import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:0")

# 1. Load backbone
from cts.model.gemma_loader import load_gemma4_e4b
from cts.backbone.gemma_adapter import GemmaCTSBackbone
BASE_MODEL_PATH = "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
from scripts.s5_train_decoder_v2 import PrefixProjector, build_canonical_prefix_prompt, load_tokenizer, DEFAULT_B1_CKPT_PATH

tokenizer = load_tokenizer()
m, _ = load_gemma4_e4b(model_id=BASE_MODEL_PATH, device_map="cuda:0", torch_dtype=torch.bfloat16)
bb = GemmaCTSBackbone(m, tokenizer)
for p in bb.cg.parameters():
    p.requires_grad = False
bb.cg.eval()
lm = bb.cg.model.language_model
lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

projector = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True).to(device)
projector.load_b1_w_proj(DEFAULT_B1_CKPT_PATH, strict=True)
projector.train()

optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-4)

# Load dataset and pick 2 worst items
import json
with open("/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl") as f:
    lines = [json.loads(line) for line in f]

# 2 worst items: 258 and 74
items = [lines[258], lines[74]]

token_ids, slot_start, slot_end, _ = build_canonical_prefix_prompt(tokenizer, k_latent=64)
target_start = slot_end + 1
prefix_prompt_tensor = torch.tensor(token_ids[:target_start], dtype=torch.long)

seq_tokens_list = []
for item in items:
    sol = item.get("solution") or item.get("answer") or ""
    sol_ids = tokenizer.encode(sol, add_special_tokens=False) + [tokenizer.eos_token_id]
    seq_tokens_list.append(torch.cat([prefix_prompt_tensor, torch.tensor(sol_ids, dtype=torch.long)]))

max_len = max(len(s) for s in seq_tokens_list)
pad_id = tokenizer.pad_token_id or 0
cur_mb_size = 2

input_ids_batch = torch.full((cur_mb_size, max_len), pad_id, dtype=torch.long, device=device)
target_ids_batch = torch.full((cur_mb_size, max_len), -100, dtype=torch.long, device=device)
for b_i, s in enumerate(seq_tokens_list):
    input_ids_batch[b_i, :len(s)] = s.to(device)
    target_ids_batch[b_i, target_start:len(s)] = s[target_start:].to(device)

z_matched = torch.randn(cur_mb_size, 64, 2560, device=device, dtype=torch.float32)
z_shuffled = torch.roll(z_matched, shifts=1, dims=0)

torch.cuda.reset_peak_memory_stats(device)
optimizer.zero_grad()

grad_accum_steps = 4
loss_lambda = 1.0
margin_m = 1.0

print(f"Testing worst-case micro-batch length: {max_len} tokens")

for cycle in range(2):
    for accum_i in range(grad_accum_steps):
        base_embeds = lm.embed_tokens(input_ids_batch)
        per_layer_inputs = lm.get_per_layer_inputs(input_ids_batch, None) if hasattr(lm, "get_per_layer_inputs") else None
        kwargs_fwd = {"per_layer_inputs": per_layer_inputs} if per_layer_inputs is not None else {}

        # 1. Shuffled evaluation with no_grad to get nll_s_val (out_m not in memory!)
        if loss_lambda > 0.0:
            with torch.no_grad():
                p_s_eval = projector(z_shuffled).to(dtype=base_embeds.dtype)
                embeds_s_eval = base_embeds.clone()
                embeds_s_eval[:, slot_start:slot_end + 1, :] = p_s_eval
                out_s_eval = lm(inputs_embeds=embeds_s_eval, return_dict=True, **kwargs_fwd)
                shift_logits_s_eval = bb.cg.lm_head(out_s_eval.last_hidden_state[:, target_start - 1:-1, :]).contiguous()
                shift_targets = target_ids_batch[:, target_start:].contiguous()
                nll_s_val = float(F.cross_entropy(shift_logits_s_eval.view(-1, shift_logits_s_eval.size(-1)), shift_targets.view(-1), ignore_index=-100).item())
                del embeds_s_eval, out_s_eval, shift_logits_s_eval, p_s_eval
                torch.cuda.empty_cache()
        else:
            nll_s_val = 0.0

        # 2. Matched forward with autograd (out_s not in memory!)
        p_m = projector(z_matched).to(dtype=base_embeds.dtype)
        embeds_m = base_embeds.clone()
        embeds_m[:, slot_start:slot_end + 1, :] = p_m
        out_m = lm(inputs_embeds=embeds_m, return_dict=True, **kwargs_fwd)
        shift_logits_m = bb.cg.lm_head(out_m.last_hidden_state[:, target_start - 1:-1, :]).contiguous()
        shift_targets = target_ids_batch[:, target_start:].contiguous()
        nll_m = F.cross_entropy(shift_logits_m.view(-1, shift_logits_m.size(-1)), shift_targets.view(-1), ignore_index=-100)
        nll_m_val = float(nll_m.item())

        if loss_lambda > 0.0:
            margin_gap = nll_s_val - nll_m_val
            margin_penalty_val = max(0.0, margin_m - margin_gap)
            margin_active = bool(margin_penalty_val > 0.0)
        else:
            margin_gap = 0.0
            margin_penalty_val = 0.0
            margin_active = False

        # Backward matched
        w_m = ((1.0 + loss_lambda) if margin_active else 1.0) / grad_accum_steps
        (w_m * nll_m).backward()

        del out_m, shift_logits_m, embeds_m, p_m, nll_m
        torch.cuda.empty_cache()

        # 3. If margin active, shuffled backward with autograd (out_m not in memory!)
        if margin_active:
            p_s = projector(z_shuffled).to(dtype=base_embeds.dtype)
            embeds_s = base_embeds.clone()
            embeds_s[:, slot_start:slot_end + 1, :] = p_s
            out_s = lm(inputs_embeds=embeds_s, return_dict=True, **kwargs_fwd)
            shift_logits_s = bb.cg.lm_head(out_s.last_hidden_state[:, target_start - 1:-1, :]).contiguous()
            nll_s = F.cross_entropy(shift_logits_s.view(-1, shift_logits_s.size(-1)), shift_targets.view(-1), ignore_index=-100)
            w_s = (-loss_lambda) / grad_accum_steps
            (w_s * nll_s).backward()
            del out_s, shift_logits_s, embeds_s, p_s, nll_s
            torch.cuda.empty_cache()

    optimizer.step()
    optimizer.zero_grad()
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"[Cycle {cycle+1}/2] peak_memory = {peak_gib:.2f} GiB (nll_m={nll_m_val:.4f}, nll_s={nll_s_val:.4f}, penalty={margin_penalty_val:.4f})")

print("=== Task 2 (b) Worst-case Memory Test SUCCESS ===")

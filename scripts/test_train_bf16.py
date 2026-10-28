import os
import sys
import torch
import torch.nn.functional as F

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/system1_5"
for p in [BASE_S5_DIR, os.path.join(BASE_S5_DIR, "scripts"), SYS_DIR, "/home/shoon/cts_e1"]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cts.model.gemma_loader import load_gemma4_e4b
from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.train.lora_compat import apply_paper_lora
from s5_train_decoder_v2 import PrefixProjector

device = torch.device("cuda:0")
base_path = "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
b1_path = "/home/shoon/retrain_1a/B1/stage1a_B1_step5000_20260907_0054.pt"

print("Loading backbone in bfloat16...")
m, tok = load_gemma4_e4b(model_id=base_path, device_map="cuda:0", torch_dtype=torch.bfloat16)
bb = GemmaCTSBackbone(m, tok)
apply_paper_lora(bb, rank=8, target_modules=("q_proj", "v_proj", "o_proj"), dropout=0.0, require_match=False, verbose=False)
ckpt = torch.load(b1_path, map_location="cpu")
sd = ckpt.get("backbone_state_dict", ckpt)
new_sd = {("routing_proj.parametrizations.weight.original" if k == "routing_proj" else k): v for k, v in sd.items()}
bb.load_state_dict(new_sd, strict=True)
bb.to(device=device, dtype=torch.bfloat16)
bb.eval()
for p in bb.parameters():
    p.requires_grad = False
print("Backbone loaded successfully.")

projector = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True).to(device)
b1_info = projector.load_b1_w_proj(b1_path, strict=True)
print("B1 loaded:", b1_info)

lm = bb.cg.model.language_model
input_ids = torch.randint(0, 1000, (8, 400), device=device)
z_m = torch.randn(8, 64, 2560, device=device, dtype=torch.float32)
z_s = torch.roll(z_m, 1, 0)
target_start = 73

per_layer = lm.get_per_layer_inputs(input_ids, None)
base_embeds = lm.embed_tokens(input_ids)

torch.cuda.reset_peak_memory_stats()

# Matched pass
with torch.autocast("cuda", dtype=torch.bfloat16):
    p_m = projector(z_m)
    embeds_m = base_embeds.clone()
    embeds_m[:, 4:68, :] = p_m.to(dtype=embeds_m.dtype)
    out_m = lm(inputs_embeds=embeds_m, per_layer_inputs=per_layer, return_dict=True)
    target_hidden_m = out_m.last_hidden_state[:, target_start - 1:-1, :]
    logits_m = bb.cg.lm_head(target_hidden_m)
    loss_m = logits_m.mean()

loss_m.backward()
mem_m = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Matched pass backward OK! Peak mem: {mem_m:.2f} GiB")

del out_m, target_hidden_m, logits_m, embeds_m
torch.cuda.empty_cache()

# Shuffled pass
with torch.autocast("cuda", dtype=torch.bfloat16):
    p_s = projector(z_s)
    embeds_s = base_embeds.clone()
    embeds_s[:, 4:68, :] = p_s.to(dtype=embeds_s.dtype)
    out_s = lm(inputs_embeds=embeds_s, per_layer_inputs=per_layer, return_dict=True)
    target_hidden_s = out_s.last_hidden_state[:, target_start - 1:-1, :]
    logits_s = bb.cg.lm_head(target_hidden_s)
    loss_s = logits_s.mean()

(-loss_s).backward()
mem_s = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Shuffled pass backward OK! Peak mem: {mem_s:.2f} GiB")

w_grad = projector.w_proj.parametrizations.weight.original.grad
print(f"Gradient norm on w_proj: {w_grad.norm().item():.4f}")
print("ALL TESTS PASSED SUCCESSFULLY!")

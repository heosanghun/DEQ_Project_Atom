#!/usr/bin/env python3
"""scripts/g2v8x_problem_latents.py - X8-0 problem-only latents (CC 2026-10-21, plan: shared/proposals/X8_READABLE_LATENT_DEQ_20261021.md §2 X8-0).
Exactly the oracle construction of g2v7_oracle_latents.py but on the PROBLEM text instead of the gold solution:
per item, pure-Gemma last-layer hidden states of the LAST 64 tokens of the problem prompt (left-padded by repeating the first state
when shorter). Contains no answer information beyond the problem text the LM already sees. Saved in the same key format
({q{idx:04d}_a0.25: {"z_star": t, "z_src": t}, q{idx:04d}: ...}) for train 993 + dev 600. Dev/train items only."""
import json, sys, torch
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_full_train import load_tokenizer, DEFAULT_PURE_GEMMA_SNAPSHOT
from cts.model.gemma_loader import load_gemma4_e4b
out_path = sys.argv[1]
recs = [json.loads(l) for l in open("/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl", encoding="utf-8") if l.strip()]
q1 = [int(x) for x in json.load(open("configs/s7_train_993.json"))]
q2 = [int(x) for x in json.load(open("results_s7/p6_item_order.json"))["indices"]]
gem, _ = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map="cuda:0", torch_dtype=torch.bfloat16, offload_vision_audio=True)
gem.eval()
lm = gem.model.language_model
tok = load_tokenizer()
out, short = {}, 0
with torch.no_grad():
    for q in q1 + q2:
        ids = torch.tensor(tok.encode(recs[q]["prompt"], add_special_tokens=True)[-512:], device="cuda:0").unsqueeze(0)
        pli = lm.get_per_layer_inputs(ids, None) if hasattr(lm, "get_per_layer_inputs") else None
        kw = {"per_layer_inputs": pli} if pli is not None else {}
        h = lm(inputs_embeds=lm.embed_tokens(ids), return_dict=True, **kw).last_hidden_state[0].float()
        if h.shape[0] < 64:
            short += 1
            h = torch.cat([h[:1].expand(64 - h.shape[0], -1), h])
        t = h[-64:].cpu()
        e = {"z_star": t, "z_src": t}
        out[f"q{q:04d}_a0.25"] = e
        out[f"q{q:04d}"] = e
torch.save(out, out_path)
print(f"[SAVE] items {len(q1) + len(q2)} (short<64: {short}) keys {len(out)} -> {out_path}")

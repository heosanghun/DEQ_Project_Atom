#!/usr/bin/env python3
# scripts/g2v4_posctl_latents.py - CC 2026-10-11 positive control: latent = pure-Gemma last-layer hidden states of the LAST 64 prompt tokens
# (left-padded by repeating the first state when the prompt is shorter). Saved like z_src files: {q0000: {"z_star": t, "z_src": t}}.
import json, sys, torch
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_full_train import load_tokenizer, DEFAULT_PURE_GEMMA_SNAPSHOT
from cts.model.gemma_loader import load_gemma4_e4b
recs = [json.loads(l) for l in open("/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl", encoding="utf-8") if l.strip()]
q1 = [int(x) for x in json.load(open("configs/s7_train_993.json"))]
q2 = [int(x) for x in json.load(open("results_s7/p6_item_order.json"))["indices"]]
gem, _ = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map="cuda:0", torch_dtype=torch.bfloat16, offload_vision_audio=True); gem.eval()
lm = gem.model.language_model; tok = load_tokenizer()
out = {}
with torch.no_grad():
    for q in q1 + q2:
        ids = torch.tensor(tok.encode(recs[q]["prompt"], add_special_tokens=True)[-512:], device="cuda:0").unsqueeze(0)
        pli = lm.get_per_layer_inputs(ids, None) if hasattr(lm, "get_per_layer_inputs") else None
        h = lm(inputs_embeds=lm.embed_tokens(ids), return_dict=True, **({"per_layer_inputs": pli} if pli is not None else {})).last_hidden_state[0].float()
        h = h[-64:] if h.shape[0] >= 64 else torch.cat([h[:1].expand(64 - h.shape[0], -1), h])
        t = h.cpu(); out[f"q{q:04d}"] = {"z_star": t, "z_src": t}
torch.save(out, "results_s7/g2v4posctl/posctl_latents.pt"); print(f"[SAVE] {len(out)} items")

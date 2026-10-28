#!/usr/bin/env python3
# scripts/g2v4_softtoken_probe.py - CC 2026-10-11 diagnostic: parameter-free latent -> token-embedding translation (tied lm_head).
# slot embedding = softmax(lm_head(z)/T) @ E_in  (E_in = scaled input embeddings). Compare NLL own vs next item, with/without text; z_src and z*'.
import json, sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_full_train import prepare_item_batch, load_tokenizer, DEFAULT_PURE_GEMMA_SNAPSHOT
from g2v3_m3_eval_latents_v2 import extract_tensor
from cts.model.gemma_loader import load_gemma4_e4b
dev = torch.device("cuda:0"); N = 100
qs = [int(x) for x in json.load(open("results_s7/p6_item_order.json"))["indices"]][:N]
recs = [json.loads(l) for l in open("/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl", encoding="utf-8") if l.strip()]
gem, _ = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map="cuda:0", torch_dtype=torch.bfloat16, offload_vision_audio=True); gem.eval()
lm, head, emb = gem.model.language_model, gem.lm_head, gem.model.language_model.embed_tokens
tok = load_tokenizer()
V = head.weight.shape[0]
with torch.no_grad():
    Ein = torch.cat([emb(torch.arange(i, min(i + 32768, V), device=dev).unsqueeze(0))[0] for i in range(0, V, 32768)])  # (V, d) scaled input embeddings
print(f"[INFO] vocab {V}, tied {head.weight.data_ptr() == emb.weight.data_ptr()}, Ein rms {float(Ein.float().pow(2).mean().sqrt()):.3f}", flush=True)
def soft(z, T):
    lg = head(z.to(torch.bfloat16)).float()
    if hasattr(gem.config, "get_text_config"):
        cap = getattr(gem.config.get_text_config(), "final_logit_softcapping", None)
        if cap: lg = torch.tanh(lg / cap) * cap
    if T == 0:
        return Ein[lg.argmax(-1)]
    return (torch.softmax(lg / T, -1).to(Ein.dtype) @ Ein)
def nll(q, fill, text):
    r = recs[q]; ids, tgt, s0, s1 = prepare_item_batch(tok, r["prompt"] if text else "", r["solution"]); ids, tgt = ids.unsqueeze(0).to(dev), tgt.unsqueeze(0).to(dev)
    e = emb(ids)
    if fill is not None: e[:, s0:s0 + 64, :] = fill.to(e.dtype)
    pli = lm.get_per_layer_inputs(ids, None) if hasattr(lm, "get_per_layer_inputs") else None
    out = lm(inputs_embeds=e, return_dict=True, **({"per_layer_inputs": pli} if pli is not None else {}))
    lg = head(out.last_hidden_state[:, :-1, :]); return float(F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), tgt[:, 1:].reshape(-1), ignore_index=-100))
for name, path in (("zsrc", "results_s7/zsrc_heldout600.pt"), ("zstar_prime", "results_s7/g2v2/zstar_zsrcinj_dev600.pt")):
    d = torch.load(path, map_location="cpu"); Z = {q: extract_tensor(d, q).float().reshape(64, 2560).to(dev) for q in qs}
    with torch.no_grad():
        for q in qs[:2]:
            ids = head(Z[q].to(torch.bfloat16)).argmax(-1).tolist()
            print(f"[DECODE] {name} q{q}: {tok.decode(ids[:40])!r} || prompt: {recs[q]['prompt'][:80]!r}", flush=True)
        for T in (0, 1.0):
            acc = {"nt_own": 0, "nt_next": 0, "t_own": 0, "t_next": 0}
            for i, q in enumerate(qs):
                o = soft(Z[q], T); n_ = soft(Z[qs[(i + 1) % N]], T)
                acc["nt_own"] += nll(q, o, False); acc["nt_next"] += nll(q, n_, False); acc["t_own"] += nll(q, o, True); acc["t_next"] += nll(q, n_, True)
            print(f"[SOFT] {name} T={T}: no-text own {acc['nt_own']/N:.4f} next {acc['nt_next']/N:.4f} | text own {acc['t_own']/N:.4f} next {acc['t_next']/N:.4f}", flush=True)
with torch.no_grad():
    b = sum(nll(q, None, False) for q in qs) / N; c = sum(nll(q, None, True) for q in qs) / N
print(f"[BASE] placeholders: no-text {b:.4f} | text {c:.4f}", flush=True)

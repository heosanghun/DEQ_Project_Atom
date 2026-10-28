#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/g2v11_x8a_latents.py - X8-A evaluation latents (CC 2026-10-23): apply the TRAINED DEQ (LoRA/routing/gamma state from
g2v11_x8a_train.py) for k_unroll M2'-form steps starting from the cached canonical z*' of each item, with the cached z_src injection,
and save z_out in the standard key format ({q{idx:04d}_a0.25: {"z_star": z_out, "z_src": z_src}, ...}) for g2v9_gen.py.
Also records per-item relative residual of the last step (G1 guard). Items: any list (dev 600 for evaluation, train 993 if needed)."""
from __future__ import annotations
import argparse, json, os, sys, torch
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
for p in ["/home/shoon/cts_e1", "/home/shoon/cts_e1/scratch"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
from g2v3_m3_full_train import DEFAULT_PURE_GEMMA_SNAPSHOT, DEFAULT_FROZEN_BACKBONE_CKPT
from g2v3_m3_eval_latents_v2 import extract_tensor
from g2v9_train import DATA
from g2v11_x8a_train import deq_unroll, TRAINABLE_KEYS
from cts.model.gemma_loader import load_gemma4_e4b
from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.train.lora_compat import apply_paper_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--zstar", required=True); ap.add_argument("--zsrc", required=True)
    ap.add_argument("--deq_state", required=True); ap.add_argument("--out", required=True); ap.add_argument("--backbone_ckpt", default=DEFAULT_FROZEN_BACKBONE_CKPT)
    ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device)
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    j = json.load(open(a.items)); items = [int(x) for x in (j["indices"] if isinstance(j, dict) else j)]
    zd_star = torch.load(a.zstar, map_location="cpu"); zd_src = torch.load(a.zsrc, map_location="cpu")
    st = torch.load(a.deq_state, map_location="cpu"); k_unroll, alpha, g0 = int(st["k_unroll"]), float(st["alpha"]), float(st["g0"])
    m_base, tok_base = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map=a.device, torch_dtype=torch.bfloat16, offload_vision_audio=True)
    bb = GemmaCTSBackbone(m_base, tok_base)
    apply_paper_lora(bb, rank=8, target_modules=("q_proj", "v_proj", "o_proj"), dropout=0.0, require_match=False, verbose=False)
    ck = torch.load(a.backbone_ckpt, map_location="cpu"); sd = ck.get("backbone_state_dict", ck)
    bb.load_state_dict({("routing_proj.parametrizations.weight.original" if k == "routing_proj" else k): v for k, v in sd.items()}, strict=True)
    missing, unexpected = bb.load_state_dict(st["state_dict"], strict=False)
    assert not unexpected, unexpected
    assert all(any(t in k for t in TRAINABLE_KEYS) for k in st["state_dict"]), "deq state has non-trainable keys"
    bb.to(device=dev, dtype=torch.bfloat16); bb.eval()
    w_g = bb.routing_matrix().to(device=dev, dtype=torch.float32)
    out, res_log, n_missing = {}, {}, 0
    with torch.no_grad():
        for i, q in enumerate(items):
            zs, zr = extract_tensor(zd_star, q), extract_tensor(zd_src, q)
            if zs is None or zr is None: n_missing += 1; continue
            z0 = zs.float().reshape(64, 2560).to(dev); zsrc = zr.float().reshape(64, 2560).to(dev)
            c = bb.encode_context(recs[q]["prompt"]); ctx = c.to(device=dev) if hasattr(c, "to") else c
            z_out, z_prev, res = deq_unroll(bb, z0, zsrc, ctx, w_g, k_unroll, alpha, g0)
            e = {"z_star": z_out.cpu(), "z_src": zsrc.cpu()}; out[f"q{q:04d}_a0.25"] = e; out[f"q{q:04d}"] = e; res_log[q] = res
            if (i + 1) % 100 == 0: print(f"[{i+1}/{len(items)}]", flush=True)
    torch.save(out, a.out)
    json.dump({"items": a.items, "deq_state": a.deq_state, "k_unroll": k_unroll, "alpha": alpha, "g0": g0, "n_items": len(res_log), "n_missing": n_missing,
               "res_last_mean": sum(v[-1] for v in res_log.values()) / max(1, len(res_log)), "res_last_max": max((v[-1] for v in res_log.values()), default=None),
               "res_by_item": {str(k): v for k, v in res_log.items()}}, open(a.out + ".res.json", "w"), indent=1)
    print(f"[SAVE] items {len(res_log)} missing {n_missing} -> {a.out}; res_last mean {sum(v[-1] for v in res_log.values())/max(1,len(res_log)):.5f}", flush=True)


if __name__ == "__main__":
    main()

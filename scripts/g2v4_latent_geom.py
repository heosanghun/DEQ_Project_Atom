#!/usr/bin/env python3
# scripts/g2v4_latent_geom.py - latent geometry on dev600 (CC, 2026-10-11, diagnostic): how much of each latent is item-specific?
import json, sys, torch
sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_eval_latents_v2 import extract_tensor
q = [int(x) for x in json.load(open("results_s7/p6_item_order.json"))["indices"]]
for name, path in (("zsrc", "results_s7/zsrc_heldout600.pt"), ("zstar_prime", "results_s7/g2v2/zstar_zsrcinj_dev600.pt")):
    d = torch.load(path, map_location="cpu")
    Z = torch.stack([extract_tensor(d, i).float().reshape(64, 2560) for i in q]).double()  # (600, 64, 2560)
    mu = Z.mean(0, keepdim=True)
    tot = Z.pow(2).sum((1, 2)).mean(); bet = (Z - mu).pow(2).sum((1, 2)).mean()
    slot_mu = Z.mean((0, 2)); per_dim = Z.abs().mean((0, 1)); top = torch.topk(per_dim, 8)
    dim_share = (Z.pow(2).sum((0, 1))); share8 = float(torch.topk(dim_share, 8).values.sum() / dim_share.sum())
    C = (Z - mu).reshape(600, -1); C = C / C.norm(dim=1, keepdim=True)
    Zf = Z.reshape(600, -1); Zn = Zf / Zf.norm(dim=1, keepdim=True); cos = Zn @ Zn.T
    off = cos[~torch.eye(600, dtype=bool)]
    slotdiff = (Z[:, 1:] - Z[:, :-1]).pow(2).sum((1, 2)).mean() / tot
    print(f"[{name}] item-specific energy share = {float(bet/tot):.4e} | mean cos(raw) {float(off.mean()):.4f} | top8 dims energy share {share8:.4f} "
          f"| top dims {top.indices.tolist()} vals {[round(float(v),1) for v in top.values]} | mean|z| {float(Z.abs().mean()):.3f} | slot-to-slot diff share {float(slotdiff):.4e}")

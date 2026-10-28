#!/usr/bin/env python3
# scripts/g2v4_pairing_check2.py - CC 2026-10-11 diagnostic: z_src recompute with the SAME loader as g2v2_zstar_run (harness_A_v1.get_model)
# vs the loader used by the M3/M4 training & eval_latents scripts; plus retrieval of stored latent by recomputed z_src.
import json, os, sys, torch
for p in ["/home/shoon/cts_e1_s5", "/home/shoon/cts_e1", "/home/shoon/cts_e1/scratch", "/home/shoon/cts_e1_s5/scripts"]:
    if p not in sys.path: sys.path.insert(0, p)
import cts, cts.deq
s5 = "/home/shoon/cts_e1_s5/cts/deq"
if s5 not in cts.deq.__path__: cts.deq.__path__.insert(0, s5)
from harness_A_v1 import get_model, get_dataset
import harness_A_v1
from g2v3_m3_eval_latents_v2 import extract_tensor
print("[HARNESS]", harness_A_v1.__file__, flush=True)
bb = get_model("cuda:0"); ds = get_dataset()
print("[BB] training mode:", bb.training, "| type", type(bb).__name__, flush=True)
def zsrc(text):
    with torch.no_grad():
        c = bb.encode_context(text); c = c.to(device="cuda:0") if hasattr(c, "to") else c
        return bb.root_fwdpass(c, K=64, output_dtype=torch.float32).reshape(-1).double().cpu()
def cs(a, b): return float((a @ b) / (a.norm() * b.norm()))
for name, items, path in (("train", "configs/s7_train_993.json", "results_s7/zsrc_train1000_P4.pt"), ("dev", "results_s7/p6_item_order.json", "results_s7/zsrc_heldout600.pt"),
                          ("fresh", "configs/g2v2_fresh1000.json", "results_s7/g2v3/fresh_zsrc_1000.pt")):
    d = torch.load(path, map_location="cpu"); j = json.load(open(items)); qs = [int(x) for x in (j["indices"] if isinstance(j, dict) else j)]
    pool = qs[:300]; S = torch.stack([extract_tensor(d, q).float().reshape(-1).double() for q in pool]); S = S / S.norm(dim=1, keepdim=True)
    hits, cl = 0, []
    for q in pool[:20]:
        z = zsrc(ds[q]["prompt"]); z2 = zsrc(ds[q]["prompt"]); zn = z / z.norm()
        sims = S @ zn; best = pool[int(sims.argmax())]; hits += best == q
        cl.append((q, round(float(sims[pool.index(q)]), 5), round(cs(z, z2), 6), best))
    print(f"[{name}] {path}: retrieval top1 {hits}/20 within first 300 | (q, cos(stored,recompute), cos(recompute,recompute), best) {cl[:6]}", flush=True)

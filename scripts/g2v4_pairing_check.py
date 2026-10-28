#!/usr/bin/env python3
# scripts/g2v4_pairing_check.py - CC 2026-10-11 diagnostic: are stored latents paired with the right problem text?
# Recompute z_src = root_fwdpass(encode_context(text)) for sample items and compare (cos) to the stored latent, under each text lookup scheme.
import hashlib, json, sys, torch
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_eval_latents_v2 import extract_tensor, DEFAULT_PURE_GEMMA_SNAPSHOT, DEFAULT_FROZEN_BACKBONE_CKPT
from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.model.gemma_loader import load_gemma4_e4b
from cts.train.lora_compat import apply_paper_lora
F1 = "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl"      # used by training (list index)
F2 = "/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl"   # used by eval latents / diag (question_idx)
L = {}
for f in (F1, F2):
    raw = open(f, "rb").read(); recs = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    qi = [r.get("question_idx", r.get("idx", r.get("q_idx"))) for r in recs]
    print(f"[FILE] {f} sha {hashlib.sha256(raw).hexdigest()[:16]} n={len(recs)} keys={sorted(recs[0].keys())} qidx==listpos {sum(1 for i, v in enumerate(qi) if v is not None and int(v) == i)}/{len(recs)} qidx_none {sum(v is None for v in qi)}")
    L[f] = recs
byq = {int(r.get("question_idx", r.get("idx", r.get("q_idx", i)))): r for i, r in enumerate(L[F2])}
m, tok = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map="cuda:0", torch_dtype=torch.bfloat16, offload_vision_audio=True)
bb = GemmaCTSBackbone(m, tok); apply_paper_lora(bb, rank=8, target_modules=("q_proj", "v_proj", "o_proj"), dropout=0.0, require_match=False, verbose=False)
sd = torch.load(DEFAULT_FROZEN_BACKBONE_CKPT, map_location="cpu"); sd = sd.get("backbone_state_dict", sd)
bb.load_state_dict({("routing_proj.parametrizations.weight.original" if k == "routing_proj" else k): v for k, v in sd.items()}, strict=True)
bb.to(device="cuda:0", dtype=torch.bfloat16).eval()
def zsrc(text):
    with torch.no_grad():
        c = bb.encode_context(text); c = c.to(device="cuda:0") if hasattr(c, "to") else c
        return bb.root_fwdpass(c, K=64, output_dtype=torch.float32).reshape(-1).double().cpu()
def cos(a, b): return float((a @ b) / (a.norm() * b.norm()))
txt = lambda r: r.get("prompt", r.get("problem", ""))
for name, items, path in (("train", "configs/s7_train_993.json", "results_s7/zsrc_train1000_P4.pt"), ("dev", "results_s7/p6_item_order.json", "results_s7/zsrc_heldout600.pt"),
                          ("train_zstar", "configs/s7_train_993.json", "results_s7/g2v2/zstar_zsrcinj_train1000.pt")):
    d = torch.load(path, map_location="cpu"); _j = json.load(open(items)); qs = [int(x) for x in (_j["indices"] if isinstance(_j, dict) else _j)]
    print(f"[{name}] {path} nkeys={len(d)} sample keys {list(d.keys())[:4]}")
    for q in qs[:4] + qs[-2:]:
        z = extract_tensor(d, q).float().reshape(-1).double()
        a = zsrc(txt(L[F1][q])); b = zsrc(txt(byq[q])); c_ = zsrc(txt(L[F2][q]))
        print(f"  q{q}: cos(stored, F1[listpos]) {cos(z, a):.5f} | cos(stored, F2[qidx]) {cos(z, b):.5f} | cos(stored, F2[listpos]) {cos(z, c_):.5f} | same text F1==F2q {txt(L[F1][q]) == txt(byq[q])}")

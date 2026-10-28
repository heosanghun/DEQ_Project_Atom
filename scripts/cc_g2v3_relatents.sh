#!/usr/bin/env bash
# scripts/cc_g2v3_relatents.sh - recompute dev600 latents with g2v3_m3_eval_latents_v2.py on GPU 7 and compare (CC, 2026-10-10). Generation 0.
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== relatents start $(date '+%F %T') ==="
sha256sum scripts/g2v3_m3_eval_latents_v2.py results_s7/g2v3/deq_backbone_A.pt results_s7/g2v3/deq_backbone_E.pt results_s7/g2v2/zstar_zsrcinj_dev600.pt results_s7/zsrc_heldout600.pt
CUDA_VISIBLE_DEVICES=$U7 nice -n 5 $PY scripts/g2v3_m3_eval_latents_v2.py --mode A --device cuda:0 --items results_s7/p6_item_order.json \
  --deq_ckpt results_s7/g2v3/deq_backbone_A.pt --zstar results_s7/g2v2/zstar_zsrcinj_dev600.pt --zsrc results_s7/zsrc_heldout600.pt \
  --out results_s7/g2v3/dev600_zstar_A_v2.pt > raw_s7/g2v3_latents_dev_A_v2.log 2>&1; echo "[A] rc=$?"; tail -3 raw_s7/g2v3_latents_dev_A_v2.log; grep LOAD-GUARD raw_s7/g2v3_latents_dev_A_v2.log
CUDA_VISIBLE_DEVICES=$U7 nice -n 5 $PY scripts/g2v3_m3_eval_latents_v2.py --mode E --device cuda:0 --items results_s7/p6_item_order.json \
  --deq_ckpt results_s7/g2v3/deq_backbone_E.pt --zsrc results_s7/zsrc_heldout600.pt \
  --out results_s7/g2v3/dev600_zsrc_E_v2.pt > raw_s7/g2v3_latents_dev_E_v2.log 2>&1; echo "[E] rc=$?"; tail -3 raw_s7/g2v3_latents_dev_E_v2.log; grep LOAD-GUARD raw_s7/g2v3_latents_dev_E_v2.log
$PY - <<'PY'
import torch, json
q = json.load(open("results_s7/p6_item_order.json"))["indices"]
for old, new in (("results_s7/g2v3/dev600_zstar_A.pt", "results_s7/g2v3/dev600_zstar_A_v2.pt"), ("results_s7/g2v3/dev600_zsrc_E.pt", "results_s7/g2v3/dev600_zsrc_E_v2.pt")):
    a = torch.load(old, map_location="cpu"); b = torch.load(new, map_location="cpu")
    d = [float((a[f"q{i:04d}"]["z_star"] - b[f"q{i:04d}"]["z_star"]).abs().max()) for i in q]
    print(f"[CMP] {new}: n={len(q)} max|d| max={max(d):.3e} exact={sum(x == 0.0 for x in d)} keys_a0.25={sum(f'q{i:04d}_a0.25' in b for i in q)}")
PY
sha256sum results_s7/g2v3/dev600_zstar_A_v2.pt results_s7/g2v3/dev600_zsrc_E_v2.pt
echo "=== relatents end $(date '+%F %T') ==="

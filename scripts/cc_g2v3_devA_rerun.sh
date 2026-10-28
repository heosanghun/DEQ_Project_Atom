#!/usr/bin/env bash
# scripts/cc_g2v3_devA_rerun.sh - PREREG_S7_G2v3.1 §3.2 dev condition A redo (CC, 2026-10-10). Generation only; scoring after B re-audit PASS.
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== devA rerun start $(date '+%F %T') ==="
[ "$(sha256sum results_s7/g2v3/dev600_zstar_A_v2.pt | cut -d' ' -f1)" = "372e42eea85b4f76aa8a008bbd7e04114c06378521bf543121aca2f7df9dd5a1" ] || { echo FATAL_latent; exit 1; }
[ "$(sha256sum results_s7/g2v2/decoder_Am3_v36.pt | cut -d' ' -f1)" = "682d05c3638e3ffadfabb6f38fd6bd3d06a3e2004964f35cbeb60be9a410cd10" ] || { echo FATAL_decoder; exit 1; }
[ "$(sha256sum scripts/s7_eval_gate_p6_v36e.py | cut -d' ' -f1)" = "a2755c1b591e12389c8efe167aa33a59a7ae6c0876234585381b6a129faa460a" ] || { echo FATAL_v36e; exit 1; }
[ -e results_s7/p6r/g2v3dev_gen_A_v2.jsonl ] && { echo FATAL_exists; exit 1; }
CUDA_VISIBLE_DEVICES=$U7 $PY scripts/s7_eval_gate_p6_v36e.py --cond A --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_Am3_v36.pt \
  --zstar results_s7/g2v3/dev600_zstar_A_v2.pt --order_sha 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c \
  --out results_s7/p6r/g2v3dev_gen_A_v2.jsonl > raw_s7/g2v3dev_gen_A_v2.log 2>&1
echo "[GEN] rc=$? rows=$(wc -l < results_s7/p6r/g2v3dev_gen_A_v2.jsonl) $(date '+%T')"
sha256sum results_s7/p6r/g2v3dev_gen_A_v2.jsonl
echo "=== devA rerun end $(date '+%F %T') ==="

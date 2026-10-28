#!/usr/bin/env bash
# scripts/cc_g2v2_confirm.sh - PREREG_S7_G2v2 §4 confirmatory evaluation (CC, 2026-10-08).
# Run ONLY after the §3 selection rule holds on dev (Gate_4.strict.delta > 0 and Gate_3.strict.delta > 0 in g2v2dev_score_ApEp.json).
# Usage: bash scripts/cc_g2v2_confirm.sh <AP_SHA> <EP_SHA>   (decoder SHAs recorded in STATE right after training)
# Shard k runs on one GPU end-to-end: z computation, then A -> C -> E1 -> B1 generation (same device per item).
set -uo pipefail
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
AP_SHA=$1; EP_SHA=$2
UU=(GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b)  # GPU 5,6,7
SH=(f4417d21de92d6a0b3c98d5fa850217f977a4f475405ac6472954cd9388b075d fee4f45a92bd8e8690cceaf8cd29d57a5daf53d5230d01aefaa59818206a976f a256039bfcc05b1f89c9e94d669abdd2617eb3aaac04a095bf4936c1ad6a2dc1)
G=results_s7/g2v2; O=results_s7/p6r/g2v2fresh; M=results_s7/p6r/g2v2fresh_merged
mkdir -p $O raw_s7
LOG=raw_s7/g2v2_confirm_master.log
exec > >(tee -a "$LOG") 2>&1
echo "=== g2v2 confirm start $(date '+%F %T') ==="
chk() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "[FATAL] SHA mismatch $1"; exit 1; }; echo "[SHA-OK] $1"; }
chk configs/g2v2_fresh1000.json 35683874284b371c6641e1280b92e6a69292a2dc5d89585158154a45b41cb3db
for k in 0 1 2; do chk configs/g2v2_fresh1000_shard$k.json ${SH[$k]}; done
chk cts/deq/zstar_inject_v2.py 7024f1f2d9ff534c3ec43251ac9c8177fa0e384717e8752b330481f6aed91650
chk scripts/g2v2_zstar_run.py 687d4480a622c845727a260ebcf49e052142269f10310b016fa82875323b4f2d
chk scripts/s7_eval_gate_p6_v36e.py a2755c1b591e12389c8efe167aa33a59a7ae6c0876234585381b6a129faa460a
chk scripts/s7_eval_score_p6_v36.py 547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd
chk scripts/s5_eval_grader_v2.py 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
chk $G/decoder_Ap_v36.pt $AP_SHA
chk $G/decoder_Ep_v36.pt $EP_SHA
[ -e $M ] && { echo "[FATAL] merged dir already exists"; exit 1; }
for u in "${UU[@]}"; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$u" | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$u" | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
# Phase 1: z for fresh shards (one GPU per shard)
for k in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${UU[$k]} $PY scripts/g2v2_zstar_run.py --items configs/g2v2_fresh1000_shard$k.json --inject zsrc \
    --out $G/fresh_zsrcinj_shard$k.pt > raw_s7/g2v2_fresh_z_shard$k.log 2>&1 &
  ZP[$k]=$!
done
for k in 0 1 2; do wait ${ZP[$k]} || { echo "[FATAL] z shard $k failed"; exit 1; }; done
$PY - <<'PYEOF'
import json, torch
q = json.load(open("configs/g2v2_fresh1000.json"))["indices"]
out = {}
for k in range(3):
    out.update(torch.load(f"results_s7/g2v2/fresh_zsrcinj_shard{k}.pt", map_location="cpu"))
keys = [f"q{i:04d}_a0.25" for i in q]
assert set(keys) == set(out.keys()) and len(out) == 1000, (len(out),)
z = {k: out[k] for k in keys}
src = {k: {"z_star": out[k]["z_src"]} for k in keys}
torch.save(z, "results_s7/g2v2/fresh_zsrcinj_1000.pt"); torch.save(src, "results_s7/g2v2/fresh_zsrc_1000.pt")
r1 = [v["r_1000"] for v in z.values()]
print(f"[ZMERGE] keys 1000, r_1000<1e-2 {sum(x < 1e-2 for x in r1)}/1000")
PYEOF
[ $? -eq 0 ] || { echo "[FATAL] z merge"; exit 1; }
sha256sum $G/fresh_zsrcinj_1000.pt $G/fresh_zsrc_1000.pt
# Phase 2: generation per shard, same GPU per item: A -> C -> E1 -> B1
gen_shard() {
  local k=$1 u=${UU[$1]} s=${SH[$1]} it=configs/g2v2_fresh1000_shard$1.json
  local base="--items $it --order_sha $s"
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36e.py --cond A --decoder_ckpt $G/decoder_Ap_v36.pt --zstar $G/fresh_zsrcinj_1000.pt $base --out $O/gen_A_s$k.jsonl > raw_s7/g2v2_fresh_gen_A_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36e.py --cond C $base --out $O/gen_C_s$k.jsonl > raw_s7/g2v2_fresh_gen_C_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36e.py --cond E1 --decoder_ckpt $G/decoder_Ep_v36.pt --zsrc $G/fresh_zsrc_1000.pt $base --out $O/gen_E1_s$k.jsonl > raw_s7/g2v2_fresh_gen_E1_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36e.py --cond B --shift 1 --decoder_ckpt $G/decoder_Ap_v36.pt --zstar $G/fresh_zsrcinj_1000.pt $base --out $O/gen_B1_s$k.jsonl > raw_s7/g2v2_fresh_gen_B1_s$k.log 2>&1 || return 1
  echo "[GEN-SHARD-DONE] $k $(date '+%T')"
}
for k in 0 1 2; do gen_shard $k & GP[$k]=$!; done
for k in 0 1 2; do wait ${GP[$k]} || { echo "[FATAL] gen shard $k failed"; exit 1; }; done
# Phase 3: merge + integrity, then score once
mkdir $M
$PY - <<'PYEOF'
import json
q = set(json.load(open("configs/g2v2_fresh1000.json"))["indices"])
for c in ("A", "C", "E1", "B1"):
    rows, ids = [], set()
    for k in range(3):
        for line in open(f"results_s7/p6r/g2v2fresh/gen_{c}_s{k}.jsonl", encoding="utf-8"):
            r = json.loads(line); rows.append(line if line.endswith("\n") else line + "\n"); ids.add(r["id"])
    assert len(rows) == 1000 and ids == q, (c, len(rows), len(ids), len(ids ^ q))
    open(f"results_s7/p6r/g2v2fresh_merged/p6_gen_{c}.jsonl", "w", encoding="utf-8").writelines(rows)
    print(f"[MERGE] {c}: 1000 rows, ids == fresh1000")
PYEOF
[ $? -eq 0 ] || { echo "[FATAL] merge integrity"; exit 1; }
sha256sum $M/p6_gen_*.jsonl
$PY scripts/s7_eval_score_p6_v36.py --gen_A $M/p6_gen_A.jsonl --gen_B1 $M/p6_gen_B1.jsonl --gen_C $M/p6_gen_C.jsonl --gen_E1 $M/p6_gen_E1.jsonl \
  --gen_dir $M --order configs/g2v2_fresh1000.json --out_json $M/gate_table.json --out_md $M/gate_table.md > raw_s7/g2v2_fresh_score.log 2>&1
echo "[SCORE] rc=$?"; sha256sum $M/gate_table.json $M/gate_table.md raw_s7/g2v2_fresh_score.log
touch raw_s7/G2V2_CONFIRM_DONE.flag
echo "=== g2v2 confirm end $(date '+%F %T') ==="

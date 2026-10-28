#!/usr/bin/env bash
# scripts/cc_g2v4_confirm.sh - PREREG_S7_G2v4 §4 confirmatory evaluation (CC, 2026-10-11). Commands exactly as §4.
# Shard k -> GPU: 0 -> GPU 5, 1 -> GPU 6, 2 -> GPU 7. Latents (one learned step) on GPU 7. Per shard: A -> C -> E1 -> B1 on the same GPU.
# rev2 (AUDIT-P014 FIX-5, R-7): all output-existence checks before any mkdir; runtime-imported files SHA-checked; LOAD-GUARD / PROJ-V2 enforced.
set -uo pipefail
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
UU=(GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b)
SH=(f8f305a1725f341d89fb38366f3d93dcdcee7c2b912b8da08dd3f010113a2253 c5a3ce59f2eb9f51a64b3b53c597d384fa5464d8400c8277f831bdc33f965f90 63935a091c9dc9a08ebe8905ee5be523990f7a730fcc1e63bd5573d526965cda)
G=results_s7/g2v4; R=results_s7/g2v4m7rep; O=results_s7/p6r/g2v4fresh; M=results_s7/p6r/g2v4fresh_merged
echo "=== g2v4 confirm start $(date '+%F %T') ==="
for f in $O $M $G/fresh_zsrcinj_shard0.pt $G/fresh_zsrcinj_shard1.pt $G/fresh_zsrcinj_shard2.pt $G/fresh_zsrcinj_1758.pt $G/fresh_zsrc_1758.pt $G/fresh_A_latent.pt $G/fresh_E_latent.pt; do
  [ -e $f ] && { echo "[FATAL] output already exists: $f"; exit 1; }
done
echo "[OUT-OK] no prior confirm outputs"
chk() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "[FATAL] SHA mismatch $1"; exit 1; }; echo "[SHA-OK] $1"; }
chk shared_prereg_PREREG_S7_G2v4.md 5923fef7e82afdd74b7c471e6f98da4d724ef9d04c20e782b1227fe1d7c5a2ce
chk configs/g2v4b_fresh.json 6e68e34c510593fb017d7dab5296c60ac506ce28d1ddeb3701231a69f696ce80
for k in 0 1 2; do chk configs/g2v4b_fresh_shard$k.json ${SH[$k]}; done
chk scripts/g2v2_zstar_run.py 687d4480a622c845727a260ebcf49e052142269f10310b016fa82875323b4f2d
chk cts/deq/zstar_inject_v2.py 7024f1f2d9ff534c3ec43251ac9c8177fa0e384717e8752b330481f6aed91650
chk configs/zstar_canonical.yaml bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157
chk scripts/g2v4_fresh_prep.py ed79108f9d8376467ea18781f6407c1140d59a69f7f3eaf22c98f929f311db54
chk scripts/g2v3_m3_eval_latents_v2.py 73db9dafd33e49e963d48aad4797c6b0eb7856e5e8e847fbb3b8dc0e5109921b
chk scripts/s7_eval_gate_p6_v36f.py fd6c2d03e3ed18d1597950edc79a01af8331d192c9f84b821a5b5bcf3f71cfe1
chk scripts/g2v4_proj_v2.py 7cabcef2e13bb524a6815add5198853f21e8437fd04d018a6cef2227161df47a
chk scripts/s7_eval_score_p6_v36.py 547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd
chk scripts/s5_eval_grader_v2.py 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
chk results_s7/g2v2/decoder_M7A_v36.pt 46977c2db533888dc13a8defe21ac287b26bd7ff1e08bb83eb22a4ca4bb99670
chk results_s7/g2v2/decoder_M7E_v36.pt d0c33599c9586d3ed131877a2439ee09d5d48804a0408965daaa0238705afd5a
chk $R/deq_Ah.pt b43de6b0eeb0a16a5f04762ac9c6a95979e504bc48f8b52249213490d6485cf7
chk $R/deq_Eh.pt 7e9dc5feb1e563996e90cc825c3567e54caadb86f5a9a25ce03435df37be9788
chk scripts/s7_train_decoder_v36.py f2951054416c7b1593075c6a94e3c1ab7e7e7be516f078d540b5bd9ce5bf3ea9
chk scripts/g2v3_m3_full_train.py 36e0f18addc8d46f6d8bf376cd95e89cf9eb75a1f15c4e8b714141efe5a49c99
chk scripts/check_p6_decoder_version.sh 1fa95f72d5e5cea2ad1836dd458544190467f0cfb1daa14e41f6459cb51d870b
mkdir -p $G $O raw_s7
for u in "${UU[@]}"; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$u" | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$u" | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
# §4-2 M2' z*'0 per shard
for k in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${UU[$k]} $PY scripts/g2v2_zstar_run.py --items configs/g2v4b_fresh_shard$k.json --inject zsrc \
    --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --config configs/zstar_canonical.yaml \
    --out $G/fresh_zsrcinj_shard$k.pt > raw_s7/g2v4_fresh_z_shard$k.log 2>&1 &
  ZP[$k]=$!
done
for k in 0 1 2; do wait ${ZP[$k]} || { echo "[FATAL] z shard $k"; exit 1; }; echo "[Z-DONE] shard $k $(date '+%T')"; done
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v4_fresh_prep.py" || { echo "[FATAL] prep"; exit 1; }
sha256sum $G/fresh_zsrcinj_1758.pt $G/fresh_zsrc_1758.pt
# §4-3 learned one step on GPU 7
CUDA_VISIBLE_DEVICES=${UU[2]} $PY scripts/g2v3_m3_eval_latents_v2.py --mode A --device cuda:0 --items configs/g2v4b_fresh.json \
  --deq_ckpt $R/deq_Ah.pt --zstar $G/fresh_zsrcinj_1758.pt --zsrc $G/fresh_zsrc_1758.pt --out $G/fresh_A_latent.pt > raw_s7/g2v4_fresh_latent_A.log 2>&1 || { echo "[FATAL] latent A"; exit 1; }
CUDA_VISIBLE_DEVICES=${UU[2]} $PY scripts/g2v3_m3_eval_latents_v2.py --mode E --device cuda:0 --items configs/g2v4b_fresh.json \
  --deq_ckpt $R/deq_Eh.pt --zsrc $G/fresh_zsrc_1758.pt --out $G/fresh_E_latent.pt > raw_s7/g2v4_fresh_latent_E.log 2>&1 || { echo "[FATAL] latent E"; exit 1; }
for t in A E; do grep -q "applied 436 DEQ tensors, unexpected 0" raw_s7/g2v4_fresh_latent_$t.log || { echo "[FATAL] LOAD-GUARD $t"; exit 1; }; done
grep -h LOAD-GUARD raw_s7/g2v4_fresh_latent_A.log raw_s7/g2v4_fresh_latent_E.log
sha256sum $G/fresh_A_latent.pt $G/fresh_E_latent.pt
# §4-4 generation per shard, same GPU per item
gen_shard() {
  local k=$1 u=${UU[$1]} base="--items configs/g2v4b_fresh_shard$1.json --order_sha ${SH[$1]}"
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36f.py --cond A --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_M7A_v36.pt --zstar $G/fresh_A_latent.pt $base --out $O/gen_A_s$k.jsonl > raw_s7/g2v4_fresh_gen_A_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36f.py --cond C --device cuda:0 $base --out $O/gen_C_s$k.jsonl > raw_s7/g2v4_fresh_gen_C_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36f.py --cond E1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_M7E_v36.pt --zsrc $G/fresh_E_latent.pt $base --out $O/gen_E1_s$k.jsonl > raw_s7/g2v4_fresh_gen_E1_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36f.py --cond B --shift 1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_M7A_v36.pt --zstar $G/fresh_A_latent.pt $base --out $O/gen_B1_s$k.jsonl > raw_s7/g2v4_fresh_gen_B1_s$k.log 2>&1 || return 1
  echo "[GEN-SHARD-DONE] $k $(date '+%T')"
}
for k in 0 1 2; do gen_shard $k & GP[$k]=$!; done
for k in 0 1 2; do wait ${GP[$k]} || { echo "[FATAL] gen shard $k"; exit 1; }; done
for k in 0 1 2; do for c in A E1 B1; do grep -q "\[PROJ-V2\]" raw_s7/g2v4_fresh_gen_${c}_s$k.log || { echo "[FATAL] PROJ-V2 missing ${c} s$k"; exit 1; }; done; done
echo "[PROJ-V2-OK] 9 logs"
# §4-5 merge + checks + score once
mkdir $M
nice -n 19 bash -c "ulimit -v 62914560; $PY - <<'PY'
import json
q = set(json.load(open('configs/g2v4b_fresh.json'))['indices'])
N = len(q)
for c in ('A', 'C', 'E1', 'B1'):
    rows, ids, bad = [], set(), 0
    for k in range(3):
        for line in open(f'results_s7/p6r/g2v4fresh/gen_{c}_s{k}.jsonl', encoding='utf-8'):
            try: r = json.loads(line); ids.add(r['id']); rows.append(line if line.endswith('\n') else line + '\n')
            except Exception: bad += 1
    assert len(rows) == N and len(ids) == N and ids == q and bad == 0, (c, len(rows), len(ids), bad)
    open(f'results_s7/p6r/g2v4fresh_merged/p6_gen_{c}.jsonl', 'w', encoding='utf-8').writelines(rows)
    print(f'[MERGE] {c}: {N} rows, {N} ids == g2v4b_fresh, json_err 0')
PY" || { echo "[FATAL] merge check"; exit 1; }
sha256sum $M/p6_gen_*.jsonl
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/s7_eval_score_p6_v36.py --gen_A $M/p6_gen_A.jsonl --gen_B1 $M/p6_gen_B1.jsonl --gen_C $M/p6_gen_C.jsonl --gen_E1 $M/p6_gen_E1.jsonl --gen_dir $M --order configs/g2v4b_fresh.json --out_json $M/gate_table.json --out_md $M/gate_table.md" > raw_s7/g2v4_fresh_score.log 2>&1
echo "[SCORE] rc=$?"; sha256sum $M/gate_table.json $M/gate_table.md raw_s7/g2v4_fresh_score.log
touch raw_s7/G2V4_CONFIRM_DONE.flag
echo "=== g2v4 confirm end $(date '+%F %T') ==="

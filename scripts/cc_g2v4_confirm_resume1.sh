#!/usr/bin/env bash
# scripts/cc_g2v4_confirm_resume1.sh - PREREG_S7_G2v4 §5 resume-1 (CC, 2026-10-11 22:xx). Base: cc_g2v4_confirm.sh d74fd9a9 (run 17:18:14).
# Cause: server /home 100% full -> g2v4_fresh_prep.py torch.save failed at 21:04 ("unexpected pos ...", [FATAL] prep). z*'0 shards 0/1/2 had finished (21:00/21:00/21:04).
# Resume from §4-2 prep. Changes vs base (only these):
#   (1) large tensor outputs (prep 2 files, latents 2 files) go to W=/tmp/cts_g2v4_shoon (root fs) via g2v4_fresh_prep_r1.py (output paths only);
#   (2) the truncated defect file results_s7/g2v4/fresh_zsrcinj_1758.pt is MOVED (kept, not deleted) to $W/defect/ to free /home for generation outputs;
#   (3) the shard z files are reused as written (SHA recorded); z computation is NOT re-run;
#   (5) /home avail 0 at upload time -> this script and g2v4_fresh_prep_r1.py are staged in S=/tmp/cts_g2v4_stage and run from there (cwd stays /home/shoon/cts_e1_s5).
#   (4) logs as base (raw_s7/...). Generation / merge / scoring commands, items, checkpoints, gates: identical to base §4-3..§4-5.
set -uo pipefail
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
UU=(GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b)
SH=(f8f305a1725f341d89fb38366f3d93dcdcee7c2b912b8da08dd3f010113a2253 c5a3ce59f2eb9f51a64b3b53c597d384fa5464d8400c8277f831bdc33f965f90 63935a091c9dc9a08ebe8905ee5be523990f7a730fcc1e63bd5573d526965cda)
S=/tmp/cts_g2v4_stage; G=results_s7/g2v4; W=/tmp/cts_g2v4_shoon; R=results_s7/g2v4m7rep; O=results_s7/p6r/g2v4fresh; M=results_s7/p6r/g2v4fresh_merged
echo "=== g2v4 confirm resume1 start $(date '+%F %T') ==="
if [ -e $O ]; then [ -d $O ] && [ -z "$(ls -A $O)" ] || { echo "[FATAL] exists and not empty: $O"; exit 1; }; echo "[NOTE] $O exists but is empty (created by base script line 36 mkdir before z)"; fi
for f in $M $W $G/fresh_zsrc_1758.pt $G/fresh_A_latent.pt $G/fresh_E_latent.pt; do [ -e $f ] && { echo "[FATAL] exists: $f"; exit 1; }; done
for k in 0 1 2; do
  [ -s $G/fresh_zsrcinj_shard$k.pt ] || { echo "[FATAL] missing shard $k"; exit 1; }
  grep -q "\[Z-DONE\] shard $k" raw_s7/cc_g2v4_confirm_master.log || { echo "[FATAL] shard $k not done in master log"; exit 1; }
done
[ -e $G/fresh_zsrcinj_1758.pt ] || { echo "[FATAL] defect file not found (unexpected state)"; exit 1; }
echo "[OUT-OK]"
chk() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "[FATAL] SHA mismatch $1"; exit 1; }; echo "[SHA-OK] $1"; }
chk shared_prereg_PREREG_S7_G2v4.md 5923fef7e82afdd74b7c471e6f98da4d724ef9d04c20e782b1227fe1d7c5a2ce
chk scripts/cc_g2v4_confirm.sh d74fd9a96dc246028c09adc31adc736c672d5dc7ec50b6c0533965af261ca9bb
chk configs/g2v4b_fresh.json 6e68e34c510593fb017d7dab5296c60ac506ce28d1ddeb3701231a69f696ce80
for k in 0 1 2; do chk configs/g2v4b_fresh_shard$k.json ${SH[$k]}; done
chk $S/g2v4_fresh_prep_r1.py e8876570b4f20360f8e40a286ad73fff28531d983e35cae24c30df0e34ed133f
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
sha256sum $G/fresh_zsrcinj_shard0.pt $G/fresh_zsrcinj_shard1.pt $G/fresh_zsrcinj_shard2.pt
mkdir -m 700 $W && mkdir -m 700 $W/defect || { echo "[FATAL] mkdir W"; exit 1; }
ls -la --time-style=full-iso $G/fresh_zsrcinj_1758.pt
sha256sum $G/fresh_zsrcinj_1758.pt
mv $G/fresh_zsrcinj_1758.pt $W/defect/fresh_zsrcinj_1758.truncated_2104.pt || { echo "[FATAL] move defect"; exit 1; }
ls -la $W/defect/
df -h /home /tmp | cat
AV=$(df --output=avail -k /home | tail -n 1 | tr -d ' '); [ "$AV" -gt 1048576 ] || { echo "[FATAL] /home avail < 1 GiB after move ($AV KiB)"; exit 1; }
AT=$(df --output=avail -k /tmp | tail -n 1 | tr -d ' '); [ "$AT" -gt 15728640 ] || { echo "[FATAL] /tmp avail < 15 GiB ($AT KiB)"; exit 1; }
for u in "${UU[@]}"; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$u" | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$u" | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
# §4-2 prep (outputs in W)
nice -n 19 bash -c "ulimit -v 62914560; $PY $S/g2v4_fresh_prep_r1.py" || { echo "[FATAL] prep"; exit 1; }
sha256sum $W/fresh_zsrcinj_1758.pt $W/fresh_zsrc_1758.pt
# §4-3 learned one step on GPU 7 (outputs in W)
CUDA_VISIBLE_DEVICES=${UU[2]} $PY scripts/g2v3_m3_eval_latents_v2.py --mode A --device cuda:0 --items configs/g2v4b_fresh.json \
  --deq_ckpt $R/deq_Ah.pt --zstar $W/fresh_zsrcinj_1758.pt --zsrc $W/fresh_zsrc_1758.pt --out $W/fresh_A_latent.pt > raw_s7/g2v4_fresh_latent_A.log 2>&1 || { echo "[FATAL] latent A"; exit 1; }
CUDA_VISIBLE_DEVICES=${UU[2]} $PY scripts/g2v3_m3_eval_latents_v2.py --mode E --device cuda:0 --items configs/g2v4b_fresh.json \
  --deq_ckpt $R/deq_Eh.pt --zsrc $W/fresh_zsrc_1758.pt --out $W/fresh_E_latent.pt > raw_s7/g2v4_fresh_latent_E.log 2>&1 || { echo "[FATAL] latent E"; exit 1; }
for t in A E; do grep -q "applied 436 DEQ tensors, unexpected 0" raw_s7/g2v4_fresh_latent_$t.log || { echo "[FATAL] LOAD-GUARD $t"; exit 1; }; done
grep -h LOAD-GUARD raw_s7/g2v4_fresh_latent_A.log raw_s7/g2v4_fresh_latent_E.log
sha256sum $W/fresh_A_latent.pt $W/fresh_E_latent.pt
mkdir -p $O
# §4-4 generation per shard, same GPU per item (identical to base; latents read from W)
gen_shard() {
  local k=$1 u=${UU[$1]} base="--items configs/g2v4b_fresh_shard$1.json --order_sha ${SH[$1]}"
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36f.py --cond A --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_M7A_v36.pt --zstar $W/fresh_A_latent.pt $base --out $O/gen_A_s$k.jsonl > raw_s7/g2v4_fresh_gen_A_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36f.py --cond C --device cuda:0 $base --out $O/gen_C_s$k.jsonl > raw_s7/g2v4_fresh_gen_C_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36f.py --cond E1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_M7E_v36.pt --zsrc $W/fresh_E_latent.pt $base --out $O/gen_E1_s$k.jsonl > raw_s7/g2v4_fresh_gen_E1_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36f.py --cond B --shift 1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_M7A_v36.pt --zstar $W/fresh_A_latent.pt $base --out $O/gen_B1_s$k.jsonl > raw_s7/g2v4_fresh_gen_B1_s$k.log 2>&1 || return 1
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
echo "=== g2v4 confirm resume1 end $(date '+%F %T') ==="

#!/usr/bin/env bash
# scripts/cc_g2v3_confirm.sh - PREREG_S7_G2v3_1 §4 confirmatory evaluation (CC, 2026-10-11). Commands exactly as §4.
# Shard k -> GPU: 0 -> GPU 5, 1 -> GPU 6, 2 -> GPU 7. Latents (one learned step) on GPU 7. Per shard: A -> C -> E1 -> B1 on the same GPU.
set -uo pipefail
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
UU=(GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b)
SH=(f4417d21de92d6a0b3c98d5fa850217f977a4f475405ac6472954cd9388b075d fee4f45a92bd8e8690cceaf8cd29d57a5daf53d5230d01aefaa59818206a976f a256039bfcc05b1f89c9e94d669abdd2617eb3aaac04a095bf4936c1ad6a2dc1)
G=results_s7/g2v3; O=results_s7/p6r/g2v3fresh; M=results_s7/p6r/g2v3fresh_merged
mkdir -p $O raw_s7
echo "=== g2v3 confirm start $(date '+%F %T') ==="
chk() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "[FATAL] SHA mismatch $1"; exit 1; }; echo "[SHA-OK] $1"; }
chk shared_prereg_PREREG_S7_G2v3_1.md 3d1a6ee0292b58fcab514ec2b71588dc78a277f6c464e61739ebb4ac96573daa
chk configs/g2v2_fresh1000.json 35683874284b371c6641e1280b92e6a69292a2dc5d89585158154a45b41cb3db
for k in 0 1 2; do chk configs/g2v2_fresh1000_shard$k.json ${SH[$k]}; done
chk scripts/g2v2_zstar_run.py 687d4480a622c845727a260ebcf49e052142269f10310b016fa82875323b4f2d
chk cts/deq/zstar_inject_v2.py 7024f1f2d9ff534c3ec43251ac9c8177fa0e384717e8752b330481f6aed91650
chk configs/zstar_canonical.yaml bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157
chk scripts/g2v3_fresh_prep.py 59889cd3cc829a5bb281e0d34a7a20899f824272f0911eb8db8547cbe741b423
chk scripts/g2v3_m3_eval_latents_v2.py 73db9dafd33e49e963d48aad4797c6b0eb7856e5e8e847fbb3b8dc0e5109921b
chk scripts/s7_eval_gate_p6_v36e.py a2755c1b591e12389c8efe167aa33a59a7ae6c0876234585381b6a129faa460a
chk scripts/s7_eval_score_p6_v36.py 547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd
chk scripts/s5_eval_grader_v2.py 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
chk results_s7/g2v2/decoder_Am3_v36.pt 682d05c3638e3ffadfabb6f38fd6bd3d06a3e2004964f35cbeb60be9a410cd10
chk results_s7/g2v2/decoder_Em3_v36.pt ba1d2788ec4f3fb38949f17cd215b1d42721c175f3a47d2659bc4dc573722e05
chk $G/deq_backbone_A.pt fec12c8faab77c31276ba4d155aaf41f32243f33212793e92f34117e80a14cbf
chk $G/deq_backbone_E.pt 255ed2731098fa5081c540e456383fa73276e9cb73c589f9f39d0da0210f581b
[ -e $M ] && { echo "[FATAL] merged dir exists"; exit 1; }
for f in $G/fresh_zsrcinj_shard0.pt $G/fresh_zsrcinj_1000.pt $G/fresh_A_latent.pt; do [ -e $f ] && { echo "[FATAL] $f exists"; exit 1; }; done
for u in "${UU[@]}"; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$u" | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$u" | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
# §4-2 M2' z*'0 per shard
for k in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${UU[$k]} $PY scripts/g2v2_zstar_run.py --items configs/g2v2_fresh1000_shard$k.json --inject zsrc \
    --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --config configs/zstar_canonical.yaml \
    --out $G/fresh_zsrcinj_shard$k.pt > raw_s7/g2v3_fresh_z_shard$k.log 2>&1 &
  ZP[$k]=$!
done
for k in 0 1 2; do wait ${ZP[$k]} || { echo "[FATAL] z shard $k"; exit 1; }; echo "[Z-DONE] shard $k $(date '+%T')"; done
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v3_fresh_prep.py" || { echo "[FATAL] prep"; exit 1; }
sha256sum $G/fresh_zsrcinj_1000.pt $G/fresh_zsrc_1000.pt
# §4-3 learned one step on GPU 7
CUDA_VISIBLE_DEVICES=${UU[2]} $PY scripts/g2v3_m3_eval_latents_v2.py --mode A --device cuda:0 --items configs/g2v2_fresh1000.json \
  --deq_ckpt $G/deq_backbone_A.pt --zstar $G/fresh_zsrcinj_1000.pt --zsrc $G/fresh_zsrc_1000.pt --out $G/fresh_A_latent.pt > raw_s7/g2v3_fresh_latent_A.log 2>&1 || { echo "[FATAL] latent A"; exit 1; }
CUDA_VISIBLE_DEVICES=${UU[2]} $PY scripts/g2v3_m3_eval_latents_v2.py --mode E --device cuda:0 --items configs/g2v2_fresh1000.json \
  --deq_ckpt $G/deq_backbone_E.pt --zsrc $G/fresh_zsrc_1000.pt --out $G/fresh_E_latent.pt > raw_s7/g2v3_fresh_latent_E.log 2>&1 || { echo "[FATAL] latent E"; exit 1; }
grep -h LOAD-GUARD raw_s7/g2v3_fresh_latent_A.log raw_s7/g2v3_fresh_latent_E.log
sha256sum $G/fresh_A_latent.pt $G/fresh_E_latent.pt
# §4-4 generation per shard, same GPU per item
gen_shard() {
  local k=$1 u=${UU[$1]} base="--items configs/g2v2_fresh1000_shard$1.json --order_sha ${SH[$1]}"
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36e.py --cond A --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_Am3_v36.pt --zstar $G/fresh_A_latent.pt $base --out $O/gen_A_s$k.jsonl > raw_s7/g2v3_fresh_gen_A_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36e.py --cond C --device cuda:0 $base --out $O/gen_C_s$k.jsonl > raw_s7/g2v3_fresh_gen_C_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36e.py --cond E1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_Em3_v36.pt --zsrc $G/fresh_E_latent.pt $base --out $O/gen_E1_s$k.jsonl > raw_s7/g2v3_fresh_gen_E1_s$k.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36e.py --cond B --shift 1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_Am3_v36.pt --zstar $G/fresh_A_latent.pt $base --out $O/gen_B1_s$k.jsonl > raw_s7/g2v3_fresh_gen_B1_s$k.log 2>&1 || return 1
  echo "[GEN-SHARD-DONE] $k $(date '+%T')"
}
for k in 0 1 2; do gen_shard $k & GP[$k]=$!; done
for k in 0 1 2; do wait ${GP[$k]} || { echo "[FATAL] gen shard $k"; exit 1; }; done
# §4-5 merge + checks + score once
mkdir $M
nice -n 19 bash -c "ulimit -v 62914560; $PY - <<'PY'
import json
q = set(json.load(open('configs/g2v2_fresh1000.json'))['indices'])
for c in ('A', 'C', 'E1', 'B1'):
    rows, ids, bad = [], set(), 0
    for k in range(3):
        for line in open(f'results_s7/p6r/g2v3fresh/gen_{c}_s{k}.jsonl', encoding='utf-8'):
            try: r = json.loads(line); ids.add(r['id']); rows.append(line if line.endswith('\n') else line + '\n')
            except Exception: bad += 1
    assert len(rows) == 1000 and len(ids) == 1000 and ids == q and bad == 0, (c, len(rows), len(ids), bad)
    open(f'results_s7/p6r/g2v3fresh_merged/p6_gen_{c}.jsonl', 'w', encoding='utf-8').writelines(rows)
    print(f'[MERGE] {c}: 1000 rows, 1000 ids == fresh1000, json_err 0')
PY" || { echo "[FATAL] merge check"; exit 1; }
sha256sum $M/p6_gen_*.jsonl
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/s7_eval_score_p6_v36.py --gen_A $M/p6_gen_A.jsonl --gen_B1 $M/p6_gen_B1.jsonl --gen_C $M/p6_gen_C.jsonl --gen_E1 $M/p6_gen_E1.jsonl --gen_dir $M --order configs/g2v2_fresh1000.json --out_json $M/gate_table.json --out_md $M/gate_table.md" > raw_s7/g2v3_fresh_score.log 2>&1
echo "[SCORE] rc=$?"; sha256sum $M/gate_table.json $M/gate_table.md raw_s7/g2v3_fresh_score.log
touch raw_s7/G2V3_CONFIRM_DONE.flag
echo "=== g2v3 confirm end $(date '+%F %T') ==="

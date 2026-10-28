#!/usr/bin/env bash
# rev2 (AUDIT-P012 FIX-3): generator v36e (accepts results_s7/g2v2 decoders), completeness checks before scoring/flag.
# scripts/cc_g2v2_train_eval.sh - G2v2 dev chain (CC, 2026-10-08). DEV ONLY: old held-out 600 (already used for G2 round 1).
# Waits for raw_s7/G2V2_COMPUTE_DONE.flag, then:
#  (1) trains 3 decoders with the registered training script s7_train_decoder_v36.py (f2951054…), same hyperparameters as A/D/E v36:
#      Ap = z*' (M2' zsrc-injected z_1000), lambda 0   [GPU 5]
#      Ep = z_src,                          lambda 0   [GPU 6]
#      Aq = z*',                            lambda 1   [GPU 7]  (pairs with existing decoder_E_v36, lambda 1)
#  (2) dev NLL (v36e --max_new_tokens 1 --nll) for Ap, Ep, Aq
#  (3) dev generation (v36e, max_new_tokens 512) for Ap [GPU 0], Ep [GPU 1], Aq [GPU 3]; C/E1 reuse P6R files.
#  (4) score with s7_eval_score_p6_v36.py (A=Ap vs C, E1=Ep) and (A=Aq vs C, E1=P6R E1).
set -uo pipefail
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
GPU0=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a  # physical GPU 5 (GPU 0/1/3 are throttled)
GPU1=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c  # physical GPU 6
GPU3=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b  # physical GPU 7
ORDER_SHA=6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c
G=results_s7/g2v2; P=results_s7/p6r
mkdir -p $G raw_s7/g2v2_train
LOG=raw_s7/g2v2_train_eval_master.log
exec > >(tee -a "$LOG") 2>&1
echo "=== g2v2 train/eval chain start $(date '+%F %T') ==="
until [ -f raw_s7/G2V2_COMPUTE_DONE.flag ]; do sleep 60; done
echo "[FLAG] compute done $(date '+%T')"
[ "$(sha256sum scripts/s7_train_decoder_v36.py | cut -d' ' -f1)" = "f2951054416c7b1593075c6a94e3c1ab7e7e7be516f078d540b5bd9ce5bf3ea9" ] || { echo "[FATAL] train script SHA"; exit 1; }
[ "$(sha256sum scripts/s7_eval_gate_p6_v36e.py | cut -d' ' -f1)" = "a2755c1b591e12389c8efe167aa33a59a7ae6c0876234585381b6a129faa460a" ] || { echo "[FATAL] v36e SHA"; exit 1; }
for u in $GPU0 $GPU1 $GPU3; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$u" | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$u" | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
COMMON="--items configs/s7_train_993.json --max_steps 3000 --batch_size 8 --micro_batch_size 2 --grad_accum_steps 4 --lr_wproj 1e-4 --lr_adapter 3e-4 --weight_decay 0.01 --output_dir $G --raw_dir raw_s7/g2v2_train"
train() {  # uuid name zfile lambda
  echo "[TRAIN] $2 on $1 lambda=$4 z=$3 $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/s7_train_decoder_v36.py $COMMON --zstar $3 --loss_lambda $4 \
    --manifest $G/manifest_$2.json --out $G/decoder_$2_v36.pt > raw_s7/g2v2_train_$2.log 2>&1
  echo "[TRAIN-DONE] $2 rc=$? $(date '+%T')"
}
train $GPU0 Ap $G/zstar_zsrcinj_train1000.pt 0.0 & T1=$!
train $GPU1 Ep results_s7/zsrc_train1000_P4.pt 0.0 & T2=$!
train $GPU3 Aq $G/zstar_zsrcinj_train1000.pt 1.0 & T3=$!
wait $T1; R1=$?; wait $T2; R2=$?; wait $T3; R3=$?
for f in Ap Ep Aq; do [ -f $G/decoder_${f}_v36.pt ] || { echo "[FATAL] missing decoder $f"; exit 1; }; done
ls -l $G/decoder_*_v36.pt; sha256sum $G/decoder_*_v36.pt
gen() {  # uuid tag cond decoder extra... ; NLL first then generation
  local u=$1 tag=$2 cond=$3 ck=$4; shift 4
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36e.py --cond $cond --decoder_ckpt $ck "$@" --max_new_tokens 1 --nll \
    --order_sha $ORDER_SHA --out $P/g2v2dev_nll_$tag.jsonl > raw_s7/g2v2dev_nll_$tag.log 2>&1
  echo "[NLL-DONE] $tag rc=$? rows=$(wc -l < $P/g2v2dev_nll_$tag.jsonl) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36e.py --cond $cond --decoder_ckpt $ck "$@" \
    --order_sha $ORDER_SHA --out $P/g2v2dev_gen_$tag.jsonl > raw_s7/g2v2dev_gen_$tag.log 2>&1
  echo "[GEN-DONE] $tag rc=$? rows=$(wc -l < $P/g2v2dev_gen_$tag.jsonl) $(date '+%T')"
}
gen $GPU0 Ap A  $G/decoder_Ap_v36.pt --zstar $G/zstar_zsrcinj_dev600.pt & G1=$!
gen $GPU1 Ep E1 $G/decoder_Ep_v36.pt & G2=$!
gen $GPU3 Aq A  $G/decoder_Aq_v36.pt --zstar $G/zstar_zsrcinj_dev600.pt & G3=$!
wait $G1 $G2 $G3
for t in Ap Ep Aq; do n=$(wc -l < $P/g2v2dev_gen_$t.jsonl 2>/dev/null || echo 0); echo "[CHECK] gen $t rows=$n"; [ "$n" -eq 600 ] || { echo "[FATAL] gen $t incomplete"; exit 1; }; done
$PY scripts/s7_eval_score_p6_v36.py --gen_A $P/g2v2dev_gen_Ap.jsonl --gen_C $P/p6_gen_C.jsonl --gen_E1 $P/g2v2dev_gen_Ep.jsonl \
  --gen_B1 $P/p6_gen_B1.jsonl --gen_dir $P --order results_s7/p6_item_order.json \
  --out_json $P/g2v2dev_score_ApEp.json --out_md $P/g2v2dev_score_ApEp.md > raw_s7/g2v2dev_score_ApEp.log 2>&1; echo "[SCORE] ApEp rc=$?"
$PY scripts/s7_eval_score_p6_v36.py --gen_A $P/g2v2dev_gen_Aq.jsonl --gen_C $P/p6_gen_C.jsonl --gen_E1 $P/p6_gen_E1.jsonl \
  --gen_B1 $P/p6_gen_B1.jsonl --gen_dir $P --order results_s7/p6_item_order.json \
  --out_json $P/g2v2dev_score_AqE.json --out_md $P/g2v2dev_score_AqE.md > raw_s7/g2v2dev_score_AqE.log 2>&1; echo "[SCORE] AqE rc=$?"
touch raw_s7/G2V2_DEV_DONE.flag
echo "=== g2v2 train/eval chain end $(date '+%F %T') ==="

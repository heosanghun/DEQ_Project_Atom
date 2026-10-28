#!/usr/bin/env bash
# scripts/cc_g2v4_dev_m7.sh - CC 2026-10-11: M7 (Ah/Eh) dev600 accuracy check (dev = already-seen items; selection evidence for a 4th prereg, not a gate).
# Latents: one learned step (g2v3_m3_eval_latents_v2.py, same loader as training). Generation v36f, 2 halves x {A,E} on GPU 5/6. Score with C = p6_gen_C.jsonl.
set -uo pipefail
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c
R=results_s7/g2v4m7rep; O=results_s7/p6r/g2v4dev; mkdir -p $O
echo "=== m7 dev start $(date '+%F %T') ==="
sha256sum scripts/s7_eval_gate_p6_v36f.py scripts/g2v4_proj_v2.py scripts/g2v3_m3_eval_latents_v2.py $R/deq_Ah.pt $R/deq_Eh.pt $R/decoder_Ah.pt $R/decoder_Eh.pt results_s7/p6_item_order.json results_s7/p6r/p6_gen_C.jsonl
$PY - <<'PY'
import json
q = json.load(open("results_s7/p6_item_order.json"))["indices"]
for k, part in enumerate((q[:300], q[300:])):
    json.dump({"indices": part}, open(f"configs/g2v4_dev600_h{k}.json", "w"))
print("[SPLIT]", len(q))
PY
H0=$(sha256sum configs/g2v4_dev600_h0.json | cut -d' ' -f1); H1=$(sha256sum configs/g2v4_dev600_h1.json | cut -d' ' -f1); echo "[SHA] h0 $H0 h1 $H1"
cp -n $R/decoder_Ah.pt results_s7/g2v2/decoder_M7A_v36.pt; cp -n $R/decoder_Eh.pt results_s7/g2v2/decoder_M7E_v36.pt
sha256sum results_s7/g2v2/decoder_M7A_v36.pt results_s7/g2v2/decoder_M7E_v36.pt
CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v3_m3_eval_latents_v2.py --mode A --device cuda:0 --items results_s7/p6_item_order.json --deq_ckpt $R/deq_Ah.pt \
  --zstar results_s7/g2v2/zstar_zsrcinj_dev600.pt --zsrc results_s7/zsrc_heldout600.pt --out $R/dev600_latent_A.pt > $R/lat_A.log 2>&1 & LA=$!
CUDA_VISIBLE_DEVICES=$U6 $PY scripts/g2v3_m3_eval_latents_v2.py --mode E --device cuda:0 --items results_s7/p6_item_order.json --deq_ckpt $R/deq_Eh.pt \
  --zsrc results_s7/zsrc_heldout600.pt --out $R/dev600_latent_E.pt > $R/lat_E.log 2>&1 & LE=$!
wait $LA || { echo "[FATAL] latent A"; exit 1; }; wait $LE || { echo "[FATAL] latent E"; exit 1; }
grep -h LOAD-GUARD $R/lat_A.log $R/lat_E.log; sha256sum $R/dev600_latent_A.pt $R/dev600_latent_E.pt
gen() { # gpu half sha
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/s7_eval_gate_p6_v36f.py --cond A --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_M7A_v36.pt --zstar $R/dev600_latent_A.pt \
    --items configs/g2v4_dev600_h$2.json --order_sha $3 --out $O/gen_A_h$2.jsonl > $O/gen_A_h$2.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/s7_eval_gate_p6_v36f.py --cond E1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_M7E_v36.pt --zsrc $R/dev600_latent_E.pt \
    --items configs/g2v4_dev600_h$2.json --order_sha $3 --out $O/gen_E1_h$2.jsonl > $O/gen_E1_h$2.log 2>&1 || return 1
  echo "[GEN-DONE] h$2 $(date '+%T')"
}
gen $U5 0 $H0 & G0=$!; gen $U6 1 $H1 & G1=$!
wait $G0 || { echo "[FATAL] gen h0"; exit 1; }; wait $G1 || { echo "[FATAL] gen h1"; exit 1; }
grep -h "PROJ-V2" $O/gen_A_h0.log $O/gen_E1_h0.log
cat $O/gen_A_h0.jsonl $O/gen_A_h1.jsonl > $O/dev_gen_A.jsonl; cat $O/gen_E1_h0.jsonl $O/gen_E1_h1.jsonl > $O/dev_gen_E1.jsonl
$PY - <<'PY'
import json
q = set(json.load(open("results_s7/p6_item_order.json"))["indices"])
for f in ("results_s7/p6r/g2v4dev/dev_gen_A.jsonl", "results_s7/p6r/g2v4dev/dev_gen_E1.jsonl"):
    ids = [json.loads(l)["id"] for l in open(f, encoding="utf-8")]
    print("[CHECK]", f, len(ids), len(set(ids)), set(ids) == q)
PY
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/s7_eval_score_p6_v36.py --gen_A $O/dev_gen_A.jsonl --gen_C results_s7/p6r/p6_gen_C.jsonl --gen_E1 $O/dev_gen_E1.jsonl --gen_B1 results_s7/p6r/p6_gen_B1.jsonl --gen_dir $O --order results_s7/p6_item_order.json --out_json $O/dev_score.json --out_md $O/dev_score.md" > $O/score.log 2>&1
echo "[SCORE] rc=$?"
sha256sum $O/dev_gen_A.jsonl $O/dev_gen_E1.jsonl $O/dev_score.json
$PY -c "import json;d=json.load(open('$O/dev_score.json'));g=d['gates'];print('[G4]',g['Gate_4']['strict']);print('[G3]',g['Gate_3']['strict']);print('[ACC]',d['condition_accuracies'])"
echo "=== m7 dev end $(date '+%F %T') ==="

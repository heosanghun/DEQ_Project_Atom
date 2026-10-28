#!/usr/bin/env bash
# scripts/cc_g2v3_dev_score_v2.sh - PREREG_S7_G2v3.1 §3.2 dev check + scoring (CC). Run after g2v3dev_gen_A_v2.jsonl reaches 600.
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
echo "=== dev score v2 $(date '+%F %T') ==="
nice -n 19 bash -c "ulimit -v 62914560; $PY - <<'PY'
import json
q = set(json.load(open('results_s7/p6_item_order.json'))['indices'])
ok = True
for f in ('results_s7/p6r/g2v3dev_gen_A_v2.jsonl', 'results_s7/p6r/g2v3dev_gen_E.jsonl', 'results_s7/p6r/p6_gen_C.jsonl'):
    rows, ids, bad = 0, set(), 0
    for line in open(f, encoding='utf-8'):
        try: r = json.loads(line); ids.add(r['id']); rows += 1
        except Exception: bad += 1
    good = rows == 600 and len(ids) == 600 and bad == 0 and ids == q
    ok &= good
    print(f'[CHECK] {f}: rows {rows} unique {len(ids)} json_err {bad} ids==order {ids == q} -> {good}')
print('[CHECK] ALL', 'PASS' if ok else 'FAIL')
raise SystemExit(0 if ok else 1)
PY"
[ $? -eq 0 ] || { echo "[FATAL] dev check failed; no scoring"; exit 1; }
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/s7_eval_score_p6_v36.py --gen_A results_s7/p6r/g2v3dev_gen_A_v2.jsonl --gen_C results_s7/p6r/p6_gen_C.jsonl --gen_E1 results_s7/p6r/g2v3dev_gen_E.jsonl --gen_B1 results_s7/p6r/p6_gen_B1.jsonl --gen_dir results_s7/p6r --order results_s7/p6_item_order.json --out_json results_s7/p6r/g2v3dev_score_AE_v2.json --out_md results_s7/p6r/g2v3dev_score_AE_v2.md" > raw_s7/g2v3dev_score_AE_v2.log 2>&1
echo "[SCORE] rc=$?"
sha256sum results_s7/p6r/g2v3dev_gen_A_v2.jsonl results_s7/p6r/g2v3dev_gen_E.jsonl results_s7/p6r/g2v3dev_score_AE_v2.json results_s7/p6r/g2v3dev_score_AE_v2.md
$PY -c "import json;d=json.load(open('results_s7/p6r/g2v3dev_score_AE_v2.json'));g=d['gates'];print('[RULE] Gate_4.strict',g['Gate_4']['strict']);print('[RULE] Gate_3.strict',g['Gate_3']['strict']);print('[ACC]',d['condition_accuracies'])"

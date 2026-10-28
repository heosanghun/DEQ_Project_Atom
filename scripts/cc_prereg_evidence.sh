#!/usr/bin/env bash
# scripts/cc_prereg_evidence.sh - raw evidence for PREREG_S7_G2v2 rev3 (AUDIT-P012 re-audit 1 re-FIX 1·2·4). Read-only.
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
echo "=== evidence $(date '+%F %T') host $(hostname) ==="
echo "--- [E1] /home/shoon/cts_e1/configs listing"; ls -la /home/shoon/cts_e1/configs/
echo "--- [E2] server configs/*.json count"; ls configs/*.json | wc -l; ls configs/*.json
echo "--- [E3] SHA-256 (server)"
sha256sum configs/zstar_canonical.yaml configs/s7_train_993.json configs/s5_decoder.yaml \
  cts/deq/transition_inject.py cts/deq/zstar_canonical.py cts/deq/zstar_canonical_lowest.py cts/deq/zstar_inject_v2.py \
  scripts/s7_zstar_lowest_run.py scripts/g2v2_zstar_run.py scripts/s7_eval_gate_p6_v36d.py scripts/s7_eval_gate_p6_v36e.py \
  scripts/s7_train_decoder_v36.py scripts/s7_eval_score_p6_v36.py scripts/s5_eval_grader_v2.py scripts/check_p6_decoder_version.sh \
  scripts/cc_g2v2_train_eval.sh scripts/cc_g2v2_compute.sh scripts/cc_g2v2_rebalance.sh \
  results_s7/zsrc_train1000_P4.pt results_s7/zsrc_heldout600.pt results_s7/decoder_E_v36.pt results_s7/anchor_ref_gpu1_check.pt \
  results_s7/p6_item_order.json results_s7/p6r/p6_gen_C.jsonl results_s7/p6r/p6_gen_B1.jsonl results_s7/p6r/p6_gen_E1.jsonl \
  /home/shoon/cts_e1/scratch/harness_A_v1.py /home/shoon/checkpoints/m1_final_frozen.pt configs/g2v2_fresh1000.json
echo "--- [E4] library versions"; $PY -c 'import torch,transformers,numpy;print("torch",torch.__version__,"transformers",transformers.__version__,"numpy",numpy.__version__)'
echo "--- [E5] tmux sessions"; tmux ls
echo "--- [E6] chain master log head"; head -3 raw_s7/g2v2_train_eval_master.log; tail -2 raw_s7/g2v2_train_eval_master.log
echo "--- [E7] item-list search over server json files (size < 20 MB) vs configs/g2v2_fresh1000.json"
$PY - <<'PYEOF'
import json, os
fresh = set(json.load(open("configs/g2v2_fresh1000.json"))["indices"])
roots = ["/home/shoon/cts_e1", "/home/shoon/cts_e1_s5"]
skip_dirs = {".git", "node_modules", ".venv", "__pycache__", ".cache", "site-packages"}
n_files = n_lists = 0
hits = {}
def collect(o, out, depth=0):
    if depth > 4: return
    if isinstance(o, list) and o and all(isinstance(x, int) for x in o[:50]) and len(o) <= 6000:
        out.update(x for x in o if isinstance(x, int))
    elif isinstance(o, dict):
        for k, v in o.items():
            collect(v, out, depth + 1)
            if isinstance(k, str) and k[:1] == "q" and k[1:5].isdigit():
                out.add(int(k[1:5]))
for root in roots:
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip_dirs]
        for fn in fns:
            if not fn.endswith(".json"): continue
            p = os.path.join(dp, fn)
            try:
                if os.path.getsize(p) > 20_000_000: continue
                d = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            n_files += 1
            s = set(); collect(d, s)
            s = {x for x in s if 0 <= x < 5000}
            if s: n_lists += 1
            ov = s & fresh
            if ov and not p.endswith(("g2v2_fresh1000.json", "g2v2_fresh1000_shard0.json", "g2v2_fresh1000_shard1.json", "g2v2_fresh1000_shard2.json")):
                hits[p] = len(ov)
print(f"json files read {n_files}, with integer lists/keys {n_lists}")
print(f"files overlapping fresh1000: {len(hits)}")
for p, n in sorted(hits.items(), key=lambda t: -t[1])[:40]:
    print(f"  {n:5d}  {p}")
PYEOF
echo "--- [E8] v36e path tests"
CUDA_VISIBLE_DEVICES='' $PY scripts/s7_eval_gate_p6_v36e.py --cond A --decoder_ckpt /tmp/decoder_A_v36.pt --order_sha 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c --out results_s7/p6r/smoke/v36e_t1.jsonl --device cpu 2>&1 | grep -E 'FATAL|Provided' | head -2
CUDA_VISIBLE_DEVICES='' $PY scripts/s7_eval_gate_p6_v36e.py --cond A --decoder_ckpt results_s7/g2v2/decoder_Zz_v36.pt --order_sha 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c --out results_s7/p6r/smoke/v36e_t2.jsonl --device cpu 2>&1 | grep -E 'FileNotFound' | head -1
ls -l results_s7/p6r/smoke/v36e_smokeA.jsonl; sha256sum results_s7/p6r/smoke/v36e_smokeA.jsonl
echo "=== evidence end $(date '+%F %T') ==="

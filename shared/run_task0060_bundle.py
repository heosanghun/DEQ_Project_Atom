#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0060_bundle.py - Bundled execution for TASK-0060 diagnosis.
Runs CPU-based format diagnosis on P6 generation jsonl files and collects
Replay progress + GPU telemetry via a single SSH call.
Uses base64-encoded Python script to prevent bash string escaping issues.
"""

import base64
import os
import subprocess
import sys

def main():
    root = r"d:\AI\cts"
    out_file = os.path.join(root, "scratch", "task0060_out.txt")
    os.makedirs(os.path.join(root, "scratch"), exist_ok=True)

    diag_code = """
import json, os, statistics, sys
sys.path.insert(0, '/home/shoon/cts_e1_s5')
from scripts.s5_eval_grader_v2 import grade_prediction

candidates = [
    '/home/shoon/system1_5/data/stage2/math_train_prompts_5000.jsonl',
    '/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl',
    '/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl'
]
dataset = None
for c in candidates:
    if os.path.exists(c):
        with open(c, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        break

order_path = '/home/shoon/cts_e1_s5/results_s7/p6_item_order.json'
with open(order_path, 'r', encoding='utf-8') as f:
    item_order = json.load(f)

conditions = ['A', 'B1', 'C', 'E1', 'F']
results = {}
repr_examples = {}

for cond in conditions:
    path = f'/home/shoon/cts_e1_s5/results_s7/p6_gen_{cond}.jsonl'
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    first_keys = list(records[0].keys()) if records else []
    has_tok = any(k in first_keys for k in ['tokens', 'token_count'])
    tok_key = 'tokens' if 'tokens' in first_keys else ('token_count' if 'token_count' in first_keys else None)
    
    c_boxed = 0
    c_512 = 0
    strict_c = 0
    lenient_c = 0
    lengths = []
    
    for rec in records:
        g_text = rec.get('generation', '')
        qid = int(rec['id'])
        gold = dataset[qid]['solution'] if dataset else ''
        
        if r'\\boxed{' in g_text:
            c_boxed += 1
            
        if tok_key and rec.get(tok_key) == 512:
            c_512 += 1
            
        lengths.append(len(g_text))
        
        sm, lm, bv, lv = grade_prediction(g_text, gold)
        if sm:
            strict_c += 1
        if lm:
            lenient_c += 1
            
        if cond in ['C', 'A', 'E1'] and (cond not in repr_examples) and (not sm) and lm:
            repr_examples[cond] = {
                'qid': qid,
                'gold': gold,
                'lenient_val': lv,
                'tail_300': g_text[-300:]
            }
            
    med_l = statistics.median(lengths) if lengths else 0
    
    results[cond] = {
        'field_names': first_keys,
        'count_boxed': c_boxed,
        'count_512': c_512 if tok_key else '필드 없음',
        'strict_correct': strict_c,
        'lenient_correct': lenient_c,
        'median_length': med_l,
        'total': len(records)
    }

print('=== DIAG_RESULTS_JSON_START ===')
print(json.dumps(results, indent=2, ensure_ascii=False))
print('=== DIAG_RESULTS_JSON_END ===')
print('=== REPR_EXAMPLES_JSON_START ===')
print(json.dumps(repr_examples, indent=2, ensure_ascii=False))
print('=== REPR_EXAMPLES_JSON_END ===')
"""

    b64_diag = base64.b64encode(diag_code.encode("utf-8")).decode("ascii")

    remote_cmd = (
        "set -u; "
        "echo '================================================================================'; "
        "echo '=== TASK-0060: BUNDLED CPU FORMAT DIAGNOSIS & REPLAY TELEMETRY ==='; "
        "echo \"Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')\"; "
        "echo '================================================================================'; "
        "echo '=== 1. CPU FORMAT DIAGNOSIS (p6_gen_*.jsonl) ==='; "
        "cd /home/shoon/cts_e1_s5; "
        "ulimit -v 62914560; "
        f"nice -n 19 env CUDA_VISIBLE_DEVICES='' /home/shoon/venv_eval/bin/python -c \"import base64; exec(base64.b64decode('{b64_diag}').decode('utf-8'))\"; "
        "echo '================================================================================'; "
        "echo '=== 2. PROMPT TEMPLATE & GENERATION PARAMS (from scripts/s7_eval_gate_p6_v36b.py) ==='; "
        "sed -n '495,508p' scripts/s7_eval_gate_p6_v36b.py; "
        "sed -n '265,274p' scripts/s7_eval_gate_p6_v36b.py; "
        "sed -n '315,316p' scripts/s7_eval_gate_p6_v36b.py; "
        "sed -n '406p' scripts/s7_eval_gate_p6_v36b.py; "
        "echo '================================================================================'; "
        "echo '=== 3. ①-a REPLAY (GPU 1) PROGRESS & GPU TELEMETRY ==='; "
        "echo '--- Replay Header Progress ---'; "
        "grep -oE '^\\[[0-9]{4}/0325\\] q[0-9]+_a[0-9.]+' raw_s7/kstar_replay_gpu1.log | tail -5; "
        "echo '--- Replay Last Header Number ---'; "
        "grep -oE '^\\[[0-9]{4}/0325\\]' raw_s7/kstar_replay_gpu1.log | tail -1; "
        "echo '--- Replay Total Header Count ---'; "
        "grep -cE '^\\[[0-9]{4}/0325\\]' raw_s7/kstar_replay_gpu1.log; "
        "echo '--- Replay Output Tensor Info ---'; "
        "ls -la --full-time results_s7/zstar_s7_a0.25_train325_kstar_replay.pt 2>/dev/null; "
        "echo '--- GPU 0~7 Telemetry ---'; "
        "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv; "
        "echo '--- GPU Compute Apps ---'; "
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv; "
        "echo '================================================================================'; "
        "echo '=== TASK-0060 BUNDLE FINISHED ==='; "
        "echo \"Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')\"; "
        "echo '================================================================================'"
    )

    cmd = [
        sys.executable,
        os.path.join(root, "shared", "cts_ssh.py"),
        "ssh",
        remote_cmd,
        "--purpose",
        "task0060_bundled_diagnosis",
    ]

    print(f"[TASK-0060] Executing bundled diagnosis SSH...")
    p = subprocess.run(cmd, cwd=root, capture_output=True, text=False)

    stdout_bytes = p.stdout
    stderr_bytes = p.stderr

    with open(out_file, "wb") as f:
        f.write(stdout_bytes)

    print(f"[TASK-0060] Exit code: {p.returncode}")
    print(f"[TASK-0060] Output bytes: {len(stdout_bytes)}, saved to {out_file}")
    if stderr_bytes:
        print(f"[TASK-0060] Stderr:\n{stderr_bytes.decode('utf-8', errors='replace')}")

    if p.returncode != 0:
        print("[ERROR] SSH command failed!")
        sys.exit(p.returncode)

if __name__ == "__main__":
    main()

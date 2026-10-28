#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0064_bundle.py - Bundled execution for TASK-0064 diagnosis.
Diagnoses stop token set in P6 generation (scripts/s7_eval_gate_p6_v36b.py),
inspects tokenizer special tokens and subwords for <turn|> and <end_of_turn>,
analyzes all results_s7/p6_gen_*.jsonl files for premature truncation,
greps training scripts, and collects GPU telemetry in a single SSH call.
Uses base64-encoded Python script to prevent bash string escaping issues.
"""

import base64
import os
import subprocess
import sys

def make_remote_script() -> str:
    script = '''# -*- coding: utf-8 -*-
import collections
import json
import os
import sys
import traceback

# 1. Load tokenizer exactly as s7_train_decoder_v36.py / s7_eval_gate_p6_v36b.py
tok_paths = [
    "/home/shoon/models/gemma4-e4b-it",
    "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377",
]
loaded_path = None
tokenizer = None
for tp in tok_paths:
    print(f"[TOKENIZER-CHECK] candidate: {tp}, exists: {os.path.exists(tp)}")
    if os.path.exists(tp):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tp, trust_remote_code=True)
            loaded_path = tp
            print(f"[TOKENIZER-LOADED] successfully loaded from: {tp}")
            break
        except Exception as e:
            print(f"[TOKENIZER-ERROR] failed on {tp}: {repr(e)}")
            traceback.print_exc()

if tokenizer is None:
    print(f"[FATAL] Failed to load tokenizer from {tok_paths}")
    sys.exit(1)

# Item 1 calculations
eos_id = tokenizer.eos_token_id
turn_ids = tokenizer.encode("<turn|>", add_special_tokens=False)
turn_toks = tokenizer.convert_ids_to_tokens(turn_ids)

end_turn_ids = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
end_turn_toks = tokenizer.convert_ids_to_tokens(end_turn_ids)

eos_set = {tokenizer.eos_token_id}
for special in ["<turn|>", "<end_of_turn>"]:
    t_ids = tokenizer.encode(special, add_special_tokens=False)
    if t_ids:
        eos_set.update(t_ids)

sorted_eos = sorted(list(eos_set))
eos_pieces = {tid: tokenizer.convert_ids_to_tokens([tid])[0] for tid in sorted_eos}

legit_ids = {tokenizer.eos_token_id, *turn_ids}
suspect_ids = set(sorted_eos) - legit_ids

# Special tokens containing 'turn'
all_special = getattr(tokenizer, "all_special_tokens", [])
special_turn = [t for t in all_special if "turn" in t]
special_turn_map = {t: tokenizer.encode(t, add_special_tokens=False) for t in special_turn}

vocab = tokenizer.get_vocab()
turn_vocab_items = [(k, v) for k, v in vocab.items() if "turn" in k.lower() or "turn" in k]
turn_vocab_sorted = sorted(turn_vocab_items, key=lambda x: x[1])

print("=== TOKENIZER_DIAG_START ===")
diag_summary_1 = {
    "loaded_path": loaded_path,
    "eos_token_id": eos_id,
    "eos_token_piece": tokenizer.convert_ids_to_tokens([eos_id])[0],
    "turn_ids": turn_ids,
    "turn_tokens": turn_toks,
    "end_of_turn_ids": end_turn_ids,
    "end_of_turn_tokens": end_turn_toks,
    "eos_set_sorted": sorted_eos,
    "eos_set_pieces": eos_pieces,
    "legit_ids": sorted(list(legit_ids)),
    "suspect_ids": sorted(list(suspect_ids)),
    "special_tokens_with_turn": special_turn_map,
    "vocab_tokens_with_turn_count": len(turn_vocab_sorted),
    "vocab_tokens_with_turn_sample": turn_vocab_sorted[:15]
}
print(json.dumps(diag_summary_1, indent=2, ensure_ascii=False))
print("=== TOKENIZER_DIAG_END ===")

# Item 2: Diagnose each p6_gen_*.jsonl file
conditions = ["A", "B1", "B2", "C", "D1", "D21", "E1", "E21", "F"]
results_by_cond = {}

for cond in conditions:
    path = f"/home/shoon/cts_e1_s5/results_s7/p6_gen_{cond}.jsonl"
    if not os.path.exists(path):
        continue
    
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    
    total_lines = len(records)
    sub_512_records = []
    for rec in records:
        tok_len = rec.get("tokens")
        if tok_len is not None and tok_len < 512:
            sub_512_records.append(rec)
    
    sub_512_count = len(sub_512_records)
    
    # Re-tokenize and examine last tokens
    last_token_counts = collections.Counter()
    suspect_count = 0
    ends_with_special_char_count = 0
    last20_has_special_char_count = 0
    suspect_samples = []
    
    for rec in sub_512_records:
        g_text = rec.get("generation", "")
        qid = rec.get("id")
        tok_field = rec.get("tokens")
        
        enc_ids = tokenizer.encode(g_text, add_special_tokens=False)
        last_id = enc_ids[-1] if enc_ids else None
        if last_id is not None:
            last_token_counts[last_id] += 1
            
            if last_id in suspect_ids:
                suspect_count += 1
                stripped = g_text.rstrip()
                ends_with_spec = stripped.endswith(("_", "<", ">")) if len(stripped) > 0 else False
                if ends_with_spec:
                    ends_with_special_char_count += 1
                
                last20 = stripped[-20:] if len(stripped) >= 20 else stripped
                has_spec_last20 = any(c in ("_", "<", ">") for c in last20)
                if has_spec_last20:
                    last20_has_special_char_count += 1
                
                if len(suspect_samples) < 5:
                    last_piece = tokenizer.convert_ids_to_tokens([last_id])[0]
                    suspect_samples.append({
                        "id": qid,
                        "tokens": tok_field,
                        "last_id": last_id,
                        "last_piece": last_piece,
                        "tail_200": g_text[-200:],
                        "ends_with_special_char": ends_with_spec
                    })
    
    # Top 15 last tokens
    top15_raw = last_token_counts.most_common(15)
    top15 = []
    for tid, cnt in top15_raw:
        p = tokenizer.convert_ids_to_tokens([tid])[0]
        top15.append({
            "id": tid,
            "piece": p,
            "count": cnt,
            "is_eos_set": tid in eos_set,
            "is_suspect": tid in suspect_ids
        })
    
    results_by_cond[cond] = {
        "path": path,
        "total_lines": total_lines,
        "sub_512_count": sub_512_count,
        "sub_512_ratio": round(sub_512_count / total_lines, 4) if total_lines > 0 else 0,
        "suspect_stop_count": suspect_count,
        "suspect_stop_ratio": round(suspect_count / sub_512_count, 4) if sub_512_count > 0 else 0,
        "ends_with_special_char_count": ends_with_special_char_count,
        "last20_has_special_char_count": last20_has_special_char_count,
        "top15_last_tokens": top15,
        "suspect_samples": suspect_samples[:3]
    }

print("=== JSONL_DIAG_START ===")
print(json.dumps(results_by_cond, indent=2, ensure_ascii=False))
print("=== JSONL_DIAG_END ===")
'''
    return script

def main():
    root = r"d:\AI\cts"
    out_file = os.path.join(root, "scratch", "task0064_out.txt")
    os.makedirs(os.path.join(root, "scratch"), exist_ok=True)

    diag_code = make_remote_script()
    b64_diag = base64.b64encode(diag_code.encode("utf-8")).decode("ascii")

    remote_cmd = (
        "set -u; "
        "echo '================================================================================'; "
        "echo '=== TASK-0064: BUNDLED STOP TOKEN DIAGNOSIS & REASONING IMPACT ==='; "
        "echo \"Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')\"; "
        "echo '================================================================================'; "
        "echo '=== 1. CPU TOKENIZER & GENERATION DIAGNOSIS ==='; "
        "cd /home/shoon/cts_e1_s5; "
        "ulimit -v 62914560; "
        f"nice -n 19 env CUDA_VISIBLE_DEVICES='' /home/shoon/system1_5/.venv/bin/python -c \"import base64; exec(base64.b64decode('{b64_diag}').decode('utf-8'))\"; "
        "echo '================================================================================'; "
        "echo '=== 2. GREP TRAINING SCRIPT (scripts/s7_train_decoder_v36.py) ==='; "
        "grep -n 'end_of_turn' scripts/s7_train_decoder_v36.py || echo 'NO MATCH for end_of_turn'; "
        "grep -n 'turn|' scripts/s7_train_decoder_v36.py || echo 'NO MATCH for turn|'; "
        "grep -n 'eos_set' scripts/s7_train_decoder_v36.py || echo 'NO MATCH for eos_set'; "
        "grep -n 'eos_token' scripts/s7_train_decoder_v36.py | head -10 || echo 'NO MATCH for eos_token'; "
        "echo '================================================================================'; "
        "echo '=== 3. GENERATOR CODE SNIPPET (scripts/s7_eval_gate_p6_v36b.py:404-411) ==='; "
        "sed -n '404,411p' scripts/s7_eval_gate_p6_v36b.py; "
        "echo '================================================================================'; "
        "echo '=== 4. HOST TELEMETRY ==='; "
        "echo 'Uptime:'; "
        "uptime -s; "
        "echo 'Tmux sessions:'; "
        "tmux ls 2>&1 || true; "
        "echo 'GPU Telemetry:'; "
        "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv; "
        "echo 'GPU Compute Apps:'; "
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv; "
        "echo '================================================================================'; "
        "echo '=== TASK-0064 BUNDLE FINISHED ==='; "
        "echo \"Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')\"; "
        "echo '================================================================================'"
    )

    cmd = [
        sys.executable,
        os.path.join(root, "shared", "cts_ssh.py"),
        "ssh",
        remote_cmd,
        "--purpose",
        "task0064_bundled_diagnosis",
    ]

    print(f"[TASK-0064] Executing bundled diagnosis SSH via cts_ssh.py...")
    p = subprocess.run(cmd, cwd=root, capture_output=True, text=False)

    stdout_bytes = p.stdout
    stderr_bytes = p.stderr

    with open(out_file, "wb") as f:
        f.write(stdout_bytes)

    print(f"[TASK-0064] Exit code: {p.returncode}")
    print(f"[TASK-0064] Output bytes: {len(stdout_bytes)}, saved to {out_file}")
    if stderr_bytes:
        print(f"[TASK-0064] Stderr:\n{stderr_bytes.decode('utf-8', errors='replace')}")

    if p.returncode != 0:
        print("[ERROR] SSH command failed!")
        sys.exit(p.returncode)

if __name__ == "__main__":
    main()

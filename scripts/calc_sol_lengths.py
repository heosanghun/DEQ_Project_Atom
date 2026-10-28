import json
import os
import sys
import numpy as np

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
for p in [BASE_S5_DIR, os.path.join(BASE_S5_DIR, "scripts"), SYS_DIR, "/home/shoon/cts_e1/scratch"]:
    if p not in sys.path:
        sys.path.insert(0, p)

from s5_train_decoder_v2 import load_tokenizer, find_boxed_char_span

items_path = os.path.join(BASE_S5_DIR, "data", "s5_items_400.json")
dataset_path = "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl"

with open(items_path, "r", encoding="utf-8") as f:
    items_data = json.load(f)

train_indices = [int(x) for x in items_data["train_indices"]]
assert len(train_indices) == 320

dataset_records = {}
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            idx = int(rec.get("index", rec.get("question_idx", rec.get("id", len(dataset_records)))))
            dataset_records[idx] = rec

tokenizer = load_tokenizer()
prefix_len = 73

sol_lens = []
full_lens = []
boxed_end_lens = []  # Token index where \boxed ends in full_seq
missing_boxed = []
truncated_at_512 = []
truncated_at_400 = []
truncated_at_1024 = []

for q in train_indices:
    sol_text = dataset_records[q]["solution"]
    enc = tokenizer(sol_text, return_offsets_mapping=True, add_special_tokens=False)
    sol_ids = enc["input_ids"]
    offsets = enc.get("offset_mapping", [])
    
    sol_len = len(sol_ids) + 1  # +1 for EOS
    total_seq_len = prefix_len + sol_len
    sol_lens.append(sol_len)
    full_lens.append(total_seq_len)
    
    span = find_boxed_char_span(sol_text)
    if span is None:
        missing_boxed.append(q)
        boxed_end_token = total_seq_len
    else:
        # Find token index where span[1] is located
        boxed_end_tok_rel = None
        for tok_i, (s, e) in enumerate(offsets):
            if e >= span[1]:
                boxed_end_tok_rel = tok_i
                break
        if boxed_end_tok_rel is None:
            boxed_end_tok_rel = len(sol_ids)
        boxed_end_token = prefix_len + boxed_end_tok_rel + 1
    
    boxed_end_lens.append(boxed_end_token)
    
    if boxed_end_token > 400:
        truncated_at_400.append((q, boxed_end_token, total_seq_len))
    if boxed_end_token > 512:
        truncated_at_512.append((q, boxed_end_token, total_seq_len))
    if boxed_end_token > 1024:
        truncated_at_1024.append((q, boxed_end_token, total_seq_len))

sol_arr = np.array(sol_lens)
full_arr = np.array(full_lens)
boxed_arr = np.array(boxed_end_lens)

print(f"Total train items: {len(train_indices)}")
print(f"Prefix tokens: {prefix_len}")
print("\n--- Solution Token Length Distribution ---")
print(f"Min:  {np.min(sol_arr)}")
print(f"Mean: {np.mean(sol_arr):.2f}")
print(f"p50:  {np.percentile(sol_arr, 50):.1f}")
print(f"p90:  {np.percentile(sol_arr, 90):.1f}")
print(f"p95:  {np.percentile(sol_arr, 95):.1f}")
print(f"p99:  {np.percentile(sol_arr, 99):.1f}")
print(f"Max:  {np.max(sol_arr)}")

print("\n--- Full Sequence Length Distribution (73 + sol + EOS) ---")
print(f"Min:  {np.min(full_arr)}")
print(f"Mean: {np.mean(full_arr):.2f}")
print(f"p50:  {np.percentile(full_arr, 50):.1f}")
print(f"p90:  {np.percentile(full_arr, 90):.1f}")
print(f"p95:  {np.percentile(full_arr, 95):.1f}")
print(f"p99:  {np.percentile(full_arr, 99):.1f}")
print(f"Max:  {np.max(full_arr)}")

print("\n--- Boxed End Token Distribution ---")
print(f"Max boxed end token: {np.max(boxed_arr)}")
print(f"p95 boxed end token: {np.percentile(boxed_arr, 95):.1f}")
print(f"p99 boxed end token: {np.percentile(boxed_arr, 99):.1f}")
print(f"Missing boxed count: {len(missing_boxed)}")

print(f"\nItems where boxed cut off if max_seq_len=400: count={len(truncated_at_400)}")
if truncated_at_400:
    print(f"  Indices: {[x[0] for x in truncated_at_400[:10]]}")

print(f"\nItems where boxed cut off if max_seq_len=512: count={len(truncated_at_512)}")
if truncated_at_512:
    print(f"  Indices: {[x[0] for x in truncated_at_512]}")

print(f"\nItems where boxed cut off if max_seq_len=1024: count={len(truncated_at_1024)}")
if truncated_at_1024:
    print(f"  Indices: {[x[0] for x in truncated_at_1024]}")

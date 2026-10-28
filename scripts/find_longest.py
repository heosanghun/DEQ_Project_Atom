import json
import os
from transformers import AutoTokenizer

with open("/home/shoon/cts_e1_s5/configs/s5_heldout_80.json") as f:
    heldout = set(json.load(f)["indices"])

with open("/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl") as f:
    lines = [json.loads(line) for line in f]

train_indices = [i for i in range(400) if i not in heldout]

tok = AutoTokenizer.from_pretrained("/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377")

lengths = []
for idx in train_indices:
    sol = lines[idx].get("solution") or lines[idx].get("answer") or ""
    tokens = tok.encode(sol, add_special_tokens=False)
    lengths.append((len(tokens), idx))

lengths.sort(reverse=True)
print("Top 10 longest train items (length, q_idx):")
for l, idx in lengths[:10]:
    print(f"  q_idx={idx:04d}: {l} tokens")

#!/usr/bin/env python3
"""scripts/g2v2_select_fresh.py - G2v2 fresh evaluation item selection (CC, 2026-10-08, PREREG_S7_G2v2 §2).

Reads only q_idx and the `level` field of data/stage2/math_train_prompts_5000.jsonl (no prompt/solution text).
Exclusion set (written to configs/g2v2_exclude.json):
  (a) union of indices in every config list given on the command line (server configs + local configs),
  (b) data/s5_items_400.json: canonical_20_indices (anchors 0, 4, ..., 76) + train_indices (320) + heldout_indices (80),
  (c) q_idx 0..99 (early experiments used dataset[:32], ds[:80] style slices),
  (d) level == "Level ?".
Selection: sorted remaining q_idx list L; numpy.random.default_rng(20261009).choice(len(L), 1000, replace=False);
the chosen q_idx sorted ascending -> configs/g2v2_fresh1000.json {"indices": [...]}.
Also writes 3 round-robin shards configs/g2v2_fresh1000_shard{0,1,2}.json (shard k = sorted list[k::3]).
"""
import glob
import hashlib
import json
import sys

import numpy as np

DATA = "data/stage2/math_train_prompts_5000.jsonl"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def indices_of(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    xs = d if isinstance(d, list) else next((d[k] for k in ("indices", "items", "valid_indices", "q_idx") if isinstance(d, dict) and isinstance(d.get(k), list)), [])
    return [int(x) for x in xs if str(x).isdigit()]


def main():
    cfg_files = sorted(set(sys.argv[1:]))
    excl, src = set(), {}
    for f in cfg_files:
        xs = indices_of(f)
        if xs:
            src[f] = len(xs)
            excl |= set(xs)
    s5 = json.load(open("data/s5_items_400.json", encoding="utf-8"))
    anchors = s5["canonical_20_indices"]
    excl |= set(int(x) for x in anchors) | set(int(x) for x in s5["train_indices"]) | set(int(x) for x in s5["heldout_indices"])
    excl |= set(range(100))
    levels = [json.loads(l)["level"] for l in open(DATA, encoding="utf-8")]
    excl |= {i for i, lv in enumerate(levels) if lv == "Level ?"}
    json.dump({"indices": sorted(excl), "sources": src, "anchors": sorted(int(x) for x in anchors),
               "rule": "configs union + s5_items_400 (anchors 20, train 320, heldout 80) + q_idx<100 + Level ?"}, open("configs/g2v2_exclude.json", "w"))
    L = sorted(i for i in range(len(levels)) if i not in excl)
    pick = np.random.default_rng(20261009).choice(len(L), 1000, replace=False)
    chosen = sorted(L[int(j)] for j in pick)
    json.dump({"indices": chosen}, open("configs/g2v2_fresh1000.json", "w"))
    for k in range(3):
        json.dump({"indices": chosen[k::3]}, open(f"configs/g2v2_fresh1000_shard{k}.json", "w"))
    print("numpy", np.__version__, "| data SHA", sha(DATA))
    print("excluded", len(excl), "| remaining", len(L), "| chosen", len(chosen), "| overlap with exclusion", len(set(chosen) & excl))
    for p in ["configs/g2v2_exclude.json", "configs/g2v2_fresh1000.json"] + [f"configs/g2v2_fresh1000_shard{k}.json" for k in range(3)]:
        print(sha(p), p)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# scripts/cc_g2v2_rebalance.sh - G2v2 compute rebalancer (CC, 2026-10-08)
# GPUs 0/1/3 run ~3x slower (throttle reasons 0x8C: HW slowdown + power brake; same PCIe group as faulty GPU 2).
# When a fast shard (3,4,5 on GPU 5,6,7) finishes, start a helper on that GPU that computes the remaining items of a slow
# shard (0,1,2) in reverse order into a separate file. When the union of all files covers the 1,600 items, stop all
# compute processes, merge (first file wins per key; device per key recorded), write G2V2_COMPUTE_DONE.flag.
set -uo pipefail
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
FAST=(GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b)
LOG=raw_s7/g2v2_rebalance.log
exec > >(tee -a "$LOG") 2>&1
echo "=== rebalance start $(date '+%F %T') ==="
remaining_rev() {  # slow shard index -> write reversed list of items not yet in its out file
  $PY - "$1" <<'PYEOF'
import json, sys, torch, os
i = int(sys.argv[1])
items = json.load(open(f"configs/g2v2_compute_shard{i}.json"))["indices"]
done = set()
p = f"results_s7/g2v2/zstar_zsrcinj_shard{i}.pt"
if os.path.exists(p):
    try: done = set(torch.load(p, map_location="cpu").keys())
    except Exception: pass
rem = [q for q in items if f"q{q:04d}_a0.25" not in done][::-1]
json.dump({"indices": rem}, open(f"configs/g2v2_helper_for_shard{i}.json", "w"))
print(len(rem))
PYEOF
}
covered() {
  $PY - <<'PYEOF'
import json, glob, torch
need = set()
for i in range(6):
    need |= {f"q{q:04d}_a0.25" for q in json.load(open(f"configs/g2v2_compute_shard{i}.json"))["indices"]}
have = set()
for f in glob.glob("results_s7/g2v2/zstar_zsrcinj_shard*.pt") + glob.glob("results_s7/g2v2/zstar_zsrcinj_helper*.pt"):
    try: have |= set(torch.load(f, map_location="cpu").keys())
    except Exception: pass
print(len(need - have))
PYEOF
}
declare -A started
for pair in "3 0" "4 1" "5 2"; do started[$pair]=0; done
while true; do
  for pair in "3 0" "4 1" "5 2"; do
    set -- $pair; fs=$1; ss=$2; gi=$((fs-3))
    if [ "${started[$pair]}" = 0 ] && grep -q "Finished. Total items saved" raw_s7/g2v2_compute_shard$fs.log 2>/dev/null; then
      n=$(remaining_rev $ss)
      echo "[HELPER] fast shard $fs done -> helper for slow shard $ss on ${FAST[$gi]} ($n items) $(date '+%T')"
      CUDA_VISIBLE_DEVICES=${FAST[$gi]} $PY scripts/g2v2_zstar_run.py --items configs/g2v2_helper_for_shard$ss.json --inject zsrc \
        --out results_s7/g2v2/zstar_zsrcinj_helper$ss.pt > raw_s7/g2v2_compute_helper$ss.log 2>&1 &
      started[$pair]=1
    fi
  done
  left=$(covered); echo "[COVER] missing=$left $(date '+%T')"
  [ "$left" = "0" ] && break
  sleep 300
done
echo "[COVER] all 1600 covered; stopping compute processes $(date '+%T')"
pkill -f "scripts/g2v2_zstar_run.py --items configs/g2v2_" || true
sleep 20
$PY - <<'PYEOF'
import json, glob, torch
order = [f"results_s7/g2v2/zstar_zsrcinj_shard{i}.pt" for i in range(6)] + sorted(glob.glob("results_s7/g2v2/zstar_zsrcinj_helper*.pt"))
dev_of = {0: "GPU0", 1: "GPU1", 2: "GPU3", 3: "GPU5", 4: "GPU6", 5: "GPU7"}
help_dev = {"0": "GPU5", "1": "GPU6", "2": "GPU7"}
out, src = {}, {}
for f in order:
    try: d = torch.load(f, map_location="cpu")
    except Exception as e:
        print("[MERGE] skip", f, e); continue
    tag = f.split("_")[-1].replace(".pt", "")
    dev = dev_of[int(tag[5:])] if tag.startswith("shard") else help_dev[tag[6:]]
    for k, v in d.items():
        if k not in out:
            out[k] = v; src[k] = dev
tr = json.load(open("configs/s7_train_1000.json"))["indices"]; dv = json.load(open("configs/g2v2_dev600.json"))["indices"]
def sub(qs): return {f"q{q:04d}_a0.25": out[f"q{q:04d}_a0.25"] for q in qs}
t, d = sub(tr), sub(dv)
torch.save(t, "results_s7/g2v2/zstar_zsrcinj_train1000.pt"); torch.save(d, "results_s7/g2v2/zstar_zsrcinj_dev600.pt")
json.dump(src, open("results_s7/g2v2/zstar_zsrcinj_device_map.json", "w"))
import collections
print("[MERGE] device counts", dict(collections.Counter(src.values())))
for name, dd in (("train1000", t), ("dev600", d)):
    r1 = [v["r_1000"] for v in dd.values()]; rl = [v["r_lowest"] for v in dd.values()]
    print(f"[MERGE] {name}: n={len(dd)} r_1000<1e-2={sum(x<1e-2 for x in r1)} r_lowest<1e-2={sum(x<1e-2 for x in rl)}")
PYEOF
sha256sum results_s7/g2v2/zstar_zsrcinj_train1000.pt results_s7/g2v2/zstar_zsrcinj_dev600.pt
touch raw_s7/G2V2_COMPUTE_DONE.flag
echo "=== rebalance end $(date '+%F %T') ==="

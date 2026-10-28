import os
import sys
import subprocess
import time
import datetime
import math

REMOTE_SCRIPT = r"""
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0046: (A) P5 ANALYSIS & CANONICAL FETCH (B) S7DEQ PREP ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 0. SYSTEM HEALTH & P6_CHAIN INTEGRITY ==="
echo "--- uptime -s ---"
uptime -s
echo "--- uptime ---"
uptime
echo "--- tmux ls ---"
tmux ls 2>&1 || echo "tmux ls failed"
echo "--- raw_s7/p6_chain.log (last 20 lines) ---"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 20 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log not found"
fi

echo "=== 1. PRE-EXECUTION SHA-256 VERIFICATION ==="
SHA_ANALYZE=$(sha256sum scripts/p5_analyze_v5.py | awk '{print $1}')
SHA_PROMOTE=$(sha256sum scripts/p5_promote_v6.py | awk '{print $1}')
SHA_HELDOUT520=$(sha256sum results_s7/zstar_s7_a0.25_heldout520_lowest.pt | awk '{print $1}')
SHA_P5B=$(sha256sum results_s7/zstar_s7_a0.25_heldout520_seed2031.pt | awk '{print $1}')
SHA_CONFIG520=$(sha256sum configs/s7_new_heldout_520.json | awk '{print $1}')
SHA_TRAIN1000=$(sha256sum results_s7/p5_inputs_train1000_v3.json | awk '{print $1}')
SHA_HELDOUT80=$(sha256sum results_s7/p5_inputs_heldout80_v3.json | awk '{print $1}')
SHA_ITEMS74=$(sha256sum configs/s6_items_74.json | awk '{print $1}')
SHA_CANONICAL=$(sha256sum cts/deq/zstar_canonical_lowest.py | awk '{print $1}')

echo "scripts/p5_analyze_v5.py: $SHA_ANALYZE"
echo "scripts/p5_promote_v6.py: $SHA_PROMOTE"
echo "results_s7/zstar_s7_a0.25_heldout520_lowest.pt: $SHA_HELDOUT520"
echo "results_s7/zstar_s7_a0.25_heldout520_seed2031.pt: $SHA_P5B"
echo "configs/s7_new_heldout_520.json: $SHA_CONFIG520"
echo "results_s7/p5_inputs_train1000_v3.json: $SHA_TRAIN1000"
echo "results_s7/p5_inputs_heldout80_v3.json: $SHA_HELDOUT80"
echo "configs/s6_items_74.json: $SHA_ITEMS74"
echo "cts/deq/zstar_canonical_lowest.py: $SHA_CANONICAL"

PRE_CHECK_FAIL=0

if [ "$SHA_ANALYZE" != "f444822620e678cda6139fbe8bc17d57dceb8e95922d7c04950413e67779d704" ]; then
  echo "MISMATCH: scripts/p5_analyze_v5.py"
  PRE_CHECK_FAIL=1
fi

if [ "$SHA_PROMOTE" != "63d9f598b05c0ea1039c73670ae344059dceee7d8a42bee1ea021395d47cb769" ]; then
  echo "MISMATCH: scripts/p5_promote_v6.py"
  PRE_CHECK_FAIL=1
fi

if [ "$SHA_HELDOUT520" != "365ac9a44150546dd424c33f2c5ae9cabc71f474f3f2cfd5e2d042fe6b4061ae" ]; then
  echo "MISMATCH: results_s7/zstar_s7_a0.25_heldout520_lowest.pt"
  PRE_CHECK_FAIL=1
fi

if [ "$SHA_CONFIG520" != "48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988" ]; then
  echo "MISMATCH: configs/s7_new_heldout_520.json"
  PRE_CHECK_FAIL=1
fi

TRAIN1000_PREFIX=$(echo "$SHA_TRAIN1000" | cut -c1-8)
if [ "$TRAIN1000_PREFIX" != "186d5b6c" ]; then
  echo "MISMATCH PREFIX: results_s7/p5_inputs_train1000_v3.json (got $TRAIN1000_PREFIX, expected 186d5b6c)"
  PRE_CHECK_FAIL=1
fi

if [ "$PRE_CHECK_FAIL" -ne 0 ]; then
  echo "ABORTING EXECUTION DUE TO PRE-CHECK SHA MISMATCH"
  exit 1
fi

echo "PRE-CHECK SHA INTEGRITY SATISFIED. PROCEEDING WITH PART (A) EXECUTION."

echo "=== 2. RUNNING PART (A) ITEM 2: NEAREST-NEIGHBOR MATCH RATE (CPU) ==="
export CUDA_VISIBLE_DEVICES=""
ulimit -v 62914560

echo "--- START: p5_analyze_v5.py at $(date '+%Y-%m-%d %H:%M:%S KST') ---"
nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/p5_analyze_v5.py \
  --target_file results_s7/zstar_s7_a0.25_heldout520_lowest.pt \
  --query_file results_s7/zstar_s7_a0.25_heldout520_seed2031.pt \
  --config_520 configs/s7_new_heldout_520.json \
  --nn_mode seedpair \
  --out_json results_s7/p5_seedpair_v5.json \
  --out_md results_s7/p5_seedpair_v5.md
EXIT_ANALYZE=$?
echo "--- END: p5_analyze_v5.py exit code $EXIT_ANALYZE at $(date '+%Y-%m-%d %H:%M:%S KST') ---"

echo "=== 3. RUNNING PART (A) ITEM 3: PROMOTION ASSESSMENT (CPU) ==="
echo "--- START: p5_promote_v6.py at $(date '+%Y-%m-%d %H:%M:%S KST') ---"
nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/p5_promote_v6.py \
  --heldout520 results_s7/zstar_s7_a0.25_heldout520_lowest.pt \
  --heldout80 results_s7/p5_inputs_heldout80_v3.json \
  --train1000 results_s7/p5_inputs_train1000_v3.json \
  --out_json results_s7/p5_promote_v6.json \
  --out_md results_s7/p5_promote_v6.md
EXIT_PROMOTE=$?
echo "--- END: p5_promote_v6.py exit code $EXIT_PROMOTE at $(date '+%Y-%m-%d %H:%M:%S KST') ---"

echo "=== 4. POST-EXECUTION ARTIFACT VERIFICATION ==="
for f in results_s7/p5_seedpair_v5.json results_s7/p5_seedpair_v5.md results_s7/p5_promote_v6.json results_s7/p5_promote_v6.md; do
  if [ -f "$f" ]; then
    sha256sum "$f"
    ls -la --time-style=full-iso "$f"
  else
    echo "FILE NOT FOUND: $f"
  fi
done

echo "=== BEGIN FILE: results_s7/p5_seedpair_v5.json ==="
cat results_s7/p5_seedpair_v5.json
echo "=== END FILE: results_s7/p5_seedpair_v5.json ==="

echo "=== BEGIN FILE: results_s7/p5_promote_v6.json ==="
cat results_s7/p5_promote_v6.json
echo "=== END FILE: results_s7/p5_promote_v6.json ==="

echo "=== BEGIN FILE: results_s7/p5_seedpair_v5.md ==="
cat results_s7/p5_seedpair_v5.md
echo "=== END FILE: results_s7/p5_seedpair_v5.md ==="

echo "=== BEGIN FILE: results_s7/p5_promote_v6.md ==="
cat results_s7/p5_promote_v6.md
echo "=== END FILE: results_s7/p5_promote_v6.md ==="

echo "=== 5. FETCH CANONICAL CODE: cts/deq/zstar_canonical_lowest.py ==="
echo "CANONICAL_SHA: $SHA_CANONICAL"
echo "=== BEGIN BASE64 cts/deq/zstar_canonical_lowest.py ==="
base64 -w 0 cts/deq/zstar_canonical_lowest.py
echo ""
echo "=== END BASE64 cts/deq/zstar_canonical_lowest.py ==="

echo "=== 6. CURRENT GPU STATUS & COMPUTE APPS ==="
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    log_path = os.path.join(root, "shared", "ssh_log.csv")
    out_path = os.path.join(root, "scratch", "task0046_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) > 1:
            last_ts = lines[-1].split(",")[0]
            try:
                last_dt = datetime.datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                diff = (datetime.datetime.now() - last_dt).total_seconds()
                if diff < 65.0:
                    wait_s = math.ceil(65.0 - diff)
                    print(f"Waiting {wait_s}s to satisfy R7-1 cooldown...")
                    time.sleep(wait_s)
            except Exception as e:
                pass

    purpose = "task0046_p5_analysis_and_canonical_fetch"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]

    print("Executing TASK-0046 bundle query via cts_ssh.py...")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"Task 0046 bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

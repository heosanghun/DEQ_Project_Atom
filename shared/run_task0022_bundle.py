import os
import sys
import subprocess

REMOTE_SCRIPT = """
echo "=== STEP 0: WAIT FOR DECODER A TO FINISH (IF RUNNING) ==="
for i in $(seq 1 60); do
  if [ -f /home/shoon/cts_e1_s5/results_s7/decoder_A_v36.pt ] && grep -q "Saved final model to results_s7/decoder_A_v36.pt" /home/shoon/cts_e1_s5/raw_s7/train_decoder_A_v36.log 2>/dev/null; then
    echo "[WAIT] decoder_A_v36.pt detected and saved in log!"
    break
  fi
  echo "[WAIT] ($(date '+%Y-%m-%d %H:%M:%S')) Waiting for decoder_A_v36.pt... ($i/60)"
  tail -n 1 /home/shoon/cts_e1_s5/raw_s7/train_decoder_A_v36.log 2>/dev/null || tail -n 1 /home/shoon/cts_e1_s5/raw_s7/train_p4_prime_suite_v36.log 2>/dev/null
  sleep 30
done

echo "=== 1. DECODER A CHECKPOINT VERIFICATION ==="
cd /home/shoon/cts_e1_s5
ls -lh results_s7/decoder_A_v36.pt
sha256sum results_s7/decoder_A_v36.pt
bash scripts/check_p6_decoder_version.sh results_s7/decoder_A_v36.pt
echo "CHECK_VERSION_RC=$?"

echo "=== 2. DECODER A LOG CITATIONS ==="
grep -C 3 "Saved final model to results_s7/decoder_A_v36.pt" raw_s7/train_decoder_A_v36.log 2>/dev/null || grep -C 3 "Saved final model to results_s7/decoder_A_v36.pt" raw_s7/train_p4_prime_suite_v36.log 2>/dev/null
echo "--- Last 20 lines of train_decoder_A_v36.log ---"
tail -n 20 raw_s7/train_decoder_A_v36.log 2>/dev/null || tail -n 20 raw_s7/train_p4_prime_suite_v36.log 2>/dev/null

echo "=== 3. DECODER D STATUS ==="
ls -lh raw_s7/train_decoder_D_v36.log 2>/dev/null || echo "D log not present"
head -n 20 raw_s7/train_decoder_D_v36.log 2>/dev/null || true
grep -C 2 "Step 0/" raw_s7/train_decoder_D_v36.log 2>/dev/null || true
echo "--- Active Decoder Processes ---"
ps -ef | grep s7_train_decoder | grep -v grep || true

echo "=== 4. P5 & P5B PROGRESS & FLAGS ==="
echo "--- P5 Log (tail 2) ---"
tail -n 2 raw_s7/heldout520_P5.log 2>/dev/null
echo "--- P5B Log (tail 2) ---"
tail -n 2 raw_s7/heldout520_P5B_seed2031.log 2>/dev/null
echo "--- Flags in raw_s7 ---"
ls -la raw_s7/*.flag 2>/dev/null

echo "=== 5. GPU STATUS (0..7) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo "=== 6. CPU SMOKE EVALUATION WITH DECODER A ==="
if [ -f results_s7/decoder_A_v36.pt ]; then
  mkdir -p scratch
  rm -f scratch/p6_smoke_A.jsonl scratch/p6_smoke_B1.jsonl scratch/p6_smoke_F.jsonl scratch/p6_smoke_A_nll.jsonl scratch/p6_smoke_B1_nll.jsonl

  ulimit -v 62914560
  export MALLOC_ARENA_MAX=1

  echo "--- RUNNING CONDITION A (smoke) ---"
  /usr/bin/time -v nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_eval_gate_p6_v36b.py \\
    --cond A --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt \\
    --max_new_tokens 16 --out scratch/p6_smoke_A.jsonl
  echo "EXIT_CODE_A=$?"

  echo "--- RUNNING CONDITION B (shift=1, smoke) ---"
  /usr/bin/time -v nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_eval_gate_p6_v36b.py \\
    --cond B --shift 1 --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt \\
    --max_new_tokens 16 --out scratch/p6_smoke_B1.jsonl
  echo "EXIT_CODE_B=$?"

  echo "--- RUNNING CONDITION F (smoke) ---"
  /usr/bin/time -v nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_eval_gate_p6_v36b.py \\
    --cond F --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt \\
    --max_new_tokens 16 --out scratch/p6_smoke_F.jsonl
  echo "EXIT_CODE_F=$?"

  echo "--- RUNNING CONDITION A (nll mode) ---"
  /usr/bin/time -v nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_eval_gate_p6_v36b.py \\
    --cond A --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt \\
    --max_new_tokens 16 --nll --out scratch/p6_smoke_A_nll.jsonl
  echo "EXIT_CODE_A_NLL=$?"

  echo "--- RUNNING CONDITION B1 (nll mode) ---"
  /usr/bin/time -v nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_eval_gate_p6_v36b.py \\
    --cond B --shift 1 --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt \\
    --max_new_tokens 16 --nll --out scratch/p6_smoke_B1_nll.jsonl
  echo "EXIT_CODE_B1_NLL=$?"

  echo "=== 7. GENERATION & NLL OUTPUT CONTENTS ==="
  echo "--- scratch/p6_smoke_A.jsonl ---"
  cat scratch/p6_smoke_A.jsonl 2>/dev/null || echo "file not found"
  echo "--- scratch/p6_smoke_B1.jsonl ---"
  cat scratch/p6_smoke_B1.jsonl 2>/dev/null || echo "file not found"
  echo "--- scratch/p6_smoke_F.jsonl ---"
  cat scratch/p6_smoke_F.jsonl 2>/dev/null || echo "file not found"
  echo "--- scratch/p6_smoke_A_nll.jsonl ---"
  cat scratch/p6_smoke_A_nll.jsonl 2>/dev/null || echo "file not found"
  echo "--- scratch/p6_smoke_B1_nll.jsonl ---"
  cat scratch/p6_smoke_B1_nll.jsonl 2>/dev/null || echo "file not found"
else
  echo "[FATAL] results_s7/decoder_A_v36.pt does not exist!"
fi
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0022_decoder_a_check_and_cpu_smoke"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]
    print(f"Launching TASK-0022 via cts_ssh: {purpose}...")
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

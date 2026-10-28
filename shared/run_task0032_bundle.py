import os
import sys
import subprocess

REMOTE_SCRIPT = """
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0032 EXECUTION: PROGRESS CHECK & CPU SMOKE WITH DECODER A ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. DECODER A FINAL CHECKPOINT & LOGS ==="
echo "--- Checkpoint existence and details ---"
if [ -f results_s7/decoder_A_v36.pt ]; then
  ls -lh results_s7/decoder_A_v36.pt
  sha256sum results_s7/decoder_A_v36.pt
  bash scripts/check_p6_decoder_version.sh results_s7/decoder_A_v36.pt
  echo "CHECK_VERSION_RC=$?"
else
  echo "RESULTS: results_s7/decoder_A_v36.pt NOT FOUND"
fi

echo "--- train_decoder_A_v36.log: final checkpoint save line ---"
grep -C 3 "Saved final model to results_s7/decoder_A_v36.pt" raw_s7/train_decoder_A_v36.log 2>/dev/null || echo "Saved final model line NOT found"

echo "--- train_decoder_A_v36.log: last 5 step lines ---"
grep "\\[Step " raw_s7/train_decoder_A_v36.log 2>/dev/null | tail -n 5 || echo "No Step lines found"

echo "--- train_decoder_A_v36.log: tail 15 lines ---"
tail -n 15 raw_s7/train_decoder_A_v36.log 2>/dev/null || echo "Log not found"

echo "=== 2. DECODER D STATUS ==="
echo "--- train_decoder_D_v36.log existence and details ---"
if [ -f raw_s7/train_decoder_D_v36.log ]; then
  ls -lh raw_s7/train_decoder_D_v36.log
  echo "--- First 10 lines of train_decoder_D_v36.log ---"
  head -n 10 raw_s7/train_decoder_D_v36.log
  echo "--- First step line of train_decoder_D_v36.log ---"
  grep -m 1 "\\[Step " raw_s7/train_decoder_D_v36.log 2>/dev/null || echo "No Step line found"
  echo "--- Last 5 step lines of train_decoder_D_v36.log ---"
  grep "\\[Step " raw_s7/train_decoder_D_v36.log 2>/dev/null | tail -n 5 || echo "No Step line found"
  echo "--- Tail 10 lines of train_decoder_D_v36.log ---"
  tail -n 10 raw_s7/train_decoder_D_v36.log
else
  echo "raw_s7/train_decoder_D_v36.log does not exist"
fi

echo "--- Active Decoder Processes ---"
ps -ef | grep -E "s7_train_decoder|train_p4_prime" | grep -v grep || echo "No active decoder process"

echo "=== 3. P5B STATUS ==="
echo "--- Flags in raw_s7 ---"
ls -la raw_s7/*P5B*.flag raw_s7/*p5b*.flag 2>/dev/null || ls -la raw_s7/*.flag 2>/dev/null || echo "No flag files found"

echo "--- P5B Output File Key Count (values uninspected) ---"
if [ -f results_s7/zstar_s7_a0.25_heldout520_seed2031.pt ]; then
  ls -lh results_s7/zstar_s7_a0.25_heldout520_seed2031.pt
  /home/shoon/system1_5/.venv/bin/python -c "
import torch
d = torch.load('results_s7/zstar_s7_a0.25_heldout520_seed2031.pt', map_location='cpu')
print('P5B_OUTPUT_TOTAL_KEYS:', len(d.keys()))
"
else
  echo "results_s7/zstar_s7_a0.25_heldout520_seed2031.pt not found"
fi

echo "--- P5B Completion Time / Log Tail ---"
tail -n 10 raw_s7/heldout520_P5b_seed2031.log 2>/dev/null || tail -n 10 raw_s7/heldout520_P5b_seed2031_gpu2.log 2>/dev/null || echo "P5B log not found"

echo "=== 4. SYSTEM RESOURCE STATUS ==="
echo "--- GPU 0..7 Status ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "--- Free Memory ---"
free -h
echo "--- Active Tmux Sessions ---"
tmux list-sessions 2>/dev/null || echo "No tmux sessions"

echo "=== 5. CPU EVALUATION WITH DECODER A (TASK-0032 §2) ==="
if [ -f results_s7/decoder_A_v36.pt ]; then
  mkdir -p scratch
  rm -f scratch/p6_final_A_smoke_A.jsonl scratch/p6_final_A_smoke_B1.jsonl scratch/p6_final_A_smoke_F.jsonl scratch/p6_final_A_smoke_A_nll.jsonl scratch/p6_final_A_smoke_B1_nll.jsonl

  ulimit -v 62914560
  export MALLOC_ARENA_MAX=1
  export CUDA_VISIBLE_DEVICES=""

  echo "--- [5.1] RUNNING CONDITION A (smoke, max_new_tokens 16) ---"
  /usr/bin/time -v nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_eval_gate_p6_v36b.py \\
    --cond A --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt \\
    --max_new_tokens 16 --out scratch/p6_final_A_smoke_A.jsonl
  echo "EXIT_CODE_A=$?"

  echo "--- [5.2] RUNNING CONDITION B (shift=1, smoke, max_new_tokens 16) ---"
  /usr/bin/time -v nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_eval_gate_p6_v36b.py \\
    --cond B --shift 1 --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt \\
    --max_new_tokens 16 --out scratch/p6_final_A_smoke_B1.jsonl
  echo "EXIT_CODE_B=$?"

  echo "--- [5.3] RUNNING CONDITION F (smoke, max_new_tokens 16) ---"
  /usr/bin/time -v nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_eval_gate_p6_v36b.py \\
    --cond F --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt \\
    --max_new_tokens 16 --out scratch/p6_final_A_smoke_F.jsonl
  echo "EXIT_CODE_F=$?"

  echo "--- [5.4] RUNNING CONDITION A (nll mode, smoke, max_new_tokens 16) ---"
  /usr/bin/time -v nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_eval_gate_p6_v36b.py \\
    --cond A --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt \\
    --max_new_tokens 16 --nll --out scratch/p6_final_A_smoke_A_nll.jsonl
  echo "EXIT_CODE_A_NLL=$?"

  echo "--- [5.5] RUNNING CONDITION B1 (nll mode, shift=1, smoke, max_new_tokens 16) ---"
  /usr/bin/time -v nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_eval_gate_p6_v36b.py \\
    --cond B --shift 1 --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt \\
    --max_new_tokens 16 --nll --out scratch/p6_final_A_smoke_B1_nll.jsonl
  echo "EXIT_CODE_B1_NLL=$?"

  echo "=== 6. INSPECT GENERATION & NLL OUTPUT FILES ==="
  echo "--- scratch/p6_final_A_smoke_A.jsonl ---"
  cat scratch/p6_final_A_smoke_A.jsonl 2>/dev/null || echo "file not found"
  echo "--- scratch/p6_final_A_smoke_B1.jsonl ---"
  cat scratch/p6_final_A_smoke_B1.jsonl 2>/dev/null || echo "file not found"
  echo "--- scratch/p6_final_A_smoke_F.jsonl ---"
  cat scratch/p6_final_A_smoke_F.jsonl 2>/dev/null || echo "file not found"
  echo "--- scratch/p6_final_A_smoke_A_nll.jsonl ---"
  cat scratch/p6_final_A_smoke_A_nll.jsonl 2>/dev/null || echo "file not found"
  echo "--- scratch/p6_final_A_smoke_B1_nll.jsonl ---"
  cat scratch/p6_final_A_smoke_B1_nll.jsonl 2>/dev/null || echo "file not found"

  echo "--- SHA-256 of Output Files ---"
  sha256sum scratch/p6_final_A_smoke_*.jsonl 2>/dev/null || true

else
  echo "[NOTICE] results_s7/decoder_A_v36.pt NOT FOUND! CPU evaluation skipped per TASK-0032 §2."
fi
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0032_check_and_cpu_smoke"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "task0032_out.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Task 0032 bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

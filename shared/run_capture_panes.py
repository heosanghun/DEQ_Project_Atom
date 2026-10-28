import os
import sys
import subprocess

SCRIPT = """
echo "=== TMUX PANE: s7_train_p4_prime ==="
tmux capture-pane -t s7_train_p4_prime -p | tail -n 30 || echo "failed to capture s7_train_p4_prime"

echo "=== TMUX PANE: p5_gpu1 ==="
tmux capture-pane -t p5_gpu1 -p | tail -n 30 || echo "failed to capture p5_gpu1"

echo "=== TMUX PANE: p5b_gpu2 ==="
tmux capture-pane -t p5b_gpu2 -p | tail -n 30 || echo "failed to capture p5b_gpu2"

echo "=== TMUX PANE: p6_chain ==="
tmux capture-pane -t p6_chain -p | tail -n 30 || echo "failed to capture p6_chain"

echo "=== ACTIVE PROCESSES ==="
ps -ef | grep -E "s7_train_decoder|s7_zstar_lowest_run|p6_chain" | grep -v grep || true

echo "=== GPU UTILIZATION ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0024_capture_tmux_panes"
    cmd = [sys.executable, cts_ssh, "ssh", SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "task0024_panes.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Panes saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

import os
import sys
import subprocess

SCRIPT = """
echo "=== TMUX PANE SCROLLBACK: s7_train_p4_prime ==="
tmux capture-pane -t s7_train_p4_prime -S -100 -p || echo "failed"

echo "=== TMUX PANE SCROLLBACK: p5_gpu1 ==="
tmux capture-pane -t p5_gpu1 -S -100 -p || echo "failed"

echo "=== TMUX PANE SCROLLBACK: p5b_gpu2 ==="
tmux capture-pane -t p5b_gpu2 -S -100 -p || echo "failed"

echo "=== P4 PRIME LATEST LOG ==="
tail -n 30 /home/shoon/cts_e1_s5/raw_s7/train_decoder_A_v36.log 2>/dev/null
tail -n 30 /home/shoon/cts_e1_s5/raw_s7/train_p4_prime_suite_v36.log 2>/dev/null

echo "=== P5 LOG LATEST ==="
tail -n 20 /home/shoon/cts_e1_s5/raw_s7/heldout520_P5.log 2>/dev/null

echo "=== P5B LOG LATEST ==="
tail -n 20 /home/shoon/cts_e1_s5/raw_s7/heldout520_P5b_seed2031.log 2>/dev/null
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0024_capture_scrollback"
    cmd = [sys.executable, cts_ssh, "ssh", SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "task0024_scrollback.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Scrollback saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

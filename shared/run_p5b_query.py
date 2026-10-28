import os
import sys
import subprocess

QUERY_SCRIPT = """
echo "=== P5B LOG TAIL ==="
tail -n 5 /home/shoon/cts_e1_s5/raw_s7/heldout520_P5b_seed2031.log 2>/dev/null || tail -n 5 /home/shoon/cts_e1_s5/raw_s7/heldout520_P5b_seed2031_gpu2.log 2>/dev/null

echo "=== DECODER A START TIME ==="
head -n 5 /home/shoon/cts_e1_s5/raw_s7/train_decoder_A_v36.log 2>/dev/null
head -n 10 /home/shoon/cts_e1_s5/raw_s7/train_p4_prime_suite_v36.log 2>/dev/null
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "query_p5b_tail_and_start_time"
    cmd = [sys.executable, cts_ssh, "ssh", QUERY_SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "p5b_query.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"P5b query saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

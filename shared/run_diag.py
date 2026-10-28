import os
import sys
import subprocess

DIAG_SCRIPT = """
echo "=== UPTIME & WHO ==="
uptime
who -b

echo "=== TMUX SESSIONS ==="
tmux list-sessions 2>&1 || echo "no tmux sessions"

echo "=== DMESG OOM / REBOOT SEARCH ==="
dmesg -T 2>/dev/null | grep -i -E "oom|killed process|out of memory|segfault" | tail -n 20 || echo "no dmesg access"

echo "=== P4 PRIME SUITE LOG (TAIL 30) ==="
tail -n 30 /home/shoon/cts_e1_s5/raw_s7/train_p4_prime_suite_v36.log 2>/dev/null || echo "no suite log"

echo "=== DECODER A LOG (TAIL 30) ==="
tail -n 30 /home/shoon/cts_e1_s5/raw_s7/train_decoder_A_v36.log 2>/dev/null || echo "no decoder A log"

echo "=== P5 LOG (TAIL 20) ==="
tail -n 20 /home/shoon/cts_e1_s5/raw_s7/heldout520_P5.log 2>/dev/null || echo "no p5 log"

echo "=== P5B LOG (TAIL 20) ==="
tail -n 20 /home/shoon/cts_e1_s5/raw_s7/heldout520_P5B_seed2031.log 2>/dev/null || echo "no p5b log"

echo "=== ALL FLAGS IN raw_s7 ==="
ls -lat /home/shoon/cts_e1_s5/raw_s7/

echo "=== RESULTS_S7 LISTING ==="
ls -lat /home/shoon/cts_e1_s5/results_s7/ | head -n 30

echo "=== GPU STATUS ==="
nvidia-smi
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "diagnose_server_crash_or_reboot"
    cmd = [sys.executable, cts_ssh, "ssh", DIAG_SCRIPT, "--purpose", purpose]
    print(f"Executing diagnostic query via cts_ssh: {purpose}...")
    out_path = os.path.join(root, "scratch", "diag_server.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Diagnostic output saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

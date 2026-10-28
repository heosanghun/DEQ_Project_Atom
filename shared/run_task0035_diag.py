import os
import sys
import subprocess

REMOTE_SCRIPT = """
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0035 DIAGNOSIS: 16:30 SECOND REBOOT & HARDWARE INSPECTION ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. REBOOT HISTORY & UPTIME ==="
echo "--- last -x reboot shutdown | head -5 ---"
last -x reboot shutdown 2>&1 | head -5

echo "--- uptime ---"
uptime

echo "--- journalctl --list-boots | tail -3 ---"
journalctl --list-boots 2>&1 | tail -3 || echo "journalctl --list-boots failed"

echo "=== 2. SYSTEM JOURNAL OF PREVIOUS BOOT (b -1) ==="
echo "--- journalctl -b -1 -p warning --no-pager | tail -80 ---"
journalctl -b -1 -p warning --no-pager 2>&1 | tail -80 || echo "journalctl -b -1 warning failed / no permission"

echo "--- journalctl -b -1 --no-pager | tail -40 ---"
journalctl -b -1 --no-pager 2>&1 | tail -40 || echo "journalctl -b -1 failed / no permission"

echo "=== 3. DMESG KERNEL MESSAGES ==="
echo "--- dmesg | grep -i -E ... | tail -20 ---"
dmesg 2>&1 | grep -i -E "xid|nvrm|thermal|mce|power|oom|watchdog" | tail -20 || echo "dmesg grep returned no matches or no permission"

echo "=== 4. NVIDIA-SMI HARDWARE DIAGNOSTIC ==="
echo "--- nvidia-smi -q query ---"
nvidia-smi -q 2>&1 | grep -i -E "xid|retired|remapped|temperature|power draw|throttle" | head -60

echo "=== 5. PROCESS CLUES & CURRENT USERS ==="
echo "--- /var/log access check ---"
ls -la /var/log/syslog* /var/log/messages* /var/log/kern.log* 2>&1 | head -10 || echo "/var/log not accessible"

echo "--- ps -eo user,pid,lstart,cmd (relevant processes) ---"
ps -eo user,pid,lstart,cmd | grep -E "python|torch|cuda|gpu|s7|shoon" | grep -v grep | head -30

echo "--- nvidia-smi --query-compute-apps=pid,used_memory --format=csv ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1 || echo "query-compute-apps failed"

echo "=== 6. DECODER D STEP 1500 CHECKPOINT DETAILS ==="
if [ -f results_s5/decoder_D_v36_step1500.pt ]; then
  ls -lh results_s5/decoder_D_v36_step1500.pt
  ls -l results_s5/decoder_D_v36_step1500.pt
  sha256sum results_s5/decoder_D_v36_step1500.pt
else
  echo "results_s5/decoder_D_v36_step1500.pt NOT FOUND"
fi
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0035_diag_second_reboot"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "task0035_diag_out.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Task 0035 diag bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

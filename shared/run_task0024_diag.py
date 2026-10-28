import os
import sys
import subprocess

SCRIPT = """
echo "=== 1. REBOOT DIAGNOSIS ==="
echo "--- last -x reboot shutdown ---"
last -x reboot shutdown 2>&1 | head -n 15 || echo "last failed"

echo "--- journalctl --list-boots ---"
journalctl --list-boots 2>&1 | tail -n 5 || echo "journalctl --list-boots failed"

echo "--- journalctl -b -1 -p warning --no-pager ---"
journalctl -b -1 -p warning --no-pager 2>&1 | tail -n 60 || echo "journalctl -b -1 failed or permission denied"

echo "--- nvidia-smi -q (retired / xid / temperature) ---"
nvidia-smi -q 2>&1 | grep -i -A3 -E "retired|xid|temperature" || echo "nvidia-smi query failed"

echo "--- dmesg (xid, nvrm, thermal, mce, oom) ---"
dmesg 2>&1 | grep -i -E "xid|nvrm|thermal|mce|oom" | tail -n 30 || echo "dmesg failed or permission denied"

echo "=== 2. INCREMENTAL FILES STATE ==="
cd /home/shoon/cts_e1_s5

echo "--- decoder_A_v36_step2500.pt ---"
ls -lh results_s5/decoder_A_v36_step2500.pt
sha256sum results_s5/decoder_A_v36_step2500.pt
/home/shoon/system1_5/.venv/bin/python -c '
import torch
p = "results_s5/decoder_A_v36_step2500.pt"
d = torch.load(p, map_location="cpu")
print("keys:", list(d.keys()))
print("step:", d.get("step"))
print("has_optimizer:", "optimizer_state_dict" in d)
print("has_torch_rng:", "torch_rng_state" in d)
print("has_cuda_rng:", "cuda_rng_state" in d)
print("has_numpy_rng:", "numpy_rng_state" in d)
'

echo "--- zstar_s7_a0.25_heldout520_lowest.pt ---"
ls -lh results_s7/zstar_s7_a0.25_heldout520_lowest.pt
sha256sum results_s7/zstar_s7_a0.25_heldout520_lowest.pt
/home/shoon/system1_5/.venv/bin/python -c '
import torch
p = "results_s7/zstar_s7_a0.25_heldout520_lowest.pt"
d = torch.load(p, map_location="cpu")
print("loadable: True")
print("key_count:", len(d))
'

echo "--- zstar_s7_a0.25_heldout520_seed2031.pt ---"
ls -lh results_s7/zstar_s7_a0.25_heldout520_seed2031.pt
sha256sum results_s7/zstar_s7_a0.25_heldout520_seed2031.pt
/home/shoon/system1_5/.venv/bin/python -c '
import torch
p = "results_s7/zstar_s7_a0.25_heldout520_seed2031.pt"
d = torch.load(p, map_location="cpu")
print("loadable: True")
print("key_count:", len(d))
'

echo "--- Flags in raw_s7 ---"
ls -la raw_s7/*.flag 2>/dev/null || echo "no flags"
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0024_reboot_diag_and_files_check"
    cmd = [sys.executable, cts_ssh, "ssh", SCRIPT, "--purpose", purpose]
    print(f"Launching diagnosis query via cts_ssh: {purpose}...")
    out_path = os.path.join(root, "scratch", "task0024_diag.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Output saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

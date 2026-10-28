import os
import sys
import subprocess

REMOTE_SCRIPT = """
cd /home/shoon/cts_e1_s5

echo "=== 1. CURRENT TMUX SESSIONS ==="
tmux list-sessions || echo "no tmux sessions"

echo "=== 2. CURRENT ACTIVE PROCESSES ==="
ps -ef | grep -E "s7_train_decoder|s7_zstar_lowest_run|p6_chain|p5_heldout520|p5b_seed2031|sha256sum" | grep -v grep || true

echo "=== 3. GPU UTILIZATION ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo "=== 4. P5 / P5B SCROLLBACK ==="
echo "--- p5_gpu1 scrollback ---"
tmux capture-pane -t p5_gpu1 -S -100 -p 2>&1 | tail -n 25 || echo "p5_gpu1 pane not found"
echo "--- p5b_gpu2 scrollback ---"
tmux capture-pane -t p5b_gpu2 -S -100 -p 2>&1 | tail -n 25 || echo "p5b_gpu2 pane not found"
echo "--- p6_chain scrollback ---"
tmux capture-pane -t p6_chain -S -100 -p 2>&1 | tail -n 15 || echo "p6_chain pane not found"

echo "=== 5. CHECK PREREQUISITES FILES ==="
python3 -c "
import os
files = [
    'scripts/s7_zstar_lowest_run.py',
    'cts/deq/zstar_canonical.py',
    'cts/deq/transition_inject.py',
    'exe_scratch/s7/ref_solvers/anderson.py',
    '/home/shoon/checkpoints/m1_final_frozen.pt',
    'configs/s7_new_heldout_520.json',
    'PREREG_S7_D_v3_7.md',
    'configs/zstar_canonical.yaml',
    '/home/shoon/cts_e1/configs/zstar_canonical.yaml',
    'scratch/harness_A_v1.py',
    '/home/shoon/cts_e1/scratch/harness_A_v1.py'
]
for f in files:
    if os.path.exists(f):
        print(f'[EXISTS] {f} ({os.path.getsize(f)} bytes)')
    else:
        print(f'[MISSING] {f}')
"

echo "=== 6. PATCH s7_train_decoder_v36.py FOR P4 RESUME ==="
python3 -c "
path = 'scripts/s7_train_decoder_v36.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

target = '''        if \"torch_rng_state\" in res_ckpt:
            torch.set_rng_state(res_ckpt[\"torch_rng_state\"])
        if \"cuda_rng_state\" in res_ckpt and res_ckpt[\"cuda_rng_state\"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(res_ckpt[\"cuda_rng_state\"])'''

replacement = '''        if \"torch_rng_state\" in res_ckpt and res_ckpt[\"torch_rng_state\"] is not None:
            t_rng = res_ckpt[\"torch_rng_state\"]
            if isinstance(t_rng, torch.Tensor):
                t_rng = t_rng.to(dtype=torch.uint8, device=\"cpu\")
            elif isinstance(t_rng, (bytes, bytearray)):
                t_rng = torch.ByteTensor(list(t_rng))
            try:
                torch.set_rng_state(t_rng)
            except Exception as e:
                print(f\"[RESUME] Warning setting torch_rng_state: {e}\", flush=True)
        if \"cuda_rng_state\" in res_ckpt and res_ckpt[\"cuda_rng_state\"] is not None and torch.cuda.is_available():
            c_rng = res_ckpt[\"cuda_rng_state\"]
            if isinstance(c_rng, torch.Tensor):
                c_rng = c_rng.to(dtype=torch.uint8, device=\"cpu\")
            elif isinstance(c_rng, (bytes, bytearray)):
                c_rng = torch.ByteTensor(list(c_rng))
            try:
                torch.cuda.set_rng_state(c_rng)
            except Exception as e:
                print(f\"[RESUME] Warning setting cuda_rng_state: {e}\", flush=True)'''

if target in c:
    c = c.replace(target, replacement, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(c)
    print('PATCHED s7_train_decoder_v36.py successfully')
elif 't_rng.to(dtype=torch.uint8, device=\"cpu\")' in c:
    print('ALREADY PATCHED s7_train_decoder_v36.py')
else:
    print('TARGET NOT FOUND')
"
python3 -m py_compile scripts/s7_train_decoder_v36.py && echo "Decoder v36 compilation: OK"

echo "=== 7. RELAUNCH P4 IN TMUX ==="
tmux kill-session -t s7_train_p4_prime 2>/dev/null || true
tmux new-session -d -s s7_train_p4_prime "bash -c 'bash /home/shoon/cts_e1_s5/scripts/s7_train_p4_prime_resume_v36.sh >> raw_s7/train_p4_prime_suite_v36.log 2>&1'"
echo "Launched s7_train_p4_prime"

echo "=== 8. P5 AND P5B TEST / LAUNCH CHECK ==="
if ! pgrep -f "s7_zstar_lowest_run.py.*zstar_s7_a0.25_heldout520_lowest.pt" >/dev/null 2>&1; then
    echo "P5 lowest is NOT running in python. Running launcher diagnostic:"
    bash -x scripts/p5_heldout520_launch_v5.sh --device cuda:1 2>&1 | head -n 45 || true
else
    echo "P5 lowest IS running in python!"
fi

if ! pgrep -f "s7_zstar_lowest_run.py.*zstar_s7_a0.25_heldout520_seed2031.pt" >/dev/null 2>&1; then
    echo "P5b seed2031 is NOT running in python. Running launcher diagnostic:"
    bash -x scripts/p5b_seed2031_launch_v2.sh --device cuda:2 2>&1 | head -n 45 || true
else
    echo "P5b seed2031 IS running in python!"
fi
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0024_patch_and_status"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "task0024_patch_out.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Output saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

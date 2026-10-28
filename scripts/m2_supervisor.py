import os
import sys
import time
import subprocess
import hashlib

def get_file_hash(path: str) -> str:
    if not os.path.exists(path):
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line)
    with open("/home/shoon/cts_e1/m2_supervisor.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def check_m2_finished() -> bool:
    log_path = "/home/shoon/cts_e1/retrain_stage1_5000.log"
    ckpt_path = "/home/shoon/retrain/ckpt/stage1_last.pt"
    ref_ckpt = "/home/shoon/retrain/ckpt/stage1_step3500.pt"
    
    if not os.path.exists(log_path):
        log("Check failed: log file not found.")
        return False
    if not os.path.exists(ckpt_path):
        log("Check failed: stage1_last.pt not found.")
        return False
    if not os.path.exists(ref_ckpt):
        log("Check failed: stage1_step3500.pt reference not found.")
        return False
        
    # Check 1: step=5000/5000 printed in log
    has_finished_log = False
    finished_line = ""
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "stage1 step=5000/5000" in line:
                has_finished_log = True
                finished_line = line.strip()
                break
    if not has_finished_log:
        log("Check failed: 'stage1 step=5000/5000' not found in log.")
        return False
        
    # Check 2: Size is at least 95% of step3500
    ref_size = os.path.getsize(ref_ckpt)
    ckpt_size = os.path.getsize(ckpt_path)
    if ckpt_size < ref_size * 0.95:
        log(f"Check failed: stage1_last.pt size {ckpt_size} is less than 95% of reference {ref_size}.")
        return False
        
    # Check 3: Checkpoint modified time is after the log's modified time
    log_mtime = os.path.getmtime(log_path)
    ckpt_mtime = os.path.getmtime(ckpt_path)
    if ckpt_mtime < log_mtime - 120:  # Allow 2 minutes leeway
        log(f"Check failed: stage1_last.pt mtime {ckpt_mtime} is older than log mtime {log_mtime}.")
        return False
        
    log(f"[WATCH-08 EVIDENCE] Log line: '{finished_line}'")
    log(f"[WATCH-08 EVIDENCE] stage1_last.pt mtime: {time.ctime(ckpt_mtime)} | size: {ckpt_size} bytes")
    return True

def main():
    target_pid = 1017004
    log(f"Starting supervisor for M2 training process PID {target_pid}...")
    
    # Loop until process ends
    while True:
        if not os.path.exists(f"/proc/{target_pid}"):
            log(f"M2 process {target_pid} has ended.")
            break
        time.sleep(30)
        
    # Now check if M2 finished successfully
    if not check_m2_finished():
        log("ERROR: M2 training process ended, but M2 did not finish successfully (missing step 5000 log, or last checkpoint corrupted/missing). locking M3.")
        sys.exit(1)
        
    ckpt_path = "/home/shoon/retrain/ckpt/stage1_last.pt"
    # Get hash of last checkpoint
    last_hash = get_file_hash(ckpt_path)
    log(f"Final checkpoint stage1_last.pt hash: {last_hash}")
    
    # 1. Run M1 Matrix experiments
    log("Starting M1 Matrix experiments...")
    matrix_cmd = [
        "/home/shoon/system1_5/.venv/bin/python",
        "/home/shoon/cts_e1/scripts/run_m1_matrix.py"
    ]
    with open("/home/shoon/cts_e1/m1_matrix_execution.log", "w", encoding="utf-8") as out:
        res = subprocess.run(matrix_cmd, stdout=out, stderr=subprocess.STDOUT)
        
    if res.returncode != 0:
        log("ERROR: run_m1_matrix.py failed. Check /home/shoon/cts_e1/m1_matrix_execution.log")
        sys.exit(1)
    log("M1 Matrix experiments completed successfully.")
    
    # 2. Run selfcheck_v2.py
    log("Running selfcheck_v2.py --task M1_MATRIX...")
    check_cmd = [
        "/home/shoon/system1_5/.venv/bin/python",
        "/home/shoon/cts_e1/selfcheck_v2.py",
        "--task", "M1_MATRIX",
        "--log_path", "/home/shoon/cts_e1/m1_matrix_results.json",
        "--file_path", "/home/shoon/cts_e1/m1_matrix_logs/"
    ]
    with open("/home/shoon/cts_e1/m1_matrix_selfcheck.log", "w", encoding="utf-8") as out:
        res_check = subprocess.run(check_cmd, stdout=out, stderr=subprocess.STDOUT)
        
    if res_check.returncode == 0:
        log("M1_MATRIX gate PASSED! Launching M3 (Stage-2 PPO 10,000 steps)...")
        
        # 3. Launch M3
        m3_env = os.environ.copy()
        m3_env["CUDA_VISIBLE_DEVICES"] = "2"
        
        m3_cmd = [
            "/home/shoon/system1_5/.venv/bin/python",
            "/home/shoon/cts_e1/scripts/run_stage2_math_ppo.py",
            "--steps", "10000",
            "--device", "cuda:0",
            "--stage1-ckpt", ckpt_path
        ]
        
        m3_log = open("/home/shoon/cts_e1/retrain_stage2.log", "w", encoding="utf-8")
        p = subprocess.Popen(m3_cmd, env=m3_env, stdout=m3_log, stderr=subprocess.STDOUT)
        log(f"M3 launched in background with PID: {p.pid}")
    else:
        log("M1_MATRIX gate FAILED or MISMATCH. M3 locked.")
        
    log("Supervisor finished task.")

if __name__ == "__main__":
    main()

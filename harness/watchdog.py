import os
import sys
import time
import json
import subprocess
import hashlib

STATE_FILE = "/home/shoon/cts_e1/harness/state.json"
WATCHDOG_LOG = "/home/shoon/cts_e1/harness/watchdog.log"
PYTHON_ENV = "/home/shoon/system1_5/.venv/bin/python"
DAG_RUNNER = "/home/shoon/cts_e1/harness/dag_runner.py"

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line)
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def get_gpu_map() -> dict:
    gpu_map = {}
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if line.strip():
                parts = line.split(",")
                gpu_map[parts[1].strip()] = int(parts[0].strip())
    except Exception as e:
        log(f"Failed to query GPU map: {e}")
    return gpu_map

def check_gpu_safety(gpu_map: dict) -> bool:
    our_pids = []
    try:
        for script in ["run_stage1_openmath.py", "run_stage2_math_ppo.py", "run_m1_matrix.py", "dag_runner.py", "watchdog.py"]:
            proc_res = subprocess.run(["pgrep", "-f", script], capture_output=True, text=True)
            for line in proc_res.stdout.splitlines():
                if line.strip():
                    our_pids.append(int(line.strip()))
    except Exception as e:
        log(f"Failed to pgrep our processes for safety check: {e}")
        return False

    try:
        res = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if line.strip():
                parts = line.split(",")
                gpu_uuid = parts[0].strip()
                pid = int(parts[1].strip())
                gpu_idx = gpu_map.get(gpu_uuid, -1)
                # Verify only our processes
                if pid in our_pids and gpu_idx >= 4:
                    log(f"ALERT: Our process PID {pid} running on forbidden GPU {gpu_idx}!")
                    return False
        return True
    except Exception as e:
        log(f"GPU safety check failed: {e}")
        return False

def check_process_gpu_occupancy(target_pids: list, gpu_map: dict) -> list:
    occupied_gpus = []
    try:
        res = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if line.strip():
                parts = line.split(",")
                gpu_uuid = parts[0].strip()
                pid = int(parts[1].strip())
                gpu_idx = gpu_map.get(gpu_uuid, -1)
                if pid in target_pids:
                    occupied_gpus.append((pid, gpu_idx))
    except Exception as e:
        log(f"Process GPU occupancy check failed: {e}")
    return occupied_gpus

def get_diagnostic_dump(log_path: str) -> str:
    dump = "\n=== DIAGNOSTIC DUMP ===\n"
    # nvidia-smi
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    dump += f"--- nvidia-smi ---\n{smi.stdout}\n"
    # dmesg
    dmesg = subprocess.run(["dmesg"], capture_output=True, text=True)
    dmesg_tail = "\n".join(dmesg.stdout.splitlines()[-20:])
    dump += f"--- dmesg tail ---\n{dmesg_tail}\n"
    # log tail
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as lf:
            lines = lf.readlines()
            log_tail = "".join(lines[-20:])
        dump += f"--- Log tail ({log_path}) ---\n{log_tail}\n"
    dump += "=======================\n"
    return dump

def main():
    log("Watchdog wake up.")
    gpu_map = get_gpu_map()
    
    # 1. GPU 4-7 Isolation check (safety constraint)
    if not check_gpu_safety(gpu_map):
        log("GPU check failed safety constraint. Watchdog halt.")
        sys.exit(1)
        
    # 2. Check dag_runner.py state
    if not os.path.exists(STATE_FILE):
        log("Initial state.json not found. Triggering dag_runner...")
        subprocess.run([PYTHON_ENV, DAG_RUNNER])
        return
        
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
        
    running_nodes = []
    for node, info in state.get("nodes", {}).items():
        if info.get("status") == "RUNNING":
            running_nodes.append(node)
            
    if not running_nodes:
        log("No nodes are running. Triggering dag_runner if pending nodes exist...")
        subprocess.run([PYTHON_ENV, DAG_RUNNER])
        return
        
    log(f"Currently running nodes: {running_nodes}")
    
    # Determine the log file of the running node
    node_log = ""
    target_script = ""
    for node in running_nodes:
        if node == "N1":
            node_log = "/home/shoon/cts_e1/retrain_stage1_5000.log"
            target_script = "run_stage1_openmath.py"
        elif node == "N2":
            node_log = "/home/shoon/cts_e1/m1_matrix_execution_dag.log"
            target_script = "run_m1_matrix.py"
        elif node == "N4":
            node_log = "/home/shoon/cts_e1/m3_preflight_100.log"
            target_script = "run_stage2_math_ppo.py"
        elif node == "N5":
            node_log = "/home/shoon/cts_e1/retrain_stage2.log"
            target_script = "run_stage2_math_ppo.py"
            
    if not node_log or not target_script:
        log("No log mapping for running nodes. Handled by dag_runner.")
        return
        
    # Find active target process PID
    target_pids = []
    try:
        proc_res = subprocess.run(["pgrep", "-f", target_script], capture_output=True, text=True)
        for line in proc_res.stdout.splitlines():
            if line.strip():
                target_pids.append(int(line.strip()))
    except Exception as e:
        log(f"Failed to pgrep target processes: {e}")
        
    if not target_pids:
        log(f"WARNING: Node {running_nodes} marked RUNNING but no active process found for {target_script}!")
        # Trigger dag_runner to recover
        subprocess.run([PYTHON_ENV, DAG_RUNNER])
        return
        
    # 3. GPU 0/1/2 Occupancy verify
    occupancies = check_process_gpu_occupancy(target_pids, gpu_map)
    log(f"Target process PIDs: {target_pids} | GPU Occupancy: {occupancies}")
    for pid, gpu in occupancies:
        if gpu not in [0, 1, 2]:
            log(f"ALERT: Target process PID {pid} is running on GPU {gpu} instead of 0, 1, or 2!")
            
    # 4. Log progress / mtime verification
    check_files = [node_log]
    if "N2" in running_nodes:
        logs_dir = "/home/shoon/cts_e1/m1_matrix_logs"
        if os.path.exists(logs_dir):
            for f in os.listdir(logs_dir):
                if f.endswith(".log"):
                    check_files.append(os.path.join(logs_dir, f))

    latest_mtime = 0
    for f in check_files:
        if os.path.exists(f):
            fmtime = os.path.getmtime(f)
            if fmtime > latest_mtime:
                latest_mtime = fmtime

    if latest_mtime > 0:
        now = time.time()
        time_since_update = now - latest_mtime
        log(f"Latest log among {len(check_files)} monitored files modified {time_since_update:.1f} seconds ago.")
        
        # If no update in 35 minutes (2100 seconds)
        if time_since_update > 2100:
            log(f"ALERT: Stagnation detected! Log {node_log} has not updated for {time_since_update:.1f}s")
            # Diagnostic context dump
            dump = get_diagnostic_dump(node_log)
            log(dump)
            
            # Reset and restart logic:
            # check retry count stored in state
            node_info = state["nodes"][running_nodes[0]]
            retries = node_info.get("watchdog_retries", 0)
            if retries < 1:
                log("Attempting recovery: Killing stale target processes and restarting runner...")
                for pid in target_pids:
                    subprocess.run(["kill", "-9", str(pid)])
                node_info["watchdog_retries"] = retries + 1
                node_info["status"] = "PENDING"
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, sort_keys=True)
                subprocess.run([PYTHON_ENV, DAG_RUNNER])
            else:
                log("Stagnation occurred again after recovery retry. Setting node status to UNRESOLVED.")
                node_info["status"] = "UNRESOLVED"
                node_info["detail"] = "Watchdog detected repeated stagnation."
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, sort_keys=True)
                subprocess.run([PYTHON_ENV, DAG_RUNNER])
    else:
        log(f"Log file {node_log} not created yet.")

if __name__ == "__main__":
    main()

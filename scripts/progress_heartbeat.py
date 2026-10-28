import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def run_ssh_cmd(cmd: str) -> str:
    ssh_cmd = f"ssh cts-server \"{cmd}\""
    try:
        proc = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=30)
        return proc.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def heartbeat_once():
    utc_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    
    # 1. Track A progress
    track_a_log = run_ssh_cmd("tail -n 5 ~/cts_e1/retrain_stage2.log")
    track_a_step = "initializing"
    if "step=" in track_a_log:
        # extract stage2 optimizer_step=X/10000
        for line in track_a_log.splitlines():
            if "optimizer_step=" in line:
                track_a_step = line.split("optimizer_step=")[-1].split(" ")[0]
                break
            elif "[wiring-check]" in line:
                track_a_step = "wiring-check: " + line.split("step=")[-1].split(" ")[0]
                
    # 2. Track B progress
    track_b_log = run_ssh_cmd("tail -n 5 ~/system1_5/logs/stage2_fwp_z300_retrain.log")
    track_b_step = "initializing"
    if "epoch=" in track_b_log:
        for line in track_b_log.splitlines():
            if "epoch=" in line:
                track_b_step = "epoch=" + line.split("epoch=")[-1].strip()
                break
    elif "[T3-RESULT]" in track_b_log:
        track_b_step = "T3 completed: " + track_b_log.splitlines()[-1]
        
    # 3. GPU Status
    gpu_out = run_ssh_cmd("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits")
    gpu_status_list = []
    for line in gpu_out.splitlines():
        if line.strip():
            parts = line.split(",")
            idx = parts[0].strip()
            mem = parts[1].strip()
            util = parts[2].strip()
            gpu_status_list.append(f"GPU {idx}: {mem}MB/{util}%")
    gpu_status = " | ".join(gpu_status_list)
    
    # 4. New output hashes
    track_a_hashes = run_ssh_cmd("sha256sum ~/cts_e1/artifacts/stage2_meta_value.pt ~/cts_e1/PROGRESS 2>/dev/null")
    track_b_hashes = run_ssh_cmd("sha256sum ~/system1_5/evidence/rm02_stage2/* ~/system1_5/checkpoints/fwp/fwp_stage2_z300_retrained_seed*.pt 2>/dev/null")
    
    new_hashes = []
    for h_line in (track_a_hashes + "\n" + track_b_hashes).splitlines():
        if h_line.strip() and "Error" not in h_line and "No such file" not in h_line:
            parts = h_line.split()
            if len(parts) >= 2:
                file_name = Path(parts[1]).name
                h_val = parts[0][:8] # use short hash for readability
                new_hashes.append(f"{file_name}({h_val})")
                
    hashes_str = ", ".join(new_hashes) if new_hashes else "none"
    
    # Assemble heartbeat line
    heartbeat_line = f"| {utc_now} | Track A: {track_a_step} | Track B: {track_b_step} | {gpu_status} | {hashes_str} |\n"
    
    # Write to PROGRESS_AUTO.md
    p_path = Path("PROGRESS_AUTO.md")
    if not p_path.is_file():
        p_path.write_text("# PROGRESS_AUTO.md - 24H Autopilot Heartbeat Log\n\n| UTC Time | Track A Status | Track B Status | GPU Status | New Outputs (Short SHA) |\n|---|---|---|---|---|\n", encoding="utf-8")
        
    with p_path.open("a", encoding="utf-8") as f:
        f.write(heartbeat_line)
        
    print(f"Heartbeat logged: {heartbeat_line.strip()}", flush=True)

def main():
    print("Heartbeat daemon started. Writing to PROGRESS_AUTO.md every 2 hours.", flush=True)
    # Perform immediate log on start
    try:
        heartbeat_once()
    except Exception as e:
        print(f"Error during first heartbeat: {e}", flush=True)
        
    # Main loop
    while True:
        try:
            time.sleep(7200) # wait 2 hours
            heartbeat_once()
        except KeyboardInterrupt:
            print("Heartbeat daemon stopped.")
            break
        except Exception as e:
            print(f"Error in heartbeat loop: {e}", flush=True)

if __name__ == "__main__":
    main()

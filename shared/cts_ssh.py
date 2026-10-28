#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 변경 사유: ESC-0003, AUDIT-0009 ⑤, AUDIT-0013 F1 — 연속 접속 실패 판정을 ssh(255·124) 및 scp/sftp(0이 아닌 모든 exit)로 정밀화하고 하위 명령 접두어(ssh:, scp:, sftp:) 자동 부여
"""
shared/cts_ssh.py — CTS 서버 접속 기계적 제어 래퍼 v2b (원장 #416, 개정안 ESC-0003/AUDIT-0013 F1) (원장 #416, AUDIT-0013 F1 보완)

기능 및 기계적 차단 요건:
(1) 60초 간격 강제: 접속 직전 ssh_log.csv 마지막 행 시각 확인 → 60초 미만이면 실행 0,
    "R7-1 대기 필요 N초" 출력 후 즉시 종료 (자동 대기·재시도 0).
(2) 1시간 20회 상한: 최근 1시간(3600초) 이내 행 수 >= 20이면 실행 0 후 즉시 종료.
(3) SSH_LOCK 확인: shared/SSH_LOCK 파일이 존재하면 실행 0 (Principal 수동 해제 전까지 차단).
(4) 단일 행 추가(append) 의무: 성공·실패·타임아웃 무관하게 ssh_log.csv에 반드시 append로 1행 기록.
    기록 시 purpose 앞에 하위 명령 접두어(ssh:, scp:, sftp:) 자동 부여.
(5) 2회 연속 접속 실패 차단 (AUDIT-0013 F1 반영):
    - scp:, sftp: 접두어 행: OpenSSH scp/sftp는 접속 실패 시에도 exit code 1을 반환하므로 0이 아닌 값은 모두 접속 실패.
    - 그 외 (ssh: 행 및 접두어 없는 과거 행): 접속 오류 255 및 타임아웃 124만 접속 실패로 판정 (원격 명령 exit 1 등은 접속 성공).

사용법:
  python shared/cts_ssh.py ssh "<bundled cmd>" [--purpose <desc>]
  python shared/cts_ssh.py sftp <batchfile> [--purpose <desc>]
  python shared/cts_ssh.py scp <args...> [--purpose <desc>]
"""

import os
import sys
import math
import subprocess
from datetime import datetime, timedelta

def get_workspace_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    log_path = os.environ.get("CTS_SSH_LOG_PATH", os.path.join(base_dir, "ssh_log.csv"))
    lock_path = os.environ.get("CTS_SSH_LOCK_PATH", os.path.join(base_dir, "SSH_LOCK"))
    config_path = os.environ.get("CTS_SSH_CONFIG_PATH", os.path.join(base_dir, "ssh_config_cts"))
    return base_dir, project_root, log_path, lock_path, config_path

def read_ssh_log(log_path):
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("timestamp"):
                continue
            parts = line.split(",", 2)
            if len(parts) == 3:
                ts_str, purpose, exit_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
                try:
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    records.append({
                        "datetime": dt,
                        "raw_timestamp": ts_str,
                        "purpose": purpose,
                        "exit_code": exit_str
                    })
                except ValueError:
                    continue
    return records

def check_ssh_lock(lock_path):
    if os.path.exists(lock_path):
        print(f"[BLOCKED] shared/SSH_LOCK 존재: {lock_path}")
        print("연속 실패 등으로 잠금 상태입니다. Principal 수동 해제 전까지 서버 접속 실행 0.")
        sys.exit(2)

def is_connection_failure(purpose, exit_code):
    """
    접속 실패 판정 (AUDIT-0013 F1 반영):
    - scp:, sftp: 접두어 행: OpenSSH scp/sftp는 접속 실패 시에도 exit code 1을 반환하므로, 0이 아닌 값은 모두 접속 실패.
    - 그 외 (ssh: 행 및 접두어가 없는 과거 행): ssh는 접속 오류 시 255, 타임아웃 124를 반환하므로 255와 124만 접속 실패로 판정 (원격 명령 exit!=0은 접속 성공).
    """
    s_code = str(exit_code).strip()
    if s_code == "0":
        return False
    p = str(purpose).strip()
    if p.startswith("scp:") or p.startswith("sftp:"):
        return True
    return s_code in ("255", "124")

def check_consecutive_failures_pre(records, lock_path):
    if len(records) >= 2:
        r1 = records[-2]
        r2 = records[-1]
        if is_connection_failure(r1["purpose"], r1["exit_code"]) and is_connection_failure(r2["purpose"], r2["exit_code"]):
            if not os.path.exists(lock_path):
                with open(lock_path, "w", encoding="utf-8") as f:
                    f.write(f"LOCK created at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Reason: Pre-check detected 2 consecutive connection failures: [{r1['purpose']}:{r1['exit_code']}], [{r2['purpose']}:{r2['exit_code']}]\n")
            print(f"[BLOCKED] 직전 2행 모두 접속 실패 감지 ([{r1['purpose']}:{r1['exit_code']}], [{r2['purpose']}:{r2['exit_code']}]).")
            print(f"shared/SSH_LOCK 생성 및 서버 접속 차단. Principal 직접 조치 필요.")
            sys.exit(2)

def check_cooldown(records, now):
    if not records:
        return
    last_dt = records[-1]["datetime"]
    elapsed = (now - last_dt).total_seconds()
    if elapsed < 60.0:
        remain = math.ceil(60.0 - elapsed)
        if remain < 1:
            remain = 1
        print(f"R7-1 대기 필요 {int(remain)}초")
        sys.exit(1)

def check_hourly_limit(records, now):
    cutoff = now - timedelta(seconds=3600)
    recent_count = sum(1 for r in records if r["datetime"] >= cutoff)
    if recent_count >= 20:
        print(f"R7-1 최근 1시간 연결 상한(20회) 도달: 현재 {recent_count}회")
        print("서버 접속 제한 초과로 실행 0 종료.")
        sys.exit(1)

def append_log(log_path, dt_str, purpose, exit_code):
    clean_purpose = purpose.replace(",", ";").replace("\n", " ").strip()
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,purpose,exit_code\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{dt_str},{clean_purpose},{exit_code}\n")

def check_consecutive_failures_post(records_before, current_purpose, current_exit, lock_path, now_str):
    if is_connection_failure(current_purpose, current_exit) and records_before:
        last_rec = records_before[-1]
        if is_connection_failure(last_rec["purpose"], last_rec["exit_code"]):
            if not os.path.exists(lock_path):
                with open(lock_path, "w", encoding="utf-8") as f:
                    f.write(f"LOCK created at {now_str}\n")
                    f.write(f"Reason: 2 consecutive connection failures: last=[{last_rec['purpose']}:{last_rec['exit_code']}], current=[{current_purpose}:{current_exit}]\n")
            print(f"[LOCK] 2회 연속 접속 실패 감지 (직전 [{last_rec['purpose']}:{last_rec['exit_code']}], 현재 [{current_purpose}:{current_exit}]).")
            print(f"shared/SSH_LOCK 파일을 생성했습니다. Principal 직접 조치 및 해제 전까지 차단됩니다.")

def parse_cli_args(args):
    purpose = None
    filtered_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--purpose" and i + 1 < len(args):
            purpose = args[i+1]
            i += 2
        elif arg.startswith("--purpose="):
            purpose = arg.split("=", 1)[1]
            i += 1
        else:
            filtered_args.append(arg)
            i += 1
    return filtered_args, purpose

def build_command(subcmd, remaining_args, config_path):
    if subcmd == "ssh":
        if not remaining_args:
            print("Error: ssh requires command argument")
            sys.exit(1)
        has_host = any(h in remaining_args[0] for h in ["cts-server", "<GPU_SERVER_IP>", "shoon@"])
        if has_host or len(remaining_args) > 1:
            cmd = ["ssh", "-F", config_path, "-o", "BatchMode=yes"] + remaining_args
        else:
            cmd = ["ssh", "-F", config_path, "-o", "BatchMode=yes", "cts-server", remaining_args[0]]
        return cmd
    elif subcmd == "sftp":
        if not remaining_args:
            print("Error: sftp requires batchfile argument")
            sys.exit(1)
        if len(remaining_args) == 1 and not remaining_args[0].startswith("-"):
            cmd = ["sftp", "-F", config_path, "-b", remaining_args[0], "cts-server"]
        else:
            cmd = ["sftp", "-F", config_path] + remaining_args
        return cmd
    elif subcmd == "scp":
        if not remaining_args:
            print("Error: scp requires arguments")
            sys.exit(1)
        cmd = ["scp", "-F", config_path] + remaining_args
        return cmd
    else:
        print(f"Error: Unsupported subcommand '{subcmd}'. Supported: ssh, sftp, scp")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python shared/cts_ssh.py <ssh|sftp|scp> [args...] [--purpose <desc>]")
        sys.exit(1)

    subcmd = sys.argv[1]
    raw_args = sys.argv[2:]
    remaining_args, user_purpose = parse_cli_args(raw_args)

    # Automatically prefix purpose with subcommand (ssh:, scp:, sftp:)
    if user_purpose:
        if user_purpose.startswith(f"{subcmd}:"):
            full_purpose = user_purpose
        else:
            full_purpose = f"{subcmd}:{user_purpose}"
    else:
        full_purpose = f"{subcmd}:{' '.join(remaining_args)[:60]}"

    base_dir, project_root, log_path, lock_path, config_path = get_workspace_paths()

    # (3) shared\SSH_LOCK 파일이 있으면 실행 0
    check_ssh_lock(lock_path)

    # Read current log records
    records = read_ssh_log(log_path)
    now = datetime.now()

    # (5-pre) 직전 2행이 모두 접속 실패이면 SSH_LOCK 생성 후 종료
    check_consecutive_failures_pre(records, lock_path)

    # (1) 60초 미만이면 실행 0, "R7-1 대기 필요 N초" 출력 후 종료
    check_cooldown(records, now)

    # (2) 최근 1시간 행 수 >= 20이면 실행 0
    check_hourly_limit(records, now)

    # Execution phase
    mock_exit_env = os.environ.get("CTS_SSH_MOCK_EXIT")
    run_timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if mock_exit_env is not None:
        exit_code = int(mock_exit_env)
        print(f"[MOCK_EXEC] CTS_SSH_MOCK_EXIT={exit_code} (서버 연결 0)")
    else:
        cmd = build_command(subcmd, remaining_args, config_path)
        try:
            res = subprocess.run(cmd)
            exit_code = res.returncode
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Command timed out: {cmd}")
            exit_code = 124
        except Exception as e:
            print(f"[EXCEPTION] Subprocess failed: {e}")
            exit_code = 255

    # (4) 성공·실패·타임아웃 모두 ssh_log.csv에 추가(append) 1행: 시각, 목적, exit
    append_log(log_path, run_timestamp_str, full_purpose, exit_code)

    # (5-post) 직전 2행이 모두 접속 실패이면 SSH_LOCK 생성 후 종료
    check_consecutive_failures_post(records, full_purpose, exit_code, lock_path, run_timestamp_str)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()

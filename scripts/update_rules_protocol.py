import os

rule_sec = """
## 7. 서버 접속 예절 규약 v1 (원장 #399, 영구)
- **배경**: 09-28 11:46 이후 회사 IP <REDACTED_IP> 가 반복 접속으로 "공격 대상"에 등록돼 차단됨(10-01 담당자 해제). 재발 시 루프 전체가 사흘 멈춤.
- R7-1 **접속 횟수 상한**: 새 SSH 연결은 60초에 1회, 1시간에 20회 이하. 연결마다 `shared\\ssh_log.csv` 에 시각·목적·exit 1행 기록.
- R7-2 **재시도 금지**: 타임아웃·거부가 2회 연속이면 즉시 중단, `STATE.md`에 "접속 불가 [시각]"만 적고 Principal 결정 대기. 자동 재시도 루프·프로브 스크립트·BatchMode 탐색·-v 반복 테스트·Test-NetConnection 반복 전부 금지.
- R7-3 **묶어서 한 번에**: 서버에서 할 명령은 하나의 ssh 호출에 모아 보냄(`"cmd1; cmd2; cmd3"` 또는 스크립트 파일을 scp 후 1회 실행). 상태 확인 하나 보려고 따로 접속하지 않음.
- R7-4 **긴 작업 폴링은 서버 안에서**: 완료 대기는 접속을 반복하지 말고 한 연결 안에서 `while …; do sleep 300; done` 으로 기다림(`ServerAliveInterval=60` 으로 끊김 방지). 모든 긴 작업은 tmux s5_exe40 안.
- R7-5 **인증 실패 0건**: `~/.ssh/config`의 cts-server 항목에 `IdentitiesOnly yes` + 정확한 `IdentityFile` 1개만. 에이전트에 키가 여러 개면 틀린 키를 차례로 내밀어 "실패 로그인"으로 집계되므로 금지. 비밀번호 인증 시도 금지.
- R7-6 **config 고정값**: `ServerAliveInterval 60`, `ServerAliveCountMax 3`, `ConnectTimeout 15`, `StrictHostKeyChecking accept-new`, `IdentitiesOnly yes`.
- R7-7 **실행자는 한 곳만**: EXE-40이 끝날 때까지 서버 실행은 노트북 루프(cts_260930)만 담당. 회사 PC는 코드·git 준비만 하고 서버 접속 0회. 교대는 Principal이 `STATE.md`로 지정할 때만.
- R7-8 **위반 시 거부**: 위 규약 위반 명령은 실행 전에 스스로 걸러 보고서에 "규약 7-2로 미실행"이라고 적음.
"""

# 1. Update shared/RULES.md
rules_path = r"D:\AI\cts\shared\RULES.md"
if os.path.exists(rules_path):
    with open(rules_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "서버 접속 예절 규약 v1" not in content:
        content = content.rstrip() + "\n" + rule_sec
        with open(rules_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Updated D:\\AI\\cts\\shared\\RULES.md")
    else:
        print("shared\\RULES.md already contains rule.")

# 2. Update .agents/AGENTS.md
agents_path = r"D:\AI\cts\.agents\AGENTS.md"
if os.path.exists(agents_path):
    with open(agents_path, "r", encoding="utf-8") as f:
        a_content = f.read()
    if "서버 접속 예절 규약 v1" not in a_content:
        a_content = a_content.rstrip() + "\n" + rule_sec
        with open(agents_path, "w", encoding="utf-8") as f:
            f.write(a_content)
        print("Updated D:\\AI\\cts\\.agents\\AGENTS.md")
    else:
        print(".agents\\AGENTS.md already contains rule.")

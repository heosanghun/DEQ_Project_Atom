# AUDIT-0000 — 킷 v1 검토 (설치자 A, 2026-09-28)

판정: **FIX** — 아래 K1·K2가 닫히기 전에는 `loop_driver.py` 가동 금지. 파일 수정은 하지 않았다(수정안은 diff로만 첨부).

## 사실 확인 (포인터: `shared/logs/server_health_20260928_A0.log`)
| 항목 | 킷 기재 | 실측 | 포인터 |
|---|---|---|---|
| 서버 SSH | 무응답 | 키 인증 접속 정상(`echo ok`), uptime 110일(재부팅 없음) | log:17 |
| 메모리 | 고갈 추정 | available 220 GB / total 251 | log:3 |
| verify_frozen_parity | 잔존 추정 | 실행 중 프로세스 0 (log:16의 2는 pgrep 자기 매칭, log:18 `count=0`이 정확) | log:18 |
| 파이프라인 | 150/400 | Step 2 `[082/680]` 진행 중, 로그 갱신 0분 전, GPU 0 98 % | log:4,14,15 |
| tmux | — | `s7_zstar_gpu0`, `s7_watcher` 생존 | log:5-13 |
| SSH 키 | 미등록 가정 | `~/.ssh/id_ed25519` 로 이미 인증됨 → ESC-0002 불필요 | — |

→ **ESC-0001(서버 물리 조치)은 사실상 불필요**. 답은 Principal이 적는다(권고: C 보류).
→ STATE.md 상단을 실측으로 갱신함.

## K1 (치명) 드라이버가 감사 없이 다음 TASK를 실행한다
`pending()`은 `exe = tasks − reports`이고 main은 `exe`를 `audit`보다 먼저 처리한다. TASK-0001~0003이 미리 들어 있으므로 REPORT-0001이 생기면 다음 사이클에 **AUDIT-0001 없이 TASK-0002, 이어서 TASK-0003(GPU 0 개입 실험)** 이 실행된다. RULES R4-5(FIX 먼저 닫기)와 TASK-0003 선행 조건(TASK-0002 PASS, 1,080 정지 플래그)을 드라이버가 강제하지 않는다.

수정안:
```diff
-            if p["exe"]:   run_exe(p["exe"][0])
-            elif p["audit"]: run_audit(p["audit"][0])
+            if p["audit"]: run_audit(p["audit"][0])
+            elif p["exe"] and exe_allowed(p["exe"][0]): run_exe(p["exe"][0])
             elif p["plan"] is not None: run_plan(p["plan"])
```
```python
def exe_allowed(n):
    """TASK-n은 TASK-(n-1)의 AUDIT가 PASS일 때만 실행 (n==1 제외)."""
    if n == 1: return True
    prev = read(f"audits/AUDIT-{n-1:04d}.md")
    return bool(re.search(r"판정\s*[:：]\s*\**PASS", prev))
```

## K2 (치명) `claude -p` 권한 모드 미정
Claude Code 2.1.220 확인 결과 `-p`, `--output-format`, `--model`, `--allowedTools`, `--permission-mode`, `--dangerously-skip-permissions` 모두 존재. 현재 `CLAUDE_CMD = ["claude","-p","--output-format","text"]`는 권한 설정이 없어 **헤드리스에서 Bash(ssh) 호출이 거부**되므로 EXE가 아무것도 실행하지 못한다. 반대로 `--dangerously-skip-permissions`는 공유 연구 서버에 무제한 명령을 허용한다. Principal 결정 필요. 권고(허용 목록):
```python
CLAUDE_CMD = ["claude", "-p", "--output-format", "text", "--model", "claude-opus-5-5",
              "--allowedTools", "Bash(ssh -o BatchMode=yes*)", "Read", "Write(shared/**)", "Edit(shared/**)"]
```
(A·B 역할은 `Read`만으로 충분 — 역할별 CLAUDE_CMD 분리 권고)

## K3 (중) 현재 노드가 N0에서 영원히 고정
`update_state()`가 STATE의 "현재 노드:"를 읽어 다시 쓰고, A_PROMPT는 TASK 본문만 출력하므로 아무도 노드를 바꾸지 않는다. 수정안: 최신 TASK 파일의 `노드:` 필드에서 읽는다.

## K4 (중) 프롬프트를 명령행 인자로 전달
Windows 명령행 한도 32,767자. 현재 컨텍스트(≈15 KB)는 들어가지만 AUDIT·REPORT가 길어지면 실패한다. 수정안: `subprocess.run(CLAUDE_CMD, input=prompt, ...)` (stdin 전달).

## K5 (경) 기타
- TASK-0002 (b) `results_s5/phase2_outputs.jsonl`: 로컬(`D:\AI\cts\results_s5`)인지 서버인지 불명확 — [대상 환경]이 서버이므로 서버 경로 명시 필요.
- GOAL G1의 r = ‖f(z)−z‖/‖z‖ ↔ 로컬 코드 `_relative_residual`은 분모 max(‖z‖,‖Φ‖,1) (broyden_forward.py:163). 서버 정본의 정의 확인 필요(‖z‖≈5,000이라 값 차이는 작을 것으로 예상되나 미검증).
- 기존 `inbox/EXE/TASK-0001.md`(킷 이전 Antigravity용 판)가 남아 있음. Antigravity가 계속 inbox를 보고 있다면 구판을 실행할 수 있음 — 건드리지 않음.
- 드라이버를 Windows 작업 스케줄러에 "로그인 시" 등록하는 것은 상시 설정이므로 Principal 승인 후. 등록 명령(미실행):
  `schtasks /Create /TN "CTS loop_driver" /SC ONLOGON /TR "python D:\AI\cts\shared\loop_driver.py"`

## 설치 기록
- 기존 GOAL/GRAPH/RULES/LESSONS/STATE/CODEBASE_MAP/tasks/TASK-0001 → `archive/pre_kit_20260928_1041/` (mv, 삭제 0). LEDGER는 병합(기존 #380–#385 보존).
- `CODEBASE_MAP.md` 새로 작성(결함 21개, 파일:행, P0/P1/P2).

LEDGER: 2026-09-28 AUDIT-0000 킷 검토 FIX — K1 감사 없는 연속 실행, K2 헤드리스 권한 미정; 서버 정상(available 220 GB, 파이프라인 82/680) server_health_20260928_A0.log

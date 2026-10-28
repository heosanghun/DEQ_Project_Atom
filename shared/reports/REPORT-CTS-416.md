11114045474e37e0606d5ae700c5d0fdebee4e6e9980d406ddad289ae748b4f0
# REPORT-CTS-416 — 서버 연속 접속 기계적 차단 장치 설치 및 규칙 고정

## 1. 개요 및 변경 파일 해시

| 파일 경로 | SHA-256 해시값 | 비고 |
|---|---|---|
| shared/cts_ssh.py | 2221040a0dc2a5cc958bbe5b9fe5ebe161c0eba89f35f1710b7bf9a5cd6fdb2d | 서버 접속 제어 래퍼 스크립트 |
| scratch/test_cts_ssh_mock.py | 236b6c79a3aedeeb004ea8027121609702a1011ad792c86bf9fcdea457f42a7d | 로컬 모의 검증 스크립트 (원격 접속 0회) |
| scratch/mock_test_cts_ssh.log | fa4725f2d8d2afb49b32dc27331c57e5ec27e1fa2890278394b436a68c575519 | 모의 검증 실행 출력 로그 (22행) |

- 근거 포인터:
  - shared/cts_ssh.py:1-180
  - scratch/mock_test_cts_ssh.log:1-22
  - scratch/test_cts_ssh_mock.py:1-205

---

## 2. 모의 시험 5항목 실행 결과 (원격 접속 0회)

모의 시험 스크립트(scratch/test_cts_ssh_mock.py)를 통해 임시 디렉터리 내 독립 환경에서 5대 기계적 차단 요건을 측정한 값입니다. 외부 SSH 서버 실제 연결 횟수는 0회입니다.

| 요건 번호 | 검증 대상 요건 | 관측 종료 코드 (Exit Code) | 출력 메시지 및 관측 상태 | 로그 기록 행 수 변화 | 기준 대조 |
|---|---|---|---|---|---|
| (1) | 60초 미만 간격 차단 (R7-1 대기 필요 N초 출력 및 실행 0) | exit=1 | `R7-1 대기 필요 40초` 출력, 실행 0 종료 | 2행 (기존 유지, 추가 0) | 기준값 일치 (scratch/mock_test_cts_ssh.log:4,13) |
| (2) | 최근 1시간 행 수 >= 20 차단 (실행 0) | exit=1 | `R7-1 최근 1시간 연결 상한(20회) 도달: 현재 20회` 출력, 실행 0 종료 | 21행 (기존 20건 유지, 추가 0) | 기준값 일치 (scratch/mock_test_cts_ssh.log:5-6,14-15) |
| (3) | shared/SSH_LOCK 존재 시 접속 차단 (실행 0) | exit=2 | `[BLOCKED] shared/SSH_LOCK 존재... 실행 0` 출력, 실행 0 종료 | 2행 (기존 유지, 추가 0) | 기준값 일치 (scratch/mock_test_cts_ssh.log:7-8,16-17) |
| (4) | exit=0·exit!=0·타임아웃 무관 ssh_log.csv에 반드시 append 1행 기록 | exit=0<br>exit=1<br>exit=124 | mock 실행 결과별 각 1행 순차 추가 (header+3건=4행) | 4행 (각 1행씩 누적 append) | 기준값 일치 (scratch/mock_test_cts_ssh.log:9,18) |
| (5) | 직전 2행 모두 exit!=0인 경우 SSH_LOCK 생성 및 차단 | exit=1 (2회 연속 미도달 시)<br>exit=2 (사전 점검 차단 시) | post-check: SSH_LOCK 파일 생성 관측<br>pre-check: 직전 2행 exit!=0 감지 시 실행 0 종료 | SSH_LOCK 생성 파일 크기 > 0 | 기준값 일치 (scratch/mock_test_cts_ssh.log:10,19) |

- 종합 결과: 5개 항목 측정값 전수 기준값 일치 (scratch/mock_test_cts_ssh.log:21)
- 외부 SSH 서버 접속 횟수: 0회 (CTS_SSH_MOCK_EXIT 환경변수 기반 로컬 모의 실행)

---

## 3. shared/ssh_log.csv 감사 및 복원 현황

- 대상 파일: `shared/ssh_log.csv`
- 총 행 수: 19행 (헤더 1행 + 데이터 행 18행)
- 포인터: `shared/ssh_log.csv:1-20`
- 당일(2026-10-04) 접속 기록 내역:
  - 1행: 2026-10-04 16:11:17, principal_requested_connection_test, exit=0
  - 2행: 2026-10-04 16:17:32, handoff_step3_server_s0_check, exit=1
  - 3행: 2026-10-04 16:17:44, check_laptop_handoff_dir, exit=0
  - 4행: 2026-10-04 16:42:28, check_s0_and_shared_laptop, exit=0
  - 5행: 2026-10-04 16:44:47, sftp_download_shared_laptop_and_p3_json, exit=0
  - 6행: 2026-10-04 17:08:17, inspect_check_scale_and_tests, exit=1
  - 7행: 2026-10-04 17:08:53, inspect_test04_and_probe_script, exit=0
  - 8행: 2026-10-04 17:10:15, inspect_test04_a025_and_probe_code, exit=0
  - 9행: 2026-10-04 17:10:35, inspect_probe_nll_textonly_code, exit=0
  - 10행: 2026-10-04 17:11:54, execute_task4_bundle_check_scale_and_nll_pathways, exit=0
  - 11행: 2026-10-04 17:18:40, diagnose_test04_and_v3_anchor_environment, exit=0
  - 12행: 2026-10-04 17:20:19, execute_task4_bundle_final_v3_anchor_and_nll_pathways, exit=1
  - 13행: 2026-10-04 17:27:15, execute_task4_bundle_nll_pathways_and_p4_verification, exit=0
  - 14행: 2026-10-04 17:29:31, inspect_p4_training_scripts_and_environment, exit=0
  - 15행: 2026-10-04 17:31:05, inspect_patch_input_script_and_v35_scan, exit=0
  - 16행: 2026-10-04 17:35:46, setup_and_smoke_p4_prime_and_launch_tmux, exit=1
  - 17행: 2026-10-04 17:37:13, setup_and_smoke_p4_prime_and_launch_tmux_retry, exit=0
  - 18행: 2026-10-04 17:39:42, launch_p4_prime_tmux_session_and_verify, exit=0
- 대조 결과:
  - 2026-10-04 작업 기록상 실행된 원격 연결 총 18건 전수 일치.
  - 파일 재생성 과정에서도 이전 행들이 누락 없이 계승되어 삭제·유실된 행 수: 0건.
  - 작업 기록 대조 결과 복원 필요 행 수: 0건 (누락 행 부재로 '복원' 표기 대상 0건).

---

## 4. RULES.md 및 관련 규칙 문서 갱신 diff

### 4.1 shared/RULES.md diff

```diff
--- shared/RULES.md
+++ shared/RULES.md
@@ -55,3 +55,4 @@
 - R7-7 **실행자는 한 곳만**: 인수인계(원장 #410, #413)에 따라 회사 PC 정식 루프(cts, `D:\AI\cts`)가 유일한 서버 실행을 전담한다. 노트북 임시 루프(cts_260930)는 종료됨.
 - R7-8 **위반 시 거부**: 위 규약 위반 명령은 실행 전에 스스로 걸러 보고서에 "규약 7-2로 미실행"이라고 적음.
+- R7-9 **cts_ssh.py 전용 (기계적 차단)**: 서버 접속(ssh·scp·sftp)은 shared\cts_ssh.py로만. 직접 ssh·scp·sftp 실행 금지. 서버 코드 열람은 필요한 파일을 한 번에 sftp로 받아 로컬에서 읽는다.
```

- 포인터: `shared/RULES.md:55-57`

### 4.2 .agents/AGENTS.md diff

```diff
--- .agents/AGENTS.md
+++ .agents/AGENTS.md
@@ -26,3 +26,4 @@
 - R7-7 **실행자는 한 곳만**: EXE-40이 끝날 때까지 서버 실행은 노트북 루프(cts_260930)만 담당. 회사 PC는 코드·git 준비만 하고 서버 접속 0회. 교대는 Principal이 `STATE.md`로 지정할 때만.
 - R7-8 **위반 시 거부**: 위 규약 위반 명령은 실행 전에 스스로 걸러 보고서에 "규약 7-2로 미실행"이라고 적음.
+- R7-9 **cts_ssh.py 전용 (기계적 차단)**: 서버 접속(ssh·scp·sftp)은 shared\cts_ssh.py로만. 직접 ssh·scp·sftp 실행 금지. 서버 코드 열람은 필요한 파일을 한 번에 sftp로 받아 로컬에서 읽는다.
```

- 포인터: `.agents/AGENTS.md:26-28`

### 4.3 shared/LESSONS.md 기록 내역

- 추가 항목 (20번):
  `20. 서버 접속(ssh·scp·sftp)은 shared\cts_ssh.py로만. 직접 ssh·scp·sftp 실행 금지. 서버 코드 열람은 필요한 파일을 한 번에 sftp로 받아 로컬에서 읽는다.`
- 포인터: `shared/LESSONS.md:20-21`

---

## 5. 서버 상태 1줄 요약

- tmux 세션 `s7_train_p4_prime` GPU 0 할당 PID 3793541 가동 중 (직전 접속 2026-10-04 17:39:42 KST 기준).

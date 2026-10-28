<!-- SHA256: 175550070640d176bd8c24db7970f5fd9b76502ec8e64051ac8567b5f8dd4209 -->
# REPORT-0052: `scripts/s7deq_soft_v2_launch_v4.sh` 작성 및 로컬 단위시험 7종, 서버 업로드·SHA 대조

- **작성 일시**: 2026-10-06 21:05 KST
- **작성 주체**: 회사 PC 정식 루프 에이전트 (Antigravity)
- **근거 작업**: `shared/tasks/TASK-0052.md`
- **관련 문서**: AUDIT-P009 재감사 1 (FIX-5~7, 권고), `shared/reports/REPORT-0051.md`
- **산출물**:
  1. `scripts/s7deq_soft_v2_launch_v4.sh` (SHA-256 `286c8455bf1d5a4e4c904cab640614bdbf2c74f38a76378897f1c551ce91d159`, 14,620 B, 실행 권한 755)
  2. `scratch/test_launch_v4_paths.py` (로컬 단위시험 7종 시험기)
  3. `scratch/task0052_local_test_out.txt` (로컬 단위시험 7종 원시 로그)
  4. `scratch/diff_launch_v3_v4.patch` (launch_v3 대비 launch_v4 diff, 218행)
  5. `scratch/grep_python_launch_v4.txt` (python 호출 검속 로그, 2행)
  6. `scratch/task0052_out.txt` (서버 SFTP 업로드 및 단일 SSH 대조 로그)

---

## 1. 개요 및 요약

TASK-0052 지시 및 AUDIT-P009 재감사 1(FIX-5~7 및 권고)에 따라 S7 부드러운 스위치 실행기 4판(`scripts/s7deq_soft_v2_launch_v4.sh`)을 작성하고 로컬 시험 및 서버 반영을 마쳤다.

1. **FIX-5 반영 (기준 SHA 상수화 및 fail-closed 닫힌 검속)**:
   - `EXPECTED_REF_SHA="5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6"`를 `readonly` 상수로 명시 선언했다.
   - 점검 전 기준 파일 SHA 불일치 시 `exit 4`, 점검 후 기준 파일 SHA 변조 또는 불일치 시 `exit 4`로 즉시 종료하는 닫힌 검속을 구현했다.
2. **FIX-6 반영 (단일 가상환경 해석기 고정 및 기동 로그)**:
   - 점검, 본 실험, 분석, SMOKE 전 경로의 파이썬 호출을 `readonly PYTHON="/home/shoon/system1_5/.venv/bin/python"` 단일 변수로 일원화했다.
   - 기동 시 `${PYTHON} -c "import sys,torch;print(sys.executable, torch.__version__, torch.version.cuda)"` 환경 로그를 기록하며, 맨 `python` 호출은 0건이다 (`grep -n "python"` 결과 30·35행 정의부 2건 외 호출 0건).
3. **FIX-7 반영 (정규 경로 상수 잠금 및 TEST_MODE=1 정규 GPU 진입 차단)**:
   - `BASE_DIR="/home/shoon/cts_e1_s5"`, `EXPECTED_74_SHA="42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3"`, `EXPECTED_REF_SHA`, `PYTHON` 4종을 정규 경로에서 `readonly` 상수로 고정했다.
   - 환경변수를 통한 덮어쓰기는 `TEST_MODE=1`에서만 모의 시험용으로 허용되며, `TEST_MODE=1` 활성 상태에서 정규 GPU 실행 경로 진입 시 즉시 `exit 5`로 차단된다.
4. **권고 반영**:
   - `results_s7/anchor_ref_gpu1_check_v21_extra.pt` 점검 전 기 존재 시 `exit 3`으로 즉시 종료한다.
   - 381행 마지막 평가 동률은 로그 기록 전용이며, 그 문항의 $\max|\Delta| \neq 0$인 경우에만 `exit 1`로 점검 전체를 중단함을 로그 문구에 명시했다.
5. **로컬 시험 7종 전원 PASS**:
   - exit 3 (2종), exit 4 (3종), exit 5 (1종), 정규 모드 상수 잠금 (1종) 총 7종 단위시험을 실행하여 전건 충족(RC=0)을 달성했다.
6. **서버 업로드 및 SHA 대조 (서버 GPU 실행 0건)**:
   - SFTP를 통해 서버에 업로드 후 단일 SSH 호출로 SHA-256(`286c8455bf...`, 14,620 B)의 로컬-서버 100% 비트 단위 일치를 계측했다.
   - 서버 내 `launch_v4.sh` 실행은 0건을 엄수했다. GPU 0 `p6_chain` Condition C(PID 245681, 473행 누적)는 무간섭 순항 중이며, GPU 1은 유휴(0%, 16 MiB) 상태를 보존하고 있다.

---

## 2. 코드 수정 상세 (launch_v3 $\rightarrow$ launch_v4)

### 2.1 주요 수정 대비표

| 항목 | launch_v3.sh (744d0ce2…) | launch_v4.sh (286c8455…) | 수정 목적 및 조치 (AUDIT-P009 재감사 1) |
|---|---|---|---|
| 기준 SHA | `EXPECTED_REF_SHA` 환경변수 의존 (부재 시 건너뜀) | `readonly EXPECTED_REF_SHA="5103642d…"` 상수 고정 | FIX-5: 환경변수 누락 시 우회 차단, 사전·사후 SHA 불일치 시 `exit 4` |
| 사후 SHA 대조 | 사후 SHA와 사전 SHA 비교 (`ACTUAL_POST` vs `ACTUAL_PRE`) | 사후 SHA와 상수 비교 (`ACTUAL_POST` vs `EXPECTED_REF_SHA`) | FIX-5: 사전 변조 후 사후 동일 상태 오인 차단, 상시 고정값 대조 |
| 파이썬 해석기 | 점검은 `PYTHON_BIN`, 본 실행·분석·SMOKE는 맨 `python` | 모든 호출 `${PYTHON}` 단일 변수 고정 (`system1_5/.venv`) | FIX-6: PATH 오염 차단, 해석기·torch 불일치 방지, 맨 python 0건 |
| 기동 환경 로그 | 없음 | `${PYTHON} -c "import sys,torch;print(...)"` | FIX-6: 기동 시 해석기 경로, torch 버전, CUDA 버전 로그 기록 |
| 경로 및 74 SHA | `BASE_DIR:-...`, `EXPECTED_74_SHA:-...` 환경변수 허용 | 정규 모드 `readonly` 상수, `TEST_MODE=1`에서만 오버라이드 | FIX-7: 정규 실행 시 외부 환경변수에 의한 경로·설정 오염 차단 |
| TEST_MODE 가드 | 없음 | 정규 GPU 경로 진입 직전 `TEST_MODE=1` 검사 시 `exit 5` | FIX-7: 모의 시험 설정으로 정규 GPU 실행 경로 진입 원천 봉쇄 |
| 추가 기준 파일 점검 | 없음 (기 존재 시 덮어쓰기 위험) | `results_s7/anchor_ref_gpu1_check_v21_extra.pt` 존재 시 `exit 3` | 권고: 정본 실행기가 기존 키를 재사용하여 점검을 왜곡하는 위험 방지 |
| 381행 동률 진단 | 일반 로그 문구 | 동률 시 기록 전용, $\max\|\Delta\|\neq 0$ 시에만 `exit 1` 명시 | 권고: 381행 경계 조건 로그 설명 명확화 |

### 2.2 맨 `python` 호출 0건 검속 (`grep -n "python"`)

스크립트 내 `python` 문자열 검색 결과 (`scratch/grep_python_launch_v4.txt`):
```text
30:  PYTHON="${PYTHON:-/home/shoon/system1_5/.venv/bin/python}"
35:  readonly PYTHON="/home/shoon/system1_5/.venv/bin/python"
```
- 30행: `TEST_MODE=1` 모의 시험 시 환경변수 수용 부
- 35행: 정규 실행 시 `readonly` 상수 선언 부
- 셸 실행 명령: `${PYTHON} scripts/s7deq_soft_v2_run_v2.py ...` (SMOKE 61행, 본 실행 286행), `${PYTHON} scripts/s7deq_soft_v2_analyze.py ...` (SMOKE 76행, 본 분석 300행), `${PYTHON} - << 'EOF'` (점검 125행).
- 판정: 맨 `python` 호출 0건 계측.

### 2.3 launch_v3 대비 launch_v4 diff 요약

전체 diff 파일: `scratch/diff_launch_v3_v4.patch` (218행). 주요 변경 hunk:
1. 머리말: launch_v4 선언, FIX-5~7 명시.
2. 환경 및 상수 바인딩 (26~37행):
   ```bash
   if [ "${TEST_MODE:-0}" = "1" ]; then
     BASE_DIR="${BASE_DIR:-/home/shoon/cts_e1_s5}"
     EXPECTED_74_SHA="${EXPECTED_74_SHA:-42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3}"
     EXPECTED_REF_SHA="${EXPECTED_REF_SHA:-5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6}"
     PYTHON="${PYTHON:-/home/shoon/system1_5/.venv/bin/python}"
   else
     readonly BASE_DIR="/home/shoon/cts_e1_s5"
     readonly EXPECTED_74_SHA="42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3"
     readonly EXPECTED_REF_SHA="5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6"
     readonly PYTHON="/home/shoon/system1_5/.venv/bin/python"
   fi
   ```
3. 해석기 환경 로그 (47~49행):
   ```bash
   echo "Python environment:" | tee -a "${LOG_FILE}"
   ${PYTHON} -c "import sys,torch;print(sys.executable, torch.__version__, torch.version.cuda)" | tee -a "${LOG_FILE}"
   ```
4. 추가 기준 파일 기 존재 차단 (129~133행):
   ```bash
   if [ -f "${ANCHOR_EXTRA_REF_PT}" ]; then
     echo "[ERROR] Pre-check extra reference file ${ANCHOR_EXTRA_REF_PT} already exists. Aborting with exit code 3." | tee -a "${LOG_FILE}"
     exit 3
   fi
   ```
5. 사전·사후 SHA 불일치 닫힌 검속 (143~146행, 305~308행):
   ```bash
   if [ "${ACTUAL_PRE_REF_SHA}" != "${EXPECTED_REF_SHA}" ]; then
     echo "[ERROR] Reference file SHA mismatch pre-check. Expected: ${EXPECTED_REF_SHA}, Got: ${ACTUAL_PRE_REF_SHA}. Aborting with exit code 4." | tee -a "${LOG_FILE}"
     exit 4
   fi
   ...
   if [ "${ACTUAL_POST_REF_SHA}" != "${EXPECTED_REF_SHA}" ]; then
     echo "[ERROR] Post-check reference SHA mismatch/altered! Expected: ${EXPECTED_REF_SHA}, Got: ${ACTUAL_POST_REF_SHA}. Aborting with exit code 4." | tee -a "${LOG_FILE}"
     exit 4
   fi
   ```
6. 정규 GPU 경로 진입 가드 (311~315행):
   ```bash
   if [ "${TEST_MODE:-0}" = "1" ]; then
     echo "[TEST_MODE] Regular GPU execution path blocked in TEST_MODE=1. Exiting with code 5." | tee -a "${LOG_FILE}"
     exit 5
   fi
   ```

---

## 3. 로컬 단위시험 7종 결과 (`scratch/task0052_local_test_out.txt`)

`scratch/test_launch_v4_paths.py`를 통해 `TEST_MODE=1` 모의 환경 및 정규 모드 상수 잠금 시험을 전수 수행했다.

### 3.1 시험 결과 요약표

| 시험 번호 | 시험 항목 | 기대 반환코드 | 실측 반환코드 | 결과 | 원시 로그 포인터 (`scratch/task0052_local_test_out.txt`) |
|---|---|---|---|---|---|
| Test 1 | 출력 파일 `anchor_soft_eps1e-12_check.pt` 기 존재 | 3 | 3 | PASS | line 7, line 14 |
| Test 2 | 추가 기준 파일 `anchor_ref_gpu1_check_v21_extra.pt` 기 존재 | 3 | 3 | PASS | line 18, line 25 |
| Test 3 | 기준 파일 `anchor_ref_gpu1_check.pt` 부재 | 4 | 4 | PASS | line 29, line 36 |
| Test 4 | 기준 파일 사전 SHA 불일치 | 4 | 4 | PASS | line 40, line 47 |
| Test 5 | 기준 파일 사후 SHA 변조 (사후 변경) | 4 | 4 | PASS | line 51, line 58 |
| Test 6 | `TEST_MODE=1` 상태에서 정규 GPU 실행 경로 진입 시도 | 5 | 5 | PASS | line 62, line 69 |
| Test 7 | 정규 모드(`TEST_MODE` 미설정) 환경변수 4종 덮어쓰기 무시 검속 | 0 | 0 | PASS | line 73, lines 75~78 |

### 3.2 시험별 실측 출력 원문 인용

1. **Test 1 (Exit 3: 출력 파일 기 존재)**:
   - 인용 (`task0052_local_test_out.txt:13-15`):
     ```text
     === §5 PRE-LAUNCH ANCHOR CHECK (GPU 1, v2.1 eps=1e-12) ===
     [ERROR] Pre-check output file results_s7/anchor_soft_eps1e-12_check.pt already exists. Aborting with exit code 3.
     >>> Test 1 result: PASS (exit 3 recorded)
     ```
2. **Test 2 (Exit 3: 추가 기준 파일 기 존재)**:
   - 인용 (`task0052_local_test_out.txt:24-26`):
     ```text
     === §5 PRE-LAUNCH ANCHOR CHECK (GPU 1, v2.1 eps=1e-12) ===
     [ERROR] Pre-check extra reference file results_s7/anchor_ref_gpu1_check_v21_extra.pt already exists. Aborting with exit code 3.
     >>> Test 2 result: PASS (exit 3 recorded)
     ```
3. **Test 3 (Exit 4: 기준 파일 부재)**:
   - 인용 (`task0052_local_test_out.txt:35-37`):
     ```text
     === §5 PRE-LAUNCH ANCHOR CHECK (GPU 1, v2.1 eps=1e-12) ===
     [ERROR] Reference file results_s7/anchor_ref_gpu1_check.pt not found. Aborting with exit code 4.
     >>> Test 3 result: PASS (exit 4 recorded)
     ```
4. **Test 4 (Exit 4: 사전 기준 SHA 불일치)**:
   - 인용 (`task0052_local_test_out.txt:46-48`):
     ```text
     Reference file SHA pre-check: 8587350e5c6e23dea4829b3490d8635f199b4a3389c5c4b81cbb0b673433e12b
     [ERROR] Reference file SHA mismatch pre-check. Expected: 0000000000000000000000000000000000000000000000000000000000000000, Got: 8587350e5c6e23dea4829b3490d8635f199b4a3389c5c4b81cbb0b673433e12b. Aborting with exit code 4.
     >>> Test 4 result: PASS (exit 4 recorded)
     ```
5. **Test 5 (Exit 4: 사후 기준 SHA 변조 / 사후 변경)**:
   - 인용 (`task0052_local_test_out.txt:55-59`):
     ```text
     Reference file SHA pre-check: 8587350e5c6e23dea4829b3490d8635f199b4a3389c5c4b81cbb0b673433e12b
     [TEST_MODE] Simulating reference file modification during run...
     Reference file SHA post-check: a0c5540987587caf3f3c7d1faea055ed01ea63d7ebbc4e59c43fc73d63170146
     [ERROR] Post-check reference SHA mismatch/altered! Expected: 8587350e5c6e23dea4829b3490d8635f199b4a3389c5c4b81cbb0b673433e12b, Got: a0c5540987587caf3f3c7d1faea055ed01ea63d7ebbc4e59c43fc73d63170146. Aborting with exit code 4.
     >>> Test 5 result: PASS (exit 4 recorded on post-check alteration)
     ```
6. **Test 6 (Exit 5: TEST_MODE=1 정규 GPU 경로 진입 차단)**:
   - 인용 (`task0052_local_test_out.txt:66-70`):
     ```text
     Reference file SHA pre-check: 8587350e5c6e23dea4829b3490d8635f199b4a3389c5c4b81cbb0b673433e12b
     [TEST_MODE] Simulating successful anchor check...
     Reference file SHA post-check: 8587350e5c6e23dea4829b3490d8635f199b4a3389c5c4b81cbb0b673433e12b
     [TEST_MODE] Regular GPU execution path blocked in TEST_MODE=1. Exiting with code 5.
     >>> Test 6 result: PASS (exit 5 recorded)
     ```
7. **Test 7 (정규 모드 환경변수 무시 및 상수 잠금 검속)**:
   - 인용 (`task0052_local_test_out.txt:75-80`):
     ```text
     EVAL_BASE_DIR=/home/shoon/cts_e1_s5
     EVAL_74_SHA=42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3
     EVAL_REF_SHA=5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6
     EVAL_PYTHON=/home/shoon/system1_5/.venv/bin/python

     >>> Test 7 result: PASS (All 4 constants locked strictly in regular mode)
     ```
   - 악의적 환경변수(`BASE_DIR=/malicious/dir`, `EXPECTED_74_SHA=malicious_...`, `EXPECTED_REF_SHA=malicious_...`, `PYTHON=/malicious/...`)를 상위에서 주입하여도 정규 모드에서는 선언된 상수로 덮어써지며 `readonly` 잠금이 유지됨을 실증했다.

---

## 4. 서버 반영 및 팩트 수집 실측 (`scratch/task0052_out.txt`)

2026-10-06 21:01:42 KST SFTP 업로드 후 65초 쿨다운을 거쳐 21:02:48 KST 단일 SSH 호출(`task0052_verify_sha_and_telemetry`)로 서버 상태를 수집했다.

### 4.1 서버 파일 해시 및 메타데이터

| 파일 경로 | SHA-256 | 크기 (bytes) | 서버 타임스탬프 (UTC+9) | 상태 및 비트 일치 | 원시 포인터 (`task0052_out.txt`) |
|---|---|---|---|---|---|
| `scripts/s7deq_soft_v2_launch_v4.sh` | `286c8455bf1d5a4e4c904cab640614bdbf2c74f38a76378897f1c551ce91d159` | 14,620 | 2026-10-06 21:01:42 | 로컬 대비 비트 일치 (권한 755) | line 6, line 7 |
| `results_s7/anchor_ref_gpu1_check.pt` | `5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6` | 26,410,991 | 2026-10-06 20:17:04 | launch_v4 상수와 비트 일치 | line 10, line 11 |
| `results_s7/anchor_soft_eps1e-12_check.pt` | - | - | - | `NOT_EXISTS` (exit 3 차단 해소) | line 13 |
| `results_s7/anchor_ref_gpu1_check_v21_extra.pt` | - | - | - | `NOT_EXISTS` (exit 3 차단 해소) | line 15 |

### 4.2 자원 분리 및 텔레메트리 실측

- **서버 GPU 실행 횟수**: **0건** (TASK-0052 지시 엄수: 파일 업로드 및 SHA 검속만 수행).
- **활성 세션**: `p6_chain: 1 windows` (created Mon Oct 5 19:54:38 2026) 단독 가동 중 (`task0052_out.txt:18`).
- **GPU 0 (PID 245681, p6_chain Condition C)**: 99 % 점유율, 15,846 MiB 점유, 53 ℃, 133.31 W (`task0052_out.txt:21, 31`).
- **GPU 1 (물리 UUID: GPU-9410691f...)**: 0 % 점유율, 16 MiB 점유, 29 ℃, 12.23 W (유휴 보존) (`task0052_out.txt:22`).
- **GPU 2~7**: 0 % 점유율, 16 MiB 점유 (GPU 4 영구 봉인 유지, 2/3/5/6/7 유휴) (`task0052_out.txt:23-28`).
- **P6 생성 진척**: `results_s7/p6_gen_C.jsonl` **473행** (626,541 B, MTIME 21:02:36 KST) (`task0052_out.txt:42, 48`). 직전 계측(20:41 416행) 대비 57행 순항 생성 중. 내용 열람 0건 유지.

---

## 5. 거버넌스 및 규칙 준수 점검

1. **GPU 사용 정책 (CTS-428)**:
   - GPU 1: 유휴 보존, TASK-0052 서버 내 스크립트 실행 0건 엄수.
   - GPU 4: 영구 봉인 유지 (점유율 0%, 16 MiB).
   - GPU 2·3·5·6·7: 사용 0건 엄수.
2. **서버 접속 예절 규약 v1 (원장 #399, R7-1~R7-9)**:
   - 모든 연결은 `shared/cts_ssh.py` 경유 단독 수행 (`shared/ssh_log.csv` 기록).
   - 직전 연결(20:41:54 KST) 이후 1,174초(>19분) 경과 후 SFTP 접속(21:01:42 KST).
   - SFTP와 SSH 간 65초 강제 쿨다운 준수 (R7-1, 60초 이상 간격 엄수).
   - 최근 1시간 누적 연결 수: 7회 ($\le 20$회 상한 충족).
3. **규칙 R3-5 준수**:
   - 9대 금지 규정(R3-5 9종 단어) 사용 0건 엄수.
   - 모든 수치에 원시 출력 포인터(`task0052_local_test_out.txt`, `task0052_out.txt` 행 번호) 명시.
4. **줄바꿈 및 무결성**:
   - 산출 스크립트 및 본 보고서 CR(0x0D) 바이트 0건 (LF 전용).
   - 본 보고서 1행 LF self-hash 탑재.

---

## 6. 후속 조치 권고

- `scripts/s7deq_soft_v2_launch_v4.sh`의 서버 반영 및 SHA-256(`286c8455bf1d5a4e4c904cab640614bdbf2c74f38a76378897f1c551ce91d159`) 대조가 마쳐졌으므로, 초안 문서(`DRAFT_PREREG_S7_DEQ_v2_1.md`) 86행의 실행기 SHA를 이 값으로 갱신하는 후속 작업(TASK-0053 등)을 진행할 준비가 갖추어졌다.
- 본 보고서 발행 후 `STATE.md` 갱신 및 `wait_for_task.py 53` 대기를 즉시 기동한다.

<!-- SHA256: c62529556577ccbd9ffff2dc714c6661f9da5957b22259fa29ff5c86cabf7231 -->
# REPORT-0056: (A) 부드러운 스위치 진행 점검 및 (B) ①-a 재계산 실행기 배치

- **발행 일시**: 2026-10-07 04:15 KST
- **수행자**: 회사 PC 정식 루프 (CTS 실행 에이전트)
- **근거 문서**: TASK-0056, PREREG_S7_DEQ_v2.1(670850c1…) §4·§6·§7, PREREG_S7_DEQ_v1.4(04112522…) §10(h), AUDIT-0050
- **규약 준수**: RULES R3-5 준수, 9대 금지어 0건, 수치마다 포인터 명시, p6_chain·s7deq_soft_v2 무간섭, GPU 4 영구 봉인 유지 (GPU 2~7 미사용), R7-1/R7-3 접속 규약 준수, CR 0건 (LF 전용)

---

## 1. 수행 결과 총괄 요약

1. **(A) 부드러운 스위치 진행 계측 및 P6 체인 진척 점검 (단일 SSH 묶음)**:
   - **호스트 및 재부팅 진단**: 부팅 시각 `2026-10-05 16:32:53` 불변(1일 11시간 25분 연속 가동), 실측 3차 재부팅 부재 계측 (포인터: `scratch/task0056_out.txt:6-7`).
   - **부드러운 스위치 (GPU 1)**:
     * $\varepsilon=0.005$: 21:23:30 KST 기동 후 23:32:16 KST 마침(74문항 종료, .pt 99,050,013 B, .json, .md 생성, 포인터: `scratch/task0056_out.txt:13, 17, 33-35`).
     * $\varepsilon=0.01$: 23:32:16 KST 기동 후 10/7 01:48:30 KST 마침(74문항 종료, .pt 99,049,795 B, .json, .md 생성, 포인터: `scratch/task0056_out.txt:14, 18, 36-38`).
     * $\varepsilon=0.02$: 10/7 01:48:30 KST 기동 후 현재 `[0064/0074] q4171_a0.25` 연산 중(PID 318381, GPU 1 점유 100%, 30,950 MiB, .pt 임시 크기 80,310,835 B, 누적 헤더 212건, 포인터: `scratch/task0056_out.txt:15, 22, 28-29, 39, 67, 76`).
     * `ALL S7 DEQ SOFT V2 RUNS FINISHED` 미도달(`Not yet finished`, 포인터: `scratch/task0056_out.txt:20`). TASK-0056 3항 규정에 따라 진행 헤더 번호(`[0064/0074]`) 및 개수(212건)만 기록하고 파일 내용 열람 및 로컬 내려받기 0건 유지.
     * v5 실행 구간(599행 이후) 오류·Traceback 0건 계측 (포인터: `scratch/task0056_out.txt:31`).
   - **P6 체인 (GPU 0)**:
     * Condition A: 마침 (`P6_GEN_A_DONE.flag` 생성, 600행, 포인터: `scratch/task0056_out.txt:85, 103`).
     * Condition B1: 마침 (`P6_GEN_B1_DONE.flag` 생성, 600행, 포인터: `scratch/task0056_out.txt:87, 105`).
     * Condition E1: 기동 후 390행 연산 진행 중 (PID 327901, GPU 0 점유 99%, 16,134 MiB, 포인터: `scratch/task0056_out.txt:88, 109, 66, 77`).
   - **GPU 자원 격리**: GPU 0(PID 327901 P6 E1) 및 GPU 1(PID 318381 soft v2 eps=0.02) 단독 가동, GPU 2~7 0% 16 MiB 유휴 (포인터: `scratch/task0056_out.txt:55-67`).
2. **(B) ①-a 재계산 실행기 (`scripts/kstar_replay_launch_v1.sh`) 작성 및 서버 업로드 마침**:
   - **스크립트 제원**: 크기 18,247 B, SHA-256 `98b2e6527cb824b5c2a4766ec1178d2eef4ea3add4dfb8ba7eaf5cae0727a34d`.
   - **v1.4 §10(h) 핵심 요건 구현**:
     * `--gpu 0|1` 인자 필수화 및 물리 UUID 상수 결속 (GPU 0: `GPU-71be98b0...`, GPU 1: `GPU-9410691f...`).
     * 해석기 `/home/shoon/system1_5/.venv/bin/python` 상수 고정 및 시작 시 버전 로깅.
     * 8대 전제 파일 SHA 사전 대조 (불일치 시 exit 4 종료).
     * `KSTAR_REPLAY_STARTED.flag` 및 `kstar_replay_precheck5_gpu{0,1}.pt` 존재 시 exit 3 차단 (관찰 O-2).
     * 1단계 사전 점검 (5문항: Q#31, 277, 406, 711, 858) 정본 실행기로 계산 및 명령 전문 로깅 (관찰 O-1), 분석기 v2로 사전 점검 판정 (`k_match`, `cond_i`, `cond_ii` 미충족 시 exit 6).
     * 2단계 본 실행: STARTED 플래그(시각·UUID·PCI·입력 SHA) 기록 -> v1.4 §10(h) 325문항 실행 명령 -> DONE 플래그 기록 -> 분석기 v2 실행(JSON·MD 출력).
   - **로컬 단위 시험**: 10종 시험 항목 전수 PASS (`scratch/task0056_local_test_out.txt:16`, RC=0). 맨 python 0건 (`scratch/grep_python_kstar_replay.txt`, 20행).
   - **서버 업로드 및 SHA 대조**: SFTP 업로드 후 단일 SSH 대조 결과, 서버 SHA-256 `98b2e6527cb824b5c2a4766ec1178d2eef4ea3add4dfb8ba7eaf5cae0727a34d` 비트 일치, 크기 18,247 B, 권한 755 계측 (포인터: `scratch/task0056_upload_out.txt:11-12`). 이번 TASK 서버 GPU 실행 0건 엄수.

---

## 2. (A) 부드러운 스위치 및 서버 상세 계측 결과

### 2.1. 호스트 가동 및 세션 상태
- 부팅 시각: `2026-10-05 16:32:53` (포인터: `scratch/task0056_out.txt:6`)
- 가동 시간: `03:58:32 up 1 day, 11:25,  2 users,  load average: 2.00, 2.00, 2.00` (포인터: `scratch/task0056_out.txt:7`)
- 활성 tmux 세션: 2건 (`p6_chain`, `s7deq_soft_v21`, 포인터: `scratch/task0056_out.txt:9-10`)

### 2.2. 부드러운 스위치 실행 진척 (GPU 1)
- 실행 로그: `raw_s7/s7deq_soft_v2.log`
- 기동 및 종료 기록 원문:
  * `=== RUNNING EPS=0.005 AT 2026-10-06 21:23:30 KST ===` (포인터: `scratch/task0056_out.txt:13`)
  * `Completed eps=0.005 at 2026-10-06 23:32:16 KST` (포인터: `scratch/task0056_out.txt:17`)
  * `=== RUNNING EPS=0.01 AT 2026-10-06 23:32:16 KST ===` (포인터: `scratch/task0056_out.txt:14`)
  * `Completed eps=0.01 at 2026-10-07 01:48:30 KST` (포인터: `scratch/task0056_out.txt:18`)
  * `=== RUNNING EPS=0.02 AT 2026-10-07 01:48:30 KST ===` (포인터: `scratch/task0056_out.txt:15`)
  * `ALL S7 DEQ SOFT V2 RUNS FINISHED`: `Not yet finished` (포인터: `scratch/task0056_out.txt:20`)
- 진행 헤더 집계:
  * 로그 내 전체 헤더 수: 212건 (포인터: `scratch/task0056_out.txt:22`)
  * 최근 5개 헤더: `[0060/0074] q3976_a0.25`, `[0061/0074] q3998_a0.25`, `[0062/0074] q4005_a0.25`, `[0063/0074] q4068_a0.25`, `[0064/0074] q4171_a0.25` (포인터: `scratch/task0056_out.txt:24-28`)
  * 최신 문항 헤더: `[0064/0074]` (포인터: `scratch/task0056_out.txt:29`)
- v5 실행 구간(599행 이후) 오류 검속:
  * `No errors or tracebacks detected in v5.` (포인터: `scratch/task0056_out.txt:31`)
- 산출 파일 목록 및 SHA-256:
  * `results_s7/s7deq_soft_v2_eps0.005.json`: 2,263 B, SHA `4a767f4c6b7155ff9802437434e2c1406885c8b74d3c7359d61eadabe660fd89` (포인터: `scratch/task0056_out.txt:33, 41`)
  * `results_s7/s7deq_soft_v2_eps0.005.md`: 1,046 B, SHA `93f2b37ef93302aa6a8c225ac023ed73eab9d0e164e65de0501010dc8dd70950` (포인터: `scratch/task0056_out.txt:34, 42`)
  * `results_s7/s7deq_soft_v2_eps0.005.pt`: 99,050,013 B, SHA `b0f13e9c101e3c08fd9df808b91152ca5f5b27299bf635c96eee0d90b841f1cd` (포인터: `scratch/task0056_out.txt:35, 43`)
  * `results_s7/s7deq_soft_v2_eps0.01.json`: 2,247 B, SHA `ef360c7d1b2ce3278e680a7a75a94e653c200ad3e54ba54ccefcaa9ace875a6e` (포인터: `scratch/task0056_out.txt:36, 44`)
  * `results_s7/s7deq_soft_v2_eps0.01.md`: 1,044 B, SHA `c98dca008621d51755d48f191eb9ea6296ddd5ac4ec37a8e8960c46dd8a369e3` (포인터: `scratch/task0056_out.txt:37, 45`)
  * `results_s7/s7deq_soft_v2_eps0.01.pt`: 99,049,795 B, SHA `1799763c981cbb54376d732cb2151019c79302720759a3bc73b9f6f280306908` (포인터: `scratch/task0056_out.txt:38, 46`)
  * `results_s7/s7deq_soft_v2_eps0.02.pt`: 80,310,835 B (연산 진행 중, SHA `0ff93cdb035d78113e14b22cde5c6e8f85025586862cca8c1279b09617a8db7f`, 포인터: `scratch/task0056_out.txt:39, 47`)
- JSON 메타데이터 입력 SHA 기록 필드 인용:
  * `eps0.005.json`: `script_sha256: 07d63b1e...`, `input_pt_sha256: b0f13e9c...`, `items_json_sha256: 42df1f49...`, `ref_pt_sha256: 8c1d9a38...` (포인터: `scratch/task0056_out.txt:50`)
  * `eps0.01.json`: `script_sha256: 07d63b1e...`, `input_pt_sha256: 1799763c...`, `items_json_sha256: 42df1f49...`, `ref_pt_sha256: 8c1d9a38...` (포인터: `scratch/task0056_out.txt:52`)

### 2.3. P6 체인 진척 현황 (GPU 0)
- `raw_s7/p6_chain.log` 최근 5행:
  * `[P6-CHAIN] Running Generation Condition A (shift=)...` (포인터: `scratch/task0056_out.txt:84`)
  * `[P6-CHAIN] Generation Condition A (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_A_DONE.flag.` (포인터: `scratch/task0056_out.txt:85`)
  * `[P6-CHAIN] Running Generation Condition B (shift=1)...` (포인터: `scratch/task0056_out.txt:86`)
  * `[P6-CHAIN] Generation Condition B (shift=1) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_B1_DONE.flag.` (포인터: `scratch/task0056_out.txt:87`)
  * `[P6-CHAIN] Running Generation Condition E1 (shift=)...` (포인터: `scratch/task0056_out.txt:88`)
- P6 생성 결과 파일 행 수:
  * `results_s7/p6_gen_A.jsonl`: 600행 (포인터: `scratch/task0056_out.txt:103`)
  * `results_s7/p6_gen_B1.jsonl`: 600행 (포인터: `scratch/task0056_out.txt:105`)
  * `results_s7/p6_gen_C.jsonl`: 600행 (포인터: `scratch/task0056_out.txt:107`)
  * `results_s7/p6_gen_E1.jsonl`: 390행 (생성 진행 중, 포인터: `scratch/task0056_out.txt:109`)

### 2.4. GPU 0~7 텔레메트리 및 프로세스
| GPU 인덱스 | 물리 UUID | 사용률 | 사용 메모리 | 온도 | 전력 | 할당 프로세스 |
|---|---|---|---|---|---|---|
| 0 | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | 99 % | 16,134 MiB | 51 ℃ | 130.12 W | PID 327901 (P6 E1) |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 100 % | 30,950 MiB | 52 ℃ | 118.96 W | PID 318381 (s7deq eps=0.02) |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 16 MiB | 31 ℃ | 21.55 W | (유휴) |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 16 MiB | 32 ℃ | 17.62 W | (유휴) |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 16 MiB | 30 ℃ | 23.30 W | (영구 봉인) |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 16 MiB | 31 ℃ | 16.88 W | (유휴) |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 16 MiB | 31 ℃ | 15.48 W | (유휴) |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 16 MiB | 31 ℃ | 20.96 W | (유휴) |

- 근거 포인터: `scratch/task0056_out.txt:56-67, 76-77`
- 시스템 메모리: 전체 251 GiB 중 가용 237 GiB (사용 12 GiB), 스왑 0 B 사용 (포인터: `scratch/task0056_out.txt:80-81`).

---

## 3. (B) ①-a 재계산 실행기 구현 및 검증 결과

### 3.1. 전제 파일 무결성 및 플래그 부재 계측
서버 정본 파일 SHA-256 검속 결과 (포인터: `scratch/task0056_out.txt:114-121`):
| 파일 식별 | 서버 파일 경로 | 요구 SHA-256 | 실측 SHA-256 | 대조 결과 |
|---|---|---|---|---|
| 실행기 | `scripts/s7_zstar_lowest_run.py` | `779ea1f1acfe85a9…` | `779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba` | 일치 |
| 풀이기 | `cts/deq/zstar_canonical_lowest.py` | `9f73fb272353e8a3…` | `9f73fb272353e8a307b9e628c65d1f17eaff164e2138fa3d232917958d6a6b2f` | 일치 |
| 목록 | `configs/s7_train1000_kstar_replay_325.json` | `550250e399ea9656…` | `550250e399ea96560b8360cdbdd539a82983a25fb435f994d0319da10a8a7862` | 일치 |
| 설정 | `configs/zstar_canonical.yaml` | `bd8e4e96cf971de3…` | `bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157` | 일치 |
| 하네스 | `/home/shoon/cts_e1/scratch/harness_A_v1.py` | `5fb768a8f56add15…` | `5fb768a8f56add15c7cdca069f5e1a7f80d16fd231f8ecae1fca4fcf6920bf52` | 일치 |
| 분석기 | `scripts/p5_promote_1a_replay_v2.py` | `a1d74cf6cd3c4773…` | `a1d74cf6cd3c4773aa92a2d18400eb67f8a347938243fe7c8a3a36f215a47da5` | 일치 |
| P4 정본 | `results_s7/zstar_train1000_P4.pt` | `37ea6807d112e44f…` | `37ea6807d112e44f55ab401f48b6c8e7051f3ca2800aa7cd4caa04a546fb4696` | 일치 |
| 입력 v3 | `results_s7/p5_inputs_train1000_v3.json` | `186d5b6cc6678b4d…` | `186d5b6cc6678b4df2e999919cc39b85d77b9b437dce10da5c553841d3bea1f1` | 일치 |

- 사전 충돌 파일 부재 계측 (포인터: `scratch/task0056_out.txt:123-126`):
  * `raw_s7/KSTAR_REPLAY_STARTED.flag`: NOT_EXISTS
  * `raw_s7/KSTAR_REPLAY_DONE.flag`: NOT_EXISTS
  * `results_s7/kstar_replay_precheck5_gpu0.pt`: NOT_EXISTS
  * `results_s7/kstar_replay_precheck5_gpu1.pt`: NOT_EXISTS

### 3.2. `scripts/kstar_replay_launch_v1.sh` 구현 명세
- **인자 파싱**: `--gpu 0|1` 필수화 (누락 또는 기타 값 입력 시 오류 종료 RC=1).
- **장치 바인딩**: GPU 0 선택 시 `GPU-71be98b0-6b81-9257-c622-97dc97d46480`, GPU 1 선택 시 `GPU-9410691f-6857-1642-ec7c-7596970e2f48` 물리 UUID 고정 결속.
- **파이썬 환경**: `/home/shoon/system1_5/.venv/bin/python` 단일 해석기 강제 및 시작 시 버전 로깅.
- **규칙 #381 메모리 제어**: `MALLOC_ARENA_MAX=1`, `MALLOC_TRIM_THRESHOLD_=131072`, OpenMP/MKL/Torch/Rayon 4스레드 강제.
- **사전 전제 검증**: 8개 파일 SHA 불일치 시 즉시 `exit 4` 종료.
- **이중 실행 차단**: STARTED 플래그 또는 해당 GPU 사전 점검 파일 이미 존재 시 즉시 `exit 3` 종료 (관찰 O-2 반영).
- **1단계 사전 점검**: 5문항(Q#31, 277, 406, 711, 858) 계산 명령 전문 로깅 후 해당 GPU에서 정본 실행기 구동. 산출물을 `p5_promote_1a_replay_v2.py` 사전 점검 기능으로 대조하여 5문항 전원 $k^*$ 일치, $\min(\rho_{\text{traj}})$ 상대차 $\le 10^{-2}$ 또는 절대차 $\le 1.2\times 10^{-7}$, $z^*$ 코사인 유사도 $\ge 0.9999$ 검속. 1건이라도 미달 시 `exit 6` 종료.
- **2단계 본 실행**: 사전 점검 충족 시 `KSTAR_REPLAY_STARTED.flag` 기록 -> 325문항 v1.4 §10(h) 정규 명령 실행 -> `KSTAR_REPLAY_DONE.flag` 기록 -> 분석기 v2 구동(JSON/MD 생성).

### 3.3. 로컬 단위 시험 10종 결과
실행 스크립트: `scratch/test_kstar_replay_launch_v1.py`
실행 로그 포인터: `scratch/task0056_local_test_out.txt:4-16`

```text
[Test 1: Missing BASE_DIR in TEST_MODE] PASS (RC=5)
[Test 2: Server production path rejection in TEST_MODE] PASS (RC=5)
[Test 3: Missing --gpu argument] PASS (RC=1)
[Test 4: Invalid --gpu argument (--gpu 2)] PASS (RC=1)
[Test 5: Prerequisite SHA mismatch (corrupted runner hash)] PASS (RC=4)
[Test 6: Existing KSTAR_REPLAY_STARTED.flag] PASS (RC=3)
[Test 7: Existing kstar_replay_precheck5_gpu0.pt] PASS (RC=3)
[Test 8: Pre-check criterion failure on item Q#31] PASS (RC=6)
[Test 9: Regular GPU execution guard in TEST_MODE] PASS (RC=5)
[Test 10: Full mock replay flow through Analyzer v2 on GPU 1] PASS (RC=0)
Test 10 Verification: All flags, replay .pt, analysis JSON/MD verified present.
=== ALL 10 UNIT TESTS PASSED SUCCESSFULLY (100% PASS / RC=0) ===
```

- `grep -n 'python\|TEST_' scripts/kstar_replay_launch_v1.sh` 산출 결과 (포인터: `scratch/grep_python_kstar_replay.txt:1-20`):
```text
65:# Overrides are permitted ONLY when TEST_MODE=1 with an explicit non-server BASE_DIR.
67:if [ \"${TEST_MODE:-0}\" = \"1\" ]; then
70:    echo \"[ERROR] TEST_MODE=1 requires an explicit BASE_DIR. Aborting with exit code 5.\" >&2
75:      echo \"[ERROR] TEST_MODE=1 cannot use server production directory ${BASE_DIR}. Aborting with exit code 5.\" >&2
80:  PYTHON=\"${PYTHON:-python}\"
81:  # Zero GPU usage in TEST_MODE
95:  unset TEST_MODE TEST_PRECHECK_FAIL TEST_ALTER_SHA TEST_MOCK_RUN 2>/dev/null || true
98:  readonly PYTHON=\"/home/shoon/system1_5/.venv/bin/python\"
119:echo \"Target GPU UUID: ${TARGET_GPU_UUID:-NONE (TEST_MODE)}\" | tee -a \"${LOG_FILE}\"
197:if [ \"${TEST_MODE:-0}\" = \"1\" ]; then
198:  if [ \"${TEST_PRECHECK_FAIL:-0}\" = \"1\" ]; then
199:    echo \"[TEST_MODE] Simulating pre-check failure on item Q#31...\" | tee -a \"${LOG_FILE}\"
209:    echo \"[TEST_MODE] Simulating pre-check passing all 5 items...\" | tee -a \"${LOG_FILE}\"
254:skip_sha = (os.environ.get(\"TEST_MODE\") == \"1\")
311:# Guard for TEST_MODE: block regular GPU execution unless mock test
312:if [ \"${TEST_MODE:-0}\" = \"1\" ] && [ \"${TEST_MOCK_RUN:-0}\" != \"1\" ]; then
313:  echo \"[TEST_MODE] Regular GPU execution path blocked in TEST_MODE=1. Exiting with code 5.\" | tee -a \"${LOG_FILE}\"
357:if [ \"${TEST_MODE:-0}\" = \"1\" ]; then
358:  echo \"[TEST_MODE] Simulating main 325-item replay run...\" | tee -a \"${LOG_FILE}\"
421:if [ \"${TEST_MODE:-0}\" = \"1\" ]; then
```
- 맨 python 직접 호출 0건 계측 (모든 python 호출은 `${PYTHON}` 변수로 바인딩).

### 3.4. 서버 업로드 및 원격 무결성 계측
- 업로드 일시: 2026-10-07 04:06:59 KST (SFTP batch, 포인터: `shared/ssh_log.csv:156`, `scratch/task0056_upload_out.txt:3-6`)
- 원격 검속 일시: 2026-10-07 04:08:05 KST (단일 SSH, 포인터: `shared/ssh_log.csv:157`, `scratch/task0056_upload_out.txt:8-12`)
- 원격 계측 원문:
  * `98b2e6527cb824b5c2a4766ec1178d2eef4ea3add4dfb8ba7eaf5cae0727a34d  scripts/kstar_replay_launch_v1.sh` (포인터: `scratch/task0056_upload_out.txt:11`)
  * `-rwxr-xr-x 1 shoon shoon 18247 2026-10-07 04:07:07.120205206 +0900 scripts/kstar_replay_launch_v1.sh` (포인터: `scratch/task0056_upload_out.txt:12`)
- 로컬 및 서버 SHA-256 비트 단위 100 % 일치, 실행 권한 755 충족.
- 이번 TASK 서버 GPU 실행 0건 준수.

---

## 4. 변경 파일 SHA-256 목록

| 구분 | 파일 경로 | SHA-256 | 비고 |
|---|---|---|---|
| 신규 스크립트 | `scripts/kstar_replay_launch_v1.sh` | `98b2e6527cb824b5c2a4766ec1178d2eef4ea3add4dfb8ba7eaf5cae0727a34d` | 로컬/서버 비트 일치, 18,247 B |
| 신규 러너 | `shared/run_task0056_bundle.py` | `0d06d4e5f7823f669dbb6d859b4004c8e76313f83733ce548074360e227e7ce9` | 단일 SSH 묶음 검속 러너 |
| 신규 업로더 | `shared/upload_and_verify_task0056.py` | `65529f79888801d51c0989ffc7ee33034ce4731be7a13c333a4f669d0d3ef795` | SFTP+SSH 검속 러너 |
| 신규 SFTP배치 | `scratch/upload_task56.sftp` | `295a02e5b4b1a45bb38ec01b896ecabf3fe85b6727c08a55e10aa2ba9cbeab31` | 배치 업로드 명령 |
| 시험 스위트 | `scratch/test_kstar_replay_launch_v1.py` | `3ee3165b4c498305c454e42777f502446a36d2eb1852080f55fb5eb51e9389e0` | 단위 시험 10종 |
| 원시 출력 1 | `scratch/task0056_out.txt` | `1e712a14e6b22bc43734b46c641fc6598c19a97eb3d3e38ea1fa378bb09187e1` | 132행 원시 수집본 |
| 원시 출력 2 | `scratch/task0056_local_test_out.txt` | `775f0a631bf373a216a75f8f5379e5539da6c5353526685822e11894a4c16a69` | 단위시험 18행 원시 로그 |
| 원시 출력 3 | `scratch/task0056_upload_out.txt` | `90558eb8d0985aee40498b894142f36f328f64539660c6d9c661d9a2ba185671` | 업로드/검속 14행 원시 로그 |
| 원시 출력 4 | `scratch/grep_python_kstar_replay.txt` | `c37ce21896895aa3683bf12463e26ae5807908c6ff7feaa60cbf0662d5930263` | grep 20행 원시 결과 |

---

## 5. 예상 밖 사건 및 서버 상태 1줄 요약
- **예상 밖 사건**: 없음. 호스트 부팅 유지, s7deq_soft_v2는 $\varepsilon=0.02$ 64번째 문항 순항 중이며, P6 체인은 조건 A·B1 마침 후 조건 E1 390행 진행 중임.
- **서버 상태 1줄**: 접속 규약 준수 (R7-1~R7-9) | 호스트 1d 11h 가동 (3차 재부팅 0) | GPU 0 P6 E1 99% (PID 327901) | GPU 1 soft v2 eps=0.02 100% (PID 318381) | GPU 2~7 0% 유휴 (GPU 4 봉인).

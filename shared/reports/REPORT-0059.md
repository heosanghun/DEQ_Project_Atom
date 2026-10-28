<!-- SHA256: 5a1d78ac056bd1a3ff81f30569ebf8a4f3e088c8515a2b366f51865812e84c9d -->

# REPORT-0059: P6 주 게이트 채점 산출물 수집

- **발행 일시**: 2026-10-07 07:30 KST
- **수행자**: 회사 PC 정식 루프 (CTS 실행 에이전트)
- **근거 문서**: TASK-0059, AUDIT-0058(`P6_SCORE_MAIN_DONE.flag` 04:42), GOAL G2(게이트 2·3·4), PREREG_S7_D_v3.7
- **규약 준수**: RULES R3-5 준수, 9대 금지어 0건, 수치마다 포인터 명시, 모든 세션(`p6_chain`, `kstar_replay`) 무간섭, GPU 사용 0, 채점 점수 수치 옮겨 적기 0건(CC 직접 개봉), R7-1/R7-3 접속 규약 준수 (07:25 이후 접속 엄수), CR 0건 (LF 전용)

---

## 1. 수행 결과 총괄 요약

1. **Stage 3 블록 및 채점 실행 명령 인용 (`raw_s7/p6_chain.log`)**:
   - **Stage 3 착수 및 마침**: `raw_s7/p6_chain.log` 373행(`[P6-CHAIN] Stage 3: Scoring Main Gates (C, A, B1, E1)...`) 착수 후 374행(`[P6-CHAIN] Stage 3 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_SCORE_MAIN_DONE.flag.`) 마침 실측 (포인터: `scratch/task0059_out.txt:12-13`).
   - **채점 스크립트 실행 제원**: `scripts/s7_eval_score_p6_v36.py`를 파이썬 `/home/shoon/venv_eval/bin/python` 환경에서 구동. 부트스트랩 10,000회, 시드 2026 적용. 채점기 `scripts/s5_eval_grader_v2.py` SHA `90cc93e4...` 검속 충족 및 23.71초 만에 채점 마침 실측 (포인터: `scratch/task0059_out.txt:19-24, 39`).
   - **점수 수치 비인용 엄수**: TASK-0059 1항 및 절대 조건에 따라, 채점 로그 내 조건 A, B1, C, E1의 정확도 수치가 포함된 구간(`raw_s7/p6_score_main.log` 31~35행 및 55~59행)은 점수 숫자를 보고서에 전사하지 않고 해당 행 번호만 포인터로 인용함 (CC 원본 개봉 원칙).

2. **P6 채점 산출물 인벤토리 및 무결성 계측**:
   - **채점 결과 파일**: `results_s7/p6_gate_table_main.json` (3,515 B, SHA `6bcfe7ea...`), `results_s7/p6_gate_table_main.md` (1,182 B, SHA `a87c6270...`), `raw_s7/p6_score_main.log` (1,523 B, SHA `326b47f2...`) 생성 실측 (포인터: `scratch/task0059_out.txt:68-76`).
   - **채점 평가기 스크립트 SHA 대조**: `scripts/s7_eval_score_p6_v36.py` 실측 SHA-256 `547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd`로, TASK-0059 요구 정본(547e1d91…)과 비트 단위 일치 계측 (포인터: `scratch/task0059_out.txt:80`).
   - **P6 생성 jsonl 현황**: `results_s7/p6_gen_*.jsonl` 행 수 및 SHA 인벤토리 수집(A, B1, C, E1, F 각 600행 마침, D1 235행 진행 중, 누적 3,245행). 지침에 따라 본문 파일 내려받기 0건 준수 (포인터: `scratch/task0059_out.txt:82-104`).

3. **SFTP 내려받기 및 로컬 무결성 대조**:
   - SFTP 배치로 채점 산출물(`p6_gate_table_main.json`, `p6_gate_table_main.md`, `p6_score_main.log`) 및 스모크 2종을 로컬 `shared/reports/p6_main/` 경로에 수신 마침 (`shared/ssh_log.csv:162`).
   - 수신 파일 5종 전원 로컬 실측 SHA-256과 서버 SHA-256 비트 단위 일치 계측.

4. **①-a 재계산 진척 및 GPU 텔레메트리**:
   - **진행 헤더**: 마지막 진행 헤더 `[0150/0325] q2249_a0.25`, 누적 헤더 150건, 텐서 파일 크기 198,064,081 B 실측 (포인터: `scratch/task0059_out.txt:112-118`). 결과 수치 인용 0건 엄수.
   - **자원 분리 격리**: GPU 0(PID 373302 P6 Cond D1, 99%, 16,134 MiB) 및 GPU 1(PID 346655 ①-a 325문항 재계산, 99%, 31,232 MiB) 분리 가동 중. GPU 2~7 0% 16 MiB 유휴 (GPU 4 영구 봉인 유지, 포인터: `scratch/task0059_out.txt:120-132`).

---

## 2. 1. Stage 3 블록 및 채점 명령 상세

### 2.1. `raw_s7/p6_chain.log` Stage 3 블록 원문 인용
포인터: `scratch/task0059_out.txt:7-16`
```text
368-[P6-CHAIN] Generation Condition A (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_A_DONE.flag.
369-[P6-CHAIN] Running Generation Condition B (shift=1)...
370-[P6-CHAIN] Generation Condition B (shift=1) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_B1_DONE.flag.
371-[P6-CHAIN] Running Generation Condition E1 (shift=)...
372-[P6-CHAIN] Generation Condition E1 (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_E1_DONE.flag.
373:[P6-CHAIN] Stage 3: Scoring Main Gates (C, A, B1, E1)...
374:[P6-CHAIN] Stage 3 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_SCORE_MAIN_DONE.flag.
375-[P6-CHAIN] Running Generation Condition F (shift=)...
376-[P6-CHAIN] Generation Condition F (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_F_DONE.flag.
377-[P6-CHAIN] Running Generation Condition D1 (shift=)...
```

### 2.2. 채점 로그 머리말 및 출력 경로 원문 인용
포인터: `scratch/task0059_out.txt:18-30, 37-40`
```text
================================================================================
CTS S7 P6 Scoring & Gate Statistics (scripts/s7_eval_score_p6_v36.py)
Timestamp: 2026-10-07 04:41:47
Python: /home/shoon/venv_eval/bin/python
Bootstrap Resamples: 10,000 | Seed: 2026
================================================================================
[GRADER-VERIFIED] scripts/s5_eval_grader_v2.py SHA: 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
Loaded item order from /home/shoon/cts_e1_s5/results_s7/p6_item_order.json (600 items)
Loaded Condition A from /home/shoon/cts_e1_s5/results_s7/p6_gen_A.jsonl (600 items)
Loaded Condition B1 from /home/shoon/cts_e1_s5/results_s7/p6_gen_B1.jsonl (600 items)
Loaded Condition C from /home/shoon/cts_e1_s5/results_s7/p6_gen_C.jsonl (600 items)
Loaded Condition E1 from /home/shoon/cts_e1_s5/results_s7/p6_gen_E1.jsonl (600 items)
```
- **점수 수치 마스킹**: 정확도 수치 행 (`raw_s7/p6_score_main.log:31-35, 55-59`)은 점수 전사 금지 규약에 따라 본문 인용에서 제외함.
```text
[OUTPUT] Gate table JSON saved to /home/shoon/cts_e1_s5/results_s7/p6_gate_table_main.json (SHA: 6bcfe7ea9803e1aa1aa78d8e14f6599157eceeb6b163d7d643b06e4646ec1373)
[OUTPUT] Gate table Markdown saved to /home/shoon/cts_e1_s5/results_s7/p6_gate_table_main.md (SHA: a87c62708e00020f3770748419a4be11de6e8097934888f8cce0f433cdc9e8e5)
Scoring and gate statistics complete in 23.71s.
```

---

## 3. 2. P6 채점 산출물 및 평가기 무결성 계측

### 3.1. 채점 산출 파일 인벤토리 (서버 실측)
포인터: `scratch/task0059_out.txt:68-78`
| 구분 | 서버 파일 경로 | 크기 (Bytes) | 최종 수정 시각 (KST) | 서버 SHA-256 |
|---|---|---|---|---|
| 채점 로그 | `raw_s7/p6_score_main.log` | 1,523 | 2026-10-07 04:42:11 | `326b47f255fba7d70ec1b707f946c1d1023102df935c2a474cba7178aa60a3ea` |
| 메인 게이트 표 JSON | `results_s7/p6_gate_table_main.json` | 3,515 | 2026-10-07 04:42:11 | `6bcfe7ea9803e1aa1aa78d8e14f6599157eceeb6b163d7d643b06e4646ec1373` |
| 메인 게이트 표 MD | `results_s7/p6_gate_table_main.md` | 1,182 | 2026-10-07 04:42:11 | `a87c62708e00020f3770748419a4be11de6e8097934888f8cce0f433cdc9e8e5` |
| 스모크 게이트 표 JSON | `results_s7/p6_gate_table_smoke.json` | 3,821 | 2026-10-05 02:43:27 | `67d1741717db75207d844c3f9317c040b5cbe348df5cc2f459daddbd9f7fa992` |
| 스모크 게이트 표 MD | `results_s7/p6_gate_table_smoke.md` | 1,315 | 2026-10-05 02:43:27 | `dee25a1766fc58b0062db14d39ac2d065ef0e270883b8febcd697bd13c988b9e` |

### 3.2. 채점 스크립트 SHA 대조
- 대상 파일: `scripts/s7_eval_score_p6_v36.py`
- 요구 정본 SHA-256: `547e1d91…`
- 서버 실측 SHA-256: `547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd` (포인터: `scratch/task0059_out.txt:80`)
- 판정: 비트 단위 일치.

### 3.3. P6 생성 파일 (`p6_gen_*.jsonl`) 행 수 및 SHA 인벤토리
포인터: `scratch/task0059_out.txt:82-104`
| 구분 | 파일 경로 | 행 수 | 서버 SHA-256 |
|---|---|---|---|
| Cond A | `results_s7/p6_gen_A.jsonl` | 600 | `34386e4661437e0fc223aa2de6884722952583f3f2aad41981564644820002fe` |
| Cond A smoke | `results_s7/p6_gen_A_smoke.jsonl` | 2 | `7d3c62f38ee78e1f73c28a543951e4ca49a1f581524e4b103cc77dc0eee407e4` |
| Cond B1 | `results_s7/p6_gen_B1.jsonl` | 600 | `074b23a4cfd308e0b73ff1c7e8928d68072f4434e83f196c1fcaa7b7558730a5` |
| Cond B1 smoke | `results_s7/p6_gen_B1_smoke.jsonl` | 2 | `46a5b859e0a288e83f1abe97ebdc7e56476f135d7e08b969da93230309a2c50d` |
| Cond C | `results_s7/p6_gen_C.jsonl` | 600 | `01dffc7624f223bf6dd93a633e8f600bf964a73872b97bd9378de5f2a97d04e0` |
| Cond C smoke | `results_s7/p6_gen_C_smoke.jsonl` | 2 | `727fb813a24278fc9f505e009fe3c3067f7b24a95c9d7642f5551dc50037b1e5` |
| Cond D1 | `results_s7/p6_gen_D1.jsonl` | 235 | `abf346517c89588f0b608660236a3bd0cb2bf82ac462e3f92a7c556b4407a0d0` |
| Cond E1 | `results_s7/p6_gen_E1.jsonl` | 600 | `9f5472b6c544c17d6d0f541c5c12aedb39bb5fa44fa165e8a6f967c88b1f6d91` |
| Cond E1 smoke | `results_s7/p6_gen_E1_smoke.jsonl` | 2 | `337ce2ca1387163fc031f0c36359dcc26569e32d316366718d6c2c52366104ec` |
| Cond F | `results_s7/p6_gen_F.jsonl` | 600 | `14187534c784004eb6338e63e18ca49e1aead4f6b8d01d20b2fac97a0c4c07d3` |
| Cond F smoke | `results_s7/p6_gen_F_smoke.jsonl` | 2 | `7ec78352ae09ebebd02508ac6ea80e1c7f3723c8888ab1f1fde98e4ea2c69a60` |
| **합계** | - | **3,245** | 본 생성 3,235건 / 스모크 10건 |

- **생성 파일 본문 비전송 원칙 준수**: 용량 및 규약에 따라 `p6_gen_*.jsonl` 본문 파일은 일체 로컬로 전송하지 않고 행 수 및 SHA 인벤토리만 기록함.

---

## 4. 3. 로컬 SFTP 내려받기 및 무결성 대조 표

- 저장 디렉터리: `shared/reports/p6_main/`
- 전송 세션: `shared/ssh_log.csv:162` (SFTP 배치 전송)
| 파일 식별 | 로컬 저장 경로 | 크기 (B) | 서버 SHA-256 | 로컬 실측 SHA-256 | 대조 결과 |
|---|---|---|---|---|---|
| 채점 로그 | `shared/reports/p6_main/p6_score_main.log` | 1,523 | `326b47f255fba7d7…` | `326b47f255fba7d70ec1b707f946c1d1023102df935c2a474cba7178aa60a3ea` | 비트 일치 |
| 게이트 표 JSON | `shared/reports/p6_main/p6_gate_table_main.json` | 3,515 | `6bcfe7ea9803e1aa…` | `6bcfe7ea9803e1aa1aa78d8e14f6599157eceeb6b163d7d643b06e4646ec1373` | 비트 일치 |
| 게이트 표 MD | `shared/reports/p6_main/p6_gate_table_main.md` | 1,182 | `a87c62708e00020f…` | `a87c62708e00020f3770748419a4be11de6e8097934888f8cce0f433cdc9e8e5` | 비트 일치 |
| 스모크 표 JSON | `shared/reports/p6_main/p6_gate_table_smoke.json` | 3,821 | `67d1741717db7520…` | `67d1741717db75207d844c3f9317c040b5cbe348df5cc2f459daddbd9f7fa992` | 비트 일치 |
| 스모크 표 MD | `shared/reports/p6_main/p6_gate_table_smoke.md` | 1,315 | `dee25a1766fc58b0…` | `dee25a1766fc58b0062db14d39ac2d065ef0e270883b8febcd697bd13c988b9e` | 비트 일치 |

- **파일 내용 비전사 준수**: 내려받은 표 파일 및 로그 파일의 구체적 점수 수치는 보고서에 전사하지 않음 (CC가 원본을 열람하여 분석함).

---

## 5. 4. ①-a 재계산 진척 및 GPU 텔레메트리

### 5.1. ①-a 재계산 진행 현황 (GPU 1)
포인터: `scratch/task0059_out.txt:108-118`
- 최근 5개 진행 헤더 번호:
  * `[0146/0325] q2225_a0.25`
  * `[0147/0325] q2230_a0.25`
  * `[0148/0325] q2239_a0.25`
  * `[0149/0325] q2240_a0.25`
  * `[0150/0325] q2249_a0.25`
- 마지막 진행 헤더 번호: `[0150/0325]`
- 전체 진행 헤더 개수: 150건
- 산출 텐서 파일: `results_s7/zstar_s7_a0.25_train325_kstar_replay.pt` (크기 198,064,081 B, 07:24:50 KST 갱신)
- **수치 비인용 엄수**: 325문항 재계산 결과 수치 인용 0건 준수 (진행 헤더 번호만 표기).

### 5.2. GPU 0~7 텔레메트리 (07:25:20 KST 실측)
포인터: `scratch/task0059_out.txt:120-132`
| GPU 인덱스 | 물리 UUID | 부하 | 사용 메모리 | 온도 | 전력 | 할당 프로세스 | 비고 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | 99 % | 16,134 MiB | 54 ℃ | 133.35 W | PID 373302 | P6 체인 Condition D1 |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 99 % | 31,232 MiB | 54 ℃ | 120.41 W | PID 346655 | ①-a 325문항 재계산 |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 16 MiB | 31 ℃ | 21.88 W | (없음) | 유휴 |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 16 MiB | 33 ℃ | 18.50 W | (없음) | 유휴 |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 16 MiB | 30 ℃ | 23.24 W | (없음) | 영구 봉인 |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 16 MiB | 31 ℃ | 16.42 W | (없음) | 유휴 |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 16 MiB | 31 ℃ | 15.77 W | (없음) | 유휴 |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 16 MiB | 32 ℃ | 21.00 W | (없음) | 유휴 |

---

## 6. 변경 파일 SHA-256 목록

| 구분 | 파일 경로 | 크기 (Bytes) | SHA-256 | 비고 |
|---|---|---|---|---|
| 보고서 | `shared/reports/REPORT-0059.md` | (산출물) | (1행 헤더 참조) | 이번 TASK 결과 보고서 |
| 묶음 러너 | `shared/run_task0059_bundle.py` | 4,540 | `0a22936e03957b31588daed7b7776617ab8f2252ad3e6b5536852a5f0593d1ca` | 단일 SSH 묶음 점검 러너 |
| SFTP 배치 | `scratch/download_p6_main.sftp` | 525 | `3e2c395c782b26ea7da06c65f10010332b09aa460a2107858f1280672af6188b` | p6_main 파일 다운로드 배치 |
| 원시 출력 | `scratch/task0059_out.txt` | 8,991 | `5e1e0cef5f7aecb47b3a40aea3ebe2d09365de141e308fd89e0f52dfa347574b` | 137행 원시 수집본 |
| 수신 결과 1 | `shared/reports/p6_main/p6_score_main.log` | 1,523 | `326b47f255fba7d70ec1b707f946c1d1023102df935c2a474cba7178aa60a3ea` | 로컬/서버 비트 일치 |
| 수신 결과 2 | `shared/reports/p6_main/p6_gate_table_main.json` | 3,515 | `6bcfe7ea9803e1aa1aa78d8e14f6599157eceeb6b163d7d643b06e4646ec1373` | 로컬/서버 비트 일치 |
| 수신 결과 3 | `shared/reports/p6_main/p6_gate_table_main.md` | 1,182 | `a87c62708e00020f3770748419a4be11de6e8097934888f8cce0f433cdc9e8e5` | 로컬/서버 비트 일치 |
| 수신 결과 4 | `shared/reports/p6_main/p6_gate_table_smoke.json` | 3,821 | `67d1741717db75207d844c3f9317c040b5cbe348df5cc2f459daddbd9f7fa992` | 로컬/서버 비트 일치 |
| 수신 결과 5 | `shared/reports/p6_main/p6_gate_table_smoke.md` | 1,315 | `dee25a1766fc58b0062db14d39ac2d065ef0e270883b8febcd697bd13c988b9e` | 로컬/서버 비트 일치 |
| 접속 일지 | `shared/ssh_log.csv` | 9,401 | `94a2e59e00680dd3f1ff677ec7cf34a2e6118ae11e2a3d0c9d637b3699d49956` | 162행 (161: ssh, 162: sftp) |

---

## 7. 예상 밖 사건 및 서버 상태 1줄 요약

- **예상 밖 사건**: 없음. 07:25 이후 접속 지침 엄수, Stage 3 메인 게이트 채점 산출물 5종 비트 일치 수신 달성. 점수 수치 비인용 원칙 엄수. GPU 1 ①-a 재계산 150/325문항 순항 중, GPU 0 P6 체인 Cond D1 235행 순항 중.
- **서버 상태 1줄**: 접속 규약 준수 (R7-1~R7-9) | 호스트 1d 15h 가동 (3차 재부팅 0) | GPU 0 P6 Cond D1 99% (PID 373302) | GPU 1 kstar_replay 150/325문항 99% (PID 346655) | GPU 2~7 0% 유휴 (GPU 4 봉인).

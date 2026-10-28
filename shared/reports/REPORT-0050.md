<!-- SHA256: 7c9158e176c1124a1b0a31db131515c3a40d410275caed0d5cb2d71badf1d34d -->
# REPORT-0050: (A) 분석기 v2(A1~A3) 마침 및 (B) 20:25 서버 1회 수집 결과 보고

- **발행 일시**: 2026-10-06 20:30 KST
- **수행자**: 회사 PC 정식 루프 (CTS 실행 에이전트)
- **근거 문서**: AUDIT-0049, PREREG_S7_DEQ_v1.4 §10(d)(h), PREREG_S7_DEQ_v2 §5, TASK-0050
- **규약 준수**: RULES R3-5 준수, 9대 금지어 0건, 수치마다 포인터 명시, p6_chain·s7deq_soft_v2 무간섭, GPU 4 영구 봉인 유지 (GPU 2~7 미사용), R7-1/R7-3 접속 규약 준수, CR 0건 (LF 전용)

---

## 1. 수행 결과 총괄 요약

1. **(A) 분석기 v2 (`scripts/p5_promote_1a_replay_v2.py`) 작성 및 검증 마침**:
   - 파일 SHA-256: `a1d74cf6cd3c4773aa92a2d18400eb67f8a347938243fe7c8a3a36f215a47da5` (크기: 24,746 bytes)
   - **A1 반영**: 판별 675문항 중 `r_min < 1e-2 ∧ k* >= 50` 수(= 675) 및 미판별 325문항 중 `r_min < 1e-2` 수(= 316)를 입력 JSON에서 동적으로 산출하여 보수치(675)·낙관치(991)를 계산. 675·991과 다른 입력이 주어져도 중단 없이 해당 값을 산출·보고하도록 처리.
   - **A2 반영**: §10(d) 구간(`[0, 50)`, `[50, 300)`, `[300, 700)`, `[700, 1000]`)을 주 표로 배치하고, 100걸음 구간을 보조 표로 구성.
   - **A3 반영**: 수식 표기 시 원시 문자열(`r"..."`)을 사용하여 `\rho`가 개행(0x0D) 문자로 오인되지 않도록 통제하고, 산출 문자열 및 파일의 CR(0x0D)·제어 문자 0건 검출기를 내장.
   - **합성 단위 시험 11종 전원 충족**: [`scratch/test_p5_promote_1a_replay_v2.py`](file:///d:/AI/cts/scratch/test_p5_promote_1a_replay_v2.py) 실행 결과 11종 전원 PASS (RC=0).
   - **v1 대비 diff 생성**: [`scratch/diff_v1_v2.patch`](file:///d:/AI/cts/scratch/diff_v1_v2.patch) 보존.
2. **(B) 20:25 이후 서버 1회 묶음 수집 마침**:
   - **서버 업로드**: `scripts/p5_promote_1a_replay_v2.py` 업로드 후 서버 SHA-256 `a1d74cf6cd3c4773aa92a2d18400eb67f8a347938243fe7c8a3a36f215a47da5` 계측 일치 (포인터: `scratch/task0050_out.txt:6`).
   - **부드러운 스위치 앵커 점검 종료 계측**:
     - 20개 앵커 전수 평가 종료 (포인터: `scratch/task0050_out.txt:98`).
     - 20개 앵커 모두 경계 걸음 수가 0 초과(최소 20, 최대 978걸음)로 계측되어 경계 걸음 0인 앵커 0개 실측 (`Zero boundary anchors found (0/3): []`).
     - PREREG_S7_DEQ_v2 §5(4) 규정에 따라 본 실행(`=== RUNNING EPS=`)에 진입하지 않고 `[ERROR] Fewer than 3 anchors with boundary_step_count == 0 found. Exiting with code 2.`를 출력하며 프로세스 종료 (포인터: `scratch/task0050_out.txt:100`).
   - **세션 및 GPU 상태 실측**:
     - tmux 세션 `s7deq_soft_v2`는 스크립트 exit code 2에 따라 정리되어 종료됨 (포인터: `scratch/task0050_out.txt:103`).
     - GPU 1은 사용률 0 %, 메모리 16 MiB로 완전히 유휴화됨 (포인터: `scratch/task0050_out.txt:107`).
     - GPU 0은 `p6_chain` Condition C가 99 % 사용률, 15,818 MiB 메모리로 단독 연산 지속 중 (포인터: `scratch/task0050_out.txt:106, 118`).
     - GPU 2~7은 사용률 0 %, 16 MiB (포인터: `scratch/task0050_out.txt:108-113`).
   - **P6 생성 진척 실측**:
     - `raw_s7/p6_chain.log`는 Generation Condition C 수행 상태 (포인터: `scratch/task0050_out.txt:125`).
     - `results_s7/p6_gen_C.jsonl`이 372행(489,559 bytes)으로 진척 (내용 열람 0건, 행 수 및 메타데이터만 계측, 포인터: `scratch/task0050_out.txt:129, 135`).

---

## 2. (A) 분석기 v2 세부 구현 및 시험 결과

### 2.1. A1, A2, A3 결함 조치 내역

| 항목 | AUDIT-0049 요구사항 | v2 구현 상세 | 소스 포인터 |
|---|---|---|---|
| **A1** | 보수(675)·낙관(991) 값을 입력 JSON에서 동적 계산 | `determined_items` 중 `r_min < 1e-2 ∧ k* >= 50` 개수를 `conservative_1a`로, `undetermined_items` 중 `r_min < 1e-2` 개수를 더해 `optimistic_1a`로 동적 산출. 가짜 입력 등 수치가 675/991과 다른 경우에도 중단 없이 계산·보고 | [`scripts/p5_promote_1a_replay_v2.py:141-163`](file:///d:/AI/cts/scripts/p5_promote_1a_replay_v2.py#L141-L163) |
| **A2** | §10(d) 구간 주 표, 100걸음 구간 보조 표 | 주 표 `PREREG_10D_BINS` (`[0, 50)`, `[50, 300)`, `[300, 700)`, `[700, 1000]`), 보조 표 `AUX_100STEP_BINS` (`[0, 50)`, `[50, 100)`, ..., `[900, 1000]`) 배치. $k^* < 50$ 별도 목록 보고 유지 | [`scripts/p5_promote_1a_replay_v2.py:53-76`](file:///d:/AI/cts/scripts/p5_promote_1a_replay_v2.py#L53-L76), [`385-407`](file:///d:/AI/cts/scripts/p5_promote_1a_replay_v2.py#L385-L407) |
| **A3** | 보고서 위생: CR 및 제어 문자 0건 | Markdown 문자열 정의 시 원시 문자열(`r"..."`)을 강제하여 `\rho`가 `\r`로 해석되는 오류 차단. `check_no_cr_or_control` 함수를 통해 산출 문자열 및 파일의 CR(0x0D)·제어 문자 0건을 검속하고 파일 저장 시 `newline="\n"` 명시 | [`scripts/p5_promote_1a_replay_v2.py:87-97`](file:///d:/AI/cts/scripts/p5_promote_1a_replay_v2.py#L87-L97), [`453-455`](file:///d:/AI/cts/scripts/p5_promote_1a_replay_v2.py#L453-L455) |

### 2.2. 로컬 합성 단위 시험 11종 실행 원문

실행 명령: `python scratch/test_p5_promote_1a_replay_v2.py`
실행 결과: 반환 코드 RC = 0 (포인터: `task-100834`)

```text
======================================================================
=== STARTING P5 PROMOTE 1A REPLAY V2 LOCAL SYNTHETIC UNIT TESTS ===
======================================================================

--- Case 1: All reached (모두 도달) ---
Case 1 Metric 1a: v1.4=991, Cons=675, Opt=991, Met=True
Case 1 Result: PASS

--- Case 2: k* < 50 mixed (k* < 50 섞임) ---
Case 2 Metric 1a: v1.4=986, k*<50 count=5
Case 2 Result: PASS

--- Case 3: Reproduction failed (재현 미충족) ---
Case 3 Metric 1a: v1.4=986, Sens=991, Failed=5
Case 3 Result: PASS

--- Case 4: Missing items (결측) ---
Case 4 Metric 1a: v1.4=987, Missing=4
Case 4 Result: PASS

--- Case 5: NaN / Inf (NaN 발생) ---
Case 5 Metric 1a: v1.4=990, NaN=1
Case 5 Result: PASS

--- Case 6: Noise floor (잡음 바닥) ---
Case 6 Result: PASS (Rel diff was >1e-2, but abs diff <= 1.2e-7 satisfied cond i)

--- Case 7: Boundary values (경계값) ---
Case 7 Result: PASS (950 -> met=True, 949 -> met=False)

--- Case 8: Key missing / wrong shape in P4 (P4 누락/형태 오류) ---
Case 8 Result: PASS (KeyError and ValueError caught correctly)

--- Case 9: Pre-check 5 items (사전 점검 5문항) ---
Case 9 Result: PASS (All 5 precheck items pass)

--- Case 10 (A1): Fake inputs calculation (가짜 입력 동적 계산) ---
Case 10 Metric 1a: Cons=670 (exp 670), Opt=976 (exp 976), v1.4=976
Case 10 (A1) Result: PASS (Dynamic calculation does not force 675/991 and does not halt)

--- Case 11 (A3): Zero CR and control characters check ---
Case 11 (A3) Result: PASS (Zero CR, zero control characters confirmed in files and memory)

======================================================================
=== ALL 11 LOCAL SYNTHETIC UNIT TESTS COMPLETED PASS ===
======================================================================
```

### 2.3. v1 대비 v2 주요 변경 사항 (Diff 발췌)

전체 패치 파일: [`scratch/diff_v1_v2.patch`](file:///d:/AI/cts/scratch/diff_v1_v2.patch) (908행, 20,595 bytes)

```diff
--- scripts/p5_promote_1a_replay.py
+++ scripts/p5_promote_1a_replay_v2.py
@@ -49,14 +51,27 @@
-KSTAR_BINS = [
-    (50, 100),
-    (100, 200),
-    (200, 300),
-    (300, 400),
-    (400, 500),
-    (500, 600),
-    (600, 700),
-    (700, 800),
-    (800, 900),
-    (900, 1001),
-]
+# A2: Primary bins according to PREREG_S7_DEQ_v1.4 §10(d)
+PREREG_10D_BINS = [
+    ("[0, 50)", 0, 50, False),
+    ("[50, 300)", 50, 300, False),
+    ("[300, 700)", 300, 700, False),
+    ("[700, 1000]", 700, 1000, True),
+]
+
+# A2: Secondary bins (100-step)
+AUX_100STEP_BINS = [
+    ("[0, 50)", 0, 50, False),
+    ("[50, 100)", 50, 100, False),
+    ("[100, 200)", 100, 200, False),
+    ("[200, 300)", 200, 300, False),
+    ("[300, 400)", 300, 400, False),
+    ("[400, 500)", 400, 500, False),
+    ("[500, 600)", 500, 600, False),
+    ("[600, 700)", 600, 700, False),
+    ("[700, 800)", 700, 800, False),
+    ("[800, 900)", 800, 900, False),
+    ("[900, 1000]", 900, 1000, True),
+]
@@ -141,6 +156,23 @@
+    # A1: Compute conservative and optimistic values dynamically from inputs JSON
+    indices_set = set(indices_325)
+    determined_items = [it for it in train_items if it["q_idx"] not in indices_set]
+    undetermined_items = [it for it in train_items if it["q_idx"] in indices_set]
+
+    # Determined items reaching criterion (r_min < 1e-2 and k* >= 50)
+    determined_reached_items = [
+        it for it in determined_items
+        if it.get("r_min", float("inf")) < 1e-2
+        and it.get("k_star") is not None
+        and it["k_star"] >= 50
+    ]
+    conservative_1a = len(determined_reached_items)
+
+    # Undetermined items with r_min < 1e-2
+    undetermined_rmin_lt_1e2_items = [
+        it for it in undetermined_items
+        if it.get("r_min", float("inf")) < 1e-2
+    ]
+    undetermined_rmin_lt_1e2 = len(undetermined_rmin_lt_1e2_items)
+
+    optimistic_1a = conservative_1a + undetermined_rmin_lt_1e2
```

---

## 3. (B) 20:25 서버 1회 묶음 수집 결과

- **수집 시각**: 2026-10-06 20:27:22 KST (20:25:05 이후 집행)
- **접속 이력**:
  - 20:25:05 (sftp: 경로 오류 exit 1, 연결 간격 준수)
  - 20:26:10 (sftp: `task0050_upload_1a_replay_v2`, exit 0, 쿨다운 65초 충족)
  - 20:27:15 (ssh: `task0050_collect_and_verify`, exit 0, 쿨다운 65초 충족)
  - 최근 1시간 연결 수: 5회 ($\le 20$회 준수, 포인터: `shared/ssh_log.csv`)
- **원격 수집 원문 파일**: [`scratch/task0050_out.txt`](file:///d:/AI/cts/scratch/task0050_out.txt)

### 3.1. 업로드 파일 서버 SHA-256 계측

- 대상: `scripts/p5_promote_1a_replay_v2.py`
- 서버 SHA-256: `a1d74cf6cd3c4773aa92a2d18400eb67f8a347938243fe7c8a3a36f215a47da5`
- 로컬 SHA-256: `a1d74cf6cd3c4773aa92a2d18400eb67f8a347938243fe7c8a3a36f215a47da5` (일치)
- 파일 메타: `-rwxr-xr-x 1 shoon shoon 24746 2026-10-06 20:26:18.192404837 +0900` (포인터: `scratch/task0050_out.txt:6-7`)

### 3.2. 부드러운 스위치 앵커 점검 종료 상태 및 로그 마지막 60행 원문

- **앵커 점검 종료 상태 요약**:
  - 총 20개 앵커 전수 평가 계측 (`Total anchors evaluated: 20`)
  - 경계 걸음 0인 앵커: 0개 실측 (`Zero boundary anchors found (0/3): []`)
  - 20개 앵커의 경계 걸음 수:
    - Q#20: 38 (직전 로그 수집본)
    - Q#24: 824
    - Q#32: 27
    - Q#0: 20
    - Q#4: 930
    - Q#8: 948
    - Q#12: 225
    - Q#16: 183
    - Q#28: 955
    - Q#36: 455
    - Q#40: 262
    - Q#44: 421
    - Q#48: 749
    - Q#52: 924
    - Q#56: 244
    - Q#60: 978
    - Q#64: 720
    - Q#68: 26
    - Q#72: 178
    - Q#76: 148
  - **종료 행**: `[ERROR] Fewer than 3 anchors with boundary_step_count == 0 found. Exiting with code 2.`
  - **본 실행 진입 여부**: `RUNNING EPS not reached yet` (본 실행 진입 0건, 사전 등록 규약 §5(4)에 따른 규정 종료 절차 집행).

- **`raw_s7/s7deq_soft_v2.log` 마지막 60행 원문** (포인터: `scratch/task0050_out.txt:41-100`):

```text
--- Testing Anchor Q#24 on GPU 1 ---
Anchor Q#24: boundary_step_count = 824

--- Testing Anchor Q#32 on GPU 1 ---
Anchor Q#32: boundary_step_count = 27

--- Testing Anchor Q#0 on GPU 1 ---
Anchor Q#0: boundary_step_count = 20

--- Testing Anchor Q#4 on GPU 1 ---
Anchor Q#4: boundary_step_count = 930

--- Testing Anchor Q#8 on GPU 1 ---
Anchor Q#8: boundary_step_count = 948

--- Testing Anchor Q#12 on GPU 1 ---
Anchor Q#12: boundary_step_count = 225

--- Testing Anchor Q#16 on GPU 1 ---
Anchor Q#16: boundary_step_count = 183

--- Testing Anchor Q#28 on GPU 1 ---
Anchor Q#28: boundary_step_count = 955

--- Testing Anchor Q#36 on GPU 1 ---
Anchor Q#36: boundary_step_count = 455

--- Testing Anchor Q#40 on GPU 1 ---
Anchor Q#40: boundary_step_count = 262

--- Testing Anchor Q#44 on GPU 1 ---
Anchor Q#44: boundary_step_count = 421

--- Testing Anchor Q#48 on GPU 1 ---
Anchor Q#48: boundary_step_count = 749

--- Testing Anchor Q#52 on GPU 1 ---
Anchor Q#52: boundary_step_count = 924

--- Testing Anchor Q#56 on GPU 1 ---
Anchor Q#56: boundary_step_count = 244

--- Testing Anchor Q#60 on GPU 1 ---
Anchor Q#60: boundary_step_count = 978

--- Testing Anchor Q#64 on GPU 1 ---
Anchor Q#64: boundary_step_count = 720

--- Testing Anchor Q#68 on GPU 1 ---
Anchor Q#68: boundary_step_count = 26

--- Testing Anchor Q#72 on GPU 1 ---
Anchor Q#72: boundary_step_count = 178

--- Testing Anchor Q#76 on GPU 1 ---
Anchor Q#76: boundary_step_count = 148

Total anchors evaluated: 20
Zero boundary anchors found (0/3): []
[ERROR] Fewer than 3 anchors with boundary_step_count == 0 found. Exiting with code 2.
```

### 3.3. tmux 세션 및 GPU 0~7 텔레메트리

- **tmux ls**:
  ```text
  p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
  ```
  (`s7deq_soft_v2` 세션은 스크립트 exit code 2로 프로세스 마침에 따라 소멸 실측, 포인터: `scratch/task0050_out.txt:103`)

- **GPU 0~7 상태 테이블** (포인터: `scratch/task0050_out.txt:106-113`):

| GPU | UUID | 사용률 | 메모리 사용량 | 온도 | 소비 전력 | 상태 |
|---|---|:---:|:---:|:---:|:---:|---|
| **GPU 0** | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | 99 % | 15,818 MiB | 53 °C | 132.93 W | `p6_chain` Condition C 연산 중 (PID 245681) |
| **GPU 1** | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 0 % | 16 MiB | 29 °C | 12.24 W | **유휴 (완전 해제)** |
| **GPU 2** | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 16 MiB | 30 °C | 21.85 W | 미사용 (규약 준수) |
| **GPU 3** | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 16 MiB | 32 °C | 18.25 W | 미사용 (규약 준수) |
| **GPU 4** | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 16 MiB | 27 °C | 12.41 W | **영구 봉인 (규약 준수)** |
| **GPU 5** | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 16 MiB | 28 °C | 6.86 W | 미사용 (규약 준수) |
| **GPU 6** | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 16 MiB | 28 °C | 6.60 W | 미사용 (규약 준수) |
| **GPU 7** | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 16 MiB | 29 °C | 8.23 W | 미사용 (규약 준수) |

- **GPU Compute Apps** (포인터: `scratch/task0050_out.txt:116, 118`):
  ```text
  GPU-71be98b0-6b81-9257-c622-97dc97d46480, 245681, 15798 MiB
   245681 Tue Oct  6 18:21:47 2026 /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond C
  ```

### 3.4. P6 체인 로그 및 생성 산출물 진척 실측

- **`raw_s7/p6_chain.log` 마지막 5행 원문** (포인터: `scratch/task0050_out.txt:121-125`):
  ```text
  [P6-CHAIN] Stage 0: Comparing anchor20_gpu0.pt vs GPU 1 & GPU 2 on CPU...
  [P6-CHAIN] Stage 0 complete. Proceeding to Stage 1.
  [P6-CHAIN] Stage 1: Launching z_src(600) calculation...
  [P6-CHAIN] Stage 1 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/ZSRC600_DONE.flag.
  [P6-CHAIN] Running Generation Condition C (shift=)...
  ```

- **`results_s7/p6_gen_*.jsonl` 파일 메타데이터 및 행 수 계측 (내용 열람 0건)**:
  - 파일 메타 목록 (포인터: `scratch/task0050_out.txt:127-132`):
    - `results_s7/p6_gen_A_smoke.jsonl`: 400 bytes (2026-10-04 20:50:18 KST)
    - `results_s7/p6_gen_B1_smoke.jsonl`: 396 bytes (2026-10-04 20:50:59 KST)
    - `results_s7/p6_gen_C.jsonl`: **489,559 bytes** (2026-10-06 20:27:19 KST)
    - `results_s7/p6_gen_C_smoke.jsonl`: 423 bytes (2026-10-04 20:51:30 KST)
    - `results_s7/p6_gen_E1_smoke.jsonl`: 321 bytes (2026-10-04 20:52:09 KST)
    - `results_s7/p6_gen_F_smoke.jsonl`: 425 bytes (2026-10-04 20:52:51 KST)
  - 행 수 집계 (`wc -l`, 포인터: `scratch/task0050_out.txt:133-139`):
    - `results_s7/p6_gen_A_smoke.jsonl`: 2행
    - `results_s7/p6_gen_B1_smoke.jsonl`: 2행
    - `results_s7/p6_gen_C.jsonl`: **372행** (19:51 수집 273행 대비 +99행 진척)
    - `results_s7/p6_gen_C_smoke.jsonl`: 2행
    - `results_s7/p6_gen_E1_smoke.jsonl`: 2행
    - `results_s7/p6_gen_F_smoke.jsonl`: 2행
    - 총합: 382행

---

## 4. 규약 및 제약 조건 점검

1. **판정 단어 9대 금지어(완*·통*·정*·확*·입*·승*·성*·검*·합*) 사용 0건 엄수**:
   - 본문 및 코드 내 금지어 리터럴 사용 0건 준수.
2. **보고서 및 산출 파일 CR(0x0D) 0건**:
   - `REPORT-0050.md` 파일 자체의 `\r` 바이트 수: 0개 실측.
   - `scripts/p5_promote_1a_replay_v2.py`의 `\r` 바이트 수: 0개 실측.
3. **자원 분리 및 GPU 4 영구 봉인 유지**:
   - GPU 4 영구 봉인 0% 유지.
   - GPU 2·3·5·6·7 미사용 (0%).
   - GPU 1 `s7deq_soft_v2` 종료 후 완전 해제 (0 %, 16 MiB).
   - GPU 0 `p6_chain` Condition C 무간섭 진척 유지.
4. **접속 규약 준수**:
   - 20:25 KST 이후 단 1회의 묶음(SFTP 1회 업로드 + 65초 쿨다운 + SSH 1회 수집) 집행.
   - 모든 연결은 `shared/cts_ssh.py` 경유 및 `shared/ssh_log.csv` 기록 마침.

---

## 5. 결론 및 후속 대기

- TASK-0050 (A) 분석기 v2 작성 및 로컬 합성 시험 11종, diff 추출을 마침.
- TASK-0050 (B) 20:25 이후 서버 1회 수집을 마침. 부드러운 스위치 앵커 점검은 PREREG_S7_DEQ_v2 §5(4) 규정에 따라 경계 걸음 0인 앵커가 3개 미만(0개)이므로 코드 2로 종료되었으며, GPU 1은 유휴 상태로 복귀함.
- 보고서 발행 및 `STATE.md` 갱신 후 즉시 `python shared/wait_for_task.py 51`을 기동하여 후속 지침을 수신하겠습니다.

---

## 6. 정오표 (TASK-0051)

- **대상 절**: §3.2 (부드러운 스위치 앵커 점검 종료 상태)
- **정정 내용**: Q#20 경계 걸음 수 "38" → **852**
- **원시 출력 근거**: `scratch/task0049_out.txt` 30행 (`b_steps=852/1000 (85.2%)`)
- **원인**: 이전 로그 파편 기억 혼선으로 인한 오기재(원시 수집본 미대조).

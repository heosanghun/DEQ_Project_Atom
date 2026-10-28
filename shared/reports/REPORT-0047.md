<!-- SHA256: 319577c3d5be5e88613ff0b2d51d9b16ea783062dcfa57ee20592bd5be71c223 -->

# REPORT-0047: 부드러운 스위치 코드 FIX (F1~F6) · 서버 업로드 · 서버 CPU SMOKE 보고서

- **작성 일시**: 2026-10-06 18:58 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, `D:\AI\cts`)
- **근거 태스크**: [TASK-0047.md](file:///d:/AI/cts/shared/tasks/TASK-0047.md)
- **근거 감사 보고서**: [AUDIT-0046.md](file:///d:/AI/cts/shared/audits/AUDIT-0046.md) (F1~F6 수정 요구)
- **근거 사전등록**: [PREREG_S7_DEQ_v2.md](file:///d:/AI/cts/shared/prereg/PREREG_S7_DEQ_v2.md) (잠금, SHA-256 `d9eef10180d228bb676f5115d97fc721f3acf67d4f94864f8ef33b018a5a72aa`)
- **절대 조건 준수 현황**:
  - `p6_chain` 세션 무간섭(kill 0, attach 0, 수정 0건): 준수 (GPU 0에서 Condition C 생성 98~99% 가동 유지 중)
  - GPU 사용 0건: 준수 (이번 TASK는 서버 CPU SMOKE만 수행, GPU 1 가동 0건)
  - 기존 파일 덮어쓰기 금지: 준수 (`run_v2.py`, `launch_v2.sh`, `__init__.py` 3종 신규 생성)
  - 서버 접속 규약 (R7): `shared/cts_ssh.py` (v2b) 경유 (SFTP 업로드 18:54:33, 65초 쿨다운 준수 후 SSH SMOKE 18:55:39)
  - 금지어 규칙: 9대 금지어 사용 0건
  - 수치 포인터 표기: 수치마다 로그 행 번호 및 SHA-256 명시

---

## 0. 서버 상태 및 p6_chain 무간섭 계측 (TASK-0047 §0)

- **호스트 가동 시간 실측 (`scratch/task0047_out.txt` 6~7행)**:
```text
$ uptime -s
2026-10-05 16:32:53

$ uptime
 18:55:46 up 1 day,  2:22,  1 user,  load average: 1.02, 1.78, 3.25
```
- **tmux 활성 세션 (`scratch/task0047_out.txt` 8행)**:
```text
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
```
- **p6_chain 로그 마지막 10행 (`scratch/task0047_out.txt` 14~23행)**:
```text
[P6-CHAIN 2026-10-06 17:39:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:44:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:49:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN] P5_DONE.flag present and GPU 0 is idle. Commencing chain execution.
[P6-CHAIN] Stage 0: Computing Anchor 20 on GPU 0...
[P6-CHAIN] Stage 0: Comparing anchor20_gpu0.pt vs GPU 1 & GPU 2 on CPU...
[P6-CHAIN] Stage 0 complete. Proceeding to Stage 1.
[P6-CHAIN] Stage 1: Launching z_src(600) calculation...
[P6-CHAIN] Stage 1 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/ZSRC600_DONE.flag.
[P6-CHAIN] Running Generation Condition C (shift=)...
```
- **GPU 0~7 하드웨어 및 연산 프로세스 (`scratch/task0047_out.txt` 174~184행)**:
```text
index, uuid, utilization.gpu [%], memory.used [MiB], temperature.gpu, clocks.current.sm [MHz], power.draw [W]
0, GPU-71be98b0-6b81-9257-c622-97dc97d46480, 99 %, 15818 MiB, 53, 1800 MHz, 135.72 W
1, GPU-9410691f-6857-1642-ec7c-7596970e2f48, 0 %, 16 MiB, 28, 210 MHz, 12.24 W
2, GPU-c81311d8-d210-3857-843e-bb38d24771e1, 0 %, 16 MiB, 30, 210 MHz, 21.64 W
3, GPU-a9fed959-7113-e747-dac4-a515ac50c160, 0 %, 16 MiB, 31, 210 MHz, 18.19 W
4, GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5, 0 %, 16 MiB, 27, 210 MHz, 11.41 W
5, GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a, 0 %, 16 MiB, 27, 210 MHz, 6.76 W
6, GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c, 0 %, 16 MiB, 28, 210 MHz, 6.54 W
7, GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b, 0 %, 16 MiB, 28, 210 MHz, 8.30 W

gpu_uuid, pid, used_gpu_memory [MiB]
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 245681, 15798 MiB
```
- **계측 소견**: `p6_chain` (PID 245681, Condition C)이 GPU 0에서 99% 부하로 가동 중이며, GPU 1은 16 MiB, 28 ℃, 0% 유휴 상태로 대기하고 있습니다.

---

## 1. 서버 업로드 파일 및 SHA-256 계측 (TASK-0047 §1·§2)

| 파일 경로 | 로컬 SHA-256 | 서버 계측 SHA-256 (`scratch/task0047_out.txt` 25~38행) | 일치 여부 |
|---|---|---|:---:|
| `exe_scratch/__init__.py` | `94f50a10185f3f2555400a5b7bb7d2f94be853117aaffdd80559f8fcbb0eab22` | `94f50a10185f3f2555400a5b7bb7d2f94be853117aaffdd80559f8fcbb0eab22` | 일치 |
| `exe_scratch/s7/__init__.py` | `1a1052b322a675cf311be8b99e4eb1fb9783ac50ba599a73cf9e48eaa4f93cd3` | `1a1052b322a675cf311be8b99e4eb1fb9783ac50ba599a73cf9e48eaa4f93cd3` | 일치 |
| `exe_scratch/s7/soft_v2/__init__.py` | `9e425fd94e9a839f6576a545359190463952eec435fcc03ccc162ed8a27d37f1` | `9e425fd94e9a839f6576a545359190463952eec435fcc03ccc162ed8a27d37f1` | 일치 |
| `exe_scratch/s7/soft_v2/zstar_soft_lowest.py` | `d1942c3cf5defbad6c148e69102b1ca6e8c64d1da5dd731434782364fd689f07` | `d1942c3cf5defbad6c148e69102b1ca6e8c64d1da5dd731434782364fd689f07` | 일치 |
| `scripts/s7deq_soft_v2_run_v2.py` | `2687a54e28b73c15e2980ad204724fb4f8903fb7d3b8f3d33d4e3ae11283db17` | `2687a54e28b73c15e2980ad204724fb4f8903fb7d3b8f3d33d4e3ae11283db17` | 일치 |
| `scripts/s7deq_soft_v2_analyze.py` | `07d63b1e9fdce284d93216cc00586bb2f4e6223a7aee13007b650dc94ec73aec` | `07d63b1e9fdce284d93216cc00586bb2f4e6223a7aee13007b650dc94ec73aec` | 일치 |
| `scripts/s7deq_soft_v2_launch_v2.sh` | `cb3ad22f61b9604da92414e3bedbf9b7c38bc387b316a59c5dd56fe6cb7c2784` | `cb3ad22f61b9604da92414e3bedbf9b7c38bc387b316a59c5dd56fe6cb7c2784` | 일치 |

---

## 2. 결함 수정 내역 (F1~F6) 상세

1. **F1 (`scripts/s7deq_soft_v2_run_v2.py`)**:
   - `configs/s6_items_74.json`에서 `all_74` 키를 명시적으로 파싱하도록 강제.
   - 문항 수 74개 계측, 중복 0건 계측, `nonconv_54` 54개 및 `control_20` 20개 상호 배타성 및 합집합 일치(`nonconv_54 | control_20 == all_74`) 점검 루틴 추가.
   - 조건 불일치 시 `ValueError`로 즉시 중단.
2. **F2 (`scripts/s7deq_soft_v2_launch_v2.sh`)**:
   - §5 앵커 사전 점검에서 경계 걸음 0인 문항 수가 3개 미만일 경우 `data/anchor20.json` 순서대로 추가 앵커를 정본과 소프트 러너 모두 순차 실행.
   - 20개 문항 탐색 후에도 경계 걸음 0 문항이 3개 미만이면 `exit 2`로 중단.
3. **F3 (`scripts/s7deq_soft_v2_launch_v2.sh`)**:
   - 경계 걸음 0인 문항에 대해 `z_star`, `z_lowest`뿐 아니라 `rho_traj` 궤적 전체에 대해서도 `max |Delta| == 0.0` 검속 수행.
   - 0이 아닌 차이 발견 시 `exit 1` 중단.
4. **F4 (`scripts/s7deq_soft_v2_launch_v2.sh`)**:
   - 앵커 점검 호출 시 `--alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --tol 0.0 --config configs/zstar_canonical.yaml` 인자 전수 명시.
5. **F5 (`scripts/s7deq_soft_v2_launch_v2.sh`)**:
   - 과거 `results_s7/anchor20_gpu1.pt` 대비 `max |Delta|` 및 `cos` 산출 보고 절 추가.
6. **F6 (`exe_scratch/__init__.py`, `exe_scratch/s7/__init__.py`, `exe_scratch/s7/soft_v2/__init__.py`)**:
   - 패키지 경로 탐색을 위한 `__init__.py` 3종 작성 및 서버 업로드.
7. **메모리 보호 규약 추가 (Rule #381)**:
   - `ulimit -v 62914560` 환경에서 glibc 멀티 아레나 폭증 방지를 위해 `MALLOC_ARENA_MAX=1`, `MALLOC_TRIM_THRESHOLD_=131072`, `TORCH_NUM_THREADS=4` 등을 셸 및 파이썬 초기화에 강제 적용.
- **수정 패치 파일**: [DIFF-soft_v2_fixes.patch](file:///d:/AI/cts/shared/reports/DIFF-soft_v2_fixes.patch) (SHA-256 `3be07928eda1e47834f4496b6a3bdd564ba6bf2a2f804d0d064eda657fa6c4f1`)

---

## 3. 로컬 CPU 단위 시험 결과 (F1 시험 포함 6종)

- **실행 명령**: `python scratch/test_soft_v2.py`
- **실행 결과 원문**:
```text
======================================================================
=== STARTING S7 DEQ SOFT V2 LOCAL CPU UNIT TESTS ===
======================================================================
--- Test (a): d > eps bitwise identical to canonical ---
  d = 0.026805 > eps = 0.01
  last_w = 1.0
  last_boundary = False
  max |Delta| = 0.0
  bitwise equal: True
  PASS: Test (a) bitwise identical.
--- Test (b): d = eps has w = 1.0 and matches Phi_S3 ---
  d = eps = 0.02, computed w = 1.0
  max |blended - phi_S3| = 0.0
  PASS: Test (b) w=1.0 verified.
--- Test (c): d = 0 rank 3-4 module swap yields identical output ---
  w at d=0: 0.5
  max |out_normal - out_swapped| = 0.0
  PASS: Test (c) symmetry at d=0 verified.
--- Test (d): Linearity of w in d ---
  eps = 0.02
  w(0) = 0.5, w(eps) = 1.0
  Expected slope = 25.0, max slope err = 1.10e-13
  PASS: Test (d) linearity verified.
--- Test (e): Analyzer synthetic input test ---
  Metric 1 (<5e-3): 54/54
  Metric 2 (min cos): 1.000000
  Metric 3 (r_1000 med): 8.000000e-04
  Metric 4 (54 mean b_ratio): 4.50%
  Metric 5 (<5e-3): 54/54
  Candidate met: True
  PASS: Test (e) synthetic analyzer test verified.
--- Test (f): F1 74-item config parsing and validation ---
[F1-CHECK] all_74 verified: total=74, nonconv=54, control=20, duplicates=0
  Mock config (nonconv_54 first): loaded 74 items
  Properly rejected all_74 len 73: F1 check failed: 'all_74' has 73 items, expected 74
  Properly rejected nonconv_54 len 53: F1 check failed: 'nonconv_54' has 53 items, expected 54
  Properly rejected overlapping nonconv/control: F1 check failed: nonconv_54 and control_20 overlap by 1 items
  Properly rejected missing all_74 without silent 54 fallback: F1 check failed: config ... must contain 'all_74' (or 'items'/'indices'), found keys: ['nonconv_54', 'control_20']
  Comma separated string parsed: [20, 24, 32]
  PASS: Test (f) F1 config parsing and validation verified.
======================================================================
=== ALL 6 UNIT TESTS PASSED SUCCESSFULLY ===
======================================================================
```

---

## 4. 서버 CPU SMOKE 실행 결과 원문 (TASK-0047 §3)

- **실행 환경**: 서버 CPU (`SMOKE=1`, `CUDA_VISIBLE_DEVICES=""`, `ulimit -v 62914560`, `nice -n 19`)
- **실행 출력 원문 (`scratch/task0047_out.txt`)**:
```text
================================================================================
=== TASK-0047: SHA VERIFICATION & SERVER CPU SMOKE ===
Timestamp: 2026-10-06 18:55:46 KST
================================================================================
=== 0. SYSTEM HEALTH & P6_CHAIN INTEGRITY ===
2026-10-05 16:32:53
 18:55:46 up 1 day,  2:22,  1 user,  load average: 1.02, 1.78, 3.25
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
[P6-CHAIN 2026-10-06 17:14:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:19:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:24:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:29:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:34:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:39:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:44:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:49:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN] P5_DONE.flag present and GPU 0 is idle. Commencing chain execution.
[P6-CHAIN] Stage 0: Computing Anchor 20 on GPU 0...
[P6-CHAIN] Stage 0: Comparing anchor20_gpu0.pt vs GPU 1 & GPU 2 on CPU...
[P6-CHAIN] Stage 0 complete. Proceeding to Stage 1.
[P6-CHAIN] Stage 1: Launching z_src(600) calculation...
[P6-CHAIN] Stage 1 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/ZSRC600_DONE.flag.
[P6-CHAIN] Running Generation Condition C (shift=)...
=== 1. UPLOADED FILES SHA-256 VERIFICATION ===
94f50a10185f3f2555400a5b7bb7d2f94be853117aaffdd80559f8fcbb0eab22  exe_scratch/__init__.py
-rw-rw-r-- 1 shoon shoon 42 2026-10-06 18:54:40.266036382 +0900 exe_scratch/__init__.py
1a1052b322a675cf311be8b99e4eb1fb9783ac50ba599a73cf9e48eaa4f93cd3  exe_scratch/s7/__init__.py
-rw-rw-r-- 1 shoon shoon 45 2026-10-06 18:54:40.303036364 +0900 exe_scratch/s7/__init__.py
9e425fd94e9a839f6576a545359190463952eec435fcc03ccc162ed8a27d37f1  exe_scratch/s7/soft_v2/__init__.py
-rw-rw-r-- 1 shoon shoon 53 2026-10-06 18:54:40.340036347 +0900 exe_scratch/s7/soft_v2/__init__.py
d1942c3cf5defbad6c148e69102b1ca6e8c64d1da5dd731434782364fd689f07  exe_scratch/s7/soft_v2/zstar_soft_lowest.py
-rw-rw-r-- 1 shoon shoon 18347 2026-10-06 18:54:40.380036327 +0900 exe_scratch/s7/soft_v2/zstar_soft_lowest.py
2687a54e28b73c15e2980ad204724fb4f8903fb7d3b8f3d33d4e3ae11283db17  scripts/s7deq_soft_v2_run_v2.py
-rwxr-xr-x 1 shoon shoon 9887 2026-10-06 18:54:40.417036310 +0900 scripts/s7deq_soft_v2_run_v2.py
07d63b1e9fdce284d93216cc00586bb2f4e6223a7aee13007b650dc94ec73aec  scripts/s7deq_soft_v2_analyze.py
-rwxr-xr-x 1 shoon shoon 13577 2026-10-06 18:54:40.454036292 +0900 scripts/s7deq_soft_v2_analyze.py
cb3ad22f61b9604da92414e3bedbf9b7c38bc387b316a59c5dd56fe6cb7c2784  scripts/s7deq_soft_v2_launch_v2.sh
-rwxr-xr-x 1 shoon shoon 10461 2026-10-06 18:54:40.491036274 +0900 scripts/s7deq_soft_v2_launch_v2.sh
=== 2. RUN SERVER CPU SMOKE VIA launch_v2.sh ===
================================================================================
=== S7 DEQ SMOOTH TOP-3 SWITCH LAUNCH (v2) ===
Timestamp: 2026-10-06 18:55:46 KST
================================================================================
Config 74 SHA: 42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3
[MODE] SMOKE=1 CPU execution mode enabled.
Running CPU SMOKE with 2 items from configs/s6_items_74.json (max_nfe=5)...
[F1-CHECK] all_74 verified: total=74, nonconv=54, control=20, duplicates=0
[SMOKE-MODE] Slicing execution from 74 items to first 2 items.
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.01, alpha=0.25, max_nfe=5, items=2
Device: cpu | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: scratch/s7deq_soft_v2_smoke_eps0.01.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377

Loading weights:   0%|          | 0/2076 [00:00<?, ?it/s]
Loading weights:  36%|███▋      | 754/2076 [00:00<00:00, 1710.91it/s]
Loading weights:  45%|████▍     | 926/2076 [00:01<00:02, 507.49it/s] 
Loading weights: 100%|██████████| 2076/2076 [00:01<00:00, 1354.59it/s]
[0001/0002] q0031_a0.25: r_1000=2.802327e-01, r_lowest=2.802327e-01 at k*=5, r_hard_lowest=2.802327e-01, b_steps=0/1000 (0.0%), in_b=False (5.25s)
[0002/0002] q0127_a0.25: r_1000=2.742026e-01, r_lowest=2.742026e-01 at k*=5, r_hard_lowest=2.742026e-01, b_steps=0/1000 (0.0%), in_b=False (5.27s)
================================================================================
Finished. Total items saved: 2 to scratch/s7deq_soft_v2_smoke_eps0.01.pt
================================================================================
Running analyzer on SMOKE output...
Analysis saved to scratch/s7deq_soft_v2_smoke_eps0.01.json and scratch/s7deq_soft_v2_smoke_eps0.01.md
Input SHA: 153fb4c7a6c25a5d7deb8f339c17966d90cfac934a547353e488be0e11902a4b
Metric 1 (<5e-3): 0/54 (0.0%)
[SMOKE-COMPLETE] SMOKE execution finished successfully.
SMOKE_RC=0
=== 3. SMOKE ARTIFACTS IN scratch/ ===
153fb4c7a6c25a5d7deb8f339c17966d90cfac934a547353e488be0e11902a4b  scratch/s7deq_soft_v2_smoke_eps0.01.pt
-rw-rw-r-- 1 shoon shoon 2624997 2026-10-06 18:56:14.782994474 +0900 scratch/s7deq_soft_v2_smoke_eps0.01.pt
4b80c7853d704737859d7a76cb125113eb1a1d2e9db0a59e2dce82e47a38eba4  scratch/s7deq_soft_v2_smoke_eps0.01.json
-rw-rw-r-- 1 shoon shoon 2064 2026-10-06 18:56:17.719993277 +0900 scratch/s7deq_soft_v2_smoke_eps0.01.json
b29272494e96f6164ba4462b7a11f521cecdb2f7d655e31dc34c5bec5b23a0f9  scratch/s7deq_soft_v2_smoke_eps0.01.md
-rw-rw-r-- 1 shoon shoon 1029 2026-10-06 18:56:17.719993277 +0900 scratch/s7deq_soft_v2_smoke_eps0.01.md
=== BEGIN FILE: scratch/s7deq_soft_v2_smoke_eps0.01.json ===
{
  "eps": 0.01,
  "metric_1": {
    "evaluated_count": 2,
    "denominator": 54,
    "count_lt_1e3": 0,
    "count_lt_5e3": 0,
    "ratio_lt_5e3": 0.0,
    "criterion_threshold_count": 52,
    "criterion_met": false
  },
  "metric_2": {
    "evaluated_count": 0,
    "denominator": 20,
    "min_cos": null,
    "count_ge_09990": 0,
    "count_lt_09950": 0,
    "criterion_met": false,
    "shift_flag": false
  },
  "metric_3": {
    "evaluated_count": 2,
    "denominator": 2,
    "median": 0.27721763745477984,
    "max": 0.2802326995442006,
    "count_lt_1e3": 0
  },
  "metric_4": {
    "group_54": {
      "mean": 0.0,
      "median": 0.0,
      "max": 0.0,
      "last100_mean": 0.0,
      "last100_median": 0.0,
      "last100_max": 0.0
    },
    "group_20": {
      "mean": 0.0,
      "median": 0.0,
      "max": 0.0,
      "last100_mean": 0.0,
      "last100_median": 0.0,
      "last100_max": 0.0
    }
  },
  "metric_5": {
    "evaluated_count": 2,
    "denominator": 54,
    "count_lt_1e3": 0,
    "count_lt_5e3": 0,
    "in_boundary_count": 0
  },
  "candidate_assessment": {
    "cond1_ratio_ge_95pct": false,
    "cond2_min_cos_ge_09990": false,
    "is_candidate": false,
    "shift_flag": false
  },
  "p3_baseline_reference": {
    "metric_1_lt_1e3": "53/54 (0.981)",
    "metric_1_lt_5e3": "52/54 (0.963)",
    "metric_2_min_cos": 0.9326308762296536,
    "metric_2_ge_09990": 10,
    "metric_2_lt_09950": 8,
    "note": "P3 metric 1 used mask-freeze mapping; v2 metric 1 uses smooth-switch mapping."
  },
  "metadata": {
    "script_path": "/home/shoon/cts_e1_s5/scripts/s7deq_soft_v2_analyze.py",
    "script_sha256": "07d63b1e9fdce284d93216cc00586bb2f4e6223a7aee13007b650dc94ec73aec",
    "input_pt_path": "scratch/s7deq_soft_v2_smoke_eps0.01.pt",
    "input_pt_sha256": "153fb4c7a6c25a5d7deb8f339c17966d90cfac934a547353e488be0e11902a4b",
    "items_json_path": "configs/s6_items_74.json",
    "items_json_sha256": "42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3",
    "ref_pt_path": null,
    "ref_pt_sha256": "NONE"
  }
}=== END FILE: scratch/s7deq_soft_v2_smoke_eps0.01.json ===
=== BEGIN FILE: scratch/s7deq_soft_v2_smoke_eps0.01.md ===
# S7 DEQ Smooth Top-3 Switch Analysis (eps=0.01)

## Comparison Table (Smooth Switch vs P3 Baseline)
| Metric | Definition | Smooth Switch (eps=0.01) | P3 Baseline (Mask Freeze) |
|---|---|:---:|:---:|
| (1) Fixed-point arrival (54) | r* < 5e-3 count / 54 | 0/54 (0.0%) | 52/54 (0.963) |
| (1) Strict arrival (54) | r* < 1e-3 count / 54 | 0/54 | 53/54 (0.981) |
| (2) Control stability min cos | min cos(z*_eps, z_ref) | N/A | 0.932631 |
| (2) Control cos >= 0.9990 | count / 20 | 0/20 | 10/20 |
| (2) Control cos < 0.9950 | count / 20 | 0/20 | 8/20 |
| (3) Final residual r_1000 | median (max) | 2.772176e-01 (2.802327e-01) | - |
| (4) Boundary step ratio (54) | mean (max) | 0.0% (0.0%) | - |
| (4) Boundary step ratio (20) | mean (max) | 0.0% (0.0%) | - |
| (5) Original machine residual (54) | r_hard < 5e-3 count | 0/54 | - |
| (5) Boundary fixed-point (54) | d <= eps count | 0/54 | - |

> [!NOTE]
> P3 metric (1) evaluated residual under mask-freeze mapping; v2 metric (1) evaluates residual under smooth-switch mapping.

=== END FILE: scratch/s7deq_soft_v2_smoke_eps0.01.md ===
=== 4. CURRENT GPU STATUS & COMPUTE APPS ===
index, uuid, utilization.gpu [%], memory.used [MiB], temperature.gpu, clocks.current.sm [MHz], power.draw [W]
0, GPU-71be98b0-6b81-9257-c622-97dc97d46480, 99 %, 15818 MiB, 53, 1800 MHz, 135.72 W
1, GPU-9410691f-6857-1642-ec7c-7596970e2f48, 0 %, 16 MiB, 28, 210 MHz, 12.24 W
2, GPU-c81311d8-d210-3857-843e-bb38d24771e1, 0 %, 16 MiB, 30, 210 MHz, 21.64 W
3, GPU-a9fed959-7113-e747-dac4-a515ac50c160, 0 %, 16 MiB, 31, 210 MHz, 18.19 W
4, GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5, 0 %, 16 MiB, 27, 210 MHz, 11.41 W
5, GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a, 0 %, 16 MiB, 27, 210 MHz, 6.76 W
6, GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c, 0 %, 16 MiB, 28, 210 MHz, 6.54 W
7, GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b, 0 %, 16 MiB, 28, 210 MHz, 8.30 W
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 245681, 15798 MiB
```

---

## 5. 요약 및 후속 조치

1. **F1~F6 결함 조치**:
   - `s7deq_soft_v2_run_v2.py` 및 `s7deq_soft_v2_launch_v2.sh` 작성.
   - 로컬 단위 시험 6개 항목(a~f) 전수 PASS.
2. **서버 반영 및 SMOKE 계측**:
   - 서버 업로드 7개 파일 전수 SHA-256 일치.
   - 실제 `configs/s6_items_74.json` 로드 시 `[F1-CHECK] all_74 verified: total=74, nonconv=54, control=20, duplicates=0` 실측 기록.
   - 2문항 DEQ 연산 및 `s7deq_soft_v2_analyze.py` 산출물(`scratch/s7deq_soft_v2_smoke_eps0.01.pt`, `.json`, `.md`) 계측 (`SMOKE_RC=0`).
3. **후속 대기**:
   - CC 감사 및 다음 태스크(TASK-0048, GPU 1 본 실행) 지시 대기.

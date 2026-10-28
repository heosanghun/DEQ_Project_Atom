<!-- SHA256: 8b38d3dda4baf69277b9a2c2a89239dbdcef4c2ba1c0bc0886c24831deeaa79a -->
# REPORT-0055: 중간 점검 계측 보고 (부드러운 스위치·P6·재부팅)

- **작성 일시**: 2026-10-06 23:45 KST
- **작성 주체**: 회사 PC 정식 루프 에이전트 (Antigravity)
- **근거 작업**: `shared/tasks/TASK-0055.md`
- **근거 규정**: AUDIT-0054, `shared/prereg/PREREG_S7_DEQ_v2.1.md` §6, CTS-428
- **원시 로그**: `scratch/task0055_out.txt` (97행, 단일 SSH 묶음 계측 로그)

---

## 1. 개요 및 핵심 요약

`TASK-0055` 지시에 따라 서버 접속 규약(R7-1~R7-9)을 준수하며 단일 SSH 묶음 호출을 통해 호스트 가동 상태, 부드러운 스위치(GPU 1), P6(GPU 0) 진행 현황을 일괄 계측했다.

1. **3차 재부팅 부재 실측**:
   - 호스트 부팅 시각: `2026-10-05 16:32:53` (`task0055_out.txt:6`).
   - 직전 기준 시각(`2026-10-05 16:32:53`)과 정확히 일치하여 **3차 재부팅 부재**를 계측했다. 연속 가동 시간은 1일 7시간 5분이다 (`task0055_out.txt:7`).
   - 활성 세션: `p6_chain`, `s7deq_soft_v21` 2개 세션 건전 가동 중 (`task0055_out.txt:9-10`).
2. **부드러운 스위치(GPU 1) 진행 현황**:
   - $\varepsilon=0.005$ 74문항 전수 연산 마침: 21:23:30 KST 기동 후 23:32:16 KST 마침 계측 (소요 시간 2시간 8분 46초, 7,726초) (`task0055_out.txt:13, 16`).
   - $\varepsilon=0.005$ 산출물 생성: `.pt`(99,050,013 B), `.json`(2,263 B), `.md`(1,046 B) (내용 열람 0건 엄수) (`task0055_out.txt:32-34`).
   - $\varepsilon=0.01$ 기동 및 진행: 23:32:16 KST 진입 (`task0055_out.txt:14`), 계측 시점(23:38:08 KST) 기준 4번째 문항(`[0004/0074] q0249_a0.25`) 연산 누적 (`task0055_out.txt:24-25`).
   - v5 실행 중 오류/Traceback: 0건 (`task0055_out.txt:26-30`).
3. **P6 체인(GPU 0) 진행 현황**:
   - Stage 2 Condition C 생성 마침: 600행 누적 (`results_s7/p6_gen_C.jsonl`, 800,733 B, MTIME 21:45:49 KST) 및 `P6_GEN_C_DONE.flag` 생성 실측 (`task0055_out.txt:43, 50, 60`).
   - Stage 2 Condition A 순항 가동: 21:45 KST 기동 후 476행 누적 (`results_s7/p6_gen_A.jsonl`, 379,333 B, MTIME 23:38:03 KST) (`task0055_out.txt:44, 47, 57`). 내용 열람 0건 엄수.
4. **자원 분리 및 거버넌스 엄수**:
   - GPU 0 (PID 280324, P6 Condition A): 99 % 점유율, 16,162 MiB 점유, 54 ℃, 133.67 W (`task0055_out.txt:68, 78, 88`).
   - GPU 1 (PID 297624, s7deq $\varepsilon=0.01$): 99 % 점유율, 30,776 MiB 점유, 55 ℃, 120.88 W (`task0055_out.txt:69, 79, 89`).
   - GPU 2~7: 0 % 점유율, 16 MiB (GPU 4 영구 봉인 보존, 2/3/5/6/7 미사용) (`task0055_out.txt:70-75`).
   - 시스템 가용 메모리: 236 GiB (여유 183 GiB, 스왑 0 B 사용) 건전 보존 (`task0055_out.txt:92-93`).

---

## 2. 부드러운 스위치 (GPU 1) 진행 상세

### 2.1 실행 로그 및 산출물 파일 메타데이터

- **실행 배너 로그 (`task0055_out.txt:12-16`)**:
  ```text
  --- 1.1 RUNNING EPS Lines ---
  === RUNNING EPS=0.005 AT 2026-10-06 21:23:30 KST ===
  === RUNNING EPS=0.01 AT 2026-10-06 23:32:16 KST ===
  --- 1.2 Completed eps Lines ---
  Completed eps=0.005 at 2026-10-06 23:32:16 KST
  ```

- **$\varepsilon=0.005$ 산출물 파일 (`task0055_out.txt:32-34`)**:
  - `results_s7/s7deq_soft_v2_eps0.005.pt`: **99,050,013 B** (2026-10-06 23:32:12 KST)
  - `results_s7/s7deq_soft_v2_eps0.005.json`: **2,263 B** (2026-10-06 23:32:16 KST, 내용 미열람)
  - `results_s7/s7deq_soft_v2_eps0.005.md`: **1,046 B** (2026-10-06 23:32:16 KST, 내용 미열람)

- **진행 헤더 및 최신 진행 번호 (수치 0 엄수) (`task0055_out.txt:18-25`)**:
  - 로그 내 전체 문항 헤더 수: **78건** ($\varepsilon=0.005$ 74건 + $\varepsilon=0.01$ 4건) (`task0055_out.txt:18`).
  - 최근 5개 문항 헤더:
    - `[0074/0074] q4773_a0.25` ($\varepsilon=0.005$ 마지막 문항) (`task0055_out.txt:20`)
    - `[0001/0074] q0031_a0.25` ($\varepsilon=0.01$ 시작) (`task0055_out.txt:21`)
    - `[0002/0074] q0127_a0.25` (`task0055_out.txt:22`)
    - `[0003/0074] q0138_a0.25` (`task0055_out.txt:23`)
    - `[0004/0074] q0249_a0.25` (`task0055_out.txt:24`)
  - 현재 최신 문항 인덱스: `[0004/0074]` (`task0055_out.txt:25`).

- **오류 및 Traceback 검속**:
  - 로그 내 출력된 오류 행(17, 48, 58, 598행)은 모두 과거 실행 및 이전 버전(v2 등)의 이력 로그이며, 금회 v5 실행(599행 이후)에서는 예외 및 오류 발생 0건을 계측했다 (`task0055_out.txt:26-30`).

---

## 3. P6 체인 (GPU 0) 진행 상세

### 3.1 P6 체인 로그 실측 (`task0055_out.txt:37-44`)

```text
[P6-CHAIN] Stage 0: Computing Anchor 20 on GPU 0...
[P6-CHAIN] Stage 0: Comparing anchor20_gpu0.pt vs GPU 1 & GPU 2 on CPU...
[P6-CHAIN] Stage 0 complete. Proceeding to Stage 1.
[P6-CHAIN] Stage 1: Launching z_src(600) calculation...
[P6-CHAIN] Stage 1 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/ZSRC600_DONE.flag.
[P6-CHAIN] Running Generation Condition C (shift=)...
[P6-CHAIN] Generation Condition C (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_C_DONE.flag.
[P6-CHAIN] Running Generation Condition A (shift=)...
```

### 3.2 P6 결과 파일 현황 (`task0055_out.txt:46-64`)

- `results_s7/p6_gen_C.jsonl`: **600행** (800,733 B, MTIME 2026-10-06 21:45:49 KST) — 생성 종결 실측 (`task0055_out.txt:50, 60`).
- `results_s7/p6_gen_A.jsonl`: **476행** (379,333 B, MTIME 2026-10-06 23:38:03 KST) — 21:45 KST 기동 후 순항 생성 중 (`task0055_out.txt:47, 57`).
- 스모크 파일 보존: `p6_gen_A_smoke.jsonl`(2행), `p6_gen_B1_smoke.jsonl`(2행), `p6_gen_C_smoke.jsonl`(2행), `p6_gen_E1_smoke.jsonl`(2행), `p6_gen_F_smoke.jsonl`(2행) (`task0055_out.txt:58, 59, 61, 62, 63`).
- P6 산출물의 본문 내용 및 점수 열람 0건을 엄수했다.

---

## 4. 자원 분리 및 시스템 상태 실측

2026-10-06 23:38:08 KST 단일 SSH 세션 실측치:

### 4.1 GPU 0~7 텔레메트리 (`task0055_out.txt:67-75`)

| GPU 인덱스 | 물리 UUID | 점유율 [%] | 메모리 점유 [MiB] | 온도 [℃] | 전력 [W] | 활성 프로세스 및 비고 |
|---|---|---|---|---|---|---|
| **GPU 0** | `GPU-71be98b0...` | 99 % | 16,162 MiB | 54 ℃ | 133.67 W | PID 280324 (`p6_chain` Condition A) |
| **GPU 1** | `GPU-9410691f...` | 99 % | 30,776 MiB | 55 ℃ | 120.88 W | PID 297624 (`s7deq_soft_v21` $\varepsilon=0.01$) |
| **GPU 2** | `GPU-c81311d8...` | 0 % | 16 MiB | 31 ℃ | 21.81 W | 유휴 보존 |
| **GPU 3** | `GPU-a9fed959...` | 0 % | 16 MiB | 33 ℃ | 18.00 W | 유휴 보존 |
| **GPU 4** | `GPU-e27bdb0d...` | 0 % | 16 MiB | 28 ℃ | 12.32 W | **영구 봉인 유지** |
| **GPU 5** | `GPU-84d7eeb0...` | 0 % | 16 MiB | 28 ℃ | 6.83 W | 유휴 보존 |
| **GPU 6** | `GPU-f99bbac4...` | 0 % | 16 MiB | 31 ℃ | 15.59 W | 유휴 보존 |
| **GPU 7** | `GPU-0eb58a21...` | 0 % | 16 MiB | 32 ℃ | 21.07 W | 유휴 보존 |

### 4.2 실행 프로세스 상세 (`task0055_out.txt:77-89`)

- **GPU 0 프로세스**:
  - PID: `280324` (`pts/2`, CPU 103%, VSZ 69.0GB, RSS 2.9GB, 누적 시간 115:45) (`task0055_out.txt:88`)
  - 명령줄: `/home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond A`
- **GPU 1 프로세스**:
  - 부모 PID: `276023` (`pts/0`, `bash scripts/s7deq_soft_v2_launch_v5.sh`) (`task0055_out.txt:87`)
  - 자식 PID: `297624` (`pts/0`, CPU 112%, VSZ 42.8GB, RSS 1.9GB, 누적 시간 6:34) (`task0055_out.txt:89`)
  - 명령줄: `/home/shoon/system1_5/.venv/bin/python scripts/s7deq_soft_v2_run_v2.py --items configs/s6_items_74.json --out results_s7/s7deq_soft_v2_eps0.01.pt --eps 0.01 --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --tol 0.0 --max_nfe 1000 --device cuda:0 --config configs/zstar_canonical.yaml`
- **시스템 메모리 (`task0055_out.txt:91-93`)**:
  - 전체: 251 GiB, 사용: 12 GiB, 여유: 183 GiB, 버퍼/캐시: 55 GiB, 가용: 236 GiB, 스왑 사용: 0 B. 시스템 자원 여유도 건전 유지.

---

## 5. 문항당 소요 시간 및 종료 시각 예측

TASK-0055 4항 규정에 따라 헤더 수 및 경과 시간을 기반으로 문항당 평균 소요 시간과 단계별 예상 종료 시각을 산출했다.

### 5.1 문항당 평균 시간 실측치

1. **$\varepsilon=0.005$ 실측 속도**:
   - 총 소요 시간: 21:23:30 ~ 23:32:16 = **7,726초** (128.77분, ~2.15시간) (`task0055_out.txt:13, 16`).
   - 문항 수: 74문항.
   - 문항당 평균 소요 시간: $7,726 \text{ s} / 74 = \mathbf{104.41} \text{ s/문항}$ (~1.74분/문항).
2. **$\varepsilon=0.01$ 초기 계측 속도**:
   - 23:32:16 ~ 23:38:08 = **352초** 경과 (`task0055_out.txt:14, 96`).
   - 연산 마침 문항: 4문항 (`task0055_out.txt:24-25`).
   - 초기 문항당 소요 시간: $352 \text{ s} / 4 = \mathbf{88.00} \text{ s/문항}$ (~1.47분/문항).

### 5.2 단계별 및 전체 예상 종료 시각 산출

- **$\varepsilon=0.01$ 예상 종료 시각**:
  - 잔여 문항: $74 - 4 = 70$문항.
  - 문항당 소요 시간 88.0초 ~ 104.4초 적용 시 잔여 시간: 약 6,160초(102.7분) ~ 7,308초(121.8분).
  - 예상 종료 시각: **2026-10-07 01:15 ~ 01:34 KST**.
- **$\varepsilon=0.02$ 예상 종료 시각**:
  - 총 74문항, 문항당 90초 ~ 110초 적용 시 예상 소요 시간: 약 6,660초(111분) ~ 8,140초(135.7분).
  - 예상 종료 시각: **2026-10-07 03:06 ~ 03:50 KST**.
- **전체 실험 종결 예상**:
  - 3개 $\varepsilon$ 전체 종료는 **2026-10-07 03:10 ~ 03:50 KST**로 예측되며, 이는 `AUDIT-0054`의 사전 예측치(10/7 03~05시)에 긴밀히 부합한다.

---

## 6. 결론 및 후속 절차

- 호스트의 3차 재부팅은 발생하지 않았으며(부팅 시각 `2026-10-05 16:32:53` 유지), 부드러운 스위치 $\varepsilon=0.005$는 74문항 연산을 마치고 $\varepsilon=0.01$ 4문항째 연산에 순항 진입했다.
- P6 체인은 Condition C(600행) 생성을 마치고 Condition A(476행) 생성을 GPU 0에서 오류 없이 진행 중이다.
- 본 보고서 발행 후 `shared/STATE.md`를 갱신하고, 지침에 따라 `python shared/wait_for_task.py 56` 대기 명령을 기동한다.

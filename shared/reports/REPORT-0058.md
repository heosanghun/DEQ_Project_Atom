<!-- SHA256: acb4e9ada9485b77599f4728516dd2b5e536d21e8cc69656da35534241ef5057 -->

# REPORT-0058: 진행 점검 (①-a 재계산·P6·재부팅)

- **발행 일시**: 2026-10-07 07:05 KST
- **수행자**: 회사 PC 정식 루프 (CTS 실행 에이전트)
- **근거 문서**: TASK-0058, AUDIT-0057, PREREG_S7_DEQ_v1.4(04112522…) §10(h)
- **규약 준수**: RULES R3-5 준수, 9대 금지어 0건, 수치마다 포인터 명시, 모든 세션 무간섭, GPU 사용 0, 재계산 결과 수치 인용 0건, P6 내용 열람 0건, R7-1/R7-3 접속 규약 준수, CR 0건 (LF 전용)

---

## 1. 수행 결과 총괄 요약

1. **0. 호스트 및 부팅 진단 (단일 SSH 묶음 점검)**:
   - **호스트 가동**: 부팅 시각 `2026-10-05 16:32:53` 불변(1일 14시간 21분 연속 가동 실측, 포인터: `scratch/task0058_out.txt:7-8`). 3차 재부팅 부재 계측.
   - **tmux 세션**: `kstar_replay` 및 `p6_chain` 2개 세션 건전 가동 중 (포인터: `scratch/task0058_out.txt:10-11`).

2. **1. ①-a 재계산 (GPU 1) 진행 현황**:
   - **진행 헤더 번호**: 마지막 진행 헤더 `[0122/0325] q1993_a0.25` 계측 (포인터: `scratch/task0058_out.txt:19, 21`).
   - **누적 헤더 수**: 122건 계측 (포인터: `scratch/task0058_out.txt:23`).
   - **오류 및 이상**: `raw_s7/kstar_replay_gpu1.log` 내 오류·Traceback 0건 계측 (`NO_ERRORS_DETECTED`, 포인터: `scratch/task0058_out.txt:25`).
   - **출력 텐서 파일**: `results_s7/zstar_s7_a0.25_train325_kstar_replay.pt` 크기 158,451,389 B, 최종 갱신 시각 2026-10-07 06:52:15 KST 실측 (포인터: `scratch/task0058_out.txt:27`).
   - **종료 플래그**: `raw_s7/KSTAR_REPLAY_DONE.flag` 미생성 (연산 진행 중, 포인터: `scratch/task0058_out.txt:29`).
   - **속도 및 예상 종료 시각**: 문항당 평균 소요 시간 약 63.2~63.7초(벽시계 63.19초/문항, 로그 소요시간 평균 63.67초/문항). 잔여 203문항 소요 예상 시간 약 3시간 33분~35분으로, 종료 예상 시각은 10/7 10:25~10:30 KST (포인터: `scratch/task0058_out.txt:33, 48`).
   - **수치 인용 0건 준수**: TASK-0058 지시에 따라 325문항 재계산 결과 수치(잔차, $k^*$, 코사인 유사도, top-3 토큰 등)는 일체 인용·출력하지 않고 진행 헤더 번호·개수·파일 크기만 기록함.

3. **2. P6 체인 (GPU 0) 진행 현황**:
   - **선행 단계 마침 실측**: Condition C(600행), Condition A(600행), Condition B1(600행), Condition E1(600행) 생성 마침 및 Stage 3 메인 게이트 채점 마침(`P6_SCORE_MAIN_DONE.flag` 생성, 04:42 KST), Condition F 600행 생성 마침(`P6_GEN_F_DONE.flag` 생성, 06:53 KST) 실측 (포인터: `scratch/task0058_out.txt:52-62, 65-70`).
   - **현재 진행 단계**: Condition D1 생성 연산 진행 중(`p6_gen_D1.jsonl` 5행 생성, PID 373302, 포인터: `scratch/task0058_out.txt:63, 78, 120`).
   - **내용 및 점수 비열람 준수**: P6 생성 텍스트 내용 및 평가 점수 열람 0건 유지(파일 목록 및 행 수만 계측).

4. **3. 하드웨어 자원 격리 및 안정성**:
   - **GPU 0**: 부하 99%, 사용 메모리 15,872 MiB, 온도 48 ℃, 전력 131.99 W, PID 373302 (P6 Cond D1, 포인터: `scratch/task0058_out.txt:90, 100, 120`).
   - **GPU 1**: 부하 99%, 사용 메모리 31,232 MiB, 온도 52 ℃, 전력 118.89 W, PID 346655 (①-a 325문항 재계산, 포인터: `scratch/task0058_out.txt:91, 101, 117`).
   - **GPU 2~7**: 부하 0%, 메모리 16 MiB 유휴 (GPU 4 영구 봉인 유지, 포인터: `scratch/task0058_out.txt:92-97`).
   - **시스템 메모리**: 전체 251 GiB 중 가용 236 GiB (사용 12 GiB), 스왑 0 B 사용 (포인터: `scratch/task0058_out.txt:125-127`).

---

## 2. 0. 호스트 가동 및 세션 상세

### 2.1. 호스트 부팅 시각 및 연속 가동
- 부팅 기준 시각: `2026-10-05 16:32:53` (불변, 포인터: `scratch/task0058_out.txt:7`)
- 현재 가동 상태: `06:54:30 up 1 day, 14:21,  2 users,  load average: 5.02, 3.17, 2.45` (포인터: `scratch/task0058_out.txt:8`)
- 진단 결과: 3차 재부팅 부재 계측 (연속 가동 1일 14시간 21분).

### 2.2. 활성 tmux 세션 목록
- 세션 1: `kstar_replay: 1 windows (created Wed Oct  7 04:38:27 2026)` (포인터: `scratch/task0058_out.txt:10`)
- 세션 2: `p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)` (포인터: `scratch/task0058_out.txt:11`)

---

## 3. 1. ①-a 재계산 (GPU 1) 상세 진척

### 3.1. 최근 진행 헤더 번호 및 누적 집계
로그 파일: `raw_s7/kstar_replay_gpu1.log` (포인터: `scratch/task0058_out.txt:14-23`)
- 최근 5개 진행 헤더 번호:
  * `[0118/0325] q1958_a0.25`
  * `[0119/0325] q1959_a0.25`
  * `[0120/0325] q1975_a0.25`
  * `[0121/0325] q1990_a0.25`
  * `[0122/0325] q1993_a0.25`
- 마지막 진행 헤더 번호: `[0122/0325]`
- 전체 진행 헤더 개수: 122건
- 로그 내 오류 및 Traceback 검속 결과: `NO_ERRORS_DETECTED` (0건 검출, 포인터: `scratch/task0058_out.txt:25`).

### 3.2. 산출 파일 및 플래그 상태
- 본 실행 산출 텐서: `results_s7/zstar_s7_a0.25_train325_kstar_replay.pt`
  * 파일 크기: 158,451,389 Bytes
  * 최종 수정 시각: `2026-10-07 06:52:15.141175691 +0900` (포인터: `scratch/task0058_out.txt:27`)
- 사전 점검 산출 텐서: `results_s7/kstar_replay_precheck5_gpu1.pt`
  * 파일 크기: 6,603,333 Bytes
  * 최종 수정 시각: `2026-10-07 04:43:42.623617062 +0900` (포인터: `scratch/task0058_out.txt:31`)
- 시작 플래그: `raw_s7/KSTAR_REPLAY_STARTED.flag` 보존 계측 (`STARTED_AT: 2026-10-07 04:43:46 KST`, 포인터: `scratch/task0058_out.txt:33`)
- 종료 플래그: `raw_s7/KSTAR_REPLAY_DONE.flag` 상태: `DONE_FLAG_NOT_EXISTS` (미종료, 연산 진행 중, 포인터: `scratch/task0058_out.txt:29`).

### 3.3. 속도 분석 및 예상 종료 시각 산출
- 시작 시각: 2026-10-07 04:43:46 KST (포인터: `scratch/task0058_out.txt:33`)
- 122번째 문항 갱신 시각: 2026-10-07 06:52:15 KST (포인터: `scratch/task0058_out.txt:27`)
- 경과 시간: 2시간 8분 29초 = 7,709초
- 실측 평균 소요 시간:
  * 벽시계 기준: $7,709\text{ s} / 122 = 63.19\text{ s/문항}$
  * 로그 개별 소요시간 합산 기준: 사전 점검 5문항 포함 총 127건 합산 8,085.56초, 평균 63.67초/문항 (포인터: `scratch/task0058_out.txt:48`)
- 잔여 문항 수: $325 - 122 = 203\text{ 문항}$
- 잔여 소요 예상 시간:
  * $203 \times 63.19\text{ s} = 12,827\text{ s} \approx 3\text{시간 } 33\text{분 } 47\text{초}$
  * $203 \times 63.67\text{ s} = 12,925\text{ s} \approx 3\text{시간 } 35\text{분 } 25\text{초}$
- 예상 종료 시각: **2026-10-07 10:25 ~ 10:30 KST**

---

## 4. 2. P6 체인 (GPU 0) 상세 진척

### 4.1. `raw_s7/p6_chain.log` 최근 행 기록
포인터: `scratch/task0058_out.txt:52-63`
```text
[P6-CHAIN] Generation Condition C (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_C_DONE.flag.
[P6-CHAIN] Running Generation Condition A (shift=)...
[P6-CHAIN] Generation Condition A (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_A_DONE.flag.
[P6-CHAIN] Running Generation Condition B (shift=1)...
[P6-CHAIN] Generation Condition B (shift=1) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_B1_DONE.flag.
[P6-CHAIN] Running Generation Condition E1 (shift=)...
[P6-CHAIN] Generation Condition E1 (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_E1_DONE.flag.
[P6-CHAIN] Stage 3: Scoring Main Gates (C, A, B1, E1)...
[P6-CHAIN] Stage 3 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_SCORE_MAIN_DONE.flag.
[P6-CHAIN] Running Generation Condition F (shift=)...
[P6-CHAIN] Generation Condition F (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_F_DONE.flag.
[P6-CHAIN] Running Generation Condition D1 (shift=)...
```

### 4.2. P6 마침 플래그 현황
포인터: `scratch/task0058_out.txt:65-70`
| 플래그 파일 | 생성 시각 (KST) | 상태 |
|---|---|---|
| `raw_s7/P6_GEN_C_DONE.flag` | 10월 6 21:45 | 생성 마침 |
| `raw_s7/P6_GEN_A_DONE.flag` | 10월 7 00:10 | 생성 마침 |
| `raw_s7/P6_GEN_B1_DONE.flag` | 10월 7 02:49 | 생성 마침 |
| `raw_s7/P6_GEN_E1_DONE.flag` | 10월 7 04:41 | 생성 마침 |
| `raw_s7/P6_SCORE_MAIN_DONE.flag` | 10월 7 04:42 | 채점 마침 |
| `raw_s7/P6_GEN_F_DONE.flag` | 10월 7 06:53 | 생성 마침 |

### 4.3. P6 생성 파일 행 수 현황
포인터: `scratch/task0058_out.txt:72-83`
| 구분 | 파일 경로 | 행 수 (건) | 비고 |
|---|---|---|---|
| Condition A | `results_s7/p6_gen_A.jsonl` | 600 | 600문항 생성 마침 |
| Condition A 스모크 | `results_s7/p6_gen_A_smoke.jsonl` | 2 | 시험 생성 |
| Condition B1 | `results_s7/p6_gen_B1.jsonl` | 600 | 600문항 생성 마침 |
| Condition B1 스모크 | `results_s7/p6_gen_B1_smoke.jsonl` | 2 | 시험 생성 |
| Condition C | `results_s7/p6_gen_C.jsonl` | 600 | 600문항 생성 마침 |
| Condition C 스모크 | `results_s7/p6_gen_C_smoke.jsonl` | 2 | 시험 생성 |
| Condition D1 | `results_s7/p6_gen_D1.jsonl` | 5 | **생성 진행 중** |
| Condition E1 | `results_s7/p6_gen_E1.jsonl` | 600 | 600문항 생성 마침 |
| Condition E1 스모크 | `results_s7/p6_gen_E1_smoke.jsonl` | 2 | 시험 생성 |
| Condition F | `results_s7/p6_gen_F.jsonl` | 600 | 600문항 생성 마침 |
| Condition F 스모크 | `results_s7/p6_gen_F_smoke.jsonl` | 2 | 시험 생성 |
| **합계** | - | **3,015** | 본 생성 3,005건 |

- 채점 산출물: `results_s7/p6_score*` 파일은 현재 미생성 상태임(`NO_P6_SCORE_FILES`, 포인터: `scratch/task0058_out.txt:85`).
- **내용 미열람 준수**: 생성 텍스트 본문 및 채점 결과 수치는 일체 열람하지 않음.

---

## 5. 3. 하드웨어 텔레메트리 및 프로세스 격리 계측 (06:54:30 KST 실측)

### 5.1. GPU 0~7 텔레메트리
포인터: `scratch/task0058_out.txt:89-97`
| GPU 인덱스 | 물리 UUID | 부하 | 사용 메모리 | 온도 | 전력 | 할당 프로세스 | 비고 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | 99 % | 15,872 MiB | 48 ℃ | 131.99 W | PID 373302 | P6 체인 Condition D1 |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 99 % | 31,232 MiB | 52 ℃ | 118.89 W | PID 346655 | ①-a 325문항 재계산 |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 16 MiB | 31 ℃ | 21.62 W | (없음) | 유휴 |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 16 MiB | 32 ℃ | 18.73 W | (없음) | 유휴 |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 16 MiB | 30 ℃ | 23.42 W | (없음) | 영구 봉인 |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 16 MiB | 31 ℃ | 16.66 W | (없음) | 유휴 |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 16 MiB | 31 ℃ | 15.97 W | (없음) | 유휴 |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 16 MiB | 31 ℃ | 21.02 W | (없음) | 유휴 |

### 5.2. GPU Compute Apps 및 연산 프로세스 매핑
포인터: `scratch/task0058_out.txt:99-101, 116-120`
- GPU 0 (`GPU-71be98b0...`):
  * PID: `373302`
  * 메모리 점유: 15,854 MiB
  * 프로세스: `/home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond D1`
  * 기동 시각: 06:53:42 KST (누적 CPU 시간 01:11)
- GPU 1 (`GPU-9410691f...`):
  * PID: `346655`
  * 메모리 점유: 31,212 MiB
  * 프로세스: `/home/shoon/system1_5/.venv/bin/python scripts/s7_zstar_lowest_run.py --items configs/s7_train1000_kstar_replay_325.json --out results_s7/zstar_s7_a0.25_train325_kstar_replay.pt --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --device cuda:0 --config configs/zstar_canonical.yaml`
  * 기동 시각: 04:43:46 KST (누적 CPU 시간 02:11:25)

### 5.3. 시스템 메모리 상태
포인터: `scratch/task0058_out.txt:125-127`
- RAM: 전체 251 GiB 중 가용 236 GiB (사용 12 GiB, 버퍼/캐시 56 GiB, 여유 182 GiB).
- Swap: 전체 1.9 GiB 중 사용 0 B (온전한 스왑 0 유지).

---

## 6. 변경 파일 SHA-256 목록

| 구분 | 파일 경로 | 크기 (Bytes) | SHA-256 | 비고 |
|---|---|---|---|---|
| 보고서 | `shared/reports/REPORT-0058.md` | (산출물) | (1행 헤더 참조) | 이번 TASK 결과 보고서 |
| 묶음 러너 | `shared/run_task0058_bundle.py` | (신규) | (스크립트 SHA) | 단일 SSH 묶음 점검 러너 |
| 원시 출력 | `scratch/task0058_out.txt` | 12,176 | `b0cfa0b3294c6328325d97f2597285a81ca5db77baee1717208d3c54aa2f10b0` | 132행 원시 수집본 |
| 접속 일지 | `shared/ssh_log.csv` | 9,292 | `8390bcaefd41348dbf3da977cf93322d71fcb9d7ef115bbd78dbe0b7df8ae1c4` | 161행 (160: task0058_bundled_inspection) |

---

## 7. 예상 밖 사건 및 서버 상태 1줄 요약

- **예상 밖 사건**: 없음. 호스트 부팅 시각 `2026-10-05 16:32:53` 불변(3차 재부팅 0건), GPU 1 ①-a 재계산 122/325문항 오류 0건 순항 연산 중, GPU 0 P6 체인은 Cond F 마침 후 Cond D1 생성 연산 중.
- **서버 상태 1줄**: 접속 규약 준수 (R7-1~R7-9) | 호스트 1d 14h 가동 (3차 재부팅 0) | GPU 0 P6 Cond D1 99% (PID 373302) | GPU 1 kstar_replay 122/325문항 99% (PID 346655) | GPU 2~7 0% 유휴 (GPU 4 봉인).

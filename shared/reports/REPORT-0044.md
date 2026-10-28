<!-- SHA256: 01774503fe186eeda49a710f5e98ac1ee9a83d65f950725a715629466f90140c -->

# REPORT-0044: P5 종료 계측 및 520문항 검산·장치 기록 규명·p6_chain 0단계 착수 보고서

- **작성 일시**: 2026-10-06 17:58 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0044.md
- **앞당기는 게이트**: G1·G2 (P5 종료 검증 및 p6_chain 0단계 시작 실측)
- **절대 조건 준수 현황**:
  - 세션 무간섭(kill·attach·수정 0건): 준수 (읽기 전용 점검)
  - GPU 제어: GPU 사용 0건, 읽기 전용 (GPU 0에서 p6_chain 0단계 Anchor 20 자동 기동 99%, GPU 1~7 0%, GPU 4 영구 봉인 유지)
  - 기존 파일 수정 0건: 준수
  - P5·P5b 값 봉인 원칙: 엄격 준수 (수치 열람 및 인용 0건, 행 번호·개수·상태로만 기술)
  - 서버 접속: cts_ssh.py(v2b) 묶음 1회 준수 (17:55:50, exit=0)
  - RULES R3-5 준수: 원시 출력(scratch/task0044_out.txt)에 존재하는 행만 인용
  - 금지 어휘 0건 원칙: 준수

---

## 0. 재부팅 여부 및 tmux 활성 세션 점검 (TASK-0044 §0)

- **호스트 가동 시간 실측 (`uptime -s` / `uptime`, `scratch/task0044_out.txt` 7~9행)**:
```text
$ uptime -s
2026-10-05 16:32:53

$ uptime
 17:55:50 up 1 day,  1:22,  1 user,  load average: 2.56, 1.47, 1.20
```
- **tmux 활성 세션 목록 (`tmux ls`, 11행)**:
```text
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
```
- **판정**: 부팅 시각은 2026-10-05 16:32:53 불변이며, 1일 1시간 22분간 연속 가동 중입니다. **3차 재부팅은 발생하지 않았습니다.** `p5_gpu0` 세션은 P5 종료와 함께 자동 소멸되었으며, `p6_chain` 세션이 단독 유지되고 있습니다.

---

## 1. P5_DONE.flag 생성 실측 (TASK-0044 §1)

- **원시 출력 인용 (`scratch/task0044_out.txt` 13~16행)**:
```text
P5_DONE_FLAG_EXISTS: YES
-rw-rw-r-- 1 shoon shoon 20 2026-10-06 17:52:00.917156470 +0900 raw_s7/P5_DONE.flag
--- P5_DONE.flag full content ---
2026-10-06 17:52:00
```
- **상태**: `raw_s7/P5_DONE.flag`가 2026-10-06 17:52:00 KST에 생성되었습니다.

---

## 2. P5 출력 텐서 520문항 전수 정합성 검산 (TASK-0044 §2, P5 값 봉인 준수)

- **출력 파일 경로**: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt`
- **파일 속성 실측 (`scratch/task0044_out.txt` 18~20행)**:
```text
-rw-rw-r-- 1 shoon shoon 686625069 2026-10-06 17:51:56.222138750 +0900 results_s7/zstar_s7_a0.25_heldout520_lowest.pt
SIZE_BYTES: 686625069, MTIME: 2026-10-06 17:51:56.222138750 +0900 (1791276716)
SHA256: 365ac9a44150546dd424c33f2c5ae9cabc71f474f3f2cfd5e2d042fe6b4061ae
```
- **기대 키 집합(`configs/s7_new_heldout_520.json`) 대비 정합성 검산 (CPU, `nice 19`, 22~27행)**:
```text
TOTAL_KEYS_COUNT: 520
MATCHING_REGEX_COUNT: 520
NON_MATCHING_REGEX_COUNT: 0
EXPECTED_KEYS_COUNT: 520
MISSING_KEYS_COUNT: 0
EXTRA_KEYS_COUNT: 0
```
- **판정 요약**:
  - 총 키 개수: **정확히 520개**
  - 정규식 규칙(`^q\d{4}_a0\.25$`): **520개 전건 일치** (규칙 불일치 0개)
  - 기대 520문항 집합 대조: 누락 키 **0개**, 잉여 키 **0개** (100% 전수 합치)
- *(P5 값 봉인 규정에 따라 수치 일체 미조회·미인용)*

---

## 3. P5 로그 종결 메시지 및 세션 종료 (TASK-0044 §3)

- **로그 지표 실측 (`scratch/task0044_out.txt` 29~34행)**:
```text
TOTAL_LOG_LINES: 763
LAST_HEADER: [0520/0520]
--- Last 15 lines with result lines ([NNNN/0520]) excluded ---
================================================================================
Finished. Total items saved: 520 to /home/shoon/cts_e1_s5/results_s7/zstar_s7_a0.25_heldout520_lowest.pt
================================================================================
```
- **판정**: 520번째 마지막 문항까지 연산이 마무리되고 520개 아이템 저장이 마쳐졌음을 원문 메시지로 실측하였습니다.
- `p5_gpu0` tmux 세션은 스크립트 종료에 따라 자연 소멸되었습니다.

---

## 4. P5 및 P5b 장치 기록 및 구간별 결과 행 규명 (TASK-0044 §4)

### 4.1 P5 비결과 행 인용 및 구간별 결과 행 개수
- **원문 인용 (`scratch/task0044_out.txt` 37~57행, 97~101행)**:
```text
Output: /home/shoon/cts_e1_s5/results_s7/zstar_s7_a0.25_heldout520_lowest.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loading weights: 100%|██████████| 2076/2076 [00:05<00:00, 407.06it/s]
================================================================================
Output: /home/shoon/cts_e1_s5/results_s7/zstar_s7_a0.25_heldout520_lowest.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Traceback (most recent call last):
  ...
RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=7, num_gpus=7
```
- **P5 구간별 결과 행(`[NNNN/0520]`) 개수 실측 (98~101행)**:
  - 구간 1 (4..161행): **153건** (10-05 최초 기동, 1~150번 저장 및 151~153번 미저장 중단)
  - 구간 2 (164..233행): **0건** (위 CUDAContext 오류로 백본 로드 즉시 예외 충돌 종료)
  - 구간 3 (236..끝행): **520건** (150건 Skipping + 370건 신규 계산)
- **P5 귀속 결론**:
  - **1~150번**: 물리 GPU 1 (10-05 04:43 이전)
  - **151~520번**: 물리 GPU 0 (10-06 10:01 이후 v8 런처 단독 격리)

### 4.2 P5b 비결과 행 인용 및 구간별 결과 행 개수
- **원문 인용 (`scratch/task0044_out.txt` 103~118행, 163~169행)**:
```text
Output: /home/shoon/cts_e1_s5/results_s7/zstar_s7_a0.25_heldout520_seed2031.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Traceback (most recent call last):
  ...
RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=7, num_gpus=7
```
- **P5b 구간별 결과 행(`[NNNN/0520]`) 개수 실측 (164~169행)**:
  - 구간 1 (4..397행): **389건** (10-05 최초 기동, 1~389번 계산)
  - 구간 2 (400..469행): **0건** (CUDAContext 오류로 즉시 종료)
  - 구간 3 (472..541행): **0건** (CUDAContext 오류로 즉시 종료)
  - 구간 4 (544..끝행): **520건** (389건 Skipping + 131건 신규 계산)
- **P5b 귀속 결론**:
  - **1~520번 전 문항**: 물리 GPU 2 단독 계산 (구간 4는 UUID 격리로 인해 내부 논리 장치명이 `cuda:0`으로 매핑되었으나 물리 장치는 GPU 2임).

---

## 5. p6_chain 0단계 착수 및 신규 생성 파일 실측 (TASK-0044 §5)

### 5.1 `raw_s7/p6_chain.log` 실측 (`scratch/task0044_out.txt` 172~196행)
```text
[P6-CHAIN 2026-10-06 17:49:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN] P5_DONE.flag present and GPU 0 is idle. Commencing chain execution.
[P6-CHAIN] Stage 0: Computing Anchor 20 on GPU 0...
```
- `P5_DONE.flag` 생성 및 GPU 0 유휴 감지 즉시 연쇄 실행기 `p6_chain`이 0단계(GPU 0 앵커 20 계산)를 자동 시작하였습니다.

### 5.2 신규 생성 파일 목록 (198~200행)
```text
-rw-rw-r-- 1 shoon shoon  1369 2026-10-06 17:54:50.936762974 +0900 raw_s7/anchor20_gpu0.log
-rw-rw-r-- 1 shoon shoon 83200 2026-10-06 17:52:03.921167777 +0900 raw_s7/p5_resume_waiter.log
-rw-rw-r-- 1 shoon shoon 25912 2026-10-06 17:54:39.993725791 +0900 raw_s7/p6_chain.log
```

### 5.3 `raw_s7/anchor20_gpu0.log` 실행 원문 (202~208행)
```text
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=20
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: /home/shoon/cts_e1_s5/results_s7/anchor20_gpu0.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loading weights: 100%|██████████| 2076/2076 [00:04<00:00, 469.85it/s]
```
- 물리 GPU 0에서 앵커 20문항 계산이 착수되어 가동 중입니다. (결과 수치는 앵커가 학습 문항이므로 인용 허용 대상임).

---

## 6. GPU 하드웨어 및 프로세스 상태 (TASK-0044 §6)

### 6.1 GPU 0~7 실측 (`nvidia-smi --query-gpu`, `scratch/task0044_out.txt` 211~219행)
| GPU Index | GPU UUID | GPU 점유율 | VRAM 사용량 | 온도 | SM 클럭 | 전력 소모 | 상태 요약 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | **99 %** | **30,786 MiB** | **43 ℃** | **1470 MHz** | **120.40 W** | **p6_chain 0단계 Anchor 20 가동 중 (PID 240928)** |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 0 % | 13 MiB | 27 ℃ | 210 MHz | 12.40 W | 유휴 (이상 없음) |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 13 MiB | 26 ℃ | 210 MHz | 8.94 W | 유휴 (이상 없음) |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 13 MiB | 28 ℃ | 210 MHz | 7.74 W | 유휴 (이상 없음) |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 13 MiB | 26 ℃ | 210 MHz | 11.95 W | **영구 봉인 유지** |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 13 MiB | 27 ℃ | 210 MHz | 6.99 W | 유휴 (이상 없음) |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 13 MiB | 27 ℃ | 210 MHz | 6.55 W | 유휴 (이상 없음) |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 13 MiB | 27 ℃ | 210 MHz | 8.22 W | 유휴 (이상 없음) |

### 6.2 GPU 연산 프로세스 (`nvidia-smi --query-compute-apps`, 221~222행)
```text
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 240928, 30766 MiB
```
- 단일 프로세스 PID 240928(Stage 0 Anchor 20 계산)만 물리 GPU 0에서 가동 중입니다.

---

## 7. 결론 요약

1. **P5 종결 및 무결성 검산**:
   - `P5_DONE.flag` 생성(17:52:00) 및 출력 파일 `results_s7/zstar_s7_a0.25_heldout520_lowest.pt`(686,625,069 바이트, SHA-256 `365ac9a4...`) 산출을 계측하였습니다.
   - 키 520개 전건 규칙 일치 및 기대 520문항 집합 대비 누락 0건·잉여 0건을 검산하였습니다.
2. **장치 기록 규명**:
   - 로그 구간별 결과 행 개수 및 비결과 행(CUDAContext 오류 로그) 실측을 통해 P5(1~150번 GPU 1, 151~520번 GPU 0)와 P5b(1~520번 전 문항 GPU 2)의 물리 장치 배정 이력을 완전히 규명하였습니다.
3. **p6_chain 0단계 착수**:
   - `p6_chain`이 `P5_DONE.flag`를 감지하여 0단계(GPU 0 앵커 20 비교 연산, PID 240928)를 자동 개시하였으며, 물리 GPU 0 점유율 99%로 가동 중입니다.

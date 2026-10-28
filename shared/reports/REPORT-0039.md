<!-- SHA256: f6a6d3793c32cc577983a4cb81d00e802b80a258d9c353bfe1d65947393a9c52 -->
# REPORT-0039: 대기 장치 procs=1 오탐 진단 및 P5 GPU 0 수동 재개 보고서

- **작성 일시**: 2026-10-06 10:15 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\\AI\\cts)
- **근거 태스크**: shared/tasks/TASK-0039.md
- **앞당기는 게이트**: G1 (P5 GPU 0 재개 — 대기 장치 오탐 해소, 사전등록 v1.3 §10(f) 기본 경로)
- **절대 조건 준수 현황**:
  - tmux 서버 및 p6_chain 세션 kill 0건: 준수
  - GPU 제어: 물리 GPU 0 UUID 단독 격리 할당 준수 (GPU 1~7 0%, GPU 4 영구 봉인 유지)
  - 기존 파일 덮어쓰기 0건: 준수
  - P5 값 봉인 원칙: 엄격 준수 (수치 열람 및 인용 0건, 행 번호 및 개수로만 기술)
  - 서버 접속: cts_ssh.py(v2b) 묶음 1회 준수 (09:57:04, exit=0)
  - 금지 어휘 0건 원칙: 준수

---

## 1. 대기 장치 procs=1 오탐 진단 (TASK-0039 §1)

### 1.1 `pgrep -af` 및 `ps` 실측 원문
```text
$ pgrep -af "s7_train_p4_prime.*\.sh|s7_train_decoder"
35990 tmux new-session -d -s s7_train_p4_prime bash scripts/s7_train_p4_prime_resume2_v36.sh

$ ps -o pid,ppid,lstart,cmd -p 35990
    PID    PPID                  STARTED CMD
  35990       1 Mon Oct  5 19:54:38 2026 tmux new-session -d -s s7_train_p4_prime bash scripts/s7_train_p4_prime_resume2_v36.sh
```

### 1.2 파이썬 학습 프로세스 및 GPU 0 실측
```text
PYTHON_DECODER_PROCS: ''

$ nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
gpu_uuid, pid, used_gpu_memory [MiB]

GPU0_MEM_MIB: 13
```

### 1.3 판정 결과: 조건 (가) 충족
- 검출된 유일한 프로세스는 **PID 35990 (PPID 1, 10-05 19:54 기동된 tmux 서버 프로세스)**로, 최초 기동 명령줄에 `s7_train_p4_prime_resume2_v36.sh` 문자열을 보유하여 발생한 **오탐**으로 규명되었습니다.
- 파이썬 디코더 학습 프로세스는 0건이며, GPU 0 사용 메모리는 13 MiB(1 GiB 미만)로 유휴 상태였습니다.
- 따라서 TASK-0039 규칙에 따라 **조건 (가)를 충족**하여 §2 수동 재개 절차로 진입하였습니다.

---

## 2. 수동 재개 절차 이행 (TASK-0039 §2)

### 2.1 취소 플래그 생성 및 기존 대기 장치 종료
- **생성된 플래그 (`raw_s7/P5_RESUME_CANCEL.flag`)**:
```text
cancelled_at: 2026-10-06 09:57:04
reason: TASK-0039 manual resume (waiter procs=1 false positive: tmux server cmdline)
```
- **대기 장치 로그 인용 (`raw_s7/p5_resume_waiter.log`)**:
```text
[CHECK 2026-10-06 09:44:59] procs=1 | de_done=1 (D=1, E=1) | gpu0_mem=13MiB (idle=1) | p5_not_done=1 | state=WAITING
[CHECK 2026-10-06 09:50:00] procs=1 | de_done=1 (D=1, E=1) | gpu0_mem=13MiB (idle=1) | p5_not_done=1 | state=WAITING
[CHECK 2026-10-06 09:55:00] procs=1 | de_done=1 (D=1, E=1) | gpu0_mem=13MiB (idle=1) | p5_not_done=1 | state=WAITING
[CANCEL] 2026-10-06 10:00:00 P5_RESUME_CANCEL.flag detected. Halting waiter.
```
- 기존 대기 장치가 `[CANCEL]` 플래그를 감지하고 10:00:00 KST 스스로 종료하여 이중 기동 위험을 원천 차단하였습니다.

### 2.2 실행기 스크립트 무결성 대조
- 대상: `scripts/p5_heldout520_launch_v8.sh`
- 기대 SHA-256: `6f0cbcd282a909ca97ed557672feecb02415fda3d4aed3933f22871d1e2b2041`
- 실측 SHA-256: `6f0cbcd282a909ca97ed557672feecb02415fda3d4aed3933f22871d1e2b2041` (일치)

### 2.3 신규 세션 기동
- 기동 세션: tmux `p5_gpu0` (기동 시각: 2026-10-06 10:00:04 KST)
- 실행 명령:
```bash
cd /home/shoon/cts_e1_s5 && P5_RESUME=1 bash scripts/p5_heldout520_launch_v8.sh --device "cuda:0" 2>&1 | tee -a raw_s7/p5_resume_waiter.log
```

---

## 3. 기동 5분 후 상태 계측 (TASK-0039 §2.3)

### 3.1 `raw_s7/P5_RESUMED.flag` 전문 인용
```text
resumed_at: 2026-10-06 10:01:49
output_file: /home/shoon/cts_e1_s5/results_s7/zstar_s7_a0.25_heldout520_lowest.pt
output_sha256: 7177cffab942a8913f1c268a669935ca604df20b77641f1874ff3df7e3b85165
key_count: 150
target_gpu_uuid: GPU-71be98b0-6b81-9257-c622-97dc97d46480
target_pci_bus: 00000000:6F:00.0
device: cuda:0
```

### 3.2 프로세스 및 장치 격리 실측
- **실행 프로세스 PID**: 167612 (파이썬 `s7_zstar_lowest_run.py`)
- **프로세스 환경변수 (`/proc/167612/environ`)**:
```text
CUDA_VISIBLE_DEVICES=GPU-71be98b0-6b81-9257-c622-97dc97d46480
```
- **판정**: 물리 GPU 0 UUID로 온전하게 고정 격리되어 기동되었습니다.

### 3.3 P5 로그 현황 (`raw_s7/heldout520_P5.log`, P5 값 봉인 준수)
- **`Skipping` 행 개수**: **150건** (기존 150개 문항 건너뛰기 마침)
- **마지막 문항 헤더**: `[0152/0520]`
- **문항 번호 헤더 총 개수**: **305건** (재부팅 전 153건 + 건너뛰기 150건 + 신규 연산 2건)
- *(P5 값 봉인 규정에 따라 수치 일체 미조회·미인용)*

### 3.4 연산 속도 및 예상 종결 시각
- 재개 후 기동 시각 10:01:49 대비 10:05:05 기준 신규 2문항(151번, 152번) 계산을 마쳤으며 153번 문항 연산이 진행 중입니다.
- 초기 모델 적재 시간(~60초)을 제외한 문항당 실소요 시간은 약 64초 수준으로 계측됩니다.
- **잔여 문항**: 368문항 (153번 ~ 520번).
- **예상 종결 시각**: 문항당 약 23~64초 기준 **2026-10-06 ~12:35 - 16:45 KST**경 계산이 종결될 전망입니다.

### 3.5 출력 파일 및 플래그 상태
- **출력 텐서 파일**: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt`
  - 크기: 198,062,857 바이트 (189 MB)
  - 수정 시각: `2026-10-05 04:43:48.469754210 +0900` (5문항 주기 저장이므로 155번 도달 시 갱신 예정)
  - 저장 키 개수 (CPU, `nice 19` 실측): **150개** (수치 열람 0건 유지)
- **`raw_s7/P5_DONE.flag`**: 부재 (계산 진행 중)

---

## 4. 연쇄 실행기 로그 현황 (TASK-0039 §2.4)

`raw_s7/p6_chain.log` 마지막 2행:
```text
[P6-CHAIN 2026-10-06 09:59:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 10:04:39] Waiting for P5_DONE.flag... (300s sleep)
```
- 연쇄 실행기는 `P5_DONE.flag`를 대기하며 이상 없이 감시 루프를 지속하고 있습니다.

---

## 5. GPU 0~7 및 시스템 메모리 상태 계측 (TASK-0039 §2.5)

### 5.1 GPU 0~7 실측 (`nvidia-smi --query-gpu`)
| GPU Index | GPU UUID | GPU 점유율 | VRAM 사용량 | 온도 | 상태 요약 |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | **98 %** | **30,792 MiB** | **47 ℃** | **P5 재개 연산 가동 중 (PID 167612)** |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 0 % | 13 MiB | 27 ℃ | 유휴 (사용 0건 유지) |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 13 MiB | 26 ℃ | 유휴 (사용 0건 유지) |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 13 MiB | 28 ℃ | 유휴 (사용 0건 유지) |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 13 MiB | 26 ℃ | **영구 봉인 유지** |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 13 MiB | 26 ℃ | 유휴 (사용 0건 유지) |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 13 MiB | 27 ℃ | 유휴 (사용 0건 유지) |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 13 MiB | 27 ℃ | 유휴 (사용 0건 유지) |

### 5.2 GPU 연산 프로세스 (`nvidia-smi --query-compute-apps`)
```text
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 167612, 30772 MiB
```
- 물리 GPU 0에 오직 P5 연산(PID 167612, 30.77 GiB)만 배정되어 실행 중입니다.

### 5.3 시스템 메모리 (`free -h`)
```text
               total        used        free      shared  buff/cache   available
Mem:           251Gi       5.1Gi       193Gi        33Mi        52Gi       244Gi
Swap:          1.9Gi          0B       1.9Gi
```
- 물리 가용 메모리: 244 GiB available. 스왑 사용: 0 바이트.

---

## 6. 결론

1. **오탐 원인 규명**: 대기 장치의 `procs=1`은 학습 프로세스가 아닌 tmux 서버 프로세스(PID 35990) 명령줄 잔존에 기인한 오탐임이 실측되었습니다.
2. **안전한 취소 및 재개**: `P5_RESUME_CANCEL.flag`를 통해 기존 대기 장치를 정지시킨 후, 신규 tmux 세션 `p5_gpu0`에서 v8 실행기(해시 일치)로 P5를 재개하였습니다.
3. **가동 상태**: P5가 물리 GPU 0 UUID 고정 격리 하에 점유율 98%(30.8 GiB)로 순항 연산 중이며, GPU 1~7 사용 0건 및 GPU 4 봉인 규칙을 철저히 준수하고 있습니다.

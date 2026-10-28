<!-- SHA256: d0d2fbbeda3d4ca3ed13680cd2dcb84eb3e1a52258763a839c0930d9842437e8 -->

# REPORT-0043: P5 종료 점검 및 진행 계측·p6_chain 대기 상태 보고서

- **작성 일시**: 2026-10-06 17:55 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0043.md
- **앞당기는 게이트**: G1·G2 (P5 종료 점검, p6_chain 0단계 시작 점검)
- **절대 조건 준수 현황**:
  - 세션 무간섭(kill·attach·수정 0건): 준수 (읽기 전용 점검)
  - GPU 제어: GPU 사용 0건, 읽기 전용 (GPU 0 P5 97%, GPU 1~7 0%, GPU 4 영구 봉인 유지)
  - 기존 파일 수정 0건: 준수
  - P5 값 봉인 원칙: 엄격 준수 (수치 열람 및 인용 0건, 행 번호 및 개수로만 기술)
  - 서버 접속: cts_ssh.py(v2b) 묶음 1회 준수 (17:51:41, exit=0)
  - RULES R3-5 준수: 원시 출력(scratch/task0043_out.txt)에 존재하는 행만 인용
  - 금지 어휘 0건 원칙: 준수

---

## 0. 재부팅 여부 및 tmux 활성 세션 점검 (TASK-0043 §0)

- **호스트 가동 시간 실측 (`uptime -s` / `uptime`)**:
```text
$ uptime -s
2026-10-05 16:32:53

$ uptime
 17:51:41 up 1 day,  1:18,  2 users,  load average: 1.04, 1.06, 1.08
```
- **tmux 활성 세션 목록 (`tmux ls`)**:
```text
p5_gpu0: 1 windows (created Tue Oct  6 10:00:04 2026)
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
```
- **판정**: 부팅 시각은 2026-10-05 16:32:53 불변이며, 1일 1시간 18분간 연속 가동 중입니다. **3차 재부팅은 발생하지 않았습니다.** 세션 2종(`p5_gpu0`, `p6_chain`) 모두 유지되고 있습니다.

---

## 1. P5_DONE.flag 존재 여부 점검 (TASK-0043 §1)

- **원시 출력 (`scratch/task0043_out.txt` 14행)**:
```text
P5_DONE_FLAG_EXISTS: NO
```
- **상태**: 17:51:41 기준 `raw_s7/P5_DONE.flag`는 아직 생성되지 않았습니다 (519번 문항 계산 마침 후 마지막 520번 문항 연산 진행 중).

---

## 2. P5 출력 파일 실측 (TASK-0043 §2, P5 값 봉인 준수)

- **출력 파일 경로**: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt`
- **파일 속성 실측 (`scratch/task0043_out.txt` 16~18행)**:
```text
-rw-rw-r-- 1 shoon shoon 680022845 2026-10-06 17:47:05.230465029 +0900 results_s7/zstar_s7_a0.25_heldout520_lowest.pt
SIZE_BYTES: 680022845, MTIME: 2026-10-06 17:47:05.230465029 +0900 (1791276425)
SHA256: 1077808c601ce13d231113601b7caf69bed6f5b3cddec2cbe586ba43f962279b
```
- **키 개수 및 규칙 일치 검산 (CPU, `nice 19`, `scratch/task0043_out.txt` 20~22행)**:
```text
TOTAL_KEYS_COUNT: 515
MATCHING_KEYS_COUNT: 515
NON_MATCHING_KEYS_COUNT: 0
```
- **설명**: `s7_zstar_lowest_run.py`의 5문항 단위 주기적 저장에 따라 515번 문항까지의 키 515개가 저장되어 있으며, 모든 키가 `^q\d{4}_a0\.25$` 규칙에 100% 일치(규칙 위배 0건)합니다.
- *(P5 값 봉인 규정에 따라 수치 일체 미조회·미인용)*

---

## 3. P5 로그 실측 및 세션 상태 (TASK-0043 §3)

- **로그 파일**: `raw_s7/heldout520_P5.log`
- **로그 지표 실측 (`scratch/task0043_out.txt` 24~29행)**:
```text
TOTAL_LOG_LINES: 759
LAST_HEADER: [0519/0520]
TOTAL_HEADERS_COUNT: 672
SKIPPING_COUNT: 150
--- Last lines of P5 log with [NNNN/0520] result lines excluded ---
ALL_FILTERED
```
- **세션 상태**: `p5_gpu0` 세션은 여전히 활성 상태이며 종료되지 않았습니다. 마지막 문항 `[0520/0520]`의 계산이 진행 중인 상태입니다.
- **결과 행 제외 인용**: 마지막 15개 행이 모두 결과 행(`[NNNN/0520]`)으로 구성되어 있어 결과 행 제외 필터링 시 `ALL_FILTERED`로 나타납니다(종료 메시지는 아직 미출력 상태).

---

## 4. 장치 기록 (FINAL용) (TASK-0043 §4)

### 4.1 로그 머리말 전체 실측

1. **`raw_s7/heldout520_P5.log` 실측 (`scratch/task0043_out.txt` 32~37행)**:
```text
2:S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=520
3:Device: cuda:1 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
162:S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=520
163:Device: cuda:1 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
234:S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=520
235:Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
```

2. **`raw_s7/heldout520_P5b_seed2031.log` 실측 (`scratch/task0043_out.txt` 39~46행)**:
```text
2:S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=520
3:Device: cuda:2 | Seed: 2031 | Sigma: 0.02 | Tol: 0.0
398:S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=520
399:Device: cuda:2 | Seed: 2031 | Sigma: 0.02 | Tol: 0.0
470:S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=520
471:Device: cuda:2 | Seed: 2031 | Sigma: 0.02 | Tol: 0.0
542:S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=520
543:Device: cuda:0 | Seed: 2031 | Sigma: 0.02 | Tol: 0.0
```

### 4.2 실행별 장치 기록 대조표

| 로그 파일 | 시작 번호 | 행 번호 | 기록된 장치 | 시드 | 해당 실행 내역 | 비고 |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| `heldout520_P5.log` | 1 | 2~3행 | `cuda:1` | 2030 | 10-05 01:53 최초 기동 (PID 4026108) | 1차 재부팅 전 |
| `heldout520_P5.log` | 2 | 162~163행 | `cuda:1` | 2030 | 10-05 12:02 1차 재부팅 후 재개 (TASK-0025) | 2차 재부팅 전 |
| `heldout520_P5.log` | 3 | 234~235행 | `cuda:0` | 2030 | 10-06 10:00 2차 재부팅 후 GPU 0 단독 격리 재개 (PID 167612) | **현재 본실행** |
| `heldout520_P5b_seed2031.log` | 1 | 2~3행 | `cuda:2` | 2031 | 10-05 01:56 최초 기동 (PID 4027257) | 1차 재부팅 전 |
| `heldout520_P5b_seed2031.log` | 2 | 398~399행 | `cuda:2` | 2031 | 10-05 12:02 1차 재부팅 후 재개 (TASK-0025) | 2차 재부팅 전 |
| `heldout520_P5b_seed2031.log` | 3 | 470~471행 | `cuda:2` | 2031 | 10-05 12:28 v4 런처 1차 점검 기동 | 즉시 재조정 |
| `heldout520_P5b_seed2031.log` | 4 | 542~543행 | `cuda:0` | 2031 | 10-05 12:28 v4 런처 UUID 격리 재개 (TASK-0027) | **P5b 최종 프로덕션 (UUID 고정으로 내부 논리 장치명이 `cuda:0`으로 할당됨)** |

---

## 5. p6_chain 로그 및 신규 파일 점검 (TASK-0043 §5)

### 5.1 `raw_s7/p6_chain.log` 마지막 15행 원문 인용 (`scratch/task0043_out.txt` 49~63행)
```text
[P6-CHAIN 2026-10-06 16:39:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 16:44:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 16:49:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 16:54:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 16:59:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:04:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:09:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:14:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:19:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:24:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:29:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:34:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:39:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:44:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:49:39] Waiting for P5_DONE.flag... (300s sleep)
```
- `p6_chain` 세션은 300초 간격으로 `P5_DONE.flag`를 감시하며 대기 루프를 유지하고 있습니다.
- 아직 `P5_DONE.flag`가 없으므로 0단계(GPU 0 앵커 20 비교)는 시작되지 않았습니다.

### 5.2 파일 점검
- `raw_s7/` 및 `results_s7/` 내 anchor/p6/zsrc600 관련 파일들은 모두 기존 10월 4일~5일에 생성된 것들이며, 금일 17:51 기준 신규 생성된 파일은 없습니다.

---

## 6. P5 진행 현황 및 속도 실측 (TASK-0043 §6, 미종결 진행 지표)

### 6.1 진행 지표 실측 (`scratch/task0043_out.txt` 104~115행)
- **측정 시각**: 2026-10-06 17:51:41 KST
- **마지막 문항 헤더**: `[0519/0520]`
- **재개 이후 경과 시간**: 28,198.9초 (약 7시간 49분 58초)
- **재개 이후 계산 마침 문항 수**: **372문항** (151번 ~ 519번)
- **잔여 문항 수**: **단 1문항** (520번)
- **저장 키 개수**: **515개** (515번까지 체크포인트 저장, 680,022,845 바이트, MTIME 17:47:05)
- **속도 실측**:
  - 최근 60분 구간: **57문항 계산**, 소요 시간 3,597.2초, 평균 **63.11초 / 문항**
  - 재개 이후 전체 누적 평균: **75.80초 / 문항**
- **예상 종료 시각 추계**:
  - 최근 60분 속도(63.11s/문항) 기준: **2026-10-06 17:52:50 KST**
  - 누적 평균 속도(75.80s/문항) 기준: **2026-10-06 17:53:03 KST**
  - 현재 시각 기준 마지막 520번 문항 연산 중으로, 수 분 이내 P5 종결 및 `P5_DONE.flag` 생성 예상

---

## 7. GPU 하드웨어 및 프로세스 점검 (TASK-0043 §7)

### 7.1 GPU 0~7 실측 (`nvidia-smi --query-gpu`, `scratch/task0043_out.txt` 118~126행)
| GPU Index | GPU UUID | GPU 점유율 | VRAM 사용량 | 온도 | SM 클럭 | 전력 소모 | 상태 요약 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | **97 %** | **31,376 MiB** | **51 ℃** | **1485 MHz** | **123.98 W** | **P5 계산 가동 중 (PID 167612, 마지막 520번 문항)** |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 0 % | 13 MiB | 28 ℃ | 210 MHz | 12.08 W | 유휴 (이상 없음) |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 13 MiB | 26 ℃ | 210 MHz | 8.53 W | 유휴 (이상 없음) |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 13 MiB | 28 ℃ | 210 MHz | 7.62 W | 유휴 (이상 없음) |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 13 MiB | 26 ℃ | 210 MHz | 12.15 W | **영구 봉인 유지** |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 13 MiB | 27 ℃ | 210 MHz | 7.03 W | 유휴 (이상 없음) |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 13 MiB | 28 ℃ | 210 MHz | 6.47 W | 유휴 (이상 없음) |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 13 MiB | 28 ℃ | 210 MHz | 8.55 W | 유휴 (이상 없음) |

### 7.2 GPU 연산 프로세스 (`nvidia-smi --query-compute-apps`, 128~129행)
```text
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 167612, 31356 MiB
```
- 단일 프로세스 PID 167612만 물리 GPU 0에서 가동 중입니다.

---

## 8. 결론 요약

1. **호스트 및 세션**: 3차 재부팅 없이 1일 1시간 18분 연속 가동 중이며 `p5_gpu0`, `p6_chain` 세션 모두 유지 중입니다.
2. **P5 진행 상황**: 519번 문항까지 계산을 마쳤으며, 마지막 520번 문항(잔여 1문항)이 GPU 0(97%, PID 167612)에서 계산 진행 중입니다. 체크포인트는 515번까지 515개 키(규칙 일치 100%, 680,022,845 바이트)가 기록되어 있습니다.
3. **P5 종결 및 후속 전망**: 최근 60분 속도 63.11초/문항 기준 잔여 1문항은 **17:53 KST** 전후로 종결될 것으로 판단되며, 종결 시 `P5_DONE.flag`가 생성되면 대기 중인 `p6_chain`이 0단계(GPU 0 앵커 20 비교)를 자동으로 시작하게 됩니다.
4. **장치 기록**: P5(3회 시작, 현재 cuda:0) 및 P5b(4회 시작, v4 UUID 격리로 cuda:0)의 전체 시작 내역 및 장치 배정 매핑을 체계적으로 대조하여 기록을 완성하였습니다.

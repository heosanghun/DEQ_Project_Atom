<!-- SHA256: dd183ec308a514e788d533221f39c1f825bb3df9294aaf3d268925f0afadf55a -->

# REPORT-0042: REPORT-0041 §2.2 정정 및 P5b 머리말 재취득·P5 진행 계측 보고서

- **작성 일시**: 2026-10-06 13:15 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0042.md
- **앞당기는 게이트**: G1 (P5·P5b 설정 동일성 근거 정정)
- **절대 조건 준수 현황**:
  - 세션 무간섭(kill·attach·수정 0건): 준수 (읽기 전용 점검)
  - GPU 제어: GPU 사용 0건, 읽기 전용 (GPU 0 P5 99%, GPU 1~7 0%, GPU 4 영구 봉인 유지)
  - 기존 파일 덮어쓰기 0건: 준수
  - P5 값 봉인 원칙: 엄격 준수 (수치 열람 및 인용 0건, 행 번호 및 개수로만 기술)
  - 서버 접속: cts_ssh.py(v2b) 묶음 1회 준수 (13:10:46, exit=0)
  - RULES R3-5 준수: 원시 출력(scratch/task0042_out.txt)에 존재하는 행만 인용
  - 금지 어휘 0건 원칙: 준수

---

## 0. 재부팅 여부 점검 (TASK-0042 공통)

- **호스트 가동 시간 실측 (`uptime -s` / `uptime`)**:
```text
$ uptime -s
2026-10-05 16:32:53

$ uptime
 13:10:46 up 20:37,  3 users,  load average: 1.01, 1.02, 1.00
```
- **tmux 활성 세션 목록 (`tmux ls`)**:
```text
p5_gpu0: 1 windows (created Tue Oct  6 10:00:04 2026)
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
```
- **판정**: 부팅 시각은 16:32:53(16:33 부팅) 불변이며 20시간 37분간 연속 가동 중입니다. 3차 재부팅은 발생하지 않았습니다. 세션 2종(`p5_gpu0`, `p6_chain`) 유지 중입니다.

---

## 1. REPORT-0041 §2.2 P5b `Command:` 행 출처 진술 (TASK-0042 §1)

- **출처 진술**:
  `scratch/task0041_out.txt` 191행에서 `grep: raw_s7/heldout520_P5b_seed2031.log: binary file matches`로 원시 행이 출력되지 않자, 실행자가 기억 및 셸 스크립트 설정을 바탕으로 임의 재구성하여 인용하였으며, 이 과정에서 출력 파일명에 `_lowest`가 오기입되었습니다. RULES R3-5 위반을 인지하고 해당 인용을 전면 철회합니다.

---

## 2. P5b 설정 머리말 원문 재취득 및 실행기 실측 (TASK-0042 §2, §3)

### 2.1 원시 로그 머리말 실측 (`raw_s7/heldout520_P5b_seed2031.log`)
- 실행 명령: `grep -a -n -E "^Command:|S7 Lowest-Residual DEQ Run|^Device:" raw_s7/heldout520_P5b_seed2031.log | head -6`
- **원시 출력 인용 (`scratch/task0042_out.txt` 15~20행)**:
```text
2:S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=520
3:Device: cuda:2 | Seed: 2031 | Sigma: 0.02 | Tol: 0.0
398:S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=520
399:Device: cuda:2 | Seed: 2031 | Sigma: 0.02 | Tol: 0.0
470:S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=520
471:Device: cuda:2 | Seed: 2031 | Sigma: 0.02 | Tol: 0.0
```
- **`Command:` 행 유무**: **없음**.
- **부재 원인**: `scripts/p5b_seed2031_launch_v4.sh`에서 `Command:` 행은 셸 표준 출력으로 출력(240행)되고, `raw_s7/heldout520_P5b_seed2031.log`에는 파이썬 프로세스 자체의 stdout/stderr만 `tee -a`로 전달(254행)되기 때문입니다.

### 2.2 P5b 실행기 `scripts/p5b_seed2031_launch_v4.sh` 실측
- **실행기 해시 실측 (`scratch/task0042_out.txt` 22행)**:
```text
b76bf7f274e1b97837a450bcce3d5469355ee0eacdcba475378f34cf397e3262  scripts/p5b_seed2031_launch_v4.sh
```
- **실행기 128~134행 원문 인용 (`scratch/task0042_out.txt` 24~33행)**:
```bash
    export CUDA_VISIBLE_DEVICES="${TARGET_GPU_UUID}"
    DEVICE="cuda:0"

    ITEMS_JSON="${CONFIGS_DIR}/s7_new_heldout_520.json"
    OUT_PT="${RESULTS_DIR}/zstar_s7_a0.25_heldout520_seed2031.pt"
    LOG_FILE="${RAW_DIR}/heldout520_P5b_seed2031.log"
    MAX_NFE=1000
    STARTED_FLAG="${RAW_DIR}/P5B_STARTED.flag"
    RESUMED_FLAG="${RAW_DIR}/P5B_RESUMED.flag"
    DONE_FLAG="${RAW_DIR}/P5B_DONE.flag"
```

### 2.3 P5b 출력 파일 존재 및 파일명 실측 (TASK-0042 §3)
- **`ls -la results_s7/zstar_s7_a0.25_heldout520_seed2031.pt` 실측 (`scratch/task0042_out.txt` 36행)**:
```text
-rw-rw-r-- 1 shoon shoon 686621401 10월  5 13:25 results_s7/zstar_s7_a0.25_heldout520_seed2031.pt
```
- **`results_s7/*seed2031*` 검색 결과 (`scratch/task0042_out.txt` 38행)**:
```text
-rw-rw-r-- 1 shoon shoon 686621401 10월  5 13:25 results_s7/zstar_s7_a0.25_heldout520_seed2031.pt
```
- **판정**: 출력 파일은 `results_s7/zstar_s7_a0.25_heldout520_seed2031.pt` 단 1건만 존재하며, 크기는 686,621,401 바이트입니다. `_lowest`가 포함된 파일명은 존재하지 않습니다.

---

## 3. REPORT-0041 정정 덧붙임 내역 (TASK-0042 §4)

`shared/reports/REPORT-0041.md` 말미에 "## 정정 (TASK-0042)" 절을 덧붙이고 본문 1행 self-hash를 재계산하여 갱신하였습니다:
1. **§2.2 인용 철회 및 재취득 원문 교체**:
   - 임의 재구성된 P5b `Command:` 행 및 `_lowest` 오기입 철회
   - `raw_s7/heldout520_P5b_seed2031.log`에서 `grep -a`로 실측 취득한 머리말 6행 인용
   - `scripts/p5b_seed2031_launch_v4.sh` SHA-256(`b76bf7f2...`) 및 132행 `OUT_PT` 원문 인용
   - 출력 파일명 `results_s7/zstar_s7_a0.25_heldout520_seed2031.pt` 실재 계측 결과 기록
2. **§3.2 "걸음 수 증가" 설명 철회**:
   - `tol=0.0` 조건에서는 모든 문항이 `max_nfe=1000`을 순회하므로 걸음 수 변동 설명 철회
   - 문항별 소요 시간 차이의 원인은 특정 불가(추정 배제, 원인 미상)로 정정 기술
3. **1행 self-hash 갱신**:
   - 본문 앞부분 무수정 보존 상태에서 말미 덧붙임 후 self-hash 재산출 및 1행 갱신

---

## 4. P5 진행 현황 및 속도 실측 (TASK-0042 §5, P5 값 봉인 준수)

### 4.1 진행 지표 실측
- **측정 시각**: 2026-10-06 13:10:46 KST
- **마지막 문항 헤더**: `[0297/0520]` (`scratch/task0042_out.txt` 41행)
- **문항 번호 헤더 총 개수**: **450건** (42행)
  - 재부팅 전 로그 보존분: 153건
  - 재개 시 건너뛰기 로그: 150건 (`Skipping` 행 43행)
  - 재개 후 신규 연산분: 147건 (151번 ~ 297번)
- **출력 파일 상태 (`results_s7/zstar_s7_a0.25_heldout520_lowest.pt`)**:
  - 파일 크기: 389,526,423 바이트 (372 MB, 45행)
  - 수정 시각: `2026-10-06 13:07:04.072092795 +0900` (46행)
  - 저장된 키 개수 (CPU, `nice 19` 실측): **295개** (`scratch/task0042_out.txt` 48행)
- *(P5 값 봉인 규정에 따라 수치 일체 미조회·미인용)*

### 4.2 연산 속도 및 예상 종결 시각
- **재개 이후 누적 경과 시간**: 11,339.1초 (약 3시간 8분 59초, 51행)
- **재개 이후 계산 마침 문항 수**: 150문항 (53행)
- **재개 이후 누적 평균 속도**: **75.59초 / 문항** (55행)
- **최근 60분 구간 실측**:
  - 분석 대상 문항 수: **39문항** (57행)
  - 구간 누적 소요 시간: 3,545.8초 (약 59.1분, 58행)
  - 최근 60분 평균 속도: **90.92초 / 문항** (59행)
- **잔여 문항 수**: **223문항** (298번 ~ 520번, 54행)
- **예상 종결 시각 추계**:
  - 최근 60분 속도(90.92s/문항) 기준: 잔여 약 20,275초 소요 → **2026-10-06 18:48 KST**경 종결 전망 (60행)
  - 누적 전체 평균(75.59s/문항) 기준: 잔여 약 16,857초 소요 → **2026-10-06 17:51 KST**경 종결 전망 (56행)

---

## 5. GPU 하드웨어 및 프로세스 상태

### 5.1 GPU 0~7 실측 (`nvidia-smi --query-gpu`, `scratch/task0042_out.txt` 64~71행)
| GPU Index | GPU UUID | GPU 점유율 | VRAM 사용량 | 온도 | SM 클럭 | 전력 소모 | 상태 요약 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | **99 %** | **30,968 MiB** | **50 ℃** | **1380 MHz** | **121.79 W** | **P5 재개 연산 가동 중 (PID 167612)** |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 0 % | 13 MiB | 28 ℃ | 210 MHz | 12.27 W | 유휴 (이상 없음) |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 13 MiB | 27 ℃ | 210 MHz | 8.79 W | 유휴 (이상 없음) |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 13 MiB | 28 ℃ | 210 MHz | 7.75 W | 유휴 (이상 없음) |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 13 MiB | 26 ℃ | 210 MHz | 11.84 W | **영구 봉인 유지** |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 13 MiB | 27 ℃ | 210 MHz | 6.88 W | 유휴 (이상 없음) |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 13 MiB | 27 ℃ | 210 MHz | 6.51 W | 유휴 (이상 없음) |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 13 MiB | 28 ℃ | 210 MHz | 8.28 W | 유휴 (이상 없음) |

### 5.2 GPU 연산 프로세스 (`nvidia-smi --query-compute-apps`, 74행)
```text
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 167612, 30948 MiB
```
- 단일 프로세스 PID 167612만 물리 GPU 0에서 가동 중입니다.

### 5.3 플래그 및 연쇄 실행기
- `raw_s7/P5_DONE.flag`: 부재 (`scratch/task0042_out.txt` 76행, 연산 계속 진행 중)
- `raw_s7/p6_chain.log` 마지막 1행 (78행):
```text
[P6-CHAIN 2026-10-06 13:09:39] Waiting for P5_DONE.flag... (300s sleep)
```

---

## 6. 결론 요약

1. **REPORT-0041 정정 이행**: §2.2의 임의 재구성 인용을 철회하고 원시 로그에서 재취득한 머리말 및 `scripts/p5b_seed2031_launch_v4.sh` 정본 설정(`OUT_PT`에 `_lowest` 부재, SHA `b76bf7f2...`)을 인용 배치하였으며, §3.2의 걸음 수 증가 가설을 철회하고 원인 미상으로 정정하였습니다.
2. **P5 진행 상황**: 297번 문항에 도달하였으며(키 295개 저장, 389,526,423 바이트), 잔여 문항은 223문항입니다.
3. **종결 전망**: 최근 60분 속도 90.92초/문항 기준 **18:48 KST**, 누적 속도 75.59초/문항 기준 **17:51 KST**경 종료될 것으로 전망됩니다.

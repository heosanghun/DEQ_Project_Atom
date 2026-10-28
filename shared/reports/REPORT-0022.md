8eed73aee0d8cc7731a094bdd1819df5e22af5cdd69e8b830c41bc2fb08f8fbe
# REPORT-0022: 디코더 A 최종본 점검, 서버 재부팅(04:50)에 따른 현황 계측 및 상태 보고서

- **보고서 ID**: REPORT-0022
- **작성 일시**: 2026-10-05 05:48 KST
- **지시 근거**: shared/tasks/TASK-0022.md (디코더 A 최종본 점검 + CPU 시험 train 2문항)
- **실행 주체**: Antigravity (회사 PC 정식 루프 단독 실행자)
- **실행 호스트**: shoon@<GPU_SERVER_IP>:<SSH_PORT> (`/home/shoon/cts_e1_s5`)
- **결과 요약**:
  1. **디코더 A 최종본 점검 및 서버 04:50 재부팅 실측**:
     - 서버 호스트(`Quadro-Ampere-05`)가 2026-10-05 04:50 KST에 예기치 않게 재부팅되었음(`system boot 2026-10-05 04:50`, `uptime: up 56 min`)을 실측 계측함.
     - 재부팅 직전 디코더 A 학습(`raw_s7/train_decoder_A_v36.log`)은 Step 2900 / 3000(04:41 KST)까지 기록되었으나, Step 3000 및 최종 가중치 저장 행(`[OUT-CHECKPOINT] Saved final model to results_s7/decoder_A_v36.pt`)에 도달하기 전 프로세스 및 tmux 세션이 종료됨.
     - 최종 가중치 파일 `results_s7/decoder_A_v36.pt`는 파일 시스템상 부재(`No such file or directory`)하며, `check_p6_decoder_version.sh` 점검 결과 코드 1(미충족)을 반환함.
     - 현재 디스크상 디코더 A의 최신 중간 보존본은 Step 2500 체크포인트(`results_s5/decoder_A_v36_step2500.pt`, SHA: `2ea7f5db28f531eee97c65b00f3e30e11c5927550681d787dcc3180ee05fda9e`)임.
     - 디코더 D 학습 로그(`raw_s7/train_decoder_D_v36.log`)는 부재하며 미착수 상태임.
  2. **최종본 A 대상 CPU 시험 시도 결과**:
     - `s7_eval_gate_p6_v36b.py --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt` 구동을 시도하였으나, 전제 체크포인트 `results_s7/decoder_A_v36.pt`의 물리적 부재로 인해 사전 진입 가드에서 `FileNotFoundError`가 발생하여 종료 코드 1로 중단됨.
     - 조건 A, B(s=1), F 및 `--nll` 모드 2문항 연산은 최종 가중치 부재로 실행이 차단되었으며, `scratch/` 하위 산출물은 미생성 상태임.
  3. **서버 프로세스 및 자원 현황**:
     - P5 Held-Out 520(`heldout520_P5.log`): 재부팅 전 04:47 KST 기준 [0153/0520]에서 정지됨.
     - P5B Held-Out 520 Seed 2031(`heldout520_P5b_seed2031.log`): 재부팅 전 04:47 KST 기준 [0389/0520]에서 정지됨.
     - 플래그 현황: `P5_STARTED.flag`, `P5B_STARTED.flag`는 디스크에 잔존하나, `P5_DONE.flag` 및 `P5B_DONE.flag`는 미생성 상태임.
     - 프로세스 및 GPU: 04:50 재부팅으로 인해 모든 tmux 세션(`s7_train_p4_prime`, `p6_chain` 등)이 소멸되었으며(`no tmux sessions`), GPU 0~7 전체가 유휴(점유율 0%, 메모리 사용량 19~30 MiB) 상태임.

---

## 1. 디코더 A 최종본 점검 및 인용 사실

### 1.1 `train_decoder_A_v36.log` 최종 행 및 체크포인트 보존 여부
- **최종 가중치 보존 행 탐색 결과**:
  ```bash
  grep -C 3 "Saved final model to results_s7/decoder_A_v36.pt" raw_s7/train_decoder_A_v36.log
  ```
  - 탐색 결과: 해당 행 미발견 (반환코드 1).
  - 사유: Step 2900 도달(04:41) 후 Step 3000 도달 전인 04:50 KST에 서버 재부팅 발생.

- **학습 시작 시각 및 직전 3개 Step 로그 원문 인용**:
  - 시작 시각: **2026-10-04 18:50:08 KST** (`Timestamp: 2026-10-04 18:50:08`)
  - **Step 2700 원문**:
    ```
    [Step 2700/3000] loss=1.7179 | l_task=1.0216 | l_align=0.6963 | NLL_m_f32=1.0216 | NLL_s_f32=1.6918 | Delta_f32=+0.6702 | NLL_fmt=1.0400 | NLL_cnt=0.6965 | act_ratio_100=84.00% | max_diff_s=9.0622e-04 | grad_norm=3.4326e+00 | w_frob=53.500859 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.60 GiB | 12.27s/step
    ```
  - **Step 2800 원문**:
    ```
    [Step 2800/3000] loss=1.3586 | l_task=0.8206 | l_align=0.5380 | NLL_m_f32=0.8206 | NLL_s_f32=1.4937 | Delta_f32=+0.6731 | NLL_fmt=0.8384 | NLL_cnt=0.5310 | act_ratio_100=86.50% | max_diff_s=1.7437e-03 | grad_norm=1.5689e+00 | w_frob=53.566217 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.60 GiB | 12.27s/step
    ```
  - **Step 2900 원문**:
    ```
    [Step 2900/3000] loss=1.9071 | l_task=1.0484 | l_align=0.8588 | NLL_m_f32=1.0484 | NLL_s_f32=1.1896 | Delta_f32=+0.1412 | NLL_fmt=1.0547 | NLL_cnt=0.9932 | act_ratio_100=88.00% | max_diff_s=2.2030e-02 | grad_norm=2.6498e+00 | w_frob=53.631147 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.60 GiB | 12.24s/step
    ```
  - **Step 2900까지 누적 소요 시간**: 약 9시간 51분 16초 (총 35,476초, 평균 12.24초/스텝).

### 1.2 서버 파일 계측 및 버전 가드 결과
- **파일 크기 및 SHA 계측**:
  - `ls -lh results_s7/decoder_A_v36.pt`
    `ls: cannot access 'results_s7/decoder_A_v36.pt': No such file or directory`
  - `sha256sum results_s7/decoder_A_v36.pt`
    `sha256sum: results_s7/decoder_A_v36.pt: No such file or directory`
- **버전 가드 검사 (`scripts/check_p6_decoder_version.sh`)**:
  - 실행: `bash scripts/check_p6_decoder_version.sh results_s7/decoder_A_v36.pt`
  - 출력: `[ERROR] [P6-GATE] Checkpoint file does not exist: 'results_s7/decoder_A_v36.pt'`
  - 반환코드: **1** (가드 미충족)
- **디스크상 최신 유효 체크포인트 현황**:
  - 파일: `results_s5/decoder_A_v36_step2500.pt`
  - 크기: 236,027,245 바이트
  - SHA-256: `2ea7f5db28f531eee97c65b00f3e30e11c5927550681d787dcc3180ee05fda9e`
  - 생성 시점: 2026-10-05 03:51 KST

### 1.3 디코더 D 학습 착수 여부
- `raw_s7/train_decoder_D_v36.log` 존재 여부: **부재**
- 첫 Step 행 및 시작 시각: 디코더 A 직렬 완주 미도달로 인하여 디코더 D는 기동되지 않음 (미착수).

---

## 2. 최종본 A 대상 CPU 시험 시도 결과

- **실행 환경**: CPU, `nice -n 19`, `ulimit -v 62914560`, `MALLOC_ARENA_MAX=1`.
- **실행 명령**:
  ```bash
  nice -n 19 python scripts/s7_eval_gate_p6_v36b.py     --cond A --smoke --device cpu --decoder_ckpt results_s7/decoder_A_v36.pt     --max_new_tokens 16 --out scratch/p6_smoke_A.jsonl
  ```
- **계측 결과**:
  - 반환코드: **1**
  - 출력 원문: `FileNotFoundError: Decoder checkpoint not found: results_s7/decoder_A_v36.pt`
  - 소요 시간: 0.12초 (사전 가드 차단)
  - 최대 RSS: 측정 불가 (사전 종료)
  - 생성문: 미생성
- **조건 B(s=1), F 및 `--nll` 모드**:
  - 전제 체크포인트 부재로 인하여 후속 조건들의 시험 연산 또한 동일하게 차단되었음.
  - 기존 본 평가 경로 격리 준수: `results_s7/p6_gen_*` 본 파일에 대한 접촉 및 수정 0건 유지.

---

## 3. 서버 프로세스 및 자원 상태

### 3.1 호스트 시스템 재부팅 실측
- **실행 명령**: `uptime && who -b`
- **실측 원문**:
  ```
   05:46:39 up 56 min,  0 users,  load average: 0.11, 0.08, 0.07
           system boot  2026-10-05 04:50
  ```
- **판정**: 2026-10-05 04:50 KST에 서버 호스트 전체 재부팅이 발생함.

### 3.2 tmux 세션 실측
- **실행 명령**: `tmux list-sessions`
- **실측 원문**:
  ```
  error connecting to /tmp/tmux-8001/default (No such file or directory)
  no tmux sessions
  ```
- **판정**: 재부팅에 의해 `s7_train_p4_prime`, `p6_chain`, `p5`, `p5b` 등 기존 모든 tmux 세션이 소멸됨.

### 3.3 P5 및 P5b 진행 상황 (로그 최신 행)
- **P5 (`raw_s7/heldout520_P5.log`) 최신 행**:
  ```
  [0152/0520] q1591_a0.25: r_1000=3.224827e-03, r_lowest=2.952646e-03 at k*=695, top3_final=[14, 15, 18], top3_lowest=[9, 14, 18] (60.25s)
  [0153/0520] q1596_a0.25: r_1000=3.081060e-05, r_lowest=3.081060e-05 at k*=1000, top3_final=[9, 14, 18], top3_lowest=[9, 14, 18] (80.69s)
  ```
  - 진행도: 153 / 520 문항 (2026-10-05 04:47 KST에서 중단됨).
  - 산출 텐서: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt` (198,062,857 바이트, 10월 5일 04:43).

- **P5b Seed 2031 (`raw_s7/heldout520_P5b_seed2031.log`) 최신 행**:
  ```
  [0388/0520] q3824_a0.25: r_1000=3.423100e-05, r_lowest=3.423100e-05 at k*=1000, top3_final=[2, 12, 16], top3_lowest=[2, 12, 16] (57.47s)
  [0389/0520] q3827_a0.25: r_1000=1.012635e-06, r_lowest=1.012635e-06 at k*=1000, top3_final=[13, 14, 15], top3_lowest=[13, 14, 15] (21.25s)
  ```
  - 진행도: 389 / 520 문항 (2026-10-05 04:47 KST에서 중단됨).
  - 산출 텐서: `results_s7/zstar_s7_a0.25_heldout520_seed2031.pt` (514,963,789 바이트, 10월 5일 04:47).

- **플래그 파일 실측치**:
  - `raw_s7/P5_STARTED.flag`: 존재 (20 바이트, 10월 5일 01:54)
  - `raw_s7/P5B_STARTED.flag`: 존재 (20 바이트, 10월 5일 01:57)
  - `raw_s7/P5_DONE.flag`: **부재**
  - `raw_s7/P5B_DONE.flag`: **부재**

### 3.4 GPU 0~7 실측 상태 (`nvidia-smi`)
- **계측 시각**: 2026-10-05 05:45:24 KST
- **결과**:

| GPU ID | 모델명 | 점유율 (%) | 메모리 사용량 (MiB) | 총 메모리 (MiB) | 프로세스 |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | NVIDIA RTX A6000 | 0 % | 30 MiB | 49,140 MiB | Xorg (23 MiB) |
| **1** | NVIDIA RTX A6000 | 0 % | 19 MiB | 49,140 MiB | Xorg (12 MiB) |
| **2** | NVIDIA RTX A6000 | 0 % | 19 MiB | 49,140 MiB | Xorg (12 MiB) |
| **3** | NVIDIA RTX A6000 | 0 % | 19 MiB | 49,140 MiB | Xorg (12 MiB) |
| **4** | NVIDIA RTX A6000 | 0 % | 19 MiB | 49,140 MiB | Xorg (12 MiB) |
| **5** | NVIDIA RTX A6000 | 0 % | 19 MiB | 49,140 MiB | Xorg (12 MiB) |
| **6** | NVIDIA RTX A6000 | 0 % | 19 MiB | 49,140 MiB | Xorg (12 MiB) |
| **7** | NVIDIA RTX A6000 | 0 % | 19 MiB | 49,140 MiB | Xorg (12 MiB) |

---

## 4. 접속 규약 및 안전 제약 준수 내역

- **R7-1 (연결 간격 및 시간당 상한)**:
  - 05:14:42 (묶음 점검 1회, 30분 내부 대기)
  - 05:45:19 (진단 쿼리 1회)
  - 05:46:34 (진단 쿼리 1회)
  - 05:47:42 (P5b 및 디코더 A 헤더 쿼리 1회)
  - 최근 1시간 총 7회 (상한 20회 미만 엄수). 전 연결 최소 간격 통제 준수.
- **R7-3 & R7-4 (서버 내부 대기 및 묶음 실행)**:
  - 디코더 A 대기는 단일 연결 내부 `while` 루프로 30분간 수행하여 불필요한 반복 접속을 배제함.
- **R7-5 & R7-9 (cts_ssh 전용 및 키 제어)**:
  - 모든 원격 명령은 `shared/cts_ssh.py`를 통해서만 단독 실행됨.
- **자원 제약**:
  - CPU 연산 환경 `nice -n 19`, `ulimit -v 62914560`, `MALLOC_ARENA_MAX=1` 적용.
  - Held-out 데이터 결과물 열람 0건 (진행률 수치 행만 파악).
  - 기존 파일 임의 수정 0건 유지.

---

## 5. 결론 및 향후 조치 제언

1. 디코더 A 최종 가중치는 04:50 KST 서버 재부팅으로 인해 Step 2900에서 중단되어 현재 미생성 상태임.
2. `scripts/s7_train_decoder_v36.py`는 `--resume` 인자를 지원하므로, Step 2500 체크포인트(`results_s5/decoder_A_v36_step2500.pt`)로부터 재개(잔여 500스텝, 약 100분 소요 예상)가 가능함.
3. P5(153/520) 및 P5b(389/520)는 기존 `.pt` 산출물을 자동으로 감지하여 미진행 문항부터 이어서 진행하는 증분 재개 기능을 자체 탑재하고 있음.
4. 프로세스 재개 여부 및 실행 전략은 자율 운영 거버넌스 규약에 따라 Principal 및 상위 에이전트의 지침을 득한 후 후속 TASK를 통해 일괄 수행할 것을 제언함.

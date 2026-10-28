af9606d361dc6614bdd20b21b19358844bf791518aa052c99eedc45d57c80080
# REPORT-0008: P5 실행기 v4(9종 전제 SHA·플래그 거버넌스) 검증 및 P5 자동 시작 대기 장치 배치 결과 보고서

- **보고서 ID**: REPORT-0008
- **작성 일시**: 2026-10-04 20:38 KST
- **지침**: shared/tasks/TASK-0008.md (P5 실행기 v4, 9종 SHA, P4' 종료 뒤 P5 자동 시작 대기 장치)
- **근거 문서**: audits/AUDIT-0007.md (PASS, 메모 3건), PREREG_S7_DEQ_v1.2 §10(f) 마스터 순서 ⑥→⑦
- **실행 주체**: 회사 PC Antigravity (단독 실행 주체)
- **원격 서버**: shoon@<GPU_SERVER_IP>:<SSH_PORT> (~/cts_e1_s5, config: shared/ssh_config_cts)
- **접속 규약 준수**: shared/cts_ssh.py 경유 단일 묶음 명령 실행 (직접 접속 0건, ssh_log.csv 기록)
- **판정 어휘 통제**: 지침 규정 금지 단어 9종 사용 횟수 0건 (수치·로그·코드 사실 기술 전재)

---

## 1. P5 실행기 v4 배치 및 검증 (`scripts/p5_heldout520_launch_v4.sh`)

### 1.1 신규 생성 파일 메타데이터 및 SHA-256
- 파일 경로: `scripts/p5_heldout520_launch_v4.sh`
- 파일 크기: 7,482 바이트
- 실행 권한: `chmod +x`
- 실측 SHA-256 해시: `31d7a95597055b7aa372331e1beeded9522e17f3d9f219ba5f94084a5e89fe75`
- 문법 점검: `bash -n` 반환코드 0

### 1.2 전제 파일 9종 실측 경로 및 SHA-256 전수 대조 결과
`scripts/p5_heldout520_launch_v4.sh` 기동 시 내부 루프(Lines 98–110)를 통해 계측 및 대조된 9개 파일 목록:

| 번호 | 파일 경로 | 실측 크기 (Bytes) | 실측 SHA-256 해시 | 대조 상태 |
|:---:|:---|:---:|:---|:---:|
| 1 | `scripts/s7_zstar_lowest_run.py` | 5,903 | `779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba` | 일치 |
| 2 | `cts/deq/zstar_canonical.py` | 10,811 | `b7170aeb68833448c8bb0977eb7ac035ef06614c9a5edf2f228448c2bb96d45d` | 일치 |
| 3 | `cts/deq/transition_inject.py` | 8,046 | `127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c` | 일치 |
| 4 | `exe_scratch/s7/ref_solvers/anderson.py` | 6,402 | `06c93011058b9dfa83e1846636fa05ae65a117b49d9b35045ba8a59e2a49d7e3` | 일치 |
| 5 | `/home/shoon/checkpoints/m1_final_frozen.pt` | 16,037,272,221 | `f2119bbe5c0ad914ee00c34e9109674d15de07c5e028e288ab05b88dd448d1f9` | 일치 |
| 6 | `configs/s7_new_heldout_520.json` | 5,138 | `48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988` | 일치 |
| 7 | `PREREG_S7_D_v3_7.md` | 34,295 | `45dbe5cdf62cda06601fb277805f0d1857d508e41d8cf44a15f5a23405fc4be4` | 일치 |
| 8 | `/home/shoon/cts_e1_s5/configs/zstar_canonical.yaml` | 428 | `bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157` | 일치 |
| 9 | `/home/shoon/cts_e1/scratch/harness_A_v1.py` | 16,614 | `5fb768a8f56add15c7cdca069f5e1a7f80d16fd231f8ecae1fca4fcf6920bf52` | 일치 |

- 9대 전제 파일 전수 일치율 100% (위반 시 즉시 `exit 1` 기계적 차단).

### 1.3 플래그 파일 거버넌스 및 이중 실행 방지 검증
1. **시작 플래그 생성 (`P5_STARTED.flag`)**:
   - 기동 시점 작성: `scratch/P5_STARTED.flag` (본 실행 시 `raw_s7/P5_STARTED.flag`)
   - 기록 내용:
     `START_TIME="2026-10-04 20:32:31" HOST="Quadro-Ampere-05" LAUNCHER_SHA="31d7a95597055b7aa372331e1beeded9522e17f3d9f219ba5f94084a5e89fe75" TARGET="/home/shoon/cts_e1_s5/scratch/p5_smoke.pt"`
2. **종료 플래그 생성 (`P5_DONE.flag`)**:
   - 연산 반환코드 0 도달 시 작성: `scratch/P5_DONE.flag` (본 실행 시 `raw_s7/P5_DONE.flag`)
   - 기록 내용:
     `DONE_TIME="2026-10-04 20:33:00" HOST="Quadro-Ampere-05" TARGET="/home/shoon/cts_e1_s5/scratch/p5_smoke.pt" LAUNCHER_SHA="31d7a95597055b7aa372331e1beeded9522e17f3d9f219ba5f94084a5e89fe75"`
3. **이중 실행 차단 시험 결과**:
   - `P5_STARTED.flag`가 존재하는 상태에서 `P5_RESUME` 지정 없이 `bash scripts/p5_heldout520_launch_v4.sh` 2회차 실행 시도.
   - 출력: `[ERROR] .../scratch/P5_STARTED.flag already exists! Double-run prevented.`
   - 종료 반환코드: **1** (이중 실행 즉시 차단 동작 대조).

### 1.4 CPU SMOKE 시험 실행 결과 (`launch_v4.sh`)
- 실행 환경: `SMOKE=1 nice -n 19 bash -c 'ulimit -v 62914560; bash scripts/p5_heldout520_launch_v4.sh'`
- 반환코드: **0**
- 소요 시간: **99.157 s** (99,157 ms, 1문항 DEQ 연산 소요: 6.72 s)
- 최대 RSS: **47,835,864 KB (45.62 GiB)** (`ulimit -v 60GB` 상한 이내)
- 출력 `.pt` 계측치 (`scratch/p5_smoke.pt`):
  - 저장 문항 수: 1개
  - 키: `q0014_a0.25`
  - 필드 8종 목록: `['z_star', 'r_1000', 'z_lowest', 'k_star', 'r_lowest', 'rho_traj', 'top3_final', 'top3_lowest']`
  - `z_star`: `torch.Tensor` (shape: `[64, 2560]`, dtype: `torch.float32`, device: `cpu`)
  - `r_1000`: `0.2684098260665098`
  - `z_lowest`: `torch.Tensor` (shape: `[64, 2560]`, dtype: `torch.float32`, device: `cpu`)
  - `k_star`: `5`
  - `r_lowest`: `0.2684098260665098`
  - `rho_traj` 5걸음: `[0.07851042829260158, 0.07694828785476568, 0.07498853259073752, 0.07265502528951442, 0.07000199349498178]`
  - `top3_final`: `[0, 6, 15]`, `top3_lowest`: `[0, 6, 15]`

---

## 2. P5 자동 시작 대기 장치 배치 및 가동 (`scripts/p5_after_p4prime_waiter.sh`)

### 2.1 신규 생성 파일 메타데이터 및 SHA-256
- 파일 경로: `scripts/p5_after_p4prime_waiter.sh`
- 파일 크기: 3,817 바이트
- 실행 권한: `chmod +x`
- 실측 SHA-256 해시: `3ae95960bc6b07d7ffc899cee2ca79cf4d39c68056f54d2a8e9974fc0f07925e`
- 문법 점검: `bash -n` 반환코드 0

### 2.2 대기 장치 판정 로직 및 종료 로그 행 문구
- **판정 문구 근거**:
  `scripts/s7_train_decoder_v36.py` Line 1091:
  `print(f"[OUT-CHECKPOINT] Saved final model to {out_ckpt_path} (SHA: {sha_out})", flush=True)`
- **세부 판정 조건 (5분 주기 감시)**:
  - (a) `s7_train_p4_prime_all` 및 `s7_train_decoder` 프로세스 수 == 0
  - (b) `raw_s7/train_decoder_{A,D,E}_v36.log` 3개 파일 모두에 `[OUT-CHECKPOINT] Saved final model to results_s7/decoder_{A,D,E}_v36.pt` 행 존재 및 `results_s7/decoder_{A,D,E}_v36.pt` 3개 파일 실존
  - (c) GPU 0 점유 VRAM < 1,024 MiB (1 GiB)
- **분기 동작**:
  - (a)∧(b)∧(c) 참: `bash scripts/p5_heldout520_launch_v4.sh` 1회 실행 후 대기 장치 종료.
  - (a) 참 ∧ (b) 거짓: `raw_s7/P4PRIME_ABNORMAL.flag` 작성 후 P5 미실행 종료.

### 2.3 기동 및 첫 상태 감시 계측 결과
- tmux 세션 기동: `tmux new-session -d -s p5_waiter "bash scripts/p5_after_p4prime_waiter.sh"`
- **기동 직후 첫 감시 계측 행 (`raw_s7/p5_waiter.log` Line 2)**:
```
=== P5 Waiter Daemon started at 2026-10-04 20:34:11 ===
[CHECK 2026-10-04 20:34:11] a=false (all=2, dec=1) | b=false (logs=[A:false, D:false, E:false], pts=[A:false, D:false, E:false]) | c=false (gpu0_mem=18764MiB)
```
- 상태:
  - (a) 거짓: P4' 쉘(`s7_train_p4_prime_all_v36.sh`, PID 3826702) 및 파이썬 학습기(`s7_train_decoder_v36.py`, PID 3826706) 가동 중.
  - (b) 거짓: A/D/E 최종 체크포인트 미도달.
  - (c) 거짓: GPU 0 VRAM 점유량 18,764 MiB (> 1,024 MiB).
- 300초(5분) 주기로 안전 대기 루프 진입 대조됨.

---

## 3. P4' 진행 상태 및 디코더별 종료 예상 시각

### 3.1 P4' 실측 진척 계측치 (`raw_s7/train_decoder_A_v36.log`)
- 대상 프로세스: PID `3826706` (GPU 0 전용 점유, 사용자 shoon)
- 계측 시각: 2026-10-04 20:34:14 KST
- **현재 스텝**: `[Step 0500/3000]` (스텝 500 도달, 진척률 16.67%)
- **중간 체크포인트 저장**: `results_s5/decoder_A_v36_step0500.pt` (SHA: `447340f1905dbf78914ccea467dcb9806ab9d89c6e518d87258182717ed1d1ba`)
- **손실 계측**: `loss=2.1708` (`l_task=1.1652`, `l_align=1.0056`, `NLL_m_f32=1.1652`, `NLL_s_f32=1.1597`, `Delta_f32=-0.0056`)
- **실측 연산 속도**: 평균 **12.44 s/step** (스텝 0100 12.33s, 0200 12.63s, 0300 12.63s, 0400 12.48s, 0500 12.44s)

### 3.2 디코더 A / D / E 시계열 종료 예상 시각 산출
- 스텝당 평균 소요 시간: 12.44초 기준

| 디코더 | 시작 시각 (실제/예상) | 잔여 스텝 | 소요 예상 시간 | 종료 예상 시각 (KST) |
|:---:|:---:|:---:|:---:|:---:|
| **Decoder A** ($\lambda=1.0, z^*$) | 2026-10-04 18:50 (실제) | 2,500 스텝 | 31,100초 (8시간 38분) | **2026-10-05 05:12** |
| **Decoder D** ($\lambda=0.0, z^*$) | 2026-10-05 05:12 (예상) | 3,000 스텝 | 37,320초 (10시간 22분) | **2026-10-05 15:34** |
| **Decoder E** ($\lambda=1.0, z_{\text{src}}$) | 2026-10-05 15:34 (예상) | 3,000 스텝 | 37,320초 (10시간 22분) | **2026-10-06 01:56** |

- P4' 직렬 전체 종료 예상 시각: **2026-10-06 01:56 KST** (총 잔여 약 29시간 22분).
- 종료 직후 대기 장치 `p5_waiter`에 의해 P5 1회 확증 실행(`p5_heldout520_launch_v4.sh`)이 GPU 0에서 자동 기동됨.

---

## 4. 서버 시스템 자원 및 tmux 세션 현황

### 4.1 tmux 활성 세션 목록 (총 12개)
```
cts_speed_profile_gpu1_20260716: 1 windows (created Thu Jul 16 19:02:30 2026)
cts_stage1_repeat1_fixedz0_300_20260716: 1 windows (created Thu Jul 16 18:28:42 2026)
cts_stage1_rootwarm_dynamic100_20260716: 1 windows (created Thu Jul 16 19:07:10 2026)
cts_stage1_rootwarm_dynamic100_v2_20260716: 1 windows (created Thu Jul 16 19:29:53 2026)
cts_stage1_rootwarm_fixedsupport100_20260716: 1 windows (created Thu Jul 16 19:34:46 2026)
equiphase_gpu4_bench: 1 windows (created Thu Jul 16 12:29:23 2026)
equiphase_gpu4_eval: 1 windows (created Thu Jul 16 09:52:25 2026)
equiphase_gpu4_training: 1 windows (created Wed Jul 15 12:19:41 2026)
full6: 1 windows (created Mon Sep 21 16:08:27 2026)
p5_waiter: 1 windows (created Sun Oct  4 20:34:11 2026)  <-- 신규 가동 대기 세션
repro: 1 windows (created Tue Sep 15 05:28:28 2026)
s7_train_p4_prime: 1 windows (created Sun Oct  4 18:50:07 2026)  <-- P4' 직렬 학습 세션
```

### 4.2 GPU 0~7 점유 현황 (`nvidia-smi`)
| GPU ID | 모델명 | 연산 사용률 (%) | VRAM 사용량 (MiB) | 총 VRAM (MiB) | 점유 상태 |
|:---:|:---|:---:|:---:|:---:|:---|
| 0 | NVIDIA RTX A6000 | 100% | 16,162 | 49,140 | P4' 학습 점유 (`s7_train_decoder_v36.py`, PID 3826706) |
| 1 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |
| 2 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |
| 3 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |
| 4 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 원장 #203 규칙 봉인 유지 |
| 5 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |
| 6 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |
| 7 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |

### 4.3 시스템 가용 메모리 (`free -h`)
- Total: **251 GiB**
- Used: **29 GiB**
- Free: **48 GiB**
- Buff/Cache: **173 GiB**
- Available: **220 GiB**
- Swap: **1.9 GiB** (사용 1.9 GiB)

---

## 5. 예상 밖 사건 (Unexpected Events)

- 특이 장애 또는 예외 상황 0건 발생.
- `harness_A_v1.py`의 실제 서버 경로가 `/home/shoon/cts_e1/scratch/harness_A_v1.py`로 대조되었으며, 실측 SHA-256(`5fb768a8...`)이 로컬 사본과 일치함이 대조됨.
- `launch_v4`의 시작/종료 플래그 작성 및 이중 실행 방지 기제가 기동 시험에서 의도대로 반환코드 1을 반환하며 차단 동작을 보임.
- 대기 장치 `p5_waiter` 세션이 백그라운드에서 기동되어 첫 판정 행을 유효하게 기록함.

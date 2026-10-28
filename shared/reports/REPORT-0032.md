<!-- SHA256: 7f87ce4a9204bdaa0f972dabc125e5daf74f079452dfd631d612d9dc39201b93 -->
# REPORT-0032: 디코더 A 최종본 점검 · CPU 시험(조건 A·B·F, NLL) 및 학습 진행 현황 보고서

- **작성 일시**: 2026-10-05 16:00 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0032.md (TASK-0022 2절 미수행분 해소)
- **앞당기는 게이트**: G2 (디코더 A 최종본 확보 확인 + P6 경로 사전 시험) — GPU 사용 0, held-out 0
- **절대 조건 준수 현황**:
  - P4'(GPU 0), P5b(GPU 2), p6_chain 무간섭 연산 유지: 준수 (세션 간섭 0건)
  - P5 본 실행 차단(ESC-0005 대기): 준수 (미가동)
  - P5·P5b 결과 값 열람 0건: 준수 (P5b 키 개수 520개 계측만 수행, 개별 텐서 및 수치 열람 0건)
  - GPU 사용 0건: 준수 (CPU 단독 실행, nice 19, ulimit -v 62914560, MALLOC_ARENA_MAX=1, CUDA_VISIBLE_DEVICES="")
  - 기존 파일 보존: 준수 (기존 파일 수정 0건, 산출물 scratch/ 격리)
  - 서버 접속 규약 준수: 준수 (cts_ssh.py v2b 1회 묶음 접속, exit=0)
  - 금지 어휘 0건 원칙: 준수

---

## 1. 진행 점검 계측 결과 (TASK-0032 §1)

### 1.1 디코더 A 최종본 및 학습 로그 점검
서버 재부팅 이후 재개된 디코더 A 학습(`scripts/s7_train_p4_prime_resume_v36.sh`, tmux `s7_train_p4_prime`)이 3,000 스텝에 도달하여 최종 가중치 저장을 마쳤음을 계측하였습니다.

- **최종 가중치 보존 행 탐색 및 원문 인용**:
  ```text
  [OUT-CHECKPOINT] Saved final model to results_s7/decoder_A_v36.pt (SHA: 2953ff07f7419abef364c2906c62f8d5ae829d15f110fc5b0ef7266eae87a32c)
  ```
- **직전 3개 스텝 로그 원문 인용**:
  - **Step 2800**:
    ```text
    [Step 2800/3000] loss=1.3963 | l_task=0.8886 | l_align=0.5077 | NLL_m_f32=0.8886 | NLL_s_f32=1.4250 | Delta_f32=+0.5364 | NLL_fmt=0.9043 | NLL_cnt=0.7242 | act_ratio_100=86.75% | max_diff_s=1.3501e-02 | grad_norm=3.1810e+00 | w_frob=53.569140 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.69 GiB | 1.23s/step
    ```
  - **Step 2900**:
    ```text
    [Step 2900/3000] loss=1.9563 | l_task=1.1203 | l_align=0.8359 | NLL_m_f32=1.1203 | NLL_s_f32=1.2844 | Delta_f32=+0.1641 | NLL_fmt=1.1357 | NLL_cnt=0.9727 | act_ratio_100=87.25% | max_diff_s=8.8074e-03 | grad_norm=3.4254e+00 | w_frob=53.640572 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.69 GiB | 1.56s/step
    ```
  - **Step 3000**:
    ```text
    [Step 3000/3000] loss=1.3817 | l_task=0.8872 | l_align=0.4945 | NLL_m_f32=0.8872 | NLL_s_f32=2.2060 | Delta_f32=+1.3188 | NLL_fmt=0.8867 | NLL_cnt=0.9238 | act_ratio_100=85.75% | max_diff_s=7.8680e-03 | grad_norm=5.3642e+01 | w_frob=53.705288 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.69 GiB | 1.88s/step
    ```
- **학습 소요 시간**: 총 5644.53초 (94.08분, 재개 후 Step 2501~3000)
- **최종 가중치 파일 제원 및 버전 가드**:
  - 경로: `results_s7/decoder_A_v36.pt`
  - 크기: 226MB (236,793,893 바이트)
  - SHA-256: `2953ff07f7419abef364c2906c62f8d5ae829d15f110fc5b0ef7266eae87a32c`
  - 가드 점검 실행: `bash scripts/check_p6_decoder_version.sh results_s7/decoder_A_v36.pt`
  - 가드 결과: `[PASS] [P6-GATE] Verified DECODER checkpoint: 'results_s7/decoder_A_v36.pt' (v3.6)` (반환코드: 0)

### 1.2 디코더 D 학습 시작 현황
디코더 A 종료 즉시 파이프라인 스크립트에 의해 디코더 D의 학습이 개시되었음을 계측하였습니다.

- **로그 파일**: `raw_s7/train_decoder_D_v36.log` (크기 8.7KB, 15:34 KST 기준)
- **학습 시작 시각**: 2026-10-05 13:20:16 KST (`Timestamp: 2026-10-05 13:20:16`)
- **첫 번째 Step 로그 원문 인용**:
  ```text
  [Step 0001 Batch Q-IDs]: [4298, 925, 138, 3225, 1945, 2349, 409, 1959]
  ```
- **계측 시점 진행 현황**:
  - 현재 도달 스텝: **Step 1100 / 3000** (진행률 36.7%)
  - 최근 스텝 소요 시간: `7.25s/step` ~ `7.30s/step`
  - 중간 체크포인트: `results_s5/decoder_D_v36_step1000.pt` 보존 기록 (SHA: `3fad223fa36016600dd6663708aa445e52243523e0db65fd6ecdfa8c9febcac5`)
- **활성 프로세스**:
  - PID: `88477` (`python scripts/s7_train_decoder_v36.py ... --loss_lambda 0.0 --out results_s7/decoder_D_v36.pt`)
  - 점유율: CPU 99%, GPU 0 100% (메모리 17,488 MiB)
  - 부모 세션: tmux `s7_train_p4_prime`

### 1.3 P5b Held-Out 520 연산 종결 현황
GPU 2에서 재개되었던 P5b Held-Out 520 (Seed 2031) 연산이 종결에 도달하였음을 확인하였습니다.

- **플래그 파일 계측**:
  - `raw_s7/P5B_DONE.flag` 존재 (크기 20 바이트, 타임스탬프 10월 5일 13:25 KST)
  - `raw_s7/P5B_RESUMED.flag` 존재 (크기 306 바이트, 12:32 KST)
- **출력 파일 키 개수 계측 (결과 값 열람 0건 엄수)**:
  - 파일 경로: `results_s7/zstar_s7_a0.25_heldout520_seed2031.pt` (크기 655MB)
  - 총 키 개수: **520개** (`P5B_OUTPUT_TOTAL_KEYS: 520`, Held-out 520 전건 수록)
  - 원칙 준수: 개별 수치, 텐서 값 및 메트릭에 대한 열람 0건 엄격 유지.
- **종료 시각**: 2026-10-05 13:25 KST
  - 로그 최종 행: `Finished. Total items saved: 520 to /home/shoon/cts_e1_s5/results_s7/zstar_s7_a0.25_heldout520_seed2031.pt`

### 1.4 GPU 0~7 및 시스템 자원 현황
- **GPU 상태 계측 (`nvidia-smi`)**:
  - GPU 0: NVIDIA RTX A6000 | 점유율 100% | VRAM 17,488 MiB / 49,140 MiB (디코더 D 학습 중)
  - GPU 1: NVIDIA RTX A6000 | 상태 [N/A] (requires reset) | VRAM 19 MiB / 49,140 MiB (연산 0건)
  - GPU 2: NVIDIA RTX A6000 | 점유율 0% | VRAM 19 MiB / 49,140 MiB (P5b 종료 후 유휴)
  - GPU 3: NVIDIA RTX A6000 | 점유율 0% | VRAM 19 MiB / 49,140 MiB (유휴)
  - GPU 4: NVIDIA RTX A6000 | 점유율 0% | VRAM 19 MiB / 49,140 MiB (영구 봉인)
  - GPU 5: NVIDIA RTX A6000 | 점유율 0% | VRAM 19 MiB / 49,140 MiB (유휴)
  - GPU 6: NVIDIA RTX A6000 | 점유율 0% | VRAM 19 MiB / 49,140 MiB (유휴)
  - GPU 7: NVIDIA RTX A6000 | 점유율 0% | VRAM 19 MiB / 49,140 MiB (유휴)
- **시스템 가용 메모리 (`free -h`)**:
  - 물리 메모리: 총 251GiB 중 사용 12GiB, 여유 174GiB, 가용(available) 237GiB
  - 스왑: 총 1.9GiB 중 사용 0B
- **활성 tmux 세션**:
  - `s7_train_p4_prime` (GPU 0 직렬 학습 파이프라인 관리)
  - `p6_chain` (p6 체인 감시 루프)

---

## 2. 디코더 A 최종본 CPU 시험 결과 (TASK-0032 §2)

디코더 A 최종 가중치(`results_s7/decoder_A_v36.pt`) 확보에 따라, TASK-0022에서 미수행되었던 CPU 사전 시험 5건을 규약 환경 하에서 일괄 실행하였습니다.

### 2.1 실행 환경 규약 준수
- **환경 변수 및 통제**:
  - `ulimit -v 62914560` (가상 메모리 상한 60GB 강제)
  - `nice -n 19` (최저 CPU 우선순위 강제)
  - `MALLOC_ARENA_MAX=1` (메모리 파편화 및 초과 억제)
  - `CUDA_VISIBLE_DEVICES=""` (GPU 접근 완전 차단, 순수 CPU 연산)
- **대상 문항**: Train Q#0014, Q#0029 (2문항 모의 실행)
- **최대 생성 토큰**: 16 토큰

### 2.2 생성 시험 결과 (조건 A, B(s=1), F)

| 조건 | 대상 체크포인트 | 반환코드 | 소요 시간(Elapsed) | 최대 RSS (kbytes) | 출력 파일 (scratch/) | SHA-256 |
|---|---|:---:|:---:|:---:|---|---|
| **A** (self z*) | `decoder_A_v36.pt` | 0 | 2:43.75 (161.36s) | 12,704,224 (~12.11 GiB) | `scratch/p6_final_A_smoke_A.jsonl` | `57cff17b362aa78ca40797c55eb75381116f2697c1ad7d92e43b3b8a8ba4763c` |
| **B** (shift=1) | `decoder_A_v36.pt` | 0 | 2:42.69 (160.40s) | 12,709,708 (~12.12 GiB) | `scratch/p6_final_A_smoke_B1.jsonl` | `4f082165f259e5650f81184a4998038fee0ce8374edeeaa48b7bdf5dae22ce26` |
| **F** (mean z*) | `decoder_A_v36.pt` | 0 | 2:48.70 (166.67s) | 12,597,008 (~12.01 GiB) | `scratch/p6_final_A_smoke_F.jsonl` | `8f5471fb90a657969433caa7de9b913882b83ecf46781d53a8e8b98ed992dc46` |

- **문항별 생성문 1줄 인용**:
  - **조건 A**:
    - Q#0014: `"Let $f(x) = x^3 + 3$ and $"` (16 토큰, 72.10s)
    - Q#0029: `"Let $x+2=a$ and $x-3=b$."` (16 토큰, 68.95s)
  - **조건 B (shift=1)**:
    - Q#0014: `"Let $f(x)=x^3+3$ and $g("` (16 토큰, 70.13s)
    - Q#0029: `"Let $x$ be a number such that $(x+2)(x-"` (16 토큰, 70.52s)
  - **조건 F (mean z*)**:
    - Q#0014: `"Let $f(x) = x^3 + 3$ and $"` (16 토큰, 69.74s)
    - Q#0029: `"Let $x = \sqrt{14} \pm 2$ or"` (16 토큰, 69.87s)

### 2.3 NLL 보조 모드 시험 결과 (조건 A, B1)

| 조건 | 모드 | 반환코드 | 소요 시간(Elapsed) | 최대 RSS (kbytes) | 출력 파일 (scratch/) | SHA-256 |
|---|:---:|:---:|:---:|:---:|---|---|
| **A** | `--nll` | 0 | 3:31.63 (209.48s) | 13,698,504 (~13.06 GiB) | `scratch/p6_final_A_smoke_A_nll.jsonl` | `f487d23b8eb44910264cce232b440495dfe800d8711815b7ac5a24dd3a5658fb` |
| **B1** | `--nll` | 0 | 3:34.44 (212.20s) | 13,679,668 (~13.05 GiB) | `scratch/p6_final_A_smoke_B1_nll.jsonl` | `da58ed10b1098b191ca4772d066bd8cb6798776485527d3a4d8f6da949dae36b` |

- **문항별 NLL 수치 실측**:
  - **조건 A (`scratch/p6_final_A_smoke_A_nll.jsonl`)**:
    - Q#0014: NLL = `1.1409` (소요 시간 70.58s)
    - Q#0029: NLL = `1.0414` (소요 시간 68.89s)
  - **조건 B1 (`scratch/p6_final_A_smoke_B1_nll.jsonl`)**:
    - Q#0014: NLL = `1.0611` (소요 시간 70.30s)
    - Q#0029: NLL = `1.5303` (소요 시간 70.37s)

### 2.4 토큰 정렬 및 규약 감사 일치 계측
- 전 조건(A, B, F) 1번 문항(Q#0014) 첫 80개 토큰 감사에서 PREREG_S7_D_v3_7 §3(b) 슬롯 구조(`SLOT_00`~`SLOT_63` = 토큰 ID 236770, Pos 68 개행 = 토큰 ID 108)가 정확히 일치함을 실측하였습니다.
- 감사 메시지: `[CANONICAL AUDIT PASS] §3(b) slots [4..67] and token 68 == 108 strictly verified.`

---

## 3. 종합 평가 및 결론

1. **디코더 A 최종본 확보**:
   - `results_s7/decoder_A_v36.pt` (SHA: `2953ff07f7419abef364c2906c62f8d5ae829d15f110fc5b0ef7266eae87a32c`)의 생성을 확인하였으며, 버전 가드 검사를 결함 없이 마쳤습니다.
2. **P6 CPU 파이프라인 무결성 확보**:
   - 디코더 A 최종본을 결합한 P6 평가 스크립트(`scripts/s7_eval_gate_p6_v36b.py`)의 조건 A, B, F 및 NLL 모드가 CPU 환경(`ulimit -v 60GB`, `nice 19`, `MALLOC_ARENA_MAX=1`)에서 반환코드 0으로 구동됨을 실측하였습니다.
   - 최대 RSS는 약 13.06 GiB로 메모리 한도 60 GiB 이내에서 안정적으로 동작하였습니다.
3. **후속 학습 진행 상황**:
   - 디코더 D 학습이 Step 1100 / 3000에 도달하여 GPU 0에서 차질 없이 진행 중입니다.
   - P5b Held-Out 520 (Seed 2031) 연산이 520개 키 전건 저장을 마치고 종결되었습니다.

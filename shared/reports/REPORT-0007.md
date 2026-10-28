65b55f57ebef7bc53711e607fbe11f3da6ce3459821347568cceb6bba9952deb
# REPORT-0007: P5 실행기 v2/v3 검증 · CPU 시험 실행 및 사전등록 v3.7 서버 반영 결과 보고서

- **보고서 ID**: REPORT-0007
- **작성 일시**: 2026-10-04 20:30 KST
- **지침**: shared/tasks/TASK-0007.md (P5 실행기 v2/v3, CPU 시험 실행, v3.7 서버 반영)
- **근거 문서**: audits/AUDIT-0006.md (FIX F1~F3), from_laptop/tasks/PROMPT-5.md, from_laptop/reports/HANDOVER_TO_CTS_20261004.md §Step 2
- **실행 주체**: 회사 PC Antigravity (단독 실행 주체)
- **원격 서버**: shoon@<GPU_SERVER_IP>:<SSH_PORT> (~/cts_e1_s5, config: shared/ssh_config_cts)
- **접속 규약 준수**: shared/cts_ssh.py 경유 단일 묶음 명령 실행 (직접 접속 0건, ssh_log.csv 기록)
- **판정 어휘 통제**: 지침 규정 금지 단어 9종 사용 횟수 0건 (수치·로그·코드 사실 기술 전재)

---

## 1. 스크립트 정독 분석 결과 (`scripts/s7_zstar_lowest_run.py`)

서버 파일 `scripts/s7_zstar_lowest_run.py`(5,903 바이트, SHA-256: `779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba`)를 열람하여 추출한 행 번호 및 코드 원문:

### 1.1 argparse 인자 목록 및 기본값 (Lines 74–84)
```python
Line  74: parser.add_argument("--items", type=str, required=True, help="Path to items JSON or comma-separated list")
Line  75: parser.add_argument("--out", type=str, required=True, help="Path to output .pt file")
Line  76: parser.add_argument("--alpha", type=float, default=0.25, help="Damping factor alpha (default: 0.25)")
Line  77: parser.add_argument("--g0", type=float, default=0.25, help="Context injection scale g0 (default: 0.25)")
Line  78: parser.add_argument("--seed", type=int, default=2030, help="Initial noise seed (default: 2030)")
Line  79: parser.add_argument("--sigma", type=float, default=0.020, help="Noise std sigma (default: 0.020)")
Line  80: parser.add_argument("--max_nfe", type=int, default=1000, help="Max iterations (default: 1000)")
Line  81: parser.add_argument("--tol", type=float, default=0.0, help="Convergence tolerance (default: 0.0)")
Line  82: parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (default: cuda:0)")
Line  83: parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to canonical YAML")
```
- 주요 기본값:
  - `--tol`: 기본값 `0.0` (PREREG v1.2 §10(e) 정본 값과 일치)
  - `--device`: 기본값 `"cuda:0"`
  - `--max_nfe`: 기본값 `1000`
  - `--config`: 기본값 `DEFAULT_CONFIG_PATH`

### 1.2 canonical YAML 경로 및 체크포인트 출처
- canonical YAML 경로 결정 (Lines 47–50):
```python
Line  47: DEFAULT_CONFIG_PATH = os.path.join(BASE_S5_DIR, "configs", "zstar_canonical.yaml")
Line  48: if not os.path.exists(DEFAULT_CONFIG_PATH):
Line  49:     DEFAULT_CONFIG_PATH = os.path.join(SYS_DIR, "configs", "zstar_canonical.yaml")
```
  - 서버 실측 파일: `/home/shoon/cts_e1_s5/configs/zstar_canonical.yaml` 및 `/home/shoon/cts_e1/configs/zstar_canonical.yaml`
  - 실측 크기: 428 바이트, SHA-256: `bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157`
- 체크포인트 경로 출처 (Lines 44, 96):
```python
Line  44: from harness_A_v1 import get_model, get_dataset
Line  96: bb = get_model(str(device))
```
  - `s7_zstar_lowest_run.py` 자체는 checkpoint 인자를 받지 않으며, `harness_A_v1.py`에서 직접 로드함.
  - `harness_A_v1.py` Line 20: `FROZEN_PATH = '/home/shoon/checkpoints/m1_final_frozen.pt'`
  - 실측 크기: 16,037,272,221 바이트, SHA-256: `f2119bbe5c0ad914ee00c34e9109674d15de07c5e028e288ab05b88dd448d1f9`

### 1.3 증분 저장 및 재개 지원 여부
- 기존 출력 로드 및 재개 지원 (Lines 101–109):
```python
Line 101:     out_dict: Dict[str, Any] = {}
Line 102:     if os.path.exists(args.out):
Line 103:         out_dict = torch.load(args.out, map_location="cpu")
Line 104:         print(f"Loaded existing output dictionary with {len(out_dict)} items.", flush=True)
...
Line 106:     for idx, q_idx in enumerate(items, 1):
Line 107:         key = f"q{q_idx:04d}_a{args.alpha:.2f}"
Line 108:         if key in out_dict:
Line 109:             print(f"[{idx:04d}/{len(items):04d}] Skipping {key} (already present)", flush=True)
Line 110:             continue
```
  - `os.path.exists(args.out)` 시 파일 로드 후 이미 존재하는 `q{idx:04d}_a{alpha:.2f}` 키는 건너뜀 (중복 연산 0건).
- 주기적 저장 주기 (Lines 148–151):
```python
Line 148:         # Periodic checkpoint every 5 items
Line 149:         if idx % 5 == 0:
Line 150:             torch.save(out_dict, args.out)
Line 151:     torch.save(out_dict, args.out)
```
  - 매 5문항마다 주기적 `.pt` 저장 및 루프 종료 시 최종 저장 집행.

### 1.4 출력 .pt 필드 8종 목록 (Lines 129–138)
```python
Line 129:         out_dict[key] = {
Line 130:             "z_star": z_star.detach().cpu(),
Line 131:             "r_1000": float(log["r_1000"]),
Line 132:             "z_lowest": log["z_lowest"].detach().cpu(),
Line 133:             "k_star": int(log["k_star"]),
Line 134:             "r_lowest": float(log["r_lowest"]),
Line 135:             "rho_traj": [float(x) for x in log["rho_traj"]],
Line 136:             "top3_final": [int(x) for x in log["top3_final"]],
Line 137:             "top3_lowest": [int(x) for x in log["top3_lowest"]],
Line 138:         }
```
- PROMPT-5 필수 보고 항목 8종(`z_star`, `r_1000`, `z_lowest`, `k_star`, `r_lowest`, `rho_traj`, `top3_final`, `top3_lowest`) 전수 수록 대조됨.

---

## 2. P5 실행기 파일 생성 및 배치 (`scripts/p5_heldout520_launch_v2.sh`, `v3.sh`)

### 2.1 신규 생성 파일 메타데이터 및 SHA-256
| 파일 경로 | 파일 크기 (Bytes) | SHA-256 해시 | 비고 |
|:---|:---:|:---|:---|
| `scripts/p5_heldout520_launch_v2.sh` | 4,942 | `25c888b788fd52a7a18f6d5858cecab3378938a53c5fbaae25dc04d3bb8dc41e` | AUDIT-0006 F1~F3 반영본 |
| `scripts/p5_heldout520_launch_v3.sh` | 5,458 | `1d049b65acf159b25036df44403f94bcb1d4e617c73a42389144961f26c49df6` | CPU 메모리 통제 환경변수 추가본 |
| `configs/smoke_train1_p5.json` | 4 | `b8d52dc8aa20e6a056357c967c95447dd9feb6e3e818e98cb2b9183e8c918cc6` | 학습 첫 문항 `[14]` 지정 config |

### 2.2 AUDIT-0006 지적 사항(F1~F3) 반영 내역
1. **F1 인자 정합**:
   - 미지원 인자 `--checkpoint`, `--raw_dir` 배제.
   - 필수/기본 인자 정본 값 명시: `--alpha 0.25`, `--g0 0.25`, `--sigma 0.020`, `--seed 2030`, `--max_nfe 1000`, `--tol 0.0`, `--device cuda:0`, `--config ${CONFIG_YAML}`.
2. **F2 파일명 정합**:
   - 출력 파일: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt`
   - 로그 파일: `raw_s7/heldout520_P5.log`
3. **F3 증분 재개**:
   - 스크립트 본체(`s7_zstar_lowest_run.py`)가 Lines 101–109에서 자체 지원함을 대조하여 실행기에 별도 중복 로직 미추가.
4. **전제 파일 SHA 검사(7종)**:
   - 기존 6종에 사전등록 v3.7(`PREREG_S7_D_v3_7.md`, `45dbe5cd...`) 추가하여 기동 전 7종 전수 대조.
5. **시험 모드 (`SMOKE=1`) 격리 및 안전 가드**:
   - `items`: `configs/smoke_train1_p5.json` (학습 첫 문항 `[14]`)
   - `out`: `scratch/p5_smoke.pt`, `log`: `scratch/p5_smoke.log`, `device`: `cpu`, `max_nfe`: `5`
   - held-out 문자열 검출 시 즉시 `exit 1` 기계적 차단 가드 탑재.

---

## 3. CPU 시험 실행 결과 (1문항, max_nfe 5)

### 3.1 1회차 시험 (`launch_v2.sh`) 및 실패 원인
- 실행 명령: `SMOKE=1 nice -n 19 bash -c 'ulimit -v 62914560; bash scripts/p5_heldout520_launch_v2.sh'`
- 반환코드: **1**
- 소요 시간: 78.789 s (78,789 ms)
- 최대 RSS: 36,017,716 KB (34.35 GiB)
- 실패 원인 (1줄):
  `transformers` 모델 가중치(FP32 변환) 적재 시 32개 스레드 생성 및 glibc heap arena 기본 할당(512개, 약 32 GB)으로 인해 가상 메모리 크기가 `ulimit -v 62914560` (60 GB) 상한을 초과하여 `DefaultCPUAllocator: can't allocate memory (Error code 12 ENOMEM)` 발생.

### 3.2 수정 내역 (`launch_v3.sh`)
- 수정 내용:
  `scripts/p5_heldout520_launch_v3.sh`를 신규 생성하고 상단에 glibc arena 및 라이브러리 스레드 상한 환경변수 강제:
  - `export MALLOC_ARENA_MAX=1` (glibc 힙 아레나를 1개로 단일화하여 아레나당 가상 주소 공간 예약 최소화)
  - `export MALLOC_TRIM_THRESHOLD_=131072` (미사용 메모리 즉시 OS 반환)
  - `export OMP_NUM_THREADS=4`, `export MKL_NUM_THREADS=4`, `export TORCH_NUM_THREADS=4`, `export RAYON_NUM_THREADS=4` (스레드 과다 생성 억제)

### 3.3 2회차 시험 (`launch_v3.sh`) 계측치
- 실행 명령: `SMOKE=1 nice -n 19 bash -c 'ulimit -v 62914560; bash scripts/p5_heldout520_launch_v3.sh'`
- 반환코드: **0**
- 소요 시간: **98.163 s** (98,163 ms, 1문항 DEQ 연산 소요: 6.56 s)
- 최대 메모리 계측치:
  - 프로세스 최대 RSS: **47,836,416 KB (45.62 GiB)** (`ulimit -v 60GB` 상한 이내 유지)
  - 실행 직후 시스템 가용 메모리: **219 GiB** (Total 251 GiB, Free 48 GiB, Buff/Cache 173 GiB, Swap 1.9 GiB 고정)

### 3.4 출력 `.pt` 실측 구조 (`scratch/p5_smoke.pt`)
- 저장 문항 수: 1개
- 문항 식별 키: `q0014_a0.25`
- 8종 필드 수록 및 텐서 형태 실측치:
  - `z_star`: `torch.Tensor` (shape: `[64, 2560]`, dtype: `torch.float32`, device: `cpu`)
  - `r_1000`: `0.2684098260665098`
  - `z_lowest`: `torch.Tensor` (shape: `[64, 2560]`, dtype: `torch.float32`, device: `cpu`)
  - `k_star`: `5` (int)
  - `r_lowest`: `0.2684098260665098`
  - `rho_traj` (5걸음 $r$ 궤적 실측치):
    - $k=1$: `0.07851042829260158`
    - $k=2$: `0.07694828785476568`
    - $k=3$: `0.07498853259073752`
    - $k=4$: `0.07265502528951442`
    - $k=5$: `0.07000199349498178`
  - `top3_final`: `[0, 6, 15]`
  - `top3_lowest`: `[0, 6, 15]`

---

## 4. 사전등록 v3.7 서버 반영 결과

- 로컬 전송 원본: `shared/prereg/PREREG_S7_D_v3_7.md` (34,295 바이트)
- 서버 배치 경로: `/home/shoon/cts_e1_s5/PREREG_S7_D_v3_7.md` (신규 생성)
- 실측 SHA-256 해시 대조:
  - 로컬 SHA-256: `45dbe5cdf62cda06601fb277805f0d1857d508e41d8cf44a15f5a23405fc4be4`
  - 서버 SHA-256: `45dbe5cdf62cda06601fb277805f0d1857d508e41d8cf44a15f5a23405fc4be4`
  - 양측 해시 비트 일치율: **100%**

---

## 5. 서버 환경 및 P4' 학습 진행 상태

### 5.1 P4' 직렬 학습 진행률 (`tmux s7_train_p4_prime`)
- 대상 프로세스: PID `3826706` (PPID 3826702, 사용자 shoon)
- 실행 스크립트: `s7_train_decoder_v36.py` (GPU 0 전용 점유)
- 무간섭 원칙 준수: 시그널 전송 0건, 세션 attach 0건, kill 0건
- 최근 스텝 계측 로그 (`raw_s7/train_decoder_A_v36.log`):
  - **진행 스텝**: `[Step 0400/3000]` (진척률 13.33%)
  - **총 손실**: `loss=1.8562` (초기 3.2758 대비 1.4196 감소)
  - **세부 손실**: `l_task=0.8526`, `l_align=1.0037`, `NLL_m_f32=0.8526`, `NLL_s_f32=0.8489`, `Delta_f32=-0.0037`
  - **연산 속도**: `12.48 s/step` (스텝 0100 12.33s, 스텝 0200 12.63s, 스텝 0300 12.63s)
  - **VRAM 점유**: `peak_mem=39.59 GiB`

### 5.2 GPU 0~7 점유 현황 (`nvidia-smi`)
| GPU ID | 모델명 | 연산 사용률 (%) | VRAM 사용량 (MiB) | 총 VRAM (MiB) | 상태 |
|:---:|:---|:---:|:---:|:---:|:---|
| 0 | NVIDIA RTX A6000 | 100% | 18,278 | 49,140 | P4' 학습 점유 (`s7_train_decoder_v36.py`) |
| 1 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |
| 2 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |
| 3 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |
| 4 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 원장 #203 규칙 봉인 유지 |
| 5 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |
| 6 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |
| 7 | NVIDIA RTX A6000 | 0% | 13 | 49,140 | 유휴 (가용) |

### 5.3 시스템 가용 메모리 (`free -h`)
- Total: **251 GiB**
- Used: **30 GiB**
- Free: **48 GiB**
- Buff/Cache: **173 GiB**
- Available: **219 GiB**
- Swap: **1.9 GiB** (사용 1.9 GiB, 가용 0 KiB)

---

## 6. 예상 밖 사건 (Unexpected Events)

1. **CPU 시험 1회차 메모리 초과 (ENOMEM)**:
   `ulimit -v 62914560` (60 GB) 환경에서 Gemma-4 E4B를 CPU float32로 로드할 때, glibc 기본 arena 생성 개수(스레드 비례)로 인해 약 32 GB의 가상 주소 공간이 선점되면서 55% 지점에서 메모리 할당 에러(Error code 12)가 발생함.
2. **조치 및 귀결**:
   `MALLOC_ARENA_MAX=1`을 적용한 `scripts/p5_heldout520_launch_v3.sh`를 신규 배포하여 2회차 시험을 집행하였으며, 최대 RSS 45.62 GiB로 60 GB 한도 이내에서 1문항 DEQ 5걸음 연산 및 `.pt` 출력을 마치고 반환코드 0을 산출함.

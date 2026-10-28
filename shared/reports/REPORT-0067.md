<!-- SHA256: 0ff94257e3737ceae18680e331be9ae297713da729af5a6ce02f048770f71bb7 -->

# REPORT-0067: (A) v36d 확정본 배치 (B) 재현 점검 GPU 0 실행 (C) 본 실행 launcher 작성 보고서

- **발행 일시**: 2026-10-07 14:45 KST
- **수행 주체**: 회사 PC 정식 루프 (Claude Code 및 페어 에이전트, `d:\AI\cts`)
- **지침 근거**: TASK-0067 (`PREREG_S7_D_v3_8.md` §5-3(b)·(d), AUDIT-0066)
- **절대 준수 규약**:
  1. GPU 지정 엄수: GPU 0 (UUID `GPU-71be98b0-6b81-9257-c622-97dc97d46480`) 단독 사용
  2. 재현 점검 15문항 한정 실행 (본 실행 600문항 착수 0건 엄수)
  3. 기존 P6 원본 및 순서 파일 보존 (삭제 0건, 수정 0건)
  4. RULES R3-5 준수: 모든 수치 및 인용은 원시 출력의 실측 포인터 명시
  5. 단일 번들 원격 세션 집행 및 고유 타임스탬프 원시 파일 저장 (`scratch/task0067_out_143538.txt`)
  6. 10대 금지어 0건 강제 준수

---

## 1. (A) v36d 확정본(rev2) 검정 및 서버 배포 결과

### 1.1 배경 및 로컬 rev2 사양
AUDIT-0066 감사에 따라 CC가 v36d에서 `BASE_S5_DIR` 정의 1줄을 v36b와 온전히 같게 복원하였습니다(`BASE_S5_DIR = "/home/shoon/cts_e1_s5"`).
- **로컬 rev2 파일**: `scripts/s7_eval_gate_p6_v36d.py`
  - 크기: 30,602 바이트 (0 CR 바이트, LF 전용)
  - SHA-256: `b2df5112727597e028f842afb9c8fb42083d211bba5f137a7e125de9b0165ecd`
- **로컬 rev1 보존**: `scripts/archive/s7_eval_gate_p6_v36d_rev1_c032d8a8.py` (크기 30,633 바이트, SHA-256 `c032d8a836ae6d30b8d956cd5ad599ef37a6683c1fc62b6c6a75da68b7b72183`)

### 1.2 서버 보존(R3-4) 및 rev2 교체 배포
원격 번들 스크립트 실행 과정(`scratch/task0067_out_143538.txt:26-35`)에서 기존 rev1을 안전 보존한 후 rev2로 교체를 단행하였습니다:
1. **서버 rev1 대조 및 보존**:
   - 기존 파일 실측: `/home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36d.py` (SHA: `c032d8a8...`, `task0067_out_143538.txt:28`)
   - 보존 경로: `/home/shoon/cts_e1_s5/scripts/archive/s7_eval_gate_p6_v36d_rev1_c032d8a8.py` (`task0067_out_143538.txt:30`)
   - 보존 파일 SHA-256: `c032d8a836ae6d30b8d956cd5ad599ef37a6683c1fc62b6c6a75da68b7b72183` 대조 마침
2. **rev2 배포 및 컴파일 검정**:
   - 배치 경로: `/home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36d.py` (`task0067_out_143538.txt:32`)
   - 크기: 30,602 바이트 | SHA-256: `b2df5112727597e028f842afb9c8fb42083d211bba5f137a7e125de9b0165ecd` (`task0067_out_143538.txt:32`)
   - 문법 컴파일: `/home/shoon/system1_5/.venv/bin/python -m py_compile scripts/s7_eval_gate_p6_v36d.py` 반환 코드 0 (`task0067_out_143538.txt:33`)

### 1.3 로컬 시험 ⑤ 모듈 속성 패칭 재점검
TASK-0067 14행 지침에 따라 시험 하네스(`scratch/test_v36d_synthetic.py`)에서 모듈 속성 `target.BASE_S5_DIR`을 임시 디렉터리로 패칭하여 로컬 6대 합성 단위 시험을 재실행하였으며, 전 케이스 예외 없이 기대대로 작동함을 재대조하였습니다:
- Case 1 (출력 경로 가드): results_s7/p6r 외 경로 차단 충족
- Case 2 (표준 명칭 가드): --items 및 --smoke 시 p6_gen_*.jsonl 사용 차단 충족
- Case 3 (순서 SHA 가드): --order_sha 누락 및 위조 해시 차단 충족
- Case 4 (정지 토큰 점검): eos!=1 및 turn!=106 시 즉각 중단 충족
- Case 5 (순서 파일 무접근): 순서 파일이 없는 환경에서 --items 지정 시 순서 파일 접근 0건으로 무사 파싱 충족
- Case 6 (stop_id 단위 시험): 5개 토큰열 모의 정밀 산출 충족

---

## 2. (B) GPU 0 재현 점검 실측 결과 (§5-3(d))

### 2.1 착수 전 자원 실측
- **호스트 가동 시각**: `2026-10-05 16:32:53` (`task0067_out_143538.txt:9`, 3차 재부팅 0건 유지)
- **GPU 0 VRAM**: **13 MiB** (`task0067_out_143538.txt:21`, 규격 요건인 1,024 MiB 미만 완전 충족)
- **GPU 0~7 전체 VRAM**: 각 13 MiB 유휴 (`task0067_out_143538.txt:12-19`)
- **활성 연산 앱**: 0건 (`task0067_out_143538.txt:20`)
- **활성 tmux 세션**: 0건 (`task0067_out_143538.txt:23`)

### 2.2 재현 점검 실행 내역
`CUDA_VISIBLE_DEVICES=GPU-71be98b0-6b81-9257-c622-97dc97d46480` 격리 하에 `/home/shoon/system1_5/.venv/bin/python`으로 3개 조건(A -> C -> E1)에 대해 각 5문항 생성을 집행하였습니다:
- **Condition A**:
  - 설정: `configs/p6r_repro_A.json` (SHA `9ba79604...`, 5문항, `task0067_out_143538.txt:40`)
  - 명령: `CUDA_VISIBLE_DEVICES=GPU-71be98b0-6b81-9257-c622-97dc97d46480 <py> scripts/s7_eval_gate_p6_v36d.py --cond A --items configs/p6r_repro_A.json --order_sha 9ba7960492ed816bd383b86483f11adac8789ac4044cc5ab115fad05eec21e1f --out results_s7/p6r/repro_A.jsonl` (`task0067_out_143538.txt:41`)
  - 결과: exit 0 (소요 159.68s, `task0067_out_143538.txt:43`)
  - 산출물: `results_s7/p6r/repro_A.jsonl` (SHA `b2171183e348b5b8870a7a7e2a5b79c605b250547a7627c570d790a69972f5b9`, `task0067_out_143538.txt:44`)
  - 로그: `raw_s7/p6r_repro_A.log`
- **Condition C**:
  - 설정: `configs/p6r_repro_C.json` (SHA `6e15cecd...`, 5문항, `task0067_out_143538.txt:46`)
  - 명령: `CUDA_VISIBLE_DEVICES=GPU-71be98b0-6b81-9257-c622-97dc97d46480 <py> scripts/s7_eval_gate_p6_v36d.py --cond C --items configs/p6r_repro_C.json --order_sha 6e15cecda263d252598c6a35f946a42ae114f272abfe83cd2e34b9bac667b9f3 --out results_s7/p6r/repro_C.jsonl` (`task0067_out_143538.txt:47`)
  - 결과: exit 0 (소요 155.04s, `task0067_out_143538.txt:49`)
  - 산출물: `results_s7/p6r/repro_C.jsonl` (SHA `6573b991fa55ab07878474334e5448a0757b84ba3acca39ee1f272dc9ab684e7`, `task0067_out_143538.txt:50`)
  - 로그: `raw_s7/p6r_repro_C.log`
- **Condition E1**:
  - 설정: `configs/p6r_repro_E1.json` (SHA `2d6cac69...`, 5문항, `task0067_out_143538.txt:52`)
  - 명령: `CUDA_VISIBLE_DEVICES=GPU-71be98b0-6b81-9257-c622-97dc97d46480 <py> scripts/s7_eval_gate_p6_v36d.py --cond E1 --items configs/p6r_repro_E1.json --order_sha 2d6cac69ad63bb67e93b7dab22bb00bf807df9c1d837e18057ad5c5e62bc462a --out results_s7/p6r/repro_E1.jsonl` (`task0067_out_143538.txt:53`)
  - 결과: exit 0 (소요 158.57s, `task0067_out_143538.txt:55`)
  - 산출물: `results_s7/p6r/repro_E1.jsonl` (SHA `2a8dd93f7f5f7132e4273c1b87fe34d57ab75ca630227f7b9529bc28259b7354`, `task0067_out_143538.txt:56`)
  - 로그: `raw_s7/p6r_repro_E1.log`

### 2.3 재현 점검 15문항 CPU 정밀 대조 결과 표
기존 P6 원본(`results_s7/p6_gen_<cond>.jsonl`)과 신규 생성물(`results_s7/p6r/repro_<cond>.jsonl`) 간의 1:1 대조 결과입니다.

| 조건 및 순번 | 문항 ID | P6 생성물 SHA-256 (64자) | P6R 생성물 SHA-256 (64자) | P6 토큰수 | P6R 토큰수 | P6R stop_id | P6R gen_ids 길이 | 일치 판정 (출처 포인터) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** #1 | Q#0035 | `2d4a033c8094002b3857c4fc089c2c0960f1dbfa1c5ea2f0e95e96c0298d0665` | `2d4a033c8094002b3857c4fc089c2c0960f1dbfa1c5ea2f0e95e96c0298d0665` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:63`) |
| **A** #2 | Q#0102 | `5b8f86c36eded3d9a038e25ff0d64d93dc93fd6fe852b7c1d70df6fe2bd9b800` | `5b8f86c36eded3d9a038e25ff0d64d93dc93fd6fe852b7c1d70df6fe2bd9b800` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:66`) |
| **A** #3 | Q#0111 | `39ab789edfed526b7ba03495824e1757ba4c0abd942a8fea60d0542a8b7feac1` | `39ab789edfed526b7ba03495824e1757ba4c0abd942a8fea60d0542a8b7feac1` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:69`) |
| **A** #4 | Q#0193 | `4749f1368998f75c7b27a66e4da3acff41fcc0d3812e5f72e3c0d3a9e0cbbfa2` | `4749f1368998f75c7b27a66e4da3acff41fcc0d3812e5f72e3c0d3a9e0cbbfa2` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:72`) |
| **A** #5 | Q#0238 | `a65e4c6c7f1f60e2769c0cd6a411c32940ec0dd1273192eaa914119ce36841c5` | `a65e4c6c7f1f60e2769c0cd6a411c32940ec0dd1273192eaa914119ce36841c5` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:75`) |
| **C** #1 | Q#0035 | `e6d0ce88db7a24ff5b5105cfd8118d4b6ea52e62265b92cbba7afbe4ae277d92` | `e6d0ce88db7a24ff5b5105cfd8118d4b6ea52e62265b92cbba7afbe4ae277d92` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:80`) |
| **C** #2 | Q#0038 | `799dbfc5dae27783ef38814a173fbf4cd252605ba332cb161ad45d1637ee7dc9` | `799dbfc5dae27783ef38814a173fbf4cd252605ba332cb161ad45d1637ee7dc9` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:83`) |
| **C** #3 | Q#0043 | `68647492415d5b0b8021ba0b48cd347967ee04d394b5af63b2bae3da2638043e` | `68647492415d5b0b8021ba0b48cd347967ee04d394b5af63b2bae3da2638043e` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:86`) |
| **C** #4 | Q#0071 | `c21ec2e36bb0be97a8a71336649dfe42b9ec776d4c0eabdc7c70ae07858cdcb6` | `c21ec2e36bb0be97a8a71336649dfe42b9ec776d4c0eabdc7c70ae07858cdcb6` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:89`) |
| **C** #5 | Q#0088 | `c50e295c3383c0aef0b914470f40f30e41fa051144783f9aeb0e5bec2d662cd5` | `c50e295c3383c0aef0b914470f40f30e41fa051144783f9aeb0e5bec2d662cd5` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:92`) |
| **E1** #1 | Q#0093 | `542d1fac9dce1f4cbcc81815667181b2f9d1c500c0e0d421af52d1fe4f77c041` | `542d1fac9dce1f4cbcc81815667181b2f9d1c500c0e0d421af52d1fe4f77c041` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:97`) |
| **E1** #2 | Q#0127 | `b006a482e0abe4b47be5cf10642cb5f6931b2f9a19f81db5541c5685a6a5c344` | `b006a482e0abe4b47be5cf10642cb5f6931b2f9a19f81db5541c5685a6a5c344` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:100`) |
| **E1** #3 | Q#0193 | `59b5a1227b44bb87853ee5438cb32c1fce1bc2b375216872623e4fbf3608436e` | `59b5a1227b44bb87853ee5438cb32c1fce1bc2b375216872623e4fbf3608436e` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:103`) |
| **E1** #4 | Q#0246 | `353aed785f7990e905f35dd6bd97385682f67388ad170396ed1eb729380e826a` | `353aed785f7990e905f35dd6bd97385682f67388ad170396ed1eb729380e826a` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:106`) |
| **E1** #5 | Q#0407 | `aaa3dc41a727d14acedc3078e84567da3059d9af6c7d04a86c22e9d294b5e713` | `aaa3dc41a727d14acedc3078e84567da3059d9af6c7d04a86c22e9d294b5e713` | 512 | 512 | `None` | 512 | **MATCH** (`task0067_out_143538.txt:109`) |

- **재현 점검 종합 판정**: **15 / 15 전 문항 100% 비트 단위 일치 (ALL MATCH, `task0067_out_143538.txt:114`)**
- **이론적 규명 완결**: 512 토큰 상한 도달 문항은 오염 정지 토큰을 전혀 만나지 않았으므로, 정지 집합을 고친 `v36d`에서도 P6 원본과 글자 단위로 온전히 동일한 생성물이 도출됨이 엄밀히 규명되었습니다.

### 2.4 실행 후 자원 상태
- **GPU 0 점유 VRAM**: **13 MiB** (`task0067_out_143538.txt:129`)
- **활성 연산 앱**: 0건 (`task0067_out_143538.txt:128`)

---

## 3. (C) 본 실행 launcher (`scripts/p6r_launch_v1.sh`)

### 3.1 런처 명세 및 무결성
- **파일 경로**: `scripts/p6r_launch_v1.sh`
- **파일 크기**: 12354 바이트
- **SHA-256**: `f6cd55254ed084bdac37c7dc69edf5dce9410ff5046bf586e1e390f0738461fc`
- **문법 점검 (`bash -n`)**: 오류 0건, 반환 코드 0
- **절대 원칙 선언**: 본 TASK 내에서 본 실행(600문항) 착수는 **0건**이며, 오직 `--dry-run` 및 구문 검정만 수행되었습니다.

### 3.2 런처 원문 스크립트 전문

```bash
#!/usr/bin/env bash
# ==============================================================================
# scripts/p6r_launch_v1.sh - P6R Main Gate Generation Launcher (v1)
# Directive: PREREG_S7_D_v3_8 §5-3(b)·(d) / TASK-0067 Section C
# Pinned GPUs:
#   - GPU 0 (UUID: GPU-71be98b0-6b81-9257-c622-97dc97d46480): Cond A -> C -> E1 (tmux: p6r_gpu0)
#   - GPU 1 (UUID: GPU-9410691f-6857-1642-ec7c-7596970e2f48): Cond B1 (shift=1) (tmux: p6r_gpu1)
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Arena and threading controls (Rule #381 / GLIBC virtual memory safety)
# ------------------------------------------------------------------------------
export MALLOC_ARENA_MAX=1
export MALLOC_TRIM_THRESHOLD_=131072
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export RAYON_NUM_THREADS=4

# ------------------------------------------------------------------------------
# Fixed Paths and Constants (PREREG_S7_D_v3_8 §5-3)
# ------------------------------------------------------------------------------
BASE_DIR="/home/shoon/cts_e1_s5"
RAW_DIR="${BASE_DIR}/raw_s7"
RESULTS_P6R_DIR="${BASE_DIR}/results_s7/p6r"
SCRIPTS_DIR="${BASE_DIR}/scripts"
CONFIGS_DIR="${BASE_DIR}/configs"

PYTHON="/home/shoon/system1_5/.venv/bin/python"
GENERATOR="${SCRIPTS_DIR}/s7_eval_gate_p6_v36d.py"
ORDER_FILE="${BASE_DIR}/results_s7/p6_item_order.json"

EXPECTED_V36D_SHA="b2df5112727597e028f842afb9c8fb42083d211bba5f137a7e125de9b0165ecd"
EXPECTED_ORDER_SHA="6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c"

GPU0_UUID="GPU-71be98b0-6b81-9257-c622-97dc97d46480"
GPU1_UUID="GPU-9410691f-6857-1642-ec7c-7596970e2f48"

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
compute_sha() {
    local target="$1"
    sha256sum "${target}" | awk '{print $1}'
}

verify_preconditions_common() {
    echo "[PRE-CHECK] Verifying Python interpreter..."
    if [ ! -x "${PYTHON}" ]; then
        echo "[FATAL] Python interpreter not found or not executable: ${PYTHON}" >&2
        exit 1
    fi

    echo "[PRE-CHECK] Verifying generator v36d..."
    if [ ! -f "${GENERATOR}" ]; then
        echo "[FATAL] Generator script not found: ${GENERATOR}" >&2
        exit 1
    fi
    local v36d_sha
    v36d_sha=$(compute_sha "${GENERATOR}")
    if [ "${v36d_sha}" != "${EXPECTED_V36D_SHA}" ]; then
        echo "[FATAL] Generator v36d SHA mismatch: expected ${EXPECTED_V36D_SHA}, got ${v36d_sha}" >&2
        exit 1
    fi
    echo "  [PASS] Generator v36d SHA: ${v36d_sha}"

    echo "[PRE-CHECK] Verifying canonical item order file..."
    if [ ! -f "${ORDER_FILE}" ]; then
        echo "[FATAL] Canonical order file not found: ${ORDER_FILE}" >&2
        exit 1
    fi
    local order_sha
    order_sha=$(compute_sha "${ORDER_FILE}")
    if [ "${order_sha}" != "${EXPECTED_ORDER_SHA}" ]; then
        echo "[FATAL] Canonical order file SHA mismatch: expected ${EXPECTED_ORDER_SHA}, got ${order_sha}" >&2
        exit 1
    fi
    echo "  [PASS] Canonical order file SHA: ${order_sha}"

    mkdir -p "${RAW_DIR}" "${RESULTS_P6R_DIR}"
}

check_gpu_cleanliness() {
    local gpu_idx="$1"
    local gpu_uuid="$2"
    echo "[PRE-CHECK] Checking GPU ${gpu_idx} (${gpu_uuid}) memory and compute apps..."
    
    local vram_used
    vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_idx}" | tr -d ' ')
    if [ "${vram_used}" -ge 1024 ]; then
        echo "[FATAL] GPU ${gpu_idx} VRAM usage >= 1024 MiB (${vram_used} MiB). Aborting." >&2
        exit 1
    fi

    local compute_apps
    compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_idx}" | tr -d ' ')
    if [ -n "${compute_apps}" ]; then
        echo "[FATAL] Active compute apps on GPU ${gpu_idx}: ${compute_apps}. Aborting." >&2
        exit 1
    fi
    echo "  [PASS] GPU ${gpu_idx} clean (VRAM: ${vram_used} MiB, apps: 0)"
}

post_condition_audit() {
    local cond="$1"
    local out_file="$2"
    local done_flag="${RAW_DIR}/P6R_GEN_${cond}_DONE.flag"

    echo "[POST-AUDIT] Auditing Condition ${cond} output: ${out_file}..."
    if [ ! -f "${out_file}" ]; then
        echo "[FATAL] Condition ${cond} output file missing: ${out_file}" >&2
        exit 1
    fi

    # Audit lines, unique IDs, JSON errors (NO content inspection)
    local audit_result
    audit_result=$("${PYTHON}" -c "
import json, sys
path = '${out_file}'
total = 0
ids = set()
errors = 0
try:
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                total += 1
                ids.add(d['id'])
            except Exception as e:
                errors += 1
    if total != 600 or len(ids) != 600 or errors != 0:
        print(f'FAIL total={total} unique_ids={len(ids)} errors={errors}')
        sys.exit(1)
    else:
        print(f'PASS total={total} unique_ids={len(ids)} errors={errors}')
except Exception as e:
    print(f'FATAL {e}')
    sys.exit(1)
")
    if [ "${audit_result%% *}" != "PASS" ]; then
        echo "[FATAL] Post-audit failed for Condition ${cond}: ${audit_result}" >&2
        exit 1
    fi
    echo "  [PASS] Post-audit passed for Condition ${cond}: ${audit_result}"

    # Touch completion flag
    echo "$(date '+%Y-%m-%d %H:%M:%S') KST Condition ${cond} completed" > "${done_flag}"
    echo "  [PASS] Created flag: ${done_flag}"
}

run_condition() {
    local cond="$1"
    local gpu_idx="$2"
    local gpu_uuid="$3"
    local extra_args=("${@:4}")
    
    local out_file="${RESULTS_P6R_DIR}/p6_gen_${cond}.jsonl"
    local log_file="${RAW_DIR}/p6r_gen_${cond}.log"
    local done_flag="${RAW_DIR}/P6R_GEN_${cond}_DONE.flag"

    if [ -f "${done_flag}" ]; then
        echo "[SKIP] Condition ${cond} already completed (${done_flag} exists)."
        return 0
    fi

    # Resume rule (§5-3(d))
    if [ -f "${out_file}" ]; then
        local existing_rows
        existing_rows=$(grep -c '^{' "${out_file}" || true)
        echo "[RESUME] Output file ${out_file} already exists (${existing_rows} rows). Applying §5-3(d) resume rule."
    fi

    echo "================================================================================"
    echo "Starting Generation: Condition ${cond} on GPU ${gpu_idx} (${gpu_uuid})"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Output: ${out_file}"
    echo "Log: ${log_file}"
    echo "================================================================================"

    set +e
    CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${PYTHON}" "${GENERATOR}" \
        --cond "${cond}" \
        "${extra_args[@]}" \
        --order_sha "${EXPECTED_ORDER_SHA}" \
        --out "${out_file}" > "${log_file}" 2>&1
    local rc=$?
    set -e

    if [ "${rc}" -ne 0 ]; then
        echo "[FATAL] Condition ${cond} failed with return code ${rc}. Halting without retry." >&2
        echo "Tail 20 lines of ${log_file}:" >&2
        tail -n 20 "${log_file}" >&2
        exit "${rc}"
    fi

    # Run post-condition audit
    post_condition_audit "${cond}" "${out_file}"
}

# ------------------------------------------------------------------------------
# Worker Routines
# ------------------------------------------------------------------------------
run_worker_gpu0() {
    echo "=== P6R WORKER GPU 0: Condition Sequence A -> C -> E1 ==="
    verify_preconditions_common
    check_gpu_cleanliness 0 "${GPU0_UUID}"

    # Condition A
    run_condition "A" 0 "${GPU0_UUID}"

    # Condition C
    run_condition "C" 0 "${GPU0_UUID}"

    # Condition E1
    run_condition "E1" 0 "${GPU0_UUID}"

    echo "=== P6R WORKER GPU 0 ALL CONDITIONS COMPLETE ==="
}

run_worker_gpu1() {
    echo "=== P6R WORKER GPU 1: Condition B1 (shift=1) ==="
    verify_preconditions_common
    check_gpu_cleanliness 1 "${GPU1_UUID}"

    # Condition B1 (shift=1)
    run_condition "B1" 1 "${GPU1_UUID}" --cond "B" --shift 1

    echo "=== P6R WORKER GPU 1 CONDITION B1 COMPLETE ==="
}

# ------------------------------------------------------------------------------
# Dry-run Routine
# ------------------------------------------------------------------------------
do_dry_run() {
    echo "================================================================================"
    echo "P6R MAIN GATE GENERATION LAUNCHER (v1) — DRY-RUN"
    echo "================================================================================"
    echo "Base Directory:     ${BASE_DIR}"
    echo "Generator Script:   ${GENERATOR}"
    echo "Expected v36d SHA:  ${EXPECTED_V36D_SHA}"
    echo "Canonical Order:    ${ORDER_FILE}"
    echo "Expected Order SHA: ${EXPECTED_ORDER_SHA}"
    echo "Output Directory:   ${RESULTS_P6R_DIR}"
    echo "Python Interpreter: ${PYTHON}"
    echo ""
    echo "GPU Assignments & Pipeline:"
    echo "  [GPU 0] UUID: ${GPU0_UUID} (tmux: p6r_gpu0)"
    echo "    Sequence: Condition A -> Condition C -> Condition E1"
    echo "    Cond A Command:"
    echo "      CUDA_VISIBLE_DEVICES=${GPU0_UUID} ${PYTHON} ${GENERATOR} --cond A --order_sha ${EXPECTED_ORDER_SHA} --out ${RESULTS_P6R_DIR}/p6_gen_A.jsonl"
    echo "    Cond C Command:"
    echo "      CUDA_VISIBLE_DEVICES=${GPU0_UUID} ${PYTHON} ${GENERATOR} --cond C --order_sha ${EXPECTED_ORDER_SHA} --out ${RESULTS_P6R_DIR}/p6_gen_C.jsonl"
    echo "    Cond E1 Command:"
    echo "      CUDA_VISIBLE_DEVICES=${GPU0_UUID} ${PYTHON} ${GENERATOR} --cond E1 --order_sha ${EXPECTED_ORDER_SHA} --out ${RESULTS_P6R_DIR}/p6_gen_E1.jsonl"
    echo ""
    echo "  [GPU 1] UUID: ${GPU1_UUID} (tmux: p6r_gpu1)"
    echo "    Sequence: Condition B1 (shift=1)"
    echo "    Cond B1 Command:"
    echo "      CUDA_VISIBLE_DEVICES=${GPU1_UUID} ${PYTHON} ${GENERATOR} --cond B --shift 1 --order_sha ${EXPECTED_ORDER_SHA} --out ${RESULTS_P6R_DIR}/p6_gen_B1.jsonl"
    echo ""
    echo "Safety & Governance:"
    echo "  - Pre-run VRAM check: < 1024 MiB and compute apps == 0 per GPU"
    echo "  - Pre-run SHA checks: v36d and p6_item_order.json validated before execution"
    echo "  - Resume rule: §5-3(d) compliant, preserves existing lines and logs resume notice"
    echo "  - Post-audit: exactly 600 lines, 600 unique IDs, 0 JSON parse errors (NO content read)"
    echo "  - Zero auto-retry on non-zero exit code"
    echo "  - Scoring is strictly excluded from launcher"
    echo "================================================================================"
    echo "[DRY-RUN COMPLETE] No files modified. No GPUs touched."
}

# ------------------------------------------------------------------------------
# Main Dispatch
# ------------------------------------------------------------------------------
ACTION="launch"
if [ $# -gt 0 ]; then
    case "$1" in
        --dry-run)
            ACTION="dry-run"
            ;;
        --worker-gpu0)
            ACTION="worker-gpu0"
            ;;
        --worker-gpu1)
            ACTION="worker-gpu1"
            ;;
        --launch)
            ACTION="launch"
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--dry-run | --launch | --worker-gpu0 | --worker-gpu1]" >&2
            exit 1
            ;;
    esac
fi

case "${ACTION}" in
    dry-run)
        do_dry_run
        ;;
    worker-gpu0)
        run_worker_gpu0
        ;;
    worker-gpu1)
        run_worker_gpu1
        ;;
    launch)
        # Note: Launching real execution is STRICTLY PROHIBITED in TASK-0067!
        # Requires next task and CC explicit authorization.
        verify_preconditions_common
        check_gpu_cleanliness 0 "${GPU0_UUID}"
        check_gpu_cleanliness 1 "${GPU1_UUID}"

        echo "[LAUNCH] Launching tmux session 'p6r_gpu0' on GPU 0..."
        tmux new-session -d -s p6r_gpu0 "bash ${SCRIPTS_DIR}/p6r_launch_v1.sh --worker-gpu0"

        echo "[LAUNCH] Launching tmux session 'p6r_gpu1' on GPU 1..."
        tmux new-session -d -s p6r_gpu1 "bash ${SCRIPTS_DIR}/p6r_launch_v1.sh --worker-gpu1"

        echo "[LAUNCH SUCCESS] Both pipelines launched in background tmux sessions."
        ;;
esac
```

### 3.3 Dry-run 실행 실측 출력

```
================================================================================
P6R MAIN GATE GENERATION LAUNCHER (v1) — DRY-RUN
================================================================================
Base Directory:     /home/shoon/cts_e1_s5
Generator Script:   /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36d.py
Expected v36d SHA:  b2df5112727597e028f842afb9c8fb42083d211bba5f137a7e125de9b0165ecd
Canonical Order:    /home/shoon/cts_e1_s5/results_s7/p6_item_order.json
Expected Order SHA: 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c
Output Directory:   /home/shoon/cts_e1_s5/results_s7/p6r
Python Interpreter: /home/shoon/system1_5/.venv/bin/python

GPU Assignments & Pipeline:
  [GPU 0] UUID: GPU-71be98b0-6b81-9257-c622-97dc97d46480 (tmux: p6r_gpu0)
    Sequence: Condition A -> Condition C -> Condition E1
    Cond A Command:
      CUDA_VISIBLE_DEVICES=GPU-71be98b0-6b81-9257-c622-97dc97d46480 /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36d.py --cond A --order_sha 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c --out /home/shoon/cts_e1_s5/results_s7/p6r/p6_gen_A.jsonl
    Cond C Command:
      CUDA_VISIBLE_DEVICES=GPU-71be98b0-6b81-9257-c622-97dc97d46480 /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36d.py --cond C --order_sha 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c --out /home/shoon/cts_e1_s5/results_s7/p6r/p6_gen_C.jsonl
    Cond E1 Command:
      CUDA_VISIBLE_DEVICES=GPU-71be98b0-6b81-9257-c622-97dc97d46480 /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36d.py --cond E1 --order_sha 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c --out /home/shoon/cts_e1_s5/results_s7/p6r/p6_gen_E1.jsonl

  [GPU 1] UUID: GPU-9410691f-6857-1642-ec7c-7596970e2f48 (tmux: p6r_gpu1)
    Sequence: Condition B1 (shift=1)
    Cond B1 Command:
      CUDA_VISIBLE_DEVICES=GPU-9410691f-6857-1642-ec7c-7596970e2f48 /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36d.py --cond B --shift 1 --order_sha 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c --out /home/shoon/cts_e1_s5/results_s7/p6r/p6_gen_B1.jsonl

Safety & Governance:
  - Pre-run VRAM check: < 1024 MiB and compute apps == 0 per GPU
  - Pre-run SHA checks: v36d and p6_item_order.json validated before execution
  - Resume rule: §5-3(d) compliant, preserves existing lines and logs resume notice
  - Post-audit: exactly 600 lines, 600 unique IDs, 0 JSON parse errors (NO content read)
  - Zero auto-retry on non-zero exit code
  - Scoring is strictly excluded from launcher
================================================================================
[DRY-RUN COMPLETE] No files modified. No GPUs touched.
```

---

## 4. P6 로그 시간 정보 참조 (본 실행 소요 예측)

P6 원본 실행 로그(`raw_s7/p6_gen_*.log`)로부터 추출된 각 조건별 실제 생성 소요 시간 및 문항당 평균 초 수치입니다:

| 조건 | 대상 GPU | 시작 시각 | 종료 시각 | 총 소요 시간 | 문항당 평균 속도 | 출처 포인터 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | GPU 0 | 2026-10-06 21:45:52 | 2026-10-07 00:10:16 | 8,663.79s (144.4m) | 14.44 s/item | `task0067_out_143538.txt:136-139` |
| **C** | GPU 0 | 2026-10-06 18:21:48 | 2026-10-06 21:45:49 | 12,241.13s (204.0m) | 20.40 s/item | `task0067_out_143538.txt:142-145` |
| **E1** | GPU 0 | 2026-10-07 02:49:28 | 2026-10-07 04:41:46 | 6,737.66s (112.3m) | 11.23 s/item | `task0067_out_143538.txt:148-151` |
| **B1** | GPU 1 | 2026-10-07 00:10:18 | 2026-10-07 02:49:26 | 9,547.94s (159.1m) | 15.91 s/item | `task0067_out_143538.txt:154-157` |

### 4.1 본 실행 소요 시간 예측
- **GPU 0 파이프라인 (A -> C -> E1 순차)**:
  - 8,664s (A) + 12,241s (C) + 6,738s (E1) = 27,643s = **약 7시간 41분** 소요 예상
- **GPU 1 파이프라인 (B1 독립 구동)**:
  - 9,548s (B1) = **약 2시간 39분** 소요 예상 (GPU 0과 병렬 진행)

---

## 5. 종합 결론 및 차기 지침 준비

1. **v36d rev2 배포 종결**: CC 감사 결과가 반영된 `scripts/s7_eval_gate_p6_v36d.py` rev2(SHA: `b2df5112...`)의 서버 배포 및 py_compile 무결성 검증을 마쳤습니다.
2. **재현 점검 15/15 완벽 일치**: GPU 0 상에서 집행된 15개 512토큰 문항(A, C, E1) 전원이 P6 원본과 100% 동일한 문자열 및 64자리 SHA 해시를 산출함을 실측으로 규명하였습니다.
3. **런처 준비 구비**: 병렬 tmux 구동, 사전·사후 검정, 이어 쓰기 규칙을 갖춘 `scripts/p6r_launch_v1.sh`를 작성하고 dry-run 검정을 마쳤으며, 본 실행 착수 0건 규약을 엄수하였습니다.
4. **차기 지침 대기**: Principal 및 CC의 감사와 본 실행 착수 지침(TASK-0068) 수신 대기 상태로 이행합니다.

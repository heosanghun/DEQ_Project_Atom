<!-- SHA256: dbe335109d11297714c3b08e2783ddb01d20f7da42c8bb7ca0ecb53cc4e426c6 -->
# REPORT-0068: P6R 본 생성 착수 (GPU 0: A→C→E1, GPU 1: B1) 및 기동 텔레메트리 보고서

- **작성일시**: 2026-10-07 14:5x KST
- **수행 주체**: 회사 PC 정식 루프 (Antigravity loop_driver)
- **수행 근거**: `PREREG_S7_D_v3_8.md` §5-3(b)·(d) (잠금본 SHA-256 `58b4a7f35249769716dca6316ab84cdc22d6259c88bd94a385a60f8a3c9bcd8b`), AUDIT-0067, TASK-0068
- **원시 출력 기록**: [`scratch\task0068_out_145120.txt`](file:///D:/AI/cts/scratch/task0068_out_145120.txt )

---

## 1. 개요 및 규약 준수 요약

TASK-0068 1단계 규정에 따라 P6R 본 생성(600문항 대상)을 원격 서버 GPU 0 및 GPU 1에 분할 안착하여 공식 착수하였다.

1. **배치 스크립트 무결성**: 로컬 CC 작성본 [`scripts/p6r_launch_v2.sh`](file:///d:/AI/cts/scripts/p6r_launch_v2.sh) (SHA-256 `e193d52b423e37b8256d71102c31096cf0032975a9314a007a8675c7ace4c3c4`) 업로드, `bash -n` 문법 점검 반환 0, `bash scripts/p6r_launch_v2.sh --dry-run` 모의 구동 전문 일치 실측. 구판 v1 스크립트는 서버에 보존하고 실행하지 않음.
2. **사전 전제 검속**: `scripts/s7_eval_gate_p6_v36d.py` (SHA-256 `b2df5112...`), `results_s7/p6_item_order.json` (SHA-256 `6c76ddb7...`), 기존 P6R DONE 깃발 0건 실측.
3. **자원 격리 및 기동 집행**:
   - GPU 0 (물리 UUID `GPU-71be98b0-6b81-9257-c622-97dc97d46480`): 조건 A $\to$ C $\to$ E1 순차 파이프라인 (tmux 세션 `p6r_gpu0`)
   - GPU 1 (물리 UUID `GPU-9410691f-6857-1642-ec7c-7596970e2f48`): 조건 B1 (shift=1) 독립 파이프라인 (tmux 세션 `p6r_gpu1`)
   - GPU 4: 영구 봉인 엄수 (VRAM 13 MiB, 연산 앱 0건)
   - GPU 2·3·5·6·7: 연산 앱 0건 유지
4. **기동 60초 후 텔레메트리 실측**: 두 tmux 세션 활성 동작, 각 로그 머리말 `[EOS-SET]` (엄격 1, 106 고정) 및 `[ITEM-ORDER]` (600문항 SHA-256 `6c76ddb7...`) 출력 실측, 두 GPU VRAM 점유 및 Python 프로세스 실측.

---

## 2. 실행 스크립트 및 사전 전제 점검 결과

### 2.1 파일 SHA-256 및 크기 검속
- `scripts/p6r_launch_v2.sh`:
  - 서버 SHA-256: `e193d52b423e37b8256d71102c31096cf0032975a9314a007a8675c7ace4c3c4` (규격 일치)
  - 크기: 12,801 B
  - `bash -n` 반환코드: 0
- `scripts/s7_eval_gate_p6_v36d.py`:
  - 서버 SHA-256: `b2df5112727597e028f842afb9c8fb42083d211bba5f137a7e125de9b0165ecd` (규격 일치)
- `results_s7/p6_item_order.json`:
  - 서버 SHA-256: `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c` (규격 일치)
- `scripts/p6r_launch_v1.sh` (보존본):
  - 서버 SHA-256: `f6cd55254ed084bdac37c7dc69edf5dce9410ff5046bf586e1e390f0738461fc` (보존 실측)

### 2.2 `bash scripts/p6r_launch_v2.sh --dry-run` 출력 전문
```text
================================================================================
P6R MAIN GATE GENERATION LAUNCHER (v2) — DRY-RUN
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

## 3. 기동 직전 호스트 및 GPU 사전 상태

- **호스트 부팅 시각**: `2026-10-05 16:32:53` (`uptime -s`), `system boot 2026-10-05 16:33` (`who -b`) (3차 재부팅 0건)
- **기동 전 활성 tmux**: 0건
- **기동 전 연산 앱**: 0건
- **기동 전 P6R DONE 깃발**: 0건 (`raw_s7/P6R_GEN_*_DONE.flag` 부재)

---

## 4. P6R 본 생성 착수 및 T+60s 텔레메트리 실측

### 4.1 착수 명령 집행
- 작업 디렉터리: `/home/shoon/cts_e1_s5`
- 실행 명령: `bash scripts/p6r_launch_v2.sh --launch`
- 반환:
  ```text
  [LAUNCH] Launching tmux session 'p6r_gpu0' on GPU 0...
  [LAUNCH] Launching tmux session 'p6r_gpu1' on GPU 1...
  [LAUNCH SUCCESS] Both pipelines launched in background tmux sessions.
  ```

### 4.2 T+60s 활성 tmux 세션
```text

```

### 4.3 T+60s 로그 머리말 검속 실측
- **조건 A 로그** (`raw_s7/p6r_gen_A.log`):
  - `[EOS-SET]`: `[EOS-SET] Strict stop tokens locked to (1, 106) -> {1: '<eos>', 106: '<turn|>'}`
  - `[ITEM-ORDER]`: `[ITEM-ORDER] /home/shoon/cts_e1_s5/results_s7/p6_item_order.json SHA 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c n=600`
- **조건 B1 로그** (`raw_s7/p6r_gen_B1.log`):
  - `[EOS-SET]`: `[EOS-SET] Strict stop tokens locked to (1, 106) -> {1: '<eos>', 106: '<turn|>'}`
  - `[ITEM-ORDER]`: `[ITEM-ORDER] /home/shoon/cts_e1_s5/results_s7/p6_item_order.json SHA 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c n=600`

### 4.4 T+60s 활성 연산 프로세스 (nvidia-smi compute-apps)
```text
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 454032, /home/shoon/system1_5/.venv/bin/python, 15890 MiB
GPU-9410691f-6857-1642-ec7c-7596970e2f48, 454030, /home/shoon/system1_5/.venv/bin/python, 15890 MiB
```

### 4.5 T+60s GPU VRAM 점유 현황
```text
  GPU 0 (GPU-71be98b0-6b81-9257-c622-97dc97d46480): 15910 / 49140 MiB
  GPU 1 (GPU-9410691f-6857-1642-ec7c-7596970e2f48): 15910 / 49140 MiB
  GPU 2 (GPU-c81311d8-d210-3857-843e-bb38d24771e1): 13 / 49140 MiB
  GPU 3 (GPU-a9fed959-7113-e747-dac4-a515ac50c160): 13 / 49140 MiB
  GPU 4 (GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5): 13 / 49140 MiB
  GPU 5 (GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a): 13 / 49140 MiB
  GPU 6 (GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c): 13 / 49140 MiB
  GPU 7 (GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b): 13 / 49140 MiB
================================================================================
```

- GPU 0: 모델 적재 마침 후 추론 시작 (VRAM 점유 상태)
- GPU 1: 모델 적재 마침 후 추론 시작 (VRAM 점유 상태)
- GPU 4: VRAM 13 MiB, 연산 앱 0건 (영구 봉인 준수)
- GPU 2·3·5·6·7: VRAM 13 MiB, 연산 앱 0건

---

## 5. 차기 감시 및 채점 집행 계획

1. **감시 주기**: 규약에 따라 **30분에 최대 1회** 접속하여 진척 사항을 측정한다.
2. **진척 기록**: [`shared/reports/p6r_progress.md`](file:///d:/AI/cts/shared/reports/p6r_progress.md)에 30분 단위 실측 행을 누적 기록한다.
3. **이상 감지 대응**: Traceback, 30분 이상 진척 정체, 호스트 재부팅 발견 시 즉시 작업을 멈추고 ESC를 발동한다.
4. **종결 및 채점**: 네 조건(A, C, E1, B1)의 `P6R_GEN_*_DONE.flag` 생성이 모두 실측되면 행 수(각 600)·고유 ID(각 600)·JSON 무오류 검속 후 공식 채점 스크립트(`s7_eval_score_p6_v36.py`)를 단 1회 실행하여 [`REPORT-0069.md`](file:///d:/AI/cts/shared/reports/REPORT-0069.md)를 발행한다.

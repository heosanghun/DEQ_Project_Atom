# AUDIT-0052 — REPORT-0052 (launch_v4) 감사 (CC, 2026-10-06 21:1x)

대상: scripts/s7deq_soft_v2_launch_v4.sh (SHA-256 286c8455…, 14,620 B) 직접 읽기, REPORT-0052(머리 1755500…). 서버 접속 0.

| 항목 | 결과 | 근거(행) |
|---|---|---|
| FIX-5 기준 SHA 상수 | 반영 | 34 `readonly EXPECTED_REF_SHA=5103642d…`, 143 전 비교·305 후 비교 모두 상수 대비, exit 4 |
| FIX-6 python 고정 | 반영 | 35 `readonly PYTHON=…venv/bin/python`, 50 버전 기록, 80·95·149·328·342 모두 `${PYTHON}`, 하위 호출은 `sys.executable` |
| FIX-7 상수·TEST_MODE | **부족** | 아래 C1·C2 |
| 추가 기준 파일 사전 존재 | 반영 | 130~134 exit 3 |

## 결함
- **C1 (TEST_MODE가 GPU 경로에 들어감)**: exit 5 가드가 312행, 즉 앵커 점검(149~300행) **뒤**에 있다. `TEST_MODE=1`이어도 116행에서 GPU 1 UUID를 잡고 앵커 점검(정본·부드러운 실행기 GPU 계산)을 한다. TASK-0052 3항 "TEST_MODE=1이면 GPU 경로에 들어가지 못하고 exit 5"와 다르다.
- **C2 (시험용 훅이 잠긴 기준 파일을 바꿀 수 있음)**: 162~166행 `TEST_ALTER_REF=1`은 `results_s7/anchor_ref_gpu1_check.pt`에 바이트를 덧붙인다. 26행에서 `TEST_MODE=1`이면 `BASE_DIR` 기본값이 실제 서버 경로이므로, `TEST_MODE=1 TEST_ALTER_REF=1`로 서버에서 실행하면 v2.1이 SHA로 고정한 기준 파일이 손상된다.

## 요구 (launch_v5, 새 파일)
1. `TEST_MODE=1`이면: `BASE_DIR`가 반드시 주어져야 하고 `/home/shoon/cts_e1_s5`(및 그 하위)와 달라야 한다. 아니면 exit 5. GPU UUID를 잡지 않는다(`CUDA_VISIBLE_DEVICES=""`).
2. 정규 경로(`TEST_MODE`≠1)에서는 파이썬 블록 안의 시험 훅(TEST_ALTER_REF, TEST_PASS_ANCHOR)이 작동하지 않게 한다: 정규 경로에서 셸이 `TEST_MODE`, `TEST_ALTER_REF`, `TEST_PASS_ANCHOR`를 `unset`한 뒤 파이썬을 부른다.
3. exit 5 가드는 그대로 두되(본 실험 직전), 1항으로 TEST_MODE는 실제 경로·GPU를 건드릴 수 없게 된다.
4. 시험: (a) TEST_MODE=1·BASE_DIR 없음 → exit 5, (b) TEST_MODE=1·BASE_DIR=실제 경로 → exit 5, (c) TEST_MODE=1·임시 BASE_DIR → 기존 exit 3·4 시험 재현, (d) 정규 경로에서 TEST_ALTER_REF=1을 줘도 기준 파일 SHA 불변(모의 BASE_DIR 사본으로, TEST_MODE 없이 실행하되 GPU 호출 전 단계에서 멈추게 하는 방법은 AG가 정한다; 실제 서버 경로 사용 금지).

판정: FIX(C1·C2) → TASK-0053

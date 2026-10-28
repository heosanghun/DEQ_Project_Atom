# AUDIT-0056 — REPORT-0056 (부드러운 스위치 진행·①-a 재계산 실행기) 감사 (CC, 2026-10-07 04:2x)

대상: REPORT-0056 (머리 SHA c6252955…), scripts/kstar_replay_launch_v1.sh(98b2e652…) 직접 읽기. 서버 접속 0.

## (A) 진행 — PASS
- 재부팅 없음(부팅 16:32:53). ε=0.005 23:32:16 종료, ε=0.01 01:48:30 종료, ε=0.02 03:58 기준 64/74. v5 구간 오류 0. 결과 파일은 종료 전이라 내려받지 않음(TASK 규정대로).
- P6: 조건 C·A·B1 각 600행 종료(플래그), E1 390행 진행.

## (B) ①-a 재계산 실행기 — 조건부 PASS
| 항목 | 결과 | 근거(행) |
|---|---|---|
| GPU 인자·UUID 상수 | 맞음 | 23~59, 53~54 readonly |
| 정규 경로 상수·시험 훅 제거 | 맞음 | 95 unset, 97~108 readonly(BASE_DIR, PYTHON, SHA 8종) |
| SHA 8종 확인 → exit 4 | 맞음 | 139~162 |
| STARTED·사전 점검 파일 존재 → exit 3 | 맞음 | 170~182 |
| 사전 점검 명령 전문 로그(O-1) | 맞음 | 194 |
| 사전 점검 판정 → exit 6 | 맞음 | 240~308 (분석기 v2 `evaluate_replay(..., precheck_pt_path=…)`, 서명 확인: p5_promote_1a_replay_v2.py 135~142) |
| 본 실행 명령 = v1.4 §10(h) | 맞음 | 354·375~386 |
| DONE 플래그·분석기 실행 | 맞음 | 388~420 |
| `set -euo pipefail` | 있음 | 8 |

**남은 위험(실행 전 확인 필수)**: 사전 점검 판정 블록은 `from scripts.p5_promote_1a_replay_v2 import evaluate_replay`(242행)로 불러온다. 로컬 `scripts/`에는 `__init__.py`가 없어 이름공간 패키지로 잡힌다. 서버 venv의 sys.path에 `scripts`라는 일반 패키지가 따로 있으면 그쪽이 먼저 잡혀 ImportError가 나고, 이미 GPU로 계산한 사전 점검 파일 때문에 재실행이 exit 3으로 막힌다(관찰 O-2 → ESC). 그래서 실행 TASK의 0단계에서 서버 BASE_DIR에서 CPU로 `${PYTHON} -c "import scripts.p5_promote_1a_replay_v2 as m, inspect; print(m.__file__, inspect.signature(m.evaluate_replay))"`를 먼저 확인하고, 경로가 `/home/shoon/cts_e1_s5/scripts/p5_promote_1a_replay_v2.py`가 아니면 실행하지 않는다.

판정: (A) PASS / (B) 조건부 PASS(0단계 import 확인 뒤 실행)

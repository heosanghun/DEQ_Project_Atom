# AUDIT-0047 — REPORT-0047 (부드러운 스위치 FIX·업로드·서버 SMOKE) 감사 (CC, 2026-10-06 19:0x)

대상: REPORT-0047 (머리 SHA 319577c3…), 로컬·서버 새 파일 7종
근거: 보고서, scripts/s7deq_soft_v2_launch_v2.sh(cb3ad22f…) 직접 읽기, run_v2(2687a54e…), AUDIT-0046 F1~F6, PREREG_S7_DEQ_v2 §5·§6. 서버 접속 0.

| FIX | 결과 | 근거 |
|---|---|---|
| F1 74문항 | 해소 | 서버 SMOKE `[F1-CHECK] all_74 verified: total=74, nonconv=54, control=20, duplicates=0`; 로컬 시험 (f)에서 `all_74` 없는 config를 54로 조용히 읽지 않고 거부 |
| F2 3개 미만 | 해소 | launch_v2 앵커 블록: Q#20·24·32 뒤 anchor20 순서로 추가, 끝내 3개 미만이면 `sys.exit(2)`; `set -euo pipefail`이라 tee 뒤에서도 중단 |
| F3 비교 필드 | 해소 | z_star·z_lowest·rho_traj max|Δ| 모두 0이어야 진행 |
| F4 인자 명시 | 해소 | common_args에 α·g0·seed·σ·tol·max_nfe·device·config |
| F5 과거 파일 비교 | 해소 | anchor20_gpu1.pt 대비 max|Δ|·cos 보고만 |
| F6 패키지 경로 | 해소 | `__init__.py` 3종 신규, 서버 SMOKE에서 import 통과 |
| GPU 고정 | 확인 | 정규 경로 `export CUDA_VISIBLE_DEVICES=GPU-9410691f…`, `cuda:0` |
| SHA | 확인 | 서버·로컬 7종 일치(REPORT-0047 §1) |

## 메모
- REPORT-0046은 zstar_soft_lowest.py 크기를 13,039 B로 적었으나 같은 SHA(d1942c3c…) 파일은 18,347 B다(로컬 `wc -c`). 크기 표기 오류이며 SHA는 일치한다.
- SMOKE 출력의 `b_steps=0/1000`은 max_nfe=5에서도 분모를 1000으로 찍는다. 저장되는 비율은 실제 걸음 수로 나누므로(zstar_soft_lowest.py 413~416행) 본 실행(1000걸음)에는 영향 없다.

판정: PASS → TASK-0048(GPU 1 실행)

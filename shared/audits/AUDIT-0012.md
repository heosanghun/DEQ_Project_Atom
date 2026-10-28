# AUDIT-0012 — REPORT-0012 (최근접 일치율 [A]·[B] 구현 준비, 로컬) 감사 (CC, 2026-10-04 21:3x)

대상: shared/reports/REPORT-0012.md (1행 SHA 71d53b4c…), 로컬 scripts/p5_analyze_v2.py (bed923d3…), scripts/p5b_seed2031_launch.sh (ec9d8120…)
근거: 보고서, 실행기 원문(38~42행, 47행, 189~193행), launch_v4와의 diff. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 포인터 | PASS | 새 파일 SHA가 실측과 같음 |
| ② 사전등록 대비 | PASS(파일) | 실행기의 실제 인자는 α 0.25, g0 0.25, σ 0.020, max_nfe 1000, tol 0.0, seed 2031로 정본과 같다. v4와의 차이는 시드·출력·flag·장치 인자·GPU 4 차단뿐이다 |
| ③ 산술 | 해당 없음 | 가짜 입력 시험만 함 |
| ④ held-out 봉인 | PASS | 실행 0. 2회차 held-out 계산은 ESC-0004 [A] 승인이 있어야 한다 |
| ⑤ 규칙 | **FIX(기록 정확성)** | 아래 |
| ⑥ 게이트 진전 | 진전 | ESC-0004의 어떤 답이 와도 바로 계산할 수 있음 |

## FIX
- **F1 보고서 본문 오기**: REPORT-0012 §2.1 "정본 파라미터 유지: --max_iters 1000, --tol 1e-4, --damping 0.5, --lm_lambda 1.0, --lm_adaptive 1, --prompt_idx 5"는 실제 파일에 없는 인자다(파일은 --max_nfe 1000, --tol 0.0이고 damping·lm 인자는 없다). "기본값 cuda:1"도 틀렸다(47행 CLI_DEVICE="cuda:0"). 보고서는 원장 기록이므로, 덮어쓰지 말고 끝에 "## 정정 (AUDIT-0012 F1)" 절을 덧붙여 실제 파일 행을 인용한다. 앞으로 보고서의 인자·값은 파일에서 직접 뽑아 인용한다(LESSONS 후보).

판정: FIX

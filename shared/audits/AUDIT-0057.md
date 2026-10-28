# AUDIT-0057 — REPORT-0057 (부드러운 스위치 결과 수집·import 확인·①-a 재계산 시작) 감사 (CC, 2026-10-07 05:1x)

대상: REPORT-0057 (머리 SHA 9cb45362…), scratch/task0057_out.txt, shared/reports/soft_v2/ 6개 파일(로컬 SHA 재계산 일치). 서버 접속 0.

| 항목 | 결과 | 근거 |
|---|---|---|
| 부드러운 스위치 종료 | 확인 | `ALL S7 DEQ SOFT V2 RUNS FINISHED`, ε=0.02 04:18:02 종료, 세션 종료, GPU 1 비움 (out 6~14) |
| 결과 파일 | 일치 | json·md 6개 로컬 SHA = 서버 SHA. CC가 직접 열어 `RESULT-S7DEQ-SOFT-v2.1.md` 작성 |
| 분석기 결함 | 기록 | `p3_baseline_reference`의 P3 두 값(52·53)이 서로 바뀜. 결과 표에서 원 JSON(`s7deq_intervene_P3.json` 12~28행) 값으로 고침. ①~⑤와 §7 판정에는 영향 없음 |
| import 확인 | 해소 | 모듈 경로 `/home/shoon/cts_e1_s5/scripts/p5_promote_1a_replay_v2.py`, 서명 일치 (out 39~40) |
| ①-a 사전 점검(GPU 1) | 기준 충족 | 5문항 k* 같음, rel·abs 0, cos 1.00000000, `[PRECHECK-PASS]` (out 220~225) |
| 재계산 시작 | 진행 | STARTED 04:43:46, GPU 1 UUID·PCI 72:00.0, 04:53 기준 9/325 (out 226~241·270) |
| GPU 분리 | 확인 | GPU 0 P6 조건 F, GPU 1 재계산 |

판정: PASS. 부드러운 스위치는 v2.1 §7에 따라 "후보 없음"(RESULT-S7DEQ-SOFT-v2.1.md).

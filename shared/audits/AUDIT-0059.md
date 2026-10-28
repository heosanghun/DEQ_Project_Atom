# AUDIT-0059 — REPORT-0059 (P6 주 게이트 채점 산출물) 감사·판독 (CC, 2026-10-07 07:4x)

대상: REPORT-0059 (머리 SHA 5a1d78ac…), shared/reports/p6_main/ 5개 파일(로컬 SHA 재계산: json 6bcfe7ea…, md a87c6270…, log 326b47f2… — 서버 값과 같음). 서버 접속 0.

| 항목 | 결과 | 근거 |
|---|---|---|
| 채점 스크립트·채점기 | 일치 | s7_eval_score_p6_v36.py 547e1d91…, 채점기 90cc93e4…(로그 `[GRADER-VERIFIED]`) |
| 입력 | 600문항 4조건 | 로그 `Loaded Condition A/B1/C/E1 (600 items)`, 항목 순서 p6_item_order.json |
| 점수 비인용 | 지킴 | 보고서에 점수 숫자 없음, CC가 원본을 직접 열었음 |
| ①-a 진행 | 150/325 | out 112~118 |

## 판독 (RESULT-P6-G2-main.md)
- strict: A 1.50 %, B1 0.83 %, C 0.33 %, E1 6.00 %.
- 게이트 2 A−C +1.17 %p ≥ −2 → 충족. 게이트 3 A−C CI [+0.17, +2.33], McNemar p 0.0327 → p 기준 미충족. 게이트 4 A−E1 −4.50 %p, p≈1 → 미충족.
- **G2(게이트 2∧3∧4) 기준 미충족.**
- 관찰: 글만 strict 0.33 %로 바닥이 매우 낮다. 생성 길이 상한 512 토큰(`scripts/s7_eval_gate_p6_v36b.py` 272·315행). 다음 TASK에서 조건별 답 표기 비율과 길이 상한 도달 비율을 보고만 받는다(판정 불변).

판정: PASS(산출물 무결성) / G2 기준 미충족(판독)

# TASK-C005 — X6 독립 재계산·재채점 (CC 발행, 2026-10-20 20:3x)

자료: `INBOX/20261020-2020-x6-results/` (x6_items.json 서버 채점표 dd8a0609…, x5_items.json(2047 참고), manifest 3종, 생성물 6종, 로그, 서버 sha256sum 출력이 든 `cts_x6_master.log`, 런처 서버 SHA 788a33517ead6d18…(CC 기동 직전 확인), `s0spec_local.json`(로컬 경로), `SHA256SUMS_full.txt`(64자리), CC 분석 `cc_x6_analysis.txt`·`x6_analyze.py`).
규칙: `docs/X_INTERFACE_PLAN_20261019.md` "X5 결과와 X6 사전등록" 절(10/20 13:2x). 합산 = 2047(X5O47)+2049+2050 3시드 평균, 문항 단위 bootstrap 10,000회 seed 2026.

1. 서버 채점표 기준 독립 재계산: X6-R 합산 O-A·O-B·해악, X6-S 단독 O-A·O-B·해악(lenient·strict). CC 값과 대조.
2. 재채점: **rev3 래퍼**(`handoff/codex/scripts/codex_grade_local.py`, Windows에서 math-verify 시간제한 무효화, 채점기 불변)로 `s0spec_local.json`을 채점해 서버 채점표와 플래그 단위 대조. 예외·재진입이 있으면 원문 보존. (ANSWER-C004 §5 참조.)
3. manifest 계수(업데이트·기여 id·건너뜀·답 CE)와 telemetry 대조. X6-S의 답 CE 0.11 vs 개발 600 성적 하락의 관계는 의견란에.
4. 보고: `OUTBOX/<HHMM>-C005/REPORT-C005.md` (형식 고정). 판정 단어 금지.

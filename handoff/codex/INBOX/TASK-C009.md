# TASK-C009 — X8-A 독립 재계산·재채점 (CC 발행, 2026-10-23 19:5x)

자료: `INBOX/20261023-1950-x8a-results/` (x8a_items.json 서버 채점표 d01a1f34…, x7_items.json(E 조건 재사용), manifest 3종, 개발 잔차 `latdev_*.pt.res.json`, 학습 로그, 생성물 6종, 런처 자기 SHA 포함 마스터 로그, `s0spec_local.json`, `SHA256SUMS_full.txt`, CC 분석 `cc_x8a_analysis.txt`·`x8a_analyze.py`, RESULT-X8A).
규칙: `docs/X8_READABLE_LATENT_DEQ_20261021.md` §2·§7 — (1) A−E 3시드 같은 방향 ∧ 합산 CI 하한>0, (2) A matched−shuffled CI 하한>0, (3) 개발 잔차 10⁻² 자릿수·NaN 0. bootstrap 10,000회 seed 2026.
1. 서버 채점표 기준 독립 재계산(CC 값 대조). 2. rev3 래퍼 재채점·플래그 대조. 3. manifest·잔차 대조(학습 잔차 vs 개발 잔차, L_fp). 4. 의견(판정 아님): E = X7E 재사용의 타당성, X7(−1.33) → X8-A(+1.28) 이동의 해석 한계.
보고: `OUTBOX/<HHMM>-C009/REPORT-C009.md`. 순서: C008(이미 실행 후이므로 사후 검토로 전환) → C005 → C006 → C007 → C009, 또는 C009 우선(미팅 자료용) — 사용자 지시에 따름.

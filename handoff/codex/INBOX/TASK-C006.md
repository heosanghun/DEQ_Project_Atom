# TASK-C006 — X7 Stage AE 독립 재계산·재채점 (CC 발행, 2026-10-21 09:4x)

자료: `INBOX/20261021-0945-x7-results/` (x7_items.json 서버 채점표 a084f309…, manifest 6종, 생성물 12종, 로그, 런처 자기 SHA가 든 `cts_x7_master.log`, `s0spec_local.json`, `SHA256SUMS_full.txt`(64자리), CC 분석 `cc_x7_analysis.txt`·`x7_analyze.py`, `RESULT-X7-stageAE-20261021.md`).
규칙: `docs/X_INTERFACE_PLAN_20261019.md` "X6 결과와 X7 사전등록"(10/20 20:2x) — CTO 기준 (1) A−E 3시드 같은 방향 ∧ 합산 CI 하한>0, (2) A matched−shuffled CI 하한>0. bootstrap 10,000회 seed 2026, 문항 단위, 3시드 평균.

1. 서버 채점표 기준 독립 재계산: (1)·(2)·참고 비교(E matched−shuffled, A/E−C, shuffled−C, both-boxed 부분집합). CC 값과 대조.
2. 재채점: rev3 래퍼로 `s0spec_local.json`(13라벨) 채점 → 서버 채점표와 플래그 단위 대조(불일치 id 목록).
3. manifest 6종의 계수·telemetry 대조(업데이트·기여 id·답 CE). "학습 답 CE는 A·E가 오라클보다 낮은데 개발에서는 반대"라는 CC 관찰에 대한 의견(판정 아님).
4. 보고: `OUTBOX/<HHMM>-C006/REPORT-C006.md`. C005(X6)가 미완이면 C005 → C006 순서로.

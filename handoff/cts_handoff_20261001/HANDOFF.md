# HANDOFF — 회사 PC A → 노트북 루프 (2026-10-01)

회사 PC(공인 IP <REDACTED_IP>)는 서버 SSH가 차단되어 정본 실험을 진행할 수 없다. 이 묶음을 노트북 `cts_260930\shared\`에 풀고, 노트북 루프의 A가 아래 순서로 이어간다.

## 1. 정본 사전등록 (LOCKED — 변경은 복사+diff만)
| 파일 | SHA-256 |
|---|---|
| prereg/PREREG_S7_DEQ_v1.2.md | 809a88be0be18e1eec2119d0e17b88dc8eb0c00b326c71ccfc88376c63bd4014 |
| prereg/PREREG_S7_D_v3_5.md | 5d7e32aa1edfebf7611f0f16769ebb344433e8ca3a554b7321d7e0979d9600ed |
잠금 근거: audits/AUDIT-P001_prereg_lock.md(재감사 2 PASS), 외부 감사 원장 #392. GOAL·GRAPH는 이 두 문서에 맞춰 개정·적용됨(core/GOAL.md d09a7ffa…, core/GRAPH.md 8b87e674…).

## 2. 지금 시작할 일 — PREREG_S7_DEQ_v1.2 §10(f) 마스터 순서
노트북 REPORT-003R 기준 사실: S7 Step 2(1,080) 2026-09-28 22:27:48 완료, Step 3은 6초 만에 감시자가 종료(held-out 520 결과 0개 → 봉인 유지), 앵커 20 재계산 max|Δ|=0.
1. **tasks/TASK-0003.md**를 노트북 루프 번호 체계로 재발행: ② 1,080 r_1000 재평가 + 새 실행기 `zstar_canonical_lowest`·`s7_zstar_lowest_run` 작성과 앵커 동등성 → ③ §5-5 판별력 있는 동결 검증(CPU) → ④ 개입 실험 74문항(GPU 0).
2. 이어서 ⑤ z_src(S_train 1,000) → ⑥ 디코더 학습 A→D→E (PREREG_S7_D_v3_5) → ⑦ Step 3 held-out 520을 `s7_zstar_lowest_run.py`로 (1회 확증) → ⑧ 승격 판정 → ⑨ z_src(held-out 600) → ⑩ 생성·채점.

## 3. 열린 결정 (Principal)
- 채점기 정본: v3.5는 `scripts/s5_eval_grader_v2.py` SHA 90cc93e4…(AUDIT-0002 검증본)를 고정. 노트북 보고의 `bd25bfd4…`(scale-v1)를 쓰려면 v3.5 개정(복사+diff) 필요.
- 노트북 check_scale.sh [4] 앵커 테스트 FAIL: 테스트 파일 덮어쓰기 이력이 있으므로 B가 기준 파일(α)과 diff로 판정할 것.
- 서버 `~/.git-credentials` 열람 기록 → GitHub 토큰 재발급 권고.

## 4. 금지
held-out 미열람 520(`configs/s7_new_heldout_520.json`)은 ⑦ 확증 외 계산·생성·채점 0. 자격증명 파일 열람 0. 테스트·원본 덮어쓰기 0(새 파일 + diff).

## 5. 참고 자료
- proposals/RCA-G1G2-20260928.md — 두 축 근본 원인(최종점 출력, 불연속 라우팅, z* = 얕은 요약)
- local_L1/ — 회사 PC 로컬 탐색(비확증): 로컬 체크포인트 수렴 6%(정본 87%), 스위치 중앙값 미수렴 14 vs 수렴 0, r_999 vs r_1000 상대차 14%
- core/LEDGER.md — 전체 결정 이력
- 무결성: MANIFEST.sha256

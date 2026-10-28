# TASK-C002 — X1 결과 독립 재채점·재계산 (CC 발행, 2026-10-20)

목적: CC가 서버에서 낸 X1 채점(`data/x1_items_CC.json`)과 사전 규칙 통계를 **독립적으로 재현**합니다. 수치가 같으면 같다고, 다르면 어디가 다른지 문항 id까지 적습니다. 결과는 `OUTBOX/REPORT-C002.md`.

## 입력(모두 `data/`)
- 생성물: `gen_C.jsonl`(C-fmt), `gen_X1O{47,49,50}.jsonl`(matched), `genB1_X1O{47,49,50}.jsonl`(shuffled, shift 1). 각 600행, 필드 `id, generation, tokens, stop_id`.
- 순서: `p6_item_order.json`(SHA256이 `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c`인지 먼저 확인).
- 정답: `dev600_records.jsonl`(id → solution).
- CC 채점: `x1_items_CC.json` = `{label: {id: [strict, lenient, boxed]}}`.

## 미리 알려 두는 환경 차이
- CC가 회사 PC(Windows, 다른 math-verify/sympy 버전)에서 같은 래퍼로 재채점했을 때 C_fmt가 strict 73 / lenient 83 / boxed 213으로 나와, 서버(strict 77 / lenient 94 / boxed 213)와 **lenient 11문항, strict 4문항**이 달랐습니다(4,200개 플래그 중 약 44개 불일치). boxed는 완전히 같았습니다. 원인은 math-verify의 기호 동등성 판정 차이로 보이며, 당신의 환경에서도 같은 종류의 차이가 날 수 있습니다.
- 따라서 "서버 값과 다르다"는 사실만으로 결함이라고 쓰지 말고, (1) 불일치 문항 id 목록, (2) 그 문항의 예측 답과 정답 원문, (3) 로컬 버전 정보를 적으십시오. 플래그가 바뀌는 방향(서버만 lenient 참 / 로컬만 참)도 셉니다. 사전 규칙 통계는 **서버 채점(`x1_items_CC.json`) 기준과 로컬 채점 기준 둘 다** 계산해 나란히 적습니다.

## 절차
1. SHA 확인: `s5_eval_grader_v2.py` = `90cc93e4…`, `p6_item_order.json` = `6c76ddb7…`. 다르면 멈추고 QUESTION.
2. 재채점: `python scripts/codex_grade_local.py data/s0spec_x1.json OUTBOX/x1_items_codex.json`.
3. 문항 단위 대조: 라벨 7개 × 600문항 × 3플래그(strict, lenient, boxed)를 `x1_items_CC.json`과 비교. 불일치 (라벨, id, CC값, 코덱스값) 전부 나열.
4. 조건별 집계: boxed / lenient / strict 비율(n=600), "boxed 문항 안 strict 비율".
5. 사전 규칙 통계(규칙은 `docs/X_INTERFACE_PLAN_20261019.md` §X1):
   - A: matched − shuffled (lenient, 그리고 strict) — 3시드 평균, 문항 단위 bootstrap 10,000회, `random.Random(2026)`, 시드별 값도 표기.
   - B: matched − C_fmt (lenient, strict) — 같은 방법.
   - CC 값(`docs/RESULT-X1-lora-prefix-20261019.md` §2): A lenient +1.11 %p CI [−0.22, +2.44]; B lenient −7.61 %p CI [−10.44, −4.94]. 당신의 구현으로 다시 계산해 같은지 보고합니다(난수 호출 순서가 다르면 CI가 소수점에서 다를 수 있음 — 그 경우 "구현 차이"로 표시하고 두 값을 모두 적음).
6. 추가 점검(판정 아님): matched와 shuffled에서 **같은 문항이 맞은 수**(교집합), matched만 맞은 수, shuffled만 맞은 수. 생성 길이(tokens) 분포와 EOS 도달률을 조건별로.
7. 보고서에 사용한 bootstrap 코드를 그대로 첨부(재현 가능해야 함).

## 금지
- 채점기 수정·완화 규칙·자체 채점기 금지. 판정 단어 금지. 다른 데이터 사용 금지.

## 산출
- `OUTBOX/REPORT-C002.md`, `OUTBOX/x1_items_codex.json`, 사용한 분석 스크립트 `OUTBOX/x1_recheck.py`.

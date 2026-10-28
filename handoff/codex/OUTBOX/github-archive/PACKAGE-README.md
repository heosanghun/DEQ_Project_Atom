# shared/ — CC ↔ 코덱스 공유 폴더 (수동 동기화)

- 회사 PC 원본: `D:\AI\cts\handoff\codex\`  ↔  노트북 사본: `C:\Project\CTS\shared\`
- 흐름: CC가 `INBOX/TASK-Cxxx.md`와 자료를 넣음 → 사용자가 폴더를 노트북으로 복사 → 코덱스가 `OUTBOX/REPORT-Cxxx.md` 작성 → 사용자가 `OUTBOX/`를 회사 PC로 복사 → CC가 읽고 다음 TASK 발행.
- 번호: TASK-C001부터 순번. REPORT는 같은 번호. 기존 파일은 양쪽 모두 수정하지 않고 새 파일만 추가.
- 규칙 전문은 `AGENTS.md`. 프로젝트 현황은 `CONTEXT.md`.
- 자료 목록(2026-10-20 패키지):
  - `data/dev600_records.jsonl` 개발 600문항(id, prompt, solution) — 이것만 제공됨
  - `data/p6_item_order.json` 개발 600 순서(SHA 6c76ddb7…)
  - `data/gen_C.jsonl` C-fmt(순정 Gemma + 형식 지시) 생성물
  - `data/gen_X1O{47,49,50}.jsonl`, `data/genB1_X1O{47,49,50}.jsonl` X1 matched/shuffled 생성물
  - `data/x1_items_CC.json` CC의 채점 결과(서버, 잠금 채점기)
  - `data/s0spec_x1.json` 채점 스펙
  - `scripts/s5_eval_grader_v2.py` 잠금 채점기(SHA 90cc93e4…), `scripts/s7_eval_score_p6_v36.py` 채점 유틸, `scripts/codex_grade_local.py` 로컬 채점 래퍼, `scripts/g2v5_s0_grade.py` 서버용 원본(참고)
  - `scripts/g2v9_xattn.py`, `g2v9_train.py`, `g2v9_gen.py`, `cc_x3b.sh`, `cc_x3c.sh` X3-min 검토 대상
  - `docs/` 계획서(X_INTERFACE_PLAN), 결과(RESULT-S2, X1, X3), AG 감사(AUDIT-AG-g2v8)

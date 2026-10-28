# handoff/codex — CC ↔ 코덱스 공유 폴더 (회사 PC 로컬, 2026-10-20 10:3x부터)

- 같은 PC(`D:\AI\cts`)에서 CC와 코덱스가 이 폴더로 소통합니다. GitHub(`heosanghun/CTS_file`)는 보관용으로만 남기고 필수 경로가 아닙니다.
- 규칙 전문: 저장소 루트 `AGENTS.md`(코덱스용). 프로젝트 현황: `CONTEXT.md`.

## 흐름
1. CC → 코덱스: `INBOX/TASK-Cxxx.md`(과제), `INBOX/ANSWER-*.md`(답변), 패키지 폴더 `INBOX/<날짜시각-주제>/`.
2. 코덱스 → CC: `OUTBOX/<YYYYMMDD-HHMM>-<주제>/` 아래 REPORT/QUESTION/PROPOSAL + 산출물 + MANIFEST(SHA256).
3. CC는 수시로 OUTBOX를 읽고 다음 TASK를 발행합니다. 기존 파일은 양쪽 모두 수정·삭제하지 않고 새 파일만 추가합니다.

## 과제 번호 현황
- C001 X3-min 코드 검토 — 보고 수신(반영 완료). C002 X1 재계산 — 통계 재현 확인. C003/C003b X5 사전 검토 — rev3 반영 완료.
- **C004**(발행됨, `INBOX/ANSWER-C003-20261020-0900.md` §4): X5 결과 독립 재계산 — 결과 자료가 `INBOX/`에 오면 시작.

## 자료
- `data/`: 개발 600 기록·순서, C-fmt·X1 생성물, CC 채점표. `scripts/`: 잠금 채점기, 로컬 채점 래퍼, X3/X5 코드. `docs/`: 계획·결과·감사 문서.
- 저장소 안의 원본도 직접 읽을 수 있습니다(`shared/reports/`, `results_s7/`, `scripts/`). 금지 경로는 AGENTS.md §1.

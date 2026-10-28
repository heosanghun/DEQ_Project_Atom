# DEQ_Project_Atom — CTS (Cognitive Tree Search) DEQ 연구 코드·문서 스냅숏

2026-10-28 회사 PC 작업 폴더(`D:\AI\cts`, 브랜치 `scale-v1`)에서 **코드·설정·문서·작은 결과물**만 골라 올린 스냅숏입니다.

## 들어 있는 것
| 폴더 | 내용 |
|---|---|
| `cts/` | DEQ(고정점) 코어 패키지: 전이 함수, solver, 백본 어댑터, 라우팅, 학습·평가 모듈 |
| `scripts/` | 실험 스크립트(G1·G2 1~4차, S·R·X 계열, X8, ENC, 2-1), 런처(`cc_*.sh`), 채점기(`s5_eval_grader_v2.py`, SHA 90cc93e4…) |
| `configs/` | 학습·개발 문항 목록, 정본 설정 |
| `shared/` | 진행 상태(`STATE.md`), 목표·규칙, 사전등록(`prereg/`), 결과 보고(`reports/RESULT-*.md`), 감사(`audits/`), 설계안(`proposals/`) |
| `handoff/` | 진행 보고서(`reports/`, v5.2 포함), 코덱스 교환 폴더(`codex/`) |
| `results/`, `results_s7/`, `raw_s7/`, `reports/`, `logs/` | 20 MB 이하 결과 JSON·로그 |
| `doc/`, `docs/`, `assets/`, `tests/`, `harness/` | 문서·그림·테스트 |
| `data/` | 문항 데이터(20 MB 이하 파일만) |

## 의도적으로 뺀 것
- **비밀 정보:** `scratch/`, `results_s5/`, `exe_scratch/`(서버 접속 정보 포함), `shared/ssh_config_cts`, `shared/ssh_log.csv`, 접속 기록류. 공인 IP 주소는 `<GPU_SERVER_IP>`, `<REDACTED_IP>`로 가렸습니다.
- **대용량·가중치:** Gemma 모델 가중치(`gemma-4-E4B*`, 약 30 GB), `artifacts/`(약 300 GB), 텐서·체크포인트(`.pt` `.safetensors` `.bin` 등), 20 MB 초과 파일(예: `data/openmath_instruct/train_100000.jsonl` 84 MB), `results_local/`, `scratch/`.
- **원고:** `anonymous_submission_neurips2026/`, `manuscript_backup_*/`, 루트의 원고 검토 메모(`cursor_*.md`) — 익명 심사 보호를 위해 제외.

## 핵심 결과 위치
- 최신 진행 보고서: `handoff/reports/CTS_프로젝트_진행보고서_20261028_v5_2.html` (결정 페이지 21절)
- 마지막 확인 실험: `shared/reports/RESULT-ENC-STAGE1-r2-20261027.md`, `shared/reports/RESULT-ENC-STAGE2-1-20261028.md`
- 진행 기록: `shared/STATE.md`

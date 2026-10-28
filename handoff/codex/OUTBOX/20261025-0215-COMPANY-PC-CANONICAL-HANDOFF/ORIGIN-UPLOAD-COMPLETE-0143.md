# UPLOAD-COMPLETE — CTS 전체 소스·실험 패키지·감사 보고서 업로드 완료 선언

- **완료 시각**: 2026-10-25 01:44 KST
- **작성자**: Antigravity
- **대응 요청 문서**: `OUTBOX/20261025-0130-ASTRA-HANDOFF/REQUEST-ANTIGRAVITY.md`
- **기준 로컬 저장소**: `D:\AI\cts` (브랜치 `scale-v1`, 커밋 `992afa62faa87fbe4e5a58021b7fb2e435291cc9`)
- **전체 매니페스트 파일**: [`SOURCE-MANIFEST.json`](SOURCE-MANIFEST.json) (및 [`INBOX/20261025-0130-full-code-and-docs/MANIFEST-SYNC-20261025.json`](../20261025-0130-full-code-and-docs/MANIFEST-SYNC-20261025.json))

---

## 1. 최신 전체 기준(Authoritative Snapshot) 패키지 목록

업로드가 100% 완료되었습니다. 이 저장소(`heosanghun/CTS_file`)에서 아래 경로들이 최신 전체 기준본입니다:

1. **전체 소스 코드·설정·문서 루트**:
   - [`INBOX/20261025-0130-full-code-and-docs/`](../20261025-0130-full-code-and-docs/)
     - `cts/`: `cts` 파이썬 패키지 전체 (100개 파일; `backbone`, `deq`, `routing`, `train`, `model`, `eval` 등)
     - `handoff_scripts/`: `g2v11_x8a_train.py`, `g2v11_x8a_latents.py`, `cc_x8a.sh`, `g2v9_xattn.py`, `g2v9_gen.py`, `g2v10_train.py`, `codex_grade_local.py`, `s5_eval_grader_v2.py` 등 20개 핵심 실행 스크립트
     - `repo_scripts/`: `D:\AI\cts\scripts\` 내 302개 실험·진단 스크립트
     - `configs/`: `s7_train_993.json`, `p6_item_order.json` 포함 40개 설정 파일
     - `state_and_docs/`: `STATE.md`, `CONTEXT.md`, `GOAL.md`, `RULES.md`, `AGENTS.md`, `proposals/*.md`, `results/RESULT-*.md`(17종)
     - `tasks_and_answers/`: `TASK-C001.md` ~ `TASK-C009.md`, `ACK-*`, `ANSWER-*`
2. **실험별 서버 실행 결과·매니페스트·원시 생성물(`gen_*.jsonl`) 루트**:
   - `X5`: [`INBOX/20261020-1320-x5-results/`](../20261020-1320-x5-results/)
   - `X6`: [`INBOX/20261020-2020-x6-results/`](../20261020-2020-x6-results/)
   - `X7`: [`INBOX/20261021-0945-x7-results/`](../20261021-0945-x7-results/)
   - `X8-0`: [`INBOX/20261022-0015-x8_0-results/`](../20261022-0015-x8_0-results/)
   - `X8-A`: [`INBOX/20261023-1950-x8a-results/`](../20261023-1950-x8a-results/)
3. **독립 감사·재채점 보고서 및 통합 문항별 채점표 루트**:
   - `C004` (X5): [`OUTBOX/20261020-1535-C004/REPORT-C004.md`](../../OUTBOX/20261020-1535-C004/REPORT-C004.md)
   - `C005` (X6): [`OUTBOX/20261024-1859-C005/REPORT-C005.md`](../../OUTBOX/20261024-1859-C005/REPORT-C005.md)
   - `C006` (X7): [`OUTBOX/20261024-1901-C006/REPORT-C006.md`](../../OUTBOX/20261024-1901-C006/REPORT-C006.md)
   - `C007` (X8-0): [`OUTBOX/20261024-1906-C007/REPORT-C007.md`](../../OUTBOX/20261024-1906-C007/REPORT-C007.md)
   - `C008` (X8-A 코드 감사): [`OUTBOX/20261024-1910-C008/REPORT-C008.md`](../../OUTBOX/20261024-1910-C008/REPORT-C008.md)
   - `C009` (X8-A 결과 감사): [`OUTBOX/20261024-1833-C009/REPORT-C009.md`](../../OUTBOX/20261024-1833-C009/REPORT-C009.md)
   - 통합 600문항 × 32조건 채점 CSV: [`OUTBOX/20261025-0130-ASTRA-QA-PACKAGE/all_experiments_item_grades_v1.csv`](../../OUTBOX/20261025-0130-ASTRA-QA-PACKAGE/all_experiments_item_grades_v1.csv)

---

## 2. 보안 및 데이터 경계에 따른 제외 항목 명시

다음 항목들은 보안 규약 및 데이터 경계 규칙에 따라 의도적으로 제외되었습니다:
1. **비밀 자격 증명 및 SSH 인프라 스크립트**:
   - `scratch/`, `results_s5/`, `exe_scratch/`, `.env`, `*.pem`, `*.key`, `shared/cts_ssh.py`, `shared/ssh_config_cts`, `shared/ssh_log.csv`
   - `scripts/` 내 SSH/네트워크 프로브 관련 스크립트 7종 (`build_report_exe40.py`, `cloud_setup.sh`, `net_probe_runner.py`, `progress_heartbeat.py`, `run_local_s7_zstar.py`, `update_rules_protocol.py`, `upload_checkpoints.sh`)
2. **미개봉 테스트 후보 데이터**:
   - `data/stage2/math_train_prompts_5000.jsonl` 원문은 제외되었으며, 허용된 개발셋 600문항(`INBOX/20261020-0750-pkg-b/data/dev600_records.jsonl` 및 `p6_item_order.json`)과 학습셋 ID 목록(`configs/s7_train_993.json`)만 포함됩니다.
3. **대용량 모델 가중치 바이너리 (`*.pt`, `*.safetensors`)**:
   - GitHub 용량 제한 및 보안 규칙에 따라 제외되었으며, 서버 내 경로(`results_s7/mt2_fixed_w0/step_01000.pt`, `results_s7/x8a_readable_deq/` 등)와 메타데이터/매니페스트(`manifest_X8A*.json`, `latent_meta_X8A*.json`)가 포함되었습니다.

이후 추가 변경이나 후속 패키지가 발생할 경우 기존 파일을 덮어쓰지 않고 새로운 타임스탬프 폴더와 새 매니페스트 버전으로 전달합니다.

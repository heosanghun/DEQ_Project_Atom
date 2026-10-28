# AUDIT-0006 — REPORT-0006 (P5 실행기 준비 · P6 실행기 현황) 감사 (CC, 2026-10-04 20:2x)

대상: shared/reports/REPORT-0006.md (1행 SHA 48eb7ee2…), shared/reports/P6_RUNNER_GAP_0006.md, 로컬 scripts/p5_heldout520_launch.sh (4,247 B; 보고서의 서버본 3,099 B와 크기 다름 → 서버본 SHA 5bb9535a…를 기준으로 본다)
근거: 보고서 본문, 실행기 정독, from_laptop/reports/HANDOVER_TO_CTS_20261004.md §Step 2, from_laptop/tasks/PROMPT-5.md. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 포인터 | PASS | 6개 전제 파일 SHA가 기대값과 일치, 채점기 90cc93e4 일치, 스크립트 16종 SHA |
| ② 사전등록 대비 | PASS | 정본 인자 α·g0·σ·seed·max_nfe가 v1.2 §10(e)와 같음. held-out 520 config SHA 48abce65…(mtime 9/27, v1.2 잠금 9/28 이전)를 이번에 처음 고정 기록함 |
| ③ 산술 | PASS | 교집합 520∩1000 = 0, 520∩993 = 0 |
| ④ held-out 봉인 | PASS | ID 집합 연산만 함, 계산·생성 0 |
| ⑤ 규칙 | PASS | 연결 2회(1회차는 원격 파이썬 SyntaxError로 rc 1, 152초 뒤 재실행). 판정 단어 0 |
| ⑥ 실행기 결함 | **FIX** | 아래 F1~F3 |

## FIX
- **F1 인자 불일치(치명)**: 실행기는 `--checkpoint`, `--raw_dir`을 넘기는데, REPORT §1.4의 `--help` 목록(`--items --out --alpha --g0 --seed --sigma --max_nfe --tol --device --config`)에는 둘 다 없다. 이대로 실행하면 argparse "unrecognized arguments"로 즉시 종료된다. P5는 1회 확증 실행이므로, 실행 전에 인자 해석까지 실제로 거치는 시험이 필요하다. `--tol`(정본 0.0)과 `--config`는 넘기지 않으므로 스크립트 기본값이 정본과 같은지도 행 번호로 확인해야 한다.
- **F2 이름 불일치**: 인수인계서와 PROMPT-5가 정한 출력은 `results_s7/zstar_s7_a0.25_heldout520_lowest.pt`, 로그는 `raw_s7/heldout520_P5.log`다. 실행기는 다른 이름(zstar_heldout520_P5.pt, p5_heldout520.log)을 쓴다. 기존 문서 쪽 이름으로 맞춘다.
- **F3 증분 저장·재개 미확인**: PROMPT-5는 "중간 재시작 시 증분 저장으로 이어서 계산(같은 문항 두 번 계산 0)"을 요구한다. 스크립트가 이를 지원하는지 행 번호로 확인되지 않았다.

## P6
- 전용 실행기가 없다는 점, 그리고 4개 갭((i) 순정 Gemma + PrefixProjector, (ii) 가드, (iii) 조건 A~F·600 병합, (iv) 디코딩 일치)을 정리한 것은 타당하다. 새 파일 `scripts/s7_eval_gate_p6_v36.py`로 작성하는 안에 동의한다. P4'가 끝나기 전(약 30시간 남음)에 TASK-0008로 작성·CPU 시험까지 진행한다.

판정: FIX

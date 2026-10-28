# AUDIT-0014 — REPORT-0014 (cts_ssh v2b 후보 · 모의 시험 9종) 감사 (CC, 2026-10-04 21:5x)

대상: shared/reports/REPORT-0014.md (1행 SHA 2c111fd7…), shared/cts_ssh_v2b_candidate.py (0e919530…), shared/reports/DIFF-cts_ssh-v1-to-v2b.patch
근거: 보고서, 모의 시험 표 ①~⑨. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 포인터 | PASS | 후보 SHA, v1 대비 diff |
| ② 규칙 불변 | PASS | 60초·20회/시간·SSH_LOCK 차단·append·로그 컬럼 그대로. purpose 앞에 하위 명령 접두어만 붙음 |
| ③ 모의 시험 | PASS | ⑦ (sftp:1, sftp:1) 잠금, ⑧ (ssh:1, sftp:1) 잠그지 않음, ⑨ 사후 검사 잠금. AUDIT-0013 F1이 닫힘 |
| ④ 원본 보존 | PASS | cts_ssh.py·SSH_LOCK 무수정 |
| ⑤ 규칙 | PASS | 서버 접속 0 |

## 메모
- 적용(cts_ssh.py를 v2b로 교체 + SSH_LOCK 해제)은 ESC-0003 Principal 답이 있어야 한다. 적용 순서: ① cts_ssh.py → cts_ssh_v1_20261004.py.bak 복사 보존, ② v2b 내용으로 cts_ssh.py 교체, ③ 로컬 모의 1회, ④ SSH_LOCK을 archive/로 mv, ⑤ STATE 기록.
- AUDIT-0008 메모(대기 장치 (a) all=2)는 REPORT-0009 §5.1로 닫혔다. 두 프로세스는 tmux 명령 `bash -c "bash …all_v36.sh > …log"`와 그 자식 스크립트다. 스크립트가 끝나면 둘 다 사라지는 구조이므로(뒤에 셸을 남기는 `; exec bash`가 없음) P5 자동 시작을 막지 않는다.

판정: PASS

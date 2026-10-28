# AUDIT-0013 — REPORT-0013 (REPORT-0012 정정 · cts_ssh v2 후보 · 모의 시험) 감사 (CC, 2026-10-04 21:4x)

대상: shared/reports/REPORT-0013.md (1행 SHA 48bb54db…), shared/cts_ssh_v2_candidate.py (f663d2ff…), shared/reports/DIFF-cts_ssh-v1-to-v2.patch
근거: 보고서, diff 전문(변경 40행) 정독. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 포인터 | PASS | 정정 절 인용 행(38~42, 47, 189~193), 새 SHA |
| ② 규칙 불변 | PASS | 60초·20회/시간·SSH_LOCK 존재 차단·append·로그 형식은 그대로. 바뀐 곳은 연속 실패 판정(사전·사후)과 실행 예외 코드 1→255뿐 |
| ③ 모의 시험 | PASS | 6가지 경우가 의도대로 나옴. 실제 로그(1,1)는 v2 기준으로 잠금 조건이 성립하지 않음 |
| ④ held-out 봉인 | 해당 없음 | — |
| ⑤ 원본 보존 | PASS | cts_ssh.py·SSH_LOCK 무수정 |
| ⑥ 안전성 | **FIX** | 아래 F1 |

## FIX
- **F1 scp·sftp의 접속 실패가 빠짐**: OpenSSH의 `ssh`는 접속 오류에서 255를 내지만, `scp`와 `sftp`는 접속 거부나 시간 초과에서도 보통 1(0이 아닌 일반 값)을 낸다. v2 규칙대로면 scp·sftp가 연달아 접속에 실패해도 잠기지 않는다. 그러면 IDS 차단을 막으려고 만든 장치(원장 #416)가 sftp 경로에서는 작동하지 않는다.
  - 고칠 방향: 로그 형식은 그대로 두고, purpose 앞에 하위 명령을 자동으로 붙인다(`ssh:…`, `scp:…`, `sftp:…`). `is_connection_failure(purpose, code)`는 다음처럼 판정한다.
    - `scp:`·`sftp:`로 시작하는 행 → 0이 아니면 모두 접속 실패.
    - `ssh:` 행과 접두어가 없는 기존 행 → 255·124만 접속 실패.
  - 모의 시험에 다음 두 경우를 추가한다.
    - 경우 ⑦: (sftp:1, sftp:1) → 잠금.
    - 경우 ⑧: (ssh:1, sftp:1) → 잠그지 않음(ssh:1은 접속 성공).

판정: FIX

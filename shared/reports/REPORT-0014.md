2c111fd7c264037ece0c77b134b7e41715e6356108502160d2bc9b1a5eaa6730
# REPORT-0014: cts_ssh v2b 후보 보완 및 9대 모의 시험 결과 보고서

- **보고서 ID**: REPORT-0014
- **작성 일시**: 2026-10-04 21:45 KST
- **지침**: shared/tasks/TASK-0014.md (cts_ssh v2 후보 보완: scp·sftp 접속 실패 판정)
- **근거 문서**: audits/AUDIT-0013.md (FIX F1), escalate/ESC-0003.md (선택지 [A] 준비), RULES.md §7 (원장 #399·#416)
- **실행 환경**: 회사 PC 로컬 (서버 접속 0건, shared/SSH_LOCK 기계적 잠금 상태 유지)
- **절대 제약 준수**:
  - 서버 접속 호출 0건.
  - 기존 `shared/cts_ssh.py` 및 `shared/SSH_LOCK` 무수정·무삭제·무이동 0건 보존 (신규 후보 파일 `shared/cts_ssh_v2b_candidate.py` 작성).
  - 모의 시험은 환경 변수(`CTS_SSH_LOG_PATH`, `CTS_SSH_LOCK_PATH`, `CTS_SSH_MOCK_EXIT`)를 활용하여 `scratch/mock_ssh_v2b` 하위 가짜 로그 및 가짜 잠금으로 격리 실행.
- **판정 어휘 통제**: 지침 규정 금지 단어 9종 사용 횟수 0건 (수치·로그·코드 사실 기술 전재).

---

## 1. `shared/cts_ssh_v2b_candidate.py` 후보 명세 및 F1 반영 내역

### 1.1 AUDIT-0013 F1 문제 배경
- OpenSSH 명령별 종료코드 특성 차이:
  - `ssh`: 호스트 접속 오류 시 종료코드 255 반환.
  - `scp` 및 `sftp`: 원격 호스트 접속 거부나 네트워크 단절 시에도 종료코드 1(일반 0 아닌 오류 코드) 반환.
- v2 단순 규칙(255·124만 접속 실패 판정) 적용 시, scp/sftp 연결 실패(exit 1)가 연속 발생해도 접속 실패로 인식되지 않아 IDS 차단 방지 목적(원장 #416)이 훼손될 위험 식별.

### 1.2 v2b 수정 로직 (`is_connection_failure`)
- **하위 명령 접두어 자동 부여**:
  - `append_log` 호출 시 purpose 필드 전면에 실행된 하위 명령 접두어(`ssh:`, `scp:`, `sftp:`)를 자동 첨부 (기존 사용자 purpose 보존).
- **접속 실패 2원 판정 규칙**:
  - `scp:` 또는 `sftp:` 로 시작하는 로그 행: 0이 아닌 모든 종료코드(exit!=0)를 접속 실패로 판정.
  - `ssh:` 로 시작하는 로그 행 및 접두어가 없는 과거 로그 행: 오직 종료코드 255(접속 오류) 및 124(시간 초과)만 접속 실패로 판정 (원격 명령 exit!=0은 연결 도달 상태로 판독).
- **머리 주석 기재**:
  ```python
  # 변경 사유: ESC-0003, AUDIT-0009 ⑤, AUDIT-0013 F1 — 연속 접속 실패 판정을 ssh(255·124) 및 scp/sftp(0이 아닌 모든 exit)로 정밀화하고 하위 명령 접두어(ssh:, scp:, sftp:) 자동 부여
  ```
- **기존 거버넌스 불변**:
  - 60초 간격 강제(R7-1), 최근 1시간 20회 상한, `SSH_LOCK` 파일 존재 시 실행 차단(exit 2), 단일 행 append 기록 의무 유지.
  - 로그 파일 컬럼 구조(`timestamp,purpose,exit_code`) 변경 없음.

---

## 2. v1 대비 v2b 차이점 패치 (`shared/reports/DIFF-cts_ssh-v1-to-v2b.patch`)

- **패치 대상**: `shared/cts_ssh.py` (v1 정본) $\to$ `shared/cts_ssh_v2b_candidate.py` (v2b 후보).
- **주요 diff 영역**:
  - `is_connection_failure(purpose, exit_code)` 함수 신설.
  - `check_consecutive_failures_pre(records, lock_path)` 및 `check_consecutive_failures_post(records_before, current_purpose, current_exit, lock_path, now_str)`에 `purpose` 인자 연동.
  - `main()`에서 `subcmd:` 접두어 자동 합성 블록 추가.
  - subprocess 실행 예외 시 `exit_code = 255` 할당 (로컬 실행 예외를 접속/실행 실패로 포착).

---

## 3. 격리 모의 시험 결과 (`scratch/test_cts_ssh_v2b_mock.py`)

서버 접속 0건 환경에서 환경 변수 격리를 통해 9대 경우 및 실제 `ssh_log.csv` 판독 시험을 수행함.

### 3.1 9대 시험 경우별 계측 결과 표

| 경우 | 시험 경우명 | 직전 조건 / 목적 | 호출 서브커맨드 | 모의 반환값 | 반환코드 | 잠금 여부 | 출력 첫 줄 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| ① | `case1_exit1_1` | 직전 (ssh:1, ssh:1) | ssh | MOCK_EXIT=0 | **0** | 잠금 미생성 (False) | `[MOCK_EXEC] CTS_SSH_MOCK_EXIT=0 (서버 연결 0)` |
| ② | `case2_exit255_255` | 직전 (ssh:255, ssh:255) | ssh | 사전 차단 | **2** | 잠금 생성 (True) | `[BLOCKED] 직전 2행 모두 접속 실패 감지 ([ssh:conn_refused1:255], [ssh:conn_refused2:255]).` |
| ③ | `case3_exit124_255` | 직전 (ssh:124, ssh:255) | ssh | 사전 차단 | **2** | 잠금 생성 (True) | `[BLOCKED] 직전 2행 모두 접속 실패 감지 ([ssh:conn_timeout:124], [ssh:conn_error:255]).` |
| ④ | `case4_exit255_1_then255` | 직전 (ssh:255, ssh:1) | ssh | MOCK_EXIT=255 | **255** | 잠금 미생성 (False) | `[MOCK_EXEC] CTS_SSH_MOCK_EXIT=255 (서버 연결 0)` |
| ⑤ | `case5_existing_lock` | `SSH_LOCK` 파일 사전 존재 | ssh | 사전 차단 | **2** | 잠금 유지 (True) | `[BLOCKED] shared/SSH_LOCK 존재: .../SSH_LOCK` |
| ⑥ | `case6_cooldown_blocked` | 10초 전 호출 이력 | ssh | 사전 차단 | **1** | 잠금 미생성 (False) | `R7-1 대기 필요 50초` |
| ⑦ | `case7_sftp1_sftp1` | 직전 (sftp:1, sftp:1) | sftp | 사전 차단 | **2** | 잠금 생성 (True) | `[BLOCKED] 직전 2행 모두 접속 실패 감지 ([sftp:download_file1:1], [sftp:download_file2:1]).` |
| ⑧ | `case8_ssh1_sftp1` | 직전 (ssh:1, sftp:1) | sftp | MOCK_EXIT=0 | **0** | 잠금 미생성 (False) | `[MOCK_EXEC] CTS_SSH_MOCK_EXIT=0 (서버 연결 0)` |
| ⑨ | `case9_sftp1_then_sftp1` | 직전 (sftp:1) | sftp | MOCK_EXIT=1 | **1** | 잠금 생성 (True) | `[MOCK_EXEC] CTS_SSH_MOCK_EXIT=1 (서버 연결 0)` |

### 3.2 신규 3개 경우(⑦·⑧·⑨) 상세 동작 분석
- **경우 ⑦ (sftp:1, sftp:1 $\to$ 잠금)**:
  - `sftp:` 접두어를 가진 행은 종료코드 1이 접속 실패로 판독됨.
  - 직전 2행이 모두 접속 실패이므로 사전 검사(`check_consecutive_failures_pre`)에서 `SSH_LOCK`을 생성하고 즉시 차단(코드 2).
- **경우 ⑧ (ssh:1, sftp:1 $\to$ 잠그지 않음)**:
  - `ssh:1`은 ssh 접속 후 원격 명령이 오류 종료된 것이므로 네트워크 연결 자체는 유효했던 상태로 판독됨.
  - 직전 2행 중 첫 행이 접속 실패가 아니므로 연속 접속 실패가 성립하지 않아 사전 검사 차단 없이 계속 진행 (코드 0).
- **경우 ⑨ (직전 sftp:1, 이번 sftp 실행 exit 1 $\to$ 사후 검사 잠금)**:
  - 직전 행이 `sftp:1`(접속 실패) 상태에서 이번 sftp 호출이 exit 1(접속 실패)로 마침.
  - 사후 검사(`check_consecutive_failures_post`)에서 2회 연속 접속 실패를 감지하여 `SSH_LOCK`을 생성함 (코드 1 반환).

### 3.3 실제 `shared/ssh_log.csv` 판독 계측 (읽기 전용)
- 전체 로그 37건은 모두 과거 행으로 접두어가 없음.
- 최신 2개 행: `task0009_p6_runner_and_cpu_smoke (exit 1)`, `task0009_p6_runner_and_cpu_smoke_retry (exit 1)`.
- v2b 판정 규칙: 접두어가 없는 과거 행은 `ssh:`와 동일하게 255·124만 접속 실패로 판독하므로, exit 1은 원격 명령 오류(네트워크 연결 유효)로 분류됨.
- v2b 기준 사전 잠금 성립 여부: `False` (성립하지 않음).
- 결론: v2b 규칙 하에서도 과거 원격 스크립트 실행 오류(1, 1)로 인한 부당 차단이 발생하지 않음을 정적 판독으로 대조함.

---

## 4. 파일 메타데이터 및 SHA-256 해시

| 번호 | 파일 경로 | 실측 크기 (Bytes) | SHA-256 해시 | 비고 |
|:---:|:---|:---:|:---|:---|
| 1 | `shared/cts_ssh_v2b_candidate.py` | 11,122 | `0e919530974b42ac69a3e84e549cd759e1e00ba0a72e573086404081a7dbda23` | v2b 후보 (scp/sftp exit!=0 및 ssh 255·124 접속 실패 판정) |
| 2 | `shared/reports/DIFF-cts_ssh-v1-to-v2b.patch` | 8,278 | `fff4660ab2e3acd3cdeef9dda6a370181d6f7701b42554e47534b2ea9d0ff142` | v1 정본 대비 v2b 변경 패치 |

---

## 5. 서버 접속 통제 현황
- `shared/SSH_LOCK` 파일 잠금 상태 유지 (`Reason: 2 consecutive failures: last=1, current=1`).
- 원격 서버 접속 횟수: **0건** (규약 R7-2, R7-8 준수).
- `shared/cts_ssh.py` 정본 무수정 보존 상태 유지.
- P4' 디코더 직렬 학습 및 tmux `p5_waiter` 세션 무간섭 지속 가동 중.
- Principal 거버넌스 결정 대기:
  - `escalate/ESC-0003.md`: `cts_ssh_v2b_candidate.py` 채택 및 SSH_LOCK 수동 해제 결정 대기.
  - `escalate/ESC-0004.md`: G1 최근접 일치율 조작적 정의 결정 대기.

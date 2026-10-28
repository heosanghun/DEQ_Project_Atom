48bb54dbcaf260033382cb3ad964170bc1d0d5e5ed27eb557137ac68bd81dbf7
# REPORT-0013: REPORT-0012 정정 및 cts_ssh.py v2 후보 작성·모의 시험 결과 보고서

- **보고서 ID**: REPORT-0013
- **작성 일시**: 2026-10-04 21:35 KST
- **지침**: shared/tasks/TASK-0013.md (REPORT-0012 정정 + cts_ssh.py 수정 후보(v2) 작성·모의 시험)
- **근거 문서**: audits/AUDIT-0012.md (FIX F1), escalate/ESC-0003.md (선택지 [A] 준비), RULES.md §7 (원장 #399·#416)
- **실행 환경**: 회사 PC 로컬 (서버 접속 0건, shared/SSH_LOCK 기계적 잠금 상태 유지)
- **절대 제약 준수**:
  - 서버 접속 호출 0건.
  - 기존 `shared/cts_ssh.py` 및 `shared/SSH_LOCK` 무수정·무삭제·무이동 0건 보존 (후보 파일 `shared/cts_ssh_v2_candidate.py` 신규 생성).
  - 모의 시험은 환경 변수(`CTS_SSH_LOG_PATH`, `CTS_SSH_LOCK_PATH`, `CTS_SSH_MOCK_EXIT`)를 활용하여 `scratch/mock_ssh_v2` 하위 가짜 로그 및 가짜 잠금으로 격리 실행.
- **판정 어휘 통제**: 지침 규정 금지 단어 9종 사용 횟수 0건 (수치·로그·코드 사실 기술 전재).

---

## 1. REPORT-0012 정정 절 추가 내역 (AUDIT-0012 F1)

- **정정 조치 방식**: 기존 보고서 본문을 덮어쓰지 않고, 파일 말미에 "## 정정 (AUDIT-0012 F1)" 절을 덧붙임.
- **오기 2개 문장 대조 및 정정 내용**:
  1. *디바이스 기본값*: §2.1의 "기본값: `cuda:1`" 표기를 실제 코드 47행 `CLI_DEVICE="cuda:0"`으로 정정.
  2. *정본 파라미터 인자*: §2.1의 "`--alpha 0.25`, `--max_iters 1000`, `--tol 1e-4`, `--damping 0.5`, `--lm_lambda 1.0`, `--lm_adaptive 1`, `--prompt_idx 5`" 기술을 실제 호출 인자(`--alpha 0.25`, `--g0 0.25`, `--seed 2031`, `--sigma 0.020`, `--max_nfe 1000`, `--tol 0.0`)로 정정하고 미존재 인자(damping, lm, prompt_idx)를 명시.
- **실제 소스 코드(`scripts/p5b_seed2031_launch.sh`) 인용 행**:
  - 38~42행 (파라미터 정의 블록)
  - 47행 (`CLI_DEVICE="cuda:0"` 설정 행)
  - 189~193행 (파이썬 스크립트 호출 인자 블록)
- **갱신된 REPORT-0012.md 1행 SHA-256 해시**: `c079cdb214f8652b398dc19f56652078507954f03258ab663f5a7274fb6b8ccc`

---

## 2. `shared/cts_ssh_v2_candidate.py` 후보 파일 명세 및 차이점

### 2.1 주요 변경 사항
- **머리 주석 기재**:
  ```python
  # 변경 사유: ESC-0003 및 AUDIT-0009 ⑤ — 연속 접속 실패 판정을 ssh/scp/sftp 접속 오류(255), 시간 초과(124), 실행 예외(255)로 한정하고 원격 명령 오류(exit!=0)는 잠금 계산에서 제외
  ```
- **접속 실패 정의 한정 (`is_connection_failure`)**:
  - SSH 네트워크/인증/호스트 오류(종료코드 255), 시간 초과(124), 로컬 프로세스 실행 예외(255)에 한정.
  - 원격 명령의 실행 오류(종료코드 1, 2 등)는 `ssh_log.csv`에 그대로 기록하되, 연속 실패 판정(사전 및 사후 검사) 대상에서 완전히 배제.
- **규약 거버넌스 불변 유지**:
  - 60초 간격 강제(R7-1), 최근 1시간 20회 상한, `SSH_LOCK` 파일 존재 시 실행 차단(exit 2), 단일 행 append 기록 의무 유지.
  - 로그 파일 형식(`timestamp,purpose,exit_code`) 변경 없음.
- **패치 파일 생성**: `shared/reports/DIFF-cts_ssh-v1-to-v2.patch`

---

## 3. 격리 모의 시험 결과 (`scratch/test_cts_ssh_v2_mock.py`)

서버 접속 0건 환경에서 환경 변수 격리를 통해 6대 경우 및 실제 `ssh_log.csv` 판독 시험을 수행함.

### 3.1 6대 시험 경우별 계측 결과 표

| 경우 | 시험 경우명 | 사전 조건 | 모의 반환값 | 반환코드 | 잠금 여부 | 출력 첫 줄 |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| ① | `case1_exit1_1` | 직전 2행 exit=(1, 1) | MOCK_EXIT=0 | **0** | 잠금 미생성 (False) | `[MOCK_EXEC] CTS_SSH_MOCK_EXIT=0 (서버 연결 0)` |
| ② | `case2_exit255_255` | 직전 2행 exit=(255, 255) | 사전 차단 | **2** | 잠금 생성 (True) | `[BLOCKED] 직전 2행 모두 접속 실패(exit 255 또는 124) (exit1=255, exit2=255).` |
| ③ | `case3_exit124_255` | 직전 2행 exit=(124, 255) | 사전 차단 | **2** | 잠금 생성 (True) | `[BLOCKED] 직전 2행 모두 접속 실패(exit 255 또는 124) (exit1=124, exit2=255).` |
| ④ | `case4_exit255_1_then255` | 직전 2행 exit=(255, 1) | MOCK_EXIT=255 | **255** | 잠금 미생성 (False) | `[MOCK_EXEC] CTS_SSH_MOCK_EXIT=255 (서버 연결 0)` |
| ⑤ | `case5_existing_lock` | `SSH_LOCK` 사전 존재 | 사전 차단 | **2** | 잠금 유지 (True) | `[BLOCKED] shared/SSH_LOCK 존재: .../SSH_LOCK` |
| ⑥ | `case6_cooldown_blocked` | 10초 전 호출 이력 | 사전 차단 | **1** | 잠금 미생성 (False) | `R7-1 대기 필요 50초` |

### 3.2 경우 ④ 결과 및 상세 이유 보고
- **계측 결과**: 종료코드 255 반환, `SSH_LOCK` 파일 미생성.
- **동작 이유**:
  - 직전 실행 기록의 종료코드가 1(원격 명령 오류)이었으므로, 해당 시점의 SSH 접속 자체는 연결에 도달했음을 나타냄.
  - v2 규칙 하에서 종료코드 1은 접속 실패가 아니므로 연속 실패 시퀀스가 단절됨.
  - 따라서 이번 호출에서 발생한 종료코드 255는 고립된 1차 접속 실패로 규정되며, 사후 검사 시 직전 이력(`exit_code=1`)이 `is_connection_failure` 조건을 만족하지 않아 잠금을 생성하지 않고 종료코드 255로 마침.

### 3.3 실제 `shared/ssh_log.csv` 판독 계측 (읽기 전용)
- 총 37건 기록 보유.
- 최신 2개 행: `2026-10-04 20:47:32 (exit=1)`, `2026-10-04 20:49:09 (exit=1)` (원격 CPU 채점 스크립트 실행 오류 기록).
- v2 기준 접속 실패(`exit_code in {255, 124}`) 판독: 2개 행 모두 `False`.
- v2 기준 사전 잠금 성립 여부: `False` (잠금 조건 미성립).
- 결론: v2가 적용될 경우 원격 명령 오류(1, 1)로 인한 접속 차단이 발생하지 않음을 정적 판독으로 대조함.

---

## 4. 파일 메타데이터 및 SHA-256 해시

| 번호 | 파일 경로 | 실측 크기 (Bytes) | SHA-256 해시 | 비고 |
|:---:|:---|:---:|:---|:---|
| 1 | `shared/reports/REPORT-0012.md` | 8,544 | `c079cdb214f8652b398dc19f56652078507954f03258ab663f5a7274fb6b8ccc` | F1 정정 절 덧붙임 및 1행 self-hash 갱신 마침 |
| 2 | `shared/cts_ssh_v2_candidate.py` | 10,087 | `f663d2ff235ade1a1bfddec938793bb247ece870c5da1b6b3aab21a7c570ea03` | SSH 제어 v2 후보 (연속 접속 실패 판정 범위 한정) |
| 3 | `shared/reports/DIFF-cts_ssh-v1-to-v2.patch` | 6,252 | `a0016ca7202f924382c29ec764d88ec70e12a9b7dec8601c6c7d143fbef727ae` | v1 대비 v2 변경 패치 |

---

## 5. 서버 접속 통제 현황
- `shared/SSH_LOCK` 파일 잠금 상태 보존 (`Reason: 2 consecutive failures: last=1, current=1`).
- 원격 서버 접속 횟수: **0건** (규약 R7-2, R7-8 준수).
- `shared/cts_ssh.py` 정본 무수정 보존 상태 유지.
- P4' 디코더 직렬 학습 및 tmux `p5_waiter` 세션 무간섭 지속 가동 중.
- Principal 거버넌스 결정 대기:
  - `escalate/ESC-0003.md`: `cts_ssh_v2_candidate.py` 채택 및 SSH_LOCK 수동 해제 결정 대기.
  - `escalate/ESC-0004.md`: G1 최근접 일치율 조작적 정의 결정 대기.

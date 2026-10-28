# Agent Instructions

## GPU Usage Policy
- **GPU 사용 지침 (GPU Usage Policy, CTS-428 2026-10-06)**: GPU 0·1 사용 허용(GPU 1은 Principal 10-06 허락). GPU 4 영구 봉인. GPU 2·3·5·6·7 사용 0.
  - CUDA_VISIBLE_DEVICES·`--device`·`docker run --gpus`에는 GPU 0 또는 1만 넣는다. 그 밖의 GPU는 Principal 허락 전 사용 0.
- **GPU 4 예외**: GPU 4는 System 1.5 봉인 실행 전용으로 예약. 어떤 스크립트도 GPU 4를 잡지 않으며 CUDA_VISIBLE_DEVICES에 4를 포함하지 않는다. 변경은 System 2 원장 기록으로만. (09-02 규칙, 원장 #203)

## CPU 진단 및 보조 연산 스크립트 상시 규칙 (원장 #381)
- **CPU 메모리 및 안전 상시 규약**:
  - **(a) Autograd 완전 차단**: 모든 CPU 진단·평가·검증 스크립트는 `torch.no_grad()` 컨텍스트 및 `model.eval()`을 기본 강제하며, 걸음 간 연산 그래프(computation graph) 및 중간 텐서 보존을 0으로 통제한다.
  - **(b) 가상 메모리 상한 강제 (`ulimit -v`)**: 스크립트 실행 wrapper 또는 셸 명령 실행 시 반드시 `ulimit -v 62914560` (가상 메모리 상한 60 GB)를 강제 설정하여 시스템 스왑 고갈 및 thrashing을 원천 차단한다.
  - **(c) 주기적 메모리 로깅**: 장시간 실행되는 CPU 스크립트는 실행 중 5분마다 `free -g` 및 현재 RSS 메모리를 로그에 기록한다.
  - **위반 시 제재**: 상기 3대 규약을 위반한 스크립트는 실행 수 0건으로 간주하며 즉시 중단한다.

## 7. 서버 접속 예절 규약 v1 (원장 #399, 영구)
- **배경**: 09-28 11:46 이후 회사 IP <REDACTED_IP> 가 반복 접속으로 "공격 대상"에 등록돼 차단됨(10-01 담당자 해제). 재발 시 루프 전체가 사흘 멈춤.
- R7-1 **접속 횟수 상한**: 새 SSH 연결은 60초에 1회, 1시간에 20회 이하. 연결마다 `shared\ssh_log.csv` 에 시각·목적·exit 1행 기록.
- R7-2 **재시도 금지**: 타임아웃·거부가 2회 연속이면 즉시 중단, `STATE.md`에 "접속 불가 [시각]"만 적고 Principal 결정 대기. 자동 재시도 루프·프로브 스크립트·BatchMode 탐색·-v 반복 테스트·Test-NetConnection 반복 전부 금지.
- R7-3 **묶어서 한 번에**: 서버에서 할 명령은 하나의 ssh 호출에 모아 보냄(`"cmd1; cmd2; cmd3"` 또는 스크립트 파일을 scp 후 1회 실행). 상태 확인 하나 보려고 따로 접속하지 않음.
- R7-4 **긴 작업 폴링은 서버 안에서**: 완료 대기는 접속을 반복하지 말고 한 연결 안에서 `while …; do sleep 300; done` 으로 기다림(`ServerAliveInterval=60` 으로 끊김 방지). 모든 긴 작업은 tmux s5_exe40 안.
- R7-5 **인증 실패 0건**: `~/.ssh/config`의 cts-server 항목에 `IdentitiesOnly yes` + 정확한 `IdentityFile` 1개만. 에이전트에 키가 여러 개면 틀린 키를 차례로 내밀어 "실패 로그인"으로 집계되므로 금지. 비밀번호 인증 시도 금지.
- R7-6 **config 고정값**: `ServerAliveInterval 60`, `ServerAliveCountMax 3`, `ConnectTimeout 15`, `StrictHostKeyChecking accept-new`, `IdentitiesOnly yes`.
- R7-7 **실행자는 한 곳만**: 인수인계(원장 #410, #413)에 따라 회사 PC 정식 루프(cts, `D:\AI\cts`)가 유일한 서버 실행을 전담한다. 노트북 임시 루프(cts_260930)는 종료됨.
- R7-8 **위반 시 거부**: 위 규약 위반 명령은 실행 전에 스스로 걸러 보고서에 "규약 7-2로 미실행"이라고 적음.
- R7-9 **cts_ssh.py 전용 (기계적 차단)**: 서버 접속(ssh·scp·sftp)은 shared\cts_ssh.py로만. 직접 ssh·scp·sftp 실행 금지. 서버 코드 열람은 필요한 파일을 한 번에 sftp로 받아 로컬에서 읽는다.

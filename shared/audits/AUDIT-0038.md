# AUDIT-0038 — REPORT-0038 (디코더 E 최종본·P5 자동 재개) 감사 (CC, 2026-10-06 10:0x)

대상: shared/reports/REPORT-0038.md (머리 SHA 31c59b00…)
근거: 보고서, 로컬 scripts/p5_resume_after_p4prime_waiter.sh(892ad12d…), scripts/p5_heldout520_launch_v8.sh(6f0cbcd2…). 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ⓪ 재부팅 | 없음 | uptime -s 16:32:53 불변(17h19m) |
| ① 디코더 E 최종본 | PASS | `[OUT-CHECKPOINT] … decoder_E_v36.pt (SHA f8115ac3…)`, 08:59, 236,027,215 B, 가드 `[PASS]` rc 0, 36,026 s |
| ② 대기 장치 | **결함** | D=1·E=1·GPU 0 유휴·P5 미종결인데 `procs=1`로 실행되지 않음(08:59 이후 약 1시간 공회전) |
| ③ P5 봉인 | PASS | 수치 인용 0, 키 150, 로그 헤더 153 |
| ④ 규칙 | PASS | 접속 1회(09:51:47), 읽기만 |

## 결함 분석
- waiter 35행 `pgrep -f "s7_train_p4_prime.*\.sh|s7_train_decoder"`는 명령줄 전체를 본다. 19:54:38에 tmux 서버를 처음 띄운 명령(`tmux new-session … s7_train_p4_prime … s7_train_p4_prime_resume2_v36.sh`)이 tmux 서버의 명령줄로 남아 이 패턴에 걸리는 것으로 추정한다. D 학습 중 4, E 학습 중 3, 지금 1로 줄어든 것도 이 추정과 맞다(남은 1 = tmux 서버).
- 대기 장치 설계를 감사(AUDIT-0033)할 때 CC도 이 경우를 놓쳤다. 이후 pgrep 기반 대기 장치는 `pgrep -f`가 아니라 python 프로세스만 세도록(`pgrep -f "python.*s7_train_decoder"`) 쓴다.
- 조치: TASK-0039(진단 후, tmux 서버뿐이면 CANCEL 플래그로 대기 장치를 끝내고 같은 명령으로 P5를 GPU 0에서 수동 1회 실행). 사전등록 v1.3 §10(f) 경로와 같다(GPU 0, v8, P5_RESUME=1).

판정: PASS(보고서) / 대기 장치 결함 → TASK-0039

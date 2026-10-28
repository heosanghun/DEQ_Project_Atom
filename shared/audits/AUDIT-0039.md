# AUDIT-0039 — REPORT-0039 (대기 장치 오탐 진단·P5 GPU 0 수동 재개) 감사 (CC, 2026-10-06 10:1x)

대상: shared/reports/REPORT-0039.md (머리 SHA f6a6d379…)
근거: 보고서, ssh_log.csv. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 진단 | PASS | 유일한 일치 프로세스 PID 35990 = `tmux new-session -d -s s7_train_p4_prime bash …resume2_v36.sh`(PPID 1). python 학습 0, GPU 0 13 MiB → 규칙 (가). CC 추정과 일치 |
| ② 이중 실행 방지 | PASS | CANCEL 플래그 09:57:04 → waiter `[CANCEL] 10:00:00 … Halting waiter.` 확인 뒤 10:00:04에 `p5_gpu0` 기동 |
| ③ 실행기 | PASS | v8 SHA 6f0cbcd2… 일치, 명령은 waiter 82행과 같음 |
| ④ 장치 | PASS | PID 167612 `CUDA_VISIBLE_DEVICES=GPU-71be98b0-…`(물리 GPU 0), `P5_RESUMED.flag` UUID·PCI 6F:00.0, 기존 키 150·출력 SHA 7177cffa…(REPORT-0024와 같음). GPU 1~7 0% |
| ⑤ 재개 정합 | PASS | Skipping 150, 신규 151·152 계산(재부팅 전 계산됐으나 저장 전이던 151~153은 다시 계산) |
| ⑥ P5 봉인 | PASS | 수치 인용 0 |
| ⑦ tmux 서버 | PASS | kill 0, p6_chain 대기 유지 |

## 메모
- 문항당 시간: 첫 2문항 기준 약 64 s로, P5b(GPU 2) 22.8 s보다 길다. 모델 적재 직후 값일 수 있으므로 11시경 다시 잰다. 64 s가 유지되면 16:45경, 23 s면 12:30경 끝난다.
- 기록할 제한(FINAL): P5는 1~150번을 GPU 1에서, 151~520번을 GPU 0에서 계산한 혼합 장치 실행이다. GPU 0 앵커 대조(p6_chain 0단계)가 이를 점검한다.
- 보고서 작성 시각(10:15)이 실제 파일 시각(10:07 이전)보다 늦게 적혀 있다(경미, 반복). 다음 TASK에 실제 시각을 쓰도록 적는다.

판정: PASS

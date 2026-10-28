# AUDIT-0036 — REPORT-0036 (디코더 D 종료·E 시작·재부팅 여부) 감사 (CC, 2026-10-05 23:2x)

대상: shared/reports/REPORT-0036.md (머리 SHA 40b6ca3a…, 본문 재계산 일치, CRLF 0)
근거: 보고서, ssh_log.csv. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ⓪ 재부팅 | 없음 | uptime -s 16:32:53 불변, tmux 3세션(19:54:38 생성) 유지 |
| ① 디코더 D 최종본 | PASS | `[OUT-CHECKPOINT] … decoder_D_v36.pt (SHA 1a09144b…)`, 236,027,279 B, 22:58, 가드 `[PASS]` rc 0. step3000 체크포인트 655d64c6… |
| ② 디코더 E 시작 | 진행 | 22:58:43 시작, configs/s7_train_993.json 993문항, z_src `results_s7/zsrc_train1000_P4.pt` 키 1000, step 1 8.24 s/step → 10/6 05:50경 종료 예상 |
| ③ 대기 장치 | PASS | waiter `D=1, E=0, state=WAITING`, p6_chain `Waiting for P5_DONE.flag` |
| ④ GPU | PASS | GPU 0만 사용(PID 65822, 23.1 GiB). GPU 1~7 0%, GPU 4 점유 0 |
| ⑤ 규칙 | PASS | 접속 1회(23:14:12, rc 0), 읽기만, P5 값 열람 0 |

## 메모
- D의 마지막 step 행 s/step(3.35~3.60)이 앞 구간(7.3)과 다르다. 1501~3000 전체 소요(약 184분)는 평균 7.3 s/step과 맞으므로, 로그의 s/step은 직전 구간 측정값으로 보이며 판정에 영향이 없다. E 시간 추정은 이후 점검에서 다시 잰다.
- **CC 정정**: ESC-0006에 쓴 "16:29 GPU 0 42.9 GB = 외부 프로세스 가능성"은 틀렸다. D 로그의 peak_mem 39.6 GiB가 이를 설명한다. ESC-0006에 정정 줄을 넣었다.
- GPU 1은 16:33 재부팅 뒤 정상 유휴 상태다. ESC-0005에 갱신 줄을 넣었다(관리자 재설정 불필요, GPU 0 유지 권고).
- 보고서 작성 시각(23:20)이 파일 생성(23:16 이전)보다 늦게 적혀 있다. 경미한 표기 문제다.

판정: PASS

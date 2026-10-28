# AUDIT-0008 — REPORT-0008 (P5 실행기 v4 · P5 자동 시작 대기 장치) 감사 (CC, 2026-10-04 20:4x)

대상: shared/reports/REPORT-0008.md (1행 SHA af9606d3…), 로컬 사본 scripts/p5_after_p4prime_waiter.sh (SHA 3ae95960… = 서버와 같음), scripts/p5_heldout520_launch_v4.sh (31d7a955… = 서버와 같음)
근거: 보고서 본문, 대기 장치 정독. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 포인터 | PASS | 9개 전제 파일 SHA, 종료 행 문구 s7_train_decoder_v36.py:1091, waiter.log 첫 행 |
| ② 사전등록 대비 | PASS | v1.2 §10(f) ⑥→⑦ 직렬을 유지함. P5는 1회만 실행되도록 STARTED flag가 막음 |
| ③ 산술 | PASS | 12.44 s/step × 2,500 = 8시간 38분 → A 10/5 05:12. D·E는 각 3,000 step으로 E 종료 10/6 01:56 |
| ④ held-out 봉인 | PASS | 시험은 train Q#14만. held-out 계산은 대기 장치 조건 충족 뒤 1회 |
| ⑤ 규칙 | PASS | 연결 1회, 판정 단어 0 |
| ⑥ 게이트 진전 | 진전 | ⑥→⑦ 사이에 사람이 개입하지 않아도 됨 |

## 메모 (다음 TASK 상태 점검에 포함)
- 대기 장치 (a)의 `all=2`: `pgrep -f s7_train_p4_prime_all`에 걸리는 프로세스 2개의 명령행을 확인해야 한다. tmux 세션의 부모 셸(`bash -c "…s7_train_p4_prime_all_v36.sh…"` 형태)이 학습이 끝난 뒤에도 남는다면 (a)가 영원히 거짓이 되어 P5가 시작되지 않는다. 이 경우는 안전한 쪽(미실행)이므로 지금 고치지는 않는다. 대신 명령행을 확인해 두고, 남는 형태라면 대기 장치 v2를 만든다.
- 중간 체크포인트가 `results_s5/decoder_A_v36_step0500.pt`로 저장된다(output_dir 기본값). P6에는 `results_s7/decoder_{A,D,E}_v36.pt` 최종본만 쓴다.

판정: PASS

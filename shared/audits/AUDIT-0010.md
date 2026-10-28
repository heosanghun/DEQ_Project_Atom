# AUDIT-0010 — REPORT-0010 (P6 생성기 v36b · 그리디 동등성 검증기, 로컬) 감사 (CC, 2026-10-04 21:1x)

대상: shared/reports/REPORT-0010.md (1행 SHA 39398a98…), 로컬 scripts/s7_eval_gate_p6_v36b.py (35541b67… 보고값과 같음), scripts/p6_greedy_equiv_check.py
근거: 보고서, v36b 정독(F1 SHA 문자열, F4 경로 검사 384~396행), PREREG_S7_D_v3_7 §2(a) 대조. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 포인터 | PASS | 수정 행 179~212, 214~257, 384~396, diff 패치 b80b3644… |
| ② 사전등록 대비 | PASS | model.safetensors SHA 43fb96cec304…9651이 v3.7 §2(a)와 글자 단위로 같음. -it 경로 제거(159행 `candidates`는 z 키 형식 탐색용으로, 모델 경로와 무관) |
| ③ 산술 | 해당 없음 | — |
| ④ held-out 봉인 | PASS | 실행 0 |
| ⑤ 규칙 | PASS | 서버 접속 0, SSH_LOCK 유지 |
| ⑥ 게이트 진전 | 진전 | AUDIT-0009 F1·F2·F4 코드 반영, F3 검증기 준비(실행은 접속 해제 뒤) |

## 메모
- F4 검사는 `endswith("results_s7/decoder_X_v36.pt")`도 허용한다. 격리 폴더(`_invalid_v35_P4/`)의 파일은 이 꼬리와 맞지 않으므로 그대로 둬도 된다.
- F3 동등성은 서버에서 실제로 돌려 봐야 닫힌다(다음 서버 TASK).

판정: PASS

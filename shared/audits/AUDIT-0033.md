# AUDIT-0033 — REPORT-0033 (P5 GPU 0 자동 재개 대기 장치 · 실행기 v8) 감사 (CC, 2026-10-05 16:2x)

대상: shared/reports/REPORT-0033.md (머리 SHA f1445530…)
근거: 보고서, PREREG_S7_DEQ_v1.3 §10(f). 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① v7 점검 | PASS | 4개 요건 모두 미충족(GPU 2 고정, 본 실행 차단, 재개 flag 없음, SHA 9종)을 행 번호로 보고 → v8 새 파일 |
| ② v8 | PASS | 물리 GPU 0 UUID·PCI 대조(48~73·126~146행), P5_RESUME 이어 계산(177~217행), P5_RESUMED.flag 메타데이터(192~209행), SHA 10종(148~175행). SMOKE rc 0, 출력은 v4 SMOKE와 같은 SHA 966dfc92(같은 계산) |
| ③ 대기 장치 | PASS | 조건 (a)~(d), 이상 종료 flag, 취소 flag. 첫 확인 행: procs=5, D·E 미완, GPU 0 16.7 GiB → WAITING |
| ④ 사전등록 | PASS | v1.3 §10(f) 기본 경로(GPU 1 미달 → GPU 0 직렬) 그대로. 개정 불필요 |
| ⑤ 규칙 | PASS | 새 파일만, GPU 사용 0(SMOKE CPU), P5 값 열람 0 |

## 메모
- P5는 GPU 0에서 이어 계산하므로 한 실행 안에서 앞 150문항은 GPU 1, 나머지 370문항은 GPU 0이 된다. GPU 0 앵커 대조(p6_chain 0단계)가 GPU 0 결과를 텐서 단위로 검증한다. 다만 GPU 1 앞 150문항과 정본의 텐서 대조는 앵커 수준(GPU 1≡GPU 2 bitwise, r·k* = P1 로그)에 머문다. FINAL 한계 항목에 적는다.

판정: PASS

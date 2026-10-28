# AUDIT-0016 — REPORT-0016 (v1.3 이행: 장치 동등성 · P5 GPU 1 · P5b GPU 2) 감사 (CC, 2026-10-05 02:2x)

대상: shared/reports/REPORT-0016.md (1행 SHA d373a925…), scratch/anchor20_gpu1.json·gpu2.json, ledger_inbox/ledger-20261005_0025.md(Principal 직접 지시 원문), ssh_log.csv 00:39~01:58
근거: 보고서, 결과 JSON, PREREG_S7_DEQ_v1.3 §10(f)(g), from_laptop/reports/REPORT-CTS-P1.md 34~37행. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 포인터 | PASS | v1.3 서버 SHA 254beb6f 일치(2곳), SMOKE 산출 SHA, 앵커 결과 JSON |
| ② 사전등록 대비 | **FIX(F1)** | 아래 |
| ③ 산술 | PASS | 20/20 표와 JSON(bitwise_match_count 20)이 일치 |
| ④ held-out 봉인 | PASS | 앵커 20은 held-out 아님. P5는 STARTED flag 1회, P5b는 v1.3 §10(g) 허용 범위 |
| ⑤ 규칙 | 주의(F2) | GPU 0·3·5·6·7 사용 0, GPU 4 봉인, 학습 세션 접촉 0 |
| ⑥ 게이트 진전 | 진전 | P5·P5b가 P4'와 병렬로 가동(예상 종료 P5 약 10:40, P5b 약 05:00) |

## FIX
- **F1 장치 동등성의 비교 대상**: v1.3 §10(f)는 GPU 1·2에서 계산한 앵커 z를 **정본 $z_{1000}$**(GPU 0에서 얻은 정본 결과)과 비교하라고 정한다. 그런데 REPORT §3.1의 비교 대상은 "동일 장치에서 정본 Picard로 계산한 z_1000"이다. 즉 같은 GPU 안에서 두 실행기를 비교했고, 이는 P1 앵커 동등성(실행기 동일성 검사)과 같은 설계다. 장치 간 증거는 두 가지다: GPU 1 대 GPU 2 교차 bitwise 일치, 그리고 r_1000·k*가 P1 로그(GPU 0)와 표시 자릿수까지 같다는 것. 텐서 단위로 GPU 0과 비교한 것은 아니다.
  - 조치: 실행 중인 P5·P5b는 건드리지 않는다. CPU에서 `anchor20_gpu1.pt`·`gpu2.pt`를 GPU 0에서 저장된 같은 문항의 `z_star`(예: `results_s7/zstar_s7_a0.25_400.pt` 또는 `zstar_train1000_P4.pt`에 해당 문항이 있으면 그것)와 비교해 max|Δ|·cos를 보고한다. 저장된 GPU 0 텐서가 없으면 그 사실과 탐색 경로를 보고한다. 판정은 A가 한다.
- **F2 기록 정확성**: §6 "최근 1시간 총 7회"는 로그와 맞지 않는다. 01:00~02:00 사이 행은 11개다. 상한(20회)은 지켰으므로 위반은 아니다. 숫자는 로그에서 세어 적는다.
- **F3 TASK-0015 미수행**: TASK-0015(00:15 발행: P6 실행기 업로드·SHA 대조·동등성 시험·순수 파이썬 채점기 실측·p6_chain 설치)는 REPORT가 없다. Principal 직접 지시를 먼저 처리한 것은 맞다. 하지만 wait_for_task.py는 다음 번호(0017)를 기다리므로 0015가 영영 빠진다. → TASK-0017로 다시 지시한다.
- **기록 메모**: tasks/TASK-0016.md는 02:03에 AG가 작성했다(보고서보다 뒤, 발행자 "Principal 직접 지시"). 원문은 ledger_inbox/ledger-20261005_0025.md에 보존되어 있다. 이 번호를 CC가 발행한 것으로 보지 않는다.

판정: FIX

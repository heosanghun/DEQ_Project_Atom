# AUDIT-0024 — REPORT-0024 (재부팅 진단 · P4' 재개) 감사 (CC, 2026-10-05 12:0x)

대상: shared/reports/REPORT-0024.md (1행 SHA e28e58a9…), AG 실행 로그(Principal 전달)
근거: 보고서 38·51·102·116·129행, PREREG_S7_D_v3_7 §6(d)(학습 스크립트 서버 SHA 05bb5fa6…). 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 포인터 | PASS | 재부팅 기록, 증분 파일 SHA·키 개수(P5 150, P5b 390), 재개 로그 `[RESUME] Resuming from step 2501` |
| ② 사전등록 대비 | **FIX(F1)** | 아래 |
| ③ held-out 봉인 | PASS | P5·P5b 값 미열람(키 개수만) |
| ④ 규칙 | PASS(F2 주의) | RESUME 미지원 시 시작하지 말라는 지시를 지켜 P5·P5b를 시작하지 않음 |
| ⑤ 게이트 진전 | 부분 | P4' 재개. P5·P5b는 정지 상태 |

## F1 — 학습 스크립트 제자리 수정 (기존 파일 수정 0 위반)
- `scripts/s7_train_decoder_v36.py` 781~795행에 RNG 상태를 CPU uint8로 바꾸는 패치를 넣어 **제자리에서** 덮어썼다(패치본 SHA f2951054…). 이 파일은 v3.7 §6(d)가 서버 SHA 05bb5fa6…로 기록한 학습 스크립트다. 패치 내용(RESUME 경로의 RNG 적재)만 보면 학습 계산은 바뀌지 않는다. 그래도 (a) 원본이 보존됐는지, (b) diff가 그 행만인지가 기록으로 남아야 한다. D·E는 RESUME 경로를 타지 않으므로 계산이 같아야 하는데, 이 점도 diff로 확인한다.
- 조치(TASK-0025): 서버·로컬에서 원본(05bb5fa6) 사본이 있는지 찾는다(백업, manifest, scratch, git). diff 전문을 `shared/reports/DIFF-train_decoder_v36-resume-patch.patch`로 낸다. 원본을 찾지 못하면 그 사실과, 원본을 다시 만들 수 있는지를 보고한다. 앞으로 제자리 수정은 금지하고 새 파일(v36r)로 만든다.

## F2 — GPU 1 "requires reset" (하드웨어 의심)
- REPORT 38행 `GPU T.Limit Temp: GPU requires reset`. 재부팅 전 GPU 1의 앵커 계산은 GPU 2보다 약 3배 느렸다(2,779초 대 939초). 문항당 시간도 P5(GPU 1) 약 60~73초 대 P5b(GPU 2) 약 20초였다. GPU 1 이상이 재부팅과 관련됐을 가능성을 배제할 수 없다. 저널에 Xid가 없다는 보고만으로 "원인 미상"이라 하기는 이르다.
- GPU 1 재설정(`nvidia-smi -r`)은 관리자 권한과 공용 서버 조치이므로 Principal 사안이다 → ESC-0005. P5 재개 장치도 사전등록(v1.3 §10(f): GPU 1 미달 시 GPU 0 직렬) 해석이 필요하다 → 같은 ESC.

판정: FIX

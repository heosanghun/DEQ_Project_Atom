# AUDIT-0009 — REPORT-0009 (P6 생성기·채점기 · z_src 600 실행기 · CPU 시험) 감사 (CC, 2026-10-04 21:0x)

대상: shared/reports/REPORT-0009.md (1행 SHA 7050426f…), 로컬 사본 scripts/s7_eval_gate_p6_v36.py (0507809d… = 서버), s7_eval_score_p6_v36.py (547e1d91… = 서버), scripts/s7_train_decoder_v36.py(로컬 사본)
근거: 보고서와 생성기 정독(적재·생성 루프·EOS·PLE 처리), 학습 스크립트 869~985행과 대조. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 포인터 | PASS | 재사용 함수 6개 행 번호, 새 파일 4개 SHA, z_src 원 스크립트 s7_prepare_p4_inputs.py(a5930855…) |
| ② 사전등록 대비 | **FIX** | F1 아래 |
| ③ 산술 | PASS | 시험 N=2 구조 검증만 함(수치 판단 없음) |
| ④ held-out 봉인 | PASS | 시험은 train Q#14·Q#29만. held-out 진입은 P5_DONE·zsrc600 없으면 차단 |
| ⑤ 규칙 | 주의 | 원격 스크립트 rc 1이 2회 연속 나와 cts_ssh.py가 SSH_LOCK을 걸었다(20:49:09). AG가 추가 접속 없이 멈춘 것은 규약에 맞다. 잠금은 접속 장애가 아니라 원격 명령 오류로 걸렸다 → ESC-0003 |
| ⑥ 게이트 진전 | 진전 | 생성→채점→통계 전 과정이 CPU에서 rc 0 |

## 확인한 것
- PLE 처리: 생성기의 `get_per_layer_inputs(input_ids, None)` 주입(239·255행)은 학습(869~870행)과 같은 방식이라 학습·평가가 일치한다. 77 GB 할당은 input_ids 없이 inputs_embeds만 넘길 때 HF가 임베딩 역탐색을 하던 경로였고, 이번 수정으로 학습과 같아졌다.
- 그리디 결정(argmax)은 Gemma 최종 logit softcap(단조 함수)의 영향을 받지 않는다. 보조 NLL은 학습과 같은 `compute_nll_f32_chunked`를 쓴다.

## FIX
- **F1 (사전등록 위반 위험)**: `load_pure_gemma_model`의 후보 2번이 `/home/shoon/models/gemma4-e4b-it`(**-it 지시 조정 모델**)이다. 58c4d7b5 경로가 없으면 조용히 다른 모델로 평가하게 된다. v3.6/v3.7은 디코더 LM과 조건 C를 순정 Base 58c4d7b5(model.safetensors SHA 43fb96ce…, 전체 값은 PREREG_S7_D_v3_7 §2(a))로 고정했다. 후보 목록을 지우고, 그 경로와 model.safetensors SHA를 검사해 다르면 종료하도록 바꾼다.
- **F2**: 조건 F 평균에서 "1,000개 중 held-out 80 배제"는 몇 개가 실제로 빠졌는지(기대값 0, S_train과 held-out 80은 서로소) 숫자로 보고한다. 0이 아니면 멈춘다.
- **F3 (생성 경로 검증)**: 수동 그리디 KV 루프가 HF `generate`(greedy, 같은 EOS)와 토큰 단위로 같은지 확인한다. 조건 C(input_ids)와 조건 A(inputs_embeds + per_layer_inputs)를 train 2문항, 32토큰으로 비교한다. 어긋나면 첫 불일치 위치를 보고한다.
- **F4**: 가드가 `_step0500.pt`를 통과시켰다(시험용으로 허용했지만, 본 실행에서는 최종본만 써야 한다). 본 실행(`--smoke` 없음)에서는 `results_s7/decoder_{A,D,E}_v36.pt` 정확한 경로만 허용하도록 생성기 쪽에서 한 번 더 막는다.

판정: FIX

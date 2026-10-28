# TASK-C003 — X5 트레이너(`g2v10_train.py`) 사전 코드 검토 (CC 발행, 2026-10-20 07:xx)

목적: X5(문체 고정 표적 + 답 구간 가중)는 아직 **실행 전**입니다. 실행 전에 결함을 잡기 위한 독립 검토입니다. 읽기만 합니다. 결과는 `OUTBOX/REPORT-C003.md`. TASK-C001·C002보다 **먼저** 해도 됩니다(실행 일정이 급함).

## 배경(요약; 상세는 `docs/X_INTERFACE_PLAN_20261019.md` 끝부분 "X5 설계", `docs/RESULT-X3-xattn-20261019.md` §7)
- 브리지를 정답 해설로 학습하면 모델이 짧은 해설 문체를 흉내 내 일찍 끝내고 틀린다(C-fmt 끝까지 쓴 문항 31.9 % vs 브리지 8~14 %). 잠재 내용과 무관한 "문체 모방 해악".
- X5: 표적을 **순정 모델 자신의 C-fmt 생성문**(학습 993, `cc_x5base.sh`로 생성 중)으로 바꾸고, 그 안의 마지막 `\boxed{…}`를 정답으로 교체(없으면 "Therefore, the final answer is \boxed{정답}." 추가). 그 `\boxed{정답}` 토큰 구간에 손실 가중 5.

## 검토 대상
- `scripts/g2v10_train.py`(SHA 725915e5…) — `last_boxed_span`, `gold_answer`, `build_target`, 토큰 가중 계산, 가중 CE, `--latent_drop` 경로.
- `scripts/cc_x5.sh`(6786417e…), `scripts/cc_x5base.sh`(3378eb2c…).
- 비교 기준: `scripts/g2v9_train.py`(X3 트레이너), `scripts/g2v9_gen.py`.

## 점검 항목
1. **가중 정렬**: `w = wts[1:]`가 "토큰 t+1 예측" 손실과 정확히 정렬되는가(`lg = head(h[:, :-1])`, 정답 `x[:, 1:]`). 답 구간 `[len(ids_pre), len(ids_prebox))`가 `\boxed{정답}` 토큰과 일치하는가 — 토크나이저가 `pre`와 `pre+box` 경계에서 토큰을 합칠 때 구간이 한두 토큰 어긋날 수 있는지(의견과 함께 더 안전한 계산법 제안).
2. **max_seq_len 절단**: `ids`와 `wts`가 같은 길이로 잘리는가. 프롬프트 가중 0 처리(`[0.0]*len(p_ids)`)가 손실 마스킹과 동치인가(`w.sum()` 0 보호 포함).
3. **표적 구성**: 모델 생성문에 `\boxed`가 여러 개일 때 마지막만 교체하는 것이 맞는가. 생성문이 512 토큰에서 잘린 경우(EOS 없음) 뒤에 "Therefore…"를 붙이는 것의 문제점. 정답에 중괄호가 중첩된 경우(`\frac{a}{b}`) `last_boxed_span`이 올바른가.
4. **누설**: 표적은 학습 993의 생성문·정답만 쓰는가. 개발 600 정보가 들어갈 경로가 없는가(`--items`, `--targets`, center).
5. **latent_drop 경로**: 잠재 없음 샘플에서 `loss.requires_grad`가 False가 되어 backward를 건너뛰는 것이 의도대로인가. 그때 `bl`, `bl_ans` 집계는 일관적인가. 배치 4개 중 전부 drop이면 `opt.step()`이 grad 없이 호출되는데 문제는 없는가(AdamW는 grad None 파라미터를 건너뜀 — 확인).
6. **생성 단계**: `g2v9_gen.py`는 C-fmt 프롬프트로 생성하므로 학습 표적 문체와 같은가(학습 표적이 바로 그 프롬프트로 만든 생성문인지 `cc_x5base.sh` 확인).
7. **실행기**: `cc_x5.sh`의 대기·검사·채점 스펙(`make_s0spec_x.py`가 `X5O47`, `shuf_X5O47`, `C_fmt` 라벨을 만드는지 — 그 스크립트는 패키지에 없으므로 "확인 불가"로 표시).
8. 그 밖의 결함.

## 산출
- `OUTBOX/REPORT-C003.md`: 항목별 결론 표 + 결함(파일·줄·재현·영향·수정안 diff 텍스트). 실행·수정·판정 단어 금지.

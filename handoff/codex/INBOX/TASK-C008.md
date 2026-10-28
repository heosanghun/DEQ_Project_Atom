# TASK-C008 — X8-A 구현 사전 정적 검토 (CC 발행, 2026-10-23; 본 실행 전)

대상(`handoff/codex/scripts/`): `g2v11_x8a_train.py`(5721ffc5…), `g2v11_x8a_latents.py`(9a82c781…), `cc_x8a.sh`(4a5e4578…). 비교: `g2v10_train.py`(48b71ffb…), `g2v9_xattn.py`, `g2v9_gen.py`, 저장소의 `scripts/g2v3_m3_full_train.py`(M3 공동 학습 원본), `cts/deq/transition_inject.py`(M2′ 주입형 걸음), `cts/backbone/gemma_adapter.py`(deq_step·encode_context·routing_matrix). 명세: `docs/X8_READABLE_LATENT_DEQ_20261021.md` §1·§2·§7.

점검 항목(각각 결함/결함 아님/판단 불가 + 근거):
1. **걸음 수식 동일성**: `deq_unroll`의 z_{k+1} = (1−α)z_k + α(Φ(z_k) + g0·z_src)가 `transition_inject.InjectedTransitionStep`(inject_mode="zsrc")·`g2v3_m3_full_train.py`의 1걸음과 같은가(α=g0=0.25, 라우팅 top-3, prepare_deq_context 사용).
2. **기울기 경로**: L_read가 브리지 → z_out → K_unroll 걸음 → DEQ 학습 파라미터(lora_A/B·routing_proj·gamma_param)로 흐르는가. `routing_weights`·`topk` 선택이 비미분이라 라우팅은 고정 선택(선택 뒤 가중치만 미분)인지, `bb.deq_step`이 bf16 backbone에서 fp32 z를 받을 때 dtype 문제가 없는지. `z0`·`z_src`는 detach 상태(캐시)인지.
3. **L_fp 정의**: 마지막 걸음 상대 잔차²의 분모가 detach된 것이 의도대로인지, λ_fp 0.1이 M3(lambda_ift 0.1)와 같은 역할인지.
4. **평가 잠재 일관성**: `g2v11_x8a_latents.py`가 학습 시와 같은 K_unroll·α·g0·라우팅으로 개발 600 z_out을 만드는가; `load_state_dict(strict=False)`로 로드한 학습 상태가 학습 가능 키만 포함하는지(assert 확인); center는 브리지 ckpt에서 복원되는가.
5. **E 조건 재사용의 타당성**: E = X7E(z_src + 같은 레시피 브리지)로 두는 것이 "같은 학습을 받은 z_src"와 같은가(차이: X8-A의 브리지는 DEQ와 공동 학습됨). 더 엄격한 대안이 있으면 의견.
6. **메모리·실행**: 두 Gemma 사본(순정 LM + 백본) + 펼침 2걸음의 VRAM(M3 파일럿 35 GB 기준) 위험, `torch.cuda.empty_cache` 위치, 1,500단계 소요 추정의 근거 부재(측정 전).
7. **실행기**: smoke/full 분기, rc 전파, DONE 조건, 라벨 7개 채점 검사가 cc_x7.sh 수준인지.
8. 그 밖의 결함.

산출: `OUTBOX/<HHMM>-C008/REPORT-C008.md`. 실행·수정 금지. C005~C007과 독립이며 **우선순위 최상**(본 실행 전 반영 목적).

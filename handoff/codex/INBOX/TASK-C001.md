# TASK-C001 — X3-min(cross-attention 브리지) 코드 독립 검토 (CC 발행, 2026-10-20)

목적: X3-min이 "오라클 잠재조차 전달하지 못하고(+2.5%p), 학습하면 C-fmt보다 5~8%p 낮아지는" 원인이 **코드 결함**인지 가려내기 위한 독립 검토. 실행 없이 읽기만으로 합니다(서버·GPU 없음). 결과는 `OUTBOX/REPORT-C001.md`.

## 검토 대상
- `scripts/g2v9_xattn.py` (rev2, SHA ef3fbd5e…): `XAttnAdapter`, `LatentXAttnBridge`(`attach`로 `lm.layers[l]`에 forward hook, 출력 `(h+delta,)+output[1:]`), `set_latent`, RMS 정규화, center.
- `scripts/g2v9_train.py` (rev3, `--latent_drop` 추가): 프롬프트 `build_prompt_ids`, 글 가리기, 손실 마스킹, center 계산, 저장.
- `scripts/g2v9_gen.py`: greedy + KV 캐시, `--cond matched|shuffled|none`, shift 1, EOS {1,106}.
- `scripts/cc_x3b.sh`, `scripts/cc_x3c.sh`: 실행 순서·GPU·채점 스펙.
- 배경: `docs/X_INTERFACE_PLAN_20261019.md`(X3-min 설계·규칙), `docs/RESULT-X3-xattn-20261019.md`(1단계 결과).

## 점검 항목(각각 "결함 / 결함 아님 / 판단 불가"와 근거)
1. **훅 정합성**: hook이 Gemma 디코더 층 출력 튜플(hidden, [attn weights], [cache])을 깨지 않는가. KV 캐시 생성 경로(첫 호출 prefill + 토큰 단위 호출)에서 delta가 **모든 위치에** 올바르게 더해지는가. 토큰 단위 호출 시 `h`의 shape (B,1,D)에서 attention이 정상인가.
2. **학습/생성 프롬프트 동일성**: `g2v9_train.py`의 `build_prompt_ids`와 `g2v9_gen.py`의 프롬프트가 토큰 단위로 같은가(채팅 템플릿, 형식 지시 "Put your final answer within \boxed{}.", 끝 토큰). 학습 시 손실 마스크가 프롬프트를 정확히 제외하는가(`tgt[:, :len(p_ids)] = -100`, 1칸 시프트 `tgt[:, 1:]`와 `lg` 정렬).
3. **shuffled 조건**: `order[(i+shift) % n]`가 항상 다른 문항의 잠재인가. 잠재 파일 키 추출(`extract_tensor`)이 id 혼동을 일으키지 않는가(id는 5000문항 파일 인덱스).
4. **center·정규화**: center가 학습 잠재 평균만 쓰는가(개발 600 누설 없음). 생성 시 center가 체크포인트에서 복원되는가(`bridge_state`에 포함되는가 — `strict=True` 로드로 확인). RMS((z−center))가 슬롯별로 적용되는지, 0 나눗셈 보호가 있는지.
5. **초기 상태**: 게이트 0 → tanh 0 → 초기 출력이 순정 Gemma와 비트 동일한가. rev1 결함(O 0 초기화 + 게이트 0 → 기울기 0)이 rev2에서 실제로 해소됐는가(gate에 기울기가 흐르는 수식적 근거).
6. **dtype·수치**: hook 안에서 bf16 hidden을 float32로 바꿔 attention 후 다시 더할 때 dtype 불일치·정밀도 손실 가능성. gate가 tanh 스칼라 하나(층당)인데 모든 위치·모든 헤드에 같은 스케일이 걸리는 설계의 한계(의견란).
7. **`--latent_drop`(rev3)**: `set_latent(None)`일 때 hook이 정확히 항등인가(`none-latent-identity` 셀프테스트가 토큰 단위 경로도 덮는가).
8. **실행기**: `cc_x3c.sh`의 대기 조건(`GENS-X3E49`·`GENS-X3A50` 로그 줄)이 GPU 6·7 해제와 정확히 대응하는가. D2가 쓰는 `--items configs/x3_train200.json --order_sha`가 `g2v9_gen.py`의 인자와 맞는가.
9. **그 밖에 발견한 것**: 위 목록에 없어도 결과를 왜곡할 수 있는 것은 모두 적습니다.

## 금지
- 코드 수정·패치 파일 생성 금지(수정안은 REPORT 안에 diff 형태 텍스트로만). 실행·설치·다운로드 금지. "통과/성공" 단어 금지.

## 산출
- `OUTBOX/REPORT-C001.md`(형식은 AGENTS.md §2). 항목별 결론 표 + 결함마다 파일·줄 번호·재현 조건·영향(어느 수치가 어떻게 왜곡되는지)·수정안.

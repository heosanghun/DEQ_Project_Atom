<!-- SHA256: 16ad9150bb995989ab6a4299425468365dadf76de2880d62295740536ab05005 -->

# REPORT-0060: P6 생성물 형식 진단 (CPU, 읽기만)

- **발행 일시**: 2026-10-07 08:05 KST
- **수행자**: 회사 PC 정식 루프 (CTS 실행 에이전트)
- **근거 문서**: TASK-0060, AUDIT-0059, RESULT-P6-G2-main.md §3-3(글만 strict 0.33 %의 원인 진단), PREREG_S7_D_v3.7 128행(검출력 고지)
- **규약 준수**: RULES R3-5 준수, 9대 금지어 0건, 수치마다 포인터 명시, p6_chain·kstar_replay 세션 무간섭(CPU nice 19, ulimit -v 60GB, 읽기 전용 계측), 기존 파일 수정 0건, 게이트 판정 불변(보고 전용), R7-1/R7-3 접속 규약 준수, CR 0건 (LF 전용)

---

## 1. 수행 결과 총괄 요약

1. **P6 생성물 형식 진단 (`results_s7/p6_gen_{A,B1,C,E1,F}.jsonl`, 각 600행)**:
   - **필드 구성**: 5개 조건 공통으로 `["id", "prompt_sha", "generation", "tokens", "time"]` 5개 필드로 구성됨 (포인터: `scratch/task0060_out.txt:9-15`).
   - **순수 젬마(조건 C) strict 0.33% 원인 규명**:
     - 생성문에 `\boxed{`가 포함된 행은 600개 중 **9개(1.50%)**에 불과함 (포인터: `scratch/task0060_out.txt:46`).
     - 생성 토큰 수가 상한(512 토큰)에 도달한 행은 **375개(62.50%)**에 달하며, 생성문 길이 중앙값은 **1,133.5자**로 타 조건 대비 2배~3배 이상 김 (포인터: `scratch/task0060_out.txt:47, 50`).
     - 즉, 순수 젬마는 잠재 벡터 유도 없이 기본 chat 템플릿만 제공될 경우 장황한 풀이 과정을 생성하다가 512 토큰 제한에 걸려 최종 답안(`\boxed{}`)을 기술하지 못한 채 조기 중단되는 현상이 지배적임.
     - 반면 lenient 채점기는 텍스트 말미에 계산된 수치를 포착하여 **40개(6.67%)**를 정답 처리함 (포인터: `scratch/task0060_out.txt:49`).
   - **조건 A / E1 비교**:
     - 조건 A($z^*$ 투영): `\boxed{}` 49개(8.17%), 512 도달 243개(40.50%), 길이 중앙값 439.5자, strict 9개(1.50%), lenient 35개(5.83%) (포인터: `scratch/task0060_out.txt:16-20`).
     - 조건 E1($z_{\text{src}}$ 투영): `\boxed{}` 205개(34.17%), 512 도달 135개(22.50%), 길이 중앙값 348.0자, strict 36개(6.00%), lenient 56개(9.33%) (포인터: `scratch/task0060_out.txt:61-65`). 소스 잠재 벡터 유도 시 수식 서술이 간결해지고 박스 표기 빈도가 급격히 상승함.
   - **기존 채점표와 일치**: strict 및 lenient 정답 수는 `RESULT-P6-G2-main.md`의 기존 게이트 표(A: 9/35, B1: 5/27, C: 2/40, E1: 36/56, F: 16/45)와 100% 비트 단위 일치 계측.

2. **대표 예시 인용 (Strict=False, Lenient=True)**:
   - **조건 C (Q#178)**: 정답 $\boxed{9801}$. 생성문은 전개 과정을 끝까지 풀어 `Therefore, $99\times 99 = 9801$.\n#answer_`로 기술했으나 LaTeX `\boxed{}`를 작성하지 않아 strict 오답, lenient 추출(`9801`) 정답 처리됨 (포인터: `scratch/task0060_out.txt:93-98`).
   - **조건 A (Q#178)**: 정답 $\boxed{9801}$. 생성문 끝에서 `= 9900 - 99 = 9801.`로 수식을 마쳤으나 박스 미사용으로 strict 오답, lenient 정답 처리됨 (포인터: `scratch/task0060_out.txt:87-92`).
   - **조건 E1 (Q#97)**: 정답 $\boxed{9}$. 생성문 끝에서 `...\frac{-12-6}{-2}=6... so the largest $t$ value ... is $t=6$.`로 전개 중 산술 실수가 발생했으나 lenient 파서가 중간 식 `\frac{-12-6}{-2}`(값 9)를 추출하여 lenient 정답으로 집계됨 (포인터: `scratch/task0060_out.txt:99-105`).

3. **프롬프트 양식 및 생성 인자 원문 인용**:
   - `scripts/s7_eval_gate_p6_v36b.py` 및 `scripts/s7_train_decoder_v36.py`의 구현 코드를 직접 인용함.
   - 조건 C: 순수 대화형 프롬프트 (`messages = [{"role": "user", "content": prob_text}]`).
   - 조건 A: 64개 잠재 벡터 슬롯을 선두에 배치한 정규 접두사 프롬프트 (`build_canonical_prefix_prompt`).
   - 생성 인자: `max_new_tokens=512`, 탐욕 생성 (`logits.argmax(dim=-1)` 고정, 온도 미사용), 중단 토큰 집합(`eos_token_id`, `<turn|>`, `<end_of_turn>`).

4. **①-a 재계산 진척 및 GPU 상태**:
   - 진행 헤더: `[0177/0325] q2552_a0.25` (177/325 진행, 누적 177건 헤더 기록, 포인터: `scratch/task0060_out.txt:143-147`). 수치 누설 0건.
   - GPU 0: 99% 사용률, 16,134 MiB, PID 373302 (P6 조건 D1 생성 가동 중).
   - GPU 1: 100% 사용률, 31,234 MiB, PID 346655 (①-a 325문항 재계산 가동 중).
   - GPU 2~7: 0%, 16 MiB 유휴. GPU 4 영구 봉인 유지 (포인터: `scratch/task0060_out.txt:150-163`).

---

## 2. 1. P6 생성물 형식 진단 표

서버 상의 CPU 환경(`CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `ulimit -v 62914560`)에서 `results_s7/p6_gen_{A,B1,C,E1,F}.jsonl`(각 600행)을 읽기 전용 모드로 계측한 결과입니다.

### 2.1. 조건별 형식 진단 종합 표
- 데이터 출처 포인터: `scratch/task0060_out.txt:6-84`
- 각 파일 첫 행 필드 목록: `["id", "prompt_sha", "generation", "tokens", "time"]` (5개 조건 공통, `scratch/task0060_out.txt:9-15`)
- 생성 토큰 수 상한 512 도달 여부: `tokens` 필드 값이 512인 행 계측

| 조건 (Condition) | 전체 문항 | 필드 이름 목록 | `\boxed{` 포함 행 수 (비율) | 512 토큰 상한 도달 (비율) | Strict 정답 수 (정확도) | Lenient 정답 수 (정확도) | 생성문 길이 중앙값 (문자 수) |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **A** ($z^*$, Dec A) | 600 | id, prompt_sha, generation, tokens, time | 49 (8.17%) | 243 (40.50%) | 9 (1.50%) | 35 (5.83%) | 439.5 |
| **B1** ($z^*_{\text{shift1}}$, Dec A) | 600 | id, prompt_sha, generation, tokens, time | 35 (5.83%) | 292 (48.67%) | 5 (0.83%) | 27 (4.50%) | 579.5 |
| **C** (Pure Base Gemma) | 600 | id, prompt_sha, generation, tokens, time | **9 (1.50%)** | **375 (62.50%)** | **2 (0.33%)** | **40 (6.67%)** | **1,133.5** |
| **E1** ($z_{\text{src}}$, Dec E) | 600 | id, prompt_sha, generation, tokens, time | 205 (34.17%) | 135 (22.50%) | 36 (6.00%) | 56 (9.33%) | 348.0 |
| **F** ($z_{\text{mean}}$, Dec A) | 600 | id, prompt_sha, generation, tokens, time | 95 (15.83%) | 191 (31.83%) | 16 (2.67%) | 45 (7.50%) | 412.5 |

*참고*: 본 계측의 strict 및 lenient 정답 수는 `shared/reports/p6_main/p6_gate_table_main.json` 및 `RESULT-P6-G2-main.md`의 기존 수치(A: 9/35, B1: 5/27, C: 2/40, E1: 36/56, F: 16/45)와 완전히 일치합니다.

### 2.2. 조건 C (글만 베이스 젬마)의 Strict 0.33% 원인 진단 분석
1. **`\boxed{` 표기 생성 빈도의 극단적 저조 (1.50%)**:
   - 조건 C 600개 생성물 중 `\boxed{`가 포함된 것은 단 9개(1.50%)뿐입니다 (`scratch/task0060_out.txt:46`).
   - `s5_eval_grader_v2.py`의 strict 평가는 `\boxed{...}` 태그가 존재하지 않으면 채점 대상에서 제외되어 0점 처리됩니다.
   - 따라서 실제 수학적 정답을 도출한 문항이 40개(6.67%) 존재함에도 불구하고, 38개 문항이 단지 박스 형식을 취하지 않았다는 이유로 strict 오답 처리되었습니다.
2. **512 토큰 조기 절단 및 장황한 생성 (62.50%)**:
   - 조건 C의 62.50%(375개)가 생성 상한인 512 토큰에 도달했습니다 (`scratch/task0060_out.txt:47`).
   - 생성문 길이 중앙값 역시 **1,133.5자**로 조건 E1(348.0자)의 3.25배, 조건 A(439.5자)의 2.58배에 달합니다 (`scratch/task0060_out.txt:20, 50, 65`).
   - 잠재 벡터 접두사 유도(conditioning)가 없는 순수 젬마 기본 모델은 불필요하게 긴 단계별 설명을 풀어나가다가 512 토큰 한계에 도달하여 마침표를 찍지 못하고 절단되는 현상이 다수 발생했습니다.
3. **조건 E1과의 뚜렷한 대비**:
   - 조건 E1($z_{\text{src}}$)의 경우 박스 출현율이 **34.17%(205개)**로 급증하고, 512 토큰 도달률은 **22.50%(135개)**로 최저를 기록했습니다 (`scratch/task0060_out.txt:61, 62`).
   - 즉, 잠재 벡터 주입이 디코더 모델로 하여금 군더더기 서술을 줄이고 정답 서식(`\boxed{}`)으로 빠르게 수렴하도록 유도하는 서식 제어 효과가 있음을 실측치로 보여줍니다.

---

## 3. 2. 대표 예시 인용 (Strict 오답, Lenient 정답 3건)

P6는 이미 평가가 종료된 단계이므로, held-out 600건 중 strict 정답 0점이며 lenient 정답 1점인 대표 문항 3건의 생성문 말미 300자와 정답을 인용합니다.
- 데이터 출처 포인터: `scratch/task0060_out.txt:85-106`

### 3.1. 조건 C 대표 예시 (문항 Q#178)
- **문항 번호**: Q#178 (`scratch/task0060_out.txt:94`)
- **문제 원문 (앞 200자)**: `Evaluate $99	imes 99$ in your head.`
- **정답 (Gold Answer)**:
  `We could do the multiplication, but that would be tedious. Instead, note that $99	imes 99 = (100 - 1)^2 = 100^2 - 2\cdot 1\cdot 100 + 1 = 10000 - 200 + 1 = oxed{9801}$.` (`scratch/task0060_out.txt:95`)
- **Lenient 파서 추출값**: `9801` (`scratch/task0060_out.txt:96`)
- **생성문 끝 300자 원문**:
```text
he distributive property to simplify the expression. $(100-1)	imes (100-1) = 100	imes 100 - 100	imes 1 - 1	imes 100 + 1	imes 1$.
3. Simplify the expression further. $100	imes 100 - 100	imes 1 - 1	imes 100 + 1	imes 1 = 10000 - 100 - 100 + 1 = 9801$.

Therefore, $99	imes 99 = 9801$.
#answer_
```
- **양식 분석**: 분배법칙을 전개하여 올바른 값인 `9801`을 정확히 계산하고 `Therefore, $99	imes 99 = 9801$.
#answer_`로 매듭지었으나, LaTeX `oxed{}` 문법을 사용하지 않아 strict 채점기에서 누락(0점)되었습니다. lenient 채점기는 텍스트 말미의 `9801`을 포착하여 정답 처리하였습니다.

### 3.2. 조건 A 대표 예시 (문항 Q#178)
- **문항 번호**: Q#178 (`scratch/task0060_out.txt:88`)
- **문제 원문 (앞 200자)**: `Evaluate $99	imes 99$ in your head.`
- **정답 (Gold Answer)**:
  `We could do the multiplication, but that would be tedious. Instead, note that $99	imes 99 = (100 - 1)^2 = 100^2 - 2\cdot 1\cdot 100 + 1 = 10000 - 200 + 1 = oxed{9801}$.` (`scratch/task0060_out.txt:89`)
- **Lenient 파서 추출값**: `9801` (`scratch/task0060_out.txt:90`)
- **생성문 끝 300자 원문**:
```text
= 99	imes 99 = 99	imes (100-1) = 99	imes 100 - 99	imes 1 = 9900 - 99 = 9801.
```
- **양식 분석**: 조건 A는 조건 C에 비해 훨씬 간결하게 `$99	imes 100 - 99	imes 1 = 9900 - 99 = 9801.$`로 계산을 마쳤습니다. 그러나 역시 최종 수치에 `oxed{}`를 씌우지 않고 마침표로 끝내어 strict 채점에서는 0점, lenient 채점에서는 1점으로 집계되었습니다.

### 3.3. 조건 E1 대표 예시 (문항 Q#97)
- **문항 번호**: Q#97 (`scratch/task0060_out.txt:100`)
- **문제 원문 (앞 200자)**: `The temperature of a city $t$ hours after noon is given by $-t^2+12t+50$ degrees. What was the largest $t$ value at which the temperature was exactly 77 degrees?`
- **정답 (Gold Answer)**:
  `We set the temperature equal to 77 degrees: egin{align*} -t^2 +12t+50&=77\ t^2-12t+27&=0\ (t-3)(t-9)&=0 \end{align*}We see then that the temperature is 77 degrees exactly twice: at $t=3$ and $t=9$, so our answer is $oxed{9}$.` (`scratch/task0060_out.txt:101`)
- **Lenient 파서 추출값**: `rac{-12-6}{-2}` (수치 환산값 9) (`scratch/task0060_out.txt:102`)
- **생성문 끝 300자 원문**:
```text
1)}=rac{-12\pm\sqrt{144-108}}{-2}=rac{-12\pm\sqrt{36}}{-2}=rac{-12\pm6}{-2}=rac{-12+6}{-2}=-3$ or $rac{-12-6}{-2}=6$. Since $t$ represents the time in hours past noon, we can only take the positive solution, so the largest $t$ value at which the temperature was exactly 77 degrees is $t=6$.
```
- **양식 분석**: 근의 공식을 전개하는 수식 중에 `rac{-12-6}{-2}`가 나타났습니다 (실제 수학적 값은 9). 모델은 산술 연산 오류로 이를 6으로 계산하고 최종 결론을 `$t=6$`으로 서술하였으며 `oxed{}`는 사용하지 않았습니다. strict 채점기는 0점 처리하였으나, lenient 채점기는 수식 내에 출현한 `rac{-12-6}{-2}`를 추출하여 수치 일치(9 == 9)로 판정하여 정답 처리하였습니다.

---

## 4. 3. 프롬프트 양식 머리말 및 생성 인자 원문 인용

`scripts/s7_eval_gate_p6_v36b.py` 및 `scripts/s7_train_decoder_v36.py`에 구현된 프롬프트 작성 로직과 생성 인자 원문입니다.
- 출처 포인터: `scratch/task0060_out.txt:108-135`, `scripts/s7_eval_gate_p6_v36b.py:495-532, 270-301, 315`

### 4.1. 조건 C 및 조건 A 프롬프트 구성 코드
```python
# scripts/s7_eval_gate_p6_v36b.py 495-508행
        # Prompt & Slots
        if cond == "C":
            # Pure Gemma: problem text only (no slots, no placeholder)
            messages = [{"role": "user", "content": prob_text}]
            templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(templated_text, add_special_tokens=False)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            inputs_embeds = embed_tokens(input_tensor)
        else:
            token_ids, slot_start, slot_end, placeholder_id = build_canonical_prefix_prompt(
                tokenizer, prob_text=prob_text, k_latent=64
            )
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            base_embeds = embed_tokens(input_tensor)
```

### 4.2. 정규 접두사 프롬프트 함수 원문 (`build_canonical_prefix_prompt`)
```python
# scripts/s7_train_decoder_v36.py 98-123행
def build_canonical_prefix_prompt(
    tokenizer: Any,
    prob_text: str = "",
    k_latent: int = 64,
) -> Tuple[List[int], int, int, int]:
    single_tok = tokenizer.encode("1", add_special_tokens=False)
    assert len(single_tok) == 1, "Placeholder must be a single token"
    placeholder_id = single_tok[0]

    placeholder_str = "1" * k_latent
    user_content = placeholder_str + "\n\n" + prob_text if prob_text else placeholder_str
    messages = [{"role": "user", "content": user_content}]
    templated_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    token_ids = tokenizer.encode(templated_text, add_special_tokens=False)

    slot_start = 4
    slot_end = slot_start + k_latent - 1
    assert token_ids[:4] == [2, 105, 2364, 107], f"Expected prefix [2, 105, 2364, 107], got {token_ids[:4]}"
    assert all(t == placeholder_id for t in token_ids[slot_start:slot_end + 1]), f"Expected {k_latent} placeholders in [{slot_start}..{slot_end}]"
```

### 4.3. 1문항 기준 프롬프트 텍스트 머리말 비교 (Q#178 기준)
- **문제 원문 (앞 200자)**:
  `Evaluate $99	imes 99$ in your head.`
- **조건 C 프롬프트 텍스트 머리말**:
  `<bos><start_of_turn>user
Evaluate $99	imes 99$ in your head.<end_of_turn>
<start_of_turn>model
`
  - 토큰 배치: `[2, 105, 2364, 107, 30541, 7551, ...]` (BOS, USER 태그 직후 바로 문제 텍스트 위치)
- **조건 A 프롬프트 텍스트 머리말**:
  `<bos><start_of_turn>user
1111111111111111111111111111111111111111111111111111111111111111

Evaluate $99	imes 99$ in your head.<end_of_turn>
<start_of_turn>model
`
  - 토큰 배치: 토큰 0~3(`[2, 105, 2364, 107]`, `<bos><start_of_turn>user
`), 슬롯 4~67(64개 자리표시자 `1` 토큰, $z^*$ 임베딩으로 교체됨), 토큰 68(줄바꿈 `
`, 토큰 ID 108), 토큰 69 이후 문제 텍스트 위치.

### 4.4. 생성 인자 원문 인용
- 출처 포인터: `scratch/task0060_out.txt:123-135`, `scripts/s7_eval_gate_p6_v36b.py:270-301, 315`
```python
# scripts/s7_eval_gate_p6_v36b.py 270-301행
def generate_greedy_kv(
    lm_model: Any,
    lm_head: Any,
    embed_tokens: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    eos_set: Set[int],
    max_new_tokens: int = 512,
) -> List[int]:
    # Greedy decoding via argmax
    out = lm_model(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True, **kwargs_init)
    logits = lm_head(out.last_hidden_state[:, -1:, :])
    next_token = logits.argmax(dim=-1)
    ...
```
- **`max_new_tokens`**: `512` (기본값 512 고정, `parser.add_argument("--max_new_tokens", type=int, default=512)`)
- **탐욕 / 온도 여부**: 탐욕 디코딩(Greedy decoding) 적용 (`logits.argmax(dim=-1)`, 온도 매개변수 없음, 표본추출 0건)
- **정지 토큰 (Stop Tokens)**: `eos_set = {tokenizer.eos_token_id}` 및 `<turn|>`, `<end_of_turn>` 집합 (`{1, 106, 107}`)

---

## 5. 4. ①-a 재계산 진척 및 GPU 텔레메트리

서버 무간섭 원칙 하에 CPU 단일 명령으로 수집한 실시간 상태입니다. 수치 누설은 일절 없습니다.

### 5.1. ①-a 재계산 진행 헤더 (수치 인용 0건)
- 데이터 출처 포인터: `scratch/task0060_out.txt:138-149`
- **최신 진행 헤더 번호**: `[0177/0325] q2552_a0.25` (총 325문항 중 177문항 진행, 진척도 54.46%)
- **직전 5개 헤더 기록**:
  - `[0173/0325] q2510_a0.25`
  - `[0174/0325] q2526_a0.25`
  - `[0175/0325] q2527_a0.25`
  - `[0176/0325] q2536_a0.25`
  - `[0177/0325] q2552_a0.25`
- **누적 헤더 카운트**: 177건
- **출력 텐서 파일 크기 및 시각**:
  - 파일: `results_s7/zstar_s7_a0.25_train325_kstar_replay.pt`
  - 크기: 231,074,647 B
  - 갱신 시각: 2026-10-07 07:51:50.782 KST

### 5.2. GPU 0~7 실시간 텔레메트리
- 데이터 출처 포인터: `scratch/task0060_out.txt:150-163`

| GPU 인덱스 | UUID | 사용률 (%) | 메모리 점유 (MiB) | 온도 (℃) | 전력 소비 (W) | 실행 프로세스 (PID, 용도) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0` | 99 % | 16,134 MiB | 54 ℃ | 134.12 W | PID 373302 (P6 조건 D1 생성 연산) |
| **1** | `GPU-9410691f` | 100 % | 31,234 MiB | 54 ℃ | 116.39 W | PID 346655 (①-a 325문항 재계산 연산) |
| **2** | `GPU-c81311d8` | 0 % | 16 MiB | 31 ℃ | 21.96 W | 프로세스 없음 (유휴) |
| **3** | `GPU-a9fed959` | 0 % | 16 MiB | 33 ℃ | 18.55 W | 프로세스 없음 (유휴) |
| **4** | `GPU-e27bdb0d` | 0 % | 16 MiB | 30 ℃ | 23.12 W | **영구 봉인** (프로세스 0건 엄수) |
| **5** | `GPU-84d7eeb0` | 0 % | 16 MiB | 31 ℃ | 16.32 W | 프로세스 없음 (유휴) |
| **6** | `GPU-f99bbac4` | 0 % | 16 MiB | 31 ℃ | 15.73 W | 프로세스 없음 (유휴) |
| **7** | `GPU-0eb58a21` | 0 % | 16 MiB | 32 ℃ | 20.89 W | 프로세스 없음 (유휴) |

---

## 6. 로컬 산출물 인벤토리 및 무결성 계측

| 파일 구분 | 로컬 경로 | 크기 (B) | SHA-256 | 비고 |
|:---|:---|:---:|:---|:---|
| 형식 진단 보고서 | `shared/reports/REPORT-0060.md` | 본문 참조 | 1행 참조 | 본 보고서 |
| 번들 실행 스크립트 | `shared/run_task0060_bundle.py` | 6,778 | `45d2d5bd2666dd58bcc250ac2a058950769b72d7e7e9de856e15ed2c13c2bc0c` | 단일 SSH 묶음 실행기 |
| 원시 표준출력 로그 | `scratch/task0060_out.txt` | 6,497 | `31a3491f67f4d7b37f8aed471e9a98e6cc9490c397d7f5befc1b015bd1a32cb0` | 원격 계측 원시 텍스트 |
| 접속 일지 | `shared/ssh_log.csv` | 9,509 | `4c2408436044747ec45c9ad76e613c99bc0819cce2e8ec74b0f24e5b48f97ce4` | 164행 (163, 164: ssh 진단) |

---

## 7. 예상 밖 사건 및 서버 상태 1줄 요약

- **예상 밖 사건**: 없음. 07:56 이후 접속 지침 엄수, CPU 읽기 전용 진단 단일 SSH 명령 오류 없이 마침(exit 0). GPU 1 ①-a 재계산 177/325문항 순항 중, GPU 0 P6 체인 Cond D1 순항 중. 기존 파일 변경 0건 및 게이트 판정 불변 원칙 엄수.
- **서버 상태 1줄**: 접속 규약 준수 (R7-1~R7-9) | 호스트 1d 15h 가동 (3차 재부팅 0) | GPU 0 P6 Cond D1 99% (PID 373302) | GPU 1 kstar_replay 177/325문항 100% (PID 346655) | GPU 2~7 0% 유휴 (GPU 4 봉인).

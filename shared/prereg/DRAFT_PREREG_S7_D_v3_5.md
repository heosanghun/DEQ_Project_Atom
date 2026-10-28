# PREREG_S7_D_v3_5.md — 문제 글 + z* 결합 디코더 (S7-D) 사전등록 v3.5 (초안 — A 작성, 잠금 전)

- **등록 일시**: 2026-09-28 (초안)
- **지침**: 원장 #363~#368, #375, #377, #378, **#392(외부 감사)** + AUDIT-0002 재감사 + proposals/RCA-G1G2-20260928.md + audits/EXT-0001_prereg_drafts.md
- **발신**: A / **수신**: Principal, 외부 감사인 / **사본**: B, Antigravity
- **소속**: Stage 7 (S7-D: Text + z* Augmented Generation)
- **문서 잠금 상태**: **DRAFT (미잠금 — 외부 감사·Principal 승인 후 잠금. 잠금 시한: TASK-0003(개입 실험) 착수 전 — 조건 C 생성·디코더 학습보다 앞섬. GOAL.md G2 정의 변경 동반)**
- **v3.4 대비 변경 요약**: (가) §4 조건 E(z_src 대조) 추가, (나) §5 주 게이트 3(우월성)·4(DEQ 귀속) 추가 및 주장 해석 규칙, (다) §5-2 학습 전 기술 진단(글만 기준선·정보 탐침) 신설, (라) §7 채점기를 검증된 v2로 교체, (마) 디코더 입력 z*를 저장된 최종점 `z_star` 한 종류로 고정(PREREG_S7_DEQ_v1.2 §10: 최소점은 2차 지표), (마-2) G2 PASS = 게이트 2 ∧ 3 ∧ 4, 게이트 1 보조 강등, 판정 통계 = 정확 McNemar + 대응 부트스트랩, 보조 조건 F(평균 z*), (바) §8 예산·일정 갱신(Step 3 자동 연쇄 정정, 마스터 순서 참조), (사) "v3.2" 잔존 표기 정정, (아) §9-1 Step 3 계산 경로 정정, §9-3 학습 착수 조건을 두 사전등록 잠금 + 마스터 순서 ⑥으로 교체, (자) math-verify 등 채점 의존성 버전 고정. §2·§3·§6·§7 학습 하이퍼파라미터와 게이트 1·2 수치는 불변.
- **선행 등록 문서 보존 상태**:
  - `PREREG_S7_D.md` (v1, SHA: `26a1f0e32d5108a9773eb4e95e14742cd898603b16b7b05b761666a09ff5ec91`)
  - `PREREG_S7_D_v2.md` (v2, SHA: `cf6628d26db7e595cc31a898696762c2dccf0f8316a5e41b434f3c153c27bfad`)
  - `PREREG_S7_D_v3.md` (v3, SHA: `ef597b3244d4d52f8132b581154a9ffb21b9c2c11b3dda296ce2bfee897954ec`)
  - `PREREG_S7_D_v3_1.md` (v3.1, SHA: `e6f72d4c2a35af6f295ebc62cc620895e9b323ac5bcb0142ebf7e5762ac3af82`)
  - `PREREG_S7_D_v3_2.md` (v3.2, SHA: `560c7fea987f9c065ed9339180f635eb70d49d8e654e3541a8bc011317e53f6c`)
  - `PREREG_S7_D_v3_3.md` (v3.3, SHA: `35881fc584ea81a49fcc8d032f52d251e4e01657d250606e1c270628cd357c17`, **무효 — §3 입력 양식 회귀 오류로 잠금 불인정**)
  - `PREREG_S7_D_v3_4.md` (v3.4, SHA: `88c308625fb10b62b29ef3426052ffd23f84ad5c856b071c272ab412f183c772`)

---

## 1. 목적 및 배경 (Background & Objectives)

S5 Phase 2의 $z$-only 디코더는 학습 문항(Seen 40문항)에서도 엄격 정확도 5.0%(2/40)에 그쳐, **"현 설계·규모(320문항·3000스텝)의 z-only 디코더는 학습 문항에서도 답을 생성하지 못함 (단, 용량 vs 데이터 부족은 미구분)"**으로 확정 및 종결되었다(원장 #363, #364).

이에 따라 문제 본문 토큰을 완전히 배제하던 $z$-only 방식을 공식 폐기하고, **문제 본문 텍스트(Problem Text)를 온전히 보존한 상태에서 인지적 고정점 $z^*$를 소프트 프리픽스로 결합하는 "문제 글 + $z^*$ 결합 디코더(S7-D)"**를 도입한다. 본 사전등록(v3.2에서 도입, v3.5까지 유지)은 외부 감사 의견(원장 #368)에 따라:
1. 조건 B 셔플 순열 횟수를 **주 게이트 평가용 1회 고정**으로 규정하고, 추가 순열은 보조 지표로 명시하며,
2. 평가 총 생성 수 및 건당 18.7초 실측치 기준 예상 시간을 산출하여 명시하고,
3. 본 문서를 최종 **잠금(Lock)** 처리하여 디코더 학습 착수 요건을 완성한다.

---

## 2. 모델 규격 및 정본 파라미터 (Architecture & Parameters)

### (a) 백본 및 디코더 프로젝터 파라미터 총계
- **기저 백본 모델**: **`Gemma 4 e4b`** (`google/gemma-4-e4b`)
  - 가중치 체크포인트 정본 SHA: `58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377`
  - $d_{\text{model}} = 2560$, 어휘 사전 크기 $V = 256,000$.
- **PrefixProjector 파라미터 총계**: **정확히 19,665,920개**
  - $W_{\text{proj}}$ (`nn.Linear(2560, 2560)`, bias 없음, Spectral Norm 적용): $2560 \times 2560 = \mathbf{6,553,600}$ 파라미터.
  - Adapter FC1 (`nn.Linear(2560, 2560)`, bias 포함): $2560 \times 2560 + 2560 = \mathbf{6,556,160}$ 파라미터.
  - Adapter FC2 (`nn.Linear(2560, 2560)`, bias 포함): $2560 \times 2560 + 2560 = \mathbf{6,556,160}$ 파라미터.
  - 총합: $6,553,600 + 6,556,160 + 6,556,160 = \mathbf{19,665,920}$ 파라미터.

---

## 3. 입력 양식 및 템플릿 검증 (Input Format & Verification)

### (a) 조건 A / B / D (프리픽스 결합 디코더) 프롬프트 생성 규약
하드코딩된 문자열 태그를 일체 배제하고, 반드시 토크나이저 정본 API인 `load_tokenizer().apply_chat_template`를 호출하여 생성한다:
- **User 메시지 내용 (`content`)**:
  $$\text{user\_content} = \mathbf{[\text{64 Placeholder Slots}]} + \text{"\textbackslash n\textbackslash n"} + \mathbf{\text{Problem Text}}$$
  - 64 슬롯: 단일 플레이스홀더 문자 `"1"`(Token ID `236770`)을 64회 반복한 문자열 `"1"*64`.
  - 문제 본문: MATH 데이터셋의 원본 문제 글 문자열.
- **템플릿 호출**:
  `tokenizer.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)`

### (b) 구성 토큰 ID 및 슬롯 위치 검증 (실측 검증 완료)
`raw_s7/s7_input_format_verification.json` 실측 검증 결과:
- **전체 슬롯 범위**: Token index `[4, 67]` (정확히 64개 토큰, ID `236770` 균일).
- **슬롯 선행 토큰 [0..3]**: `[2, 105, 2364, 107]` (`<bos>`, `<|turn>`, `user`, `\n`).
- **슬롯 직후 토큰 [68..]**: Token index 68은 `108` (`\n\n`), 69부터 문제 본문 토큰 시작.
- **앞 80개 토큰 ID 정본 배열**:
  ```python
  [2, 105, 2364, 107, 236770, 236770, ..., 236770, 108, 11823, 506, 2324, 529, 784, 8355, 4945, 2344, 1082, 236743, 236778]
  ```
  *(슬롯 64개가 문제 글보다 물리적으로 선행하여 인지적 조율 컨텍스트를 형성함을 사전 검증 완료).*

### (c) 조건 C ("글만", `text_only_base`) 프롬프트 생성 규약
- **동일 백본 및 동일 템플릿 API**:
  `tokenizer.apply_chat_template([{"role": "user", "content": problem_text}], tokenize=False, add_generation_prompt=True)`
- **User 메시지 내용 (`content`)**:
  - 오직 문제 본문 문자열만 포함 (`content = problem_text`).
  - **선행 64 슬롯 및 선행 `\n\n`은 일체 포함되지 않음** (Prefix 및 구분자 완전 배제).

---

## 4. 4대 평가 조건 및 순열 횟수 규약 (Conditions & Controls — 원장 #368)

1. **조건 A (`text_matched_z`)**: 문제 글 $i$ + 자기 고정점 $z_i^*$ 결합 ($\lambda=1.0$ 학습 디코더).
2. **조건 B (`text_shuffled_z`)**: 문제 글 $i$ + 타 문제 고정점 $z_j^*$ 결합 ($j \ne i$, $\lambda=1.0$ 학습 디코더).
   - **주 게이트 평가용 순열 횟수 (권고 확정)**: **1회 고정 (Fixed 1 Permutation)**.
     - 결정론적 순환 시프트 오프셋($\text{shift}=1$) 적용: $j = (i + 1) \pmod N$.
     - 주 게이트 1 판정은 엄밀히 본 1회 고정 순열 기준의 대응 차이($\Delta_{\text{disc}} = \text{acc}(\text{matched}) - \text{acc}(\text{shuffled}_1)$)로 수행.
   - **보조 지표 순열 (Auxiliary Robustness)**:
     - 추가 순열($\text{shift}=2, 3$ 및 3개 시프트 pooled 평균)은 주 게이트 판정 후 셔플 강건성 입증을 위한 보조 지표로 산출 및 보고.
3. **조건 C (`text_only_base`, "글만")**: 동일 백본(Gemma 4 e4b), 동일 프롬프트 양식에서 선행 슬롯과 `\n\n` 없이 문제 글만 전달한 순수 텍스트 기준선.
4. **조건 D (`control_projector_l0`, $\lambda=0$)**:
   - **정의**: $\lambda=0$(마진 항 없음) NLL 손실 단독으로 $\lambda=1.0$과 완전히 동일한 하이퍼파라미터 및 스텝(3,000스텝)으로 학습한 디코더 (그래디언트 차단 없음, $z^*$ 역전파 정상 수행).
   - **평가 분기**:
     - `D1`: 문제 글 $i$ + 자기 $z_i^*$ ($\lambda=0$ 디코더)
     - `D2`: 문제 글 $i$ + 타 $z_j^*$ ($\lambda=0$ 디코더, 1회 고정 시프트)
   - **통제 목적**: 대조 마진 손실($\mathcal{L}_{\text{margin}}$) 도입이 문제 특이적 차별성 강화에 미치는 순수 효과를 엄밀히 절제(Ablation) 통제.
5. **조건 E (`text_zsrc`, v3.5 추가 — DEQ 반복 귀속 대조)**:
   - **정의**: $z^*$ 대신 반복 전 초기 잠재 $z_{\text{src}}$(정본 `root_fwdpass(ctx, K=64)` 출력, 잡음 추가 전)를 같은 64 슬롯에 넣는 디코더. $\lambda=1.0$, 조건 A와 완전히 같은 하이퍼파라미터·스텝(3,000)·데이터로 **별도 학습**.
   - **평가 분기**: `E1`: 문제 글 $i$ + 자기 $z_{\text{src},i}$ / `E2`: 문제 글 $i$ + 타 $z_{\text{src},j}$ (shift=1).
   - **학습·일정**: 조건 A와 같은 레시피로 따로 학습한다(z*-학습 디코더에 $z_{\text{src}}$를 넣는 것은 분포 불일치라 무효). $z_{\text{src}}$는 $S_{\text{train}}$ 1,000 + $S_{\text{heldout}}$ 600 = **1,600문항**에 대해 `root_fwdpass` 1회로 계산(결정적, 잡음 없음). 일정은 `PREREG_S7_DEQ_v1.2` §10(f) 마스터 순서 ⑤·⑥·⑨.
   - **통제 목적**: 문제 글의 얕은 요약만으로 얻는 효과와 DEQ 고정점 반복이 더하는 효과를 분리. $z^*$는 문제 글과 시드만의 함수이므로(RCA-G1G2 §G2), 조건 A의 이득이 E1과 같다면 이득의 원천은 DEQ 반복이 아니다.
6. **조건 F (`text_mean_z`, v3.5 추가 — 보조)**: 조건 A 디코더에 문제 글 $i$ + **$S_{\text{train}}$ 1,000문항**(held-out 80 제외) `z_star`의 평균(모든 문항 동일 프리픽스). 추가 학습 없음, 생성 600건.

---

## 5. 게이트 통과 기준 (Gate Criteria — 고정 수치)

사후 조정이 불가한 2대 주 게이트 기준:

| 게이트 구분 | 지표 수식 | 합격 기준 (Fixed Cutoff) | 검증 방법 및 근거 |
| :--- | :--- | :--- | :--- |
| **주 게이트 1 (차별성)** | $\Delta_{\text{disc}} = \text{acc}(\text{matched}) - \text{acc}(\text{shuffled}_1)$ | **$95\%\text{ CI}_{\text{lower}}(\Delta_{\text{disc}}) > \mathbf{0.0\%\text{p}}$**<br>*(점추정 관측 기준: $\Delta_{\text{disc}}^{\text{obs}} \ge \mathbf{+3.0\%\text{p}}$)* | **10,000회 대응 부트스트랩 (Strict 정확도, 1회 고정 셔플)**.<br>N=600에서 95% CI 하한>0은 단측 $\alpha=0.025$ 기각을 엄격 보장하며, +3%p(약 18문항 순증)는 실질적 효과 크기를 만족함 (2종 오류 방지). |
| **주 게이트 2 (비열등성 — 바닥선)** | $\Delta_{\text{non-inf}} = \text{acc}(\text{matched}) - \text{acc}(\text{text\_only})$ | **$\Delta_{\text{non-inf}} \ge \mathbf{-2.0\%\text{p}}$** | 프리픽스 결합에 따른 Base 모델 대비 성능 저하를 2.0%p 이내로 방어. |
| **주 게이트 3 (기여 — v3.5 추가)** | $\Delta_{\text{sup}} = \text{acc}(\text{matched}) - \text{acc}(\text{text\_only})$ | 정확 McNemar 단측 $p < 0.025$ **그리고** 대응 부트스트랩 $95\%\text{ CI}_{\text{lower}} > 0.0\%\text{p}$ | 10,000회 대응 부트스트랩 + 불일치 쌍 정확 이항검정 (Strict, 문항 단위). |
| **주 게이트 4 (DEQ 귀속 — v3.5 추가)** | $\Delta_{\text{deq}} = \text{acc}(\text{matched}) - \text{acc}(\text{E1})$ | 정확 McNemar 단측 $p < 0.025$ **그리고** 대응 부트스트랩 $95\%\text{ CI}_{\text{lower}} > 0.0\%\text{p}$ | 동일. 조건 E는 별도 학습 디코더(§4-5). |


*판정 규칙 (v3.5, 원장 #392 — 결과 열람 전 고정)*:
- **G2 PASS = 주 게이트 2(바닥선) ∧ 주 게이트 3 ∧ 주 게이트 4.** 모두를 요구하는 교집합 판정이므로 다중성 보정을 하지 않는다.
- **주 게이트 1은 보조 지표로 강등**: 통과해도 "$z^*$가 **문항 특이적**이다"까지만 인정한다(타 문항 $z^*$가 문제 글과 충돌해 성능을 깎아도 통과할 수 있음). 통계는 게이트 3·4와 같은 방식으로 보고하고, "+3.0%p"는 합격 조건이 아니라 관측 보고값이다.
- 판정 통계: 대응 퍼센타일 부트스트랩만으로는 N=600에서 불일치 4:0도 CI 하한 > 0이 될 수 있으므로(EXT-0001 M12), 불일치 쌍에 대한 정확 McNemar 단측 $p<0.025$를 함께 요구한다.
- 검출력 고지: N=600 대응 설계에서 게이트 3·4는 대략 +4~5%p 이상의 차이가 있어야 통과 가능하다(원장 #392). §5-2의 글만 기준선으로 사전 검출력을 보고하되, 기준 수치는 바꾸지 않는다.
- GOAL.md G2 정의를 위 규칙으로 개정한다(`shared/prereg/DRAFT_GOAL_GRAPH.diff`, Principal 승인).

*보조 관찰 지표*:
- 정답 해설 토큰의 float32 청크 NLL(LESSONS 1): $\text{NLL}(A)$ vs $\text{NLL}(C)$, $\text{NLL}(A)$ vs $\text{NLL}(B)$, $\text{NLL}(A)$ vs $\text{NLL}(E1)$ 문항 단위 대응 차이의 95% CI. 정답 수가 적을 때 정확도보다 검출력이 높은 연속 지표.
- 조건 D 통제 분리: $(\Delta_{\text{disc}})_{\lambda=1} - (\Delta_{\text{disc}})_{\lambda=0} > 0$.
- 조건 F(평균 $z^*$ 프리픽스, 조건 A 디코더): $\text{acc}(A) - \text{acc}(F)$. 게이트 3의 차이가 $z^*$ 내용이 아니라 학습된 프리픽스의 서식 효과(EXT-0001 M11)에서 오는지 가르는 보조 지표.
- 조건 B 추가 2순열(시프트 2, 3) 및 Pooled 차별성 강건성 확인.
- Lenient 수치 정답 추출 정확도에서도 동일 방향성 확인.

---

## 5-2. 학습 전 기술 진단 (v3.5 추가 — 게이트 아님, 결과로 게이트·조건을 바꾸지 않음)

1. **글만 기준선 선계산**: 조건 C는 학습과 무관하므로 디코더 학습 전에 held-out 600에서 먼저 생성·채점해도 된다. 목적: 기준 정확도를 알아 게이트 3·4의 검출력을 사전 보고. 본 문서 게이트 수치는 이 결과를 보기 전에 잠근다.
2. **정보 탐침**: $S_{\text{train}}$ 1,000문항에 대해 $z^*$(평균 풀링), $z_{\text{src}}$(평균 풀링), 문제 글 입력 임베딩 평균의 세 표현으로 같은 로지스틱 탐침(5겹 교차검증, 고정 시드)을 학습해 MATH `subject`(5종: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory — `math_train_prompts_5000.jsonl` 집계)·`level`(Level 1~5, `Level ?` 2건 제외) 예측 정확도를 보고.
   - 해석(사전 고정): $z^*$ 정확도가 $z_{\text{src}}$ 대비 두 과제 모두 +1%p 미만이면 "DEQ 반복의 정보 이득 미관측"으로 기록하고 S8(음함수 미분 공동학습) 우선순위를 올린다. S7-D 진행 여부는 바꾸지 않는다.

---

## 5-1. DEQ 고정점 수렴 게이트 척도 정정 (원장 #375·#377·#378)

### (a) 기존 척도의 물리적 한계 및 정정 배경
- 기존 $\rho = \frac{\|z_{k+1} - z_k\|}{\|z_k\|}$는 감쇠 보폭 $\alpha$가 곱해진 **1보 이동량(Step Displacement)**에 불과함:
  $$\rho = \alpha \cdot \frac{\|f(z_k) - z_k\|}{\|z_k\|} = \alpha \cdot r$$
- 따라서 $\alpha$가 $0.5 \to 0.25$ 등으로 변경되면 $\rho$의 수치가 절반으로 축소되어 참 고정점 오차를 과소평가하게 됨.
- 이에 따라 DEQ 수렴 게이트(①-a, ①-b) 및 수렴 구간 표의 공식 척도를 **$\alpha$ 무관 참 고정점 잔차(True Fixed-Point Residual)**:
  $$r = \frac{\|f(z) - z\|}{\|z\|} = \frac{\rho}{\alpha}$$
  로 전면 정정한다.

### (b) 이중 수렴 기준 (Dual Convergence Criteria)
1. **엄격 기준 (Strict DEQ Convergence)**:
   $$r < \mathbf{5.0 \times 10^{-3}}$$
   *(DEQ 원 논문 Bai et al. 2019 및 엄밀 고정점 수렴 기준)*
2. **S5 동치 기준 (S5 Equivalent Tolerance)**:
   $$r < \mathbf{1.0 \times 10^{-2}}$$
   *(S5 시절 $\alpha=0.5$ 감쇠 이동량 $\rho < 5.0 \times 10^{-3} \iff r = \frac{\rho}{0.5} < 1.0 \times 10^{-2}$와 물리적으로 동일한 허용 오차)*

### (c) 수렴 구간 표 정의 (4구간)
| 구간 명칭 | 참 잔차 $r$ 범위 | 물리적 상태 및 의미 |
| :--- | :--- | :--- |
| **초고정밀 수렴** | $r < 1.0 \times 10^{-3}$ | 초고정밀 고정점 도달 |
| **엄격 수렴** | $1.0 \times 10^{-3} \le r < 5.0 \times 10^{-3}$ | 원 논문 기준 고정점 엄격 도달 |
| **S5 동치 완보 수렴** | $5.0 \times 10^{-3} \le r < 1.0 \times 10^{-2}$ | S5 기존 $\rho < 5\times 10^{-3}$ 동치 수렴 |
| **미수렴 (Non-converged)** | $r \ge 1.0 \times 10^{-2}$ | 고정점 미도달 (미수렴군 잔류) |

### (d) DEQ 수렴 게이트 ①-a / ①-b 정정 기준
- **①-a (Train 수렴 비율)**: Train 1,000문항 중 $r < 1.0 \times 10^{-2}$ (S5 동치) 비율 $\ge 95\%$, $r < 5.0 \times 10^{-3}$ (엄격) 비율 병기 보고.
- **①-b (Held-out 수렴 비율)**: Held-out 600문항 중 $r < 1.0 \times 10^{-2}$ (S5 동치) 비율 $\ge 90\%$, $r < 5.0 \times 10^{-3}$ (엄격) 비율 병기 보고.
- **v3.5 주석**: ①-a·①-b의 1차 판정값은 최종점 $z_{1000}$에서 새로 평가한 $r_{1000}$이다(위 (e)의 $4\rho_{1000}$은 실제로 $r_{999}$로 보고용). 최소 잔차점은 `PREREG_S7_DEQ_v1.2` §10의 2차 등록 지표이며, 승격 판정과 승격 시 ①-a·①-b 정의는 그 문서 §10(c)가 정본이다.

### (e) S7 기존 400문항 완료 보고 규약
- 현재 $\alpha=0.25$로 GPU 0에서 산출되는 $\rho_{1000}$에 대해, $r = \frac{\rho_{1000}}{0.25} = 4 \times \rho_{1000}$으로 즉시 환산하여 이중 기준(엄격 $5\times 10^{-3}$, S5 동치 $1\times 10^{-2}$) 및 4구간 표로 보고한다.

### (f) S6 중간 보고 원장 정정 기록 명시 (원장 #377 지침 2)
- S6 중간 보고의 "43/54 해소"는 감쇠 보폭 $\rho < 5\times 10^{-3}$ 기준이었으며, 참 잔차 $r$ 기준으로 정정한 수치는 다음과 같다:
  - $\alpha=0.25$: $r < 1.0 \times 10^{-2}$ (S5 동치) **19/54**, $r < 5.0 \times 10^{-3}$ (엄격) **5/54**.
  - $\alpha=0.35$: $r < 1.0 \times 10^{-2}$ (S5 동치) **17/54**, $r < 5.0 \times 10^{-3}$ (엄격) **4/54**.
  - **결론**: 감쇠 채택 없음(감쇠 단독으로 미수렴 해소 불가) 판정 불변 유지.

---

## 6. 학습·평가 데이터 분할 및 서로소·앵커 배제 증명 (Data Splits)

### (a) 데이터 분할 총괄
- **전체 풀**: MATH train 5,000문항 (`math_train_prompts_5000.jsonl`, 인덱스 $0 \sim 4999$).
- **기존 베이스 제외 풀 (420문항)**:
  - 기존 400문항: `train_320` (320문항) + `heldout_80` (80문항).
  - 앵커 20문항: `canonical_20_indices` (Stage 4/5 대조 20문항).
  - *실측 확인: 앵커 20문항과 기존 400문항 간의 중복 수는 엄밀히 0개 ($\text{Overlap} = 0$). 따라서 제외 풀은 정확히 420문항.*
- **신규 1,200문항 추출 풀**: $5,000 - 420 = \mathbf{4,580}$문항.

### (b) SHA-256 엔트로피 시드 추출 규약 (실측 완료)
- 엔트로피 문자열: `CTS-random-select-v1|1293f2789a31a8ae449f9d10924e2bbc3d06c459|S7_SPLIT_1200_NEW_ITEMS`
- 엔트로피 SHA-256: `d5c63ec1d1bade5e477ea3815bad6110acea1a19212e263d1fccb83eced8f861`
- 유도 시드 (`uint64`): `15404068577635655262`
- **추출 결과**:
  - **신규 학습 680문항** 추출 (`configs/s7_new_train_680.json`) $\implies$ 기존 train 320과 병합하여 **S7 학습 1,000문항 ($S_{\text{train}}$)** 확정 (`configs/s7_train_1000.json`).
  - **신규 평가 520문항** 추출 (`configs/s7_new_heldout_520.json`) $\implies$ 기존 heldout 80과 병합하여 **S7 평가 600문항 ($S_{\text{heldout}}$)** 확정 (`configs/s7_heldout_600.json`).

### (c) 서로소 및 앵커 배제 증명 (Disjointness Proof)
- $S_{\text{train}} \cap S_{\text{heldout}} = \emptyset$ (정확히 0개 중복, `isdisjoint = True` 확인).
- $S_{\text{train}} \cap \text{Canonical\_20} = \emptyset$ (앵커 20 완전 배제 확인).
- $S_{\text{heldout}} \cap \text{Canonical\_20} = \emptyset$ (앵커 20 완전 배제 확인).
- **검증 로그**: `raw_s7/s7_split_verification.log` (SHA: `df661e7dcf53f2d304381254dc215019bb690fadb23e0d2739b7f5d51ad3ea7d`).

---

## 7. 학습 및 평가 설정 규약 (S5 코드 및 로그 실측 정본 대조)

본 설정은 S5 실측 코드(`scripts/s5_train_decoder_v2.py`) 및 Phase 2 Run 4 실행 로그(`raw_s5/phase2_pipeline_run4.log`)의 파라미터를 행 번호 단위로 대조하여 정확히 일치시킨 정본 규약이다:

| 파라미터 항목 | 설정 값 (S7-D 확정값) | S5 코드 출처 (`s5_train_decoder_v2.py`) | S5 실행 로그 출처 (`phase2_pipeline_run4.log`) |
| :--- | :--- | :--- | :--- |
| **옵티마이저 (Optimizer)** | AdamW (분리 학습률) | `lines 686-691` (`torch.optim.AdamW`) | — |
| **Weight Decay** | `0.01` (코드 기본값) | `line 1207` (`default=0.01`) | — |
| **학습률 $W_{\text{proj}}$** | **상수 `1.0e-4` (`1.0000e-04`)** | `line 1205` (`--lr_wproj default=1e-4`) | `line 34` (`lr=1.0000e-04 (const)`) |
| **학습률 Adapter** | **상수 `3.0e-4` (`3.0000e-04`)** | `line 1206` (`--lr_adapter default=3e-4`) | `line 1156` (`adapter: 3e-4`) |
| **스케줄러 (Scheduler)** | **`None` (스케줄러 없음, 완전 상수 lr)** | `line 692` (`scheduler = None`) | `line 34` (`sched=None`) |
| **Micro-batch 크기** | **`2`** | `line 1202` (`--micro_batch_size default=2`) | `line 17` (`micro: 2`) |
| **누적 스텝 (Grad Accum)** | **`4`** | `line 1203` (`--grad_accum_steps default=4`) | `line 17` (`accum: 4`) |
| **유효 배치 크기 (Effective)** | **`8`** ($2 \times 4$) | `line 1201` (`--batch_size default=8`) | `line 17` (`Batch: 8`), `line 34` (`eff 8`) |
| **대조 마진 ($m$)** | **`1.0` nat** | `line 1199` (`--margin_m default=1.0`) | `line 17` (`Margin: 1.0`) |
| **대조 손실 가중치 ($\lambda$)** | **$\lambda \in \{1.0, 0.0\}$** | `lines 1197-1198` (`--loss_lambda default=1.0`) | `line 17` (`Lambda: 1.0`) |
| **최적화 총 스텝 (Steps)** | **`3,000` steps** | `line 1200` (`--max_steps default=3000`) | `line 17` (`Steps: 3000`) |
| **Gradient Checkpointing** | **활성화 (`use_reentrant=False`)** | `lines 699-700` (`gradient_checkpointing_enable`) | `line 31` (`ENABLED`) |
| **Target Loss Masking** | 문제 글 및 슬롯 마스킹, **해설 토큰만 CE 역전파** | `lines 776-780, 894-895` (`ignore_index=-100`) | — |
| **평가 디코딩 규약** | Greedy (`do_sample=False`, `max_new_tokens=512`) | `s5_eval_gate_v3.py:108-115` | — |
| **채점 정규화 규약 (v3.5 교체)** | Strict = `\boxed` 추출 + math-verify 기호 동등성(첫 숫자 fallback 없음), Lenient = 단어 경계 산문 패턴(마지막 숫자 fallback 없음, **보조 지표 전용**) | `scripts/s5_eval_grader_v2.py` SHA `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` (venv `/home/shoon/venv_eval`: math-verify 0.9.0, sympy 1.14.0, latex2sympy2_extended 1.11.0 — 2026-09-28 서버 확인, 버전 불일치 시 채점 실행 0). 단위 20/20·Phase 2 재채점 (`raw_s7/unit_grader_TASK0002.log`, AUDIT-0002 PASS). 알려진 lenient 오판 유형: record 159 (`2x - 1.`→1) | — |
| **디코더 입력 z* (v3.5 추가, 한 종류 고정)** | S7 결과 파일의 `z_star` 필드 = 정본 풀이기 최종점 $z_{1000}$ (학습 1,080: `zstar_s7_a0.25_400.pt`·`_train680.pt`, 평가 520: `s7_zstar_lowest_run.py`의 `z_star` — 정본과 같은 필드). 최소점 `z_lowest`는 디코더에 쓰지 않는다. | `scripts/s6_damping_run.py` 저장 필드 `z_star`, 서버 `zstar_canonical.py` picard 분기 `z_star = z_cur` | 미결: S5 Phase 2 디코더 입력이 같은 종류였는지 B가 S5 기록으로 확인(비교 가능성) |
| **수렴 집계 (v3.5 추가)** | G1 판정값은 저장 텐서로 새로 평가한 $r_{1000}$ 하나(`PREREG_S7_DEQ_v1.2` §10(b)). 로그 기반 값은 `rho@1000`만 보고용으로 사용($4\rho_{1000} = r_{999}$). `rho@2000`은 `max_nfe=1000`에서 마지막 값의 복사이므로 사용 금지(AUDIT-0001 재감사) | — | — |

- **손실 함수 수식**:
  - $\lambda = 1.0$ (조건 A, B):
    $$\mathcal{L} = \mathcal{L}_{\text{NLL}}(\text{matched}) + 1.0 \times \max\Big(0,\; 1.0 - \big(\mathcal{L}_{\text{NLL}}(\text{shuffled}) - \mathcal{L}_{\text{NLL}}(\text{matched})\big)\Big)$$
  - $\lambda = 0.0$ (조건 D):
    $$\mathcal{L} = \mathcal{L}_{\text{NLL}}(\text{matched})$$
    *(마진 페널티 항 배제, NLL 단독 역전파, 그래디언트 차단 없음).*

---

## 8. z* 계산 총량 및 평가 총 생성 예산 (원장 #368 명시)

### (a) z* 계산 총량 (실측치 71.2초/문항 근거)
- **기준 실측치**: 현재 가동 중인 S7 GPU 0 실측치 = **문항당 평균 71.2초** (중앙값 58.2초, Picard 1,000 NFE 고정, $tol=0.0$).
- **총 계산 대상**: 1,600문항 (기존 400문항 재계산 + 신규 1,200문항).

| 단계 (Phase) | 대상 문항 수 | 문항당 소요 시간 | GPU 0 단독 시간 | 진행 상태 |
| :--- | :---: | :---: | :---: | :--- |
| **Step 1 (기존 400문항 재계산)** | 400문항 | 71.2초 (실측 평균) | **7.91시간** (475분) | **실행 중 (GPU 0 tmux `s7_zstar_gpu0`)** |
| **Step 2 (신규 train 680문항)** | 680문항 | 71.2초 (실측 평균) | **13.45시간** (807분) | Step 1 직후 자동 연쇄 실행 |
| **Step 3 (신규 heldout 520문항)** | 520문항 | 71.2초 (실측 평균) | **10.28시간** (617분) | Step 2 직후 자동 연쇄 실행 |
| **전체 합계 (Total)** | **1,600문항** | 71.2초 (평균) | **31.64시간** | **원격 파이프라인 자동화 완료** |

### (b) 평가 총 생성 수 및 소요 시간 (실측치 18.7초/건 기준 — 원장 #368)
- **기준 실측치**: S5 Phase 2 실측치 = **문항 생성당 평균 18.7초** (`max_new_tokens=512`, Greedy decoding).
- **평가 데이터셋**: S7 Held-out $N = 600$문항.

| 평가 조건 및 범위 | 조건별 대상 문항 수 | 총 생성 건수 (Generations) | 단일 GPU 소요 시간 (18.7s/건) | 2 GPU 분산 시 소요 시간 | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **① 주 게이트 핵심 3조건**<br>(조건 A + B 1회 고정 + C 글만) | 각 600문항 | **1,800건** | **9.35시간** (561분) | **4.68시간** (280분) | **주 게이트 1, 2 즉시 판정** |
| **② 4대 전체 조건 정규 평가**<br>(A + B 1회 + C + D1 + D2 1회) | 각 600문항 | **3,000건** | **15.58시간** (935분) | **7.79시간** (468분) | **$\lambda=0$ 마진 효과 절제 통제 포함** |
| **③ 보조 지표 순열 확장**<br>(B 추가 2회 + D2 추가 2회) | 각 1,200건 | **+2,400건**<br>*(총 5,400건)* | **+12.47시간**<br>*(총 28.05시간)* | **+6.23시간**<br>*(총 14.03시간)* | 셔플 강건성 보조 확인용 |
| **④ 조건 E 추가 (v3.5)**<br>(E1 + E2 1회) | 각 600문항 | **+1,200건** | **+6.23시간** | **+3.12시간** | 주 게이트 4 판정. 조건 E 디코더 학습 3,000스텝 별도(조건 A와 같은 시간) |

- **주 게이트 1~4 최소 생성 세트 (v3.5)**: A + B(1회) + C + E1 = **2,400건**, 단일 GPU 약 **12.47시간**. 보조 조건 F +600건(약 3.12시간).
- **v3.5 정정**: 위 (a)표의 \"Step 3 자동 연쇄 실행\"은 사실과 다르다 — 감시 스크립트가 Step 2 완료 시 tmux 세션을 종료하므로 Step 3는 `PREREG_S7_DEQ_v1.2` §10(f) 순서로 따로 착수한다. 문항당 71.2초 대신 Step 2 관측 평균 약 125초(EXT-0001 집계)를 쓴다.
- **GPU 0 직렬 일정 (v3.5)**: `PREREG_S7_DEQ_v1.2` §10(f) 마스터 순서를 따른다 — ⑤ $z_{\text{src}}$($S_{\text{train}}$) → ⑥ 학습 A($\lambda=1$) → D($\lambda=0$) → E($\lambda=1$, $z_{\text{src}}$) → ⑦ Step 3 held-out 520 → ⑨ $z_{\text{src}}$($S_{\text{heldout}}$) → ⑩ 생성: C → A·B·E1·F → D1·D2·E2·보조 순열.

---

## 9. 즉시 집행 및 자원 할당 규약 (원장 #368)

1. **z* 계산 파이프라인 가동**:
   - 2026-09-27 22:09 가동 개시 확인 완료.
   - 실행 순서: ① 기존 400 (`results_s7/zstar_s7_a0.25_400.pt`) $\to$ ② 신규 학습 680 (`results_s7/zstar_s7_a0.25_train680.pt`) $\to$ ③ 신규 평가 520 (`results_s7/zstar_s7_a0.25_heldout520.pt`).
   - **v3.5 정정**: 파이프라인은 감시 스크립트가 1,080(①+②)에서 정지시킨다. ③ 신규 평가 520은 파이프라인이 아니라 `PREREG_S7_DEQ_v1.2` §10(e)의 `scripts/s7_zstar_lowest_run.py`로, 같은 문서 §10(f) 마스터 순서 ⑦에서 따로 계산한다(`z_star` 필드는 정본과 동일).
   - 문항 단위 증분 저장(`.partial.pt`) 및 재개(Resume) 동작 중.
   - **§0 중간 보고 시점**: 학습 문항 $z^*$ 완료 시점 (기존 400 + 신규 학습 680 = 총 1,080문항 완료 시 즉시 보고).
2. **GPU 1 투입 분기 규약**:
   - **Principal의 GPU 1 허용 문구 수신 시**:
     - GPU 0: 학습 문항 $z^*$ 완료 즉시 디코더 학습($\lambda=1.0$) 착수.
     - GPU 1: 평가 520문항 $z^*$ 계산 담당 $\to$ 이후 디코더 학습 분산($\lambda=0.0$ 병렬 진행) $\to$ 평가 생성 600문항 절반(300문항씩) 분산.
   - **허용 문구 미수신 시**:
     - GPU 0 단독으로 전 과정 순차 진행 (GPU 1 미할당 유지).
3. **디코더 학습 착수 승인**:
   - 본 문서(`PREREG_S7_D_v3_5.md`)와 `PREREG_S7_DEQ_v1.2` 잠금 후, 그 문서 §10(f) 마스터 순서 ⑥에서 디코더 학습(조건 A·D·E)에 착수한다. 학습 입력은 저장된 `z_star` 그대로(재생 없음).

---

## 10. 사전등록 잠금 (초안 — 미잠금)

본 문서(`PREREG_S7_D_v3_5.md`)는 v3.4의 학습 하이퍼파라미터·입력 양식·데이터 분할·게이트 1·2 수치를 그대로 유지하고, 조건 E·주 게이트 3·4·주장 해석 규칙·학습 전 기술 진단·검증된 채점기·z* 정의 통일을 추가한 **초안**이다. 외부 감사인과 Principal의 승인 문구(및 GOAL.md G2 개정 결정) 수신 후 잠금하며, 잠금 전에는 디코더 학습에 착수하지 않는다.

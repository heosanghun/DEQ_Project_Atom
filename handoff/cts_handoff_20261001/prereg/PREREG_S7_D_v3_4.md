# PREREG_S7_D_v3_4.md — 문제 글 + z* 결합 디코더 (S7-D) 사전등록 v3.4 (원장 #377, #378 — 척도 정정 및 잠금)

- **등록 일시**: 2026-09-28
- **지침**: 원장 #363, #364, #365, #366, #367, #368, #375, #377, #378 (S7-D v3.4 최종 잠금안)
- **발신**: System 2 / **수신**: Antigravity / **사본**: Principal, 외부 감사인
- **소속**: Stage 7 (S7-D: Text + z* Augmented Generation)
- **문서 잠금 상태**: **LOCKED (잠금 완료 — 본 문서 기준 척도 정정 반영 및 디코더 학습 착수 승인)**
- **선행 등록 문서 보존 상태**:
  - `PREREG_S7_D.md` (v1, SHA: `26a1f0e32d5108a9773eb4e95e14742cd898603b16b7b05b761666a09ff5ec91`)
  - `PREREG_S7_D_v2.md` (v2, SHA: `cf6628d26db7e595cc31a898696762c2dccf0f8316a5e41b434f3c153c27bfad`)
  - `PREREG_S7_D_v3.md` (v3, SHA: `ef597b3244d4d52f8132b581154a9ffb21b9c2c11b3dda296ce2bfee897954ec`)
  - `PREREG_S7_D_v3_1.md` (v3.1, SHA: `e6f72d4c2a35af6f295ebc62cc620895e9b323ac5bcb0142ebf7e5762ac3af82`)
  - `PREREG_S7_D_v3_2.md` (v3.2, SHA: `560c7fea987f9c065ed9339180f635eb70d49d8e654e3541a8bc011317e53f6c`)
  - `PREREG_S7_D_v3_3.md` (v3.3, SHA: `35881fc584ea81a49fcc8d032f52d251e4e01657d250606e1c270628cd357c17`, **무효 — §3 입력 양식 회귀 오류로 잠금 불인정**)

---

## 1. 목적 및 배경 (Background & Objectives)

S5 Phase 2의 $z$-only 디코더는 학습 문항(Seen 40문항)에서도 엄격 정확도 5.0%(2/40)에 그쳐, **"현 설계·규모(320문항·3000스텝)의 z-only 디코더는 학습 문항에서도 답을 생성하지 못함 (단, 용량 vs 데이터 부족은 미구분)"**으로 확정 및 종결되었다(원장 #363, #364).

이에 따라 문제 본문 토큰을 완전히 배제하던 $z$-only 방식을 공식 폐기하고, **문제 본문 텍스트(Problem Text)를 온전히 보존한 상태에서 인지적 고정점 $z^*$를 소프트 프리픽스로 결합하는 "문제 글 + $z^*$ 결합 디코더(S7-D)"**를 도입한다. 본 사전등록 v3.2는 외부 감사 의견(원장 #368)에 따라:
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

---

## 5. 게이트 통과 기준 (Gate Criteria — 고정 수치)

사후 조정이 불가한 2대 주 게이트 기준:

| 게이트 구분 | 지표 수식 | 합격 기준 (Fixed Cutoff) | 검증 방법 및 근거 |
| :--- | :--- | :--- | :--- |
| **주 게이트 1 (차별성)** | $\Delta_{\text{disc}} = \text{acc}(\text{matched}) - \text{acc}(\text{shuffled}_1)$ | **$95\%\text{ CI}_{\text{lower}}(\Delta_{\text{disc}}) > \mathbf{0.0\%\text{p}}$**<br>*(점추정 관측 기준: $\Delta_{\text{disc}}^{\text{obs}} \ge \mathbf{+3.0\%\text{p}}$)* | **10,000회 대응 부트스트랩 (Strict 정확도, 1회 고정 셔플)**.<br>N=600에서 95% CI 하한>0은 단측 $\alpha=0.025$ 기각을 엄격 보장하며, +3%p(약 18문항 순증)는 실질적 효과 크기를 만족함 (2종 오류 방지). |
| **주 게이트 2 (비열등성)** | $\Delta_{\text{non-inf}} = \text{acc}(\text{matched}) - \text{acc}(\text{text\_only})$ | **$\Delta_{\text{non-inf}} \ge \mathbf{-2.0\%\text{p}}$** | 프리픽스 결합에 따른 Base 모델 대비 성능 저하를 2.0%p 이내로 방어. |

*보조 관찰 지표*:
- 조건 D 통제 분리: $(\Delta_{\text{disc}})_{\lambda=1} - (\Delta_{\text{disc}})_{\lambda=0} > 0$.
- 조건 B 추가 2순열(시프트 2, 3) 및 Pooled 차별성 강건성 확인.
- Lenient 수치 정답 추출 정확도에서도 동일 방향성 확인.

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
| **채점 정규화 규약** | MATH official boxed strict match + lenient numeric | `cts/eval/math500.py`, `lenient_extract.py` | — |

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

---

## 9. 즉시 집행 및 자원 할당 규약 (원장 #368)

1. **z* 계산 파이프라인 가동**:
   - 2026-09-27 22:09 가동 개시 확인 완료.
   - 실행 순서: ① 기존 400 (`results_s7/zstar_s7_a0.25_400.pt`) $\to$ ② 신규 학습 680 (`results_s7/zstar_s7_a0.25_train680.pt`) $\to$ ③ 신규 평가 520 (`results_s7/zstar_s7_a0.25_heldout520.pt`).
   - 문항 단위 증분 저장(`.partial.pt`) 및 재개(Resume) 동작 중.
   - **§0 중간 보고 시점**: 학습 문항 $z^*$ 완료 시점 (기존 400 + 신규 학습 680 = 총 1,080문항 완료 시 즉시 보고).
2. **GPU 1 투입 분기 규약**:
   - **Principal의 GPU 1 허용 문구 수신 시**:
     - GPU 0: 학습 문항 $z^*$ 완료 즉시 디코더 학습($\lambda=1.0$) 착수.
     - GPU 1: 평가 520문항 $z^*$ 계산 담당 $\to$ 이후 디코더 학습 분산($\lambda=0.0$ 병렬 진행) $\to$ 평가 생성 600문항 절반(300문항씩) 분산.
   - **허용 문구 미수신 시**:
     - GPU 0 단독으로 전 과정 순차 진행 (GPU 1 미할당 유지).
3. **디코더 학습 착수 승인**:
   - 본 `PREREG_S7_D_v3_2.md` 잠금에 따라, 학습 문항 $z^*$ 연산 완료 즉시 디코더 학습 착수 가능 상태로 전환.

---

## 10. 사전등록 잠금 (Document Locked)

본 문서(`PREREG_S7_D_v3_4.md`)는 원장 #363부터 #368, #375, #377, #378에 이르는 모든 외부 감사 의견, DEQ 참 잔차 척도 정정($r = \rho/\alpha$), 파라미터 실측치, 입력 템플릿 검증 배열, 데이터 분할 서로소 증명, 게이트 컷오프 수치, 평가 생성 예산이 완비되었으므로 **최종 잠금(LOCKED)** 처리한다. 사후 수정은 절대 불가하다.

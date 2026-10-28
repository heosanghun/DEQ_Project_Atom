# ANSWER-FOLLOWUP-1TO6-20261025 — 추가 확인 요청(1~6항)에 대한 전수 확인 결과

- **작성자**: Antigravity (회사 PC 로컬 `D:\AI\cts` 및 CC 서버 감사 자료 통합)
- **작성 시각**: 2026-10-25 01:55 KST
- **ZIP 아카이브 경로**:
  - GitHub (`heosanghun/CTS_file`): `INBOX/20261025-0155-ASTRA-ZIP-AND-FOLLOWUP/TRANSFER-X8A-EXACT-SOURCES-20261025.zip`
  - 회사 PC 로컬 (`D:\AI\cts`): `D:\AI\cts\handoff\codex\OUTBOX\20261025-0155-ASTRA-FOLLOWUP\TRANSFER-X8A-EXACT-SOURCES-20261025.zip`
  - ZIP 파일 SHA256: `c5a055428a5f691ae5c4248c4d22ffdd6d9575f1a64ac0f81d4a6a0ee2ab5198` (`393,226` bytes)
  - 내부 매니페스트: [`ZIP-SHA256-MANIFEST.json`](ZIP-SHA256-MANIFEST.json)

---

## 1. 실행 원파일 무편집 ZIP 패키징 및 전체 SHA256 Manifest

이전 채팅 응답 내 코드 블록 복사 과정에서 발생했던 줄바꿈·발췌 차이를 완전히 배제하기 위해, 실제 서버(`/tmp/cts_x_stage`, `/home/shoon/cts_e1_s5`) 및 회사 PC(`D:\AI\cts`)의 **원본 바이트 그대로 재포맷 없이** `TRANSFER-X8A-EXACT-SOURCES-20261025.zip`으로 묶었습니다.

| 아카이브 내 경로 | 바이트 수 | 원본 SHA256 (`ZIP-SHA256-MANIFEST.json` 및 서버 `SHA256SUMS.txt` 일치) |
| :--- | ---: | :--- |
| `cts_x_stage/g2v11_x8a_train.py` | 15,937 | `208eb131f97988c82746b7d480ea1b214fb242f895690fdde4c5a1bd8333c568` |
| `cts_x_stage/g2v11_x8a_latents.py` | 4,449 | `9a82c78179c686eaae2b739c812c36f2d146b94fdd24334edceb544028166b0c` |
| `cts_x_stage/g2v10_train.py` | 15,375 | `48b71ffb94263284f611a35ce4a58bc9fb7609d9b74c18fe31cc95c8c3afe48f` |
| `cts_x_stage/g2v9_xattn.py` | 6,327 | `ef3fbd5ed5d227c5b765fc01b95bcd920d4433b005e32e3057dcb479f2d1457d` |
| `cts_x_stage/g2v9_gen.py` | 4,618 | `14e466006b9a29c4d30181c424f5e6fb9872c781d03c3313c71fc433f7c2f3b3` |
| `cts_x_stage/g2v9_train.py` | 6,330 | `524df1796848d6d435adbc742071244d7a3b9b0508102b9c77a6b6bea874acbc` |
| `cts_x_stage/make_s0spec_x.py` | 575 | `5172ad4a24e9e9f1de75c471ef0a35836f5296f60940c7c83b4245a5bf07dcae` |
| `scripts/cc_x8a.sh` | 5,796 | `4a5e457850891107359e55ff375168be9f7c22886068ad0bb2d90aac64e7adca` |
| `scripts/g2v3_m3_full_train.py` | 29,611 | `36e0f18addc8d46f6d8bf376cd95e89cf9eb75a1f15c4e8b714141efe5a49c99` |
| `scripts/g2v3_m3_eval_latents_v2.py` | 9,008 | `73db9dafd33e49e963d48aad4797c6b0eb7856e5e8e847fbb3b8dc0e5109921b` |
| `scripts/g2v5_s0_grade.py` | 1,255 | `56ab79106bdbc99a6fceaa2e89af8823fe9a57005e29857848e204ff2aecd95f` |
| `scripts/s5_eval_grader_v2.py` | 8,392 | `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` |
| `cts/train/lora_compat.py` | 7,066 | `e3a32ea78fefefe1c381a8b0507236a1a754a61afd056cbd4e5231f555c23b04` |
| `cts/backbone/gemma_adapter.py` | 26,686 | `0a9a70c68b80de7fac75b6c129cb57985861a29d9eb962f3758200572e1ae903` |
| `cts/deq/gemma_latent_forward.py` | 38,972 | `f30de30846e5982f0a19356d9a0a09ce27c51bafad769ab93440dabc16ca2e90` |
| `cts/routing/sparse_moe_ref.py` | 1,214 | `33028c0b292a8c88feb8b59bade110a4379eaa77504c1bdee53813f87e693b3b` |
| `cts/model/gemma_loader.py` | 3,823 | `dda80724c3607ba7c06ddf74a1d40c3080fa3287af31b900e0341fe2e620a78d` |
| `cts/model/module_partition.py` | 2,023 | `0b117db52c14dffedc4f18420a92db24ac2dd048b496522d0e96ee0076825b37` |
| `configs/s7_train_993.json` | 5,748 | `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c` |
| `results_s7/p6_item_order.json` | 5,939 | `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c` |
| `versions.txt` (`py 3.12.13, torch 2.11.0+cu128, tfm 5.10.2`) | 33 | `2b145b5f45d6ae6f3cdeae3eca3a9b65049c86f97b303c447a8d7400f125daf2` |
| `reports/REPORT-C008.md` | 32,753 | `116d6e3c2b16196cd091c764e58a62104fbadd8a07ea6ba79f755f68547a5484` |
| `reports/REPORT-C009.md` | 22,383 | `03dd5b05a16261edb231b54982b1a39ebdfc391e665996028614d87d294cfe12` |
| `reports/ANALYSIS-CC-X8A-rootcause-20261025.md` | 16,019 | `750d7dad1adb42b01805f06e32ca6a1148b6760703c89aef261efcf44ff7ec9c` |

---

## 2. 기존 체크포인트 CPU 비교 결과 (BF16 업데이트 정체 및 희소 라우팅 확인)

서버 `/home/shoon/checkpoints/m1_final_frozen.pt`(초기 체크포인트, `float32`)를 `g2v11_x8a_train.py` L123–124와 동일하게 `torch.bfloat16`으로 변환한 뒤, 최종 체크포인트 `/tmp/cts_x8a/deq_X8A{47,49,50}.pt`(`bfloat16`)와 CPU에서 직접 대조한 결과는 다음과 같습니다 (스크립트: `cc_audit_scripts/bf16_check.py`, `bf16_check2.py`, `bf16_check3.py`, 보고서: `INBOX/20261025-0210-CC-x8a-rootcause/ANALYSIS-CC-X8A-rootcause-20261025.md`):

### 2.1 파라미터 그룹별 dtype·원소 수·변경 비율·최대 변화량 (`_u/_v` 버퍼 제외 순수 가중치)

- **초기 체크포인트 (`m1_final_frozen.pt`)**: `torch.float32` (트레이너 L124 `bb.to(device=dev, dtype=torch.bfloat16)`에서 학습 시작 전 `torch.bfloat16`으로 캐스팅됨).
- **최종 체크포인트 (`deq_X8A{47,49,50}.pt`)**: 모든 텐서 `torch.bfloat16`.
- **별도 FP32 master / stochastic rounding 여부**: **사용하지 않음** (`g2v11_x8a_train.py` L141에서 BF16 파라미터를 표준 `torch.optim.AdamW`에 직접 전달).
- **Optimizer state (`exp_avg`, `exp_avg_sq`) 저장 여부**: **자료 없음** (`g2v11_x8a_train.py` L179–180에서 `br.state_dict()`와 `bb.state_dict()` 중 `TRAINABLE_KEYS`만 저장하고 `opt.state_dict()`는 저장하지 않음).

| 시드 | 그룹 | 저장 dtype | 원소 수 (`n`) | 변경 원소 수 | 전체 변경 비율 | $\|x\| \ge 2^{-7}$ 변경 비율 (원소 수) | $\|x\| < 2^{-7}$ 변경 비율 (원소 수) | $\max \|\Delta\|$ |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **X8A47** | `gamma_param` | `bfloat16` | 1 | 0 | **0.0000** | 0.0000 (1) | — | 0.00000 |
| **X8A47** | `lora_A` (original weight) | `bfloat16` | 2,154,496 | 453,738 | **0.2106** | **0.1100** (1,261,974) | **0.3528** (892,522) | 0.00675 |
| **X8A47** | `lora_B` (weight) | `bfloat16` | 1,777,664 | 679,701 | **0.3824** | **0.3781** (110,740) | **0.3826** (1,666,924) | 0.00827 |
| **X8A47** | `routing_proj` (original weight) | `bfloat16` | 48,640 | 11,991 | **0.2465** | **0.1320** (42,171) | **0.9929** (6,469) | 0.00658 |
| **X8A49** | `gamma_param` | `bfloat16` | 1 | 0 | **0.0000** | 0.0000 (1) | — | 0.00000 |
| **X8A49** | `lora_A` (original weight) | `bfloat16` | 2,154,496 | 481,790 | **0.2236** | **0.1196** (1,261,974) | **0.3707** (892,522) | 0.00684 |
| **X8A49** | `lora_B` (weight) | `bfloat16` | 1,777,664 | 761,846 | **0.4286** | **0.4712** (110,740) | **0.4257** (1,666,924) | 0.00748 |
| **X8A49** | `routing_proj` (original weight) | `bfloat16` | 48,640 | 12,114 | **0.2491** | **0.1350** (42,171) | **0.9927** (6,469) | 0.00529 |
| **X8A50** | `gamma_param` | `bfloat16` | 1 | 0 | **0.0000** | 0.0000 (1) | — | 0.00000 |
| **X8A50** | `lora_A` (original weight) | `bfloat16` | 2,154,496 | 436,057 | **0.2024** | **0.1078** (1,261,974) | **0.3361** (892,522) | 0.00623 |
| **X8A50** | `lora_B` (weight) | `bfloat16` | 1,777,664 | 570,285 | **0.3208** | **0.4136** (110,740) | **0.3146** (1,666,924) | 0.00749 |
| **X8A50** | `routing_proj` (original weight) | `bfloat16` | 48,640 | 11,854 | **0.2437** | **0.1282** (42,171) | **0.9969** (6,469) | 0.00435 |

### 2.2 모듈 단위 완전 불변(0% 변경) 및 `_u/_v`·`B@A` 확인
1. **19개 라우팅 모듈 중 완전 불변 모듈 (가중치 텐서 218개 중 0회 변경 텐서 수)**:
   - **X8A47**: 218개 가중치 텐서 중 **125개 완전 불변** → 19개 모듈 중 **11개 모듈(`{1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 16}`)**의 `lora_A` 및 `lora_B`가 1,500 step 후에도 초기 BF16 값과 원소 단위로 100% 동일합니다.
   - **X8A49**: 218개 중 **113개 완전 불변** → 19개 모듈 중 **10개 모듈(`{1, 2, 3, 4, 7, 8, 10, 11, 12, 16}`)** 완전 불변.
   - **X8A50**: 218개 중 **137개 완전 불변** → 19개 모듈 중 **12개 모듈(`{1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 16, 17}`)** 완전 불변.
2. **Spectral norm `_u/_v` 버퍼**:
   - `lora_A.parametrizations.weight.0._u` 및 `._v` (및 `routing_proj`의 `._u`, `._v`)는 `bb.state_dict()`에 포함되어 `bfloat16`으로 함께 저장되었으며, 선택된 모듈에서는 power iteration으로 인해 $\max \|\Delta\| \approx 1.02 \sim 1.07$까지 변했습니다.
3. **Scaling 포함 유효 `(alpha / r) * (B @ A)` 변화**:
   - `cts/train/lora_compat.py` L62, L144에서 `rank = 8`, `alpha = 16`이므로 **`scaling = 2.0`**이며, 유효 어댑터 연산은 `2.0 * (lora_B.weight @ lora_A_effective_weight)`입니다.
   - 위 10~12개 불변 모듈에서는 `lora_A`(original), `lora_B`, `_u/_v` 모두 0회 호출/갱신되었으므로 유효 `2.0 * (B @ A)` 변화도 정확히 **0**입니다. 반면 선택된 7~9개 모듈에서는 가중치 변화($\max \|\Delta\| \le 0.0083$)와 `lora_A`의 spectral norm 버퍼(`_u/_v`) 갱신이 결합되어 유효 `B @ A` 및 2걸음 출력 $z_2$가 변했습니다. (전 모듈의 개별 $\|2(B_1 A_1 - B_0 A_0)\|_F$ 정밀 분포는 로컬 PC에 `.pt` 파일이 없어 **미측정**이며, 서버 `/tmp/cts_x8a/deq_X8A*.pt`에 Astra의 `checkpoint_delta_audit.py --lora-products`를 실행하면 즉시 산출 가능합니다.)

---

## 3. 정답 학습 신호(`grad(L_read)`)와 다른 변화 요인의 분리 확인

1. **기존 기록에 `grad(L_read)` vs `grad(L_fp)` 분리 측정, clip 전후 norm, 실제 update norm 존재 여부**:
   - **자료 없음 (미측정)**.
   - 실제 `g2v11_x8a_train.py` L165–171을 보면:
     ```python
     loss = (l_read + a.lambda_fp * l_fp) / a.batch_size
     loss.backward()
     ...
     torch.nn.utils.clip_grad_norm_(list(br.adapters.parameters()) + deq_params, 1.0)
     opt.step()
     ```
     두 손실을 합산한 뒤 1회 `backward()`만 호출하며, `clip_grad_norm_`의 반환값(총 클리핑 전 norm)도 변수에 저장하거나 로그에 출력하지 않았습니다.
2. **`bb.train()`에서 spectral normalization `_u/_v` 버퍼가 갱신되는지 실제 구현 확인**:
   - **갱신됨 (코드로 확정)**.
   - 근거 1 (`cts/train/lora_compat.py` L56–58): `from torch.nn.utils.parametrizations import spectral_norm; spectral_norm(self.lora_A, name="weight")`로 PyTorch 기본 parametrization을 등록합니다.
   - 근거 2 (`g2v11_x8a_train.py` L140): 학습 루프 진입 직전 **`bb.train()`**을 호출합니다.
   - 근거 3: PyTorch의 `spectral_norm` parametrization은 모듈이 `training=True` 상태일 때 forward에서 `.weight` 프로퍼티에 접근할 때마다(`self.lora_A(adapter_input)`) power iteration을 1회 수행하여 `_u`, `_v` 버퍼를 in-place 갱신합니다. 또한 `routing_proj` 역시 `bb.routing_matrix()` 호출 시 `training=True`에서 `_u/_v`가 갱신됩니다.
   - 따라서 Astra의 지적대로 step 0(`gate=0`)에서 $\nabla_{z_{\text{out}}} \mathcal{L}_{\text{read}} = 0$이더라도, **(a) $\mathcal{L}_{\text{fp}}$의 gradient, (b) AdamW `weight_decay=0.001`, (c) `bb.train()` 상태에서의 spectral norm `_u/_v` 버퍼 갱신**만으로 DEQ 출력과 잔차가 변할 수 있으므로, 잔차 변화나 `n_unique_ids_contributing_grad = 992`만으로 정답 읽기 신호($\mathcal{L}_{\text{read}}$)가 DEQ를 유효하게 학습시켰다고 단정할 수 없습니다.

---

## 4. G1 잔차의 상태 위치 확인 및 `r(z1)` 전수 집계

1. **실제 출력 상태 $z_2$의 정본 잔차 $r(z_2) = \|F_\theta(z_2, x) - z_2\|_F / \|z_2\|_F$ 존재 여부**:
   - **미측정 (기존 자료 없음)**.
   - `g2v11_x8a_train.py` L38–53 및 `g2v11_x8a_latents.py` L34, L45의 `deq_unroll(..., k_unroll=2)`는 $k=0$에서 $\rho(z_0) = \|z_1 - z_0\| / \|z_0\|$, $k=1$에서 $\rho(z_1) = \|z_2 - z_1\| / \|z_1\|$만 기록하고 $z_2$를 반환하며, $F_\theta(z_2, x)$를 계산하지 않았습니다.
2. **기존 `latdev_X8A{47,49,50}.pt.res.json` (개발 600문항, $\alpha=0.25$)에서 $r(z_1) = \text{res}[1] / \alpha = 4 \times \text{res}[1]$ 및 $r(z_0) = 4 \times \text{res}[0]$ 집계**:

| 시드 (`latdev`) | 문항 수 | 지표 | 최솟값 (`min`) | 중앙값 (`p50`) | 평균 (`mean`) | 상위 10% (`p90`) | 최댓값 (`max`) | `< 0.01` 충족 문항 수 |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **2047** (`X8A47`) | 600 | $\rho(z_1) = \text{res}[1]$ | 0.029040 | 0.035525 | 0.035484 | 0.037496 | 0.040403 | **0 / 600** |
| **2047** (`X8A47`) | 600 | **$r(z_1) = 4 \cdot \text{res}[1]$** | **0.116162** | **0.142102** | **0.141936** | **0.149983** | **0.161610** | **0 / 600** |
| **2047** (`X8A47`) | 600 | $r(z_0) = 4 \cdot \text{res}[0]$ | 0.135795 | 0.163076 | 0.162943 | 0.171996 | 0.186888 | **0 / 600** |
| **2049** (`X8A49`) | 600 | $\rho(z_1) = \text{res}[1]$ | 0.026802 | 0.032455 | 0.032318 | 0.033932 | 0.036139 | **0 / 600** |
| **2049** (`X8A49`) | 600 | **$r(z_1) = 4 \cdot \text{res}[1]$** | **0.107209** | **0.129821** | **0.129270** | **0.135727** | **0.144554** | **0 / 600** |
| **2049** (`X8A49`) | 600 | $r(z_0) = 4 \cdot \text{res}[0]$ | 0.124014 | 0.148645 | 0.148099 | 0.154979 | 0.162994 | **0 / 600** |
| **2050** (`X8A50`) | 600 | $\rho(z_1) = \text{res}[1]$ | 0.028496 | 0.035825 | 0.035693 | 0.037392 | 0.039469 | **0 / 600** |
| **2050** (`X8A50`) | 600 | **$r(z_1) = 4 \cdot \text{res}[1]$** | **0.113986** | **0.143301** | **0.142770** | **0.149569** | **0.157875** | **0 / 600** |
| **2050** (`X8A50`) | 600 | $r(z_0) = 4 \cdot \text{res}[0]$ | 0.131478 | 0.163428 | 0.162902 | 0.170058 | 0.177701 | **0 / 600** |

- **이전 M1의 G1 결과와 새 가중치(X8-A)의 구분**:
  - 세 시드 모두 $r(z_1) < 0.01$을 만족하는 문항은 **0 / 600개**이며, 3시드 전체 1,800개 평가 중 $r(z_1)$의 최솟값조차 **`0.107209`**(임계값 `0.01`의 10.7배)입니다.
  - 이전 M1(`m1_final_frozen.pt`)에서 달성했던 G1 통과(수렴성·다중 초기값 유일성)는 동결된 M1 연산자 $\Phi_{\theta_{\text{M1}}}$에 대한 결과일 뿐이며, X8-A의 변경된 연산자 $\Phi_{\theta_{\text{X8A}}}$와 2걸음 전개 출력 $z_2$에는 상속되지 않습니다.

---

## 5. 학습 타깃 교차 집계 존재 여부 및 개발 600문항(`C_fmt`, `X8A`, `X7E`) 서버/로컬 채점표

1. **`replaced/appended × 원래 정답 여부 × finished/truncated` 교차 집계 및 본문/답 토큰·CE 총합 존재 여부**:
   - **기존 자료 없음 (미측정)**.
   - 기존 `manifest_X8A{47,49,50}.json`의 `target_counts`에는 1차원 주변합(`replaced: 341`, `appended: 651`, `skip_no_gold: 1`, `budget_truncated: 41`, `finished_own: 943`, 총 유효 `992`)만 기록되어 있고, `RESULT-X5-stylematched-20261020.md`에는 학습 993문항 전체의 기준선 정답 수(`lenient: 159/993`, `strict: 134/993`, `boxed: 351/993`)만 기록되어 있습니다.
   - 학습 타깃 원문 파일(`/tmp/cts_x5base/gen_Ctrain.jsonl` 또는 `results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl`)은 서버에만 있고 회사 PC 로컬(`D:\AI\cts`)에는 내려받아져 있지 않아 로컬에서 새 교차 집계를 내지 않았습니다. Astra의 지적대로 교집합 자료 없이 `341 - 134 = 207`로 모순 표본 수를 추산하거나 가중치 질량 예시(`7.3%` vs `92.7%`)를 실제 loss/gradient 기여도로 취급해서는 안 됩니다.
2. **개발 600문항 (`C_fmt`, `X8A matched/shuffled`, `X7E matched`) 서버 vs 로컬 채점 분리 CSV/JSON**:
   - 요청하신 10개 조건(`C_fmt` 1개 + `X8A_matched` 3시드 + `X8A_shuffled` 3시드 + `X7E_matched` 3시드 = 총 6,000행)에 대해 서버 판정(`x8a_items.json`, `x7_items.json`)과 로컬 판정(`x8a_items_local.json`, `x7_items_local.json`)을 나란히 기록한 파일을 생성하여 폴더 및 ZIP 안에 포함했습니다:
     - CSV: [`dev600_cfmt_x8a_x7e_server_vs_local.csv`](dev600_cfmt_x8a_x7e_server_vs_local.csv) (SHA256 `bd890d4bc81fd957980af47f1890b873c01a86e35cf92e6cdef71966dbf33954`)
     - JSON: [`dev600_cfmt_x8a_x7e_server_vs_local.json`](dev600_cfmt_x8a_x7e_server_vs_local.json) (SHA256 `5e832168415fbe039580222a69a141a0dbd8822ba9da0cab4deb5b6bdc0a637f`)
   - **서버 판정 vs 로컬 판정 비교 요약 (6,000행 전수 대조)**:
     - `C_fmt`, `X8A_matched`(3시드), `X8A_shuffled`(3시드), `X7E_matched`(2047, 2049): 서버와 로컬이 `strict`, `lenient`, `boxed_present` **전 문항 100% 일치 (불일치 0건)**.
     - `X7E_matched` 시드 `2050`: 600문항 중 **단 1문항(`item_id = 3982`)**에서 `lenient` 판정이 서버 `1` vs 로컬 `0` (`math-verify 0.5.2` vs `0.9.0` 파서 차이; `REPORT-C006.md` §3에 보고됨)이며, 나머지 5,999행은 100% 일치합니다.
3. **`C_fmt`와 `g2v9_gen.py --cond none` 경로의 동일성 근거**:
   - **Mock selftest 및 코드 정적 구조에 기반한 것이며, 개발 600문항 실제 생성물 전수 대조는 미수행**입니다.
   - 구체적으로: (a) `g2v9_xattn.py --selftest`는 `MockLM(D=32, K=8)`에서 `gate=0` 및 `set_latent(None)`일 때 `torch.equal(ref, lm(x))`를 확인하는 CPU mock test이고, (b) `g2v9_gen.py` L50–51에서 `if a.cond != "none": br.attach(lm)`으로 훅을 부착하지 않으나, (c) 개발 600문항의 `C_fmt` 기준선(`results_s7/p6r/g2v8s1/gen_C.jsonl`)은 S1(`v36g`)에서 생성된 파일을 재사용한 것이며 `g2v9_gen.py --cond none`으로 개발 600문항을 다시 생성해 `v36g` 출력과 bit-by-bit 대조한 실험은 수행되지 않았습니다 (`g2v9_gen.py --cond none`은 `cc_x5base.sh`에서 학습 993문항 `gen_Ctrain.jsonl` 생성에만 실행됨).

---

## 6. 다음 수정 방향(4단계)의 구현 제약 검토 및 3대 해석 정정 수용

### 6.1 제안하신 4단계 수정 순서에 대한 코드 수준 구현 제약 검토

업로드된 실제 코드(`cts/train/lora_compat.py`, `cts/backbone/gemma_adapter.py`, `g2v9_xattn.py`, `g2v11_x8a_train.py`)를 기준으로 검토한 결과, 제안하신 4단계를 가로막는 구조적 제약은 없으며 아래 구현 세부사항만 준수하면 즉시 동작합니다:

1. **학습 parameter와 업데이트 누적을 FP32로 보장 (P1)**:
   - **호환성 확인**: `cts/train/lora_compat.py` L67–70의 `LoraLinear.forward`는 이미 `adapter_input = self.lora_dropout(x.to(self.lora_A.weight.dtype))`로 입력을 `lora_A.weight.dtype`에 맞춰 캐스팅하고 출력을 `b.to(out.dtype)`으로 되돌립니다. 또한 `g2v11_x8a_train.py` L129의 `bb.routing_matrix().to(device=dev, dtype=torch.float32)`와 `cts/deq/gemma_latent_forward.py`의 `torch.sigmoid(gamma_param).to(out.dtype)` 역시 FP32 파라미터와 충돌하지 않습니다.
   - **주의할 구현 제약 2가지**:
     - **(a) Spectral norm 버퍼 dtype 일치**: `lora_A`와 `routing_proj`에는 `torch.nn.utils.parametrizations.spectral_norm`이 걸려 있어 `.parametrizations.weight.original`(파라미터)과 `.parametrizations.weight.0._u`, `._v`(버퍼)가 내적(`torch.mv`)을 수행합니다. 따라서 `TRAINABLE_KEYS` 파라미터를 FP32로 유지할 때 **반드시 해당 모듈의 `_u`, `_v` 버퍼도 함께 `torch.float32`로 유지**해야 `RuntimeError: expected scalar type Float but found BFloat16`이 발생하지 않습니다.
     - **(b) 평가 스크립트(`g2v11_x8a_latents.py`) 일관성**: `g2v11_x8a_latents.py` L37–40에서도 `bb.load_state_dict(deq_state, strict=False)` 직후 `bb.to(device=dev, dtype=torch.bfloat16)`을 호출하고 있으므로, 학습에서 FP32로 저장한 `deq_state`가 평가 시 다시 BF16으로 반올림되지 않도록 평가 스크립트에서도 동일하게 `TRAINABLE_KEYS` 파라미터와 `_u/_v` 버퍼를 FP32로 유지해야 합니다.
2. **본문 loss 제거(`T1: w_pre=0`)와 풀이 입력 제거(`T2: 짧은 정답 타깃`) 별도 비교**:
   - **기존 수행 여부**: **미수행** (기존 X5~X8-A는 모두 `w_pre=1.0, w_box=5.0, w_post=1.0`인 `T0`만 수행함).
   - **구현 제약**: `g2v11_x8a_train.py` L105에서 `ids_pre` 구간의 가중치를 `0.0`으로 두면 `T1`이 되고, `ids_pre`를 고정 답 지시문(` CONNECTOR = "\nTherefore, the final answer is "`)으로 대체하면 `T2`가 됩니다. 단, `T2`는 출력 형식이 짧은 답으로 바뀌므로 평가 시에도 동일 프롬프트·생성 예산의 순정 기준선을 함께 비교해야 합니다.
3. **실제 출력 상태 $z_{\text{out}} = z_K$에서의 정본 잔차 $r(z_{\text{out}}) = \|F_\theta(z_{\text{out}}, x) - z_{\text{out}}\|_F / \|z_{\text{out}}\|_F$ 측정 (P2)**:
   - **구현 제약**: `deq_unroll` 루프 종료 직후 `z_out`에서 1회 추가 forward(`phi_out = bb.deq_step(z_out, ...)`)를 평가하면 `r_map_out = (phi_out + g0 * z_src - z_out).norm() / z_out.norm().clamp_min(1e-6)`을 정확히 얻을 수 있습니다.
   - 단, 학습 중 `bb.train()`에서 추가 `deq_step`을 호출하면 선택된 top-3 모듈의 `lora_A` spectral norm `_u/_v` 버퍼가 1회 더 power iteration될 수 있으므로, 순수 진단 목적일 때는 `with torch.no_grad():` 및 버퍼 보존(또는 `bb.eval()` 전환 후 복구) 여부를 명시적으로 통제해야 합니다.
4. **잠재 전용 answer reader 교정·동결(Stage R) 후 DEQ만 정답 loss로 학습(Stage Z) (P4)**:
   - **기존 수행 여부**: **미수행**.
   - **구현 제약**: `LatentXAttnBridge`의 cross-attention은 쿼리 `Q_l(h)`의 시퀀스 길이 $T$에 제약을 두지 않으므로, 문제 본문 없이 고정 답 지시문 토큰만 입력하는 `T3` 설정에서도 훅(`lm_model.layers[l]`)이 그대로 작동합니다. Stage Z에서는 `br.eval()` 및 `for p in br.parameters(): p.requires_grad = False`로 판독기와 게이트를 동결하되, `br.set_latent(z_out)`에 들어가는 `z_out`의 `requires_grad=True`만 유지하면 `XAttnAdapter.forward`를 통해 $\nabla_{z_{\text{out}}} \mathcal{L}_{\text{task}}$가 DEQ의 `deq_params`로 정상 역전파됩니다.

### 6.2 세 가지 이론·해석 정정 사항 확인
지적해 주신 아래 세 가지 해석 정정을 공식 기록으로 수용하며, 이후 보고서·안건 문서에서 오용하지 않습니다:
1. 결정론적 매핑 $Z = f(X)$에서 조건부 상호정보량 $I(Y; Z(X) \mid X) = 0$은 항등적으로 성립하는 정보이론적 성질일 뿐이므로, 문제 텍스트만으로 잠재를 계산한다는 사실 자체를 추론 불가능의 근거로 삼지 않습니다(핵심은 제한된 디코더가 읽기 쉬운 표현으로 변환하는 계산적 유용성/V-information임).
2. 비선형·상태 의존 연산자 $F_\theta(z)$ (및 희소 top-k 라우팅 경계)에서는 고정 목표 수축률 $1 - (1-\alpha)^2 = 43.75\%$가 일반적으로 성립하지 않으며, 2걸음의 $\rho$ 감소만으로 이후 고정점 수렴을 보장할 수 없습니다.
3. 학습 NLL 감소와 개발 자유 생성 정확도 간극만으로 "문항 ID 암기(lookup-key memorization)"를 100% 확정하거나 BF16 정밀도/기울기 병목을 기각할 수 없으며, 실제로 이번 체크포인트 감사에서 BF16 저장 + 희소 라우팅으로 인한 DEQ 부분 미학습(10~12/19 모듈 불변)이 실측으로 확인되었습니다.

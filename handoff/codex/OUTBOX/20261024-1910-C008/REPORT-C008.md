# REPORT-C008 — X8-A (읽기 손실이 DEQ를 직접 미는 결합 학습) 구현 정적 검토

- **과제**: [TASK-C008.md](file:///d:/AI/cts/handoff/codex/INBOX/TASK-C008.md) (`55f5ecd410a4a083b217fffba27ed3a1ca222bb8ba164b38d1d8494dcca32f8c`)
- **검토 대상 스크립트**:
  - [g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py) (`208eb131f97988c82746b7d480ea1b214fb242f895690fdde4c5a1bd8333c568`)
  - [g2v11_x8a_latents.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_latents.py) (`9a82c78179c686eaae2b739c812c36f2d146b94fdd24334edceb544028166b0c`)
  - [cc_x8a.sh](file:///d:/AI/cts/handoff/codex/scripts/cc_x8a.sh) (`4a5e457850891107359e55ff375168be9f7c22886068ad0bb2d90aac64e7adca`)
- **참조 스크립트 및 채점기**:
  - [g2v10_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v10_train.py) (`48b71ffb94263284f611a35ce4a58bc9fb7609d9b74c18fe31cc95c8c3afe48f`)
  - [g2v9_xattn.py](file:///d:/AI/cts/handoff/codex/scripts/g2v9_xattn.py) (`ef3fbd5ed5d227c5b765fc01b95bcd920d4433b005e32e3057dcb479f2d1457d`)
  - [g2v9_gen.py](file:///d:/AI/cts/handoff/codex/scripts/g2v9_gen.py) (`14e466006b9a29c4d30181c424f5e6fb9872c781d03c3313c71fc433f7c2f3b3`)
  - [g2v3_m3_full_train.py](file:///d:/AI/cts/scripts/g2v3_m3_full_train.py) (`36e0f18addc8d46f6d8bf376cd95e89cf9eb75a1f15c4e8b714141efe5a49c99`)
  - [transition_inject.py](file:///d:/AI/cts/cts/deq/transition_inject.py) (`127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c`)
  - [gemma_adapter.py](file:///d:/AI/cts/cts/backbone/gemma_adapter.py) (`0a9a70c68b80de7fac75b6c129cb57985861a29d9eb962f3758200572e1ae903`)
  - [gemma_latent_forward.py](file:///d:/AI/cts/cts/deq/gemma_latent_forward.py) (`f30de30846e5982f0a19356d9a0a09ce27c51bafad769ab93440dabc16ca2e90`)
  - [s5_eval_grader_v2.py](file:///d:/AI/cts/scripts/s5_eval_grader_v2.py) (`90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`)
- **작성일**: 2026-10-24

---

## 0. 한 줄 요약

`X8-A` 구현 3종([g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py), [g2v11_x8a_latents.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_latents.py), [cc_x8a.sh](file:///d:/AI/cts/handoff/codex/scripts/cc_x8a.sh))의 8개 검토 항목을 정적 분석하고 실행 산출물(`INBOX/20261023-1950-x8a-results/`)과 교차 대조한 결과, M2' 1걸음 식 일치성·기울기 흐름·`L_fp` 정의·평가 잠재 생성 일관성·VRAM/실행 안정성에서 실험 결과를 무효화하는 결함은 없었으며, 경미한 코드/메타데이터 결함 3건(`ad.gate` 로깅 시 `.detach()` 누락으로 인한 `UserWarning`, `bridge_X8A*.pt` 내 `"zfile": a.zstar` 기록, `rng` 미사용으로 인한 시드 간 동일 미니배치 순서)과 구조적 관찰 사항 4건을 식별하여 파일·줄 번호·재현 조건과 함께 보고한다.

---

## 1. 수행한 것

1. **스크립트 SHA256 및 서버 실행본 일치 확인**:
   - 로컬 저장소 `handoff/codex/scripts/`의 3개 스크립트 SHA256을 계산하고 [cts_x8a_master.log](file:///d:/AI/cts/handoff/codex/INBOX/20261023-1950-x8a-results/cts_x8a_master.log#L1-L6)의 서버 실행 해시와 대조하였다.
   - `TASK-C008.md` 본문(3행)에는 `g2v11_x8a_train.py` 해시 접두사가 `5721ffc5…`로 기재되어 있으나, 실제 저장소 파일 및 서버 실행본([cts_x8a_master.log](file:///d:/AI/cts/handoff/codex/INBOX/20261023-1950-x8a-results/cts_x8a_master.log#L4))은 모두 `208eb131f97988c82746b7d480ea1b214fb242f895690fdde4c5a1bd8333c568`로 정확히 일치함을 확인하였다.
2. **검토 항목 1~8 전수 정적 분석**:
   - **항목 1 (한 걸음 식의 일치)**: [g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L38-L53)의 `deq_unroll`을 [g2v3_m3_full_train.py](file:///d:/AI/cts/scripts/g2v3_m3_full_train.py#L516-L532) 및 [transition_inject.py](file:///d:/AI/cts/cts/deq/transition_inject.py#L109-L148)와 줄 단위로 비교하였다.
   - **항목 2 (기울기 흐름)**: `z0`, `zs` 상수 취급 여부, `TRAINABLE_KEYS` 파라미터 수(218 텐서, 3,980,801 가중치), top-k 이산 선택과 가중합 경로([gemma_latent_forward.py](file:///d:/AI/cts/cts/deq/gemma_latent_forward.py#L932-L1026)), `fp32` 잠재 텐서와 `bf16` 백본 사이의 캐스팅([gemma_latent_forward.py](file:///d:/AI/cts/cts/deq/gemma_latent_forward.py#L519-L527), [gemma_adapter.py](file:///d:/AI/cts/cts/backbone/gemma_adapter.py#L344-L347)), `bb.train()` 모드에서의 `spectral_norm` 및 KV 캐시 동작을 검증하였다.
   - **항목 3 (`L_fp` 정의)**: [g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L154)의 분모 `z_prev.detach()` 필요성과 분자 양방향 기울기 특성을 [g2v3_m3_full_train.py](file:///d:/AI/cts/scripts/g2v3_m3_full_train.py#L536)와 비교하였다.
   - **항목 4 (평가 잠재 일관성)**: [g2v11_x8a_latents.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_latents.py#L31-L55)의 `deq_X8A*.pt` 로드 방식(`strict=False` + assertions + spectral_norm 버퍼 `_u`, `_v` 포함 여부) 및 `br.center` 버퍼 일관성을 확인하였다.
   - **항목 5 (`E` 대조 재사용 타당성)**: `X7E`와 `X8-A`의 입력 파일·초매개변수·타깃 구성을 대조하고, `z_src` 고정 학습 대조군으로서의 성립 범위와 한계를 정리하였다.
   - **항목 6 (메모리·실행 시간)**: 2개 Gemma-4-E4B 인스턴스 + `k_unroll=2` 그래프 + 매 스텝 `empty_cache()` 호출의 메모리·속도 영향을 분석하고 실제 학습 로그(`1860s~1886s`)와 대조하였다.
   - **항목 7 (`cc_x8a.sh`) & 항목 8 (기타 결함)**: 런처의 GPU 가드·사후 검증 게이트 및 `g2v10_train.py` 대비 차이점을 전수 점검하였다.
3. **정적 검증 및 런타임 증거 추출 스크립트 실행**:
   - [verify_c008_v2.py](file:///d:/AI/cts/handoff/codex/OUTBOX/20261024-1910-C008/verify_c008_v2.py)를 작성·실행하여 [c008_static_verification.json](file:///d:/AI/cts/handoff/codex/OUTBOX/20261024-1910-C008/c008_static_verification.json) 및 [INPUT_MANIFEST.json](file:///d:/AI/cts/handoff/codex/OUTBOX/20261024-1910-C008/INPUT_MANIFEST.json)을 생성하였다.

---

## 2. 결과 표(숫자만)

### 2.1 검토 항목별 정적 분석 요약표

| 항목 번호 | 검토 주제 | 대상 파일 및 줄 번호 | 대조 기준 | 정적 검토 분류 | 핵심 수치 / 확인 결과 |
|---:|---|---|---|---|---|
| 1 | 한 걸음 식(`deq_unroll`) 일치 | `g2v11_x8a_train.py:38-53` | `g2v3_m3_full_train.py:516-532` | 결함 아님 | `top_k=3`, `alpha=0.25`, `g0=0.25`, `z_src` 주입 식 100% 일치 |
| 2 | 기울기 흐름·dtype·`bb.train()` | `g2v11_x8a_train.py:119-171` | `gemma_adapter.py:255-347`, `gemma_latent_forward.py:519-1026` | 결함 아님 (관찰 2건) | DEQ 학습 텐서 `218`개 (`3,980,801` 파라미터), 브리지 `20,971,524` 파라미터 |
| 3 | 고정점 벌점 `L_fp` 정의 | `g2v11_x8a_train.py:154` | `g2v3_m3_full_train.py:536` | 결함 아님 | `lambda_fp=0.1`, 분모 `z_prev.detach()`로 $\|z_1\|^2$ 팽창 해킹 차단 |
| 4 | 평가 잠재(`g2v11_x8a_latents.py`) 일관성 | `g2v11_x8a_latents.py:31-55` | `g2v11_x8a_train.py:179-180`, `g2v9_gen.py:45-56` | 결함 아님 (관찰 1건) | `deq_state` 내 `218` 파라미터 + `routing_proj` `_u`/`_v` 버퍼 정확 복원, dev `600/600` 생성 |
| 5 | `E = X7E` 대조 재사용 타당성 | `cc_x8a.sh:74-86` | `X7E` (`manifest_X7E*.json`) | 결함 아님 (의견 기재) | `z_src`에는 `k_unroll` 대상 고정점 이동이 없으므로 브리지 단독 학습(`X7E`)과 동치 |
| 6 | 메모리·실행 시간·`empty_cache()` | `g2v11_x8a_train.py:115-177` | `train_X8A{47,49,50}.log` | 결함 아님 | `1,500` step 소요 시간 `1860s` / `1886s` / `1863s` (`~1.24s/step`), OOM `0`건 |
| 7 | 런처 `cc_x8a.sh` 가드 및 중단 조건 | `cc_x8a.sh:1-92` | `cts_x8a_master.log:1-67` | 결함 아님 (관찰 1건) | `set -euo pipefail`, `SELFTEST-OK`, GPU 4 미사용(`U5,U6,U7`), `rows=600` 검증 |
| 8 | 기타 코드 결함 (`g2v10` 대비 차이 등) | `g2v11_x8a_train.py:68,146,175,179` | `g2v10_train.py:90,169,208,212` | 경미 결함 3건 | `ad.gate` `.detach()` 누락 경고, `bridge` 내 `"zfile": a.zstar` 기록, `rng` 미사용 |

### 2.2 실행 매니페스트 및 잔차 텔레메트리 교차 검증 수치 (`INBOX/20261023-1950-x8a-results/`)

| 시드 태그 | `seed` | 학습 문항 수 (`n_items_trained`) | 업데이트 수 (`n_updates`) | 스킵 수 (`n_samples_skipped`) | Step 0 `loss` / `ans_ce` | Step 0 `l_fp` / `res_mean` | 마지막 200스텝 평균 `loss` / `ans_ce` | 마지막 200스텝 평균 `l_fp` / `res_mean` | Dev 600 1걸음 잔차 평균 (`res_step1_mean`) | Dev 600 2걸음 잔차 평균 (`res_last_mean`) | Dev 600 2걸음 잔차 최댓값 (`res_last_max`) | Dev 600 수축 비율 (`res[1] < res[0]`) |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| `X8A47` | 2047 | 992 | 1500 | 0 | 0.852929 / 6.933695 | 1.6986e-07 / 0.000439 | 0.058347 / 0.291897 | 0.001553 / 0.042057 | 0.040736 | 0.035484 | 0.040403 | 600 / 600 |
| `X8A49` | 2049 | 992 | 1500 | 0 | 0.852929 / 6.933695 | 1.6986e-07 / 0.000439 | 0.061653 / 0.312520 | 0.001267 / 0.038020 | 0.037025 | 0.032318 | 0.036139 | 600 / 600 |
| `X8A50` | 2050 | 992 | 1500 | 0 | 0.852929 / 6.933695 | 1.6986e-07 / 0.000439 | 0.060099 / 0.311632 | 0.001300 / 0.038474 | 0.040725 | 0.035693 | 0.039469 | 600 / 600 |

### 2.3 원본 소스 코드 및 실행 로그 핵심 원문 행

- **[cts_x8a_master.log](file:///d:/AI/cts/handoff/codex/INBOX/20261023-1950-x8a-results/cts_x8a_master.log#L1-L6)** (서버 실행 스크립트 SHA256 기록):
```text
=== X8-A full start 2026-10-23 11:54:33 launcher_sha 4a5e457850891107359e55ff375168be9f7c22886068ad0bb2d90aac64e7adca ===
ef3fbd5ed5d227c5b765fc01b95bcd920d4433b005e32e3057dcb479f2d1457d  /tmp/cts_x_stage/g2v9_xattn.py
48b71ffb94263284f611a35ce4a58bc9fb7609d9b74c18fe31cc95c8c3afe48f  /tmp/cts_x_stage/g2v10_train.py
208eb131f97988c82746b7d480ea1b214fb242f895690fdde4c5a1bd8333c568  /tmp/cts_x_stage/g2v11_x8a_train.py
9a82c78179c686eaae2b739c812c36f2d146b94fdd24334edceb544028166b0c  /tmp/cts_x_stage/g2v11_x8a_latents.py
14e466006b9a29c4d30181c424f5e6fb9872c781d03c3313c71fc433f7c2f3b3  /tmp/cts_x_stage/g2v9_gen.py
```

- **[g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L38-L53)** (`deq_unroll` 원문):
```python
def deq_unroll(bb, z0, z_src, ctx, w_g, k_unroll, alpha, g0, top_k=3):
    """K differentiable M2'-form steps from z0. Returns (z_out, z_prev, residuals list)."""
    z_cur = z0; z_prev = z0; res = []
    for _ in range(k_unroll):
        f_weights = sparse_module_weights(routing_weights(z_cur.float(), w_g, 1.0), top_k)
        sel_ids = torch.sort(torch.topk(f_weights.reshape(-1), top_k).indices.to(dtype=torch.long)).values
        r_ids = [int(x) for x in sel_ids.detach().cpu().tolist()]
        ex = {"top_k": top_k, "deq_map_mode": bb.deq_map_mode, "routing_module_ids": r_ids, "routing_module_ids_tensor": sel_ids}
        hook = getattr(bb, "prepare_deq_context", None)
        if callable(hook):
            ex["eq3_static_inputs"] = hook(z_cur, ctx, r_ids)
        phi = bb.deq_step(z_cur, ctx, f_weights, ex).float()
        z_next = (1.0 - alpha) * z_cur + alpha * (phi + g0 * z_src)
        res.append(float(((z_next - z_cur).norm() / z_cur.norm().clamp_min(1e-6)).detach()))
        z_prev, z_cur = z_cur, z_next
    return z_cur, z_prev, res
```

- **[train_X8A47.log](file:///d:/AI/cts/handoff/codex/INBOX/20261023-1950-x8a-results/train_X8A47.log#L8-L14)** 및 **[train_X8A47.log](file:///d:/AI/cts/handoff/codex/INBOX/20261023-1950-x8a-results/train_X8A47.log#L74-L75)**:
```text
[DEQ] trainable tensors 218 (3,980,801 weights), map_mode parallel
[CACHE] contexts 992
[BRIDGE] layers [10, 20, 30, 40], params 20,971,524, center norm/slot 312.5
/tmp/cts_x_stage/g2v11_x8a_train.py:175: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
  gates = [round(float(torch.tanh(ad.gate)), 3) for ad in br.adapters.values()]
[Step    0] loss 0.8529 ans_ce 6.9337 l_fp 0.00000 res 0.0004 gates [0.0, -0.0, -0.0, -0.0] (2s)
...
[Step 1499] loss 0.0652 ans_ce 0.2333 l_fp 0.00125 res 0.0379 gates [0.002, -0.016, -0.043, -0.037] (1860s)
[SAVE] /tmp/cts_x8a/bridge_X8A47.pt /tmp/cts_x8a/deq_X8A47.pt updates 1500 skipped 0
```

---

## 3. 불일치·결함 목록(파일, 줄, 재현 조건)

### 3.1 문서-코드 간 SHA256 표기 불일치 (1건, 실행 무관)

1. **`TASK-C008.md` 내 `g2v11_x8a_train.py` SHA256 접두사 표기 불일치**:
   - **위치**: [TASK-C008.md](file:///d:/AI/cts/handoff/codex/INBOX/TASK-C008.md#L3) vs [g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L1) 및 [cts_x8a_master.log](file:///d:/AI/cts/handoff/codex/INBOX/20261023-1950-x8a-results/cts_x8a_master.log#L4).
   - **내용**: `TASK-C008.md` 3행에는 `g2v11_x8a_train.py`(`5721ffc5…`)로 기재되어 있으나, 실제 저장소의 `handoff/codex/scripts/g2v11_x8a_train.py`와 서버에서 실행된 `/tmp/cts_x_stage/g2v11_x8a_train.py`의 SHA256은 모두 `208eb131f97988c82746b7d480ea1b214fb242f895690fdde4c5a1bd8333c568`이다. (`g2v11_x8a_latents.py`의 `9a82c781…` 및 `cc_x8a.sh`의 `4a5e4578…`는 `TASK-C008.md`와 정확히 일치함.)

### 3.2 경미한 코드/메타데이터 결함 (3건, 결과 수치 무효화 아님)

1. **결함 C008-D1 (`ad.gate` 스칼라 변환 시 `.detach()` 누락으로 인한 PyTorch `UserWarning`)**:
   - **위치**: [g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L175) (동일 패턴이 [g2v10_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v10_train.py#L208)에도 존재).
   - **재현 조건**: `step % 25 == 0`에서 `gates = [round(float(torch.tanh(ad.gate)), 3) for ad in br.adapters.values()]` 실행 시 `ad.gate`가 `requires_grad=True`여서 `torch.tanh(ad.gate)` 역시 ` grad_fn=<TanhBackward0>`를 가진다. 이를 `float(...)`로 직접 변환하면서 [train_X8A47.log](file:///d:/AI/cts/handoff/codex/INBOX/20261023-1950-x8a-results/train_X8A47.log#L11-L13)에 `UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior`가 발생한다.
   - **영향 및 수정 제안(diff)**: 그래프나 기울기 값 자체는 손상되지 않는다. `float(torch.tanh(ad.gate.detach()))`로 변경하면 경고가 제거된다.
   ```diff
   -            gates = [round(float(torch.tanh(ad.gate)), 3) for ad in br.adapters.values()]
   +            gates = [round(float(torch.tanh(ad.gate.detach())), 3) for ad in br.adapters.values()]
   ```

2. **결함 C008-D2 (`bridge_X8A*.pt` 체크포인트 메타데이터에 `"zfile": a.zstar` 기록)**:
   - **위치**: [g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L179).
   - **재현 조건**: `torch.save({"bridge_state": br.state_dict(), ..., "zfile": a.zstar, ...}, a.out_bridge)`에서 브리지가 실제로 학습에 입력받은 잠재는 고정된 `a.zstar`가 아니라 `deq_unroll`을 거친 `z_out`임에도 `"zfile": a.zstar`로 저장된다.
   - **영향**: [g2v9_gen.py](file:///d:/AI/cts/handoff/codex/scripts/g2v9_gen.py#L45-L50)는 체크포인트 내부의 `"zfile"` 메타데이터를 읽지 않고 CLI 인자 `--zfile $W/latdev_$T.pt`([cc_x8a.sh](file:///d:/AI/cts/handoff/codex/scripts/cc_x8a.sh#L56-L65))를 직접 사용하므로 생성 및 평가 결과에는 영향이 없다.

3. **결함 C008-D3 (`rng = np.random.default_rng(a.seed)` 선언 후 미사용으로 시드 간 미니배치 방문 순서 동일)**:
   - **위치**: [g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L68), [g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L146-L147).
   - **재현 조건**: 68행에서 `rng = np.random.default_rng(a.seed)`를 생성하지만 `items` 리스트를 셔플하지 않고 147행에서 `q = items[ptr % len(items)]; ptr += 1`로 순차 순회한다. (`g2v10_train.py`에서는 `rng`가 `text_drop`·`latent_drop`에 쓰였으나 `g2v11_x8a_train.py`에는 드롭아웃이 없어 `rng`가 전혀 호출되지 않는다.)
   - **영향**: 세 시드(`2047`, `2049`, `2050`) 모두 매 스텝 정확히 동일한 4개 문항 묶음(`items[ptr % 992]`)을 방문한다. 이로 인해 `step 0`에서 세 시드의 `loss`(`0.852929`), `ans_ce`(`6.933695`), `l_fp`(`1.6986e-07`), `res_mean`(`0.000439`)가 소수점 끝자리까지 동일하다. 다만 68행의 `torch.manual_seed(a.seed)`에 의해 137행 `LatentXAttnBridge`의 `q_proj, k_proj, v_proj, o_proj`(`std=1e-3`) 초기 가중치가 시드마다 다르게 초기화되므로, `step 0` 역전파 시 `ad.gate`로 흐르는 기울기가 시드마다 달라져 `step 1` 이후부터는 3개 시드의 궤적과 최종 성능이 서로 독립적으로 분기한다.

### 3.3 항목별 정적 분석 상세 결과 (검토 항목 1~7)

1. **항목 1 — 한 걸음 식(`deq_unroll`)의 일치 ([g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L38-L53))**: **결함 아님**.
   - `routing_weights(z_cur.float(), w_g, 1.0)` → `sparse_module_weights(..., top_k=3)` → `sel_ids` 정렬 → `bb.prepare_deq_context(z_cur, ctx, r_ids)` → `phi = bb.deq_step(z_cur, ctx, f_weights, ex).float()` → `z_next = (1.0 - alpha) * z_cur + alpha * (phi + g0 * z_src)` 구조가 [g2v3_m3_full_train.py](file:///d:/AI/cts/scripts/g2v3_m3_full_train.py#L516-L532)와 완전히 일치한다.
   - 참고로 [transition_inject.py](file:///d:/AI/cts/cts/deq/transition_inject.py#L109-L148)(`InjectedTransitionStep`)는 기본 주입 항이 `g0 * c_hat_0`인 반면, `zstar_zsrcinj_train1000.pt`와 `g2v3_m3_full_train.py`는 `g0 * z_src`를 주입하는 M2'(`zsrcinj`) 변형이므로 `deq_unroll`은 초기점 `z*'`를 만든 연산자와 정확히 짝을 이룬다(단, 초기점 `z*'`가 생성될 당시의 고정 백본에서 시작하므로 `step 0`의 상대 잔차가 `0.000439`로 0에 매우 가깝게 출발함).

2. **항목 2 — 기울기 흐름·dtype·캐시 ([g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L106-L171))**: **결함 아님 (관찰 2건)**.
   - **상수 입력 분리**: `Z0[q]`와 `ZS[q]`는 디스크 체크포인트에서 로드된 텐서(`requires_grad=False`)로, 매 스텝 `.to(dev)`로 복사되어 상수로만 작동한다.
   - **기울기 경로**: `loss.backward()` 호출 시 `l_read` → ` gem`(동결, `requires_grad=False`이나 입력 임베딩 이후 레이어 출력이 `br.adapters`의 출력과 더해지므로 활성화 텐서는 `requires_grad=True` 유지) → `br.adapters` (`q/k/v/o_proj`, `gate`, `norm_x`, `norm_z`) → `z_out` (`z_2`) → 2번째 `deq_step` → `z_1` → 1번째 `deq_step`으로 역전파된다.
   - **라우팅 미분 가능성**: `torch.topk`에 의한 인덱스 선택(`sel_ids`) 자체는 조각별 상수(piecewise constant)이지만, [gemma_latent_forward.py](file:///d:/AI/cts/cts/deq/gemma_latent_forward.py#L932-L933) 및 [gemma_latent_forward.py](file:///d:/AI/cts/cts/deq/gemma_latent_forward.py#L1021-L1026)에서 `chosen = module_alpha.index_select(1, support); chosen = chosen / chosen.sum(dim=1, keepdim=True).clamp_min(1e-8)`를 통해 선택된 top-3 모듈의 출력에 연속 가중치 `weight`가 곱해진다. 따라서 활성화된 3개 모듈의 가중치를 통해 `routing_proj`와 `z_cur`로 기울기가 정상 전달된다.
   - **dtype 일관성 (`fp32` `z` ↔ `bf16` 백본)**:
     - [gemma_latent_forward.py](file:///d:/AI/cts/cts/deq/gemma_latent_forward.py#L525)에서 `z = z.to(device=..., dtype=model_dtype)`를 통해 `fp32` `z_cur`가 `bf16`으로 캐스팅(Autograd `ToCopyBackward0` 보존)되어 Gemma 레이어를 통과한다.
     - [gemma_adapter.py](file:///d:/AI/cts/cts/backbone/gemma_adapter.py#L344-L347)에서 `out = out.to(dtype=z.dtype)`로 다시 `fp32`로 변환된 뒤 `g * out + (1.0 - g) * z`가 계산된다. 따라서 `dtype` 불일치 예외는 발생하지 않는다.
   - **구조적 관찰 2a (`w_g_fn()`의 마이크로배치당 호출과 `spectral_norm` 버퍼 갱신)**:
     - [g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L140)에서 `bb.train()`이 설정되고, 153행에서 마이크로배치 루프(`batch_size=4`)마다 `w_g_fn()`(`bb.routing_matrix()`)이 호출된다. `routing_proj`에는 `spectral_norm`이 적용되어 있으므로([gemma_adapter.py](file:///d:/AI/cts/cts/backbone/gemma_adapter.py#L64)), `bb.train()` 상태에서 호출될 때마다 power iteration 버퍼 `_u`, `_v`가 1회씩(옵티마이저 1스텝당 4회) 갱신된다. 반면 `deq_unroll` 내부의 `k_unroll=2` 걸음 동안에는 가중치가 바뀌지 않으므로 동일한 `w_g` 텐서를 재사용하는 것이 수학적으로 일관된다.
   - **구조적 관찰 2b (`bb.train()`과 `bb.eval()` 간 `prepare_deq_context`의 prefix KV 경로 차이)**:
     - [gemma_adapter.py](file:///d:/AI/cts/cts/backbone/gemma_adapter.py#L255-L258)에서 `cache_shared_kv = (needs_shared_kv and (not self.training or self.eq3_suffix_cache_train_enabled))`로 정의되어 있다. 환경변수 `CTS_EQ3_SUFFIX_CACHE_TRAIN`이 기본값 `"0"`이므로, 학습(`bb.train()`) 시에는 `shared_prefix_kv`를 미리 잘라 두지 않고 `_prefill_shared_kv`([gemma_latent_forward.py](file:///d:/AI/cts/cts/deq/gemma_latent_forward.py#L801))에서 전체 `[prefix; z]` 시퀀스에 대해 한 번에 계산하며, 평가 잠재 생성(`g2v11_x8a_latents.py`, `bb.eval()`) 시에는 `cache_shared_kv=True`가 되어 `prefix` KV를 먼저 계산한 뒤 `suffix` KV를 이어 붙인다([gemma_latent_forward.py](file:///d:/AI/cts/cts/deq/gemma_latent_forward.py#L810-L821)). 인과적 어텐션(causal attention) 구조와 `dropout=0.0` 설정 덕분에 두 경로는 bf16 부동소수점 연산 순서 차이 범위 내에서 동치이다.

3. **항목 3 — 고정점 벌점 `L_fp` 정의 ([g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L154))**: **결함 아님**.
   - 원문 식: `l_fp = (z_out - z_prev).square().sum() / z_prev.detach().square().sum().clamp_min(1e-6)`.
   - `M3`([g2v3_m3_full_train.py](file:///d:/AI/cts/scripts/g2v3_m3_full_train.py#L536))에서는 `k_unroll=1`이었기 때문에 `z_prev`가 상수 `z0`여서 `.detach()` 여부가 무관했으나, `X8-A`에서는 `k_unroll=2`이므로 `z_prev = z_1`이 DEQ 파라미터의 함수이다. 만약 분모의 `z_prev`를 `.detach()`하지 않으면 모델이 $\|z_2 - z_1\|^2$를 줄이는 대신 $\|z_1\|^2$의 노름을 키워 `l_fp`를 낮추는 해킹이 가능해지므로, 분모에 `z_prev.detach()`를 둔 것은 올바른 설계이다.
   - 분자 `(z_out - z_prev).square().sum()`에서는 `z_prev`(`z_1`)가 분리되지 않으므로 $z_2$와 $z_1$을 서로 끌어당기는 양방향 기울기가 작용한다. 실제 학습 결과 마지막 200스텝 평균 `l_fp`는 `0.00127~0.00155`($\sqrt{L_{\text{fp}}} \approx 0.036\sim 0.039$)로 안정적으로 유지되었다.

4. **항목 4 — 평가 잠재 생성의 일관성 ([g2v11_x8a_latents.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_latents.py#L31-L55))**: **결함 아님 (관찰 1건)**.
   - `g2v11_x8a_latents.py`는 `g2v11_x8a_train`에서 `deq_unroll`과 `TRAINABLE_KEYS`를 직접 임포트하고, `deq_X8A*.pt`에 저장된 `k_unroll`, `alpha`, `g0`를 그대로 읽어 동일한 2걸음 식을 수행한다.
   - [g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L180)에서 `bb.state_dict().items()` 중 `any(t in k for t in TRAINABLE_KEYS)`를 만족하는 키를 모두 저장하므로, `lora_A`, `lora_B`, `gamma_param`, `routing_proj.parametrizations.weight.original`뿐만 아니라 `routing_proj.parametrizations.weight.0._u` 및 `_v` 버퍼까지 빠짐없이 저장·복원된다.
   - `br.center`는 `LatentXAttnBridge`에서 `register_buffer`로 등록되어 있어([g2v9_xattn.py](file:///d:/AI/cts/handoff/codex/scripts/g2v9_xattn.py#L44)) `bridge_X8A*.pt`의 `bridge_state`에 포함되며, [g2v9_gen.py](file:///d:/AI/cts/handoff/codex/scripts/g2v9_gen.py#L49)에서 `strict=True`로 정확히 로드된다.
   - **구조적 관찰 4a (`br.center` 고정과 `z_out` 이동)**: `br.center`는 학습 시작 시 초기 `Z0`(`z*'`)의 평균으로 고정되며([g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L139)), 학습 중 `z_out`이 초기 `z*'`로부터 약 `6~7%`(1걸음 잔차 `~0.04` + 2걸음 잔차 `~0.035`) 이동하더라도 재계산되지 않는다. 단, 학습과 평가 모두 동일한 고정 `br.center`를 사용하고 `LatentXAttnAdapter` 내부에 `RMSNorm`(`norm_z`)이 후속 적용되므로 학습-평가 간 불일치는 없다.

5. **항목 5 — `E = X7E` 대조 재사용의 타당성 ([cc_x8a.sh](file:///d:/AI/cts/handoff/codex/scripts/cc_x8a.sh#L74-L86))**: **결함 아님 (한계는 §7에 기재)**.
   - `X7E`는 `zsrc_train1000_P4.pt`(`340ef5c2...`)와 `gen_Ctrain.jsonl`(`0c026ea2...`) 위에서 `steps=1500, batch_size=4, lr=3e-4, answer_weight=5.0, gen_budget=480, layers=10,20,30,40, seeds=2047/2049/2050`으로 학습된 `z_src` 브리지이다.
   - `z_src`는 DEQ 반복을 돌리지 않은 warm-start 잠재이므로(`k_unroll=0`), `z_src`에 대해 브리지를 학습하는 실험은 수학적으로 `X7E`와 동일하다.

6. **항목 6 — 메모리 및 실행 시간 ([g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L115-L177))**: **결함 아님**.
   - 마이크로배치 크기 1 단위로 `loss.backward()`를 즉시 호출하여([g2v11_x8a_train.py](file:///d:/AI/cts/handoff/codex/scripts/g2v11_x8a_train.py#L167)) `k_unroll=2` 활성화 그래프가 한 번에 1문항분만 메모리에 상주한다.
   - 177행의 `torch.cuda.empty_cache()`가 매 옵티마이저 스텝(1,500회) 호출되어 약간의 할당기 플러시 오버헤드가 있으나, 실제 실행 시간은 시드당 `1,860~1,886초`(`31.0~31.4분`, 스텝당 `1.24초`)로 48 GB VRAM 한도 내에서 OOM 없이 완주하였다.

7. **항목 7 — 런처 `cc_x8a.sh` 가드 및 중단 조건 ([cc_x8a.sh](file:///d:/AI/cts/handoff/codex/scripts/cc_x8a.sh#L1-L92))**: **결함 아님 (관찰 1건)**.
   - `set -euo pipefail`, `SELFTEST-OK` 확인, GPU 5·6·7 UUID 고정(`GPU 4` 미포함), 백그라운드 잡 개별 `wait $p || exit 1` 체크, 생성 JSONL 600행 및 고유 ID 600개 검증이 모두 정상 작동하였다.
   - **구조적 관찰 7a**: [cc_x8a.sh](file:///d:/AI/cts/handoff/codex/scripts/cc_x8a.sh#L37)의 smoke 모드 로그 출력(`nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n 6p`)은 호스트의 6번째 줄(GPU 5)을 하드코딩하여 출력하나, 본 실행(`full`) 경로와는 무관하다.

---

## 4. 하지 못한 것과 이유

- **GPU 상에서 `g2v11_x8a_train.py` smoke 직접 실행**: `AGENTS.md` §1 규칙 1(서버 접속 금지) 및 규칙 6(로컬 PC에 실험용 GPU가 없고 모델을 돌리지 않음)에 따라 로컬에서는 GPU 학습을 실행하지 않고, 정적 코드 분석 및 서버 실행 로그·매니페스트(`INBOX/20261023-1950-x8a-results/`) 전수 검증으로 대체하였다.

---

## 5. 재현 명령

```powershell
# 작업 디렉터리: D:\AI\cts
python -B handoff/codex/OUTBOX/20261024-1910-C008/verify_c008_v2.py
```

---

## 6. 산출물 SHA256

### 6.1 입력 파일 SHA256 (`INPUT_MANIFEST.json`)
- `handoff/codex/INBOX/TASK-C008.md`: `55f5ecd410a4a083b217fffba27ed3a1ca222bb8ba164b38d1d8494dcca32f8c`
- `handoff/codex/scripts/g2v11_x8a_train.py`: `208eb131f97988c82746b7d480ea1b214fb242f895690fdde4c5a1bd8333c568`
- `handoff/codex/scripts/g2v11_x8a_latents.py`: `9a82c78179c686eaae2b739c812c36f2d146b94fdd24334edceb544028166b0c`
- `handoff/codex/scripts/cc_x8a.sh`: `4a5e457850891107359e55ff375168be9f7c22886068ad0bb2d90aac64e7adca`
- `handoff/codex/scripts/g2v10_train.py`: `48b71ffb94263284f611a35ce4a58bc9fb7609d9b74c18fe31cc95c8c3afe48f`
- `handoff/codex/scripts/g2v9_xattn.py`: `ef3fbd5ed5d227c5b765fc01b95bcd920d4433b005e32e3057dcb479f2d1457d`
- `handoff/codex/scripts/g2v9_gen.py`: `14e466006b9a29c4d30181c424f5e6fb9872c781d03c3313c71fc433f7c2f3b3`
- `scripts/g2v3_m3_full_train.py`: `36e0f18addc8d46f6d8bf376cd95e89cf9eb75a1f15c4e8b714141efe5a49c99`
- `cts/deq/transition_inject.py`: `127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c`
- `cts/backbone/gemma_adapter.py`: `0a9a70c68b80de7fac75b6c129cb57985861a29d9eb962f3758200572e1ae903`
- `cts/deq/gemma_latent_forward.py`: `f30de30846e5982f0a19356d9a0a09ce27c51bafad769ab93440dabc16ca2e90`
- `scripts/s5_eval_grader_v2.py`: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`
- `handoff/codex/INBOX/20261023-1950-x8a-results/cts_x8a_master.log`: `91cd0f619225c2b2c9c59f65b054634c946243e247a5192ec00490677becbc78`
- `handoff/codex/INBOX/20261023-1950-x8a-results/manifest_X8A47.json`: `43cd2cf842a133043c8e8ee0b0fc3eda705809545663f020b627a4634e364a97`
- `handoff/codex/INBOX/20261023-1950-x8a-results/manifest_X8A49.json`: `98aa44028862e63e4f609555de26b0323ba732dccdac600a4f19c3af091e5215`
- `handoff/codex/INBOX/20261023-1950-x8a-results/manifest_X8A50.json`: `b66b637952cc6955be85264ca4b45ef18ed33e13d871a5fbb8515d5cefb63a7e`
- `handoff/codex/INBOX/20261023-1950-x8a-results/train_X8A47.log`: `afbb5ba1158be3273494906defaf743586bb1823131cceb324d7be6c365ad10c`
- `handoff/codex/INBOX/20261023-1950-x8a-results/train_X8A49.log`: `130a1214d32a9509cadee72d3afb927ac95ddc405fe06b1972ce07cbda00634b`
- `handoff/codex/INBOX/20261023-1950-x8a-results/train_X8A50.log`: `6e10c35bf7f65b03f5f61fd7f54b1013866791c5e48004c93301138cd11e11ce`
- `handoff/codex/INBOX/20261023-1950-x8a-results/latdev_X8A47.pt.res.json`: `79503289d47109f93549db200881610611a8b017bc79fe5ff560604e84ad6f30`
- `handoff/codex/INBOX/20261023-1950-x8a-results/latdev_X8A49.pt.res.json`: `be19d67f29163ee341bc866dec07eb5d8416e38e860d6e1a01cea84ebc1926ce`
- `handoff/codex/INBOX/20261023-1950-x8a-results/latdev_X8A50.pt.res.json`: `0022fc1911237c50189a92da63e5e601ce71cb0d9f71c4b700a8675fa2b45857`

### 6.2 산출물 파일 SHA256 (`OUTBOX/20261024-1910-C008/`)
- `c008_static_verification.json`: `4de2a07e1f796ee264b5656da69bb773e99e3a80259bf20278480501146b779b`
- `verify_c008_v2.py`: `081e468941c3c27d586e1805411c52f2f0db153c48bd5d4df25e846214017db3`
- `INPUT_MANIFEST.json`: `1eb69c84417f8426f9065a629873d0586016b1be18cf4f986fd3303dc3cb6515`
- (`REPORT-C008.md` 자신을 포함한 전체 산출물 SHA256은 동일 폴더의 `OUTPUT_MANIFEST.json`에 기록됨)

---

## 7. 의견(판정 아님)

1. **`X8-A`에서 DEQ와 브리지의 결합 학습 자체는 설계 의도대로 작동하였으나, 왜 평가(`REPORT-C009`)에서 문항별 정보 전달(`matched - shuffled`)이 나타나지 않았는가에 대한 구조적 해석**:
   - 정적 검토와 실행 텔레메트리 모두에서 `X8-A`의 기울기 흐름과 고정점 제약(`L_fp`)은 결함 없이 작동하였다: 학습 중 `ans_ce`는 `6.93`에서 `0.29~0.31`로 감소했고, Dev 600의 모든 문항(`600/600`)에서 2번째 걸음의 상대 잔차(`0.0323~0.0357`)가 1번째 걸음의 상대 잔차(`0.0370~0.0407`)보다 작아 수축성(contraction)도 유지되었다.
   - 그럼에도 불구하고 `REPORT-C009`에서 확인된 것처럼 자유 생성 평가 시 `matched - shuffled`가 시드 2049(`+4`문항)와 시드 2050(`-2`문항)에서 상쇄되고 3시드 평균 브리지 게이트가 `[-0.0003, -0.0153, -0.0183, +0.0097]`로 매우 작게 머문 이유는 코드 버그 때문이 아니라, **교사 강제(teacher-forced) 문맥에서는 질문 프롬프트와 이전 생성 토큰만으로도 정답 구간 예측 손실을 줄일 수 있어, 브리지 게이트(`ad.gate`)를 크게 열거나 `z_out`에 문항별 교차 어텐션 신호를 강하게 실을 유인이 약하기 때문**으로 해석된다.
2. **대조군 `E`에 관한 보조 관찰**:
   - `X7E`는 고정된 `z_src` 위에서 브리지만 학습한 대조군으로 타당하나, 만약 후속 실험에서 "`z*'`에서 시작한 2걸음 언롤(`DEQ^2(z*')`)과 `z_src`에서 시작한 2걸음 언롤(`DEQ^2(z_src)`)의 차이"를 엄밀히 분리하고자 한다면, `z0 = z_src`에서 동일하게 `k_unroll=2`로 DEQ+브리지를 결합 학습하는 대조가 더 대칭적인 비교가 될 것이다.

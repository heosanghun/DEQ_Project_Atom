# REPORT-LOCAL-02 — 로컬 4090 동등성 차이 원인 분석 보고 (EXE → B)

## 1. 결과 표 (수치 + 포인터)

### 1-1. 체크포인트 경로 및 로딩 방식 대조
- 포인터: `scratch/harness_A_v1.py:20,36,39`, `scripts/run_local_s7_zstar.py:35,137,148`, `scripts/verify_local_equivalence.py:28,126,138`

| 환경 | 체크포인트 경로 | 파일 크기 | 로딩 코드 행 번호 | strict 옵션 |
| :--- | :--- | :--- | :--- | :--- |
| 서버 정본 | `/home/shoon/checkpoints/m1_final_frozen.pt` | 미상(서버 보관) | `scratch/harness_A_v1.py:20,36,39` | `strict=True` |
| 로컬 실행 | `artifacts/stage1_last.lora_only.pt` | 8,053,367 bytes | `scripts/run_local_s7_zstar.py:35,137,148` | `strict=False` |

### 1-2. `bb.load_state_dict` 반환값 분석 (로컬 `artifacts/stage1_last.lora_only.pt`)
- 총 백본 파라미터 수(`bb.state_dict()`): 2,518개
- 체크포인트 보유 텐서 수(`sd`): 259개

| 구분 | missing_keys 수 | unexpected_keys 수 | 체크포인트 텐서 반영 수 |
| :--- | :--- | :--- | :--- |
| 재매핑 전 (원문 로딩) | 2,368 | 109 | 150 / 259 |
| 재매핑 후 (`lora_A` + `routing_proj`) | 2,259 | 0 | 259 / 259 |

- **재매핑 전 unexpected_keys 앞 5개**:
  1. `routing_proj`
  2. `cg.model.language_model.layers.0.self_attn.q_proj.lora_A.weight`
  3. `cg.model.language_model.layers.0.self_attn.v_proj.lora_A.weight`
  4. `cg.model.language_model.layers.0.self_attn.o_proj.lora_A.weight`
  5. `cg.model.language_model.layers.1.self_attn.q_proj.lora_A.weight`
- **재매핑 후 missing_keys 앞 20개**:
  1. `gamma_param`
  2. `cg.model.language_model.embed_tokens.weight`
  3. `cg.model.language_model.layers.0.self_attn.q_proj.base.weight`
  4. `cg.model.language_model.layers.0.self_attn.q_proj.lora_A.parametrizations.weight.0._u`
  5. `cg.model.language_model.layers.0.self_attn.q_proj.lora_A.parametrizations.weight.0._v`
  6. `cg.model.language_model.layers.0.self_attn.q_norm.weight`
  7. `cg.model.language_model.layers.0.self_attn.k_norm.weight`
  8. `cg.model.language_model.layers.0.self_attn.k_proj.weight`
  9. `cg.model.language_model.layers.0.self_attn.v_proj.base.weight`
  10. `cg.model.language_model.layers.0.self_attn.v_proj.lora_A.parametrizations.weight.0._u`
  11. `cg.model.language_model.layers.0.self_attn.v_proj.lora_A.parametrizations.weight.0._v`
  12. `cg.model.language_model.layers.0.self_attn.o_proj.base.weight`
  13. `cg.model.language_model.layers.0.self_attn.o_proj.lora_A.parametrizations.weight.0._u`
  14. `cg.model.language_model.layers.0.self_attn.o_proj.lora_A.parametrizations.weight.0._v`
  15. `cg.model.language_model.layers.0.mlp.gate_proj.weight`
  16. `cg.model.language_model.layers.0.mlp.up_proj.weight`
  17. `cg.model.language_model.layers.0.mlp.down_proj.weight`
  18. `cg.model.language_model.layers.0.input_layernorm.weight`
  19. `cg.model.language_model.layers.0.post_attention_layernorm.weight`
  20. `cg.model.language_model.layers.0.pre_feedforward_layernorm.weight`

### 1-3. 실행 dtype 및 로그 머리글 설정
- 로컬 백본 모델 로딩 dtype: `torch.bfloat16` (`scripts/run_local_s7_zstar.py:123`)
- 로컬 라우팅 행렬 및 누적 dtype: `torch.float32` (`scripts/run_local_s7_zstar.py:153`)
- 서버 정본 모델 로딩 dtype: `torch.float32` (`scratch/harness_A_v1.py:33`)
- 로컬 409문항 실행 로그 머리글 원문 (`raw_s7/local_s7_a0.25_20260928_193646.log:1-3`):
  ```
  === LOCAL RTX 4090 S7 z* TRACE (alpha=0.25) ===
  Run ID: 20260928_193646 | Device: cuda:0 | Items: 409
  --------------------------------------------------------------------------------
  ```
- 로컬 동등성 검사 로그 머리글 원문 (`raw_s7/equivalence_test_TASK0001.log:1-3`):
  ```
  === EQUIVALENCE TEST (RTX 4090 vs A100 SERVER) ===
  Timestamp: 2026-09-28 23:27:22
  Device: cuda:0 | Alpha: 0.25 | Max NFE: 1000 | Tol: 0.0 | Sigma: 0.02 | Seed: 2030 | g0: 0.25
  ```

---

## 2. 변경·생성 파일 SHA-256

| 파일 경로 | SHA-256 |
| :--- | :--- |
| `results_s7/quarantine_local_20260928/local_zstar_s7_a0.25_train680.pt` | `c3725f28d38e7430464a7507edd616684893f6d3386c9a9ad98afc65a788e0da` |
| `results_s7/quarantine_local_20260928/README.md` | `c21db33fbc50e12d53bf38bc602b9e68c2f1f3a9e6b72a6a0e67610fae1f7cbe` |
| `shared/reports/REPORT-LOCAL-02.md` | 본 보고서 |

---

## 3. 예상 밖 사건

1. **체크포인트 계통 불일치**:
   - 서버 정본은 `m1_final_frozen.pt`를 `strict=True`로 로드하여 기저 모델 가중치 전체를 체크포인트에서 공급받음.
   - 로컬은 어댑터 전용 가중치(`stage1_last.lora_only.pt`, 259개)만 보유하여 `strict=False`로 로드하였으며, 기저 2,259개 파라미터는 HuggingFace 허브 스냅샷 원본 상태로 유지됨.
2. **모델 로딩 정밀도 불일치**:
   - 서버는 `torch.float32`로 기저 모델을 인스턴스화(`harness_A_v1.py:33`), 로컬은 `torch.bfloat16`으로 인스턴스화(`run_local_s7_zstar.py:123`).

---

## 4. 서버 상태 1줄
- SSH 서버 접속: 2026-09-28 23:44:38 KST 기준 `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` (장애 지속).

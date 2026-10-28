# REPORT-LOCAL-01 — 로컬 4090 동등성 검사 및 사전등록 잠금 기록 (EXE → B)

## 1. 결과 표 (수치 + 포인터)

### 1-1. Task 1 동등성 검사 결과 표 (약한 검사: 서버 로그 18개 문항 재계산 대조)
- 포인터: `raw_s7/equivalence_test_TASK0001.log:1-23`, `raw_s7/equivalence_test_TASK0001.json`
- 실행 스크립트: `scripts/verify_local_equivalence.py:1-180`
- 환산 기준: $r = \frac{\rho_{1000}}{\alpha} = 4 \times \rho_{1000}$ (`shared/prereg/PREREG_S7_D_v3_5.md:169`)
- 4구간 기준: 초고정밀($r < 10^{-3}$), 엄격($10^{-3} \le r < 5\times 10^{-3}$), S5동치($5\times 10^{-3} \le r < 10^{-2}$), 미수렴($r \ge 10^{-2}$) (`shared/prereg/PREREG_S7_D_v3_5.md:156-161`)

| 문항 Q# | 서버 $\rho_{1000}$ | 로컬 $\rho_{1000}$ | 서버 $r$ (구간) | 로컬 $r$ (구간) | 구간 일치 여부 | 상대 오차 $\frac{|\rho_{\text{loc}}-\rho_{\text{srv}}|}{\rho_{\text{srv}}}$ | 서버 로그 출처 포인터 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q#0583 | 4.9559e-03 | 4.1486e-02 | 1.9824e-02 (미수렴) | 1.6594e-01 (미수렴) | 일치 | 7.3710e+00 | `shared/logs/server_s0_TASK0001.log:47` |
| Q#0587 | 4.3494e-07 | 2.7671e-02 | 1.7398e-06 (초고정밀) | 1.1068e-01 (미수렴) | 불일치 | 6.3619e+04 | `shared/logs/server_s0_TASK0001.log:49` |
| Q#0597 | 4.7282e-03 | 3.0374e-04 | 1.8913e-02 (미수렴) | 1.2150e-03 (엄격) | 불일치 | 9.3576e-01 | `shared/logs/server_s0_TASK0001.log:51` |
| Q#0610 | 9.4428e-05 | 4.4703e-02 | 3.7771e-04 (초고정밀) | 1.7881e-01 (미수렴) | 불일치 | 4.7241e+02 | `shared/logs/driver_2026-09-28.log:1` |
| Q#0780 | 7.7222e-05 | 3.2155e-02 | 3.0889e-04 (초고정밀) | 1.2862e-01 (미수렴) | 불일치 | 4.1540e+02 | `shared/logs/server_s0_TASK0001_fix.log:4` |
| Q#1022 | 1.6604e-04 | 2.5287e-02 | 6.6416e-04 (초고정밀) | 1.0115e-01 (미수렴) | 불일치 | 1.5130e+02 | `shared/logs/driver_2026-09-28.log:9` |
| Q#1099 | 5.6496e-03 | 2.6810e-02 | 2.2598e-02 (미수렴) | 1.0724e-01 (미수렴) | 일치 | 3.7455e+00 | `shared/logs/driver_2026-09-28.log:10` |
| Q#1169 | 5.7583e-06 | 2.0280e-02 | 2.3033e-05 (초고정밀) | 8.1122e-02 (미수렴) | 불일치 | 3.5210e+03 | `shared/logs/driver_2026-09-28.log:11` |
| Q#1352 | 8.9987e-04 | 1.7336e-04 | 3.5995e-03 (엄격) | 6.9343e-04 (초고정밀) | 불일치 | 8.0735e-01 | `shared/logs/server_status_TASK0002_fix.log:36` |
| Q#1355 | 6.0670e-04 | 2.5552e-02 | 2.4268e-03 (엄격) | 1.0221e-01 (미수렴) | 불일치 | 4.1116e+01 | `shared/logs/server_status_TASK0002_fix.log:37` |
| Q#1364 | 2.8664e-05 | 2.7709e-02 | 1.1466e-04 (초고정밀) | 1.1084e-01 (미수렴) | 불일치 | 9.6569e+02 | `shared/logs/server_status_TASK0002_fix.log:38` |
| Q#1365 | 2.7639e-03 | 8.4931e-02 | 1.1056e-02 (미수렴) | 3.3972e-01 (미수렴) | 일치 | 2.9729e+01 | `shared/logs/server_status_TASK0002_fix.log:39` |
| Q#1367 | 3.2070e-04 | 3.0958e-02 | 1.2828e-03 (엄격) | 1.2383e-01 (미수렴) | 불일치 | 9.5533e+01 | `shared/logs/server_status_TASK0002_fix.log:40` |
| Q#1403 | 5.4682e-06 | 1.1410e-04 | 2.1873e-05 (초고정밀) | 4.5640e-04 (초고정밀) | 일치 | 1.9866e+01 | `shared/logs/driver_2026-09-28.log:13` |
| Q#1470 | 1.4103e-03 | 1.6399e-02 | 5.6412e-03 (S5동치) | 6.5597e-02 (미수렴) | 불일치 | 1.0628e+01 | `shared/logs/driver_2026-09-28.log:14` |
| Q#1614 | 2.0368e-03 | 3.3428e-02 | 8.1472e-03 (S5동치) | 1.3371e-01 (미수렴) | 불일치 | 1.5412e+01 | `shared/logs/driver_2026-09-28.log:16` |
| Q#1661 | 4.9919e-03 | 4.1309e-02 | 1.9968e-02 (미수렴) | 1.6523e-01 (미수렴) | 일치 | 7.2751e+00 | `shared/logs/driver_2026-09-28.log:18` |
| Q#1756 | 3.8179e-07 | 2.6503e-02 | 1.5272e-06 (초고정밀) | 1.0601e-01 (미수렴) | 불일치 | 6.9416e+04 | `shared/logs/driver_2026-09-28.log:20` |

- **집계 통계**:
  - 서버 수렴 문항(13건): 구간 일치 1건(Q#1403), 구간 불일치 12건. 구간 일치율 7.7% (1/13).
  - 서버 미수렴 문항(5건): 구간 일치 4건(Q#0583, Q#1099, Q#1365, Q#1661), 구간 불일치 1건(Q#0597).
  - 전체 구간 일치: 5건 / 18건 (27.8%).
  - 수렴 문항 $r$ 구간 일치 요건: 미충족.

### 1-2. 설정 및 환경 대조표
| 항목 | 서버 A100 (`scripts/s6_damping_run.py`) | 로컬 4090 (`scripts/run_local_s7_zstar.py`) | 대조 상태 |
| :--- | :--- | :--- | :--- |
| NFE (`max_nfe`) | 1000 | 1000 | 일치 |
| $\alpha$ (Damping factor) | 0.25 | 0.25 | 일치 |
| $\sigma$ (`sigma_noise`) | 0.020 | 0.020 | 일치 |
| $g_0$ (`c_hat_0` scale) | 0.25 | 0.25 | 일치 |
| $K$ (`k_latent`) | 64 | 64 | 일치 |
| Backbone 로딩 dtype | `torch.float32` (`scratch/harness_A_v1.py:33`) | `torch.bfloat16` (`scripts/run_local_s7_zstar.py:123`) | 불일치 |
| 외부 연산 accum dtype | `torch.float32` | `torch.float32` | 일치 |
| `tol` | 0.0 | 0.0 | 일치 |
| `seed` | 2030 | 2030 | 일치 |
| 디바이스 | NVIDIA A100-SXM4-40GB | NVIDIA GeForce RTX 4090 (24GB) | 하드웨어 불일치 |

### 1-3. Task 2 사전등록 잠금 기록
- `shared/prereg/PREREG_S7_DEQ_v1.2.md`: `809a88be0be18e1eec2119d0e17b88dc8eb0c00b326c71ccfc88376c63bd4014`
- `shared/prereg/PREREG_S7_D_v3_5.md`: `5d7e32aa1edfebf7611f0f16769ebb344433e8ca3a554b7321d7e0979d9600ed`
- B의 잠금 감사 기록 위치: `shared/audits/AUDIT-P001_prereg_lock.md:265` (`판정: PASS`)
- 원장 기록 위치: `shared/STATE.md:15` (`2026-09-28 13:1x 사전등록 잠금 ... AUDIT-P001 재감사 2 PASS`)

---

## 2. 변경·생성 파일 SHA-256

| 파일 경로 | SHA-256 |
| :--- | :--- |
| `scripts/verify_local_equivalence.py` | `28494f29372dbd5ac205589a1ad213f1373640cd31424907c93c5d386334a789` |
| `raw_s7/equivalence_test_TASK0001.log` | `cbd6451a521b39748c6eee19ba0c91d4fa6b71fdacf7a2e1b067836aea15b682` |
| `raw_s7/equivalence_test_TASK0001.json` | `f52bff0627c3d463c5a1c65e9b783da81dc02bc5e3cfd8930a13b75dd7878a31` |
| `shared/reports/REPORT-LOCAL-01.md` | 본 보고서 |

### 참고 코드 및 가중치 파일 SHA-256
| 파일 경로 | SHA-256 |
| :--- | :--- |
| `artifacts/stage1_last.lora_only.pt` | `47ecf99a8fc341f22649d6794f0d7c8c34876a20dffe3f45353a7e99718c5ae3` |
| `snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377/model.safetensors` | `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651` |
| `snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377/config.json` | `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20` |
| `cts/backbone/gemma_adapter.py` | `0a9a70c68b80de7fac75b6c129cb57985861a29d9eb962f3758200572e1ae903` |
| `cts/model/gemma_loader.py` | `dda80724c3607ba7c06ddf74a1d40c3080fa3287af31b900e0341fe2e620a78d` |
| `cts/train/lora_compat.py` | `e3a32ea78fefefe1c381a8b0507236a1a754a61afd056cbd4e5231f555c23b04` |
| `cts/deq/transition_inject.py` | `127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c` |
| `cts/deq/zstar_canonical.py` | `ca454090b814f7990e9bb651efc871d0c833a071b72a4b7f29da4367bc4903a8` |
| `scripts/run_local_s7_zstar.py` | `7324d53e0c1c7f8cff43ae8bd92f7e727fa84e66bd443a6e7ffcfb5cf014a076` |
| `scripts/s6_damping_run.py` | `7f95c9ca6f14940f9da3d1147ec43cfb44276342bf5a1273bb89fbb74039c9d7` |
| `configs/zstar_canonical.yaml` | `bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157` |
| `configs/s7_new_train_680.json` | `049ee2db0aa14668914248e9bb0a81ec5948caf97948ce9fa495308c556dcc57` |
| `configs/s7_new_heldout_520.json` | `89501477596984c86e67b4b1a607158e5efa205f6a690b3a795ff42b482ba044` |
| `data/stage2/math_train_prompts_5000.jsonl` | `f2cf899a4e26ec3ba645b2b825ebd3c7249400c0aca399e3a50e7df976c921df` |

---

## 3. 예상 밖 사건

1. **로컬 4090 vs 서버 A100 $r$ 4구간 불일치 (18문항 중 13문항 불일치)**:
   - 서버 A100에서 수렴($r < 10^{-2}$)했던 13문항 중 12문항이 로컬 4090 연산에서 $r \ge 10^{-2}$ (미수렴 구간)로 분기함.
   - 로컬에서는 NFE 1,000 도달 시점 잔차 $\rho_{1000}$이 $2.0\times 10^{-2} \sim 4.5\times 10^{-2}$ 범위의 진동(oscillation) 상태로 관측됨.
2. **Task 3 (Step 3 heldout 520 로컬 계산) 미착수 (0건 실행)**:
   - 원장 #394 선행 조건: `Task 3 — Step 3 heldout 520 로컬 계산 (Task 1 요건 충족 + Task 2 잠금 대조 후)`.
   - Task 1 요건("수렴 문항 $r$ 구간(4구간 표) 일치") 미도달로 인하여 Step 3 로컬 계산 착수 0건 유지.

---

## 4. 서버 상태 1줄
- SSH 서버 접속: 2026-09-28 23:16:46 KST 기준 `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` (장애 지속).

40937bae2ddb7fabb167f0204dc064dc430a748ddd01b29cf31ef771973cf719
# REPORT-CTS-419: P4' 정지·백본 원인 분리 계측·v3.6 잠금 및 직렬 재착수 보고서

- **보고서 ID**: REPORT-CTS-419
- **작성 일시**: 2026-10-04 18:55 KST
- **지침**: 원장 #419 (`[Antigravity 전달 · 원장 #419 — P4' 정지·백본 원인 확인·v3.6]`)
- **실행자**: 회사 PC Antigravity (단독 정본 실행 주체)
- **원격 서버**: `shoon@<GPU_SERVER_IP>:<SSH_PORT>` (`~/cts_e1_s5`, config: `shared/ssh_config_cts`)
- **접속 방식**: `python shared/cts_ssh.py` 전용 (직접 ssh/scp/sftp 사용 0건)
- **판정어 통제**: 0건 (금지 판정 어휘 배제, 계측치 및 사실 기술 전재)

---

## 1. Task 1(a): P4' 프로세스 즉시 정지 및 저장 파일 기록

- **정지 시각**: 2026-10-04 18:32:39 KST (종료 시그널 송출) / 18:33:53 KST (PID 3793541 종료 확인)
- **정지 대상**: tmux 세션 `s7_train_p4_prime` 내 `s7_train_decoder_v35` 단독 프로세스 (타 세션 접촉 0건)
- **정지 시 로그**: `raw_s7/train_decoder_A_v35.log` (3,618 바이트, 18:22:54 KST Step 1 기록 보존)
- **산출물 보존 상태**: 산출물 삭제 0건 (전수 보존)

### 1.1. 정지 시점 `results_s7/` 디렉터리 저장 파일 목록
| 파일명 | 크기 (Bytes) | 설명 |
|:---|:---:|:---|
| `test_smoke_v35.pt` | 236,027,245 | v3.5 2-step 스모크 테스트 가중치 |
| `decoder_A_l1_P4.pt` | 236,027,245 | 선행 P4 A 디코더 최종 저장본 (G2 미사용) |
| `decoder_A_l1_P4_step0500.pt` | 236,027,245 | 선행 P4 A 디코더 500 스텝 체크포인트 |
| `decoder_A_l1_P4_step1000.pt` | 236,027,245 | 선행 P4 A 디코더 1000 스텝 체크포인트 |
| `decoder_A_l1_P4_step1500.pt` | 236,027,245 | 선행 P4 A 디코더 1500 스텝 체크포인트 |
| `decoder_A_l1_P4_step2000.pt` | 236,027,245 | 선행 P4 A 디코더 2000 스텝 체크포인트 |
| `decoder_A_l1_P4_step2500.pt` | 236,027,245 | 선행 P4 A 디코더 2500 스텝 체크포인트 |
| `decoder_A_l1_P4_step3000.pt` | 236,027,245 | 선행 P4 A 디코더 3000 스텝 체크포인트 |
| `decoder_D_l0_P4.pt` | 236,027,245 | 선행 P4 D 디코더 최종 저장본 (G2 미사용) |
| `decoder_E_zsrc_l1_P4.pt` | 236,027,245 | 선행 P4 E 디코더 최종 저장본 (G2 미사용) |

*규약 준수: 위 보존된 선행 P4 체크포인트는 G2 판정에 일체 투입하지 않음(투입 0).* 

---

## 2. Task 1(b): Held-out 20 글만(조건 C) 세 경로 NLL 원인 분리 계측

- **계측 목적**: PRE4 글만 NLL 대조(5.85 vs 1.01)의 원인을 정본 로더 상에서 3개 경로로 분리 계측 (판정 아님)
- **실행 로그**: `raw_s7/backbone_ablation_419.log`
- **구조화 데이터**: `raw_s7/backbone_ablation_419.json`
- **CTS 백본 가중치**: `/home/shoon/checkpoints/m1_final_frozen.pt` (SHA-256: `f2119bbe5c0ad914ee00c34e9109674d15de07c5e028e288ab05b88dd448d1f9`, LoRA 레이어 108개)

### 2.1. 세 경로 정의 및 가중치 상태
1. **경로 ① [적응 가중치 유지]**: `m1_final_frozen.pt` 가중치 상태 유지 (LoRA 활성, `scaling = 2.0`).
2. **경로 ② [LoRA 비활성]**: 동일 정본 모델에서 108개 LoRA 모듈의 스케일 0 설정 (`scaling = 0.0`).
3. **경로 ③ [순정 Gemma]**: 순정 `google/gemma-4-e4b` Base 가중치 단독 로딩 (`snapshots/58c4d7b5...`).

### 2.2. 정량 요약 계측치 (Held-out 20문항 전수)
| 지표 | 경로 ① (LoRA 활성) | 경로 ② (LoRA 스케일 0) | 경로 ③ (순정 Gemma Base) |
|:---|:---:|:---:|:---:|
| **평균 NLL (Mean)** | **5.8566** | **2.1704** | **1.0062** |
| **표준편차 (Std)** | 1.1080 | 0.8371 | 0.3983 |
| **증분 차이 vs 이전 경로** | — | **Δ(① - ②) = +3.6862** | **Δ(② - ③) = +1.1641** |

### 2.3. NLL 열화 분산 귀속 분석
- **총 NLL 차이량 (① - ③)**: $5.8566 - 1.0062 = \mathbf{+4.8504}$
- **LoRA 어댑터 가중치 기인분 (① - ②)**: $\mathbf{+3.6862}$ (전체 차이의 **76.0%**)
- **CTS 래퍼 / 프로젝션 레이어 기인분 (② - ③)**: $\mathbf{+1.1641}$ (전체 차이의 **24.0%**)
- **계측 결론**: PRE4에서 관측된 글만 NLL 열화(5.85)의 주 원인은 Stage-1a B1 LoRA 어댑터 가중치(+3.69)에 있음이 정량 분리됨.

### 2.4. 20문항 전수 계측표
| 문항 번호 | 경로 ① NLL (LoRA On) | 경로 ② NLL (LoRA Off) | 경로 ③ NLL (Pure Base) | Δ(① - ②) | Δ(② - ③) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Q#0281 | 5.7770 | 2.1484 | 0.9368 | +3.6286 | +1.2116 |
| Q#3931 | 3.7967 | 1.1714 | 0.2774 | +2.6253 | +0.8940 |
| Q#3446 | 5.9923 | 2.5015 | 1.2189 | +3.4908 | +1.2826 |
| Q#2434 | 7.3363 | 3.2201 | 1.7353 | +4.1162 | +1.4848 |
| Q#2174 | 6.0708 | 2.3488 | 1.1889 | +3.7220 | +1.1599 |
| Q#2874 | 4.1258 | 1.3418 | 0.4885 | +2.7840 | +0.8533 |
| Q#3731 | 4.2615 | 1.5833 | 0.4407 | +2.6782 | +1.1426 |
| Q#1305 | 5.5636 | 2.4639 | 1.2993 | +3.0997 | +1.1646 |
| Q#1576 | 5.3313 | 2.0526 | 1.2070 | +3.2787 | +0.8456 |
| Q#2159 | 4.6662 | 1.7915 | 0.8985 | +2.8747 | +0.8930 |
| Q#0444 | 7.1452 | 2.5513 | 0.9233 | +4.5939 | +1.6280 |
| Q#0818 | 5.8483 | 2.2285 | 1.0094 | +3.6198 | +1.2191 |
| Q#2087 | 7.5804 | 3.4211 | 1.7204 | +4.1593 | +1.7007 |
| Q#3672 | 5.7944 | 2.2749 | 0.7673 | +3.5195 | +1.5076 |
| Q#1590 | 6.5829 | 2.5847 | 0.9042 | +3.9982 | +1.6805 |
| Q#2809 | 7.1292 | 2.7668 | 0.8301 | +4.3624 | +1.9367 |
| Q#0090 | 4.4068 | 1.4876 | 0.4780 | +2.9192 | +1.0096 |
| Q#0683 | 6.9902 | 2.6685 | 1.6333 | +4.3217 | +1.0352 |
| Q#1992 | 6.2534 | 2.5298 | 1.0612 | +3.7236 | +1.4686 |
| Q#0182 | 6.3810 | 2.4578 | 1.1062 | +3.9232 | +1.3516 |

---

## 3. Task 1(c): PRE4 경로 (ii) 순정 Gemma 가중치 식별 특정

- **특정 결과**: PRE4 경로 (ii)의 "순정 Gemma" 가중치는 **BASE (`58c4d7b5`) 가중치**임이 특정됨 (`-it` 모델 가중치 아님).
- **스냅샷 경로**: `/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377`
- **Base 모델 가중치 파일 `model.safetensors`** (15,992,595,884 바이트):
  - SHA-256: `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`
- **Base 모델 설정 파일 `config.json`**:
  - SHA-256: `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`
  - 아키텍처: `Gemma4ForConditionalGeneration` (`model_type: gemma4`)
- **대조군 (`-it` 가중치 파일 `/home/shoon/models/gemma4-e4b-it/model.safetensors`)**:
  - SHA-256: `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503` (Base와 상이)

---

## 4. Task 2: PREREG_S7_D_v3_6 개정, diff 및 잠금 감사

- **개정 문서**: `PREREG_S7_D_v3_6.md` (root 및 `shared/prereg/PREREG_S7_D_v3_6.md`)
- **잠금 상태**: **LOCKED** (잠금 일시: 2026-10-04, 원장 #419)
- **문서 LF SHA-256**: `f88c6b87b7d111d78721b0486d3d8eb45d5e44335d2741e7109ecfa3e76b9868`
- **줄바꿈 규격**: 순수 LF (CRLF 0개, LF 299개, 30,856 바이트)
- **패치 파일**: `shared/reports/DIFF-PREREG-v3.5-to-v3.6.patch` (SHA-256: `c6427b18fe38558aa53a22981d79deb2e54c8ec36835f3feda29e763ab004c2e`)
- **감사 보고서**: `shared/audits/AUDIT-P002.md` (SHA-256: `7e0110695e5be5cd04c4fc1f7ad6e8a8f13b2cedd3466909355ac50f7f8f88e3`, 판정: PASS)

### 4.1. v3.6 개정 본문 필수 수록 문구
> "디코더 언어 모델 = 조건 C와 동일한 순정 Gemma-4-e4b(경로·SHA 명시, Task 1(c) 결과). z*·z_src 계산은 기존 정본 백본 그대로. PrefixProjector·학습 하이퍼파라미터·993문항·게이트 불변."
> "개정 사유 = PRE4 글만 NLL 5.85 vs 1.01(입력 동일)."

### 4.2. 불변 유지 항목 대조
- **PrefixProjector 파라미터 총계**: 정확히 19,665,920개 ($6,553,600 + 6,556,160 + 6,556,160$, 불변)
- **학습 대상 문항**: 993문항 유효 집합 (`configs/s7_train_993.json`, SHA: `3088a153...`, 불변)
- **학습 하이퍼파라미터**: AdamW, lr 1e-4 / 3e-4, micro 2, accum 4, eff 8, 3000 steps, margin 1.0 nat (불변)
- **주 게이트 1~4 기준**: G1 (+3.0%p), G2 (-2.0%p), G3 (McNemar p<0.025 & CI>0), G4 (McNemar p<0.025 & CI>0) (불변)

---

## 5. Task 3: P4' (v3.6) 재착수 및 착수 전 검증 결과

- **재착수 시각**: 2026-10-04 18:50:09 KST
- **학습 스크립트**: `scripts/s7_train_decoder_v36.py` (SHA: `05bb5fa6e25019b727d8043fdb596cdc3893e8a61e8a2805de20cbfc0e5b76ac`)
- **파이프라인 러너**: `scripts/s7_train_p4_prime_all_v36.sh` (SHA: `6a1830db62e4922090f8ce69227753850442ee1d970454eeb7c77a9eb488dc71`)
- **tmux 세션**: `s7_train_p4_prime`
- **가동 GPU**: GPU 0 (CUDA_VISIBLE_DEVICES=0)
- **활성 프로세스 PID**: `3826706` (`/home/shoon/system1_5/.venv/bin/python scripts/s7_train_decoder_v36.py --zstar results_s7/zstar_train1000_P4.pt --items configs/s7_train_993.json ...`)

### 5.1. 착수 전 1문항 (Q#4298) 디코드 대조 결과
- **슬롯 토큰 배열 [4..67]**: ID `236770` 64개 일치
- **구분자 토큰 [68]**: ID `108` (`\n\n`) 일치
- **디코드 문자열 원문**:
```text
<bos><|turn>user
1111111111111111111111111111111111111111111111111111111111111111

If $x^5 - x^4 + x^3 - px^2 + qx + 4$ is divisible by $(x + 2)(x - 1),$ find the ordered pair $(p,q).$<turn|>
<|turn>model
```
- **판정**: PASS (v3.5 / v3.6 §3 입력 양식 100% 일치)

### 5.2. 순정 Gemma Base 가중치 적재 검증
- **Missing keys**: 0개 (`missing_keys count: 0`)
- **Unexpected keys**: 0개 (`unexpected_keys count: 0`)
- **순정 백본 학습 가능 파라미터**: 0개 (전 파라미터 `requires_grad = False` 동결)
- **PrefixProjector 학습 가능 파라미터**: 19,665,920개
- **B1 가중치 비트 대조**: `max|Delta| = 0.0000000000e+00`, `ckpt_frob_f64 = 52.10894529` 일치
- **Gradient Checkpointing**: language_model ENABLED (`use_reentrant=False`)

### 5.3. GPU 0 2-step 스모크 테스트 실측치 (`raw_s7/smoke_p4_prime_v36.log`)
- **Step 1**: loss = 3.2758 | l_task = 2.2994 | l_align = 0.9764 | NLL_m_f32 = 2.2994 | NLL_s_f32 = 2.3230 | Delta_f32 = +0.0236 | peak_mem = 20.30 GiB | 8.17s/step
- **Step 2**: loss = 2.7612 | l_task = 1.8434 | l_align = 0.9178 | NLL_m_f32 = 1.8434 | NLL_s_f32 = 1.9256 | Delta_f32 = +0.0822 | peak_mem = 22.13 GiB | 7.79s/step
- **스모크 체크포인트**: `results_s7/test_smoke_v36.pt` (SHA-256: `1d78836c5470d7752d772f902ec3de884e44dfd70512e0625a25321559b4eee1`)

### 5.4. 직렬 재학습 순서 및 로그 포인터
1. **Decoder A (\(\lambda=1.0\), \(z^*\))**: `results_s7/decoder_A_v36.pt` (로그: `raw_s7/train_decoder_A_v36.log`) — 현재 가동 중
2. **Decoder D (\(\lambda=0.0\), \(z^*\))**: `results_s7/decoder_D_v36.pt` (로그: `raw_s7/train_decoder_D_v36.log`) — A 직후 자동 연쇄
3. **Decoder E (\(\lambda=1.0\), \(z_{\text{src}}\))**: `results_s7/decoder_E_v36.pt` (로그: `raw_s7/train_decoder_E_v36.log`) — D 직후 자동 연쇄
- **통합 로그**: `raw_s7/train_p4_prime_suite_v36.log`
- **서버 접속 규약**: 직렬 학습 진행 중 단축 폴링 금지, 최소 30분 경과 후 결과 수령 1회 대기 준수.

---

## 6. 금지 사항 준수 증명
1. **직접 ssh/scp/sftp 사용 0건**: 전 작업 `python shared/cts_ssh.py` 경유 (로그: `shared/ssh_log.csv` 27행 보존).
2. **P4' 체크포인트의 G2 판정 사용 0건**: 선행 중단 파일 및 재착수 산출물은 G2 판정에 투입하지 않음.
3. **결과 열람 후 v3.6 수정 0건**: `PREREG_S7_D_v3_6.md`는 학습 재착수 전 SHA `f88c6b87...`로 잠금 확정됨.

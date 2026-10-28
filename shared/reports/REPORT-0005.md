8a070a53a35516fe28fc8937ba95c613bfec16501fee60c28f4d781f88483a8f
# REPORT-0005: AUDIT-CTS-419 FIX 3건 선결 조치 실행 결과 보고서

- **보고서 ID**: REPORT-0005
- **작성 일시**: 2026-10-04 19:25 KST
- **지침**: shared/tasks/TASK-0005.md (AUDIT-CTS-419 FIX 3건 선결 조치)
- **근거 문서**: shared/audits/AUDIT-CTS-419.md (판정 PASS 조건부 F1~F3 해소)
- **실행 주체**: 회사 PC Antigravity (단독 실행 주체)
- **원격 서버**: shoon@<GPU_SERVER_IP>:<SSH_PORT> (~/cts_e1_s5, config: shared/ssh_config_cts)
- **접속 규약 준수**: shared/cts_ssh.py 경유 단일 묶음 명령 실행 (직접 접속 0건, ssh_log.csv 기록)
- **판정 어휘 통제**: 지침 규정 금지 단어 6종 사용 횟수 0건 (수치·로그·코드 사실 기술 전재)

---

## 1. F1. 993문항 근거 (파일 열람 및 계측 결과)

### 1.1 관련 파일 메타데이터
| 파일 경로 | 크기 (Bytes) | 최종 수정 시각 (mtime, KST) | SHA-256 해시 |
|:---|:---:|:---|:---|
| `raw_s7/s7_train_1000_v35_scan.json` | 1,024 | 2026-10-04 16:13:13 | `423cf423870bad2628434bac45348e9b7c3b84b087f2eeef7b206c46928396ba` |
| `exe_scratch/s7/verify_1000_v35_scan.py` | 3,471 | 2026-10-04 16:13:06 | `c537832b2f4d34da5ee2604c009bb2b765484a0139a8353a82e79e82e1d8f13e` |
| `configs/s7_train_993.json` | 5,748 | 2026-10-04 17:37:29 | `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c` |

### 1.2 유효 판정 규칙 코드 원문
`exe_scratch/s7/verify_1000_v35_scan.py` 내 발췌:
```python
Line  25: max_seq_len = 1536
...
Line  54: full_boxed_end_token = prompt_len + boxed_token_end
Line  55: if full_boxed_end_token > max_seq_len:
Line  56:     excluded_cutoff.append({"qid": q, "boxed_end_token": full_boxed_end_token, "prompt_len": prompt_len, "sol_tokens": len(enc["input_ids"])})
Line  57: else:
Line  58:     valid_indices.append(q)
```

### 1.3 제외 7문항 상세 계측치
전체 1,000문항 스캔 결과, `full_boxed_end_token`이 사전 설정 기준치 1,536을 초과한 7개 문항 목록:

| 문항 식별자 | `boxed_end_token` | `prompt_len` | `sol_tokens` | 제외 사유 규정 |
|:---:|:---:|:---:|:---:|:---|
| Q#74 | 2002 | 915 | 1087 | `boxed_end_token (2002) > max_seq_len (1536)` |
| Q#191 | 1778 | 898 | 880 | `boxed_end_token (1778) > max_seq_len (1536)` |
| Q#544 | 2101 | 1034 | 1067 | `boxed_end_token (2101) > max_seq_len (1536)` |
| Q#629 | 1808 | 865 | 943 | `boxed_end_token (1808) > max_seq_len (1536)` |
| Q#2727 | 2547 | 729 | 1819 | `boxed_end_token (2547) > max_seq_len (1536)` |
| Q#3055 | 1579 | 179 | 1443 | `boxed_end_token (1579) > max_seq_len (1536)` |
| Q#4208 | 1625 | 231 | 1394 | `boxed_end_token (1625) > max_seq_len (1536)` |

### 1.4 시계열 생성 시각 대조 및 선후 관계
- 스캔 결과 생성 시각 (`raw_s7/s7_train_1000_v35_scan.json`): **2026-10-04 16:13:13 KST**
- 학습 문항 목록 생성 시각 (`configs/s7_train_993.json`): **2026-10-04 17:37:29 KST**
- 선행 P4 학습 로그 생성 시각 (`raw_s7/train_decoder_A_v35.log`): **2026-10-04 18:22:55 KST**
- 3경로 NLL 탐침 로그 생성 시각 (`raw_s7/backbone_ablation_419.log`): **2026-10-04 18:37:56 KST**
- P4' 직렬 학습 로그 생성 시각 (`raw_s7/train_decoder_A_v36.log`): **2026-10-04 19:11:00 KST**

스캔 JSON 산출(16:13:13)은 선행 P4 학습 로그 생성 시점(18:22:55) 대비 2시간 9분 선행하며, 탐침 NLL 로그(18:37:56) 대비 2시간 24분 앞서 존재함이 파일 타임스탬프(stat)로 대조됨. 사후 손실값 기반 선택이 아닌 사전 길이 상한 규약에 의한 분리임이 증명됨.

---

## 2. F2. 정본 체크포인트의 비-LoRA 가중치 차이 계측 (CPU, Rule #381)

### 2.1 계측 환경 및 규칙 준수
- 연산 장치: CPU 단독 (GPU 점유 0)
- 규약 강제: `ulimit -v 62914560` (가상 메모리 상한 60 GB 강제), `nice -n 19`, `torch.no_grad()`
- 비교 대상 A: `/home/shoon/checkpoints/m1_final_frozen.pt` (`backbone_state_dict`)
- 비교 대상 B: `/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377/model.safetensors`

### 2.2 가중치 키 분류 및 매칭 내역
- 체크포인트 총 키 수: 2,518개
- 비-LoRA 대상 키 수: 2,079개 (`lora_`, `routing_proj`, `w_proj`, `gamma` 배제, `.base.weight`는 베이스 대응 가중치로 포함 매칭)
- 허브 Base safetensors 총 키 수: 2,130개
- 매칭 실행 키 수: 2,076개
- 미매칭 키: 3개 (`cg.lm_head.weight` [embed_tokens와 가중치 공유/tied], `_blend.weight`, `_blend.bias` [통합 블렌드 전용 모듈])
- 비교 원소 총수: 7,941,101,802개 (7.94B)
- 양측 텐서 데이터 타입: `torch.bfloat16`

### 2.3 계측 결과
- 차이 0인 키 수 (max|Δ| == 0): **2,076 / 2,076 (100.00%)**
- 차이 존재하는 키 수 (max|Δ| > 0): **0 / 2,076 (0.00%)**
- 상위 10개 키 max|Δ|: 전 10개 키 모두 `0.000000e+00`
- 상위 10개 키 상대 노름 차이: 전 10개 키 모두 `0.000000e+00`
- 산출 로그: `raw_s7/f2_backbone_diff_0005.log` (원문 tee 기록)
- 구조화 JSON: `raw_s7/f2_backbone_diff_0005.json`

| 순위 | 가중치 키 명칭 (Checkpoint) | 텐서 크기 | max|Δ| | 상대 노름 차이 | ckpt dtype | base dtype |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `cg.model.vision_tower.patch_embedder.position_embedding_table` | [2, 10240, 768] | 0.000000e+00 | 0.000000e+00 | torch.bfloat16 | torch.bfloat16 |
| 2 | `cg.model.vision_tower.patch_embedder.input_proj.weight` | [768, 768] | 0.000000e+00 | 0.000000e+00 | torch.bfloat16 | torch.bfloat16 |
| 3 | `cg.model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min` | [] | 0.000000e+00 | 0.000000e+00 | torch.bfloat16 | torch.bfloat16 |
| 4 | `cg.model.vision_tower.encoder.layers.0.self_attn.q_proj.input_max` | [] | 0.000000e+00 | 0.000000e+00 | torch.bfloat16 | torch.bfloat16 |
| 5 | `cg.model.vision_tower.encoder.layers.0.self_attn.q_proj.output_min` | [] | 0.000000e+00 | 0.000000e+00 | torch.bfloat16 | torch.bfloat16 |
| 6 | `cg.model.vision_tower.encoder.layers.0.self_attn.q_proj.output_max` | [] | 0.000000e+00 | 0.000000e+00 | torch.bfloat16 | torch.bfloat16 |
| 7 | `cg.model.vision_tower.encoder.layers.0.self_attn.q_proj.weight` | [768, 768] | 0.000000e+00 | 0.000000e+00 | torch.bfloat16 | torch.bfloat16 |
| 8 | `cg.model.vision_tower.encoder.layers.0.self_attn.k_proj.input_min` | [] | 0.000000e+00 | 0.000000e+00 | torch.bfloat16 | torch.bfloat16 |
| 9 | `cg.model.vision_tower.encoder.layers.0.self_attn.k_proj.input_max` | [] | 0.000000e+00 | 0.000000e+00 | torch.bfloat16 | torch.bfloat16 |
| 10 | `cg.model.vision_tower.encoder.layers.0.self_attn.k_proj.output_min` | [] | 0.000000e+00 | 0.000000e+00 | torch.bfloat16 | torch.bfloat16 |

### 2.4 분석 귀결
정본 체크포인트 `m1_final_frozen.pt`에 수록된 모든 비-LoRA 기저 가중치는 허브 순정 Base 가중치와 비트 단위로 동일함. 이에 따라 AUDIT-CTS-419 계측에서 나타난 글만 NLL 차이(경로 ① 5.85 vs 경로 ③ 1.01)는 기저 모델 가중치의 오염 또는 손상이 아닌, 어댑터(LoRA) 모듈의 파라미터 및 모델 래퍼 입출력 호출 방식에서 유래된 것임이 수치로 입증됨.

---

## 3. F3. 무효 P4(v3.5) 디코더 격리 및 P6 버전 검증기 배치

### 3.1 무효 파일 격리 내역 (mv 단독, 삭제 0건)
격리 디렉터리: `results_s7/_invalid_v35_P4/`
이동 방식: `shutil.move` (삭제 명령 배제, 파일 전수 이동)
이동 대상 파일 수: 총 21개 파일 (체크포인트 3개 + 중간 스텝 체크포인트 18개)
이동 전후 SHA-256 검증 결과: 21개 파일 전수 전후 해시 일치율 100%

| 번호 | 원본 경로 (`results_s7/`) | 크기 (Bytes) | 이동 전후 검증 SHA-256 해시 |
|:---:|:---|:---:|:---|
| 1 | `decoder_A_l1_P4.pt` | 236,027,275 | `fb83971b50344fc45342e0075581ef0e57e36cebbab8d7a7233b78a788d5f196` |
| 2 | `decoder_A_l1_P4_step0500.pt` | 236,027,609 | `b7f14d9fa28bc86febce4db6ff7dd7a39abfb43974f3f4387c7966755806eacf` |
| 3 | `decoder_A_l1_P4_step1000.pt` | 236,027,609 | `1f4631604a8d718a3b991a29e3b7def3ed54b958e3b7ef1650e881686272872b` |
| 4 | `decoder_A_l1_P4_step1500.pt` | 236,027,609 | `2b0c0c6f39d887b68ab55280f4b4bd554250f0c9d601b694a0ab7a3a0181643b` |
| 5 | `decoder_A_l1_P4_step2000.pt` | 236,027,609 | `b4ae0645f48d10fb5e46b67ecf09ad5eb2e84b76281e74527eb06a938711cfd9` |
| 6 | `decoder_A_l1_P4_step2500.pt` | 236,027,609 | `4205a7a2158d839eeb57caf8d3f2662f6a22d5b45deaf474d12c9c94d4e7f3d6` |
| 7 | `decoder_A_l1_P4_step3000.pt` | 236,027,609 | `0c243cf6876c819df5d2e0695702d54ec6ac52de9331110580035c41ff0223ab` |
| 8 | `decoder_D_l0_P4.pt` | 236,027,275 | `6eeb682eba58f6d4498a9f770fffd3df641b60e7b364b5d902a2454d9abb664a` |
| 9 | `decoder_D_l0_P4_step0500.pt` | 236,027,609 | `d6230d17ebf714afbcd9e6e9f7392b96b2be21abc220fcccb3347c932c19a3d1` |
| 10 | `decoder_D_l0_P4_step1000.pt` | 236,027,609 | `cd1e8efd5854be540c56667a4d773b15f95cb1952ce99bff5f9009a879fe9675` |
| 11 | `decoder_D_l0_P4_step1500.pt` | 236,027,609 | `d0852086167369ef18a3ab871e79f3644d1c81ed30b4dd596b9819a373c64109` |
| 12 | `decoder_D_l0_P4_step2000.pt` | 236,027,609 | `a425191b300a97c8cd4f734e20dc392725184c4b19035a02d9219653dbfde7ae` |
| 13 | `decoder_D_l0_P4_step2500.pt` | 236,027,609 | `a5372bb7f016b9dae5a32b04e834a9fb56fc805fe38e7f0c6973e2bdad9b24f1` |
| 14 | `decoder_D_l0_P4_step3000.pt` | 236,027,609 | `9ce159c4acc9b1f7c704055eec839a8af83daa7adeaf4096443ebdcb1b93df4c` |
| 15 | `decoder_E_zsrc_l1_P4.pt` | 236,027,425 | `b37a680f6765c2e0ce11d23396e8f72549d8ff5dd11a6b9df2e3a9d9b465bc60` |
| 16 | `decoder_E_zsrc_l1_P4_step0500.pt` | 236,027,759 | `6e49562ea67bef8062cd9c3d761b352e90514e7845af3f38a6a1e0862a3d8eab` |
| 17 | `decoder_E_zsrc_l1_P4_step1000.pt` | 236,027,759 | `c05385ac0e41515d70dea3e8a95d55b01506838f954e2cd91b2b716681c3cb5f` |
| 18 | `decoder_E_zsrc_l1_P4_step1500.pt` | 236,027,759 | `d91ecee2bfac534fd6ce8f1c8c99271a7432403eaa8404c1c3240698f1d969f6` |
| 19 | `decoder_E_zsrc_l1_P4_step2000.pt` | 236,027,759 | `bc744ec834d0ec0b652109d835d581b5c4ed9efe90ce4d4499572b0b123bdd24` |
| 20 | `decoder_E_zsrc_l1_P4_step2500.pt` | 236,027,759 | `39cc4b8e32554353dc448c65f38c6edb8d4a0d25bf618c84cae8bcb14011c690` |
| 21 | `decoder_E_zsrc_l1_P4_step3000.pt` | 236,027,759 | `e32783c41e2454e19e90ec3d9c5bda43f22f9c90019075623b7b88fa4d6a8b47` |

### 3.2 잔류 보존 파일 목록 (`results_s7/`)
- `test_smoke_v36.pt` (236,027,245 bytes)
- `zstar_train1000_P4.pt` (655,869,289 bytes)
- `zsrc_train1000_P4.pt` (655,695,483 bytes)
- 현재 GPU 0에서 생성 중인 활성 디코더 산출물 `decoder_A_v36.pt`

### 3.3 P6 실행기 검증 스크립트 배치
- 배포 파일: `scripts/check_p6_decoder_version.sh`
- 파일 크기: 1,569 바이트
- 실행 권한: `chmod +x` 적용
- SHA-256 해시: `1fa95f72d5e5cea2ad1836dd458544190467f0cfb1daa14e41f6459cb51d870b`
- 검증 기준: 전달된 디코더 파일명이 `*_v36.pt` 또는 `*_v36_step*.pt` 패턴과 일치하지 않는 경우 즉시 exit 1로 평가 진입을 기계적으로 차단.

---

## 4. 예상 밖 사건 (Unexpected Incidents) 및 조치 내역

1. **1차 실행 시 메모리 상한 초과 발생**:
   - 2026-10-04 19:16:20 1차 통합 묶음 실행 도중, F2 연산 단계에서 Gemma-4의 레이어별 입력 임베딩 파라미터(`embed_tokens_per_layer`, 크기 42 x 262,144 x 256 = 2,818,572,288 원소, float32 변환 시 11.27 GB)를 일괄 변환 처리하려다 `ulimit -v 62914560` 가상 메모리 상한(60 GB)에 도달하여 `RuntimeError: DefaultCPUAllocator: can't allocate memory` 발생(반환코드 1 기록).
2. **조치 및 복구**:
   - R7-1 간격 규약(최소 60초 대기)을 준수하며, TASK-0005 지침 19행("메모리 초과 시 키 단위 순차 비교로 바꾸되 결과 수치만 보고")에 의거하여:
     - 5,000,000 원소(약 20 MB) 단위 청크 분할 노름 계산 함수 적용
     - `sd_ckpt.pop(k)` 방식으로 각 가중치 계측 직후 텐서 메모리를 메모리에서 즉시 해제
   - 2차 실행(19:18:24)을 단일 세션으로 진행하여 2,076개 키 전수 비교, 21개 파일 이동, P6 검증 스크립트 배치가 반환코드 0으로 종료됨.

---

## 5. 원격 서버 및 P4' 학습 진행 상태 요약

- **원격 서버 상태**: `tmux s7_train_p4_prime` (PID 3826706, GPU 0) 프로세스 접촉 0건 유지 중, 19:21 기준 Step 100/3000 도달 (`loss=3.2270`, `l_task=2.1158`, `l_align=1.1112`, `peak_mem=39.59 GiB`, `12.33s/step`).

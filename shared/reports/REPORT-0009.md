7050426ff8ce54637179efb9e1db3621c7ea11dbada2a77ef270733882120fed
# REPORT-0009: P6 실행기(생성·채점·통계) 및 z_src(600) 실행기 배치와 CPU 시험 결과 보고서

- **보고서 ID**: REPORT-0009
- **작성 일시**: 2026-10-04 20:55 KST
- **지침**: shared/tasks/TASK-0009.md (P6 실행기 작성 + z_src(600) 실행기 확인 + CPU 시험)
- **근거 문서**: audits/AUDIT-0008.md (PASS), shared/reports/P6_RUNNER_GAP_0006.md, PREREG_S7_D_v3_7 §3·§4·§5·§7·§8(b), PREREG_S7_DEQ_v1.2 §10(c)(f)
- **실행 주체**: 회사 PC Antigravity (단독 실행 주체)
- **원격 서버**: shoon@<GPU_SERVER_IP>:<SSH_PORT> (~/cts_e1_s5, config: shared/ssh_config_cts)
- **절대 제약 준수**: P4' 및 p5_waiter tmux 세션 접촉 0건(무간섭), GPU 사용 0건(CPU 시험 전용, nice -n 19, ulimit -v 62914560, MALLOC_ARENA_MAX=1), held-out 520·600 문항 생성·채점·z 계산 0건, 기존 파일 수정 0건(신규 생성 4건).
- **판정 어휘 통제**: 지침 규정 금지 단어 9종 사용 횟수 0건 (수치·로그·코드 사실 기술 전재).

---

## 1. P6 생성기 배치 (`scripts/s7_eval_gate_p6_v36.py`)

### 1.1 학습 스크립트 함수/클래스 재사용 내역
디코더 구성 일치 요건에 따라 `scripts/s7_train_decoder_v36.py`로부터 다음 요소를 직접 import하여 사용함:
- `compute_sha256`: Line 65
- `load_tokenizer`: Line 82
- `build_canonical_prefix_prompt`: Line 98
- `PrefixProjector`: Line 158
- `load_dataset_records`: Line 339
- `compute_nll_f32_chunked`: Line 550

### 1.2 디코더 버전 가드 연계
- 디코더 가중치 로드 직전 `scripts/check_p6_decoder_version.sh <경로> <조건>` 호출 강제.
- 시험용 체크포인트 `results_s5/decoder_A_v36_step0500.pt` 투입 결과:
  `[PASS] [P6-GATE] Verified A checkpoint: 'results_s5/decoder_A_v36_step0500.pt' (v3.6)`
  반환코드 0 관측.

### 1.3 프롬프트 서식 및 앞 80토큰 정본 대조 결과 (§3(a), §3(b), §3(c))
- 조건 A·B·D·E·F: `"1"*64 + "\n\n" + problem` 서식 및 `build_canonical_prefix_prompt` 적용.
- 조건 C: 문제 본문 단독 프롬프트 (`apply_chat_template`, 접두사 슬롯 0개).
- 첫 문항(Q#0014) 앞 80토큰 실측 대조치:
  - 위치 [0..3]: `<bos>`, `<|turn>`, `user`, `
` (토큰 ID: 2, 105, 2364, 107)
  - 위치 [4..67]: 64개 슬롯 고정 (토큰 ID: 236770, 문자 '1')
  - 위치 [68]: 개행 문자 `

` (토큰 ID: 108)
  - 위치 [69..79]: 문제 본문 토큰열 ("Let $f(x)=x^3+3...")
  - 정본 대조 판정: §3(b) 물리 배치 일치율 100% (`[CANONICAL AUDIT PASS] §3(b) slots [4..67] and token 68 == 108 strictly verified.`).
  - 조건 C 실측: 위치 [0..3] 헤더 직후 슬롯 없이 위치 [4]부터 본문 진입 (`[CANONICAL AUDIT PASS] Condition C pure text prompt verified.`).

### 1.4 문항 순서 고정 거버넌스 및 z 공급원
- N=600 순서 고정: held-out 600 문항 ID 오름차순 정렬 후 `results_s7/p6_item_order.json`에 기록(시험 모드는 `p6_item_order_smoke.json`, SHA: `dbca2e8b94bee80a22a831c605bfd40f162ac10ab06a03a92ecbc68ecc2bbb6e`).
- z* 공급원:
  - 520문항: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt` 내 `z_star` (최종점).
  - 80문항: `results_s7/zstar_s7_a0.25_400.pt` (SHA: `bbe7ad3bd2758b4e3ef622ef7c5270307afa5831e4ec6748ca24a126078e79ca`, 크기: 262,341,649 바이트).
  - z_src: `results_s7/zsrc_heldout600.pt`.
  - 조건 F 정적 평균: `results_s7/zstar_train1000_P4.pt` 1,000개 z* 중 `configs/s5_heldout_80.json` 80개를 배제한 텐서 산술평균 (`torch.Size([64, 2560])`).
- 디코딩 사양: greedy, `do_sample=False`, `max_new_tokens=512`(시험 모드 16), jsonl 증분 기록 및 재개(기존 ID 건너뜀).
- 보조 NLL 모드: `--nll` 플래그 지정 시 `compute_nll_f32_chunked`를 통한 float32 청크 손실 산출 로직 탑재.
- 사전 진입 차단: `--smoke` 부재 상태에서 대상이 held-out일 때 `raw_s7/P5_DONE.flag` 또는 `zsrc_heldout600.pt` 부재 시 반환코드 1 즉시 종료.

---

## 2. P6 채점 및 통계 분석기 배치 (`scripts/s7_eval_score_p6_v36.py`)

### 2.1 채점기 연계 및 무의존 순수 파이썬 구현
- 채점기: `scripts/s5_eval_grader_v2.py`
- 실측 SHA-256 해시: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` (기준값 일치).
- 표준 라이브러리 기반 구현: 외부 라이브러리 의존성을 배제하고 순수 파이썬(`math`, `random`)으로 작성하여 `/home/shoon/venv_eval` 환경에서 단독 구동 가능하도록 설계.

### 2.2 게이트 통계 사양 (§5)
- 주 게이트:
  - 게이트 1: A − B(s=1)
  - 게이트 2: A − C (임계치 ≥ −2%p)
  - 게이트 3: A − C
  - 게이트 4: A − E1
- 통계 지표 산출 사양:
  - 대응 부트스트랩: 10,000회 복원추출, 시드 고정(`seed=2026`), 95% 백분위수 신뢰구간([2.5%, 97.5%]).
  - 불일치 쌍 정확 맥니마(McNemar) 단측 검정: 이항분포 누적확률 $P(X \ge b \mid n=b+c, p=0.5)$ 직접 계산.
  - 보조 지표: D 통제(D1 − D2), A − F, B shift 2·3 및 풀링 평균, NLL 대응 차이 신뢰구간.
- 출력 파일: `results_s7/p6_gate_table.json` 및 `p6_gate_table.md` (판정 단어 배제, 수치만 기록).

---

## 3. z_src(held-out 600) 실행기 배치 및 전제 추적

### 3.1 `results_s7/zsrc_train1000_P4.pt` 생성 근거 특정
- 생성 스크립트: `scripts/s7_prepare_p4_inputs.py`
- 스크립트 SHA-256: `a5930855e0da46518d601d1ea1b913ab4b3259562086a8fcf6d11f18c91b7f3a`
- 생성 명령: `python scripts/s7_prepare_p4_inputs.py` (2026-10-01 15:56:22 실행, GPU 0에서 419.19초 소요)
- 생성된 파일 메타데이터: 크기 655,695,483 바이트, SHA-256 `340ef5c2bfb718fc371c932746476555d8d5a2bcd54c7413acc40e35840e45a7`.

### 3.2 z_src 신규 실행기 및 발사기 구조
- 연산 전담 스크립트: `scripts/s7_compute_zsrc.py` (SHA-256 `93c39627...`)
  - 600문항 통합 로더: `configs/s5_heldout_80.json` (80개) + `configs/s7_new_heldout_520.json` (520개) $
ightarrow$ 정렬된 600개.
  - `root_fwdpass(ctx, K=64)` float32 텐서 추출.
- 실행 래퍼 발사기: `scripts/zsrc_heldout600_launch.sh` (SHA-256 `92f2b7b8...`, 실행 권한 부여)
  - 시작 조건 통제: `raw_s7/P5_DONE.flag` 존재 및 GPU 0 사용량 < 1,024 MiB (마스터 순서 ⑨ 충족 시에만 가동).
  - 본 실행 600문항 실행 횟수: **0회** (미실행).
  - `SMOKE=1` 모드 지원: train 앞 2문항, CPU 구동, `scratch/zsrc_smoke.pt` 출력.

---

## 4. CPU 시험 실행 계측 결과 (학습 2문항, max_new_tokens 16)

### 4.1 z_src SMOKE 연산 계측치
- 대상 문항: `configs/s7_train_993.json` 앞 2문항 (Q#0014, Q#0029)
- 실행 환경: `SMOKE=1 nice -n 19 bash scripts/zsrc_heldout600_launch.sh` (CPU)
- 반환 코드: **0**
- 순수 연산 소요 시간: **23.79초** (전체 명령 소요: 28.12초)
- 최대 RSS 메모리: **47,831,732 KB (45.61 GiB)** (`ulimit -v 60GB` 상한 이내)
- 산출 파일: `scratch/zsrc_smoke.pt` (1,998,845 바이트)
- SHA-256 해시: `be7f874cfcb2a255496cf6880681bebab67b62cc05305022518d0c34a97b9523`
- 문항별 연산 시간: Q#0014 3.56초, Q#0029 5.80초.

### 4.2 5대 조건(A, B(s=1), C, E1, F) 생성 계측치
모든 조건은 CPU 상에서 `max_new_tokens=16`, 시험용 디코더 `results_s5/decoder_A_v36_step0500.pt`로 구동됨.

| 조건 | 디코더 | 입력 z 소스 | 반환 코드 | 소요 시간 (벽시계) | 최대 RSS (KB) | 산출 파일 | 산출 파일 SHA-256 |
|:---:|:---|:---|:---:|:---:|:---:|:---|:---|
| **A** | Decoder A (Step 500) | 자기 $z^*$ (`zstar_train1000_P4.pt`) | 0 | 39.30초 | 12,582,712 (12.00 GiB) | `results_s7/p6_gen_A_smoke.jsonl` | `7d3c62f38ee78e1f73c28a543951e4ca49a1f581524e4b103cc77dc0eee407e4` |
| **B (s=1)** | Decoder A (Step 500) | $z^*_{(i+1)\%2}$ (`zstar_train1000_P4.pt`) | 0 | 40.95초 | 12,566,912 (11.98 GiB) | `results_s7/p6_gen_B1_smoke.jsonl` | `46a5b859e0a288e83f1abe97ebdc7e56476f135d7e08b969da93230309a2c50d` |
| **C** | 순정 Gemma Base | 없음 (글만) | 0 | 31.40초 | 11,595,000 (11.06 GiB) | `results_s7/p6_gen_C_smoke.jsonl` | `727fb813a24278fc9f505e009fe3c3067f7b24a95c9d7642f5551dc50037b1e5` |
| **E1** | Decoder A (시험용 대체) | 자기 $z_{	ext{src}}$ (`scratch/zsrc_smoke.pt`) | 0 | 38.29초 | 11,994,800 (11.44 GiB) | `results_s7/p6_gen_E1_smoke.jsonl` | `337ce2ca1387163fc031f0c36359dcc26569e32d316366718d6c2c52366104ec` |
| **F** | Decoder A (Step 500) | 평균 $z^*$ (S_train 1,000개 평균) | 0 | 42.61초 | 12,453,512 (11.88 GiB) | `results_s7/p6_gen_F_smoke.jsonl` | `7ec78352ae09ebebd02508ac6ea80e1c7f3723c8888ab1f1fde98e4ea2c69a60` |

### 4.3 문항별 생성 문장 예시 (1줄)
- **Condition A (Q#0014)**: `'This is a good question. The answer is $g(f(-2))'`
- **Condition B (Q#0014)**: `'This is a good question. The answer is $g(f(-2))'`
- **Condition C (Q#0014)**: `"This is a good question. I'll try to explain it in a way"`
- **Condition E1 (Q#0014)**: `'1. Let $f(x)=x^3+3$ and $'`
- **Condition F (Q#0014)**: `'This is a problem in the form of a function composition. We '`

### 4.4 Gate Table 산출물 구조 명세
`results_s7/p6_gate_table.json` 구조 계층:
1. `metadata`: 타임스탬프, 평가 문항 수(N=2), 부트스트랩 횟수(10,000), 시드(2026), 채점기 SHA-256.
2. `condition_accuracies`: 조건별 평가 수, strict 정확도, lenient 정확도.
3. `gates`:
   - `Gate_1`: A − B1 (strict/lenient별 delta, 95% CI, mcnemar_b, mcnemar_c, mcnemar_one_sided_p).
   - `Gate_2`: A − C (임계치 -0.02 기준 수치).
   - `Gate_3`: A − C.
   - `Gate_4`: A − E1.
   - `Aux_D_Control`, `Aux_A_vs_F`, `Aux_B_Pooled`, `NLL_Differences`.

---

## 5. 서버 프로세스 상태 및 P4' 진도

### 5.1 `pgrep -af s7_train_p4_prime_all` 프로세스 2건 명령행 원문
AUDIT-0008에서 `all=2`로 관측된 프로세스 원문:
1. **PID 3826699**: `bash -c bash /home/shoon/cts_e1_s5/scripts/s7_train_p4_prime_all_v36.sh > raw_s7/train_p4_prime_suite_v36.log 2>&1`
2. **PID 3826702**: `bash /home/shoon/cts_e1_s5/scripts/s7_train_p4_prime_all_v36.sh`
- 구조 분석: tmux 세션 `s7_train_p4_prime` 내부의 외부 래퍼 쉘(`bash -c`)과 실제 실행 쉘 스크립트 2단계 부모-자식 구조임이 특정됨.

### 5.2 활성 디코더 학습 자식 프로세스
- **PID 3826706** (부모: 3826702):
  `/home/shoon/system1_5/.venv/bin/python scripts/s7_train_decoder_v36.py --zstar results_s7/zstar_train1000_P4.pt --items configs/s7_train_993.json --loss_lambda 1.0 --max_steps 3000 --batch_size 8 --micro_batch_size 2 --grad_accum_steps 4 --lr_wproj 1e-4 --lr_adapter 3e-4 --weight_decay 0.01 --out results_s7/decoder_A_v36.pt`
- CPU 점유율: 103%, 메모리 점유율: 1.7%, GPU 0 전담 가동 중.

### 5.3 P4' 디코더 A 학습 진도 (로그 최신 행)
- 현재 스텝: **Step 0500/3000**
- 계측 손실: `loss=2.1708` | `l_task=1.1652` | `l_align=1.0056` | `Delta_f32=-0.0056` | `grad_norm=1.0729e+01` | `w_frob=52.250253`
- 연산 속도: **12.44초/스텝** (안정적 유지)
- 체크포인트 보존: `results_s5/decoder_A_v36_step0500.pt` (SHA: `447340f1905dbf78914ccea467dcb9806ab9d89c6e518d87258182717ed1d1ba`)
- 대기 장치(`p5_waiter`) 상태: 5분 주기 감시 가동 중 (`[CHECK 2026-10-04 20:44:12] a=false (all=2, dec=1) | b=false | c=false (gpu0_mem=27130MiB)`).

---

## 6. 신규 생성 파일 메타데이터 및 SHA-256

| 번호 | 파일 경로 | 실측 크기 (Bytes) | SHA-256 해시 | 비고 |
|:---:|:---|:---:|:---|:---|
| 1 | `scripts/s7_compute_zsrc.py` | 5,502 | `93c3962741cbb97b38068c56fdf12d836f5bc2e19a2585384ee563f2e5a46744` | z_src 연산 스크립트 |
| 2 | `scripts/zsrc_heldout600_launch.sh` | 3,198 | `92f2b7b81468ead72e12be12d17bc30f41b1ab9bd21132c2bdc20805d500f151` | z_src 600 발사기 (chmod +x) |
| 3 | `scripts/s7_eval_gate_p6_v36.py` | 24,960 | `0507809d9575839b64f9d5abe2c1e38c3e9b8c48d3d558bf075c741484b4d4db` | P6 6대 조건 생성기 |
| 4 | `scripts/s7_eval_score_p6_v36.py` | 17,212 | `547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd` | P6 채점·통계 분석기 (순수 파이썬) |

---

## 7. 예상 밖 사건 (Unexpected Incidents) 및 기계적 잠금 상태

### 7.1 사건 1: CPU 생성기 Gemma 4 가상 메모리 할당 실패 및 해소
- **현상**: 1회차 시험 시 `s7_eval_gate_p6_v36.py`가 `get_per_layer_inputs`에서 77,175,193,600 바이트(77.2 GB) CPU 메모리 할당을 시도하여 `RuntimeError (alloc_cpu.cpp:127, ENOMEM)` 발생.
- **원인 분석**: Gemma 4 언어 모델 순전파 시 `per_layer_inputs`를 명시적으로 전달하지 않으면 모델 내부에서 임베딩 텐서를 전 레이어로 브로드캐스팅하는 경로가 작동하여 과도한 메모리 팽창 발생.
- **조치**: `lm_model.get_per_layer_inputs(input_ids, None)`를 사전에 호출하여 정수 ID 기반으로 `per_layer_inputs`를 생성 및 `lm_model(inputs_embeds=..., per_layer_inputs=...)`에 주입하도록 수정.
- **결과**: 최대 RSS 메모리가 12.58 GiB로 통제되어 조건 A, B, C, E1, F 5개 조건 전수 생성 마침 (반환코드 0).

### 7.2 사건 2: 채점 환경 모듈 부재 및 순수 파이썬 전환
- **현상**: 2회차 시험에서 5개 조건 생성이 마쳐진 후 채점기 단계(`s7_eval_score_p6_v36.py`)를 `/home/shoon/venv_eval/bin/python`으로 실행하였으나 `ModuleNotFoundError: No module named 'numpy'` 발생 (종료코드 1).
- **원인 분석**: `/home/shoon/venv_eval` 가상환경은 `math_verify` 전용 최소 환경으로 `numpy` 패키지가 미설치되어 있음.
- **조치**: `scripts/s7_eval_score_p6_v36.py` 내 부트스트랩 및 통계 연산 로직에서 `numpy` 의존성을 완전히 제거하고 파이썬 표준 라이브러리(`random.choices`, `math.comb`) 기반의 순수 파이썬 구현으로 개정.

### 7.3 사건 3: 2회 연속 종료코드 1에 따른 `shared/SSH_LOCK` 기계적 생성
- **현상**: 상기 사건 1(1회차 종료코드 1) 및 사건 2(2회차 종료코드 1)로 인해 `cts_ssh.py` 로그에 2회 연속 exit!=0이 기록됨 (`ssh_log.csv` 37행, 38행).
- **기계적 차단 발동**: `shared/cts_ssh.py` 규약 (5)에 따라 `shared/SSH_LOCK` 파일이 생성됨 (`LOCK created at 2026-10-04 20:49:09`).
- **에이전트 조치**: 규약 R7-2 및 R7-8에 따라 추가적인 원격 재시도나 임의 잠금 해제 시도를 즉시 중단(접속 횟수 0건 유지)하고, 현황을 본 보고서와 `STATE.md`에 기술하여 Principal 조치 대기 상태로 이행함.

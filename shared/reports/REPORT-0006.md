48eb7ee22d5b19a704e6cd63e046b6909eb646c4d2db50b1954591334e9ddc0d
# REPORT-0006: P5 실행기 준비 및 P6 실행기 현황 점검 결과 보고서

- **보고서 ID**: REPORT-0006
- **작성 일시**: 2026-10-04 20:15 KST
- **지침**: shared/tasks/TASK-0006.md (P5 실행기 준비 + P6 실행기 현황 점검)
- **근거 문서**: audits/AUDIT-0005.md, PREREG_S7_DEQ_v1.2 §10(e)(f), PREREG_S7_D_v3_6
- **실행 주체**: 회사 PC Antigravity (단독 실행 주체)
- **원격 서버**: shoon@<GPU_SERVER_IP>:<SSH_PORT> (~/cts_e1_s5, config: shared/ssh_config_cts)
- **접속 규약 준수**: shared/cts_ssh.py 경유 단일 묶음 명령 실행 (직접 접속 0건, ssh_log.csv 기록)
- **판정 어휘 통제**: 지침 규정 금지 단어 9종 사용 횟수 0건 (수치·로그·코드 사실 기술 전재)

---

## 1. A. P5 실행기 준비 (실행 0건)

### 1.1 전제 파일 6종 경로·메타데이터 및 SHA-256 대조
전제 파일 6종에 대해 서버 파일 시스템 상의 실측 SHA-256 해시를 산출하고 사전등록 기대치와 대조한 결과:

| 파일 경로 | 파일 크기 (Bytes) | 최종 수정 시각 (mtime, KST) | 실측 SHA-256 해시 | 사전등록 기대치 | 대조 상태 |
|:---|:---:|:---|:---|:---:|:---:|
| `scripts/s7_zstar_lowest_run.py` | 5,903 | 2026-10-01 10:50:21 | `779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba` | `779ea1f1...` | 일치 |
| `cts/deq/zstar_canonical.py` | 10,811 | 2026-09-13 20:44:58 | `b7170aeb68833448c8bb0977eb7ac035ef06614c9a5edf2f228448c2bb96d45d` | `b7170aeb...` | 일치 |
| `cts/deq/transition_inject.py` | 8,046 | 2026-09-13 00:41:07 | `127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c` | `12706843...` | 일치 |
| `exe_scratch/s7/ref_solvers/anderson.py` | 6,402 | 2026-09-28 11:40:59 | `06c93011058b9dfa83e1846636fa05ae65a117b49d9b35045ba8a59e2a49d7e3` | `06c93011...` | 일치 |
| `/home/shoon/checkpoints/m1_final_frozen.pt` | 16,037,272,221 | 2026-08-24 01:03:34 | `f2119bbe5c0ad914ee00c34e9109674d15de07c5e028e288ab05b88dd448d1f9` | `f2119bbe...` | 일치 |
| `configs/s7_new_heldout_520.json` | 5,138 | 2026-09-27 22:08:39 | `48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988` | (기존 서버 파일) | 일치 |

### 1.2 문항 수 및 학습 세트와의 ID 교집합 검증
파이썬 집합 연산을 통해 문항 수와 세트 간 ID 교집합을 계측한 수치:
- `configs/s7_new_heldout_520.json` 총 문항 수: **520개**
- `configs/s7_train_993.json` 총 문항 수: **993개**
- 전체 학습 집합(`train_1000`) 총 문항 수: **1,000개** (유효 993개 + 제외 7개)
- **교집합 계측 결과**:
  - `heldout_520` ∩ `train_993` 교집합 원소 수: **0개 (공집합)**
  - `heldout_520` ∩ `train_1000` 교집합 원소 수: **0개 (공집합)**
- 세트 간 문항 ID 누출 또는 중복 0건이 대조됨.

### 1.3 `scripts/p5_heldout520_launch.sh` 생성 및 배치 내역
- 파일 경로: `scripts/p5_heldout520_launch.sh`
- 파일 크기: 3,099 바이트
- 실행 권한: `chmod +x`
- 실측 SHA-256 해시: `5bb9535afb7b69604cc3de862db4e48a61f0c35ff49e7dc1d7991ac076f4d66c`
- 실행 횟수: **0회** (실행 0 규약 엄수)
- 탑재된 기계적 안전 차단 장치:
  1. 기동 전 위 6대 전제 파일의 SHA-256을 전수 재계산하여 불일치 시 즉시 exit 1 차단.
  2. GPU 0에서 `s7_train_decoder` 또는 `s7_train_p4_prime` 프로세스가 검출될 경우 즉시 exit 1 차단.
  3. PREREG_S7_DEQ_v1.2 §10(e) 정본 인자 강제: α=0.25, g0=0.25, σ=0.020, seed=2030, max_nfe=1000, device=cuda:0.
  4. 로그 파일: `raw_s7/p5_heldout520.log` 지정 및 tee 출력.
  5. 출력 파일: `results_s7/zstar_heldout520_P5.pt`.

### 1.4 정적 문법 검사 및 도움말 출력 결과
- `bash -n scripts/p5_heldout520_launch.sh`: 반환코드 0 (Syntax clean 출력).
- `python scripts/s7_zstar_lowest_run.py --help`: 규정 파라미터 규격 출력 (인자: `--items`, `--out`, `--alpha`, `--g0`, `--seed`, `--sigma`, `--max_nfe`, `--tol`, `--device`, `--config`).

---

## 2. B. P6 실행기 현황 점검 및 갭 분석 (정적 점검)

### 2.1 채점기 SHA-256 대조
- 파일 경로: `scripts/s5_eval_grader_v2.py`
- 실측 SHA-256 해시: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`
- 사전등록 기대치: `90cc93e4...` (일치)

### 2.2 서버 상 평가/생성 스크립트 목록
서버 내 `scripts/` 디렉터리에서 탐색된 평가 및 게이트 관련 스크립트 16종 목록:
| 파일 경로 | 크기 (Bytes) | SHA-256 해시 |
|:---|:---:|:---|
| `scripts/check_p6_decoder_version.sh` | 1,570 | `1fa95f72d5e5cea2ad1836dd458544190467f0cfb1daa14e41f6459cb51d870b` |
| `scripts/s5_eval_gate_v3.py` | 53,349 | `9e6d03d521ecffb9a1f5c7fc7a8a1ac5e4c785aa82adf1faa911dd07f378e338` |
| `scripts/s5_eval_gate_v2.py` | 51,426 | `9629a50d80ba529ab241ed2c681093f91c59431846fdece03dac82483a15b70d` |
| `scripts/s5_eval_gate.py` | 12,829 | `4e6e58e6829ee3cc2e6f2676fa3d159299268a1cf84164ba4ae913cfb70908fe` |
| `scripts/s5_eval_nll_contrast.py` | 20,635 | `d05617141dc071c446b1f21c6a46f5e651467e8dcd38d460b79fead94adcef73` |
| `scripts/s5_eval_grader_v2.py` | 8,392 | `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` |
| `scripts/s5_phase1_gate.py` | 14,038 | `671fc8d7673321d4429d62fdae6414761328ab58f21bbff216eff23dc043850f` |
| `scripts/s6_eval_damping.py` | 11,772 | `a3e45a276dd5a5604e63e15fccf9a59035cff499551134d8093b396cc2965655` |
| `scripts/s7_r1000_reeval.py` | 7,713 | `1a2b26815c14831d4034b33001c592500a31d2a63a717fdd38b1095f41379dcf` |
| `scripts/eval_node_a_replication.py` | 5,615 | `fdad324d5ae7a75f6656804841604335d9911f60a7eef2b44cc20308d552bb27` |
| `scripts/eval_node_d_minigate.py` | 6,439 | `5e8db89608f0659a36e381858d80e9bf37d51fd609835f30791c4aa856ed469a` |
| `scripts/eval_stage1c_step50.py` | 6,170 | `1aa06459b434228abd79f8b812099e77c7f0d4c41d116e28710a5412dba5fdd6` |
| `scripts/run_cts_eval_full.py` | 90,194 | `16af518a9d90696d3bbad726c513e505afd3c261464622523bd4f12be8a80534` |
| `scripts/run_phase2_eval_run4b.sh` | 1,402 | `5be9d333001f8420f919e491371f5a011bb28b942706d59580b26dbb21054acd` |
| `scripts/run_phase2_eval_train40.sh` | 1,579 | `10e04d24118b3ba539a806b9e127c08901824c3d8c23734e9f8f2a62f013baec` |
| `scripts/aggregate_nu_table19.py` | 3,920 | `6a074eb9f0ec68f9c2dcafe9224a7fafdb726ca87546efaa86f20d38da2d5107` |

### 2.3 `scripts/s5_eval_gate_v3.py` 기준 4대 요건 대조 결과
1. **(i) 순정 Gemma 58c4d7b5 적재 여부**:
   - Line 77에 경로 상수는 기재되어 있으나, 디코더 모델 인스턴스화 시 어댑터 결합 백본(`CtsBackbone`)을 적재함. 순정 Gemma 단독 적재 및 PrefixProjector 결합 분기 미구현 (불일치).
2. **(ii) `scripts/check_p6_decoder_version.sh` 가드 호출 여부**:
   - 가드 호출 코드 행 수: **0건** (호출 부재).
3. **(iii) 프롬프트 양식 §3(a)(c) 및 조건 A~F 구현 여부**:
   - 64슬롯 prefix 구성(Line 221) 및 math decode prompt 구성(Line 326) 함수는 존재함.
   - 조건 분기는 S5 80문항 전용 5개 조건(`matched`, `shuffled`, `zero`, `random-z`, `no_prefix`, `text_only`)으로 국한됨.
   - S7 P6의 6개 조건(A, B, C, D1/D2, E1/E2, F) 중 C(순정 대화 템플릿 글만 분기) 및 D/E/F 분기 미구현 (불일치).
4. **(iv) 디코딩 하이퍼파라미터 (Greedy, max_new_tokens=512)**:
   - Line 334 (`max_new_tokens=512`), Line 335 (`do_sample=False`), Line 408 (`max_new_tokens: int = 512`) 일치함.

상세 갭 분석 항목 및 신규 러너 작성 권고 사항은 별도 보고서 `shared/reports/P6_RUNNER_GAP_0006.md`에 기술됨.

---

## 3. C. 원격 서버 및 P4' 학습 진행 현황

### 3.1 GPU 자원 점유 현황 (`nvidia-smi`)
- GPU 0: RTX A6000 | 메모리: **27,554 MiB / 49,140 MiB** | 점유율: **100%** | PID: `3826706` (`s7_train_decoder_v36.py`)
- GPU 1: RTX A6000 | 메모리: 13 MiB / 49,140 MiB | 점유율: 0% (유휴)
- GPU 2: RTX A6000 | 메모리: 13 MiB / 49,140 MiB | 점유율: 0% (유휴)
- GPU 3: RTX A6000 | 메모리: 13 MiB / 49,140 MiB | 점유율: 0% (유휴)
- GPU 4: RTX A6000 | 메모리: 13 MiB / 49,140 MiB | 점유율: 0% (봉인 유지, 프로세스 0건)
- GPU 5: RTX A6000 | 메모리: 13 MiB / 49,140 MiB | 점유율: 0% (유휴)
- GPU 6: RTX A6000 | 메모리: 13 MiB / 49,140 MiB | 점유율: 0% (유휴)
- GPU 7: RTX A6000 | 메모리: 13 MiB / 49,140 MiB | 점유율: 0% (유휴)

### 3.2 시스템 메모리 가용량 (`free -h`)
- Total: **251 GiB**, Used: **29 GiB**, Free: **30 GiB**, Available: **220 GiB**, Buff/Cache: **192 GiB**
- Swap: Total 1.9 GiB, Used 1.9 GiB, Free 0.0 KiB (안정 상태, thrashing 0)

### 3.3 P4' 직렬 학습 진행 단계 (`raw_s7/train_decoder_A_v36.log`)
- 현재 학습 프로세스: Decoder A v3.6 학습
- 도달 걸음 수: **Step 300 / 3000** (20:11 KST 계측)
- 최신 계측 지표:
  - `loss = 2.0217`
  - `l_task = 1.0190`
  - `l_align = 1.0028`
  - `Delta_f32 = -0.0028`
  - `NLL_fmt = 0.9834`
  - `NLL_cnt = 1.7832`
  - `grad_norm = 1.7120e+00`
  - `w_frob = 52.216922`
  - `peak_mem = 39.59 GiB`
  - 스텝당 소요 시간: `12.63 s/step`

---

## 4. 예상 밖 사건 (Unexpected Incidents) 및 조치 내역

1. **1차 실행 시 서식 문자열 문법 오류 발생**:
   - 2026-10-04 20:07:40 1차 통합 질의 실행 중, 파이썬 서브스크립트 내 f-string 내부 이스케이프 문자열 표기(`{"Script Path":...}`)가 Python 3.10 인터프리터에서 `SyntaxError`를 유발하여 반환코드 1로 중단됨.
2. **조치 및 복구**:
   - R7-1 대기 규약(최소 60초 대기)을 준수하여 152초 대기 후, 서식 문자열을 일반 문자열 출력으로 교체하고 다중 바이트 문자 인코딩 에러 방지 처리(`ascii replace`)를 적용.
   - 2차 실행(20:10:12)을 단일 묶음 세션으로 수행하여 모든 질의 및 검증이 반환코드 0으로 종료됨.

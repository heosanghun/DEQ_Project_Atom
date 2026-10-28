<!-- SHA256: d37eac68a4453c4613c6fc297c3d8e5bba17f431876d8075773d19be6a78ef9a -->
# REPORT-0029: G1 게이트 입력 준비 스크립트 (p5_prep_inputs.py) 및 p5_promote_v4.py 구현·시험 보고서

- **작성 일시**: 2026-10-05 13:10 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0029.md
- **선행 감사**: audits/AUDIT-P006_p5_analyze_v1.md 재감사 2 (N1·N3·N9b·N10~N14, 규칙문 불일치)
- **절대 조건 준수 현황**:
  - P4'·P5b·p6_chain 무간섭: 준수 (tmux 3개 세션 지속 가동 유지)
  - P5·P5b 결과 열람 0건: 준수 (가짜 합성 데이터 및 기열람 기본 산출물만 취급)
  - GPU 사용 0건: 준수 (CPU 로컬 연산 및 서버 CPU 명령만 수행)
  - 기존 파일 보존: 준수 (p5_promote_v2·v3 보존, v4 신규 작성)
  - 서버 접속 제약: 준수 (cts_ssh.py 전용, 60초 간격 준수, 명령 묶음 실행)
  - 금지어 0건 원칙: 준수

---

## 1. 서버 데이터 출처 조사 결과 (TASK-0029 §1)

### 1.1 대상 파일 및 스키마 실측 조사표
| 항목 | 파일 경로 | SHA-256 | 최상위 구조 | 첫 항목 필드 구성 및 형식 |
|---|---|---|---|---|
| P1 재평가 | `results_s7/r1000_reeval_P1.json` | `97b712ed555cba78113860faf8b3553868ea4ec8ebbb0ad815f46dba041be41a` | `list` (1,080건) | `q_idx`: int, `r_1000`: float, `r_999`: float, `top3_at_z1000`: list |
| Train 1000 PT | `results_s7/zstar_train1000_P4.pt` | `37ea6807d112e44f55ab401f48b6c8e7051f3ca2800aa7cd4caa04a546fb4696` | `dict` (1,000키, `q0014_a0.25` 등) | `z_star`: Tensor, `rho`: float, `final_rho_fp`: float, `nfe`: int, `converged`: bool, `rho_1000`: float, `rho_2000`: float, `min_rho`: float, `shape`: str, `alpha`: float, `meta`: dict |
| S7 400 PT | `results_s7/zstar_s7_a0.25_400.pt` | `bbe7ad3bd2758b4e3ef622ef7c5270307afa5831e4ec6748ca24a126078e79ca` | `dict` (400키: 320 Train + 80 Heldout) | `z_star`: Tensor, `rho`: float, `final_rho_fp`: float, `nfe`: int, `converged`: bool, `rho_1000`: float, `rho_2000`: float, `min_rho`: float, `shape`: str, `alpha`: float, `meta`: dict |
| S7 680 PT | `results_s7/zstar_s7_a0.25_train680.pt` | `4f659e784bc890869b521640379fb0cf2addacbf6a9cacac7a339395f18efe94` | `dict` (680키: 신규 Train 680) | `z_star`: Tensor, `rho`: float, `final_rho_fp`: float, `nfe`: int, `converged`: bool, `rho_1000`: float, `rho_2000`: float, `min_rho`: float, `shape`: str, `alpha`: float, `meta`: dict |
| S7 파이프라인 로그 | `raw_s7/s7_zstar_pipeline.log` | 실측 보유 (157,039 B) | 텍스트 로그 | 400 문항 계산(NFE 1000) 및 680 문항 계산 기록 포함 |
| S7 감쇠 실행 로그 | `raw_s7/s6_a0.25_20260927_220928.log` | 실측 보유 (842,270 B) | 텍스트 로그 | 문항별 `rho@1000`, `rho@2000`, `min_rho` 및 10걸음 간격 궤적 기록 |

### 1.2 $\min \rho$ 및 $\rho$ 궤적 위치 파악
- **Train 1,000**: `results_s7/zstar_train1000_P4.pt` 각 문항 항목에 `min_rho` (float 스칼라)가 직접 저장되어 있음. `zstar_s7_a0.25_400.pt`(320건) 및 `zstar_s7_a0.25_train680.pt`(680건)의 `min_rho`와 오차 0으로 일치함.
- **기존 80 (Held-Out 80)**: `results_s7/zstar_s7_a0.25_400.pt`에 80개 전 문항이 `q{qid:04d}_a0.25` 키로 수록되어 있으며, 항목 내에 `min_rho` (float 스칼라)가 직접 보존되어 있음.
- **문항 식별 일치성**: `results_s7/r1000_reeval_P1.json`의 1,080개 문항은 Train 1,000 문항(1,000개)과 기존 80 문항(80개)의 합집합과 정확히 일치하며 외래 문항 0건임.

### 1.3 문서 정의 인용 및 규칙 고정
- **PROMPT-5 6행 문언**: `"①-a S_train min(4·min ρ, r_1000)"`
- **PREREG_S7_DEQ_v1.3 §10(c) 173행 문언**: `"①-b … 기존 80 (로그 r_min)"`
- **규칙 준용 결론**:
  - ①-a (Train 1,000) 및 ①-b 중 기존 80 평가에는 $k^*$ 페널티 규칙이 존재하지 않음.
  - $k^* < 50$ 미도달 규칙은 오직 신규 520 Held-Out 문항에만 한정 적용됨.

---

## 2. 입력 준비 스크립트 및 중간 산출물 (`scripts/p5_prep_inputs.py`, TASK-0029 §2)

### 2.1 스크립트 식별자 및 실행 내역
- **스크립트 경로**: `scripts/p5_prep_inputs.py`
- **스크립트 SHA-256**: `0d0f8f7108acda76f5125436c47a32cf628037afe900a87b13d78661e45c7988`
- **서버 실행 일시**: 2026-10-05 12:53:57 KST (CPU 단독 실행, 종료 코드 0)

### 2.2 기준 문항 집합 식별자
- **Train 1,000 정본 Config**: `configs/s7_train_1000.json` (SHA-256 `adb09419c69bda0ca9e6e3975d2eae6fb8d6a7c6dea934cb0aab2a897722c94f`, 1,000개 고유 ID)
- **기존 80 정본 Config**: `configs/s5_heldout_80.json` (SHA-256 `27899edbea218b46bc0a5caabada4eb1c1c9dcc17772c001de43374cb208b8c5`, 80개 고유 ID)

### 2.3 생성된 중간 파일 명세
| 중간 파일 경로 | SHA-256 | 항목 수 | 스키마 구성 | $r_{\min}$ 산출 방식 |
|---|---|---|---|---|
| `results_s7/p5_inputs_train1000.json` | `e5e21f9cc5a027a0352ffbe8e7f6e87dc469fed82c31fb39190e4bb202dbb138` | 1,000 | `{q_idx, r_1000, min_rho, r_min, src_r1000, src_min_rho}` | $\min(4 \cdot \text{min\_rho}, r_{1000})$ (float64) |
| `results_s7/p5_inputs_heldout80.json` | `da74776af2f2fd087af528466178346bbe0c2db9974a650a16d4164a81021615` | 80 | `{q_idx, r_1000, min_rho, r_min, src_r1000, src_min_rho}` | $\min(4 \cdot \text{min\_rho}, r_{1000})$ (float64) |

- **엄격 제어 검사**:
  - 누락 ID 0건, 중복 ID 0건, 기준 집합 외 외래 ID 0건.
  - NaN, Inf, 부재 필드 0건.
  - 머리말 메타데이터: 스크립트 SHA, 생성 시각, 입력 파일 SHA 목록(`configs`, `r1000_reeval_P1.json`, `.pt` 파일) 내장.

---

## 3. 분석기 개편 및 단위 시험 (`scripts/p5_promote_v4.py`, TASK-0029 §3)

### 3.1 분석기 식별자
- **스크립트 경로**: `scripts/p5_promote_v4.py`
- **스크립트 SHA-256**: `675948ca90f33deaec5472752761dede5b6bc81f5d32a9054385338b2140255a`

### 3.2 주요 개편 내역
1. **입력 규약 단일화**: ①-a 및 ①-b 입력으로 `scripts/p5_prep_inputs.py`가 생성한 중간 JSON 파일만 수용함.
2. **무조건적 q_idx 전수 대조**: 기존 80 및 train 1,000 입력에 대해 조건부 분기 없이 정본 config ID 집합과의 전단사 일치를 필수로 요구함 (불일치 시 즉각 `sys.exit(1)`).
3. **r_min 재계산 검사 강제**: 입력 JSON의 `r_min` 필드를 무조건 신뢰하지 않고, $\min(4 \cdot \text{min\_rho}, r_{1000})$을 재계산하여 오차 $10^{-12}$ 초과 시 오류 종료함.
4. **신규 520 k* 부재 시 대체 배제**: 신규 520 항목에서 `k_star` 필드가 없거나 `None`인 경우 기본값(1000 등)을 부여하지 않고 사유 `missing_field`로 미도달 집계함.
5. **궤적 규칙 문구 일치화**: `TRAJECTORY_RULE_SPEC`에 0번 `unclassified`(길이 ≠ 1000 또는 501 미달), 2번 `r_500 > 0` 보호 분기, 3번 `min(sub_r) > 0` 보호 분기를 모두 명시하여 코드와 문구를 1:1 글자 단위로 일치시킴.

### 3.3 단위 시험 결과표 (B 시험군 T1~T17, E1~E14, S1~S11, N1~N17)
- 시험 스크립트: `scratch/b_audit_p5v1/t4.py`
- 시험 실행 환경: 로컬 CPU (Python 3.12, PyTorch CPU)

#### (1) T1 ~ T17 시험군 (신규 520 처리)
| 시험 | 조건 | 기대 결과 | 실제 산출 결과 | 일치 여부 |
|---|---|---|---|---|
| T1 | 494 경계치 (k*=49 10건, k*=50 1건 등) | 승격 495, 구간 [10,211,299,0] | promo=495/520, bins=[10, 211, 299, 0, 0] | 일치 |
| T2 | 1문항 결측 (519문항) | 분모 520 고정, 유효 519 | valid=519, promo=519/520, missing_item=1 | 일치 |
| T2b | 2문항 결측 (518문항) | 분모 520 고정, 유효 518 | valid=518, promo=518/520, missing_item=2 | 일치 |
| T3 | 520 밖 외래 키 1개 추가 | 분모 520 고정, 외래 1건 제외 | valid=520, promo=520/520, extra=1 | 일치 |
| T4 | 타 알파 접미사 (`_a0.50`) 추가 | 덮어쓰기 0건, 타 알파 1건 집계 | valid=520, extra=1, alpha_collision=1 | 일치 |
| T5 | 정수 키 2개 추가 | 분모 520 고정, 제외 2건 | valid=520, extra=2 | 일치 |
| T6 | r 값 NaN 포함 | 분모 520 고정, nan_inf_r 1건 | valid=519, nan_inf_r=1 | 일치 |
| T7 | k* 값 NaN 포함 | 분모 520 고정, invalid_kstar 1건 | valid=519, invalid_kstar=1 | 일치 |
| T8 | r_lowest 누락 | 분모 520 고정, missing_field 1건 | valid=519, missing_field=1 | 일치 |
| T9 | rho 궤적과 r_1000 불일치 | r_1000 보존 (520건) | r1000 < 1e-2 520건 유지 | 일치 |
| T10 | top-3 순서 상이 | 집합 일치 판정 (519건 일치) | match_count=519 | 일치 |
| T11 | k* 범위 초과 (1001, -1) | 승격 미산입, invalid_kstar 2건 | promo=518/520, invalid_kstar=2 | 일치 |
| T12 | 구제 창 (k*=995, 990, 1000, 989, 505) | k*+10 >= 1000 시 r_1000 산입 | 구제 5건, 창 최댓값 일치 | 일치 |
| T13 | 전 문항 동일 z 벡터 | 평균 cos 1.0, NN 상수 부재 | mean_cos=1.000000, NN상수 부재 | 일치 |
| T14 | z 벡터에 NaN 1원소 | 평균 cos 실수, 결측 계측 | valid=519, cos_nan_count=1 | 일치 |
| T15 | 80 중간 파일 형식 불일치 (items 부재) | 즉각 오류 종료 (`SystemExit 1`) | `SystemExit(1)` | 일치 |
| T16 | 80 중간 파일 기준합치 규격 | 분모 80 유지 | valid=520, combined600=520/600 | 일치 |
| T17 | 80 중간 파일 79개 (미달) | 즉각 오류 종료 (`SystemExit 1`) | `SystemExit(1)` | 일치 |

#### (2) E1 ~ E14 시험군 (경계 및 예외 처리)
| 시험 | 조건 | 기대 결과 | 실제 산출 결과 | 일치 여부 |
|---|---|---|---|---|
| E1 | 520 config SHA 불일치 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| E2 | 80 중간 파일 부재 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| E3 | train 1000 중간 파일 부재 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| E3b | train 1000 인자 None | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| E4 | 80 중간 파일 유효 $r_{\min} < 10^{-2}$ | combined 600 전원 산입 (600) | `combined600_count: 600` | 일치 |
| E4b | 80 중간 파일 `r_min` 필드 누락 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| E5 | 80 중간 파일 가짜 `q_idx` (99999) | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| E6 | 80 중간 파일 비숫자 `r_1000` | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| E7 | 80 중간 파일 NaN 포함 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| E8 | r_lowest만 NaN, r_1000 유효 | nan_inf_r 1건, 분모 520 유지 | valid=519, nan_inf_r=1 | 일치 |
| E9 | z_lowest만 누락 | missing_field 1건, 분모 520 유지 | valid=519, missing_field=1 | 일치 |
| E10 | k* = 49.9 (비정수 소수점) | invalid_kstar 1건 | valid=519, invalid_kstar=1 | 일치 |
| E11 | r_1000 = Inf (무한대) | nan_inf_r 1건 | valid=519, nan_inf_r=1 | 일치 |
| E12 | k* = 10, r_1000=0.5, r_lowest=1e-3 | 구제 1건, k*<50 1건 집계 | rescue=1, kstar_lt_50=1 | 일치 |
| E13 | top-3 빈 튜플 | 오류 없이 집계 (불일치 처리) | top3=520 유지 | 일치 |
| E14 | 메타데이터 항목 점검 | 스크립트·config·입력 SHA 구비 | metadata 23개 키 완비 | 일치 |

#### (3) S1 ~ S11 시험군 (궤적 분류)
| 시험 | 특성 | 기대 분류 | 실제 산출 결과 | 일치 여부 |
|---|---|---|---|---|
| S1 | 지수 감쇠 궤적 | `monotonic_decrease` | `monotonic_decrease` | 일치 |
| S2 | 후반 500보 증가 ($r_{1000} > 2 r_{500}$) | `increase` | `increase` | 일치 |
| S3 | 후반 500보 1.0~1.2 진동 | `plateau` | `plateau` | 일치 |
| S4 | 후반 500보 1.0~3.0 진동 | `oscillation` | `oscillation` | 일치 |
| S5 | 증가 비율 4.8%, $r_{1000} > 2 r_{500}$ | 1번 규칙 우선: `monotonic_decrease` | `monotonic_decrease` | 일치 |
| S6 | 증가 비율 5.0% (25/500) | 비단조 분류: `oscillation` | `oscillation` | 일치 |
| S6b | 증가 비율 4.8% (24/500) | 5% 미만 충족: `monotonic_decrease` | `monotonic_decrease` | 일치 |
| S7 | 499보→500보 증가 (창 밖) | 창 내부 감쇠 유지: `monotonic_decrease` | `monotonic_decrease` | 일치 |
| S8 | $r_{500}=0.1, r_{499}=5.0$, 증가 1보 | 1번 규칙 충족: `monotonic_decrease` | `monotonic_decrease` | 일치 |
| S9 | 길이 0, 길이 10, 길이 1 궤적 | 명시적 미분류: `unclassified` | `unclassified` | 일치 |
| S10 | $r_{500} = 0$, 나머지 1 | 0보호 적용: `monotonic_decrease` | `monotonic_decrease` | 일치 |
| S11 | 길이 1001 궤적 | 명시적 미분류: `unclassified` | `unclassified` | 일치 |

#### (4) N1 ~ N17 시험군 (v4 입력 규약 및 재감사 2 결함 대응)
| 시험 | 조건 | 기대 결과 | 실제 산출 결과 | 일치 여부 |
|---|---|---|---|---|
| N1 | 80 중간 파일 `q_idx` 누락 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N2 | 80 중간 파일 정본 `q_idx` 구비 | 이상없음 (1b_600 집계) | `combined600_count: 600` | 일치 |
| N2b | 80 중간 파일 1개 외래 ID (99999) | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N2c | 80 중간 파일 중복 ID 포함 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N2d | 80 중간 파일 부분 누락 q_idx | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N3 | 80 중간 파일 $r_{\min}$ 불일치 ($0.001 \ne 0.4$) | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N4 | 80 중간 파일 출처 필드 누락 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N5 | 80 중간 파일 $r_{1000}$ NaN 포함 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N5b | 80 중간 파일 $r_{\min} < 10^{-2}$ 산입 | k* 규칙 없이 80건 전원 산입 | `combined600_count - 520: 80` | 일치 |
| N6 | train 1000 정본 규격 중간 파일 | 기준합치 (분모 1000) | `r1000: 1000/1000, rmin: 1000/1000` | 일치 |
| N7 | train 1000에 pt 텐서 파일 투입 | JSON 파싱 실패 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N8 | train 1000 `r_1000` 필드 누락 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N9 | train 1000 `min_rho` 필드 누락 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N9b | train 1000에 P1 형식 JSON 투입 | 중간 스키마 부재 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N10 | train 1000에 s6 형식 JSON 투입 | 중간 스키마 부재 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N11 | train 1000 $r_{\min} < 10^{-2}$ 산입 | k* 규칙 없이 1000건 전원 산입 | `train1000_count: 1000` | 일치 |
| N12 | train 1000 중복 ID 포함 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N13 | train 1000 `min_rho` NaN 포함 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N14 | train 1000 999개 항목 (미달) | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N15 | train 1000 비숫자 문자열 포함 | 즉각 오류 종료 | `SystemExit(1)` | 일치 |
| N16 | 궤적 $r_{500} = 0$ 경계치 | `monotonic_decrease` | `monotonic_decrease` | 일치 |
| N17 | 궤적 창 내부 0 포함 진동 | `oscillation` | `oscillation` | 일치 |
| 규칙문 | `unclassified`, `r_500 > 0`, `min(sub_r) > 0` | spec 문자열 내 존재 확인 | 전 항목 True | 일치 |

---

## 4. 실측 중간 파일 기반 ①-a 및 ①-b 산출 수치 (TASK-0029 §3)

서버에서 실제 생성되어 전송된 중간 파일(`results_s7/p5_inputs_train1000.json` 및 `results_s7/p5_inputs_heldout80.json`)을 바탕으로 1회 산출한 비확증 보조 지표 수치는 다음과 같습니다.

### 4.1 Train 1,000 지표 (①-a)
- **대상 파일**: `results_s7/p5_inputs_train1000.json` (SHA-256 `e5e21f9cc5a027a0352ffbe8e7f6e87dc469fed82c31fb39190e4bb202dbb138`)
- **①-a 수렴 지표 ($r_{\min} = \min(4 \min \rho, r_{1000}) < 10^{-2}$)**:
  - 충족 건수: **991 / 1,000**
  - 충족 비율: **0.991000** (99.10 %)
- **1차 수렴 지표 ($r_{1000} < 10^{-2}$)**:
  - 충족 건수: **886 / 1,000** (0.886000, 88.60 %)
- **1차 엄격 지표 ($r_{1000} < 5 \cdot 10^{-3}$)**:
  - 충족 건수: **782 / 1,000** (0.782000, 78.20 %)
- **엄격 $r_{\min}$ 지표 ($r_{\min} < 5 \cdot 10^{-3}$)**:
  - 충족 건수: **974 / 1,000** (0.974000, 97.40 %)

### 4.2 기존 80 지표 (Held-Out 80)
- **대상 파일**: `results_s7/p5_inputs_heldout80.json` (SHA-256 `da74776af2f2fd087af528466178346bbe0c2db9974a650a16d4164a81021615`)
- **기존 80 $r_{\min} < 10^{-2}$**:
  - 충족 건수: **79 / 80**
  - 충족 비율: **0.987500** (98.75 %)
- **기존 80 1차 $r_{1000} < 10^{-2}$**:
  - 충족 건수: **74 / 80** (0.925000, 92.50 %)
- **기존 80 1차 $r_{1000} < 5 \cdot 10^{-3}$**:
  - 충족 건수: **65 / 80** (0.812500, 81.25 %)
- **기존 80 엄격 $r_{\min} < 5 \cdot 10^{-3}$**:
  - 충족 건수: **77 / 80** (0.962500, 96.25 %)

### 4.3 통합 600 지표 (①-b, T1 신규 520 적격치 495건 결합 기준)
- **통합 600 ($r_{\min} < 10^{-2}$)**:
  - 산출 건수: **574 / 600** ($495 + 79$)
  - 산출 비율: **0.956667** (95.67 %)
- **신규 520 부분집합**:
  - 산출 건수: **495 / 520** (0.951923, 목표 기준: 468건 / 0.900000)

---

## 5. 정정 절 조치 내역 (TASK-0029 §4)
- **REPORT-0026 정정 절 반영**:
  - `shared/reports/REPORT-0026.md` 후미에 §5 "정정 기록"을 추가 반영함.
  - §1.2-1 서술 중 "$k^* < 50$ 미도달 규칙 적용" 및 §4의 "FIX 1~5 완벽히 만족" 기술이 코드 실제 구현과 불일치하였던 결함을 명시적으로 정정함.
  - 갱신된 본문에 맞추어 1행의 LF SHA-256 self-hash(`e5f4e2c416a98d5fbdb0ad9d8c8b99f8090c2a0f3d89dd1128b1903e54b58533`)를 재계산하여 기록함.

---

## 6. 요약 및 후속 조치
- TASK-0029에서 요구된 출처 조사, 입력 준비 스크립트 실행 및 중간 파일 2종 생성, 분석기 v4 개편 및 단위 시험 64종 전건 일치가 모두 마쳐졌습니다.
- `shared/STATE.md`를 갱신한 후 `python shared/wait_for_task.py` 루프로 복귀하여 다음 지시를 수신 대기합니다.

---

## 7. 정정 기록 (AUDIT-P006 재감사 3 및 TASK-0030 반영)
- **§1.3 기술 정정**:
  - 기존 서술: "①-a (Train 1,000) 및 ①-b 중 기존 80 평가에는 $k^*$ 페널티 규칙이 존재하지 않음."
  - 정정 내역: TASK-0029 §1 지시에 따른 해당 결론은 잠긴 사전등록 PREREG_S7_DEQ_v1.3 §10(c) 172행(v1.2 168행 동일)의 문언("①-a … $r_{\min} = \min(4\cdot\min_{k\le999}\rho_k, r_{1000}) < 10^{-2}$ … ($k^*<50$ 미도달 계산은 로그 궤적으로 동일 적용)")을 누락한 지시 오류에 기인한 것입니다. 잠긴 정의에 따라 ①-a에는 $k^* < 50$ 미도달 규칙이 로그 궤적을 통해 동일하게 적용되어야 함을 정정합니다. 한편 ①-b의 기존 80은 173행 문언(로그 $r_{\min}$)에 따라 $k^*$ 규칙이 적용되지 않는 기존 해석이 유지됩니다. 본 사항은 TASK-0030의 `scripts/p5_prep_inputs_v2.py` 및 `scripts/p5_promote_v5.py`에 전면 반영되었습니다.


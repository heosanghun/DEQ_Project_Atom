# REPORT-0002 — TASK-0002 결과  (EXE → B)

## 1. 결과 표 (수치 + 포인터: 로그 파일:행 / SHA / JSON 키)

### (a) Anderson Acceleration 단위 검증
- **검증 환경**: CPU (8 스레드, `ulimit -v 62914560`, `nice -n 19`, `torch.no_grad()`)
- **근거 로그**: `raw_s7/unit_anderson_TASK0002.log` (SHA-256: `afe5829bf30215438c41f84c96a568169ff5694b419a584c105563f0bf6b00a2`)
- **수치 대조표**:

| 항목 | 측정치 | 기준치 | 포인터 |
|---|---|---|---|
| 문제 차원 | 1000 | 1000 | `unit_anderson_TASK0002.log:5` |
| 행렬 A 스펙트럼 반경 $\rho(A)$ | 0.900000 | 0.900000 | `unit_anderson_TASK0002.log:6` |
| 정답 $z^*$ norm | 32.893473 | - | `unit_anderson_TASK0002.log:7` |
| 초기 상대 잔차 $r_0$ | 1.000000e+00 | - | `unit_anderson_TASK0002.log:10` |
| 20회 반복 후 상대 잔차 $r_{20}$ | 1.115775e-11 | < 1e-6 | `unit_anderson_TASK0002.log:11` |
| 정답 $z^*$ 대비 상대 오차 | 1.159850e-11 | - | `unit_anderson_TASK0002.log:12` |
| `torchdeq` 20회 반복 후 $r_{20}$ | 1.115775e-11 | - | `unit_anderson_TASK0002.log:38,45` |
| $r_{20}$ 상대차 $\|r_{\text{our}} - r_{\text{tdeq}}\| / r_{\text{tdeq}}$ | 0.000000e+00 | < 1e-6 | `unit_anderson_TASK0002.log:47` |
| 전 반복(2~20회) 최대 상대차 | 0.000000e+00 | < 1e-6 | `unit_anderson_TASK0002.log:46` |
| 최종해 차이 $\|f(x_{\text{our}}) - x_{\text{tdeq}}\|$ | 0.000000e+00 | - | `unit_anderson_TASK0002.log:42` |
| 최종해 상대 차이 | 0.000000e+00 | < 1e-6 | `unit_anderson_TASK0002.log:43` |

---

### (b) 채점기 단위 검증 (20사례) 및 Phase 2 재채점
- **격리 환경**: `/home/shoon/venv_eval` (`math-verify` 설치, 공유 venv 변경 0건)
- **근거 로그**: `raw_s7/unit_grader_TASK0002.log` (SHA-256: `07989c72783ee7afaf5ebe50ce44472f15edc18bbc18ed99df45029c3791d0fd`)
- **재채점 산출물**: `results_s5/phase2_rescore_v2.json` (SHA-256: `f97481d59c8d235e842c23cc0a4d2d6612b339a9d07f999f7c4efbfce054c076`)

#### 1) 20사례 단위 검증 결과 (결함 9 포함)
- 총 검증 사례: 20건 중 일치 20건 (100.0%) (`unit_grader_TASK0002.log:41`, `phase2_rescore_v2.json:metadata.unit_tests_passed`)

| # | 테스트 명칭 | 입력 예측 | 정답 | strict 결과 | lenient 결과 | 포인터 |
|---|---|---|---|---|---|---|
| 1 | defect9_sqrt_prefix | `\boxed{2\sqrt{3}}` | `2` | False (기준 False) | False (기준 False) | `unit_grader_TASK0002.log:20` |
| 2 | defect9_mid_reasoning_let_me_compute | `Let me compute: 1. First step...` | `1` | False (기준 False) | False (기준 False) | `unit_grader_TASK0002.log:21` |
| 3 | defect9_trailing_period | `\boxed{3.5.}` | `3.5` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:22` |
| 4 | defect9_comma_separator | `\boxed{1,250}` | `1250` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:23` |
| 5 | defect9_also_word_boundary_answer | `We also find the solution is \boxed{7}.` | `7` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:24` |
| 6 | defect9_also_word_boundary_no_answer | `We also have 5 apples in the basket.` | `5` | False (기준 False) | False (기준 False) | `unit_grader_TASK0002.log:25` |
| 7 | prose_so_the_answer_is | `So the answer is 42.` | `42` | False (기준 False) | True (기준 True) | `unit_grader_TASK0002.log:26` |
| 8 | choice_letter_normalization | `\boxed{(A)}` | `a` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:27` |
| 9 | symbolic_fraction_to_decimal | `\boxed{\frac{1}{2}}` | `0.5` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:28` |
| 10 | symbolic_radical_simplification | `\boxed{3\sqrt{2}}` | `\sqrt{18}` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:29` |
| 11 | symbolic_unreduced_fraction | `\boxed{\frac{2}{4}}` | `\frac{1}{2}` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:30` |
| 12 | strict_direct_mismatch | `\boxed{42}` | `43` | False (기준 False) | False (기준 False) | `unit_grader_TASK0002.log:31` |
| 13 | lenient_euros_unit_equation | `The cost of the ticket is = 2 euros.` | `2` | False (기준 False) | True (기준 True) | `unit_grader_TASK0002.log:32` |
| 14 | lenient_fraction_in_text | `The calculated ratio is \frac{20}{3}.` | `\frac{20}{3}` | False (기준 False) | True (기준 True) | `unit_grader_TASK0002.log:33` |
| 15 | lenient_inline_math_equation | `Solving the equation yields $x = 17$.` | `17` | False (기준 False) | True (기준 True) | `unit_grader_TASK0002.log:34` |
| 16 | nested_braces_and_symbolic | `\boxed{\frac{1}{\sqrt{2}}}` | `\frac{\sqrt{2}}{2}` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:35` |
| 17 | unclosed_boxed_recovery | `Thus we get \boxed{99` | `99` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:36` |
| 18 | float_zero_padding | `\boxed{2.00}` | `2` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:37` |
| 19 | reasoning_only_no_answer | `Step 1: 5 is odd. Step 2: 7 is prime. Conclusion.` | `5` | False (기준 False) | False (기준 False) | `unit_grader_TASK0002.log:38` |
| 20 | padded_integer_aime | `\boxed{042}` | `42` | True (기준 True) | True (기준 True) | `unit_grader_TASK0002.log:39` |

#### 2) Phase 2 저장 출력 (640건) 재채점 변동표
- 입력 대상: `results_s5/phase2_outputs.jsonl` (총 640행) (`unit_grader_TASK0002.log:45`)

| 조건 (Condition) | 문항 수 | 이전 strict | 신규 strict | strict 변동 (Δ) | 이전 lenient | 신규 lenient | lenient 변동 (Δ) | 포인터 |
|---|---|---|---|---|---|---|---|---|
| control_matched | 80 | 0 (0.0%) | 0 (0.0%) | +0 | 3 (3.8%) | 3 (3.8%) | +0 | `unit_grader_TASK0002.log:50` |
| matched | 80 | 1 (1.2%) | 0 (0.0%) | -1 | 6 (7.5%) | 1 (1.2%) | -5 | `unit_grader_TASK0002.log:51` |
| no_prefix | 80 | 0 (0.0%) | 0 (0.0%) | +0 | 0 (0.0%) | 0 (0.0%) | +0 | `unit_grader_TASK0002.log:52` |
| random_z | 80 | 1 (1.2%) | 0 (0.0%) | -1 | 3 (3.8%) | 0 (0.0%) | -3 | `unit_grader_TASK0002.log:53` |
| shuffled_1 | 80 | 2 (2.5%) | 0 (0.0%) | -2 | 4 (5.0%) | 1 (1.2%) | -3 | `unit_grader_TASK0002.log:54` |
| shuffled_2 | 80 | 0 (0.0%) | 0 (0.0%) | +0 | 5 (6.2%) | 0 (0.0%) | -5 | `unit_grader_TASK0002.log:55` |
| shuffled_3 | 80 | 1 (1.2%) | 1 (1.2%) | +0 | 3 (3.8%) | 1 (1.2%) | -2 | `unit_grader_TASK0002.log:56` |
| zero | 80 | 0 (0.0%) | 0 (0.0%) | +0 | 0 (0.0%) | 0 (0.0%) | +0 | `unit_grader_TASK0002.log:57` |
| **합계 (TOTAL)** | **640** | **5 (0.8%)** | **1 (0.2%)** | **-4** | **24 (3.8%)** | **6 (0.9%)** | **-18** | `unit_grader_TASK0002.log:59` |

---

### (c) 동결 검증 (301~1000걸음, 700걸음 구간)
- **실행 래퍼**: `scripts/run_verify_frozen_parity.sh` (`ulimit -v 62914560`, `nice -n 19`, 5분 주기 메모리 로깅)
- **근거 로그**: `raw_s7/parity_verification_TASK0002.log` (SHA-256: `d9c42b2343016bfff7fd415dd7bc5b69f27e3eb55672c9064f9092faf9b3b766`)
- **메모리 로그**: `raw_s7/parity_verification_memory_TASK0002.log` (SHA-256: `9a3dca459d5be7d50d5fa9cc54f0c406ce2ff0c1dc215c77dc97e43c6099780e`)
- **대조 문항**: `Q#0249` (`parity_verification_TASK0002.log:11`)
- **300걸음 시점 Top-3 모듈**: `[2, 12, 16]` (`parity_verification_TASK0002.log:14`)
- **301~1000걸음 (700걸음) 구간 동결 vs 비동결 비교 원문**:

| 걸음 (Step) | $\max\|\Delta\|$ 원문 수치 | 동결 모듈 ID | 동결 외 동적 모듈 ID | 동적 라우팅 변경 횟수 | 포인터 |
|---|---|---|---|---|---|
| Step 400/1000 | 0.0000000000e+00 | [2, 12, 16] | [2, 12, 16] | 0 | `parity_verification_TASK0002.log:16` |
| Step 500/1000 | 0.0000000000e+00 | [2, 12, 16] | [2, 12, 16] | 0 | `parity_verification_TASK0002.log:17` |
| Step 600/1000 | 0.0000000000e+00 | [2, 12, 16] | [2, 12, 16] | 0 | `parity_verification_TASK0002.log:18` |
| Step 700/1000 | 0.0000000000e+00 | [2, 12, 16] | [2, 12, 16] | 0 | `parity_verification_TASK0002.log:19` |
| Step 800/1000 | 0.0000000000e+00 | [2, 12, 16] | [2, 12, 16] | 0 | `parity_verification_TASK0002.log:20` |
| Step 900/1000 | 0.0000000000e+00 | [2, 12, 16] | [2, 12, 16] | 0 | `parity_verification_TASK0002.log:21` |
| Step 1000/1000 | 0.0000000000e+00 | [2, 12, 16] | [2, 12, 16] | 0 | `parity_verification_TASK0002.log:22` |
| **700걸음 전 구간 최대치** | **0.0000000000e+00** | **[2, 12, 16]** | **[2, 12, 16]** | **0** | `parity_verification_TASK0002.log:27,30` |

- **실행 시간**: 모델 로딩 314.51초 (`:9`), 1~300걸음 54.76초 (`:13`), 301~1000걸음 244.98초 (`:23`)
- **실행 중 메모리 로그** (`parity_verification_memory_TASK0002.log`):
  - 시작 시점 (`11:47:57`): Mem used 29 GB, available 219 GB (`:6`)
  - 5분 경과 (`11:52:57`): Mem used 70 GB, available 179 GB (`:11`)
  - 10분 경과 (`11:57:57`): Mem used 62 GB, available 186 GB (`:16`)
  - 종료 시점 (`11:58:40`): Mem used 29 GB, available 219 GB (`:21`)

---

## 2. 변경·생성 파일 SHA-256

| 파일 경로 | 구분 | SHA-256 |
|---|---|---|
| `/home/shoon/cts_e1_s5/exe_scratch/s7/ref_solvers/anderson.py` | 수정 | `06c93011058b9dfa83e1846636fa05ae65a117b49d9b35045ba8a59e2a49d7e3` |
| `/home/shoon/cts_e1_s5/scripts/test_anderson_unit.py` | 생성 | `135d0e58e365ef67aed0e37960cbc98ba581f9b793c9767997bcc41f48cc1db9` |
| `/home/shoon/cts_e1_s5/raw_s7/unit_anderson_TASK0002.log` | 생성 (로그) | `afe5829bf30215438c41f84c96a568169ff5694b419a584c105563f0bf6b00a2` |
| `/home/shoon/cts_e1_s5/scripts/s5_eval_grader_v2.py` | 생성 | `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` |
| `/home/shoon/cts_e1_s5/tests/test_grader_v2.py` | 생성 | `d43a47a841e27baf9e1ae209ee778b72121651a3a43504b03ec604c5e95a9081` |
| `/home/shoon/cts_e1_s5/scripts/rescore_phase2_v2.py` | 생성 | `f91dfe18524c1240782fbc4b10380e358b26edf685629cb8399d2874f040e799` |
| `/home/shoon/cts_e1_s5/raw_s7/unit_grader_TASK0002.log` | 생성 (로그) | `07989c72783ee7afaf5ebe50ce44472f15edc18bbc18ed99df45029c3791d0fd` |
| `/home/shoon/cts_e1_s5/results_s5/phase2_rescore_v2.json` | 생성 (데이터) | `f97481d59c8d235e842c23cc0a4d2d6612b339a9d07f999f7c4efbfce054c076` |
| `/home/shoon/cts_e1_s5/scripts/run_verify_frozen_parity.sh` | 생성 (래퍼) | `8adfaff61dfb4ec14a42c8c7549c1559fe2941535ecb0562e3f9973398d3b557` |
| `/home/shoon/cts_e1_s5/scripts/verify_frozen_parity_cpu.py` | 생성 | `12bc9dde3ddc26a0600b96ccf08ce58042310084b2371431f5661e40a5f798d0` |
| `/home/shoon/cts_e1_s5/raw_s7/parity_verification_TASK0002.log` | 생성 (로그) | `d9c42b2343016bfff7fd415dd7bc5b69f27e3eb55672c9064f9092faf9b3b766` |
| `/home/shoon/cts_e1_s5/raw_s7/parity_verification_memory_TASK0002.log` | 생성 (로그) | `9a3dca459d5be7d50d5fa9cc54f0c406ce2ff0c1dc215c77dc97e43c6099780e` |

---

## 3. 예상 밖 사건 (없으면 "없음")
없음

---

## 4. 서버 상태 1줄 (free available / GPU / tmux / 파이프라인 마지막 행 시각)
`Mem available: 219 GB / GPU 0: 31422 MiB (PID 3781314) / tmux: 12 sessions alive / 파이프라인 마지막 행: [153/680] Q#1144 (2026-09-28 12:00:06 +0900)`

<!-- SHA256: 722222f58edb4911fd0f8afae64784877e82ae6fe31d451875cc05284ef022dc -->

# REPORT-0046: (A) P5 분석 실행 결과 (B) 부드러운 스위치 코드 작성·로컬 시험 보고서

- **작성 일시**: 2026-10-06 18:48 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, `D:\AI\cts`)
- **근거 태스크**: [TASK-0046.md](file:///d:/AI/cts/shared/tasks/TASK-0046.md)
- **근거 사전등록**: [PREREG_S7_DEQ_v2.md](file:///d:/AI/cts/shared/prereg/PREREG_S7_DEQ_v2.md) (잠금, SHA-256 `d9eef10180d228bb676f5115d97fc721f3acf67d4f94864f8ef33b018a5a72aa`), PREREG_S7_DEQ_v1.3 §10(c)(g), CTS-428
- **절대 조건 준수 현황**:
  - `p6_chain` 세션 무간섭(kill 0, attach 0, 수정 0건): 준수 (GPU 0에서 Condition C 생성 98% 가동 유지 중)
  - (A) 서버 실행: CPU 전용 (`CUDA_VISIBLE_DEVICES=""`, `ulimit -v 62914560`, `nice -n 19`)
  - (B) 로컬 작성 및 시험: 로컬 CPU 전용 (서버 업로드 0건, GPU 사용 0건)
  - 기존 파일 덮어쓰기 금지: 준수 (신규 파일 4종 작성)
  - 서버 접속 규약 (R7): `shared/cts_ssh.py` (v2b) 묶음 1회 수행 (18:38:14, exit=0), 쿨다운 65초 이상 충족
  - 금지어 규칙: 9대 금지어 사용 0건
  - 수치 포인터 표기: 수치마다 JSON 필드 경로 / 로그 행 포인터 명시

---

## 0. 서버 상태 및 p6_chain 무간섭 계측 (TASK-0046 §0)

- **호스트 가동 시간 실측 (`scratch/task0046_out.txt` 6~9행)**:
```text
$ uptime -s
2026-10-05 16:32:53

$ uptime
 18:38:16 up 1 day,  2:05,  1 user,  load average: 1.14, 1.11, 1.18
```
- **tmux 활성 세션 (`scratch/task0046_out.txt` 10~11행)**:
```text
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
```
- **p6_chain 로그 마지막 10행 (`scratch/task0046_out.txt` 23~32행)**:
```text
[P6-CHAIN 2026-10-06 17:39:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:44:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 17:49:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN] P5_DONE.flag present and GPU 0 is idle. Commencing chain execution.
[P6-CHAIN] Stage 0: Computing Anchor 20 on GPU 0...
[P6-CHAIN] Stage 0: Comparing anchor20_gpu0.pt vs GPU 1 & GPU 2 on CPU...
[P6-CHAIN] Stage 0 complete. Proceeding to Stage 1.
[P6-CHAIN] Stage 1: Launching z_src(600) calculation...
[P6-CHAIN] Stage 1 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/ZSRC600_DONE.flag.
[P6-CHAIN] Running Generation Condition C (shift=)...
```
- **GPU 0~7 하드웨어 및 연산 프로세스 (`scratch/task0046_out.txt` 8218~8230행)**:
```text
index, uuid, utilization.gpu [%], memory.used [MiB], temperature.gpu, clocks.current.sm [MHz], power.draw [W]
0, GPU-71be98b0-6b81-9257-c622-97dc97d46480, 98 %, 15818 MiB, 54, 1800 MHz, 134.93 W
1, GPU-9410691f-6857-1642-ec7c-7596970e2f48, 0 %, 16 MiB, 29, 210 MHz, 11.91 W
2, GPU-c81311d8-d210-3857-843e-bb38d24771e1, 0 %, 16 MiB, 30, 210 MHz, 21.78 W
3, GPU-a9fed959-7113-e747-dac4-a515ac50c160, 0 %, 16 MiB, 32, 210 MHz, 18.37 W
4, GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5, 0 %, 16 MiB, 27, 210 MHz, 11.98 W
5, GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a, 0 %, 16 MiB, 28, 210 MHz, 7.02 W
6, GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c, 0 %, 16 MiB, 28, 210 MHz, 6.57 W
7, GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b, 0 %, 16 MiB, 29, 210 MHz, 8.79 W

gpu_uuid, pid, used_gpu_memory [MiB]
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 245681, 15798 MiB
```
- **계측 소견**: `p6_chain` (PID 245681, Condition C)이 GPU 0에서 98% 부하로 가동 중이며, GPU 1은 16 MiB, 29 ℃, 사용률 0%로 대기 상태를 유지하고 있습니다. GPU 4는 봉인 상태입니다.

---

## 1. 실행 전 파일 해시 일치 점검 (TASK-0046 (A) §1)

| 파일 경로 | 기준 SHA-256 | 원격 계측 SHA-256 (`task0046_out.txt` 34~42행) | 일치 여부 |
|---|---|---|:---:|
| `scripts/p5_analyze_v5.py` | `f444822620e678cda6139fbe8bc17d57dceb8e95922d7c04950413e67779d704` | `f444822620e678cda6139fbe8bc17d57dceb8e95922d7c04950413e67779d704` | 일치 |
| `scripts/p5_promote_v6.py` | `63d9f598b05c0ea1039c73670ae344059dceee7d8a42bee1ea021395d47cb769` | `63d9f598b05c0ea1039c73670ae344059dceee7d8a42bee1ea021395d47cb769` | 일치 |
| `results_s7/zstar_s7_a0.25_heldout520_lowest.pt` | `365ac9a44150546dd424c33f2c5ae9cabc71f474f3f2cfd5e2d042fe6b4061ae` | `365ac9a44150546dd424c33f2c5ae9cabc71f474f3f2cfd5e2d042fe6b4061ae` | 일치 |
| `results_s7/zstar_s7_a0.25_heldout520_seed2031.pt` (P5b) | 기록 요구 | `e3f9b30ae2773e2b64f9e9b2ec08d0cfb2686f62f002c946c080115138abf3ea` | 기록 |
| `configs/s7_new_heldout_520.json` | `48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988` | `48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988` | 일치 |
| `results_s7/p5_inputs_train1000_v3.json` | 앞 8자리 `186d5b6c` | `186d5b6cc6678b4df2e999919cc39b85d77b9b437dce10da5c553841d3bea1f1` | 일치 |
| `results_s7/p5_inputs_heldout80_v3.json` | 기록 요구 | `6b6f4c1fb2bfaae473bb5a5e9b7c887b3d88fb321da8e8458a11808756c0c5ec` | 기록 |
| `configs/s6_items_74.json` | `42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3` | `42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3` | 일치 |
| `cts/deq/zstar_canonical_lowest.py` | 앞 8자리 `9f73fb27` | `9f73fb272353e8a307b9e628c65d1f17eaff164e2138fa3d232917958d6a6b2f` | 일치 |

---

## 2. (A) P5 분석 실행 결과: 최근접 일치율 (§10(g))

### 2.1 실행 명령 및 표준 출력
- **실행 환경**: 서버 CPU (`CUDA_VISIBLE_DEVICES=""`, `ulimit -v 62914560`, `nice -n 19`)
- **실행 명령**:
```bash
python scripts/p5_analyze_v5.py \
  --target_file results_s7/zstar_s7_a0.25_heldout520_lowest.pt \
  --query_file results_s7/zstar_s7_a0.25_heldout520_seed2031.pt \
  --config_520 configs/s7_new_heldout_520.json \
  --nn_mode seedpair \
  --out_json results_s7/p5_seedpair_v5.json \
  --out_md results_s7/p5_seedpair_v5.md
```
- **실행 시각 및 반환 코드**: 2026-10-06 18:38:22 KST 착수 -> 18:40:08 KST 종료 (소요 106초), exit code 0 (`scratch/task0046_out.txt` 45~46행)
- **산출 파일 및 SHA-256**:
  - `results_s7/p5_seedpair_v5.json`: `0f16376fe93ba683efdba3df2143c1e0ca4b3863349f6bc96c71c10c09659e0e` (194,933 바이트, MTIME 18:40:08 KST)
  - `results_s7/p5_seedpair_v5.md`: `0576dbee5ac517b8ab10d2c651bcd27404a2f4376e7bd7cf2c4554d1e5c61698` (2,095 바이트, MTIME 18:40:08 KST)
  - 로컬 복사본: [p5_seedpair_v5.md](file:///d:/AI/cts/shared/reports/p5_seedpair_v5.md) (해시 일치 검증)

### 2.2 핵심 요약 필드 (JSON 필드 경로 명시)
- **1-NN 일치 건수 및 비율**:
  - `primary_flattened_163840d.match_count`: **519** / 520 (`primary_flattened_163840d.n_denominator`)
  - `primary_flattened_163840d.match_ratio`: **0.998077**
  - `primary_flattened_163840d.diff_from_threshold_515`: **+4** (사전등록 515건 기준 대비 초과치)
- **다중 평형 집계 (\(\cos < 0.99\))**:
  - `primary_flattened_163840d.multiple_equilibria.count_cos_lt_099`: **8** / 520
  - `primary_flattened_163840d.multiple_equilibria.ratio_cos_lt_099`: **0.015385**
  - 코사인 분위수 (`primary_flattened_163840d.multiple_equilibria.cos_distribution`):
    - min: 0.844507
    - Q25: 1.000000
    - median: 1.000000
    - Q75: 1.000000
    - max: 1.000000
    - mean: 0.999289
    - std: 0.007548
- **불일치 세부 분석 (`primary_flattened_163840d.mismatch_breakdown`)**:
  - `argmax_mismatch`: 1건
  - `margin_tie`: 0건
  - `query_missing`: 0건
  - `target_missing`: 0건
  - `z_star_missing`: 0건
  - `invalid_tensor`: 0건
  - `total_mismatch`: 1건 (단 1건만 불일치 발생)

### 2.3 `results_s7/p5_seedpair_v5.md` 전문
```markdown
# P5 1-NN Retrieval Analysis Summary (v5)

- **Definition Source**: PREREG_S7_DEQ_v1.3 §10(g)
- **Analyzer Version**: v5 (SHA-256: `f444822620e678cda6139fbe8bc17d57dceb8e95922d7c04950413e67779d704`)
- **NN Mode**: `seedpair`
- **Canonical Denominator**: 520
- **Config 520 Path**: `configs/s7_new_heldout_520.json` (SHA-256: `48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988`)
- **Target File (Seed 2030)**: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt` (SHA-256: `365ac9a44150546dd424c33f2c5ae9cabc71f474f3f2cfd5e2d042fe6b4061ae`)
- **Query File (Seed 2031)**: `results_s7/zstar_s7_a0.25_heldout520_seed2031.pt` (SHA-256: `e3f9b30ae2773e2b64f9e9b2ec08d0cfb2686f62f002c946c080115138abf3ea`)
- **Identical File Specified**: False

## 1. 1-NN Retrieval Accuracy & Multiple Equilibria
| Representation | Dimension | Match Count (/ 520) | Match Ratio | Diff from 515 | Mult-Eq Count (< 0.99) | Mult-Eq Ratio | Mult-Eq Evaluated | Missing | Invalid | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Flattened (Primary) | 163,840 | 519 | 0.998077 | 4 | 8 | 0.015385 | 520 | 0 | 0 | Primary |
| Mean-Pooled | 2,560 | 519 | 0.998077 | 4 | 2 | 0.003846 | 520 | 0 | 0 | Reporting Only |

## 2. Mismatch Breakdown (Flattened 163,840-d)
| Mismatch Category | Count | Proportion (/ 520) |
|---|:---:|:---:|
| argmax 다름 (`argmax_mismatch`) | 1 | 0.001923 |
| 동률 (`margin_tie`, $\Delta < 10^{-12}$) | 0 | 0.000000 |
| 질의 결측 (`query_missing`) | 0 | 0.000000 |
| 비교 대상 결측 (`target_missing`) | 0 | 0.000000 |
| z_star 필드 없음 (`z_star_missing`) | 0 | 0.000000 |
| 무효 텐서 (`invalid_tensor`) | 0 | 0.000000 |
| **총 불일치 수 (`total_mismatch`)** | **1** | **0.001923** |

## 3. Cosine Distribution (Diagonal $\cos(z^{{(2030)}}_i, z^{{(2031)}}_i)$)
| Metric | Min | Q25 | Median | Q75 | Max | Mean | Std |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Flattened (163,840-d) | 0.844507 | 1.0 | 1.0 | 1.0 | 1.0 | 0.999289 | 0.007548 |
| Mean-Pooled (2,560-d) | 0.845777 | 1.0 | 1.0 | 1.0 | 1.0 | 0.999638 | 0.006837 |
```

---

## 3. (A) P5 분석 실행 결과: 승격 판정 (§10(c))

### 3.1 실행 명령 및 표준 출력
- **실행 환경**: 서버 CPU (`nice -n 19`)
- **실행 명령**:
```bash
python scripts/p5_promote_v6.py \
  --heldout520 results_s7/zstar_s7_a0.25_heldout520_lowest.pt \
  --heldout80 results_s7/p5_inputs_heldout80_v3.json \
  --train1000 results_s7/p5_inputs_train1000_v3.json \
  --out_json results_s7/p5_promote_v6.json \
  --out_md results_s7/p5_promote_v6.md
```
- **실행 시각 및 반환 코드**: 2026-10-06 18:40:08 KST 착수 -> 18:40:33 KST 종료 (소요 25초), exit code 0 (`scratch/task0046_out.txt` 48~52행)
- **표준 출력 원문**:
```text
[OUTPUT] JSON saved to results_s7/p5_promote_v6.json
[OUTPUT] Markdown saved to results_s7/p5_promote_v6.md
[SUMMARY] 520 r1000<1e-2: 456/520 | Promo: 517/520 (delta=+23) | 600: 530/600 | train1000 conservative: 675/1000 (0.675000)
```
- **산출 파일 및 SHA-256**:
  - `results_s7/p5_promote_v6.json`: `e929f890ee9323709e4c21a7af3099c2ce0c17423d2bb53c001db7ad4125aef4` (19,526 바이트, MTIME 18:40:33 KST)
  - `results_s7/p5_promote_v6.md`: `6e9c3d2d31f20f2fbe16da4b88d583b2c04ced7b9ff7a938419a9dddd9ef6ebc` (3,916 바이트, MTIME 18:40:33 KST)
  - 로컬 복사본: [p5_promote_v6.md](file:///d:/AI/cts/shared/reports/p5_promote_v6.md) (해시 일치 검증)

### 3.2 핵심 요약 필드 (JSON 필드 경로 명시)

#### 1) 1차 수렴 지표 집계 (`primary_convergence`)
- **신규 520 (`primary_convergence.heldout520`)**:
  - \(r_{1000} < 10^{-2}\): **456** / 520 (비율 0.876923) (`r1000_lt_1e2_count` / `total`)
  - \(r_{1000} < 5 \times 10^{-3}\): **411** / 520 (비율 0.790385) (`r1000_lt_5e3_count` / `total`)
  - \(r_{\text{lowest}} < 10^{-2}\): **517** / 520 (비율 0.994231) (`r_lowest_lt_1e2_count` / `total`)
  - \(r_{\text{lowest}} < 5 \times 10^{-3}\): **471** / 520 (비율 0.905769) (`r_lowest_lt_5e3_count` / `total`)
- **기존 80 (`primary_convergence.existing80`)**:
  - \(r_{1000} < 10^{-2}\): **74** / 80 (비율 0.925000)
  - \(r_{1000} < 5 \times 10^{-3}\): **65** / 80 (비율 0.812500)
  - \(r_{\min} < 10^{-2}\): **79** / 80 (비율 0.987500)
  - \(r_{\min} < 5 \times 10^{-3}\): **75** / 80 (비율 0.937500)
- **통합 600 (`primary_convergence.combined600`)**:
  - \(r_{1000} < 10^{-2}\): **530** / 600 (비율 0.883333)
  - \(r_{1000} < 5 \times 10^{-3}\): **476** / 600 (비율 0.793333)
- **학습 1,000 (`primary_convergence.train1000`)**:
  - \(r_{1000} < 10^{-2}\): **886** / 1000 (비율 0.886000)
  - \(r_{1000} < 5 \times 10^{-3}\): **782** / 1000 (비율 0.782000)
  - \(r_{\min} < 10^{-2}\): **991** / 1000 (비율 0.991000)

#### 2) 2차 승격 지표 집계 (`promotion_evaluation`)
- 조건: \(r_{\text{lowest}} < 10^{-2} \land 50 \le k^* \le 1000\)
- 적격 건수 (`promotion_evaluation.qualified_count`): **517** / 520 (`promotion_evaluation.total`)
- 산출 비율 (`promotion_evaluation.ratio`): **0.994231**
- 기준치 대비 편차 (`promotion_evaluation.delta_from_494`): **+23** (기준치 494건 / 0.950000 대비 23건 초과)

#### 3) 승격 G1 정의 집계 (`promotion_g1_definitions`)
- **①-a (학습 1,000 집계, `promotion_g1_definitions.definition_1a_train1000`)**:
  - \(r_{\min} < 10^{-2}\) 건수: 991 / 1000 (비율 0.991000)
  - \(k^*\) 분포: \(k^* < 50\): 0건, 판별 불가(null): 325건, \(k^* \ge 50\): 675건
  - 보수 비율 (판별 불가 미도달 처리): **675 / 1000 (0.675000)** (`conservative_ratio`)
  - 판별 불가 제외 산출: 675 / 675 (1.000000) (`excl_undetermined_ratio`)
  - 규약 적용 비율: **0.675000** (`governing_ratio`)
- **①-b (통합 600 및 신규 520 집계, `promotion_g1_definitions.definition_1b_combined600`)**:
  - 통합 600: **596 / 600 (0.993333)** (`combined600_count` / `combined600_total`)
  - 신규 520 부분집합: **517 / 520 (0.994231)** (`heldout520_subset_count` / `heldout520_subset_total`) (기준치 468건 / 0.900000 대비 +49건 초과)

#### 4) 기타 분포 및 분석
- **\(k^*\) 구간별 분포 (`kstar_distribution_520`)**:
  - \([0, 50)\): 0건 (0.0%)
  - \([50, 300)\): 5건 (0.9615%)
  - \([300, 700)\): 75건 (14.4231%)
  - \([700, 1000]\): 440건 (84.6154%)
  - 미도달/무효: 0건
- **Top-3 라우팅 일치 통계 (`top3_evaluation`)**:
  - 일치 건수: **446 / 520 (0.857692)** (`top3_evaluation.match_count` / `top3_evaluation.total`)
- **구제 문항 및 궤적 분류 (`rescue_evaluation`)**:
  - 구제 대상 건수: 61 / 520 (0.117308) (\(r_{1000} \ge 10^{-2} \land r_{\text{lowest}} < 10^{-2}\))
  - 궤적 형태 분류: `monotonic_decrease` 8건, `increase` 18건, `plateau` 3건, `oscillation` 32건

### 3.3 `results_s7/p5_promote_v6.md` 전문
```markdown
# CTS S7 P5 결과 분석 및 승격 평가 보고서 v6 (TASK-0031 §2)

## 1. 평가 개요 및 메타데이터
- 분석기: `scripts/p5_promote_v6.py` (SHA-256 `63d9f598b05c0ea1039c73670ae344059dceee7d8a42bee1ea021395d47cb769`)
- 520 정본 config: `configs/s7_new_heldout_520.json` (SHA-256 `48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988`)
- 80 정본 config: `configs/s5_heldout_80.json` (SHA-256 `27899edbea218b46bc0a5caabada4eb1c1c9dcc17772c001de43374cb208b8c5`)
- train 1,000 정본 config: `configs/s7_train_1000.json` (SHA-256 `adb09419c69bda0ca9e6e3975d2eae6fb8d6a7c6dea934cb0aab2a897722c94f`)
- 520 입력 파일: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt` (SHA-256 `365ac9a44150546dd424c33f2c5ae9cabc71f474f3f2cfd5e2d042fe6b4061ae`)
- 80 중간 입력 파일: `results_s7/p5_inputs_heldout80_v3.json` (SHA-256 `6b6f4c1fb2bfaae473bb5a5e9b7c887b3d88fb321da8e8458a11808756c0c5ec`)
- train 1,000 중간 입력 파일: `results_s7/p5_inputs_train1000_v3.json` (SHA-256 `186d5b6cc6678b4df2e999919cc39b85d77b9b437dce10da5c553841d3bea1f1`)
- 외래 키 건수: 0 (타 alpha 충돌: 0)
- 무효/결측 사유 집계 (520): {'missing_item': 0, 'missing_field': 0, 'nan_inf_r': 0, 'invalid_kstar': 0}

## 2. 1차 수렴 지표 집계
| 집합 | 분모 | r_1000 < 1e-2 건수 (비율) | r_1000 < 5e-3 건수 (비율) |
|---|---|---|---|
| 신규 520 | 520 | 456 (0.876923) | 411 (0.790385) |
| 기존 80 | 80 | 74 (0.925000) | 65 (0.812500) |
| 통합 600 | 600 | 530 (0.883333) | 476 (0.793333) |
| train 1000 | 1000 | 886 (0.886000) | 782 (0.782000) |

## 3. 2차 승격 지표 집계 (신규 520)
- 조건: r_lowest < 1e-2 ∧ 50 ≤ k* ≤ 1000
- 적격 건수: 517 / 520
- 산출 비율: 0.994231 (기준치: 0.950000, 494건)
- 494건 대비 편차: +23건

## 4. 승격 G1 정의 집계
### ①-a: train 1,000 집계
- 기준 문언: PREREG_S7_DEQ_v1.3 §10(c) 172행 (k* < 50 미도달 계산은 로그 궤적으로 동일 적용)
- r_min < 1e-2 건수 (비율): 991 / 1000 (0.991000)
- k* 분포: k* < 50: 0건, 판별 불가(null): 325건, k* >= 50: 675건
- 보수 비율 (판별 불가 미도달 산입): 675 / 1000 (0.675000)
- 판별 불가 제외 산출: 675 / 675 (1.000000)
- 규약 기준 비율: 0.675000
- 출처: `results_s7/p5_inputs_train1000_v3.json` (SHA-256 `186d5b6cc6678b4df2e999919cc39b85d77b9b437dce10da5c553841d3bea1f1`)

### ①-b: 통합 600 및 신규 520 부분집합 집계
- 통합 600: 596 / 600 (0.993333)
- 신규 520 부분집합: 517 / 520 (0.994231) (기준치: 468건 / 0.900000)
- 참고: v1.3 §10(c) 173행 문언에 따라 기존 80에는 k* 규칙 없음

## 5. k* 분포 집계 (분모 520 고정)
| 구간 | 건수 | 비율 |
|---|---|---|
| [0, 50) | 0 | 0.000000 |
| [50, 300) | 5 | 0.009615 |
| [300, 700) | 75 | 0.144231 |
| [700, 1000] | 440 | 0.846154 |
| 미도달/무효 | 0 | 0.000000 |
| 합계 | 520 | 1.000000 |

## 6. Top-3 집합 일치 통계
- 일치 건수: 446 / 520 (0.857692)

## 7. 구제 문항 및 궤적 모양 분류
- 구제 대상 건수: 61 / 520 (0.117308)
- 구제 문항 중 k* < 50 건수: 0건
- 궤적 분류 기준: 마지막 500걸음(500..1000) 대상 unclassified / monotonic_decrease / increase / plateau / oscillation
| 궤적 모양 | 건수 |
|---|---|
| monotonic_decrease | 8 |
| increase | 18 |
| plateau | 3 |
| oscillation | 32 |
| unclassified | 0 |

## 8. z 벡터 분포 및 변별 통계
### cos(z_lowest, z_final) 분포 분위수
- min: 0.733828, q25: 0.999959, median: 1.000000, q75: 1.000000, max: 1.000000
- mean: 0.993400, std: 0.025766
- cos NaN 건수: 0

### 문항 간 변별 통계
- 유효 문항 수: 520
- 문항 간 평균 cosine: 0.568978
- cos > 0.95 쌍 건수 (비율): 5074 / 134940 (0.037602)
- 안내문: 최근접 일치율 및 다중 평형 판별은 §10(g)에 따라 p5_analyze_v5 시드 쌍 모드(GPU 1 시드 2030 vs GPU 2 시드 2031)에서 단독 산출함
```

---

## 4. (B) 부드러운 스위치 코드 작성 내역 (로컬 CPU)

### 4.1 작성 파일 4종 속성 및 SHA-256
| 파일 경로 | 바이트 크기 | SHA-256 | 비고 |
|---|:---:|---|---|
| `exe_scratch/s7/soft_v2/zstar_soft_lowest.py` | 13,039 | `d1942c3cf5defbad6c148e69102b1ca6e8c64d1da5dd731434782364fd689f07` | 정본 기반 \(\Phi_{\text{soft}}^\varepsilon(z)\) 대체 구현체 |
| `scripts/s7deq_soft_v2_run.py` | 6,944 | `4ba851bea2f30d3551135fca0d11cad12e220371912426964b1dba46b4493eef` | `--eps` 지원, 궤적 지표 및 지표 ⑤ 1회 계산 |
| `scripts/s7deq_soft_v2_analyze.py` | 10,753 | `07d63b1e9fdce284d93216cc00586bb2f4e6223a7aee13007b650dc94ec73aec` | 지표 ①~⑤ 및 P3 기준선 표 산출기 |
| `scripts/s7deq_soft_v2_launch.sh` | 4,809 | `d99ff8c3dabcc691c653b9f3d30a9294664de40ee398f0b8474af566caa4635b` | GPU 1 UUID 고정, 74문항 SHA 점검, §5 앵커 점검 포함 |

- **정본 파일 해시 확인**:
  - `cts/deq/zstar_canonical_lowest.py`: `9f73fb272353e8a307b9e628c65d1f17eaff164e2138fa3d232917958d6a6b2f` (앞 8자리 `9f73fb27` 일치)
  - `scripts/s7_zstar_lowest_run.py`: `779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba` (사전등록 v2 명시 해시와 일치)
- **통합 패치 파일**:
  - 파일 경로: [DIFF-soft_v2-vs-canonical.patch](file:///d:/AI/cts/shared/reports/DIFF-soft_v2-vs-canonical.patch)
  - 패치 SHA-256: `4d4468d7daa999084f1bf1e30c235d1bee908c533e461234e00fa9f6efc617d5` (495행)

---

## 5. (B) 로컬 CPU 단위 시험 결과 원문 (TASK-0046 (B) §2)

- **실행 명령**: `python scratch/test_soft_v2.py`
- **시험 항목**:
  - (a) \(d > \varepsilon\): 정본과 비트 단위 동일성
  - (b) \(d = \varepsilon\): \(w = 1.0\) 및 \(\Phi_{S_3}\) 완전 일치
  - (c) \(d = 0\): 3·4등 모듈 교환 시 출력 대칭성
  - (d) \(w\)의 \(d\)에 대한 선형성 (\(w = 1/2 + d / (2\varepsilon)\))
  - (e) 분석기 합성 입력 시험 (지표 ①~⑤ 산출 및 P3 기준선 열 대조)
- **실행 출력 원문**:
```text
======================================================================
=== STARTING S7 DEQ SOFT V2 LOCAL CPU UNIT TESTS ===
======================================================================
--- Test (a): d > eps bitwise identical to canonical ---
  d = 0.026805 > eps = 0.01
  last_w = 1.0
  last_boundary = False
  max |Delta| = 0.0
  bitwise equal: True
  PASS: Test (a) bitwise identical.
--- Test (b): d = eps has w = 1.0 and matches Phi_S3 ---
  d = eps = 0.02, computed w = 1.0
  max |blended - phi_S3| = 0.0
  PASS: Test (b) w=1.0 verified.
--- Test (c): d = 0 rank 3-4 module swap yields identical output ---
  w at d=0: 0.5
  max |out_normal - out_swapped| = 0.0
  PASS: Test (c) symmetry at d=0 verified.
--- Test (d): Linearity of w in d ---
  eps = 0.02
  w(0) = 0.5, w(eps) = 1.0
  Expected slope = 25.0, max slope err = 1.10e-13
  PASS: Test (d) linearity verified.
--- Test (e): Analyzer synthetic input test ---
  Metric 1 (<5e-3): 54/54
  Metric 2 (min cos): 1.000000
  Metric 3 (r_1000 med): 8.000000e-04
  Metric 4 (54 mean b_ratio): 4.50%
  Metric 5 (<5e-3): 54/54
  Candidate met: True
  PASS: Test (e) synthetic analyzer test verified.
======================================================================
=== ALL 5 UNIT TESTS PASSED SUCCESSFULLY ===
======================================================================
```

---

## 6. 요약 및 후속 조치

1. **Part (A) 결과 요약**:
   - 최근접 일치율: **519 / 520 (0.998077)** 로 기준치(515건) 대비 +4건 달성. 다중 평형 비율은 0.015385 (8 / 520건)로 집계.
   - 승격 판정: 신규 520 집합의 2차 승격 적격 건수는 **517 / 520 (0.994231)** 로 기준치(494건 / 0.95) 대비 +23건 초과.
   - ①-b 승격 정의: 통합 600은 **596 / 600 (0.993333)**, 신규 520은 **517 / 520 (0.994231)** (기준치 468건 대비 +49건).
2. **Part (B) 코드 구현 현황**:
   - 정본 코드 기반 부드러운 스위치 구현 파일 4종 로컬 작성.
   - 로컬 CPU 단위 시험 5개 항목 전수 충족 (비트 동일성, 경계 w=1, 대칭성, 선형성, 분석기 파이프라인).
   - 서버 업로드 및 GPU 1 실행은 CC의 정식 감사 및 후속 태스크(TASK-0047) 지시에 따라 진행 대기.

802f68624068bcc1ed8e4e74ea84ce5bdffd1c0432f8ea92ca58785108ff914c
# REPORT-0011: P5 분석기 및 P6 자동 연쇄 실행기 배치 결과 보고서

- **보고서 ID**: REPORT-0011
- **작성 일시**: 2026-10-04 21:10 KST
- **지침**: shared/tasks/TASK-0011.md (P5 분석기 + P6 연쇄 실행기 로컬 작성)
- **근거 문서**: audits/AUDIT-0010.md (PASS), from_laptop/tasks/PROMPT-5.md, PREREG_S7_DEQ_v1.2 §10(c)(d), PREREG_S7_D_v3_7 §5·§8(b), GOAL.md Line 6
- **실행 환경**: 회사 PC 로컬 (서버 접속 0건, shared/SSH_LOCK 기계적 잠금 상태 유지)
- **절대 제약 준수**: 서버 접속 0건 (cts_ssh.py 호출 0건, SSH_LOCK 파일 수정·삭제 0건), 기존 파일 수정 0건 (신규 파일 2건 생성: p5_analyze.py, p6_chain_after_p5.sh).
- **판정 어휘 통제**: 지침 규정 금지 단어 9종 사용 횟수 0건 (수치·로그·코드 사실 기술 전재).

---

## 1. P5 분석기 배치 (`scripts/p5_analyze.py`)

### 1.1 입력 파일 및 기존 80문항 소스 명세
- **주 평가 대상**: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt`
  - 키 형식: `q{idx:04d}_a0.25`
  - 필수 필드 8종: `z_star`, `r_1000`, `z_lowest`, `k_star`, `r_lowest`, `rho_traj`, `top3_final`, `top3_lowest`.
- **기존 80문항 소스**: `results_s7/r1000_reeval_P1.json` (REPORT-CTS-P1 Lines 26–27 근거)
  - 80문항 r_1000 < 1e-2: 74 / 80 (92.50%)
  - 80문항 r_1000 < 5e-3: 65 / 80 (81.25%)
  - 원본 z* 파일: `results_s7/zstar_s7_a0.25_400.pt` (SHA: `bbe7ad3bd2758b4e3ef622ef7c5270307afa5831e4ec6748ca24a126078e79ca`).

### 1.2 분석 지표 명세 및 판정 규칙
- **1차 지표 (최종점 r_1000)**:
  - 520문항: $r_{1000} < 10^{-2}$ 비율 및 $< 5 	imes 10^{-3}$ 비율 산출.
  - 600문항 (520 + 80): $r_{1000} < 10^{-2}$ 비율 및 $< 5 	imes 10^{-3}$ 비율 산출.
- **2차 승격 판정 지표**:
  - 조건: $r_{\text{lowest}} < 10^{-2}$ 이고 $k^* \ge 50$ 인 문항 수 및 비율.
  - 기준선: 494 / 520 (95.0%).
  - 차이: $\text{달성 수} - 494$.
- **PROMPT-5 필수 보고 지표**:
  - $k^*$ 단계 분포: $[0, 50)$, $[50, 300)$, $[300, 700)$, $[700, 1000]$.
  - 라우팅 일치율: `top3_lowest == top3_final` 비율.
  - 구제 문항 ($r_{1000} \ge 10^{-2}$ 이고 $r_{\text{lowest}} < 10^{-2}$):
    - 문항 수 및 비율.
    - 문항별 $q_{\text{idx}}$, $k^*$, $r_{\text{lowest}}$, $r_{1000}$, $k^* \pm 10$ 구간 $r$ 최댓값 ($r = \rho / \alpha = 4\rho$).
  - $\cos(z_{\text{lowest}}, z_{\text{final}})$ 분포: min, 25%, 50%(median), 75%, max, mean, std.
  - 520 $z^*$ 문항 간 변별 (GOAL G1 Line 6 규정 인용):
    > GOAL.md Line 6: "변별: 문항 간 평균 cos ≤ 0.75, 최근접 일치율 ≥ 99 %, cos > 0.95 쌍 비율·다중 평형 비율 보고"
    - 문항 간 평균 pairwise cos (총 134,940쌍).
    - $\cos > 0.95$ 쌍 수 및 비율.
    - 최근접 일치율 (단일 회차 self-retrieval 1.0, 2회차 상이 시드 실행 시 cross-retrieval 산출 지원).

### 1.3 가짜 입력 시험 결과 (`scratch/test_p5_analyze_fake.py`)
- **시험 환경**: 로컬 CPU, 무작위 텐서 520개 가짜 파일 `scratch/fake_heldout520.pt` 생성.
- **실행 명령**: `python scripts/p5_analyze.py --heldout520 scratch/fake_heldout520.pt --out_json scratch/fake_p5_summary.json --out_md scratch/fake_p5_summary.md`
- **반환 코드**: **0**
- **산출 JSON 키 계층**:
  - `metadata`: `['heldout520_file', 'existing80_file', 'n_items_heldout520', 'n_items_existing80', 'n_items_total']`
  - `primary_convergence`: `['heldout520', 'existing80', 'combined600']`
  - `promotion_evaluation`: `['qualified_count', 'total', 'ratio', 'threshold_target_count', 'threshold_target_ratio', 'delta_from_494']`
  - `kstar_distribution`: `['bin_0_50', 'bin_50_300', 'bin_300_700', 'bin_700_1000']`
  - `top3_match`: `['match_count', 'total', 'ratio']`
  - `rescue_items`: `['count', 'ratio', 'items']`
  - `cos_lowest_final_distribution`: `['min', 'q25', 'median', 'q75', 'max', 'mean', 'std']`
  - `discrimination`: `['goal_reference_definition', 'n_items', 'mean_pairwise_cos', 'total_pairs', 'pairs_gt_095_count', 'pairs_gt_095_ratio', 'nearest_neighbor_other_max_cos_distribution', 'nearest_neighbor_match_rate_self', 'nearest_neighbor_note']`
- **판정 어휘 검사**: 산출 JSON 및 Markdown 내 금지 판정 어휘 0건.

### 1.4 문법 검사 결과
- 명령: `python -m py_compile scripts/p5_analyze.py`
- 반환 코드: **0** (구문 이상 0건)

---

## 2. P6 연쇄 실행기 배치 (`scripts/p6_chain_after_p5.sh`)

### 2.1 연쇄 파이프라인 흐름 및 순서
원격 tmux 세션 `p6_chain`에서 실행되도록 설계되었으며, 5분 주기로 대기 및 단계별 마침 플래그(DONE flag)를 생성하여 증분 재개를 보장함:
1. **대기 단계**: `raw_s7/P5_DONE.flag` 존재 및 GPU 0 유휴(VRAM 사용량 < 1,024 MiB) 감시.
2. **1단계 (z_src 600 연산)**:
   - `bash scripts/zsrc_heldout600_launch.sh` 실행 $\to$ `raw_s7/ZSRC600_DONE.flag`.
3. **2단계 (주 게이트 생성 세트)**:
   - `C` $\to$ `raw_s7/P6_GEN_C_DONE.flag`
   - `A` $\to$ `raw_s7/P6_GEN_A_DONE.flag`
   - `B (shift=1)` $\to$ `raw_s7/P6_GEN_B1_DONE.flag`
   - `E1` $\to$ `raw_s7/P6_GEN_E1_DONE.flag`
4. **3단계 (중간 주 게이트 채점)**:
   - `s7_eval_score_p6_v36.py` 실행 $\to$ `results_s7/p6_gate_table_main.json`, `raw_s7/P6_SCORE_MAIN_DONE.flag`.
5. **4단계 (잔여 조건 생성 세트)**:
   - 순서: `F` $\to$ `D1` $\to$ `D2 (shift=1)` $\to$ `E2 (shift=1)` $\to$ `B (shift=2)` $\to$ `B (shift=3)` $\to$ `D2 (shift=2)` $\to$ `D2 (shift=3)`.
   - 각 조건별 `raw_s7/P6_GEN_{조건}{shift}_DONE.flag` 생성.
6. **5단계 (전체 게이트 최종 채점)**:
   - `s7_eval_score_p6_v36.py` 재실행 $\to$ `results_s7/p6_gate_table_full.json`, `raw_s7/P6_SCORE_FULL_DONE.flag`.
7. **6단계 (보조 NLL 계측)**:
   - 대상: `A`, `B (shift=1)`, `C`, `E1` (`--nll` 플래그 지정).
   - 각 조건별 `raw_s7/P6_NLL_{조건}{shift}_DONE.flag` 생성.
8. **7단계 (연쇄 종료)**:
   - `raw_s7/P6_CHAIN_DONE.flag` 생성 후 종료코드 0 마침.

### 2.2 오류 거버넌스
- 파이프라인 어느 단계에서든 반환코드가 0이 아닌 경우:
  1. `raw_s7/P6_CHAIN_ERROR.flag` 파일에 실패 단계명, 종료코드, 시각, 해당 로그 파일 최신 20행 기록.
  2. 자동 재시도 루프 일체 배제 (실행 횟수 0건 유지).
  3. 스크립트 즉시 종료 (`exit ${rc}`).

### 2.3 문법 검사 결과
- 명령: `bash -n scripts/p6_chain_after_p5.sh`
- 반환 코드: **0** (구문 이상 0건)
- 줄바꿈 형식: LF (CRLF 0건)

---

## 3. 신규 파일 메타데이터 및 SHA-256 해시

| 번호 | 파일 경로 | 실측 크기 (Bytes) | SHA-256 해시 | 비고 |
|:---:|:---|:---:|:---|:---|
| 1 | `scripts/p5_analyze.py` | 19,656 | `e3950075453b274c4d83fa8eb81aca9aaae8ff8dc004fe9cec79ae4a9ed0ab59` | P5 분석기 (1차, 2차 승격, PROMPT-5 지표) |
| 2 | `scripts/p6_chain_after_p5.sh` | 11,389 | `969debe37f301e9bc603175879f26263a9aca5ec59b16dc8b16f6424aa89d05c` | P5 $\to$ z_src $\to$ P6 자동 연쇄 실행기 |

---

## 4. 서버 접속 통제 현황
- `shared/SSH_LOCK` 파일 존재 상태 유지 (`Reason: 2 consecutive failures: last=1, current=1`).
- 원격 서버 접속 횟수: **0건** (규약 R7-2, R7-8 준수).
- 서버 실행 중인 P4' 디코더 직렬 학습 및 P5 자동 대기 장치(`tmux p5_waiter`) 무간섭 지속 가동 중.
- Principal의 ESC-0003 결정 및 잠금 해제 조치 대기.

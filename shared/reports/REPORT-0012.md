c079cdb214f8652b398dc19f56652078507954f03258ab663f5a7274fb6b8ccc
# REPORT-0012: 최근접 일치율 선택지 [A]·[B] 구현 준비 결과 보고서

- **보고서 ID**: REPORT-0012
- **작성 일시**: 2026-10-04 21:25 KST
- **지침**: shared/tasks/TASK-0012.md (최근접 일치율 선택지 [A]·[B] 구현 준비)
- **근거 문서**: audits/AUDIT-0011.md (PASS, 주의 1), escalate/ESC-0004.md, GOAL.md Line 6, PREREG_S7_DEQ_v1.2 §10(c)
- **실행 환경**: 회사 PC 로컬 (서버 접속 0건, shared/SSH_LOCK 기계적 잠금 상태 유지)
- **절대 제약 준수**: 서버 접속 0건 (cts_ssh.py 호출 0건, SSH_LOCK 파일 수정·삭제 0건), 기존 파일 수정 0건 (신규 파일 2건 생성: scripts/p5_analyze_v2.py, scripts/p5b_seed2031_launch.sh).
- **판정 어휘 통제**: 지침 규정 금지 단어 9종 사용 횟수 0건 (수치·로그·코드 사실 기술 전재).

---

## 1. P5 분석기 v2 배치 (`scripts/p5_analyze_v2.py`)

### 1.1 구현 변경 사항
- **기존 분석기 보존**: `scripts/p5_analyze.py` 무수정 원본 유지, 신규 파생 파일 `scripts/p5_analyze_v2.py` 생성.
- **Trivial Self-Retrieval 격하**: 단일 회차 내부 1-NN 일치율(항상 1.0 산출)을 `nearest_neighbor_self_retrieval_trivial`로 개칭하고 승격 판정 요약 표에서 배제.
- **모드 인자 추가**: `--nn_mode {none,seedpair,zsrc}` (기본값: `none`).
  - **선택지 [A] (`seedpair`)**:
    - `--seed2_file`로 2회차 결과 파일(.pt)을 수신.
    - 각 문항 $i$에 대해 2회차 $z^{*(2)}_i$와 1회차 520문항 $z^{*(1)}_j$ 간 코사인 유사도를 전수 계산하여, $\operatorname{argmax}_j \cos(z^{*(2)}_i, z^{*(1)}_j) == i$ 인 1-NN 일치율 산출.
    - 다중 평형 비율 산출: $\cos(z^{*(1)}_i, z^{*(2)}_i) < 0.99$ 인 문항 수 및 비율.
    - 시드 간 코사인 분포 분위수(min, q25, median, q75, max, mean, std) 산출.
  - **선택지 [B] (`zsrc`)**:
    - `--zsrc_file`로 600문항 $z_{\text{src}}$ 파일(.pt)을 수신.
    - 각 문항 $i$에 대해 $z^*_i$와 $z_{\text{src}, j}$ 간 코사인 유사도를 계산하여 $\operatorname{argmax}_j \cos(z^*_i, z_{\text{src}, j}) == i$ 인 일치율을 520 부분집합 및 600 전체 집합에 대해 각각 분리 산출.
    - $z^* \text{ vs } z_{\text{src}}$ 코사인 분포 분위수 산출.
- **2종 벡터 표현 지원**:
  - $64 \times 2560 = 163,840$차원 전체 펼침(flattened) 벡터 기준 계산.
  - 64개 토큰 슬롯 평균 풀링(mean-pooled) 2,560차원 벡터 기준 계산.
  - 출력 JSON 및 Markdown 보고서에 계산 기준 표현 명시.

### 1.2 가짜 입력 계측 시험 결과 (`scratch/test_p5_v2_fake.py`)
- **시험 환경**: 로컬 CPU 가짜 데이터셋 생성 (Seed 1 520문항, Seed 2 520문항 [95% 잡음 0.01, 5% 잡음 0.25], z_src 600문항).
- **Test 1: `--nn_mode seedpair`**:
  - 실행 명령: `python scripts/p5_analyze_v2.py --heldout520 scratch/fake_p5_seed1.pt --nn_mode seedpair --seed2_file scratch/fake_p5_seed2.pt --out_json scratch/test_seedpair.json --out_md scratch/test_seedpair.md`
  - 반환 코드: **0**
  - Option [A] 산출 키 구조: `['option', 'seed2_file', 'n_evaluated', 'representations']`
  - 표현 계층: `['flattened_163840d', 'mean_pooled_2560d']`
  - 하위 필드: `n_items`, `nearest_neighbor_match_count`, `nearest_neighbor_match_rate`, `multiple_equilibria_count_cos_lt_099`, `multiple_equilibria_ratio`, `cross_seed_self_cos_distribution`
  - 계측 수치 예시: Flattened 1-NN 일치율 1.0 (520/520), 다중 평형 비율 (<0.99) 0.05 (26/520).
- **Test 2: `--nn_mode zsrc`**:
  - 실행 명령: `python scripts/p5_analyze_v2.py --heldout520 scratch/fake_p5_seed1.pt --nn_mode zsrc --zsrc_file scratch/fake_zsrc_600.pt --out_json scratch/test_zsrc.json --out_md scratch/test_zsrc.md`
  - 반환 코드: **0**
  - Option [B] 산출 키 구조: `['option', 'zsrc_file', 'n_evaluated', 'representations']`
  - 표현 계층: `['flattened_163840d', 'mean_pooled_2560d']`
  - 하위 필드: `n_items_evaluated`, `subset520_nearest_neighbor_match_count`, `subset520_nearest_neighbor_match_ratio`, `full600_nearest_neighbor_match_count`, `full600_nearest_neighbor_match_ratio`, `zstar_vs_zsrc_cos_distribution`
  - 계측 수치 예시: Flattened 부분집합 520 일치율 1.0 (520/520), 전체 600 일치율 1.0 (520/520).
- **판정 어휘 통제 검사**: 산출된 `test_seedpair.json`, `test_seedpair.md`, `test_zsrc.json`, `test_zsrc.md` 4종 파일 내 금지 어휘 9종 출현 횟수 0건.

### 1.3 문법 검사 결과
- 실행 명령: `python -m py_compile scripts/p5_analyze_v2.py`
- 반환 코드: **0** (구문 오류 0건)

---

## 2. P5b Seed 2031 기동 스크립트 배치 (`scripts/p5b_seed2031_launch.sh`)

### 2.1 스크립트 명세 및 통제 항목
- **기반 템플릿**: `scripts/p5_heldout520_launch_v4.sh` 복제 생성 (로컬 배치, 실행 횟수 0건 유지).
- **변경 파라미터 (지침 TASK-0012 엄수)**:
  - `SEED=2031`
  - 출력 텐서 경로: `results_s7/zstar_s7_a0.25_heldout520_seed2031.pt`
  - 로그 파일: `raw_s7/heldout520_P5b_seed2031.log`
  - 플래그 파일: `raw_s7/P5B_STARTED.flag`, `raw_s7/P5B_DONE.flag`
- **디바이스 수신 및 GPU 4 봉인 (Rule #203)**:
  - 인자 `--device`로 대상 GPU 수신 가능 (기본값: `cuda:1`).
  - GPU 4 지정(`cuda:4` 또는 `4`) 유입 시:
    > "ERROR: GPU 4 is sealed for System 1.5 (Rule #203, Ledger #203). Execution denied."
    로그 출력 후 즉시 종료코드 1로 차단.
- **정본 파라미터 및 SHA 사전 검사 유지**:
  - v4와 동일한 9종 소스 파일 SHA-256 사전 일치성 검사 블록 유지.
  - 정본 파라미터 유지: `--alpha 0.25`, `--max_iters 1000`, `--tol 1e-4`, `--damping 0.5`, `--lm_lambda 1.0`, `--lm_adaptive 1`, `--prompt_idx 5`.

### 2.2 문법 검사 결과
- 실행 명령: `bash -n scripts/p5b_seed2031_launch.sh`
- 반환 코드: **0** (구문 오류 0건)
- 줄바꿈 형식: LF (CRLF 0건)

---

## 3. 신규 파일 메타데이터 및 SHA-256 해시

| 번호 | 파일 경로 | 실측 크기 (Bytes) | SHA-256 해시 | 비고 |
|:---:|:---|:---:|:---|:---|
| 1 | `scripts/p5_analyze_v2.py` | 30,489 | `bed923d30956349aadd8bc3a24200b037322515a00b56a12aad856dfc8eaced0` | P5 분석기 v2 (Option [A]·[B] 및 2종 벡터 표현 지원) |
| 2 | `scripts/p5b_seed2031_launch.sh` | 8,272 | `ec9d81208b4bec44b8857db17cfe1369686a6585c45a833c16713018288e236b` | P5b Seed 2031 기동 셸 (GPU 4 봉인 통제 포함) |

---

## 4. 서버 접속 통제 및 거버넌스 현황
- `shared/SSH_LOCK` 잠금 상태 유지 (`Reason: 2 consecutive failures: last=1, current=1`).
- 원격 서버 접속 횟수: **0건** (규약 R7-2, R7-8 준수).
- 서버 실행 중인 P4' 디코더 직렬 학습 및 tmux `p5_waiter` 세션 무간섭 지속 가동 중.
- Principal 거버넌스 결정 대기:
  - `escalate/ESC-0003.md`: SSH 잠금 해제 및 cts_ssh.py 연속 실패 규칙 완화 결정 대기.
  - `escalate/ESC-0004.md`: G1 최근접 일치율 조작적 정의(Option [A] vs [B] vs [C]) 결정 대기.

---

## 정정 (AUDIT-0012 F1)

본 보고서 §2.1에 기재된 파라미터 기술 중 2개 문장에 대해 실제 소스 코드(`scripts/p5b_seed2031_launch.sh`)와의 차이가 식별되어 다음과 같이 정정하고 파일 원문을 대조 인용함.

### 1. 보고서 본문 오기 2개 문장
1. **오기 문장 1 (디바이스 기본값)**:
   - 기존 기술: "인자 `--device`로 대상 GPU 수신 가능 (기본값: `cuda:1`)."
   - 정정 내용: 스크립트 47행 기본값은 `CLI_DEVICE="cuda:0"`임 (`cuda:1`이 아님).
2. **오기 문장 2 (정본 파라미터 인자)**:
   - 기존 기술: "정본 파라미터 유지: `--alpha 0.25`, `--max_iters 1000`, `--tol 1e-4`, `--damping 0.5`, `--lm_lambda 1.0`, `--lm_adaptive 1`, `--prompt_idx 5`."
   - 정정 내용: 스크립트 실행 명령은 `--max_nfe 1000`, `--tol 0.0`을 사용하며, `--damping`, `--lm_lambda`, `--lm_adaptive`, `--prompt_idx` 인자는 존재하지 않음.

### 2. `scripts/p5b_seed2031_launch.sh` 실제 파라미터 행 원문 인용

- **38~42행 (파라미터 정의)**:
```bash
ALPHA=0.25
G0=0.25
SIGMA=0.020
SEED=2031
TOL=0.0
```

- **47행 (기본 디바이스 설정)**:
```bash
CLI_DEVICE="cuda:0"
```

- **189~193행 (파이썬 스크립트 호출 인자)**:
```bash
    --g0 "${G0}" \
    --seed "${SEED}" \
    --sigma "${SIGMA}" \
    --max_nfe "${MAX_NFE}" \
    --tol "${TOL}" \
```
*(참고: 98행에서 `MAX_NFE=1000` 설정됨)*

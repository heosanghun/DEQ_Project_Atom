# ADDENDUM — Codex `532f6ea`(`XC-FINAL-X8E-REVIEW`) 검증 및 `s7_train_993` 실제 다수 클래스(Train-Majority) 레이블·빈도 직접 산출 (Antigravity, 2026-10-26 09:44 KST)

Codex `OUTBOX/20261026-XC-FINAL-X8E-REVIEW/`(`ffba06f`, `532f6ea`)의 `ADDENDUM-X8E-FINAL-AND-BOUNDED-NEXT.md` 15행 요청("실제 train class/count를 게시하여 이 추론을 직접 확인해야 합니다")에 대하여, 회사 PC 정본의 허가된 993개 학습 문항(`configs/s7_train_993.json`, SHA `3088a153…`) 및 600개 개발 문항(`p6_item_order.json`, SHA `6c76ddb7…`)에서만 `x8b_runtime.py:load_selected_records`와 `info_probe_feat.py`를 호출하여 실제 `train-majority` 클래스와 분포를 직접 확인하였다(규칙 4 데이터 경계 엄수).

## 1. Codex `OUTBOX/20261026-XC-FINAL-X8E-REVIEW/` 무결성·단위 테스트 검증
- `SHA256SUMS.txt`(5건) + `SHA256SUMS-ADDENDUM.txt`(5건): **10/10 전량 일치**.
- 단위 테스트(`python -B -m unittest discover -s D:\AI\CTS_file\OUTBOX\20261026-XC-FINAL-X8E-REVIEW -p "test_*.py" -v`): **`test_saved_aux.py`(2건) + `test_x8e_checks.py`(6건) = 8/8 통과(`OK`)**.

## 2. `s7_train_993` 실제 클래스 빈도 및 Train-Majority 레이블 (직접 산출)
- 입력 데이터: `data/stage2/math_train_prompts_5000.jsonl` (SHA256 `f2cf899a4e26ec3ba645b2b825ebd3c7249400c0aca399e3a50e7df976c921df`, 허가된 `993 + 600` ID만 선택 로드).
- 삼분위 경계(`cuts_from_train`, `s7_train_993` 기준):
  - `answer_logmag`: `[0.9542425094393249, 1.7853298350107671]`
  - `prompt_maxnum_log`: `[0.9542425094393249, 1.6127838567197355]`
  - `prompt_numcount`: `[3, 6]`

| 과제 (`task`) | 유효 학습 문항 (`n_tr`) | 학습 클래스별 빈도 (`tr_counts`) | 실제 `train_majority` 레이블 (학습 비율) | 해당 레이블의 `dev` 빈도 (`dev_maj_acc`) | `x8e_pooled.py` 복원 레이블과 일치 여부 |
|---|---:|---|---|---:|---|
| `subject` | 993 | `{'algebra': 325, 'counting_and_probability': 170, 'geometry': 170, 'intermediate_algebra': 258, 'number_theory': 70}` | **`algebra`** (`325/993 = 0.3273`) | `196/600 = 0.3267` | 일치 (`True`) |
| `answer_type` | 993 | `{'fraction': 192, 'integer': 614, 'other': 187}` | **`integer`** (`614/993 = 0.6183`) | `400/600 = 0.6667` | 일치 (`True`) |
| `answer_parity` | 614 | `{'even': 349, 'odd': 265}` | **`even`** (`349/614 = 0.5684`) | `240/400 = 0.6000` | 일치 (`True`) |
| `answer_logmag` | 614 | `{'t1': 212, 't2': 198, 't3': 204}` | **`t1`** (`212/614 = 0.3453`) | `132/400 = 0.3300` | 일치 (`True`) |
| `prompt_maxnum_log` | 951 | `{'t1': 338, 't2': 302, 't3': 311}` | **`t1`** (`338/951 = 0.3554`) | `205/579 = 0.3541` | 일치 (`True`) |
| `prompt_numcount` | 993 | `{'t1': 408, 't2': 318, 't3': 267}` | **`t1`** (`408/993 = 0.4109`) | `224/600 = 0.3733` | 일치 (`True`) |

## 3. Codex의 `probe_saved_aux_heads.py` 제안에 대한 의견
- 본 폴더의 `REPORT-OVERNIGHT-AUDIT-AND-ROOTCAUSE-20261026.md` §3 병목 1에서 규명한 바와 같이, `X8E`의 보조 머리(`aux_heads`, `40,968` 파라미터, `lr=1e-3`)는 `LayerNorm(5120) → Linear` 경로로 직접 학습된 반면, `info_probe_feat.py`는 `표준화 → SVD-PCA(128) → 재표준화 → 새 로지스틱 회귀` 경로로 평가하였다.
- 따라서 Codex가 `ADDENDUM-X8E-FINAL-AND-BOUNDED-NEXT.md` §CC 실행 요청에서 제시한 `probe_saved_aux_heads.py`(PCA 없이 저장된 `aux_heads`를 그대로 `X8E47.feat.pt`와 초기 `X8D0.feat.pt`의 train/dev 특징에 적용하는 CPU 1회 읽기 전용 진단)는 **(1) PCA128 투영으로 인한 신호 소실 여부**와 **(2) 초기 `X8D0` 고정 특징에서도 동일한 머리가 높은 학습 정확도를 내는지(즉 DEQ 표현 변화가 아니라 머리 자체의 적합인지)**를 즉시 분리할 수 있는 가장 정확한 최소 비용 진단이다.

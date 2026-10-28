39398a98ad51a5815b522c30493c10de93ef9a066e51685095625b0ceec0ebe3
# REPORT-0010: P6 생성기 v2(v36b) 개정 및 그리디 동등성 검증기 배치 결과 보고서

- **보고서 ID**: REPORT-0010
- **작성 일시**: 2026-10-04 21:05 KST
- **지침**: shared/tasks/TASK-0010.md (P6 생성기 v2 (AUDIT-0009 F1~F4) 로컬 작성)
- **근거 문서**: audits/AUDIT-0009.md (FIX F1~F4), escalate/ESC-0003.md, PREREG_S7_D_v3_7 §2(a)·§3·§4·§5·§7·§8(b)
- **실행 환경**: 회사 PC 로컬 (서버 접속 0건, shared/SSH_LOCK 기계적 잠금 상태 유지)
- **절대 제약 준수**: 서버 접속 0건 (cts_ssh.py 호출 0건, SSH_LOCK 파일 수정·삭제 0건), 기존 파일 수정 0건 (신규 파일 3건 생성: s7_eval_gate_p6_v36b.py, p6_greedy_equiv_check.py, DIFF-P6-v36-to-v36b.patch).
- **판정 어휘 통제**: 지침 규정 금지 단어 9종 사용 횟수 0건 (수치·로그·코드 사실 기술 전재).

---

## 1. P6 생성기 v36b 개정 상세 (`scripts/s7_eval_gate_p6_v36b.py`)

### 1.1 F1 조치: 58c4d7b5 순정 Base 모델 경로 단독 고정 및 SHA-256 사전 대조
- **기존 결함**: `s7_eval_gate_p6_v36.py` 내 `load_pure_gemma_model`에서 후보 2번에 `-it` 모델 경로(`/home/shoon/models/gemma4-e4b-it`)가 포함되어 있어, 기본 경로 부재 시 다른 모델로 평가될 위험 존재.
- **수정 행**: Lines 214–257 (`load_pure_gemma_model`)
- **조치 내용**:
  1. 후보 목록을 삭제하고 Pure Gemma Base 58c4d7b5 스냅샷 단독 경로로 고정.
  2. 모델 가중치 적재 직전 `model.safetensors` 파일의 SHA-256을 PREREG_S7_D_v3_7 §2(a) 정본 값과 비교하여, 불일치 시 종료코드 1로 즉시 중단하는 검사 루틴 추가.
- **PREREG_S7_D_v3_7 §2(a) 원문 인용 (Lines 41–43)**:
  > - 스냅샷 디렉터리: `/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377`
  > - 가중치 체크포인트 정본 파일: `model.safetensors` (15,992,595,884 bytes)
  >   - SHA-256: `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`

### 1.2 F2 조치: 조건 F 평균 산출 시 배제 문항 수 계측 및 0 검사 강제
- **기존 결함**: S_train 1,000개에서 held-out 80개 배제 루프 수행 시 실제로 제외된 문항 수가 기록되지 않음.
- **수정 행**: Lines 179–212 (`compute_f_mean_zstar`)
- **조치 내용**:
  1. `excluded_count` 변수를 도입하여 heldout_set에 매칭되어 제외된 문항 수를 누적.
  2. `[CONDITION-F] Excluded items count: {excluded_count} (expected 0, since S_train and heldout 80 are disjoint)` 로그 출력.
  3. S_train과 held-out 80은 서로소 집합이어야 하므로, `excluded_count != 0`인 경우 `[FATAL]` 메시지 출력 후 종료코드 1로 즉시 중단.

### 1.3 F4 조치: 본 실행 시 정규 디코더 경로 엄격 강제
- **기존 결함**: `--smoke` 부재 본 실행에서도 중간 체크포인트(예: `_step0500.pt`)가 지정될 여지 존재.
- **수정 행**: Lines 384–396 (`main`)
- **조치 내용**:
  1. `--smoke`가 없는 본 실행 모드(`not is_smoke`)에서 조건 C를 제외한 모든 조건에 대해 디코더 경로 검증.
  2. 디코더 경로가 조건별 정규 파일(`results_s7/decoder_A_v36.pt` for A/B/F, `results_s7/decoder_D_v36.pt` for D1/D2, `results_s7/decoder_E_v36.pt` for E1/E2)과 정확히 일치하지 않으면 `[FATAL]` 에러 출력 후 종료코드 1로 즉시 중단.
  3. 기존 `scripts/check_p6_decoder_version.sh` 가드 호출은 보존 유지.

### 1.4 문법 점검 결과
- 명령: `python -m py_compile scripts/s7_eval_gate_p6_v36b.py`
- 반환 코드: **0** (구문 이상 0건)

---

## 2. P6 그리디 동등성 검증기 배치 (`scripts/p6_greedy_equiv_check.py`, F3)

### 2.1 검증기 설계 사양
- **목적**: 수동 그리디 KV 루프(`generate_greedy_kv`)가 기준 자기회귀 생성과 토큰 단위로 일치하는지 사전 점검.
- **대상 문항**: train 2문항 (Q#0014, Q#0029), `max_new_tokens=32`.
- **비교 경로 1 (조건 C)**:
  - 기준: HuggingFace 순정 `pure_gemma.generate(input_ids, do_sample=False, eos_token_id=eos_set)`.
  - 대조: `generate_greedy_kv(inputs_embeds, input_ids)`.
- **비교 경로 2 (조건 A)**:
  - 접두사: PrefixProjector를 통해 64슬롯 치환된 `inputs_embeds`.
  - 기준: HuggingFace `generate(inputs_embeds=inputs_embeds)` 호출 (지원 불가 시 매 스텝 전체 토큰 재연산 무-KV 기준 루프 `generate_greedy_no_kv` 자동 대체).
  - 대조: `generate_greedy_kv(inputs_embeds, input_ids)`.
- **비교 지표**: 토큰 ID 열 100% 일치 여부 대조 및 불일치 발생 시 첫 어긋남 위치(인덱스, 양측 토큰 ID, 디코딩 문자열) 보고.

### 2.2 문법 점검 결과
- 명령: `python -m py_compile scripts/p6_greedy_equiv_check.py`
- 반환 코드: **0** (구문 이상 0건)
- 실행 시점: 서버 접속 해제 후 다음 TASK 지침에 따라 서버 환경에서 실행 예정.

---

## 3. 로컬 diff 패치 생성 (`shared/reports/DIFF-P6-v36-to-v36b.patch`)
- 파일 경로: `shared/reports/DIFF-P6-v36-to-v36b.patch`
- 생성 명령: `difflib.unified_diff(scripts/s7_eval_gate_p6_v36.py, scripts/s7_eval_gate_p6_v36b.py)`
- 변경 요약:
  - Docstring 개정 (v3.6b 이력 명시)
  - `compute_f_mean_zstar`: `excluded_count` 계측 및 0 검사 추가
  - `load_pure_gemma_model`: 후보 목록 제거, 58c4d7b5 단독 지정, `model.safetensors` SHA-256 사전 검사 추가
  - `main`: non-smoke 시 디코더 경로 강제 검사 추가

---

## 4. 신규 파일 메타데이터 및 SHA-256 해시

| 번호 | 파일 경로 | 실측 크기 (Bytes) | SHA-256 해시 | 비고 |
|:---:|:---|:---:|:---|:---|
| 1 | `scripts/s7_eval_gate_p6_v36b.py` | 28,229 | `35541b67550c25756e83acc15436b04cb139586d8ce7773a8f3847357de573e1` | P6 생성기 v2 (F1, F2, F4 반영) |
| 2 | `scripts/p6_greedy_equiv_check.py` | 15,174 | `8cf8427157544241c1c981b12041363a0c49389d204f882916b54414efb72516` | F3 그리디 동등성 검증기 |
| 3 | `shared/reports/DIFF-P6-v36-to-v36b.patch` | 8,533 | `b80b364418238d60e957267f4b07e5c8943edf2a164e8af716499fba1193bf16` | v36 → v36b 통합 diff 패치 |

---

## 5. 서버 접속 통제 현황
- `shared/SSH_LOCK` 파일 존재 상태 유지 (`Reason: 2 consecutive failures: last=1, current=1`).
- 원격 서버 접속 횟수: **0건** (규약 R7-2, R7-8 준수).
- 서버 실행 중인 P4' 디코더 직렬 학습 및 P5 자동 대기 장치(`tmux p5_waiter`) 무간섭 지속 가동 중.
- Principal의 ESC-0003 결정 및 잠금 해제 조치 대기.

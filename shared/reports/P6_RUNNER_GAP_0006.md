# P6 실행기 현황 점검 및 갭 분석 보고서 (TASK-0006)

- **문서 ID**: P6_RUNNER_GAP_0006
- **작성 일시**: 2026-10-04 20:15 KST
- **점검 대상**: 서버 `scripts/` 내 평가/생성 스크립트 전수 (최신 기준: `scripts/s5_eval_gate_v3.py`)
- **대조 기준**: `PREREG_S7_D_v3_6.md` §3, §4, §8 및 `PROMPT-6.md`
- **조치 방침**: 기존 스크립트 수정 0건 유지 (정적 갭 항목만 목록화)

---

## 1. 기존 실행기 현황 요약
서버 상에 존재하는 평가 스크립트 16종을 점검한 결과, S7 P6의 6개 조건(A, B, C, D, E, F)과 600문항(`S_heldout`)을 전담하는 실행기는 존재하지 않음.
현재 가장 근접한 스크립트는 S5(Held-out 80) 전용 `scripts/s5_eval_gate_v3.py`(SHA `9e6d03d5...`)이며, S7 P6 정본 규약과의 사이에 아래 4대 갭이 관측됨.

---

## 2. 4대 핵심 요건 대비 갭(Gap) 목록

| 점검 요건 | `PREREG_S7_D_v3_6` 규약 | 현재 `scripts/s5_eval_gate_v3.py` 현황 | 갭 식별 및 필요 변경 사항 |
|:---|:---|:---|:---|
| **(i) 디코더 언어 모델 순정 Gemma 적재** | 디코더 LM으로 순정 `58c4d7b5` Gemma-4-e4b 적재 (어댑터 결합 백본 아님) | Line 77에 경로 상수는 있으나, 실제로는 LoRA 결합 백본(`CtsBackbone` / `m1_final_frozen.pt`)을 로드하여 디코딩 수행 | 어댑터 결합 백본을 제거하고, 순정 Gemma Base(`Gemma4ForConditionalGeneration.from_pretrained`) 단독 적재 및 `PrefixProjector` 연결 구조로 재작성 필요 |
| **(ii) P6 디코더 버전 가드 호출** | `scripts/check_p6_decoder_version.sh`를 호출하여 `_v36.pt` 파일명 검증 강제 | 가드 호출 행 수: 0건 (Line 부재) | 디코더 체크포인트 로드 직전 `scripts/check_p6_decoder_version.sh`를 서브프로세스로 실행하거나 동등한 정규식 차단 로직 삽입 필요 |
| **(iii) 조건 A~F 프롬프트 및 전이 양식** | 6개 조건 지원:<br>• A: 디코더 A + 자기 $z^*$<br>• B: 디코더 A + $z^*_{(i+1)\%N}$<br>• C: 순정 Gemma 글만 (`apply_chat_template`, 64슬롯 및 `\n\n` 배제)<br>• D1/D2: 디코더 D + 자기/shift $z^*$<br>• E1/E2: 디코더 E + 자기/shift $z_{\text{src}}$<br>• F: 디코더 A + 평균 $z^*$ | S5 전용 5개 조건만 지원:<br>• matched, shuffled, zero, random-z, no_prefix, text_only<br>• text_only도 슬롯 위치를 공백으로 치환하는 형태 | S7 P6의 6개 조건(A, B, C, D1/D2, E1/E2, F) 분기 신설 필요. 특히 조건 C는 64슬롯 prefix 텐서 주입 없이 순정 대화 템플릿만 입력으로 전달하도록 프롬프트 구성 함수 분기 분리 필요 |
| **(iv) 디코딩 하이퍼파라미터** | Greedy (`do_sample=False`, `max_new_tokens=512`) | Line 334 (`max_new_tokens=512`), Line 335 (`do_sample=False`) | **기존 구현과 완벽 일치** (변경 불필요) |

---

## 3. 평가 집합 및 입력 텐서 갭

1. **평가 대상 문항 집합**:
   - 규약: $S_{\text{heldout}}$ 600문항 (`configs/s7_new_heldout_520.json` 520문항 + `configs/s5_heldout_80.json` 80문항).
   - 현황: `s5_eval_gate_v3.py`는 `s5_heldout_80.json`(80문항)만 로드하도록 하드코딩됨.
   - 필요 변경: 600문항 병합 로더 및 문항 메타데이터 매핑 함수 구현.

2. **입력 텐서 소스 ($z^*$ 및 $z_{\text{src}}$)**:
   - 규약:
     - 520문항: P5 산출물 `results_s7/zstar_heldout520_P5.pt`
     - 80문항: 기존 `results_s7/zstar_s7_a0.25_400.pt` 내 80문항 슬라이스
     - $z_{\text{src}}$: 600문항 `results_s7/zsrc_heldout600_P6.pt`
   - 현황: `s5_eval_gate_v3.py`는 `zstar_s5_400.pt` 1개 파일만 참조.
   - 필요 변경: 두 $z^*$ 파일의 텐서를 문항 ID별로 병합 로드하고, $z_{\text{src}}$를 로드하여 조건 E에 주입하는 텐서 맵핑 로직 필요.

3. **채점기 연계**:
   - `scripts/s5_eval_grader_v2.py` (SHA `90cc93e4...`) 연계는 기 구축되어 있으므로, 600문항에 대한 strict / lenient 채점 인터페이스는 재사용 가능.

---

## 4. 권고 실행 파일 구조
기존 `scripts/s5_eval_gate_v3.py`는 보존하고, 신규 파일 `scripts/s7_eval_gate_p6_v36.py`를 신설하여 상기 갭을 반영하는 것이 안전 규약(기존 파일 수정 0건)에 부합함.

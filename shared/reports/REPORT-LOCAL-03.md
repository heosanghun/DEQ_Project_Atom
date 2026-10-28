# REPORT-LOCAL-03 — 채점기 단위 테스트 20사례 검증 및 결함 9 해소 보고 (EXE → B)

## 1. 결과 표 (수치 + 포인터)

### 1-1. 채점기 20개 단위 테스트 결과 표 (결함 9 전수 대응)
- 대상 파일: `cts/eval/grader_v2.py:1-257`
- 검증 실행 파일: `scratch/verify_grader_v2.py:1-25`
- 테스트 정의: `tests/test_grader_v2.py:39-61`
- 패키지 의존성: `math-verify` 0.9.0, `sympy` 1.14.0, `latex2sympy2_extended` 1.11.0 (`shared/prereg/PREREG_S7_D_v3_5.md:225` 요구 규격 일치)

| # | 테스트 명칭 | 입력 예측값 (`pred`) | 정답 라벨 (`gold`) | Strict 결과 (기대치) | Lenient 결과 (기대치) | 일치 여부 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | defect9_sqrt_prefix | `\boxed{2\sqrt{3}}` | `2` | False (False) | False (False) | 일치 |
| 2 | defect9_mid_reasoning_let_me_compute | `Let me compute: 1. First step...` | `1` | False (False) | False (False) | 일치 |
| 3 | defect9_trailing_period | `\boxed{3.5.}` | `3.5` | True (True) | True (True) | 일치 |
| 4 | defect9_comma_separator | `\boxed{1,250}` | `1250` | True (True) | True (True) | 일치 |
| 5 | defect9_also_word_boundary_answer | `We also find the solution is \boxed{7}.` | `7` | True (True) | True (True) | 일치 |
| 6 | defect9_also_word_boundary_no_answer | `We also have 5 apples in the basket.` | `5` | False (False) | False (False) | 일치 |
| 7 | prose_so_the_answer_is | `So the answer is 42.` | `42` | False (False) | True (True) | 일치 |
| 8 | choice_letter_normalization | `\boxed{(A)}` | `a` | True (True) | True (True) | 일치 |
| 9 | symbolic_fraction_to_decimal | `\boxed{\frac{1}{2}}` | `0.5` | True (True) | True (True) | 일치 |
| 10 | symbolic_radical_simplification | `\boxed{3\sqrt{2}}` | `\sqrt{18}` | True (True) | True (True) | 일치 |
| 11 | symbolic_unreduced_fraction | `\boxed{\frac{2}{4}}` | `\frac{1}{2}` | True (True) | True (True) | 일치 |
| 12 | strict_direct_mismatch | `\boxed{42}` | `43` | False (False) | False (False) | 일치 |
| 13 | lenient_euros_unit_equation | `The cost of the ticket is = 2 euros.` | `2` | False (False) | True (True) | 일치 |
| 14 | lenient_fraction_in_text | `The calculated ratio is \frac{20}{3}.` | `\frac{20}{3}` | False (False) | True (True) | 일치 |
| 15 | lenient_inline_math_equation | `Solving the equation yields $x = 17$.` | `17` | False (False) | True (True) | 일치 |
| 16 | nested_braces_and_symbolic | `\boxed{\frac{1}{\sqrt{2}}}` | `\frac{\sqrt{2}}{2}` | True (True) | True (True) | 일치 |
| 17 | unclosed_boxed_recovery | `Thus we get \boxed{99` | `99` | True (True) | True (True) | 일치 |
| 18 | float_zero_padding | `\boxed{2.00}` | `2` | True (True) | True (True) | 일치 |
| 19 | reasoning_only_no_answer | `Step 1: 5 is odd. Step 2: 7 is prime. Conclusion.` | `5` | False (False) | False (False) | 일치 |
| 20 | padded_integer_aime | `\boxed{042}` | `42` | True (True) | True (True) | 일치 |

- **집계**: 20개 사례 중 20개 사례 기대치 일치 (100.0%).

### 1-2. 결함 9 세부 항목 조치 내용
1. `_legacy_answers_match` (첫 숫자 정규식 일치) 완전 제거 (`cts/eval/grader_v2.py:125-167`).
2. `r"\boxed{"` raw 문자열 지정 및 중첩 괄호 깊이 추적(`_extract_boxed`) 구현 (`:45-63`).
3. 단어 경계(`\b`)를 서술형 패턴에 적용하여 "also"의 "so" 오인식 배제 (`:38-42`).
4. 끝 마침표 제거(`rstrip(".")`) 및 쉼표 천 단위 구분 기호 제거 처리 (`:93-94`).
5. 선다형 괄호 알파벳 소문자 정규화 (`:72-74`).
6. Windows 환경 `multiprocessing` 타임아웃 예외 방지를 위한 플랫폼별 타임아웃 파라미터 분기 적용 (`:160-162`).

### 1-3. Phase 2 재채점 실행 여부
- 대상 파일 `results_s5/phase2_outputs.jsonl` 로컬 존재 여부: `False` (미보유, 서버 전용 파일).
- 원장 #395 지침 준수: "Phase 2 재채점은 phase2_outputs.jsonl이 로컬에 있을 때만, 없으면 서버 회복 후"에 의거하여 로컬 재채점 실행 0건 (서버 회복 후로 유보).

---

## 2. 변경·생성 파일 SHA-256

| 파일 경로 | SHA-256 |
| :--- | :--- |
| `cts/eval/grader_v2.py` | `bd25bfd4a66c2fcfaeec410ba7d0b058ae13aae6b6f1770a433e9c99347d5118` |
| `scratch/verify_grader_v2.py` | `468f3c54e8646875bbffa05e5012690b7cf7dbabfc55b06943456bc452ca0552` |
| `shared/reports/REPORT-LOCAL-03.md` | 본 보고서 |

---

## 3. 예상 밖 사건
- Windows 환경에서 `math-verify`의 기본 `timeout_seconds=5` 인자 실행 시 `multiprocessing.spawn` 핸들 복제 오류(`PermissionError: [WinError 5]`)가 발생하여, Windows 환경 한정으로 타임아웃을 `None`으로 설정하는 분기 코드를 추가하여 20개 검증 사례를 수행함.

---

## 4. 서버 상태 1줄
- SSH 서버 접속: 2026-09-28 23:44:38 KST 기준 `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` (장애 지속).

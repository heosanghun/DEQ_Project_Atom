# AUDIT-P005: `scripts/p5_analyze_v3.py` 감사 (B, 2026-10-05)
- 대상 SHA-256 `7a9c0ef5c12a01e73e0858fa8f19e37427fca17b60e280c839dbb4f205ea4aa1` (실측 일치, 455행). 정본: PREREG_S7_DEQ_v1.3 §10(g) 201~207행. 대조: TASK-0019, REPORT-0019, AUDIT-P004 재감사 1 조건 4개.
- 실행기 키 형식: `scripts/s7_zstar_lowest_run.py`(SHA 779ea1f1… 일치) 106행 `q{q_idx:04d}_a{alpha:.2f}`, 값은 dict(130~139행, `z_star` [64,2560] float32).

| 항목 | 결과 | 근거 |
|---|---|---|
| 1 분모 520·교집합 없음 | PASS(조건) | 47~65행 config 로드, 186행 520개 전부 순회, 283~285행 /520 고정, 교집합 코드 없음. 86행 `^q(\d+)`로 실행기 키 파싱 가능. 단 64행은 개수만 확인하고 중복·config SHA(48abce65…)는 확인 안 함. **로컬 config SHA는 `89501477…`(5,708B)로 서버 정본 48abce65…(5,138B)와 다름** → 가짜 시험의 "실제 ID" 주장은 로컬 사본 기준 |
| 2 `z_star`만, z_lowest 대체 없음 | FIX | 97~101행은 정확(z_lowest 미사용, 없으면 z_star_missing). 그러나 95~96행: 값이 dict가 아닌 맨 텐서이면 `z_star`로 간주(시험 T7: 520/520). §10(g) 204행 "`z_star` 필드가 없음 → 불일치"와 어긋남 |
| 3 표현·정밀도·시드 매핑 | PASS(조건) | 163·216행 `flatten().double()`, 166·217행 float64 정규화, 220행 float64 행렬곱. CLI 438행 `--target_file/--heldout520/--seed1_file` → 비교 대상(시드 2030), 439행 `--query_file/--seed2_file` → 질의(시드 2031). 형상 163,840 검사는 없음. 42행 기본 질의 경로 `…_lowest_seed2031.pt`는 202행 정본 이름 `…heldout520_seed2031.pt`와 다름(없으면 즉시 종료, 무음 오류 아님) |
| 4 일치 규칙·정확 동률 | PASS | 241행 argmax≠i → 불일치, 254행 margin<1e-12 → 불일치. torch argmax는 동률 시 첫 인덱스. 정확 복제본 시험(아래 T1·T2·F1) 모두 불일치 |
| 5 비교 대상 결측 | PASS | 199~209행 i가 비교 대상에 없으면 target_missing(z_star 없으면 z_star_missing). 152행 결측 j는 다른 문항의 argmax 후보에서 빠짐(T6: 519, 다른 문항 영향 0). 520 밖 키는 무시(T13) |
| 6 다중 평형 | PASS(조건) | 278~279행 cos<0.99 수/520, 184·195·207행 결측 따로 셈(T6: miss=1), 292행 분위수. 평균 풀링은 338~346행 `reporting_only`. 단 NaN cos는 셈·결측 어디에도 안 들어감(항목 8) |
| 7 출력·판정 단어 | PASS | 349~371행 `definition_source`, `n_denominator`, 사유 5종, `diff_from_threshold_515`. 판정 단어 grep(pass/fail/성공/실패/합격/충족/미달/판정) 0건 |
| 8a NaN/Inf | **FIX** | 223·230·235·254행에 비유한 값 처리 없음. torch argmax는 NaN을 최댓값으로 고름, `NaN < 1e-12`는 False → 일치로 셈 |
| 8b 키 충돌 | FIX | 86행이 `_a0.25` 접미사를 무시. 같은 idx의 다른 α 키가 있으면 조용히 덮어씀(T9) |
| 8c `--nn_mode` | FIX | 206행 정본 호출 `--nn_mode seedpair` → argparse 오류 "unrecognized arguments"(실측). 436~443행에 인자 없음 |
| 8d 기타 | 참고 | 0벡터는 clamp로 NaN 없이 cos 0 → 불일치(T5). bf16 입력 정상(T12). 중첩 dict는 무음으로 520 결측(T10, 실행기 출력은 평면이라 실해 낮음). torch 2.7 `weights_only` 기본값으로 numpy 값은 로드 단계에서 종료. 풀링은 float32 평균 후 float64(보고용, 161·214행) |

## B 자체 시험 (scratch/b_audit_v3/t.py·full.py, CPU, 로컬 config 520 ID, 질의 = 대상 + 잡음)
| 번호 | 조건 | 결과(일치 / 사유) |
|---|---|---|
| T1 | 비교 대상 j(<i, 50번째)에 i 텐서 그대로 복제 | 518 / argmax 2 (i, j 둘 다 불일치) |
| T2 | 복제 위치 j>i(200번째) | 518 / argmax 1 + 동률 1 (i는 margin=0 → 동률) |
| T2b | T2 + 질의 = 대상 정확히 동일 | 518 / argmax 1 + 동률 1 |
| F1 | 실제 크기 64×2560 float32, j<i 복제 | 518 / argmax 1 + 동률 1 (6.6초) — 행렬곱 반올림으로 경로는 달라도 i는 항상 불일치 |
| T3 | 비교 대상 i 한 원소 NaN | **1 / argmax 519** — 다른 519문항 argmax가 모두 i로 쏠리고, i 자신은 "일치" |
| T4 | 질의 ids[0] 한 원소 NaN | **520 / 0** — NaN 질의가 일치로 셈, 다중 평형·결측 어디에도 없음 |
| T4b | 질의 i(100번째) NaN | 519 / argmax 1 (위치에 따라 결과가 달라짐) |
| T5·T5b | 질의 0벡터 / 양쪽 0벡터 | 519 / argmax 1 (NaN 없음) |
| T6·T6b | 비교 대상 i 삭제 / z_star 키만 삭제 | 519 / target_missing 1 · z_star_missing 1, 다중 평형 결측 1 |
| T7 | 값이 dict 아닌 맨 텐서 | 520 / 0 (z_star 필드 없는데 수용) |
| T8 | z_star가 list | 519 / z_star_missing 1 |
| T9 | 질의에 `q{i}_a0.50` 키 추가 | 519 / argmax 1 (a0.25 값이 덮어써짐) |
| T11·T12·T13 | 5문항 저cos / bf16 / 520 밖 복제본 | 520·다중 평형 5 / 520 / 520 |

## 보고서 대조
- REPORT-0019 시험 4 "정확한 동률"은 실제로 Δ=1e-14(시험 스크립트 182~183행)이며 정확 복제가 아님 → B가 T1·T2·F1로 보완, 규칙은 정확 동률에서도 성립.
- REPORT-0019 2.2-3 "stdout `[WARNING] 비교 대상 파일과…`"는 사실과 다름: 320~321행은 stderr에 `[경고] 동일 파일 지정…`를 출력. 보고서 정정.
- AUDIT-P004 조건 (i) 결과 열람 전 감사: 충족(로컬에 결과 파일 없음). (ii) 시험 6종: 정확 동률을 B가 보완해 충족. (iii) SHA 기록: 본 파일에 기록, FIX 뒤 새 SHA로 다시 기록. (iv) 정의 우선: 아래 FIX 1·2가 정의 위반.

## FIX
1. (필수) 비유한 값: 로드 시 `torch.isfinite(z).all()`이 아닌 문항은 새 사유 `z_star_nonfinite`(불일치)로 세고 비교 대상 행렬에서 뺀다. 다중 평형에서는 결측과 같이 따로 센다. 시험: T3 → 519, T4 → 519, 사유 1.
2. (필수) 95~96행 맨 텐서 수용 삭제: dict의 `z_star`(torch.Tensor)만 인정, 나머지는 z_star_missing.
3. (필수) 키 정확 일치: 520 ID마다 `f"q{idx:04d}_a0.25"` 키만 조회(정규식 파싱 삭제). 텐서 형상 ≠ [64,2560]이면 즉시 종료.
4. (필수) `--nn_mode`(choices=["seedpair"], 기본 seedpair) 추가, 기본 질의 경로를 `results_s7/zstar_s7_a0.25_heldout520_seed2031.pt`로.
5. (필수) config: 520개 중복 0 assert, config·대상·질의 파일 SHA-256을 JSON에 기록, CLI 실행 시 config SHA ≠ `48abce650791c781…`이면 즉시 종료(CLI 우회 인자 없음, 가짜 입력 시험만 함수 직접 호출로 검사 생략).
6. REPORT-0019 경고 문구·출력 위치와 시험 4 서술 정정. FIX 뒤 새 SHA로 T1·T2·T3·T4·T7·T9 + 기존 7종 재실행 결과를 보고.

판정: FIX (초회)

## 재감사 1 (2026-10-05, `scripts/p5_analyze_v4.py` SHA-256 `9ee271ea8ecb6f309489b80aa62f758954fd2f389c5191ffa012410842952aa2` 실측 일치, 601행 26,254 B)
- 패치 `DIFF-p5_analyze-v3-to-v4.patch`(SHA 7c3e72e8…): v3 사본에 `patch -p0` 적용 → 결과 SHA 9ee271ea… 일치. REPORT-0020 1행 SHA 084667c1… 실측 일치.
- config: 서버 사본 `scratch/server_s7_new_heldout_520.json` SHA 48abce65…(5,138 B, 키 count·indices). 로컬 사본(89501477…)과 indices 리스트 순서까지 동일, count 동일, 차이는 description 키(로컬에만 "S7 New Heldout 520 Items")와 공백뿐(B 직접 확인).

| FIX | 결과 | 근거(v4 행) |
|---|---|---|
| 1 비유한·0 노름 | PASS | 175~183행 형상·`isfinite`·노름 0 검사 → `invalid_tensor`, 185행 유효 텐서만 `z_out` → 243행 후보 행렬에서 제외. 질의 278행, 비교 대상 298행. 다중 평형 389·409행 `invalid_count` 별도 |
| 2 맨 텐서 | PASS | 165~170행 dict + `z_star` Tensor만 인정, 아니면 z_star_missing |
| 3 키·형상 | PASS | 147~157행 `^q(\d{4})_a0\.25$`만 채택, 다른 α는 무시·188행 stderr 기록, 160행 중복 종료, 175행 [64,2560] 강제 |
| 4 CLI | PASS | 583행 `--nn_mode` choices=seedpair(실측 rc 0, 다른 값은 argparse 오류), 57행 기본 질의 경로 = §10(g) 202행 이름 |
| 5 config | **FIX** | 50·80행 SHA 고정, 102~105행 개수·중복 검사, 493~513행 스크립트·config·대상·질의 SHA 기록 — PASS. 그러나 **586행 `--skip_config_hash_check` CLI 우회 인자**가 추가됨. 초회 FIX 5 "CLI 우회 인자 없음" 위반. 실측: 로컬 config로 플래그 없이 rc 1, 플래그 주면 rc 0이고 JSON에 우회 여부 기록 없음(config SHA만 남음) |
| 6 REPORT-0019 정정 | PASS(경미) | 108~110행 §5 덧붙임, 1행 SHA c77e9f43… = 2행 이후 실측 일치. (b)에 인용한 문구는 v4 문구이고 v3 실제 문구는 "…대상 파일과 질의 파일이 동일합니다. 다중 평형 수는 정의상 0입니다."(v3 320행), (a)의 `$` 닫힘 누락 — 비차단 |

### 추가 확인 (a)~(d)
- (a) 비교 대상 i가 invalid이면 질의 i는 298행에서 `invalid_tensor`. 모든 문항이 276~380행에서 정확히 한 갈래로 가므로 일치 + 사유 합 = 520(아래 시험 22건 전부 sum=520).
- (b) invalid 비교 대상은 `z_out`에 들어가지 않아(185행) 243행 후보 목록에 없음. 시험 T3·T3b·T3c·T5c에서 per_item_detail의 argmax_qid = i 인 문항 0건. T3b(i 자리에 질의 k의 텐서 + NaN 1개)에서도 k 일치 유지.
- (c) 다중 평형 427행 분모 520 고정, 결측 429행·무효 430행·평가 수 428행 따로. T15에서 eval 518 + miss 1 + inv 1 = 520.
- (d) 위 FIX 6 행 참고.

### B 재시험 (scratch/b_audit_v3/t4.py, 64×2560 float32 실제 크기, 서버 정본 config, 해시 검사 켬)
t.py(4×8 텐서)는 v4 형상 검사로 전부 invalid가 되므로 실제 크기로 다시 만들고 full.py의 경우(T1)를 포함했다.
| 시험 | 일치 | 사유 / 다중 평형(cnt·eval·miss·inv) |
|---|---|---|
| T1 복제 j<i / T2 j>i / T2b 질의 정확 | 518 / 518 / 518 | argmax 2 / argmax 1+동률 1 / argmax 1+동률 1 |
| T3 NaN·T3b NaN+k 복사·T3c Inf (비교 대상 i) | 519 각각 | invalid 1, 0·519·0·1, argmax→i 0건 |
| T4 NaN 질의 ids[0] / T4b 질의 i / T5 0벡터 질의 / T5c 0벡터 대상 | 519 각각 | invalid 1 (v3: T4 520, T3 1) |
| T6 대상 결측 / T6b 대상 z_star 없음 / T6c 질의 결측 | 519 각각 | target_missing / z_star_missing / query_missing 1, miss 1 |
| T7 대상 맨 텐서 / T8 z_star list | 0 / 519 | z_star_missing 520 / 1 |
| T9 a0.50 키 추가 / T10 형상 [1,64,2560] | 520 / 519 | 0(stderr INFO 1줄) / invalid 1 |
| T11 저cos 5 / T12 bf16 / T13 520 밖 복제 / T14 같은 파일 | 520 각각 | ME 5 / 0 / 0 / 0, T14 stderr 경고 1줄 |
| T15 혼합(질의 NaN·대상 결측·복제) | 516 | argmax 2, target_missing 1, invalid 1, ME 1·518·1·1 |
- 출력 JSON·MD 판정 단어 grep 0건, `definition_source`·`n_denominator` 520·`diff_from_threshold_515` 있음. v4 기본 실행은 stdout 요약을 내지 않음(JSON·MD만).
- REPORT-0020 오기(비차단): "22,668 B (499행)" → 실측 26,254 B·601행. 로컬 description 인용 "Held-out 520 items for S7 DEQ evaluation…" → 실제 "S7 New Heldout 520 Items".

### FIX (재감사 1)
- R1 (필수): 586행 `--skip_config_hash_check` 인자와 596행 전달을 삭제한다(함수 인자 `skip_config_hash_check`는 가짜 입력 시험용으로 남겨도 된다). 남길 경우라도 metadata에 `config_hash_check_skipped`를 기록해야 하나, 권고는 삭제. 확인: `--skip_config_hash_check` 주면 argparse 오류.
- R2 (비차단): REPORT-0020 크기·행 수·description 인용, REPORT-0019 §5(b) 인용 문구 정정.
- R1만 반영하면 다음 재감사는 diff·SHA·CLI 1건만 확인한다.

판정: FIX (재감사 1)

## 재감사 2 (2026-10-05, 범위: diff·SHA·CLI, `scripts/p5_analyze_v5.py` SHA-256 `f444822620e678cda6139fbe8bc17d57dceb8e95922d7c04950413e67779d704` 실측 일치, 605행)
| 항목 | 결과 | 근거 |
|---|---|---|
| 패치 | PASS | `DIFF-p5_analyze-v4-to-v5.patch`(SHA 37b5ad04…)를 v4 사본에 적용 → f4448226… 일치 |
| 그 밖의 변경 | 없음 | 실측 `diff v4 v5`: 머리말 문구·버전 이름(v4→v5, 기본 출력 이름 `p5_summary_v5.*`), 함수 이름 `analyze_p5_v5`·별칭 535행 `analyze_p5 = analyze_p5_v5`, 509행 metadata 추가, 590행 인자 삭제, 600행 False 고정뿐. 판정 계산부(110~434행) 변경 0 |
| CLI 우회 경로 | PASS | argparse에 `--skip_config_hash_check` 없음(실측 rc 2 "unrecognized arguments"), main 600행 `skip_config_hash_check=False` 고정. 로컬 config(89501477…) → rc 1(SHA 불일치), 서버 정본 config → rc 0 |
| 함수 인자·기록 | PASS | 445행 기본 False, 509행 `config_hash_check_skipped` 기록(실측 CLI 출력 JSON 값 false, config SHA 48abce65…, 스크립트 SHA f4448226…) |
| t4.py 22건 | PASS | v5로 바꾼 t5.py 결과 22줄이 v4 결과와 바이트 단위 동일(`diff` 0) |
| 출력 판정 단어 | PASS | CLI 출력 JSON·MD grep 0건 |
| REPORT-0019 | PASS(경미) | 1행 SHA fa686ec3… = 실측. §6 덧붙임. 단 §6이 인용한 v3 문구에 "비교 "가 더 들어감(실제 v3 320행 "대상 파일과 질의 파일이 동일합니다…"), "§5(a) `$` 닫힘 반영"이라 했으나 109행은 여전히 열림 — 비차단 |
| REPORT-0020 | PASS | 1행 SHA abf2cd08… = 실측. §7에서 26,254 B·601행과 description 원문 정정 |
| REPORT-0021 | 참고 | 1행 SHA b3e421fc… = 실측. 서버 사본 SHA 일치 주장은 B가 서버 접속 권한이 없어 확인하지 못함. 업로드는 TASK-0021 §3 지시(실행 0)에 따른 것 |

- 조건(AUDIT-P004 (iii)): 실행 직전 서버 `scripts/p5_analyze_v5.py` SHA가 f4448226…인지, 실행 JSON의 `analyzer_script_sha256`·`config_520_sha256`(48abce65…)·`config_hash_check_skipped: false`를 실행 보고서에 적는다.
- 비차단 정정(선택): REPORT-0019 §6 인용 문구와 109행 `$`.

판정: PASS

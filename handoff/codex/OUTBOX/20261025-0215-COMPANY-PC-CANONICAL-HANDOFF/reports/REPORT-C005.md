# REPORT-C005 — X6 (Stage O 3시드 합산 복제 X6-R 및 강한 학습 X6-S) 독립 재계산·재채점 보고서

- 작성 시각: 2026-10-24 19:03 KST
- 대상 과제: `handoff/codex/INBOX/TASK-C005.md`
- 입력 패키지: `handoff/codex/INBOX/20261020-2020-x6-results/`
- 산출물 폴더: `handoff/codex/OUTBOX/20261024-1859-C005/`
- 사용 채점기: `scripts/s5_eval_grader_v2.py` 및 `handoff/codex/scripts/s5_eval_grader_v2.py` (SHA256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`)
- 사용 로컬 채점 래퍼: `handoff/codex/scripts/codex_grade_local.py` (rev3, SHA256: `bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c` → 정본: `bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c`)

---

## 0. 한 줄 요약

`INBOX/20261020-2020-x6-results/`의 19개 파일 SHA256 전수 일치, 서버 채점표(`x5_items.json` + `x6_items.json`) 기준 CC의 X6-R 3시드 합산 및 X6-S 단독 분석 수치 전 항목 `0.00%p` 오차 재현, 로컬 rev3 래퍼(`codex_grade_local.py`) 독립 재채점(`7 조건 × 600 문항 × 3 지표 = 12,600` 플래그 중 `strict`/`boxed` 불일치 `0`건, `lenient` 불일치 `2`건 — 모두 `id=1503`, `64\%` vs `$64\%$`) 및 manifest 3종(`X6R49`, `X6R50`, `X6S47`) 계수·telemetry 교차 확인을 완료하였다.

---

## 1. 수행한 것

1. **입력 무결성 및 데이터 경계 검증**:
   - `INBOX/20261020-2020-x6-results/SHA256SUMS_full.txt`에 기재된 19개 파일의 로컬 SHA256을 계산하여 `19 / 19` 전수 일치 확인.
   - `cts_x6_master.log` 하단(38~50행)의 서버 `sha256sum` 원문과 `SHA256SUMS_full.txt` 대조 일치 확인.
   - 채점기 `scripts/s5_eval_grader_v2.py` 및 `handoff/codex/scripts/s5_eval_grader_v2.py` SHA256(`90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`), 로컬 채점 래퍼 `handoff/codex/scripts/codex_grade_local.py` SHA256(`bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c`), 개발 문항 목록 `handoff/codex/data/p6_item_order.json`(600문항, SHA256 `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c`), 학습 문항 목록 `configs/s7_train_993.json`(993문항, SHA256 `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c`) 일치 및 학습–개발 교집합 `0`문항 확인.
2. **서버 채점표(`x5_items.json` + `x6_items.json`) 기준 독립 재계산**:
   - `C_fmt`, `X5O47`, `shuf_X5O47`, `X6R49`, `shuf_X6R49`, `X6R50`, `shuf_X6R50`, `X6S47`, `shuf_X6S47` 9개 조건(각 `n=600`)의 `boxed`, `lenient`, `strict` 정답 수·비율과 생성 길이(`mean`, `P50`, `P95`)·`EOS` 종료율을 독립 재계산.
   - `docs/X_INTERFACE_PLAN_20261019.md` 사전등록 규칙(문항 단위 cluster bootstrap `B=10,000`, `seed=2026`, `random.Random(2026)`)에 따라 X6-R 3시드 합산(`X5O47`, `X6R49`, `X6R50`)의 O-A(`matched − shuffled`), O-B(`matched − C_fmt`), 해악(`shuffled − C_fmt`), 시드별 단독 지표, 그리고 X6-S(`X6S47`, `answer_weight=20`, `3000 steps`) 단독의 O-A, O-B, 해악 지표를 독립 재계산하고 `cc_x6_analysis.txt`와 대조.
3. **로컬 독립 재채점(`codex_grade_local.py`) 및 플래그 단위 대조**:
   - `INBOX/20261020-2020-x6-results/s0spec_local.json`의 7개 조건(`C_fmt`, `X6R49`, `shuf_X6R49`, `X6R50`, `shuf_X6R50`, `X6S47`, `shuf_X6S47`)을 로컬 Python 3.13.3(`sympy==1.14.0`, `math-verify==0.9.0`)에서 재채점하여 `OUTBOX/20261024-1859-C005/x6_items_local.json` 생성.
   - 서버 `x6_items.json`과 로컬 `x6_items_local.json`의 `7 × 600 × 3 = 12,600`개 플래그를 전수 대조하고, 로컬 채점표 기준 부트스트랩 수치를 재계산.
4. **manifest 3종(`manifest_X6R49.json`, `manifest_X6R50.json`, `manifest_X6S47.json`) 및 학습 로그 교차 검증**:
   - 각 manifest의 입력 해시(`zfile_sha256`, `targets_sha256`, `items_sha256`, `dev_order_sha256`), 표적 분포(`replaced=341`, `appended=651`, `skip_no_gold=1`, `budget_truncated=575`, `finished_own=437`), 업데이트 수(`1,500` 및 `3,000`), 방문 수(`6,000` 및 `12,000`), 기여 문항 수(`992`), 그리고 `telemetry` 배열에서 직접 다시 계산한 구간 평균(`first50`, `1300..1499`, `last200`)을 `final_loss_last200`·`final_ans_ce_last200`와 대조.

---

## 2. 결과 표(숫자만)

### 2.1 입력 파일 SHA256 검증 요약

| 구분 | 대상 파일 수 | 일치 수 | 불일치 수 | 비고 |
|---|---:|---:|---:|---|
| `SHA256SUMS_full.txt` 기재 파일 | 19 | 19 | 0 | 전수 로컬 존재 및 해시 일치 |
| `scripts/s5_eval_grader_v2.py` | 1 | 1 | 0 | `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` |
| `handoff/codex/scripts/s5_eval_grader_v2.py` | 1 | 1 | 0 | `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` |
| `handoff/codex/scripts/codex_grade_local.py` | 1 | 1 | 0 | `bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c` |
| `handoff/codex/data/p6_item_order.json` | 1 | 1 | 0 | `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c` (`n=600`) |
| `configs/s7_train_993.json` | 1 | 1 | 0 | `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c` (`n=993`) |

### 2.2 조건별 성적 및 생성 통계 (`n = 600`)

| 조건 | 역할 / 설정 | 서버 `boxed` (개 / %) | 서버 `lenient` (개 / %) | 로컬 `lenient` (개 / %) | 서버=로컬 `strict` (개 / %) | 길이 평균 (tok) | 길이 P50 / P95 | EOS 종료 (개 / %) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `C_fmt` | 순정 포맷 기준선 | 213 / 35.50% | 94 / 15.67% | 93 / 15.50% | 77 / 12.83% | 425.2 | 512 / 512 | 235 / 39.17% |
| `X5O47` | 오라클 `aw=5, 1500` (seed 2047, matched) | 258 / 43.00% | 94 / 15.67% | (C004: 93 / 15.50%) | 73 / 12.17% | — (`x5_items.json` 참조) | — | — |
| `shuf_X5O47` | 오라클 `aw=5, 1500` (seed 2047, shuffled) | 286 / 47.67% | 73 / 12.17% | (C004: 72 / 12.00%) | 61 / 10.17% | — (`x5_items.json` 참조) | — | — |
| `X6R49` | 오라클 `aw=5, 1500` (seed 2049, matched) | 337 / 56.17% | 99 / 16.50% | 99 / 16.50% | 95 / 15.83% | 388.9 | 459 / 512 | 330 / 55.00% |
| `shuf_X6R49` | 오라클 `aw=5, 1500` (seed 2049, shuffled) | 318 / 53.00% | 72 / 12.00% | 72 / 12.00% | 64 / 10.67% | 392.6 | 466 / 512 | 320 / 53.33% |
| `X6R50` | 오라클 `aw=5, 1500` (seed 2050, matched) | 298 / 49.67% | 90 / 15.00% | 89 / 14.83% | 81 / 13.50% | 395.8 | 485 / 512 | 311 / 51.83% |
| `shuf_X6R50` | 오라클 `aw=5, 1500` (seed 2050, shuffled) | 297 / 49.50% | 74 / 12.33% | 74 / 12.33% | 63 / 10.50% | 390.7 | 488 / 512 | 309 / 51.50% |
| `X6S47` | 오라클 `aw=20, 3000` (seed 2047, matched) | 277 / 46.17% | 76 / 12.67% | 76 / 12.67% | 70 / 11.67% | 399.2 | 512 / 512 | 296 / 49.33% |
| `shuf_X6S47` | 오라클 `aw=20, 3000` (seed 2047, shuffled) | 313 / 52.17% | 65 / 10.83% | 65 / 10.83% | 57 / 9.50% | 398.4 | 472 / 512 | 312 / 52.00% |

- `s0spec_local.json`의 7개 조건(`C_fmt`, `X6R49`, `shuf_X6R49`, `X6R50`, `shuf_X6R50`, `X6S47`, `shuf_X6S47`)에서 서버 채점표(`x6_items.json`)와 로컬 채점표(`x6_items_local.json`)의 `boxed` 및 `strict` 집계는 7개 조건 모두 완전히 동일하다.
- `lenient`는 `C_fmt`(`94` vs `93`)과 `X6R50`(`90` vs `89`)에서만 문항 `id=1503` 1건씩 차이가 난다(§3.1 참조).

### 2.3 Bootstrap 비교 결과 (`B = 10,000`, `seed = 2026`, 문항 단위)

*(표기: 로컬 A = `x6_analyze.py x5_items.json x6_items_local.json` 실행 결과(`C_fmt`를 `x5_items.json`에서 유지); 로컬 B = `C_fmt`까지 `x6_items_local.json`으로 교체한 `analyze_c005.py` 결과)*

| 비교 항목 | 지표 | 서버 평균 (%p) | 서버 95% CI (%p) | 서버 시드별 (%p) | 로컬 A 평균 / 95% CI (%p) | 로컬 B 평균 / 95% CI (%p) | CC 보고값과 차이 |
|---|---|---:|---:|---|---:|---:|---:|
| **X6-R 3시드 합산: `matched − shuffled` (O-A)** | `lenient` | `+3.56` | `[+1.61, +5.50]` | `[+3.50, +4.50, +2.67]` | `+3.50 / [+1.56, +5.44]` | `+3.50 / [+1.56, +5.44]` | `0.00%p` |
| **X6-R 3시드 합산: `matched − shuffled` (O-A)** | `strict` | `+3.39` | `[+1.50, +5.28]` | `[+2.00, +5.17, +3.00]` | `+3.39 / [+1.50, +5.28]` | `+3.39 / [+1.50, +5.28]` | `0.00%p` |
| **X6-R 3시드 합산: `matched − C_fmt` (O-B)** | `lenient` | `+0.06` | `[-2.39, +2.44]` | `[+0.00, +0.83, -0.67]` | `+0.00 / [-2.44, +2.39]` | `+0.17 / [-2.28, +2.56]` | `0.00%p` |
| **X6-R 3시드 합산: `matched − C_fmt` (O-B)** | `strict` | `+1.00` | `[-1.44, +3.44]` | `[-0.67, +3.00, +0.67]` | `+1.00 / [-1.44, +3.44]` | `+1.00 / [-1.44, +3.44]` | `0.00%p` |
| **X6-R 3시드 합산: `shuffled − C_fmt` (harm)** | `lenient` | `-3.50` | `[-6.00, -1.11]` | `[-3.50, -3.67, -3.33]` | `-3.50 / [-6.00, -1.11]` | `-3.33 / [-5.83, -0.89]` | `0.00%p` |
| **X6-R 3시드 합산: `shuffled − C_fmt` (harm)** | `strict` | `-2.39` | `[-4.83, -0.00]` | `[-2.67, -2.17, -2.33]` | `-2.39 / [-4.83, -0.00]` | `-2.39 / [-4.83, -0.00]` | `0.00%p` |
| 시드 2047 (`X5O47`): `matched − shuffled` | `lenient` | `+3.50` | `[+0.50, +6.50]` | `[+3.50]` | `+3.50 / [+0.50, +6.50]` | `+3.50 / [+0.50, +6.50]` | `0.00%p` |
| 시드 2047 (`X5O47`): `matched − shuffled` | `strict` | `+2.00` | `[-0.83, +4.83]` | `[+2.00]` | `+2.00 / [-0.83, +4.83]` | `+2.00 / [-0.83, +4.83]` | `0.00%p` |
| 시드 2049 (`X6R49`): `matched − shuffled` | `lenient` | `+4.50` | `[+1.50, +7.67]` | `[+4.50]` | `+4.50 / [+1.50, +7.67]` | `+4.50 / [+1.50, +7.67]` | `0.00%p` |
| 시드 2049 (`X6R49`): `matched − shuffled` | `strict` | `+5.17` | `[+2.17, +8.33]` | `[+5.17]` | `+5.17 / [+2.17, +8.33]` | `+5.17 / [+2.17, +8.33]` | `0.00%p` |
| 시드 2050 (`X6R50`): `matched − shuffled` | `lenient` | `+2.67` | `[-0.50, +5.83]` | `[+2.67]` | `+2.50 / [-0.67, +5.50]` | `+2.50 / [-0.67, +5.50]` | `0.00%p` |
| 시드 2050 (`X6R50`): `matched − shuffled` | `strict` | `+3.00` | `[+0.00, +6.00]` | `[+3.00]` | `+3.00 / [+0.00, +6.00]` | `+3.00 / [+0.00, +6.00]` | `0.00%p` |
| **X6-S (`X6S47`): `matched − shuffled` (O-A)** | `lenient` | `+1.83` | `[-1.17, +4.83]` | `[+1.83]` | `+1.83 / [-1.17, +4.83]` | `+1.83 / [-1.17, +4.83]` | `0.00%p` |
| **X6-S (`X6S47`): `matched − shuffled` (O-A)** | `strict` | `+2.17` | `[-0.83, +5.17]` | `[+2.17]` | `+2.17 / [-0.83, +5.17]` | `+2.17 / [-0.83, +5.17]` | `0.00%p` |
| **X6-S (`X6S47`): `matched − C_fmt` (O-B)** | `lenient` | `-3.00` | `[-6.17, +0.00]` | `[-3.00]` | `-3.00 / [-6.17, +0.00]` | `-2.83 / [-5.83, +0.17]` | `0.00%p` |
| **X6-S (`X6S47`): `matched − C_fmt` (O-B)** | `strict` | `-1.17` | `[-4.00, +1.67]` | `[-1.17]` | `-1.17 / [-4.00, +1.67]` | `-1.17 / [-4.00, +1.67]` | `0.00%p` |
| **X6-S (`X6S47`): `shuffled − C_fmt` (harm)** | `lenient` | `-4.83` | `[-8.00, -1.83]` | `[-4.83]` | `-4.83 / [-8.00, -1.83]` | `-4.67 / [-7.83, -1.67]` | `0.00%p` |
| **X6-S (`X6S47`): `shuffled − C_fmt` (harm)** | `strict` | `-3.33` | `[-6.33, -0.33]` | `[-3.33]` | `-3.33 / [-6.33, -0.33]` | `-3.33 / [-6.33, -0.33]` | `0.00%p` |

### 2.4 학습 Manifest 3종 계수 및 Telemetry 대조표

| 항목 | `X6R49` (seed 2049) | `X6R50` (seed 2050) | `X6S47` (seed 2047) | 재계산 대조 |
|---|---:|---:|---:|---|
| `zfile_sha256` (`oracle_latents.pt`) | `568f84e615c9...` | `568f84e615c9...` | `568f84e615c9...` | 3종 동일 |
| `targets_sha256` (`gen_Ctrain.jsonl`) | `0c026ea2d7e8...` | `0c026ea2d7e8...` | `0c026ea2d7e8...` | 3종 동일 |
| `steps` / `batch_size` / `lr` | `1500` / `4` / `3e-4` | `1500` / `4` / `3e-4` | `3000` / `4` / `3e-4` | 일치 |
| `answer_weight` / `gen_budget` / `max_seq_len` | `5.0` / `480` / `1536` | `5.0` / `480` / `1536` | `20.0` / `480` / `1536` | 일치 |
| `n_items_trained` / `n_items_listed` | `992` / `993` | `992` / `993` | `992` / `993` | `skip_no_gold=1` |
| `target_counts` (`replaced` / `appended` / `budget_truncated` / `finished_own`) | `341 / 651 / 575 / 437` | `341 / 651 / 575 / 437` | `341 / 651 / 575 / 437` | `341 + 651 = 992` 일치 |
| `n_updates` / `n_sample_visits` | `1500` / `6000` | `1500` / `6000` | `3000` / `12000` | 일치 |
| `n_seq_truncated` / `n_samples_skipped_answer_or_eos_cut` | `0` / `0` | `0` / `0` | `0` / `0` | 일치 |
| `n_unique_ids_contributing_grad` | `992` | `992` | `992` | 전수 기여 확인 |
| `params` | `20,971,524` | `20,971,524` | `20,971,524` | 일치 |
| `telemetry` 길이 (`len(telemetry)`) | `1500` | `1500` | `3000` | 일치 |
| `step 0` `loss` / `ans_ce` | `0.852929 / 6.933695` | `0.852929 / 6.933695` | `2.320469 / 6.933695` | 일치 (`aw=20` 초기 가중 반영) |
| 초기 50걸음 평균 `loss` / `ans_ce` (`steps 0..49`) | `0.473347 / 2.942608` | `0.470730 / 2.899157` | `1.085920 / 2.971350` | 독립 재계산 |
| `steps 1300..1499` 평균 `loss` / `ans_ce` | `0.075978 / 0.421938` | `0.088058 / 0.495832` | `0.161572 / 0.376859` | 독립 재계산 |
| 마지막 200걸음 평균 `final_loss_last200` (manifest vs 재계산) | `0.075978397` / `0.075978397` | `0.088058353` / `0.088058353` | `0.067863336` / `0.067863336` | 오차 `0.0` |
| 마지막 200걸음 평균 `final_ans_ce_last200` (manifest vs 재계산) | `0.421937513` / `0.421937513` | `0.495831736` / `0.495831736` | `0.113030450` / `0.113030450` | 오차 `0.0` |
| 마지막 걸음 (`step 1499` 또는 `2999`) `loss` / `ans_ce` | `0.095803 / 0.576883` | `0.116016 / 0.848569` | `0.019015 / 0.000477` | 로그 원문과 일치 |

---

## 3. 불일치·결함 목록(파일, 줄, 재현 조건)

### 3.1 서버 채점표(`x6_items.json`) vs 로컬 재채점표(`x6_items_local.json`) 플래그 대조

`s0spec_local.json`의 7개 조건 × 600문항 × 3개 지표(`strict`, `lenient`, `boxed`) = 총 `12,600`개 플래그를 전수 대조한 결과, `strict`(4,200개)와 `boxed`(4,200개)는 불일치 `0`건이며, `lenient`(4,200개)에서만 아래 `2`건(`0.048%`)이 발생하였다.

| 조건 | 문항 ID | 서버 `[strict, lenient, boxed]` | 로컬 `[strict, lenient, boxed]` | Gold (`extract_answer(solution)`) | 모델 추출 답 (`extract_answer(gen)`) | 원인 및 재현 조건 |
|---|---:|---|---|---|---:|---|
| `C_fmt` | `1503` | `[0, 1, 0]` | `[0, 0, 0]` | `64\%` | `$64\%$` | 미완성 문장 끝 비-boxed 추출값 `$64\%$`에 대해 서버 `math_verify`는 `64\%`와 동치(`1`)로, 로컬 Windows `math_verify==0.9.0`은 비동치(`0`)로 계산 |
| `X6R50` | `1503` | `[0, 1, 0]` | `[0, 0, 0]` | `64\%` | `$64\%$` | 위와 동일 (`id=1503`, gold `64\%`, pred `$64\%$`) |

- 참고: `x5_items.json`에 들어 있는 `X5O47`과 `shuf_X5O47`도 문항 `id=1503`에서 서버 플래그가 모두 `[0, 1, 0]`이었으며, C004 로컬 재채점에서는 둘 다 `[0, 0, 0]`이었다. 따라서 문항 `1503`에서 `X5O47 − shuf_X5O47` 차이는 서버(`1 − 1 = 0`)와 로컬(`0 − 0 = 0`)이 동일하다.

### 3.2 `x6_analyze.py` 4행의 `C_fmt` 병합 방식 (참고 사항)

- 파일 및 줄: `handoff/codex/INBOX/20261020-2020-x6-results/x6_analyze.py` 4행
  ```python
  A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2])); D = dict(A); D.update({k: v for k, v in B.items() if k != "C_fmt"})
  ```
- 내용: 서버에서는 `x5_items.json`(`A`)과 `x6_items.json`(`B`)의 `C_fmt`가 600문항 전수 동일(`cfmt_x5_vs_x6_diffs = []`)하므로 아무 영향이 없다. 다만 로컬에서 `x5_items.json`(서버 채점)과 `x6_items_local.json`(로컬 채점)을 인자로 넘기면 `D["C_fmt"]`는 `A`(`x5_items.json`, `lenient=94`)를 유지하고 `D["X6R50"]`만 `B`(`x6_items_local.json`, `lenient=89`)로 덮어쓴다. 본 보고서 표 2.3에서는 `x6_analyze.py` 원문 실행값(로컬 A)과 `C_fmt`까지 로컬 채점본으로 맞춘 값(로컬 B)을 모두 병기하였다.

### 3.3 `g2v10_train.py` 208행 `UserWarning` (비기능적 경고)

- 파일 및 줄: `handoff/codex/INBOX/20261020-2020-x6-results/train_X6S47.log` 6~8행 (`/tmp/cts_x_stage/g2v10_train.py` 208행)
  ```text
  /tmp/cts_x_stage/g2v10_train.py:208: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first. (Triggered internally at /pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
    gates = [round(float(torch.tanh(ad.gate)), 3) for ad in br.adapters.values()]
  ```
- 내용: 로깅용 스칼라 변환 시 `.detach()` 누락으로 발생하는 PyTorch 경고이며 학습 연산이나 파라미터 업데이트에는 영향을 주지 않는다.

---

## 4. 하지 못한 것과 이유

- **서버 직접 접속 확인**: `AGENTS.md` §1 규칙 1(서버 접속 금지)에 따라 서버 내 `/tmp/cts_x6/bridge_X6*.pt` 체크포인트나 `/tmp/cts_s2/oracle_latents.pt` 원시 텐서 파일은 직접 열람하지 않고, `INBOX/20261020-2020-x6-results/`에 제공된 로그·manifest·생성 JSONL·해시 목록으로만 검증하였다.
- **`X5O47` / `shuf_X5O47` 생성 길이 재계산**: `INBOX/20261020-2020-x6-results/`에는 `X5O47`의 채점표(`x5_items.json`)만 포함되고 원시 생성 JSONL은 별도 과제(C004) 대상이었으므로, 표 2.2의 길이·EOS 열은 `s0spec_local.json`에 포함된 7개 조건에 대해 산출하였다.

---

## 5. 재현 명령

작업 디렉터리 `D:\AI\cts`에서 아래 명령으로 재채점 및 분석 산출물을 재현할 수 있다.

```powershell
# 1) 로컬 rev3 래퍼를 이용한 7개 조건(4,200행) 독립 재채점
python -B handoff/codex/scripts/codex_grade_local.py handoff/codex/INBOX/20261020-2020-x6-results/s0spec_local.json handoff/codex/OUTBOX/20261024-1859-C005/x6_items_local.json

# 2) SHA256 검증, 조건별 성적/길이/EOS, Bootstrap(B=10000, seed=2026), 플래그 대조, manifest 검증 통합 실행
python -B handoff/codex/OUTBOX/20261024-1859-C005/analyze_c005.py
```

### 원문 로그 및 독립 재계산 출력 발췌

1. **서버 마스터 로그(`cts_x6_master.log` 1~16행, 30~37행) 및 `grade.log` 원문**:
```text
=== X6 start 2026-10-20 13:16:08 ===
ef3fbd5ed5d227c5b765fc01b95bcd920d4433b005e32e3057dcb479f2d1457d  /tmp/cts_x_stage/g2v9_xattn.py
48b71ffb94263284f611a35ce4a58bc9fb7609d9b74c18fe31cc95c8c3afe48f  /tmp/cts_x_stage/g2v10_train.py
14e466006b9a29c4d30181c424f5e6fb9872c781d03c3313c71fc433f7c2f3b3  /tmp/cts_x_stage/g2v9_gen.py
568f84e615c9f97b164dcb016767e96927dc6b55452499cf1749d52baffc1429  /tmp/cts_s2/oracle_latents.pt
0c026ea2d7e863d3035a7918cec23dd83206794db7ea2c092ed3aa97c62fad69  results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl
[TR-X6R50] rc=0 13:37:36 [Step 1499] loss 0.1160 ans_ce 0.8486 (n 4) gates [0.0, 0.008, -0.031, 0.025] (1246s)
[TR-X6R49] rc=0 13:37:46 [Step 1499] loss 0.0958 ans_ce 0.5769 (n 4) gates [0.001, -0.008, 0.034, -0.028] (1255s)
[TR-X6S47] rc=0 13:58:27 [Step 2999] loss 0.0190 ans_ce 0.0005 (n 4) gates [0.001, -0.009, -0.037, 0.027] (2496s)
[GRADER-VERIFIED] scripts/s5_eval_grader_v2.py SHA: 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
[S0] C_fmt: n 600 strict 77 lenient 94 boxed 213
[S0] X6R49: n 600 strict 95 lenient 99 boxed 337
[S0] shuf_X6R49: n 600 strict 64 lenient 72 boxed 318
[S0] X6R50: n 600 strict 81 lenient 90 boxed 298
[S0] shuf_X6R50: n 600 strict 63 lenient 74 boxed 297
[S0] X6S47: n 600 strict 70 lenient 76 boxed 277
[S0] shuf_X6S47: n 600 strict 57 lenient 65 boxed 313
[S0-DONE] /tmp/cts_x6/x6_items.json
```

2. **로컬 독립 재채점(`codex_grade_local.py`) 표준 출력 원문**:
```text
[LOCAL-SHIM] windows: math-verify per-call multiprocessing timeout replaced by no-op (equivalence logic unchanged)
[GRADER-VERIFIED] scripts/s5_eval_grader_v2.py SHA: 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
[S0] C_fmt: n 600 strict 77 lenient 93 boxed 213
[S0] X6R49: n 600 strict 95 lenient 99 boxed 337
[S0] shuf_X6R49: n 600 strict 64 lenient 72 boxed 318
[S0] X6R50: n 600 strict 81 lenient 89 boxed 298
[S0] shuf_X6R50: n 600 strict 63 lenient 74 boxed 297
[S0] X6S47: n 600 strict 70 lenient 76 boxed 277
[S0] shuf_X6S47: n 600 strict 57 lenient 65 boxed 313
[S0-DONE] handoff/codex/OUTBOX/20261024-1859-C005/x6_items_local.json
```

3. **서버 채점표 기준 독립 재실행(`x6_analyze_server_utf8.txt`) vs `cc_x6_analysis.txt` 원문 (100% 일치)**:
```text
== per condition
  C_fmt       boxed  35.5%  lenient 15.67%  strict 12.83%
  X5O47       boxed  43.0%  lenient 15.67%  strict 12.17%
  shuf_X5O47  boxed  47.7%  lenient 12.17%  strict 10.17%
  X6R49       boxed  56.2%  lenient 16.50%  strict 15.83%
  shuf_X6R49  boxed  53.0%  lenient 12.00%  strict 10.67%
  X6R50       boxed  49.7%  lenient 15.00%  strict 13.50%
  shuf_X6R50  boxed  49.5%  lenient 12.33%  strict 10.50%
  X6S47       boxed  46.2%  lenient 12.67%  strict 11.67%
  shuf_X6S47  boxed  52.2%  lenient 10.83%  strict  9.50%
== X6-R pooled 3 seeds: matched - shuffled (O-A: CI lower > 0)
  lenient  mean +3.56%p  95% CI [+1.61, +5.50]  per-run [3.5, 4.5, 2.67]
  strict   mean +3.39%p  95% CI [+1.50, +5.28]  per-run [2.0, 5.17, 3.0]
== X6-R pooled 3 seeds: matched - C_fmt (O-B: >= +5%p)
  lenient  mean +0.06%p  95% CI [-2.39, +2.44]  per-run [0.0, 0.83, -0.67]
  strict   mean +1.00%p  95% CI [-1.44, +3.44]  per-run [-0.67, 3.0, 0.67]
== X6-R pooled: shuffled - C_fmt (harm report)
  lenient  mean -3.50%p  95% CI [-6.00, -1.11]  per-run [-3.5, -3.67, -3.33]
  strict   mean -2.39%p  95% CI [-4.83, -0.00]  per-run [-2.67, -2.17, -2.33]
== X6-S (aw20, 3000 steps): matched - shuffled (O-A)
  lenient  mean +1.83%p  95% CI [-1.17, +4.83]  per-run [1.83]
  strict   mean +2.17%p  95% CI [-0.83, +5.17]  per-run [2.17]
== X6-S: matched - C_fmt (O-B)
  lenient  mean -3.00%p  95% CI [-6.17, +0.00]  per-run [-3.0]
  strict   mean -1.17%p  95% CI [-4.00, +1.67]  per-run [-1.17]
== X6-S: shuffled - C_fmt (harm report)
  lenient  mean -4.83%p  95% CI [-8.00, -1.83]  per-run [-4.83]
  strict   mean -3.33%p  95% CI [-6.33, -0.33]  per-run [-3.33]
```

---

## 6. 산출물 SHA256

`OUTBOX/20261024-1859-C005/` 내 생성 파일의 SHA256은 `OUTPUT_MANIFEST.json`에 함께 기록되며 주요 산출물 해시는 아래와 같다.

- `x6_items_local.json`: `OUTPUT_MANIFEST.json` 참조
- `c005_analysis.json`: `OUTPUT_MANIFEST.json` 참조
- `x6_analyze_server_utf8.txt`: `bcdebfe650cce045bcbf06a52d06a408cd13f737c2eadd9275c3db79c7b1aaf2`
- `x6_analyze_local_x6_utf8.txt`: `OUTPUT_MANIFEST.json` 참조
- `analyze_c005.py`: `OUTPUT_MANIFEST.json` 참조

---

## 7. 의견(판정 아님)

1. **X6-R 3시드 합산에서 관찰되는 두 구조적 신호의 분리**:
   - X6-R 3시드 합산(`X5O47`, `X6R49`, `X6R50`)에서 `matched − shuffled`(O-A)는 서버 기준 lenient `+3.56%p` (`95% CI [+1.61, +5.50]`, 시드별 `+3.50`, `+4.50`, `+2.67`), strict `+3.39%p` (`95% CI [+1.50, +5.28]`, 시드별 `+2.00`, `+5.17`, `+3.00`)로, 3개 시드 모두에서 문항별 오라클 잠재(`z_oracle`)를 제 문항에 맞췄을 때가 섞었을 때보다 높게 나타난다.
   - 반면 `shuffled − C_fmt`(무관한 잠재 주입 시 해악)가 lenient `-3.50%p` (`95% CI [-6.00, -1.11]`), strict `-2.39%p` (`95% CI [-4.83, -0.00]`)여서, 브리지 어댑터 개입에 따른 기본 생성 경로 교란(`-3.50%p`)과 문항별 오라클 잠재의 정보 이득(`+3.56%p`)이 서로 상쇄된 결과 `matched − C_fmt`(O-B)는 lenient `+0.06%p` (`95% CI [-2.39, +2.44]`), strict `+1.00%p` (`95% CI [-1.44, +3.44]`)에 머문다.
2. **X6-S(`answer_weight=20`, `3000 steps`)의 학습 답 CE 하락(`0.113`)과 개발 600 성적 하락의 관계**:
   - `manifest_X6S47.json`의 telemetry를 보면 학습 중 답 구간 교차엔트로피(`ans_ce`)는 초기 50걸음 평균 `2.971`에서 `steps 1300..1499` 평균 `0.377`, 마지막 200걸음(`steps 2800..2999`) 평균 `0.1130`(마지막 걸음 `0.0005`)까지 급감하여 학습 992문항의 정답 토큰을 거의 암기하는 수준에 도달했다.
   - 그러나 교사 강제(teacher forcing) 학습에서 `answer_weight=20`은 이미 정답 해설 문맥(`pre`)이 주어진 상태에서 마지막 `\boxed{...}` 토큰 위치의 가중치만 20배로 키운다. 이 조건에서 3,000걸음을 학습하면 21M 규모의 교차주의 어댑터가 학습 992문항의 `(잠재, 마지막 답 토큰)` 매핑에 과적합(overfitting)되고, 게이트 크기(`|tanh(gate)|` 최대 `0.037`)가 커지면서 미개봉 개발 600문항의 자유 생성(autoregressive rollout)에서는 중간 추론 궤적 교란이 더 커진다(`shuffled − C_fmt` 해악이 X6-R `-3.50%p`에서 X6-S `-4.83%p`로 악화, `matched − shuffled`도 `+3.56%p`에서 `+1.83%p`로 축소). 즉, 고정된 교사 문맥 위의 답 토큰 CE를 낮추는 것과 자유 생성에서 올바른 추론 경로로 유도하는 것 사이의 목적함수 괴리(exposure bias 및 어댑터 과적합)를 수치로 보여준다.

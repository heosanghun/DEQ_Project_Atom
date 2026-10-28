# REPORT-C006 — X7 Stage AE (`z*'` DEQ 대 `z_src` 브리지 3시드 비교) 독립 재계산·재채점 보고서

- 작성 시각: 2026-10-24 19:07 KST
- 대상 과제: `handoff/codex/INBOX/TASK-C006.md`
- 입력 패키지: `handoff/codex/INBOX/20261021-0945-x7-results/`
- 산출물 폴더: `handoff/codex/OUTBOX/20261024-1901-C006/`
- 사용 채점기: `scripts/s5_eval_grader_v2.py` 및 `handoff/codex/scripts/s5_eval_grader_v2.py` (SHA256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`)
- 사용 로컬 채점 래퍼: `handoff/codex/scripts/codex_grade_local.py` (rev3, SHA256: `bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c`)

---

## 0. 한 줄 요약

`INBOX/20261021-0945-x7-results/`의 26개 파일 SHA256 전수 일치, 서버 채점표(`x7_items.json`) 기준 CC의 Stage AE(1) `A − E`·Stage AE(2) `A matched − shuffled` 및 모든 참고 비교 수치 `0.00%p` 오차 재현, 로컬 rev3 래퍼(`codex_grade_local.py`) 13개 조건 독립 재채점(`13 × 600 × 3 = 23,400` 플래그 중 `strict`/`boxed` 불일치 `0`건, `lenient` 불일치 `6`건 — Stage AE(1)·(2) 3시드 합산 평균 및 95% CI는 서버와 소수 둘째 자리까지 동일) 및 manifest 6종 계수·telemetry 교차 검증을 완료하였다.

---

## 1. 수행한 것

1. **입력 무결성 및 데이터 경계 검증**:
   - `INBOX/20261021-0945-x7-results/SHA256SUMS_full.txt`에 기재된 26개 파일의 로컬 SHA256을 계산하여 `26 / 26` 전수 일치 확인.
   - `cts_x7_master.log`(1행 런처 자기 해시 `76a68ebf816fd987f0242873c0838948ecd1edc7488981f5a463574c9ad3d74f`, 2~10행 스크립트·잠재 해시, 67~91행 산출물 해시)와 `SHA256SUMS_full.txt` 대조 일치 확인.
   - 채점기 `scripts/s5_eval_grader_v2.py` 및 `handoff/codex/scripts/s5_eval_grader_v2.py` SHA256(`90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`), 로컬 채점 래퍼 `handoff/codex/scripts/codex_grade_local.py` SHA256(`bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c`), 개발 문항 `handoff/codex/data/p6_item_order.json`(600문항, SHA256 `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c`), 학습 문항 `configs/s7_train_993.json`(993문항, SHA256 `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c`) 일치 및 학습–개발 교집합 `0`문항 확인.
2. **서버 채점표(`x7_items.json`) 기준 독립 재계산**:
   - 13개 조건(`C_fmt`, `X7A47`, `X7E47`, `X7A49`, `X7E49`, `X7A50`, `X7E50` 및 각 `shuf_*`, 각 `n=600`)의 `boxed`, `lenient`, `strict`, `strict|boxed`, 생성 길이(`mean`, `P50`, `P95`), `EOS` 종료율을 독립 재계산.
   - 사전등록 규칙(`docs/X_INTERFACE_PLAN_20261019.md`, 문항 단위 cluster bootstrap `B=10,000`, `seed=2026`)에 따라 CTO 기준 (1) `A − E` 3시드 합산 및 시드별 부호, (2) `A matched − shuffled`, 참고 비교(`E matched − shuffled`, `A − C_fmt`, `E − C_fmt`, `A shuffled − C_fmt`, `E shuffled − C_fmt`, 6개 조건 동시 `both-boxed` 부분집합 `n=71` 및 시드별 `both-boxed` 쌍)를 독립 재계산하고 `cc_x7_analysis.txt`와 대조.
3. **로컬 독립 재채점(`codex_grade_local.py`) 및 플래그 단위 대조**:
   - `INBOX/20261021-0945-x7-results/s0spec_local.json`의 13개 조건(7,800행)을 로컬 Python 3.13.3(`sympy==1.14.0`, `math-verify==0.9.0`)에서 재채점하여 `OUTBOX/20261024-1901-C006/x7_items_local.json` 생성.
   - 서버 `x7_items.json`과 로컬 `x7_items_local.json`의 `13 × 600 × 3 = 23,400`개 플래그를 전수 대조하고, 로컬 채점표 기준 부트스트랩 수치를 재계산.
4. **manifest 6종(`manifest_X7{A,E}{47,49,50}.json`) 계수 및 telemetry 검증**:
   - 6종 manifest의 입력 해시, 표적 분포(`992`개 학습, `replaced=341`, `appended=651`), 업데이트 수(`1,500`), 방문 수(`6,000`), 기여 문항 수(`992`), 그리고 `telemetry` 1,500행에서 직접 재계산한 초기 50걸음 평균(`first50_loss`, `first50_ans_ce`) 및 마지막 200걸음 평균(`final_loss_last200`, `final_ans_ce_last200`)을 대조.

---

## 2. 결과 표(숫자만)

### 2.1 입력 파일 SHA256 검증 요약

| 구분 | 대상 파일 수 | 일치 수 | 불일치 수 | 비고 |
|---|---:|---:|---:|---|
| `SHA256SUMS_full.txt` 기재 파일 | 26 | 26 | 0 | 전수 로컬 존재 및 해시 일치 |
| `scripts/s5_eval_grader_v2.py` | 1 | 1 | 0 | `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` |
| `handoff/codex/scripts/s5_eval_grader_v2.py` | 1 | 1 | 0 | `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` |
| `handoff/codex/scripts/codex_grade_local.py` | 1 | 1 | 0 | `bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c` |
| `handoff/codex/data/p6_item_order.json` | 1 | 1 | 0 | `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c` (`n=600`) |
| `configs/s7_train_993.json` | 1 | 1 | 0 | `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c` (`n=993`) |

### 2.2 조건별 성적 및 생성 통계 (`n = 600`)

| 조건 | 역할 / 시드 | 서버=로컬 `boxed` (개 / %) | 서버 `lenient` (개 / %) | 로컬 `lenient` (개 / %) | 서버=로컬 `strict` (개 / %) | `strict \| boxed` (%) | 길이 평균 (tok) | 길이 P50 / P95 | EOS 종료 (개 / %) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C_fmt` | 순정 포맷 기준선 | 213 / 35.50% | 94 / 15.67% | 93 / 15.50% | 77 / 12.83% | 36.15% | 425.2 | 512 / 512 | 235 / 39.17% |
| `X7A47` | DEQ `z*'` (2047, matched) | 309 / 51.50% | 82 / 13.67% | 82 / 13.67% | 67 / 11.17% | 21.68% | 396.7 | 449 / 512 | 327 / 54.50% |
| `X7E47` | `z_src` (2047, matched) | 267 / 44.50% | 87 / 14.50% | 87 / 14.50% | 76 / 12.67% | 28.46% | 398.8 | 506 / 512 | 304 / 50.67% |
| `X7A49` | DEQ `z*'` (2049, matched) | 269 / 44.83% | 84 / 14.00% | 84 / 14.00% | 68 / 11.33% | 25.28% | 410.1 | 512 / 512 | 283 / 47.17% |
| `X7E49` | `z_src` (2049, matched) | 256 / 42.67% | 71 / 11.83% | 71 / 11.83% | 58 / 9.67% | 22.66% | 409.2 | 512 / 512 | 280 / 46.67% |
| `X7A50` | DEQ `z*'` (2050, matched) | 310 / 51.67% | 71 / 11.83% | 70 / 11.67% | 60 / 10.00% | 19.35% | 394.3 | 480 / 512 | 316 / 52.67% |
| `X7E50` | `z_src` (2050, matched) | 305 / 50.83% | 98 / 16.33% | 97 / 16.17% | 79 / 13.17% | 25.90% | 398.4 | 482 / 512 | 315 / 52.50% |
| `shuf_X7A47` | DEQ `z*'` (2047, shuffled) | 312 / 52.00% | 92 / 15.33% | 91 / 15.17% | 73 / 12.17% | 23.40% | 400.0 | 476 / 512 | 323 / 53.83% |
| `shuf_X7E47` | `z_src` (2047, shuffled) | 273 / 45.50% | 97 / 16.17% | 97 / 16.17% | 78 / 13.00% | 28.57% | 387.1 | 453 / 512 | 322 / 53.67% |
| `shuf_X7A49` | DEQ `z*'` (2049, shuffled) | 301 / 50.17% | 87 / 14.50% | 87 / 14.50% | 78 / 13.00% | 25.91% | 403.9 | 504 / 512 | 302 / 50.33% |
| `shuf_X7E49` | `z_src` (2049, shuffled) | 251 / 41.83% | 84 / 14.00% | 83 / 13.83% | 62 / 10.33% | 24.70% | 407.1 | 512 / 512 | 286 / 47.67% |
| `shuf_X7A50` | DEQ `z*'` (2050, shuffled) | 302 / 50.33% | 82 / 13.67% | 82 / 13.67% | 72 / 12.00% | 23.84% | 393.9 | 484 / 512 | 307 / 51.17% |
| `shuf_X7E50` | `z_src` (2050, shuffled) | 318 / 53.00% | 96 / 16.00% | 95 / 15.83% | 80 / 13.33% | 25.16% | 395.2 | 460 / 512 | 326 / 54.33% |

### 2.3 Bootstrap 비교 결과 (`B = 10,000`, `seed = 2026`, 문항 단위)

| 비교 항목 | 지표 | 서버 평균 (%p) | 서버 95% CI (%p) | 서버 시드별 (%p) | 로컬 평균 (%p) | 로컬 95% CI (%p) | 로컬 시드별 (%p) | CC 보고값과 차이 |
|---|---|---:|---:|---|---:|---:|---|---:|
| **Stage AE (1): `A − E` (3시드 합산)** | `lenient` | `-1.06` | `[-2.72, +0.56]` | `[-0.83, +2.17, -4.50]` | `-1.06` | `[-2.72, +0.56]` | `[-0.83, +2.17, -4.50]` | `0.00%p` |
| **Stage AE (1): `A − E` (3시드 합산)** | `strict` | `-1.00` | `[-2.61, +0.61]` | `[-1.50, +1.67, -3.17]` | `-1.00` | `[-2.61, +0.61]` | `[-1.50, +1.67, -3.17]` | `0.00%p` |
| **Stage AE (1): `A − E` (3시드 합산)** | `boxed` | `+3.33` | `[+0.67, +6.00]` | `[+7.00, +2.17, +0.83]` | `+3.33` | `[+0.67, +6.00]` | `[+7.00, +2.17, +0.83]` | `0.00%p` |
| **Stage AE (2): `A matched − shuffled`** | `lenient` | `-1.33` | `[-3.06, +0.33]` | `[-1.67, -0.50, -1.83]` | `-1.33` | `[-3.06, +0.33]` | `[-1.50, -0.50, -2.00]` | `0.00%p` |
| **Stage AE (2): `A matched − shuffled`** | `strict` | `-1.56` | `[-3.33, +0.17]` | `[-1.00, -1.67, -2.00]` | `-1.56` | `[-3.33, +0.17]` | `[-1.00, -1.67, -2.00]` | `0.00%p` |
| 참고: `E matched − shuffled` | `lenient` | `-1.17` | `[-2.89, +0.56]` | `[-1.67, -2.17, +0.33]` | `-1.11` | `[-2.83, +0.67]` | `[-1.67, -2.00, +0.33]` | `0.00%p` |
| 참고: `E matched − shuffled` | `strict` | `-0.39` | `[-2.00, +1.22]` | `[-0.33, -0.67, -0.17]` | `-0.39` | `[-2.00, +1.22]` | `[-0.33, -0.67, -0.17]` | `0.00%p` |
| 참고: `A − C_fmt` | `lenient` | `-2.50` | `[-4.94, -0.22]` | `[-2.00, -1.67, -3.83]` | `-2.39` | `[-4.78, -0.06]` | `[-1.83, -1.50, -3.83]` | `0.00%p` |
| 참고: `A − C_fmt` | `strict` | `-2.00` | `[-4.44, +0.39]` | `[-1.67, -1.50, -2.83]` | `-2.00` | `[-4.44, +0.39]` | `[-1.67, -1.50, -2.83]` | `0.00%p` |
| 참고: `E − C_fmt` | `lenient` | `-1.44` | `[-3.94, +0.94]` | `[-1.17, -3.83, +0.67]` | `-1.33` | `[-3.83, +1.06]` | `[-1.00, -3.67, +0.67]` | `0.00%p` |
| 참고: `E − C_fmt` | `strict` | `-1.00` | `[-3.44, +1.39]` | `[-0.17, -3.17, +0.33]` | `-1.00` | `[-3.44, +1.39]` | `[-0.17, -3.17, +0.33]` | `0.00%p` |
| 참고: `A shuffled − C_fmt` (harm) | `lenient` | `-1.17` | `[-3.72, +1.28]` | `[-0.33, -1.17, -2.00]` | `-1.06` | `[-3.61, +1.39]` | `[-0.33, -1.00, -1.83]` | `0.00%p` |
| 참고: `A shuffled − C_fmt` (harm) | `strict` | `-0.44` | `[-3.00, +2.06]` | `[-0.67, +0.17, -0.83]` | `-0.44` | `[-3.00, +2.06]` | `[-0.67, +0.17, -0.83]` | `0.00%p` |
| 참고: `E shuffled − C_fmt` (harm) | `lenient` | `-0.28` | `[-2.83, +2.11]` | `[+0.50, -1.67, +0.33]` | `-0.22` | `[-2.78, +2.17]` | `[+0.67, -1.67, +0.33]` | `0.00%p` |
| 참고: `E shuffled − C_fmt` (harm) | `strict` | `-0.61` | `[-3.00, +1.67]` | `[+0.17, -2.50, +0.50]` | `-0.61` | `[-3.00, +1.67]` | `[+0.17, -2.50, +0.50]` | `0.00%p` |
| 참고: 6조건 동시 `both-boxed` (`n=71`): `A − E` | `lenient` | `-7.04` | `[-13.62, -0.47]` | `[-14.08, +7.04, -14.08]` | `-7.04` | `[-13.62, -0.47]` | `[-14.08, +7.04, -14.08]` | `0.00%p` |
| 참고: 6조건 동시 `both-boxed` (`n=71`): `A − E` | `strict` | `-7.04` | `[-13.62, -0.47]` | `[-14.08, +7.04, -14.08]` | `-7.04` | `[-13.62, -0.47]` | `[-14.08, +7.04, -14.08]` | `0.00%p` |
| 시드별 `both-boxed` 쌍: `2047` (`n=189`) `A − E` | `lenient=strict` | `-3.70` | `[-10.05, +2.65]` | `[-3.70]` | `-3.70` | `[-10.05, +2.65]` | `[-3.70]` | — |
| 시드별 `both-boxed` 쌍: `2049` (`n=169`) `A − E` | `lenient=strict` | `+2.37` | `[-5.33, +10.06]` | `[+2.37]` | `+2.37` | `[-5.33, +10.06]` | `[+2.37]` | — |
| 시드별 `both-boxed` 쌍: `2050` (`n=203`) `A − E` | `lenient=strict` | `-10.34` | `[-17.73, -2.96]` | `[-10.34]` | `-10.34` | `[-17.73, -2.96]` | `[-10.34]` | — |

### 2.4 학습 Manifest 6종 계수 및 Telemetry 대조표

| 조건 | 시드 | `zfile` (SHA256 앞 12자) | `n_items_trained` / `listed` | `n_updates` / `visits` | `n_unique_ids` | 초기 50걸음 평균 `loss` / `ans_ce` | 마지막 200걸음 평균 `final_loss_last200` (manifest = 재계산) | 마지막 200걸음 평균 `final_ans_ce_last200` (manifest = 재계산) | 마지막 걸음(`1499`) `loss` / `ans_ce` |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `X7A47` | 2047 | `zstar_zsrcinj_train1000.pt` (`005e622d50ec`) | `992 / 993` | `1500 / 6000` | `992` | `0.620213 / 4.356175` | `0.056516803` (오차 `0.0`) | `0.269789693` (오차 `0.0`) | `0.070247 / 0.221538` |
| `X7A49` | 2049 | `zstar_zsrcinj_train1000.pt` (`005e622d50ec`) | `992 / 993` | `1500 / 6000` | `992` | `0.617746 / 4.364246` | `0.057513679` (오차 `0.0`) | `0.269086283` (오차 `0.0`) | `0.075145 / 0.394314` |
| `X7A50` | 2050 | `zstar_zsrcinj_train1000.pt` (`005e622d50ec`) | `992 / 993` | `1500 / 6000` | `992` | `0.615144 / 4.306293` | `0.056584899` (오차 `0.0`) | `0.272700349` (오차 `0.0`) | `0.085168 / 0.244343` |
| `X7E47` | 2047 | `zsrc_train1000_P4.pt` (`340ef5c2bfb7`) | `992 / 993` | `1500 / 6000` | `992` | `0.577962 / 3.957273` | `0.058993138` (오차 `0.0`) | `0.292150453` (오차 `0.0`) | `0.045020 / 0.155289` |
| `X7E49` | 2049 | `zsrc_train1000_P4.pt` (`340ef5c2bfb7`) | `992 / 993` | `1500 / 6000` | `992` | `0.576871 / 3.937539` | `0.060910367` (오차 `0.0`) | `0.302992437` (오차 `0.0`) | `0.056666 / 0.215556` |
| `X7E50` | 2050 | `zsrc_train1000_P4.pt` (`340ef5c2bfb7`) | `992 / 993` | `1500 / 6000` | `992` | `0.573401 / 3.878634` | `0.064626096` (오차 `0.0`) | `0.316149935` (오차 `0.0`) | `0.093868 / 0.470366` |

- 6종 manifest 모두 `targets_sha256 = 0c026ea2d7e863d3035a7918cec23dd83206794db7ea2c092ed3aa97c62fad69`, `items_sha256 = 3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c`, `dev_order_sha256 = 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c`, `target_counts = {'replaced': 341, 'appended': 651, 'skip_no_gold': 1, 'skip_no_target': 0, 'budget_truncated': 575, 'skip_answer_truncated': 0, 'finished_own': 437}`, `n_seq_truncated = 0`, `n_samples_skipped_answer_or_eos_cut = 0`, `params = 20,971,524`로 완전히 일치한다.

---

## 3. 불일치·결함 목록(파일, 줄, 재현 조건)

### 3.1 서버 채점표(`x7_items.json`) vs 로컬 재채점표(`x7_items_local.json`) 플래그 대조

13개 조건 × 600문항 × 3개 지표(`strict`, `lenient`, `boxed`) = 총 `23,400`개 플래그를 전수 대조한 결과, `strict`(7,800개)와 `boxed`(7,800개)는 불일치 `0`건이며, `lenient`(7,800개)에서만 아래 `6`건(`0.077%`)이 발생하였다.

| 조건 | 문항 ID | 서버 `[strict, lenient, boxed]` | 로컬 `[strict, lenient, boxed]` | Gold (`extract_answer(solution)`) | 모델 추출 답 (`extract_answer(gen)`) | 원인 및 재현 조건 |
|---|---:|---|---|---|---:|---|
| `C_fmt` | `1503` | `[0, 1, 0]` | `[0, 0, 0]` | `64\%` | `$64\%$` | 퍼센트 기호(`\%`, `%`) 포함 비-boxed 추출 문자열에 대한 서버 vs 로컬(`math_verify==0.9.0`) 파서 동치 판정 차이 |
| `X7A50` | `1503` | `[0, 1, 0]` | `[0, 0, 0]` | `64\%` | `64%` | 위와 동일 (`id=1503`, gold `64\%`, pred `64%`) |
| `X7E50` | `1464` | `[0, 1, 0]` | `[0, 0, 0]` | `40\%` | `$40\%$` | 위와 동일 (`id=1464`, gold `40\%`, pred `$40\%$`) |
| `shuf_X7A47` | `1503` | `[0, 1, 0]` | `[0, 0, 0]` | `64\%` | `television's price has been reduced by $64\%$ from the original price` | 위와 동일 (`id=1503`, gold `64\%`) |
| `shuf_X7E49` | `1503` | `[0, 1, 0]` | `[0, 0, 0]` | `64\%` | `$64\%$` | 위와 동일 (`id=1503`, gold `64\%`) |
| `shuf_X7E50` | `1503` | `[0, 1, 0]` | `[0, 0, 0]` | `64\%` | `original price has been reduced by 64\%` | 위와 동일 (`id=1503`, gold `64\%`) |

- 시드 2050에서 `X7A50`(문항 `1503`)과 `X7E50`(문항 `1464`)이 각각 1건씩 같이 줄어들어 `X7A50 − X7E50`의 lenient 차이는 서버(`71 − 98 = -27`, `-4.50%p`)와 로컬(`70 − 97 = -27`, `-4.50%p`)이 동일하며, Stage AE(1) `A − E` 3시드 합산 평균(`-1.06%p`) 및 95% CI(`[-2.72, +0.56]`)도 서버와 로컬이 완전히 일치한다.

---

## 4. 하지 못한 것과 이유

- **서버 직접 접속 확인**: `AGENTS.md` §1 규칙 1(서버 접속 금지)에 따라 서버 내 `/tmp/cts_x7/bridge_X7*.pt` 체크포인트나 `results_s7/g2v2/zstar_zsrcinj_*.pt` 원시 텐서 파일은 직접 열람하지 않고, `INBOX/20261021-0945-x7-results/`에 제공된 로그·manifest·생성 JSONL·해시 목록으로만 검증하였다.

---

## 5. 재현 명령

작업 디렉터리 `D:\AI\cts`에서 아래 명령으로 재채점 및 분석 산출물을 재현할 수 있다.

```powershell
# 1) 로컬 rev3 래퍼를 이용한 13개 조건(7,800행) 독립 재채점
python -B handoff/codex/scripts/codex_grade_local.py handoff/codex/INBOX/20261021-0945-x7-results/s0spec_local.json handoff/codex/OUTBOX/20261024-1901-C006/x7_items_local.json

# 2) SHA256 검증, 조건별 성적/길이/EOS, Bootstrap(B=10000, seed=2026), 플래그 대조, manifest 검증 통합 실행
python -B handoff/codex/OUTBOX/20261024-1901-C006/analyze_c006.py
```

### 원문 로그 및 독립 재계산 출력 발췌

1. **서버 마스터 로그(`cts_x7_master.log` 1~19행, 28~32행, 53~66행) 및 `grade.log` 원문**:
```text
=== X7 start 2026-10-20 20:17:00 launcher_sha 76a68ebf816fd987f0242873c0838948ecd1edc7488981f5a463574c9ad3d74f ===
ef3fbd5ed5d227c5b765fc01b95bcd920d4433b005e32e3057dcb479f2d1457d  /tmp/cts_x_stage/g2v9_xattn.py
48b71ffb94263284f611a35ce4a58bc9fb7609d9b74c18fe31cc95c8c3afe48f  /tmp/cts_x_stage/g2v10_train.py
14e466006b9a29c4d30181c424f5e6fb9872c781d03c3313c71fc433f7c2f3b3  /tmp/cts_x_stage/g2v9_gen.py
5172ad4a24e9e9f1de75c471ef0a35836f5296f60940c7c83b4245a5bf07dcae  /tmp/cts_x_stage/make_s0spec_x.py
0c026ea2d7e863d3035a7918cec23dd83206794db7ea2c092ed3aa97c62fad69  results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl
005e622d50ec5efdfa309a6dd8e4663880194f9c10e13c276573e636050aee8a  results_s7/g2v2/zstar_zsrcinj_train1000.pt
d43613d6b640b965c9c8a3d6ad422d7a8492fe0b0c2eed9a379064495c200433  results_s7/g2v2/zstar_zsrcinj_dev600.pt
340ef5c2bfb718fc371c932746476555d8d5a2bcd54c7413acc40e35840e45a7  results_s7/zsrc_train1000_P4.pt
0da3748af2a3776c7fa91da50268a7829233939f56218d73f4b4511a6a8e5cff  results_s7/zsrc_heldout600.pt
[TR-X7E47] rc=0 20:38:36 [Step 1499] loss 0.0450 ans_ce 0.1553 (n 4) gates [-0.002, -0.014, -0.047, 0.034] (1255s)
[TR-X7A47] rc=0 20:38:45 [Step 1499] loss 0.0702 ans_ce 0.2215 (n 4) gates [-0.001, -0.016, 0.056, 0.033] (1254s)
[TR-X7A49] rc=0 20:38:50 [Step 1499] loss 0.0751 ans_ce 0.3943 (n 4) gates [0.002, -0.015, 0.045, 0.04] (1259s)
[TR-X7A50] rc=0 03:10:17 [Step 1499] loss 0.0852 ans_ce 0.2443 (n 4) gates [-0.002, 0.016, -0.043, 0.04] (1252s)
[TR-X7E50] rc=0 03:10:20 [Step 1499] loss 0.0939 ans_ce 0.4704 (n 4) gates [-0.001, 0.013, -0.041, -0.034] (1259s)
[TR-X7E49] rc=0 03:10:32 [Step 1499] loss 0.0567 ans_ce 0.2156 (n 4) gates [-0.001, -0.013, 0.041, 0.035] (1270s)
[GRADER-VERIFIED] scripts/s5_eval_grader_v2.py SHA: 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
[S0] C_fmt: n 600 strict 77 lenient 94 boxed 213
[S0] X7A47: n 600 strict 67 lenient 82 boxed 309
[S0] shuf_X7A47: n 600 strict 73 lenient 92 boxed 312
[S0] X7E47: n 600 strict 76 lenient 87 boxed 267
[S0] shuf_X7E47: n 600 strict 78 lenient 97 boxed 273
[S0] X7A49: n 600 strict 68 lenient 84 boxed 269
[S0] shuf_X7A49: n 600 strict 78 lenient 87 boxed 301
[S0] X7E49: n 600 strict 58 lenient 71 boxed 256
[S0] shuf_X7E49: n 600 strict 62 lenient 84 boxed 251
[S0] X7A50: n 600 strict 60 lenient 71 boxed 310
[S0] shuf_X7A50: n 600 strict 72 lenient 82 boxed 302
[S0] X7E50: n 600 strict 79 lenient 98 boxed 305
[S0] shuf_X7E50: n 600 strict 80 lenient 96 boxed 318
[S0-DONE] /tmp/cts_x7/x7_items.json
```

2. **로컬 독립 재채점(`codex_grade_local.py`) 표준 출력 원문**:
```text
[LOCAL-SHIM] windows: math-verify per-call multiprocessing timeout replaced by no-op (equivalence logic unchanged)
[GRADER-VERIFIED] scripts/s5_eval_grader_v2.py SHA: 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
[S0] C_fmt: n 600 strict 77 lenient 93 boxed 213
[S0] X7A47: n 600 strict 67 lenient 82 boxed 309
[S0] shuf_X7A47: n 600 strict 73 lenient 91 boxed 312
[S0] X7E47: n 600 strict 76 lenient 87 boxed 267
[S0] shuf_X7E47: n 600 strict 78 lenient 97 boxed 273
[S0] X7A49: n 600 strict 68 lenient 84 boxed 269
[S0] shuf_X7A49: n 600 strict 78 lenient 87 boxed 301
[S0] X7E49: n 600 strict 58 lenient 71 boxed 256
[S0] shuf_X7E49: n 600 strict 62 lenient 83 boxed 251
[S0] X7A50: n 600 strict 60 lenient 70 boxed 310
[S0] shuf_X7A50: n 600 strict 72 lenient 82 boxed 302
[S0] X7E50: n 600 strict 79 lenient 97 boxed 305
[S0] shuf_X7E50: n 600 strict 80 lenient 95 boxed 318
[S0-DONE] handoff/codex/OUTBOX/20261024-1901-C006/x7_items_local.json
```

3. **서버 채점표 기준 독립 재실행(`x7_analyze_server_utf8.txt`) vs `cc_x7_analysis.txt` 원문 (100% 일치)**:
```text
== 0) per condition (n=600): boxed / lenient / strict / strict|boxed
  C_fmt       boxed  35.5%  lenient 15.67%  strict 12.83%  strict|boxed  36.2% (nb 213)
  X7A47       boxed  51.5%  lenient 13.67%  strict 11.17%  strict|boxed  21.7% (nb 309)
  X7E47       boxed  44.5%  lenient 14.50%  strict 12.67%  strict|boxed  28.5% (nb 267)
  X7A49       boxed  44.8%  lenient 14.00%  strict 11.33%  strict|boxed  25.3% (nb 269)
  X7E49       boxed  42.7%  lenient 11.83%  strict  9.67%  strict|boxed  22.7% (nb 256)
  X7A50       boxed  51.7%  lenient 11.83%  strict 10.00%  strict|boxed  19.4% (nb 310)
  X7E50       boxed  50.8%  lenient 16.33%  strict 13.17%  strict|boxed  25.9% (nb 305)
  shuf_X7A47  boxed  52.0%  lenient 15.33%  strict 12.17%  strict|boxed  23.4% (nb 312)
  shuf_X7E47  boxed  45.5%  lenient 16.17%  strict 13.00%  strict|boxed  28.6% (nb 273)
  shuf_X7A49  boxed  50.2%  lenient 14.50%  strict 13.00%  strict|boxed  25.9% (nb 301)
  shuf_X7E49  boxed  41.8%  lenient 14.00%  strict 10.33%  strict|boxed  24.7% (nb 251)
  shuf_X7A50  boxed  50.3%  lenient 13.67%  strict 12.00%  strict|boxed  23.8% (nb 302)
  shuf_X7E50  boxed  53.0%  lenient 16.00%  strict 13.33%  strict|boxed  25.2% (nb 318)
== Stage AE (1): A - E (rule: same sign all seeds AND pooled CI lower > 0) 
  lenient (primary)    seed-mean -1.06%p  95% CI [-2.72, +0.56]  per-seed [-0.83, 2.17, -4.5]
  strict (secondary)   seed-mean -1.00%p  95% CI [-2.61, +0.61]  per-seed [-1.5, 1.67, -3.17]
== Stage AE (2): A matched - shuffled (rule: CI lower > 0) 
  lenient (primary)    seed-mean -1.33%p  95% CI [-3.06, +0.33]  per-seed [-1.67, -0.5, -1.83]
  strict (secondary)   seed-mean -1.56%p  95% CI [-3.33, +0.17]  per-seed [-1.0, -1.67, -2.0]
== Stage AE (ref): E matched - shuffled 
  lenient (primary)    seed-mean -1.17%p  95% CI [-2.89, +0.56]  per-seed [-1.67, -2.17, 0.33]
  strict (secondary)   seed-mean -0.39%p  95% CI [-2.00, +1.22]  per-seed [-0.33, -0.67, -0.17]
== Stage AE (ref): A - C_fmt 
  lenient (primary)    seed-mean -2.50%p  95% CI [-4.94, -0.22]  per-seed [-2.0, -1.67, -3.83]
  strict (secondary)   seed-mean -2.00%p  95% CI [-4.44, +0.39]  per-seed [-1.67, -1.5, -2.83]
== Stage AE (ref): E - C_fmt 
  lenient (primary)    seed-mean -1.44%p  95% CI [-3.94, +0.94]  per-seed [-1.17, -3.83, 0.67]
  strict (secondary)   seed-mean -1.00%p  95% CI [-3.44, +1.39]  per-seed [-0.17, -3.17, 0.33]
== Stage AE (ref): A shuffled - C_fmt (harm) 
  lenient (primary)    seed-mean -1.17%p  95% CI [-3.72, +1.28]  per-seed [-0.33, -1.17, -2.0]
  strict (secondary)   seed-mean -0.44%p  95% CI [-3.00, +2.06]  per-seed [-0.67, 0.17, -0.83]
== Stage AE (ref): E shuffled - C_fmt (harm) 
  lenient (primary)    seed-mean -0.28%p  95% CI [-2.83, +2.11]  per-seed [0.5, -1.67, 0.33]
  strict (secondary)   seed-mean -0.61%p  95% CI [-3.00, +1.67]  per-seed [0.17, -2.5, 0.5]
== Stage AE (1) both-boxed subset: A - E (subset n=71)
  lenient (primary)    seed-mean -7.04%p  95% CI [-13.62, -0.47]  per-seed [-14.08, 7.04, -14.08]
  strict (secondary)   seed-mean -7.04%p  95% CI [-13.62, -0.47]  per-seed [-14.08, 7.04, -14.08]
```

---

## 6. 산출물 SHA256

`OUTBOX/20261024-1901-C006/` 내 생성 파일의 SHA256은 `OUTPUT_MANIFEST.json`에 함께 기록되며 주요 산출물 해시는 아래와 같다.

- `x7_items_local.json`: `OUTPUT_MANIFEST.json` 참조
- `c006_analysis.json`: `OUTPUT_MANIFEST.json` 참조
- `x7_analyze_server_utf8.txt`: `45ee13b4994fa70c019ca7e9b781f9bdc9aefe559abcc42c7144ae4fe09d5aae`
- `x7_analyze_local_utf8.txt`: `OUTPUT_MANIFEST.json` 참조
- `analyze_c006.py`: `OUTPUT_MANIFEST.json` 참조

---

## 7. 의견(판정 아님)

1. **"학습 답 CE는 `A`·`E`가 오라클보다 낮은데 개발에서는 반대"라는 현상의 telemetry 기반 구조 분석**:
   - **학습 후반(마지막 200걸음, `steps 1300..1499`)과 학습 초반(첫 50걸음, `steps 0..49`)의 역전 현상**:
     - 마지막 200걸음 평균 `final_ans_ce_last200`는 `X7A`(`0.270`, `0.269`, `0.273`) < `X7E`(`0.292`, `0.303`, `0.316`) < `X5/X6-R` 오라클(`0.409`, `0.422`, `0.496`) 순으로 `A`·`E`가 오라클보다 낮다.
     - 그러나 본 감사에서 계산한 **초기 50걸음(`steps 0..49`) 평균 `ans_ce`**(표 2.4 및 C005 표 2.4)를 비교하면 정반대 순서가 나타난다: 오라클(`2.943`, `2.899`) < `X7E`(`3.957`, `3.938`, `3.879`) < `X7A`(`4.356`, `4.364`, `4.306`).
   - **기전 해석**:
     - 학습 초반(0~49걸음)에는 992개 문항을 아직 한 번도 다 보지 않은 상태(`50 × 4 = 200` 샘플 방문)이므로 문항 암기가 불가능하다. 이때 `z_oracle`은 정답 해설의 은닉 상태에서 직접 추출되어 문항 간 공통으로 읽을 수 있는 답 단서를 실제로 담고 있기 때문에 초반 `ans_ce`가 `2.9`까지 빠르게 떨어진다. 반면 `z_src`(`X7E`)와 `z*'`(`X7A`)는 초반에 답 정보를 거의 주지 못해 `3.9~4.3`에 머문다.
     - 반면 학습 후반(1,300~1,499걸음, 문항당 평균 ~6회 반복 방문)에 이르면 21M 파라미터 교차주의 어댑터가 992개 학습 문항의 잠재 벡터를 **문항 식별 키(item-ID hash key)**로 삼아 해당 문항의 정답 토큰을 외우는(lookup memorization) 구간에 진입한다. 이때 `z*'`(수축 사상 DEQ 고정점)와 `z_src`(P4 투영)는 정규화·수축을 거쳐 노름과 분포가 매끄럽게 정돈된 저분산 벡터이므로, 해설 길이나 토큰 위치에 따라 변동이 큰 원시 은닉 상태(`z_oracle`)보다 992개 학습 샘플의 키-값 암기 테이블을 적합하기가 오히려 더 쉽다.
     - 그러나 미개봉 개발 600문항에서는 학습 때 외운 992개 문항의 키-값 매핑이 전혀 작동하지 않으며, 잠재 자체에 정답 정보가 들어 있지 않으므로 `X7A`(`matched − shuffled` lenient `-1.33%p`)와 `X7E`(`-1.17%p`) 모두 개발 세트에서 양(+)의 정보 이득을 내지 못한다.
2. **형식 효과(`boxed` 상승)와 정답 정밀도(`strict | boxed`) 하락**:
   - `X7A`(`44.8%~51.7%`)와 `X7E`(`42.7%~50.8%`)는 `C_fmt`(`35.5%`) 대비 `\boxed{}` 출력 비율을 `+7~+16%p` 높이지만, 이는 `shuffled` 잠재를 넣었을 때도 동일하게 나타나는(`41.8%~53.0%`) 문항 비특이적 형식 효과이다. 오히려 `strict | boxed`(박스를 친 문항 중 정답 비율)는 `C_fmt` `36.15%`에서 `X7A` `19.35%~25.28%`, `X7E` `22.66%~28.46%`로 낮아져, 내용 정보 없이 형식만 강제할 때 오답 박스 생성이 늘어남을 보여준다.

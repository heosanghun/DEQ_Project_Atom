# REPORT-C009 — X8-A(DEQ + 브리지 공동 학습, K_unroll=2) 독립 재계산·재채점 및 산출물 감사

- 작성 시각: `2026-10-24 18:54 KST`
- 대상 과제: `handoff/codex/INBOX/TASK-C009.md`
- 입력 패키지: `handoff/codex/INBOX/20261023-1950-x8a-results/` 및 대조군 `handoff/codex/INBOX/20261021-0945-x7-results/` (`X7E47`, `X7E49`, `X7E50`, `shuf_X7E*`)
- 고정 채점기: `scripts/s5_eval_grader_v2.py` / `handoff/codex/scripts/s5_eval_grader_v2.py`
  - 검증된 SHA256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`
- 로컬 재채점 래퍼: `handoff/codex/scripts/codex_grade_local.py` (rev3, SHA256 `bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c`)
- 고정 개발 문항 순서: `handoff/codex/data/p6_item_order.json` (600문항, SHA256 `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c`)

---

## 0. 한 줄 요약

X8-A 3시드(`2047`, `2049`, `2050`) 및 대조군(`X7-E`, `C_fmt`)의 개발 600문항 지표를 문항 단위 cluster bootstrap(10,000회, `seed=2026`)으로 독립 재계산한 결과 `cc_x8a_analysis.txt`의 모든 수치와 `0.00 %p` 차이로 일치했으며(`X8-A − E` lenient 합산 `+0.61 %p`, 95% CI `[-1.11, +2.33]`, 양(+) 시드 `2/3`; `X8-A matched − shuffled` lenient 합산 `+1.28 %p`, 95% CI `[-0.44, +3.00]`, 양(+) 시드 `2/3`), 로컬 rev3 재채점(`codex_grade_local.py`)은 전체 `12,600`개 플래그 중 `12,596`개(`99.968%`)가 서버 채점표(`x8a_items.json`)와 일치했다(`strict`·`boxed` 불일치 0건, `lenient` 불일치 4건).

---

## 1. 수행한 것

1. **고정 채점기·데이터 경계·입력 패키지 SHA256 전수 대조**:
   - `scripts/s5_eval_grader_v2.py` 및 `handoff/codex/scripts/s5_eval_grader_v2.py`의 SHA256이 `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`임을 확인.
   - `handoff/codex/data/p6_item_order.json`(SHA256 `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c`, 600문항)과 `configs/s7_train_993.json`(SHA256 `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c`, 993문항)의 교집합이 `0`임을 확인.
   - `INBOX/20261023-1950-x8a-results/SHA256SUMS_full.txt`에 기재된 24개 파일 전체의 SHA256 해시를 독립 계산하여 `24/24` 전수 일치를 확인.
   - 실행 스크립트 `handoff/codex/scripts/g2v11_x8a_train.py`(`208eb131f97988c82746b7d480ea1b214fb242f895690fdde4c5a1bd8333c568`), `g2v11_x8a_latents.py`(`9a82c78179c686eaae2b739c812c36f2d146b94fdd24334edceb544028166b0c`), `cc_x8a.sh`(`4a5e457850891107359e55ff375168be9f7c22886068ad0bb2d90aac64e7adca`)가 `cts_x8a_master.log` 상단 기록 해시와 일치함을 확인.
2. **로컬 독립 재채점 실행 (`codex_grade_local.py` rev3)**:
   - `handoff/codex/INBOX/20261023-1950-x8a-results/s0spec_local.json`의 7개 조건(`C_fmt`, `X8A47`, `shuf_X8A47`, `X8A49`, `shuf_X8A49`, `X8A50`, `shuf_X8A50`) 각 600문항을 로컬에서 독립 채점하여 `x8a_items_local.json` 생성.
   - 서버 채점 결과(`x8a_items.json`)와 `7 × 600 × 3 = 12,600`개 플래그를 1:1 대조하고 불일치 문항 4건의 원문·추출 답·정답을 확인.
3. **CTO 기준 및 보조 지표 독립 재계산 (bootstrap `B=10,000`, `seed=2026`)**:
   - 서버 채점표(`x8a_items.json` + `x7_items.json`)와 로컬 재채점표(`x8a_items_local.json` + `x7_items.json`) 양쪽에서 조건별 정답률(`lenient`, `strict`, `boxed`), 생성 토큰 길이(평균·P50·P95), EOS 종료율(`stop_id in {1, 106}`), 그리고 3시드 합산 및 시드별 bootstrap 평균·95% CI·부호 일치를 독립 계산.
4. **학습 manifest 3종·개발 잠재 잔차(`latdev_*.pt.res.json`) 3종·로그 감사**:
   - `manifest_X8A{47,49,50}.json`의 하이퍼파라미터, 표적 분류 계수(`target_counts`), 1,500걸음 telemetry 배열로부터 마지막 200걸음 평균(`final_loss_last200`, `final_ans_ce_last200`, `final_lfp_last200`, `final_res_last200`)을 재계산하여 대조.
   - `latdev_X8A{47,49,50}.pt.res.json`의 600문항별 2걸음 잔차(`[res_step0, res_step1]`)로부터 `res_step0_mean`, `res_last_mean`, `res_last_max` 및 `res_step1 < res_step0` 성립 문항 수를 독립 집계.
   - `RESULT-X8A-readable-deq-20261023.md` 본문 표와 원시 로그/manifest 사이의 표기 불일치를 검출.

---

## 2. 결과 표(숫자만)

### 2.1 원본 로그 및 독립 재계산 원문 행 대조

- **서버 채점 로그 원문 (`handoff/codex/INBOX/20261023-1950-x8a-results/grade.log`)**:
```text
[GRADER-VERIFIED] scripts/s5_eval_grader_v2.py SHA: 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
[S0] C_fmt: n 600 strict 77 lenient 94 boxed 213 len 425.2 eos 39.2%
[S0] X8A47: n 600 strict 67 lenient 89 boxed 301 len 389.9 eos 54.7%
[S0] shuf_X8A47: n 600 strict 55 lenient 68 boxed 308 len 386.8 eos 55.7%
[S0] X8A49: n 600 strict 83 lenient 95 boxed 306 len 391.6 eos 55.3%
[S0] shuf_X8A49: n 600 strict 82 lenient 91 boxed 351 len 378.7 eos 60.8%
[S0] X8A50: n 600 strict 70 lenient 83 boxed 314 len 387.2 eos 56.0%
[S0] shuf_X8A50: n 600 strict 71 lenient 85 boxed 290 len 401.3 eos 50.3%
[S0-DONE] /tmp/cts_x8a/x8a_items.json
```

- **로컬 독립 재채점 실행 출력 원문 (`codex_grade_local.py` rev3)**:
```text
[LOCAL-SHIM] windows: math-verify per-call multiprocessing timeout replaced by no-op (equivalence logic unchanged)
[GRADER-VERIFIED] scripts/s5_eval_grader_v2.py SHA: 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
[S0] C_fmt: n 600 strict 77 lenient 93 boxed 213
[S0] X8A47: n 600 strict 67 lenient 88 boxed 301
[S0] shuf_X8A47: n 600 strict 55 lenient 67 boxed 308
[S0] X8A49: n 600 strict 83 lenient 94 boxed 306
[S0] shuf_X8A49: n 600 strict 82 lenient 91 boxed 351
[S0] X8A50: n 600 strict 70 lenient 83 boxed 314
[S0] shuf_X8A50: n 600 strict 71 lenient 85 boxed 290
[S0-DONE] handoff/codex/OUTBOX/20261024-1833-C009/x8a_items_local.json
```

- **서버 마스터 로그 학습·잠재 생성 요약 원문 (`cts_x8a_master.log` lines 17–28)**:
```text
[TR-X8A47] rc=0 13:20:32 [Step 1499] loss 0.0652 ans_ce 0.2333 l_fp 0.00125 res 0.0379 gates [0.002, -0.016, -0.043, -0.037] (1860s)
[TR-X8A50] rc=0 13:20:35 [Step 1499] loss 0.0731 ans_ce 0.2027 l_fp 0.00131 res 0.0386 gates [-0.002, -0.014, -0.055, 0.031] (1863s)
[TR-X8A49] rc=0 13:20:58 [Step 1499] loss 0.0574 ans_ce 0.2198 l_fp 0.00106 res 0.0349 gates [-0.002, -0.016, 0.043, 0.035] (1886s)
[LAT-X8A47] rc=0 13:22:36 [SAVE] items 600 missing 0 -> /tmp/cts_x8a/latdev_X8A47.pt; res_last mean 0.03548
[LAT-X8A50] rc=0 13:22:38 [SAVE] items 600 missing 0 -> /tmp/cts_x8a/latdev_X8A50.pt; res_last mean 0.03569
[LAT-X8A49] rc=0 13:23:02 [SAVE] items 600 missing 0 -> /tmp/cts_x8a/latdev_X8A49.pt; res_last mean 0.03232
```

- **독립 재계산 스크립트 출력 원문 (`x8a_analyze_server_utf8.txt` vs `cc_x8a_analysis.txt` 핵심 행)**:
```text
== CTO (1): X8-A - E (rule: 3 seeds same sign AND pooled lenient CI lower > 0)
  lenient (primary) mean +0.61%p  95% CI [-1.11, +2.33]  per-seed [0.33, 4.0, -2.5] (+seeds 2/3)
  boxed (aux)       mean +5.17%p  95% CI [+2.44, +7.83]  per-seed [5.67, 8.33, 1.5] (+seeds 3/3)
  strict (aux)      mean +0.39%p  95% CI [-1.28, +2.06]  per-seed [-1.5, 4.17, -1.5] (+seeds 1/3)
== CTO (2): X8-A matched - shuffled (rule: lenient CI lower > 0)
  lenient (primary) mean +1.28%p  95% CI [-0.44, +3.00]  per-seed [3.5, 0.67, -0.33] (+seeds 2/3)
  boxed (aux)       mean -1.56%p  95% CI [-4.28, +1.17]  per-seed [-1.17, -7.5, 4.0] (+seeds 1/3)
  strict (aux)      mean +0.67%p  95% CI [-0.94, +2.33]  per-seed [2.0, 0.17, -0.17] (+seeds 2/3)
```

### 2.2 조건별 성적·길이·EOS 요약 표 (`n = 600`)

| 조건 | 서버 `boxed` (개 / %) | 서버 `lenient` (개 / %) | 서버 `strict` (개 / %) | 로컬 `lenient` (개 / %) | 로컬 `strict` (개 / %) | 평균 토큰 길이 | 길이 P50 / P95 | EOS 종료 (개 / %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C_fmt` | 213 / 35.50% | 94 / 15.67% | 77 / 12.83% | 93 / 15.50% | 77 / 12.83% | 425.21 | 512.0 / 512 | 235 / 39.17% |
| `X8A47` | 301 / 50.17% | 89 / 14.83% | 67 / 11.17% | 88 / 14.67% | 67 / 11.17% | 389.91 | 456.0 / 512 | 328 / 54.67% |
| `shuf_X8A47` | 308 / 51.33% | 68 / 11.33% | 55 / 9.17% | 67 / 11.17% | 55 / 9.17% | 386.76 | 439.5 / 512 | 334 / 55.67% |
| `X8A49` | 306 / 51.00% | 95 / 15.83% | 83 / 13.83% | 94 / 15.67% | 83 / 13.83% | 391.59 | 452.5 / 512 | 332 / 55.33% |
| `shuf_X8A49` | 351 / 58.50% | 91 / 15.17% | 82 / 13.67% | 91 / 15.17% | 82 / 13.67% | 378.70 | 420.5 / 512 | 365 / 60.83% |
| `X8A50` | 314 / 52.33% | 83 / 13.83% | 70 / 11.67% | 83 / 13.83% | 70 / 11.67% | 387.23 | 430.0 / 512 | 336 / 56.00% |
| `shuf_X8A50` | 290 / 48.33% | 85 / 14.17% | 71 / 11.83% | 85 / 14.17% | 71 / 11.83% | 401.34 | 509.0 / 512 | 302 / 50.33% |
| `X7E47` (대조) | 267 / 44.50% | 87 / 14.50% | 76 / 12.67% | 87 / 14.50% | 76 / 12.67% | 398.85 | 505.5 / 512 | 304 / 50.67% |
| `shuf_X7E47` | 273 / 45.50% | 97 / 16.17% | 78 / 13.00% | 97 / 16.17% | 78 / 13.00% | 387.07 | 452.0 / 512 | 322 / 53.67% |
| `X7E49` (대조) | 256 / 42.67% | 71 / 11.83% | 58 / 9.67% | 71 / 11.83% | 58 / 9.67% | 409.18 | 512.0 / 512 | 280 / 46.67% |
| `shuf_X7E49` | 251 / 41.83% | 84 / 14.00% | 62 / 10.33% | 84 / 14.00% | 62 / 10.33% | 407.05 | 512.0 / 512 | 286 / 47.67% |
| `X7E50` (대조) | 305 / 50.83% | 98 / 16.33% | 79 / 13.17% | 98 / 16.33% | 79 / 13.17% | 398.39 | 482.0 / 512 | 315 / 52.50% |
| `shuf_X7E50` | 318 / 53.00% | 96 / 16.00% | 80 / 13.33% | 96 / 16.00% | 80 / 13.33% | 395.16 | 460.0 / 512 | 326 / 54.33% |

### 2.3 Bootstrap 비교 검증 표 (`B = 10,000`, `seed = 2026`, 단위 `%p`)

| 비교 항목 | 지표 | CC 보고 평균 (`cc_x8a_analysis.txt`) | CC 보고 95% CI | 독립 재계산 평균 (서버 채점표) | 독립 재계산 95% CI (서버 채점표) | 시드별 차이 `[2047, 2049, 2050]` | 양(+) 시드 수 | 로컬 재채점 평균 [95% CI] |
|---|---|---:|---:|---:|---:|---|---:|---:|
| **CTO (1) `X8-A − E`** (3시드 합산) | `lenient` | `+0.61` | `[-1.11, +2.33]` | `+0.61` | `[-1.11, +2.33]` | `[+0.33, +4.00, -2.50]` | `2/3` | `+0.50 [-1.22, +2.22]` |
| **CTO (1) `X8-A − E`** (3시드 합산) | `boxed` | `+5.17` | `[+2.44, +7.83]` | `+5.17` | `[+2.44, +7.83]` | `[+5.67, +8.33, +1.50]` | `3/3` | `+5.17 [+2.44, +7.83]` |
| **CTO (1) `X8-A − E`** (3시드 합산) | `strict` | `+0.39` | `[-1.28, +2.06]` | `+0.39` | `[-1.28, +2.06]` | `[-1.50, +4.17, -1.50]` | `1/3` | `+0.39 [-1.28, +2.06]` |
| **CTO (2) `X8-A matched − shuf`** (3시드 합산) | `lenient` | `+1.28` | `[-0.44, +3.00]` | `+1.28` | `[-0.44, +3.00]` | `[+3.50, +0.67, -0.33]` | `2/3` | `+1.22 [-0.50, +2.94]` |
| **CTO (2) `X8-A matched − shuf`** (3시드 합산) | `boxed` | `-1.56` | `[-4.28, +1.17]` | `-1.56` | `[-4.28, +1.17]` | `[-1.17, -7.50, +4.00]` | `1/3` | `-1.56 [-4.28, +1.17]` |
| **CTO (2) `X8-A matched − shuf`** (3시드 합산) | `strict` | `+0.67` | `[-0.94, +2.33]` | `+0.67` | `[-0.94, +2.33]` | `[+2.00, +0.17, -0.17]` | `2/3` | `+0.67 [-0.94, +2.33]` |
| **보조 `X8-A matched − C_fmt`** (3시드 합산) | `lenient` | `-0.83` | `[-3.28, +1.56]` | `-0.83` | `[-3.28, +1.56]` | `[-0.83, +0.17, -1.83]` | `1/3` | `-0.78 [-3.22, +1.61]` |
| **보조 `X8-A matched − C_fmt`** (3시드 합산) | `boxed` | `+15.67` | `[+11.78, +19.50]` | `+15.67` | `[+11.78, +19.50]` | `[+14.67, +15.50, +16.83]` | `3/3` | `+15.67 [+11.78, +19.50]` |
| **보조 `X8-A matched − C_fmt`** (3시드 합산) | `strict` | `-0.61` | `[-2.94, +1.61]` | `-0.61` | `[-2.94, +1.61]` | `[-1.67, +1.00, -1.17]` | `1/3` | `-0.61 [-2.94, +1.61]` |
| **보조 `X8-A shuffled − C_fmt`** (3시드 합산) | `lenient` | `-2.11` | `[-4.61, +0.33]` | `-2.11` | `[-4.61, +0.33]` | `[-4.33, -0.50, -1.50]` | `0/3` | `-2.00 [-4.50, +0.44]` |
| **보조 `X8-A shuffled − C_fmt`** (3시드 합산) | `boxed` | `+17.22` | `[+13.28, +21.06]` | `+17.22` | `[+13.28, +21.06]` | `[+15.83, +23.00, +12.83]` | `3/3` | `+17.22 [+13.28, +21.06]` |
| **보조 `X8-A shuffled − C_fmt`** (3시드 합산) | `strict` | `-1.28` | `[-3.83, +1.11]` | `-1.28` | `[-3.83, +1.11]` | `[-3.67, +0.83, -1.00]` | `1/3` | `-1.28 [-3.83, +1.11]` |
| **참고 `E matched − C_fmt`** (3시드 합산) | `lenient` | `-1.44` | `[-3.94, +0.94]` | `-1.44` | `[-3.94, +0.94]` | `[-1.17, -3.83, +0.67]` | `1/3` | `-1.28 [-3.78, +1.11]` |
| **참고 `E matched − C_fmt`** (3시드 합산) | `strict` | `-1.00` | `[-3.44, +1.39]` | `-1.00` | `[-3.44, +1.39]` | `[-0.17, -3.17, +0.33]` | `1/3` | `-1.00 [-3.44, +1.39]` |
| **참고 `both-boxed` 시드 2047 (`n=179`) `X8A47 − X7E47`** | `lenient` / `strict` | `-3.35` | `[-10.61, +3.91]` | `-3.35` | `[-10.61, +3.91]` | `[-3.35]` | `0/1` | `-3.35 [-10.61, +3.91]` |
| **참고 `both-boxed` 시드 2049 (`n=183`) `X8A49 − X7E49`** | `lenient` / `strict` | `+1.64` | `[-4.92, +8.74]` | `+1.64` | `[-4.92, +8.74]` | `[+1.64]` | `1/1` | `+1.64 [-4.92, +8.74]` |
| **참고 `both-boxed` 시드 2050 (`n=207`) `X8A50 − X7E50`** | `lenient` / `strict` | `-5.31` | `[-11.11, +0.48]` | `-5.31` | `[-11.11, +0.48]` | `[-5.31]` | `0/1` | `-5.31 [-11.11, +0.48]` |

### 2.4 시드별 단일-시드 Bootstrap 95% CI 상세 표 (`n = 600`, 서버 채점 기준)

| 시드 | 비교 | `lenient` 평균 `%p` [95% CI] | `boxed` 평균 `%p` [95% CI] | `strict` 평균 `%p` [95% CI] |
|---|---|---:|---:|---:|
| `2047` | `X8A47 − X7E47` | `+0.33 [-2.67, +3.17]` | `+5.67 [+1.00, +10.33]` | `-1.50 [-4.33, +1.33]` |
| `2049` | `X8A49 − X7E49` | `+4.00 [+0.83, +7.17]` | `+8.33 [+3.83, +12.83]` | `+4.17 [+1.33, +7.00]` |
| `2050` | `X8A50 − X7E50` | `-2.50 [-5.67, +0.50]` | `+1.50 [-3.17, +6.17]` | `-1.50 [-4.33, +1.33]` |
| `2047` | `X8A47 − shuf_X8A47` | `+3.50 [+0.67, +6.33]` | `-1.17 [-6.00, +3.67]` | `+2.00 [-0.67, +4.67]` |
| `2049` | `X8A49 − shuf_X8A49` | `+0.67 [-2.33, +3.67]` | `-7.50 [-12.33, -2.67]` | `+0.17 [-2.67, +3.17]` |
| `2050` | `X8A50 − shuf_X8A50` | `-0.33 [-3.33, +2.67]` | `+4.00 [-0.50, +8.67]` | `-0.17 [-3.17, +2.67]` |

### 2.5 학습 Manifest 및 개발 600문항 잠재 잔차(`latdev_*.pt.res.json`) 감사 표

| 항목 | `X8A47` (`seed=2047`) | `X8A49` (`seed=2049`) | `X8A50` (`seed=2050`) | 독립 재계산 일치 여부 |
|---|---:|---:|---:|---|
| `steps` / `n_updates` / `n_sample_visits` / `n_samples_skipped` | `1500 / 1500 / 6000 / 0` | `1500 / 1500 / 6000 / 0` | `1500 / 1500 / 6000 / 0` | 일치 |
| `n_items_trained` / `n_unique_ids_contributing_grad` | `992 / 992` | `992 / 992` | `992 / 992` | 일치 |
| `target_counts` (`replaced / appended / skip_no_gold / budget_truncated / finished_own`) | `341 / 651 / 1 / 575 / 437` | `341 / 651 / 1 / 575 / 437` | `341 / 651 / 1 / 575 / 437` | 일치 |
| `deq_trainable_weights` / `bridge_params` | `3,980,801 / 20,971,524` | `3,980,801 / 20,971,524` | `3,980,801 / 20,971,524` | 일치 |
| `k_unroll` / `alpha` / `g0` / `lambda_fp` / `lr` / `lr_deq` | `2 / 0.25 / 0.25 / 0.1 / 3e-4 / 3e-5` | `2 / 0.25 / 0.25 / 0.1 / 3e-4 / 3e-5` | `2 / 0.25 / 0.25 / 0.1 / 3e-4 / 3e-5` | 일치 |
| `step 0` (`loss / ans_ce / l_fp / res_mean`) | `0.852929 / 6.933695 / 1.70e-7 / 0.000439` | `0.852929 / 6.933695 / 1.70e-7 / 0.000439` | `0.852929 / 6.933695 / 1.70e-7 / 0.000439` | 3시드 완전 동일 |
| `step 1499` (`loss / ans_ce / l_fp / res_mean`) | `0.065248 / 0.233292 / 0.001253 / 0.037943` | `0.057374 / 0.219767 / 0.001063 / 0.034897` | `0.073131 / 0.202698 / 0.001306 / 0.038602` | 일치 |
| `final_loss_last200` (manifest 기록 vs 재계산) | `0.05834734` vs `0.05834734` | `0.06165290` vs `0.06165290` | `0.06009912` vs `0.06009912` | 일치 (`Δ = 0`) |
| `final_ans_ce_last200` (manifest 기록 vs 재계산) | `0.29189711` vs `0.29189711` | `0.31252046` vs `0.31252046` | `0.31163235` vs `0.31163235` | 일치 (`Δ = 0`) |
| `final_lfp_last200` (manifest 기록 vs 재계산) | `0.00155295` vs `0.00155295` | `0.00126697` vs `0.00126697` | `0.00129961` vs `0.00129961` | 일치 (`Δ = 0`) |
| `final_res_last200` (manifest 기록 vs 재계산) | `0.04205670` vs `0.04205670` | `0.03801978` vs `0.03801978` | `0.03847383` vs `0.03847383` | 일치 (`Δ = 0`) |
| 개발 600 잠재 `n_items` / `n_missing` | `600 / 0` | `600 / 0` | `600 / 0` | 일치 |
| 개발 600 잠재 `res_step0_mean` / `res_step0_max` (1번째 걸음) | `0.04073587 / 0.04672207` | `0.03702472 / 0.04074851` | `0.04072544 / 0.04442524` | 독립 계산 |
| 개발 600 잠재 `res_last_mean` / `res_last_max` (2번째 걸음) | `0.03548404 / 0.04040253` | `0.03231760 / 0.03613853` | `0.03569259 / 0.03946863` | 일치 (`Δ = 0`) |
| 개발 600 잠재 `res_step1 < res_step0` 성립 문항 수 | `600 / 600` | `600 / 600` | `600 / 600` | 전 문항 단조 감소 |

---

## 3. 불일치·결함 목록(파일, 줄, 재현 조건)

### 3.1 서버 채점(`x8a_items.json`) 대 로컬 재채점(`x8a_items_local.json`) 플래그 차이 (`4 / 12,600` 플래그)

- `strict`(`4,200`개) 및 `boxed`(`4,200`개) 플래그는 `100%`(`8,400/8,400`) 일치.
- `lenient`(`4,200`개)에서 아래 4건만 `server=1, local=0` 차이가 발생함(모두 정답이 퍼센트 기호 `\%`를 포함한 문항이며, 서버 Linux 환경과 로컬 Windows Python 3.13 환경 간 `math-verify` 버전의 퍼센트 기호 파싱 차이에서 기인):

| 조건(`label`) | `item_id` | 플래그 | 서버 값 | 로컬 값 | 정답(`gold`) | 추출된 예측(`extracted_pred`) |
|---|---:|---|---:|---:|---|---|
| `C_fmt` | `1503` | `lenient` | `1` | `0` | `64\%` | `$64\%$` |
| `X8A47` | `1503` | `lenient` | `1` | `0` | `64\%` | `price after the first discount is $P - 0` (비-boxed 절단문; 본문 내 `$0.64P$` 포함) |
| `shuf_X8A47` | `1503` | `lenient` | `1` | `0` | `64\%` | `$64\%$` |
| `X8A49` | `1464` | `lenient` | `1` | `0` | `40\%` | `$40\%$` |

### 3.2 결과 보고서(`RESULT-X8A-readable-deq-20261023.md`) 표기 불일치 2건

1. **파일**: `shared/reports/RESULT-X8A-readable-deq-20261023.md` (줄 7–11, `## 1. 학습·잠재 잔차 (G1 가드)` 표)
   - **현상**: 열 제목이 `학습 잔차(마지막 걸음)`으로 표기되어 있고 값으로 `0.042` (`X8A47`), `0.038` (`X8A49`), `0.039` (`X8A50`)가 적혀 있음.
   - **근거 수치**:
     - 실제 마지막 걸음(`Step 1499`)의 `res_mean`은 `0.03794` (`X8A47`), `0.03490` (`X8A49`), `0.03860` (`X8A50`)임(`train_X8A*.log` 줄 74 및 `manifest_X8A*.json` `telemetry[-1]`).
     - 반면 마지막 200걸음 평균(`final_res_last200`)은 `0.04206` (`X8A47`), `0.03802` (`X8A49`), `0.03847` (`X8A50`)임.
   - **원인**: `X8A47`(`0.042`)과 `X8A49`(`0.038`)는 `final_res_last200` 값을 적었고, `X8A50`(`0.039`)은 `Step 1499`의 `0.0386`(`0.039`)을 적어 한 열 안에서 기준(`마지막 200걸음 평균` vs `Step 1499 단일 걸음`)이 혼재되었거나 열 제목(`마지막 걸음` vs `마지막 200걸음`)이 부정확함.
2. **파일**: `shared/reports/RESULT-X8A-readable-deq-20261023.md` (줄 3) 및 `handoff/codex/scripts/g2v11_x8a_train.py` (줄 175)
   - **현상**: 보고서 줄 3에 `트레이너 g2v11_x8a_train.py 208eb131…(스모크판 5721ffc5…에서 detach 1줄 수정)`이라고 기록되어 있으나, 본 실행 로그 `train_X8A47.log`, `train_X8A49.log`, `train_X8A50.log` 줄 11–13 모두에 여전히 아래 `UserWarning`이 출력됨:
     ```text
     /tmp/cts_x_stage/g2v11_x8a_train.py:175: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
     Consider using tensor.detach() first.
       gates = [round(float(torch.tanh(ad.gate)), 3) for ad in br.adapters.values()]
     ```
   - **원인**: 스모크판(`5721ffc5…`)에서 본 실행판(`208eb131…`)으로 수정된 `detach`는 줄 51(`deq_unroll` 내 `res.append`)에만 적용되었고, 줄 175의 `torch.tanh(ad.gate)`에는 `.detach()`가 빠진 채 실행됨(수치·그래디언트 계산 자체에는 영향 없음).

---

## 4. 하지 못한 것과 이유

- 서버 내 원본 체크포인트(`bridge_X8A*.pt`, `deq_X8A*.pt`) 및 잠재 텐서(`latdev_X8A*.pt`) 바이너리 파일 자체의 직접 로드는 로컬 패키지에 포함되지 않아(로그 및 `.res.json`, `manifest_*.json`만 포함) 수행하지 못함(`cts_x8a_master.log`의 SHA256 기록과 대조로 갈음).

---

## 5. 재현 명령

```powershell
python -B handoff/codex/scripts/codex_grade_local.py handoff/codex/INBOX/20261023-1950-x8a-results/s0spec_local.json handoff/codex/OUTBOX/20261024-1833-C009/x8a_items_local.json
python -B handoff/codex/OUTBOX/20261024-1833-C009/analyze_c009_fast.py
```

---

## 6. 산출물 SHA256

- 입력 파일 24종의 검증 해시는 `INPUT_MANIFEST.json`에 수록.
- 본 OUTBOX 폴더(`handoff/codex/OUTBOX/20261024-1833-C009/`)의 산출물 해시는 `OUTPUT_MANIFEST.json`에 수록:
  - `x8a_items_local.json`: `make_manifests_c009.py` 실행 시 기록
  - `c009_analysis.json`: `make_manifests_c009.py` 실행 시 기록
  - `x8a_analyze_server_utf8.txt`: `make_manifests_c009.py` 실행 시 기록
  - `x8a_analyze_local_utf8.txt`: `make_manifests_c009.py` 실행 시 기록
  - `analyze_c009_fast.py`: `make_manifests_c009.py` 실행 시 기록

---

## 7. 의견(판정 아님)

1. **지표 재현성**: `cc_x8a_analysis.txt` 및 `RESULT-X8A-readable-deq-20261023.md`의 모든 정답률·bootstrap 평균·95% 신뢰구간·시드별 부호 집계가 독립 재계산에서 소수점 둘째 자리까지 오차 없이 일치한다. 로컬 rev3 재채점에서도 4개 플래그 차이(`lenient` 한정)만 존재하며, 모든 비교 구간의 95% CI가 0을 포함한다.
2. **잔차(G1 가드) 거동**: 학습 중 고정점 잔차 페널티(`lambda_fp = 0.1`) 하에서 개발 600문항 전체(`600/600`, 3시드 모두)에서 2번째 걸음 잔차(`res_last_mean` `0.0323~0.0357`)가 1번째 걸음 잔차(`res_step0_mean` `0.0370~0.0407`)보다 작아, 2걸음 전개 내에서 발산하지 않고 수축 방향을 유지했음이 수치로 확인된다.
3. **시드 간 비일관성**: `X8-A − E` lenient 차이는 시드 `2049`에서 `+4.00 %p`(`95% CI [+0.83, +7.17]`)인 반면 시드 `2050`에서 `-2.50 %p`(`[-5.67, +0.50]`)이며, `X8-A matched − shuffled` lenient 차이도 시드 `2047`에서만 `+3.50 %p`(`[+0.67, +6.33]`)이고 시드 `2049`(`+0.67 %p`)·`2050`(`-0.33 %p`)에서는 신호가 재현되지 않는다.

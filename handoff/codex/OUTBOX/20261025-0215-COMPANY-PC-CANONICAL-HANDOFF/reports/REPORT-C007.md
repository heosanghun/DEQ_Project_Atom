# REPORT-C007 — X8-0 (문제만 보는 잠재 대조, Problem-Only Latent Control) 독립 재계산·재채점 보고서

- 작성 시각: 2026-10-24 19:11 KST
- 대상 과제: `handoff/codex/INBOX/TASK-C007.md`
- 입력 패키지: `handoff/codex/INBOX/20261022-0015-x8_0-results/`
- 산출물 폴더: `handoff/codex/OUTBOX/20261024-1906-C007/`
- 사용 채점기: `scripts/s5_eval_grader_v2.py` 및 `handoff/codex/scripts/s5_eval_grader_v2.py` (SHA256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`)
- 사용 로컬 채점 래퍼: `handoff/codex/scripts/codex_grade_local.py` (rev3, SHA256: `bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c`)

---

## 0. 한 줄 요약

`INBOX/20261022-0015-x8_0-results/`의 18개 파일 SHA256 전수 일치, 서버 채점표(`x8_0_items.json`) 기준 CC의 X8-0 3시드 합산 및 조건별 수치 전 항목 `0.00%p` 오차 재현, 로컬 rev3 래퍼(`codex_grade_local.py`) 7개 조건 독립 재채점(`X80P*`/`shuf_X80P*` 신규 6개 조건 10,800개 플래그 불일치 `0`건, `C_fmt` 포함 12,600개 플래그 중 `id=1503` `1`건만 차이 — 주 지표 `matched − shuffled`는 서버와 로컬이 `0.00%p` 차이로 완전 일치) 및 manifest 3종·교차 실험 비교 분석을 완료하였다.

---

## 1. 수행한 것

1. **입력 무결성 및 데이터 경계 검증**:
   - `INBOX/20261022-0015-x8_0-results/SHA256SUMS_full.txt`에 기재된 18개 파일의 로컬 SHA256을 계산하여 `18 / 18` 전수 일치 확인.
   - `cts_x8_0_master.log`(1행 런처 자기 해시 `d71b45943ed17a49665a127c6d43e83c46dc303f0ba8497f01712350ad5f2346`, 2~7행 스크립트·타깃 해시, 13행·41~54행 산출물 해시)와 `SHA256SUMS_full.txt` 대조 일치 확인.
   - 채점기 `scripts/s5_eval_grader_v2.py` 및 `handoff/codex/scripts/s5_eval_grader_v2.py` SHA256(`90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`), 로컬 채점 래퍼 `handoff/codex/scripts/codex_grade_local.py` SHA256(`bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c`), 개발 문항 `handoff/codex/data/p6_item_order.json`(600문항, SHA256 `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c`), 학습 문항 `configs/s7_train_993.json`(993문항, SHA256 `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c`) 일치 및 학습–개발 교집합 `0`문항 확인.
2. **서버 채점표(`x8_0_items.json`) 기준 독립 재계산**:
   - 7개 조건(`C_fmt`, `X80P47`, `shuf_X80P47`, `X80P49`, `shuf_X80P49`, `X80P50`, `shuf_X80P50`, 각 `n=600`)의 `boxed`, `lenient`, `strict`, `strict|boxed`, 생성 길이(`mean`, `P50`, `P95`), `EOS` 종료율을 독립 재계산.
   - 사전등록 규칙(`docs/X8_READABLE_LATENT_DEQ_20261021.md` §2·§5, 문항 단위 cluster bootstrap `B=10,000`, `seed=2026`)에 따라 3시드 합산 `matched − shuffled`(주 지표), `matched − C_fmt`, `shuffled − C_fmt` 및 시드별 지표를 독립 재계산하고 `cc_x8_0_analysis.txt`와 대조.
3. **로컬 독립 재채점(`codex_grade_local.py`) 및 플래그 단위 대조**:
   - `INBOX/20261022-0015-x8_0-results/s0spec_local.json`의 7개 조건(4,200행)을 로컬 Python 3.13.3(`sympy==1.14.0`, `math-verify==0.9.0`)에서 재채점하여 `OUTBOX/20261024-1906-C007/x8_0_items_local.json` 생성.
   - 서버 `x8_0_items.json`과 로컬 `x8_0_items_local.json`의 `7 × 600 × 3 = 12,600`개 플래그를 전수 대조하고, `grade.log` 8행의 `Timeout during comparison` 발생 조건(`shuf_X80P50`)에서도 서버·로컬 플래그 불일치가 `0`건임을 확인.
4. **manifest 3종(`manifest_X80P{47,49,50}.json`) 및 `latents.log` 검증 + X5/X6·X7·X8-0·X8-A 통합 대조**:
   - `latents.log`의 잠재 추출 로그(`items 1593 (short<64: 1031) keys 3186`)와 manifest 3종의 계수·telemetry(`first50`, `last200`)를 검증하고, X5/X6(오라클), X7(DEQ `z*'`·`z_src`), X8-0(문제 잠재), X8-A(공동 학습 DEQ)를 단일 비교표로 정리하였다.

---

## 2. 결과 표(숫자만)

### 2.1 입력 파일 SHA256 검증 요약

| 구분 | 대상 파일 수 | 일치 수 | 불일치 수 | 비고 |
|---|---:|---:|---:|---|
| `SHA256SUMS_full.txt` 기재 파일 | 18 | 18 | 0 | 전수 로컬 존재 및 해시 일치 |
| `scripts/s5_eval_grader_v2.py` | 1 | 1 | 0 | `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` |
| `handoff/codex/scripts/s5_eval_grader_v2.py` | 1 | 1 | 0 | `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` |
| `handoff/codex/scripts/codex_grade_local.py` | 1 | 1 | 0 | `bd0c7e78e3f5046376d3fb8f4ebe211add08e1e2c99921da54340876f1128d7c` |
| `handoff/codex/data/p6_item_order.json` | 1 | 1 | 0 | `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c` (`n=600`) |
| `configs/s7_train_993.json` | 1 | 1 | 0 | `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c` (`n=993`) |

### 2.2 조건별 성적 및 생성 통계 (`n = 600`)

| 조건 | 역할 / 시드 | 서버=로컬 `boxed` (개 / %) | 서버 `lenient` (개 / %) | 로컬 `lenient` (개 / %) | 서버=로컬 `strict` (개 / %) | `strict \| boxed` (%) | 길이 평균 (tok) | 길이 P50 / P95 | EOS 종료 (개 / %) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C_fmt` | 순정 포맷 기준선 | 213 / 35.50% | 94 / 15.67% | 93 / 15.50% | 77 / 12.83% | 36.15% | 425.2 | 512 / 512 | 235 / 39.17% |
| `X80P47` | 문제 잠재 (2047, matched) | 326 / 54.33% | 72 / 12.00% | 72 / 12.00% | 64 / 10.67% | 19.63% | 402.8 | 512 / 512 | 279 / 46.50% |
| `shuf_X80P47` | 문제 잠재 (2047, shuffled) | 322 / 53.67% | 73 / 12.17% | 73 / 12.17% | 66 / 11.00% | 20.50% | 405.9 | 512 / 512 | 284 / 47.33% |
| `X80P49` | 문제 잠재 (2049, matched) | 317 / 52.83% | 77 / 12.83% | 77 / 12.83% | 71 / 11.83% | 22.40% | 400.8 | 512 / 512 | 291 / 48.50% |
| `shuf_X80P49` | 문제 잠재 (2049, shuffled) | 330 / 55.00% | 80 / 13.33% | 80 / 13.33% | 73 / 12.17% | 22.12% | 395.4 | 505 / 512 | 303 / 50.50% |
| `X80P50` | 문제 잠재 (2050, matched) | 319 / 53.17% | 76 / 12.67% | 76 / 12.67% | 69 / 11.50% | 21.63% | 397.6 | 488 / 512 | 312 / 52.00% |
| `shuf_X80P50` | 문제 잠재 (2050, shuffled) | 346 / 57.67% | 69 / 11.50% | 69 / 11.50% | 66 / 11.00% | 19.08% | 386.4 | 453 / 512 | 327 / 54.50% |

- `X80P47`, `shuf_X80P47`, `X80P49`, `shuf_X80P49`, `X80P50`, `shuf_X80P50` 6개 신규 조건은 `boxed`, `lenient`, `strict` 모두 서버 채점표와 로컬 재채점표가 600문항 전수 일치한다.

### 2.3 Bootstrap 비교 결과 (`B = 10,000`, `seed = 2026`, 문항 단위)

| 비교 항목 | 지표 | 서버 평균 (%p) | 서버 95% CI (%p) | 서버 시드별 (%p) | 로컬 평균 (%p) | 로컬 95% CI (%p) | 로컬 시드별 (%p) | CC 보고값과 차이 |
|---|---|---:|---:|---|---:|---:|---|---:|
| **X8-0 3시드 합산: `matched − shuffled` (O-A)** | `lenient` | `+0.17` | `[-1.50, +1.94]` | `[-0.17, -0.50, +1.17]` | `+0.17` | `[-1.50, +1.94]` | `[-0.17, -0.50, +1.17]` | `0.00%p` |
| **X8-0 3시드 합산: `matched − shuffled` (O-A)** | `strict` | `-0.06` | `[-1.67, +1.61]` | `[-0.33, -0.33, +0.50]` | `-0.06` | `[-1.67, +1.61]` | `[-0.33, -0.33, +0.50]` | `0.00%p` |
| **X8-0 3시드 합산: `matched − shuffled` (O-A)** | `boxed` | `-2.00` | `[-4.61, +0.61]` | `[+0.67, -2.17, -4.50]` | `-2.00` | `[-4.61, +0.61]` | `[+0.67, -2.17, -4.50]` | `0.00%p` |
| **X8-0 3시드 합산: `matched − C_fmt` (O-B)** | `lenient` | `-3.17` | `[-5.67, -0.78]` | `[-3.67, -2.83, -3.00]` | `-3.00` | `[-5.56, -0.56]` | `[-3.50, -2.67, -2.83]` | `0.00%p` |
| **X8-0 3시드 합산: `matched − C_fmt` (O-B)** | `strict` | `-1.50` | `[-3.94, +0.89]` | `[-2.17, -1.00, -1.33]` | `-1.50` | `[-3.94, +0.89]` | `[-2.17, -1.00, -1.33]` | `0.00%p` |
| **X8-0 3시드 합산: `shuffled − C_fmt` (harm)** | `lenient` | `-3.33` | `[-5.89, -0.89]` | `[-3.50, -2.33, -4.17]` | `-3.17` | `[-5.72, -0.72]` | `[-3.33, -2.17, -4.00]` | `0.00%p` |
| **X8-0 3시드 합산: `shuffled − C_fmt` (harm)** | `strict` | `-1.44` | `[-4.00, +1.00]` | `[-1.83, -0.67, -1.83]` | `-1.44` | `[-4.00, +1.00]` | `[-1.83, -0.67, -1.83]` | `0.00%p` |
| 시드 2047 (`X80P47`): `matched − shuffled` | `lenient` | `-0.17` | `[-3.00, +2.67]` | `[-0.17]` | `-0.17` | `[-3.00, +2.67]` | `[-0.17]` | `0.00%p` |
| 시드 2047 (`X80P47`): `matched − shuffled` | `strict` | `-0.33` | `[-3.00, +2.33]` | `[-0.33]` | `-0.33` | `[-3.00, +2.33]` | `[-0.33]` | `0.00%p` |
| 시드 2049 (`X80P49`): `matched − shuffled` | `lenient` | `-0.50` | `[-3.33, +2.33]` | `[-0.50]` | `-0.50` | `[-3.33, +2.33]` | `[-0.50]` | `0.00%p` |
| 시드 2049 (`X80P49`): `matched − shuffled` | `strict` | `-0.33` | `[-3.17, +2.50]` | `[-0.33]` | `-0.33` | `[-3.17, +2.50]` | `[-0.33]` | `0.00%p` |
| 시드 2050 (`X80P50`): `matched − shuffled` | `lenient` | `+1.17` | `[-1.67, +4.00]` | `[+1.17]` | `+1.17` | `[-1.67, +4.00]` | `[+1.17]` | `0.00%p` |
| 시드 2050 (`X80P50`): `matched − shuffled` | `strict` | `+0.50` | `[-2.33, +3.33]` | `[+0.50]` | `+0.50` | `[-2.33, +3.33]` | `[+0.50]` | `0.00%p` |

### 2.4 학습 Manifest 3종 계수 및 Telemetry 대조표

| 조건 | 시드 | `zfile_sha256` (`problem_latents.pt`) | `n_items_trained` / `listed` | `n_updates` / `visits` | `n_unique_ids` | 초기 50걸음 평균 `loss` / `ans_ce` | 마지막 200걸음 평균 `final_loss_last200` (manifest = 재계산) | 마지막 200걸음 평균 `final_ans_ce_last200` (manifest = 재계산) | 마지막 걸음(`1499`) `loss` / `ans_ce` |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `X80P47` | 2047 | `019ebefd40aea9c3...` | `992 / 993` | `1500 / 6000` | `992` | `0.538081 / 3.535675` | `0.096576704` (오차 `0.0`) | `0.564533763` (오차 `0.0`) | `0.131207 / 0.937461` |
| `X80P49` | 2049 | `019ebefd40aea9c3...` | `992 / 993` | `1500 / 6000` | `992` | `0.514311 / 3.287165` | `0.094036207` (오차 `0.0`) | `0.559792392` (오차 `0.0`) | `0.094188 / 0.694142` |
| `X80P50` | 2050 | `019ebefd40aea9c3...` | `992 / 993` | `1500 / 6000` | `992` | `0.522630 / 3.325060` | `0.094808869` (오차 `0.0`) | `0.547494371` (오차 `0.0`) | `0.124139 / 0.636744` |

- `latents.log` 확인: `items 1593 (short<64: 1031) keys 3186 -> /tmp/cts_x8_0/problem_latents.pt` (`993 + 600 = 1,593`문항, 정수 키+문자열 키 `3,186`개, 64토큰 미만 좌측 패딩 문항 `1,031`개).

### 2.5 교차 실험 종합 대조표 (X5/X6 오라클 vs X7 DEQ·`z_src` vs X8-0 문제 잠재 vs X8-A 공동 학습 DEQ, 3시드 합산 서버 채점 기준)

*(동일 브리지 아키텍처 `LatentXAttnBridge` 20,971,524 params, 동일 타깃 `gen_Ctrain.jsonl`, 동일 학습 설정 `steps=1500, bs=4, lr=3e-4, aw=5.0, seeds=2047/2049/2050`)*

| 실험군 | 잠재 원천 및 공간 | 문제 지문 외 정답 정보 유무 | 학습 초반 50걸음 `ans_ce` (3시드 평균) | 학습 마지막 200걸음 `ans_ce` (3시드 평균) | 개발 600 `matched` lenient / strict (%) | 개발 600 `shuffled` lenient / strict (%) | **`matched − shuffled` lenient (95% CI)** | **`matched − shuffled` strict (95% CI)** | `matched − C_fmt` lenient (95% CI) | `shuffled − C_fmt` lenient (95% CI) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`X5/X6-R` (`O`)** | 순정 LM 층 20 (문제+정답 해설 마지막 64tok) | **있음** (정답 해설 포함) | `2.92` | `0.442` | `15.72% / 13.83%` | `12.17% / 10.44%` | **`+3.56%p` (`[+1.61, +5.50]`)** | **`+3.39%p` (`[+1.50, +5.28]`)** | `+0.06%p` (`[-2.39, +2.44]`) | `-3.50%p` (`[-6.00, -1.11]`) |
| **`X8-0` (`P`)** | 순정 LM 층 20 (문제만 마지막 64tok) | **없음** (문제만) | `3.38` | `0.557` | `12.50% / 11.33%` | `12.33% / 11.39%` | **`+0.17%p` (`[-1.50, +1.94]`)** | **`-0.06%p` (`[-1.67, +1.61]`)** | `-3.17%p` (`[-5.67, -0.78]`) | `-3.33%p` (`[-5.89, -0.89]`) |
| **`X7-E` (`E`)** | M1 백본 P4 투영 `z_src` (문제만) | **없음** (문제만) | `3.92` | `0.304` | `14.22% / 11.83%` | `15.39% / 12.22%` | **`-1.17%p` (`[-2.89, +0.56]`)** | **`-0.39%p` (`[-2.00, +1.22]`)** | `-1.44%p` (`[-3.94, +0.94]`) | `-0.28%p` (`[-2.83, +2.11]`) |
| **`X7-A` (`A`)** | 동결 DEQ `z*'` (문제만) | **없음** (문제만) | `4.34` | `0.271` | `13.17% / 10.83%` | `14.50% / 12.39%` | **`-1.33%p` (`[-3.06, +0.33]`)** | **`-1.56%p` (`[-3.33, +0.17]`)** | `-2.50%p` (`[-4.94, -0.22]`) | `-1.17%p` (`[-3.72, +1.28]`) |
| **`X8-A` (`A_unroll`)** | 2걸음 펼침 공동학습 DEQ `z_out` (문제만) | **없음** (문제만) | `4.39` | `0.258` | `13.33% / 10.78%` | `14.00% / 11.89%` | **`-0.67%p` (`[-2.50, +1.17]`)** | **`-1.11%p` (`[-2.78, +0.56]`)** | `-2.33%p` (`[-4.72, +0.06]`) | `-1.67%p` (`[-4.17, +0.67]`) |

---

## 3. 불일치·결함 목록(파일, 줄, 재현 조건)

### 3.1 서버 채점표(`x8_0_items.json`) vs 로컬 재채점표(`x8_0_items_local.json`) 플래그 대조

7개 조건 × 600문항 × 3개 지표(`strict`, `lenient`, `boxed`) = 총 `12,600`개 플래그를 전수 대조한 결과:
- `X80P47`, `shuf_X80P47`, `X80P49`, `shuf_X80P49`, `X80P50`, `shuf_X80P50` 6개 조건(`10,800`개 플래그)에서는 불일치가 **`0`건**이다.
- `C_fmt` 조건에서만 기존 알려진 문항 `id=1503` `lenient` 1건(`0.008%`)이 발생하였다.

| 조건 | 문항 ID | 서버 `[strict, lenient, boxed]` | 로컬 `[strict, lenient, boxed]` | Gold (`extract_answer(solution)`) | 모델 추출 답 (`extract_answer(gen)`) | 원인 및 재현 조건 |
|---|---:|---|---|---|---:|---|
| `C_fmt` | `1503` | `[0, 1, 0]` | `[0, 0, 0]` | `64\%` | `$64\%$` | 비-boxed 추출값 `$64\%$`에 대한 서버 vs 로컬(`math_verify==0.9.0`) 파서 동치 판정 차이 |

### 3.2 `grade.log` 8행 `Timeout during comparison` 확인 (불일치 유발 없음)

- 파일 및 줄: `handoff/codex/INBOX/20261022-0015-x8_0-results/grade.log` 8행
  ```text
  [S0] X80P50: n 600 strict 69 lenient 76 boxed 319
  Timeout during comparison
  [S0] shuf_X80P50: n 600 strict 66 lenient 69 boxed 346
  ```
- 내용: 서버에서 `shuf_X80P50` 채점 중 한 문항에서 `math_verify` 시간제한(`Timeout during comparison`)이 발생하여 `lenient=0`으로 처리되었다. 시간제한을 무효화한 로컬 rev3 래퍼(`codex_grade_local.py`)로 끝까지 동치 판정을 수행한 결과에서도 해당 문항은 비동치(`lenient=0`)여서, `shuf_X80P50`의 600문항 플래그(`strict 66, lenient 69, boxed 346`)는 서버와 로컬이 100% 일치하였다.

---

## 4. 하지 못한 것과 이유

- **서버 직접 접속 확인**: `AGENTS.md` §1 규칙 1(서버 접속 금지)에 따라 서버 내 `/tmp/cts_x8_0/problem_latents.pt` 및 `/tmp/cts_x8_0/bridge_X80P*.pt` 원시 텐서 파일은 직접 열람하지 않고, `INBOX/20261022-0015-x8_0-results/`에 제공된 로그·manifest·생성 JSONL·해시 목록으로만 검증하였다.

---

## 5. 재현 명령

작업 디렉터리 `D:\AI\cts`에서 아래 명령으로 재채점 및 분석 산출물을 재현할 수 있다.

```powershell
# 1) 로컬 rev3 래퍼를 이용한 7개 조건(4,200행) 독립 재채점
python -B handoff/codex/scripts/codex_grade_local.py handoff/codex/INBOX/20261022-0015-x8_0-results/s0spec_local.json handoff/codex/OUTBOX/20261024-1906-C007/x8_0_items_local.json

# 2) SHA256 검증, 조건별 성적/길이/EOS, Bootstrap(B=10000, seed=2026), 플래그 대조, manifest 검증 통합 실행
python -B handoff/codex/OUTBOX/20261024-1906-C007/analyze_c007.py
```

### 원문 로그 및 독립 재계산 출력 발췌

1. **서버 마스터 로그(`cts_x8_0_master.log` 1~19행, 33~40행) 및 `grade.log` 원문**:
```text
=== X8-0 start 2026-10-21 16:57:49 launcher_sha d71b45943ed17a49665a127c6d43e83c46dc303f0ba8497f01712350ad5f2346 ===
ef3fbd5ed5d227c5b765fc01b95bcd920d4433b005e32e3057dcb479f2d1457d  /tmp/cts_x_stage/g2v9_xattn.py
48b71ffb94263284f611a35ce4a58bc9fb7609d9b74c18fe31cc95c8c3afe48f  /tmp/cts_x_stage/g2v10_train.py
14e466006b9a29c4d30181c424f5e6fb9872c781d03c3313c71fc433f7c2f3b3  /tmp/cts_x_stage/g2v9_gen.py
115eb94aa2f9cdb3b3e7344a03316f34bee7eb4ca6185d08546a63a30a02a46c  /tmp/cts_x_stage/g2v8x_problem_latents.py
5172ad4a24e9e9f1de75c471ef0a35836f5296f60940c7c83b4245a5bf07dcae  /tmp/cts_x_stage/make_s0spec_x.py
0c026ea2d7e863d3035a7918cec23dd83206794db7ea2c092ed3aa97c62fad69  results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl
[LAT] rc=0 16:59:27 [SAVE] items 1593 (short<64: 1031) keys 3186 -> /tmp/cts_x8_0/problem_latents.pt
019ebefd40aea9c3af5236d124eefba7869eaaa8dece93679374fb401d22a3ae  /tmp/cts_x8_0/problem_latents.pt
[TR-X80P47] rc=0 17:20:54 [Step 1499] loss 0.1312 ans_ce 0.9375 (n 4) gates [0.0, -0.01, -0.032, 0.019] (1250s)
[TR-X80P49] rc=0 17:20:57 [Step 1499] loss 0.0942 ans_ce 0.6941 (n 4) gates [0.0, -0.008, -0.035, -0.019] (1250s)
[TR-X80P50] rc=0 17:21:01 [Step 1499] loss 0.1241 ans_ce 0.6367 (n 4) gates [-0.0, 0.01, 0.031, 0.018] (1255s)
[GRADER-VERIFIED] scripts/s5_eval_grader_v2.py SHA: 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
[S0] C_fmt: n 600 strict 77 lenient 94 boxed 213
[S0] X80P47: n 600 strict 64 lenient 72 boxed 326
[S0] shuf_X80P47: n 600 strict 66 lenient 73 boxed 322
[S0] X80P49: n 600 strict 71 lenient 77 boxed 317
[S0] shuf_X80P49: n 600 strict 73 lenient 80 boxed 330
[S0] X80P50: n 600 strict 69 lenient 76 boxed 319
Timeout during comparison
[S0] shuf_X80P50: n 600 strict 66 lenient 69 boxed 346
[S0-DONE] /tmp/cts_x8_0/x8_0_items.json
```

2. **로컬 독립 재채점(`codex_grade_local.py`) 표준 출력 원문**:
```text
[LOCAL-SHIM] windows: math-verify per-call multiprocessing timeout replaced by no-op (equivalence logic unchanged)
[GRADER-VERIFIED] scripts/s5_eval_grader_v2.py SHA: 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13
[S0] C_fmt: n 600 strict 77 lenient 93 boxed 213
[S0] X80P47: n 600 strict 64 lenient 72 boxed 326
[S0] shuf_X80P47: n 600 strict 66 lenient 73 boxed 322
[S0] X80P49: n 600 strict 71 lenient 77 boxed 317
[S0] shuf_X80P49: n 600 strict 73 lenient 80 boxed 330
[S0] X80P50: n 600 strict 69 lenient 76 boxed 319
[S0] shuf_X80P50: n 600 strict 66 lenient 69 boxed 346
[S0-DONE] handoff/codex/OUTBOX/20261024-1906-C007/x8_0_items_local.json
```

3. **서버 채점표 기준 독립 재실행(`x8_0_analyze_server_utf8.txt`) vs `cc_x8_0_analysis.txt` 원문 (100% 일치)**:
```text
== per condition
  C_fmt       boxed  35.5%  lenient 15.67%  strict 12.83%
  X80P47      boxed  54.3%  lenient 12.00%  strict 10.67%
  shuf_X80P47 boxed  53.7%  lenient 12.17%  strict 11.00%
  X80P49      boxed  52.8%  lenient 12.83%  strict 11.83%
  shuf_X80P49 boxed  55.0%  lenient 13.33%  strict 12.17%
  X80P50      boxed  53.2%  lenient 12.67%  strict 11.50%
  shuf_X80P50 boxed  57.7%  lenient 11.50%  strict 11.00%
== X8-0 pooled 3 seeds: matched - shuffled (O-A: CI lower > 0)
  lenient  mean +0.17%p  95% CI [-1.50, +1.94]  per-run [-0.17, -0.5, 1.17]
  strict   mean -0.06%p  95% CI [-1.67, +1.61]  per-run [-0.33, -0.33, 0.5]
== X8-0 pooled 3 seeds: matched - C_fmt (O-B: >= +5%p)
  lenient  mean -3.17%p  95% CI [-5.67, -0.78]  per-run [-3.67, -2.83, -3.0]
  strict   mean -1.50%p  95% CI [-3.94, +0.89]  per-run [-2.17, -1.0, -1.33]
== X8-0 pooled: shuffled - C_fmt (harm report)
  lenient  mean -3.33%p  95% CI [-5.89, -0.89]  per-run [-3.5, -2.33, -4.17]
  strict   mean -1.44%p  95% CI [-4.00, +1.00]  per-run [-1.83, -0.67, -1.83]
```

---

## 6. 산출물 SHA256

`OUTBOX/20261024-1906-C007/` 내 생성 파일의 SHA256은 `OUTPUT_MANIFEST.json`에 함께 기록되며 주요 산출물 해시는 아래와 같다.

- `x8_0_items_local.json`: `OUTPUT_MANIFEST.json` 참조
- `c007_analysis.json`: `OUTPUT_MANIFEST.json` 참조
- `x8_0_analyze_server_utf8.txt`: `57941e9d39b56dec6e60bd165b767a4f5e34eedea0c5c18fec7e812a6a848f52`
- `x8_0_analyze_local_utf8.txt`: `OUTPUT_MANIFEST.json` 참조
- `analyze_c007.py`: `OUTPUT_MANIFEST.json` 참조

---

## 7. 의견(판정 아님)

표 2.5에서 `X5/X6-R`(오라클), `X7-A`(DEQ `z*'`), `X7-E`(`z_src`), `X8-0`(문제만 본 순정 LM 층 20 잠재), 그리고 `X8-A`(공동 학습 DEQ)를 나란히 놓고 볼 때, **"통로는 문제 글에 없는 정보가 있을 때만 값을 낸다"**는 CC의 해석에 대한 타당성과 대안·보완 해석은 다음과 같다.

1. **CC 해석의 타당성 (좌표계 불일치 가설의 배제와 조건부 상호정보량 $I(Y; Z \mid X)$ 관점)**:
   - X7까지 남아 있던 핵심 의문은 "`X7-A`/`X7-E`가 작동하지 않은 이유가 잠재에 답 정보가 없어서인가, 아니면 `M1` 백본(`P4`/DEQ)의 표현 공간이 순정 LM 층 20 표현 공간(`z_oracle`)과 달라서(representation mismatch) 브리지가 읽지 못한 것인가?"였다.
   - `X8-0`은 `z_oracle`과 **완전히 동일한 모델(순정 LM)·동일한 층(층 20)·동일한 슬롯 수(`64 × 2560`)**에서 단지 입력만 `[문제 + 정답 해설]` 대신 `[문제]`로 바꾼 대조군이다. 그런데 `X5/X6-R`에서 `+3.56%p` (`[+1.61, +5.50]`)였던 `matched − shuffled` 신호가 `X8-0`에서는 `+0.17%p` (`[-1.50, +1.94]`, strict `-0.06%p`)로 소멸했다.
   - 이는 단순히 표현 공간이나 스케일을 순정 LM 층 20에 맞춘다고 통로가 열리는 것이 아니며, 자기 주의(self-attention)로 이미 문제 지문 $X$ 전체를 직접 참조하고 있는 언어 모델에게 **$X$만 그대로 다시 요약한 잠재 $Z(X)$는 조건부 추가 정보($I(Y; Z(X) \mid X)$)가 0에 가까워 교차주의 어댑터가 $Z(X)$의 문항별 내용에 의존하도록 학습되지 않는다**는 CC의 해석을 직접 뒷받침한다.
2. **대안·보완 해석 1 ("문제만 본 모든 잠재"가 아니라 "추가 계산이 없는 단일 패스 은닉 상태 복사본"의 한계)**:
   - 이론적으로 잠재 $Z = f(X)$가 문제 지문 $X$만의 결정론적 함수라 하더라도, 만약 $f$가 순정 LM의 단일 forward 패스(층 0~20)에서 아직 계산되지 않은 **다단계 추론·산술 중간 결과(computational depth gain)**를 미리 계산해 담고 있다면, 생성 초반 토큰 위치의 LM에게는 계산 지름길(computational shortcut)로서 유효한 신호가 될 수 있다.
   - `X8-0`에서 사용한 $Z_{\text{prob}}(X)$는 별도의 반복 추론 없이 LM 자신의 문제 프롬프트 층 20 은닉 상태를 그대로 복사한 것이므로, 생성 시 이미 LM 자신의 KV 캐시(층 0~20)에 들어 있는 표현과 완전히 중복된다. 즉 `X8-0`은 "문제만 본 잠재는 어떤 형태든 원리적으로 무용하다"기보다, **"LM 자신이 이미 KV 캐시에 갖고 있는 문제 층 20 표현을 외부 잠재 슬롯으로 중복 주입하는 것은 정보 이득이 전혀 없고 브리지 개입 해악(`-3.33%p`)만 남긴다"**는 것을 엄밀하게 입증한 실험이다.
3. **대안·보완 해석 2 (학습 동역학 차이 — 암기 키로도 쓰이지 못한 `X8-0`과 좌측 영벡터 패딩)**:
   - 표 2.5의 학습 마지막 200걸음 `ans_ce`를 보면 `X7-A`(`0.271`)·`X7-E`(`0.304`)는 학습 후반에 992개 문항 식별 키로 과적합되어 오라클(`0.442`)보다 낮은 `ans_ce`를 보인 반면, `X8-0`은 `0.557`로 모든 실험군 중 가장 높다.
   - `latents.log` 4행(`items 1593 (short<64: 1031)`)에서 확인되듯, 정답 해설을 제외한 순수 문제 지문은 전체 1,593문항 중 `1,031`문항(`64.7%`)이 64토큰 미만이어서 `problem_latents.pt`의 64개 슬롯 중 상당 부분이 좌측 영벡터 패딩(`zero-padding`)으로 채워져 있다(반면 `z_oracle`은 긴 해설이 붙어 거의 모든 문항이 64개 유효 슬롯을 꽉 채움). 유효 슬롯 밀도가 낮고 자기 주의 KV 캐시와 완전히 중복되다 보니, 학습 중 어댑터가 잠재에 주의를 기울일 유인이 더 약해졌고, 결과적으로 개발 세트에서 `matched`(`12.50%`)와 `shuffled`(`12.33%`)가 완전히 구분 불가능한 상태가 된 것으로 해석된다.

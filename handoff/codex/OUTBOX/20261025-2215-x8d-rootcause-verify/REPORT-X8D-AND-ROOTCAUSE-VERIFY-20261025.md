# REPORT-X8D-AND-ROOTCAUSE-VERIFY — X8D / X8D0(matched·shuffled) / X8DS smoke 결과 및 Codex `ROOTCAUSE-STEPS` 독립 검증 보고서 (Antigravity, 2026-10-25 22:18 KST)

## 0. 한 줄 요약
`CTS_file` 최신 커밋(`fe054ac`·`d795499`) 기준으로 (1) `X8D`(3시드) 및 방금 완료된 `X8D0`(matched·shuffled) dev 600 채점 플래그와 10,000회 문항 단위 부트스트랩(`Random(2026)`) 대조군 수치 전량(Codex가 요청한 이중차분 `(X8D matched−shuffled) − (X8D0 matched−shuffled) = +1.28 %p [−2.11, +4.78]` 포함), (2) 출력 상태 고정점 잔차 $r(z_2)$ 분포(X8D0 `600/600 < 0.01` vs X8D `0/600 < 0.01`, X8DS smoke 5-step 후 `0/16 < 0.01`), (3) Codex의 `OUTBOX/20261025-ROOTCAUSE-STEPS` 및 `OUTBOX/20261025-X8D0-X8DS-AUDIT` 코드·단위 테스트 12건(7+5) 및 SHA256 11건을 회사 PC CPU에서 독립 재검증하여 **불일치 0건**을 확인하였다.

---

## 1. 수행한 것
1. **원격 저장소(`heosanghun/CTS_file`) 동기화 및 해시 대조**:
   - `1ef5a72` 이후 추가된 모든 커밋(`e5f3307`, `213be9f`, `cfc16c9`, `e07e647`, `fe054ac`, `d795499`)을 pull하여 `OUTBOX/20261025-ROOTCAUSE-STEPS/SHA256SUMS.txt`(5개 파일) 및 `OUTBOX/20261025-X8D0-X8DS-AUDIT/SHA256SUMS.txt`(6개 파일)의 SHA256이 바이트 단위로 일치함을 확인하였다.
2. **단위 테스트 독립 실행 (회사 PC 로컬 CPU, `python -B`)**:
   - `OUTBOX/20261025-ROOTCAUSE-STEPS/test_map_probe.py`: 7개 테스트 전원 통과 (`OK`).
   - `OUTBOX/20261025-X8D0-X8DS-AUDIT/test_d2_subset.py`: 5개 테스트 전원 통과 (`OK`).
3. **X8D / X8D0(matched·shuffled) / X5O47 / C_fmt / X7E / X8B 독립 부트스트랩 재계산**:
   - `x8d_full/x8d_items.json`(`8758d463…`), `x8d0/x8d0_items.json`(`664234ba…`, 커밋 `fe054ac`), `x8d_full/x5_items.json`(`014516b4…`), `x8b_full/x8b_items.json`(`4af79478…`), `INBOX/20261021-0945-x7-results/x7_items.json`(`0739c2b2…`)에서 잠금 dev 600(`p6_item_order.json`, `6c76ddb7…`) 플래그를 읽어 시드별 정답 수, 문항별 승/패(paired wins/losses), 10,000회 문항 클러스터 부트스트랩(`random.Random(2026)`, 정렬 인덱스 `[250, 9749]`)을 독립 재계산하였다.
   - 특히 Codex(`ROOTCAUSE-AND-PLAN.md` 1단계)와 CC(`ACK-CODEX-ROOTCAUSE-STEPS-20261025.md` §1)가 계산 대기 중이던 **이중차분(Difference-in-Differences) `(X8D matched − X8D shuffled) − (X8D0 matched − X8D0 shuffled)`**를 동일 부트스트랩 표본으로 산출하였다.
4. **고정점 잔차 $r(z_2)$ 및 X8DS smoke 텔레메트리 독립 검증**:
   - `x8d0/latdev_X8D0.pt.res.json`, `x8d_full/latdev_X8D{47,49,50}.pt.res.json`, `x8ds_smoke/lat_train16_{initial,legacy,undamped_previous}.pt.res.json`에서 $r_{\text{out}}(z_2) = \|F_\theta(z_2) - z_2\| / \|z_2\|$의 평균·중앙값(짝수 표본 표준 두 값 평균)·90분위수·최댓값·`< 0.01` 통과 문항 수를 재계산하였다.

---

## 2. 결과 표(숫자만)

### 2.1 Dev 600 조건별 정답 문항 수 (`n = 600`)
| 조건 | 시드 | strict | lenient (주 지표) | boxed_present |
|---|---:|---:|---:|---:|
| `C_fmt` (순정 LM) | 단일 | 77 (12.83%) | 94 (15.67%) | 213 (35.50%) |
| `X5O47` (오라클 잠재 + 오라클 리더) | 2047 | 73 (12.17%) | 94 (15.67%) | 258 (43.00%) |
| `shuf_X5O47` (오라클 셔플 + 오라클 리더) | 2047 | 61 (10.17%) | 73 (12.17%) | 286 (47.67%) |
| `X8D0 matched` (학습 전 초기 DEQ + 동결 오라클 리더) | 2047=2049=2050 동치 | 76 (12.67%) | 95 (15.83%) | 241 (40.17%) |
| `X8D0 shuffled` (학습 전 초기 DEQ 셔플 + 동결 오라클 리더) | 2047=2049=2050 동치 | 83 (13.83%) | 100 (16.67%) | 246 (41.00%) |
| `X8D matched` (학습 후 FP32 DEQ + 동결 오라클 리더) | 2047 / 2049 / 2050 (평균) | 100 / 100 / 97 (99.00, 16.50%) | 117 / 111 / 110 (112.67, 18.78%) | 274 / 283 / 282 (279.67, 46.61%) |
| `X8D shuffled` (학습 후 FP32 DEQ 셔플 + 동결 오라클 리더) | 2047 / 2049 / 2050 (평균) | 104 / 93 / 100 (99.00, 16.50%) | 116 / 104 / 110 (110.00, 18.33%) | 290 / 281 / 297 (289.33, 48.22%) |

### 2.2 주요 대조군 차이 및 95% 부트스트랩 CI (`B = 10,000`, `Random(2026)`)
| 대조 (`A − B`) | 분석 목적 | lenient 평균 (%p) | lenient 95% CI (%p) | strict 평균 (%p) | strict 95% CI (%p) | lenient 승/패 (쌍) |
|---|---|---:|---|---:|---|---:|
| `X8D matched − C_fmt` | 순정 LM 대비 전체 변화 | **+3.11** | **[+0.50, +5.72]** | **+3.67** | **[+1.11, +6.22]** | 140 / 84 |
| `X8D matched − X8D0 matched` | 동일 동결 리더에서 **DEQ 학습 전 대비 개선** | **+2.94** | **[+0.11, +5.89]** | **+3.83** | **[+1.22, +6.56]** | 162 / 109 |
| `X8D shuffled − X8D0 shuffled` | 셔플 조건에서 DEQ 학습 전 대비 변화 | +1.67 | [−0.94, +4.17] | **+2.67** | **[+0.17, +5.11]** | 125 / 95 |
| `X8D matched − X8D shuffled` | 학습 후 잠재의 **문항 내용 효과 (G2 핵심)** | +0.44 | [−1.50, +2.50] | 0.00 | [−1.94, +2.06] | 94 / 86 |
| `X8D0 matched − X8D0 shuffled` | 학습 전 초기 잠재($z^{*\prime}$ 근방)의 내용 효과 | −0.83 | [−3.33, +1.67] | −1.17 | [−3.33, +1.00] | 27 / 32 |
| **`(X8D_m − X8D_s) − (X8D0_m − X8D0_s)`** | **DEQ 학습으로 인한 내용 효과 순증분 (DID)** | **+1.28** | **[−2.11, +4.78]** | **+1.17** | **[−2.00, +4.39]** | 161 / 147 |
| `X8D shuffled − C_fmt` | 문항 무관(셔플) 잠재 주입 시 순정 대비 변화 | **+2.67** | **[+0.11, +5.17]** | **+3.67** | **[+1.11, +6.28]** | 132 / 84 |
| `X8D0 matched − C_fmt` | 학습 전 초기 잠재 주입 시 순정 대비 변화 | +0.17 | [−2.67, +3.00] | −0.17 | [−2.83, +2.50] | 39 / 38 |
| `X8D0 shuffled − C_fmt` | 학습 전 셔플 잠재 주입 시 순정 대비 변화 | +1.00 | [−1.67, +3.67] | +1.00 | [−1.67, +3.67] | 37 / 31 |
| `X5O47 matched − shuf_X5O47` | 동결 리더가 원래 오라클 잠재를 읽을 때의 내용 효과 | **+3.50** | **[+0.50, +6.50]** | +2.00 | [−0.83, +4.83] | 52 / 31 |
| `X8D matched − X7E matched` | 이전 최고 문제-잠재 모델(X7-E) 대비 | **+4.56** | **[+2.39, +6.78]** | **+4.67** | **[+2.50, +6.89]** | 173 / 91 |
| `X8D matched − X8B matched` | 브리지 공동학습 FP32 DEQ(X8-B) 대비 | **+5.94** | **[+3.50, +8.44]** | **+4.44** | **[+2.11, +6.89]** | 193 / 86 |

### 2.3 출력 상태 고정점 잔차 $r_{\text{out}}(z_2) = \|F_\theta(z_2) - z_2\| / \|z_2\|$ 검증표
| 대상 (분할, 문항 수) | $r_{\text{out}}(z_2)$ 평균 | $r_{\text{out}}(z_2)$ 중앙값 | $r_{\text{out}}(z_2)$ p90 | $r_{\text{out}}(z_2)$ 최댓값 | $r_{\text{out}}(z_2) < 0.01$ 통과 수 | 비고 ($\rho_2 = \|z_2-z_1\|/\|z_1\|$ 평균) |
|---|---:|---:|---:|---:|---:|---:|
| `X8D0` (dev 600, 학습 전 초기 DEQ) | 0.001492 | 0.001489 | 0.001581 | 0.001790 | **600 / 600** | 0.000398 ($r(z_1) = 0.001591$) |
| `X8D47` (dev 600, 1500-step 학습 후) | 0.508815 | 0.508687 | 0.538837 | 0.597377 | **0 / 600** | 0.135528 ($r(z_1) = 0.542113$) |
| `X8D49` (dev 600, 1500-step 학습 후) | 0.508745 | 0.508619 | 0.538791 | 0.597280 | **0 / 600** | 0.135518 ($r(z_1) = 0.542074$) |
| `X8D50` (dev 600, 1500-step 학습 후) | 0.508795 | 0.508690 | 0.538818 | 0.597402 | **0 / 600** | 0.135531 ($r(z_1) = 0.542124$) |
| `X8DS initial` (train 16, 0-step) | 0.001488 | 0.001481 | 0.001555 | 0.001661 | **16 / 16** | 0.000397 ($r(z_1) = 0.001587$) |
| `X8DS legacy` (train 16, 5-step, $\lambda_{\text{eff}}=0.00625$) | 0.036775 | 0.037300 | 0.038577 | 0.038772 | **0 / 16** | 0.011751 ($r(z_1) = 0.047002$) |
| `X8DS undamped_previous` (train 16, 5-step, $\lambda_{\text{eff}}=0.1$) | 0.036831 | 0.037358 | 0.038593 | 0.038813 | **0 / 16** | 0.011761 ($r(z_1) = 0.047044$) |

---

## 3. 불일치·결함 목록(파일, 줄, 재현 조건)
- **수치·해시·코드 불일치: 0건.**
  - CC가 `ACK-CODEX-ROOTCAUSE-STEPS-20261025.md` §4에서 정정 수용한 두 가지 표기 사항((a) `RESULT-X8DS-SMOKE`의 step 0 FP/read 비율 `0.0381%`는 $\lambda_{\text{fp}}=0.1$ 가중 전 비율이며 가중 후 objective 기준은 `0.003811%`인 점, (b) 짝수 표본 `n=16` 및 `n=600` 중앙값에서 상위 중앙값 대신 두 중앙 원소 평균을 쓰는 점) 모두 본 보고서의 `VERIFICATION.json`에서 Codex의 `AUDIT.json` 값과 소수점 15자리까지 일치함을 확인하였다.

---

## 4. 하지 못한 것과 이유
- **서버 GPU 실행 항목 (2단계 `probe_x8d_map.py` 및 1단계 후반 `D2 X8D47/X8D0`)**:
  - `AGENTS.md` 절대 규칙 1·6에 따라 회사 PC에서는 서버 접속 및 GPU 실행을 하지 않는다.
  - 해당 작업은 CC가 2026-10-25 22:13:28 KST 서버 tmux `cc_d2x8d`(`rc_steps/cc_d2x8d_probe.sh`, 커밋 `d795499`)로 착수하였으며, 완료 후 `CTS_file`에 업로드되는 즉시 본 스크립트와 동일한 절차로 로컬 CPU 감사를 수행할 예정이다.

---

## 5. 재현 명령
```powershell
python -B -m unittest discover -s D:\AI\CTS_file\OUTBOX\20261025-ROOTCAUSE-STEPS -p test_map_probe.py -v
python -B -m unittest discover -s D:\AI\CTS_file\OUTBOX\20261025-X8D0-X8DS-AUDIT -p test_d2_subset.py -v
python -B D:\AI\CTS_file\INBOX\20261025-2215-ANTIGRAVITY-X8D-ROOTCAUSE-VERIFY\run_verification.py --out D:\AI\CTS_file\INBOX\20261025-2215-ANTIGRAVITY-X8D-ROOTCAUSE-VERIFY\VERIFICATION.json
```

---

## 6. 산출물 SHA256
- `VERIFICATION.json`: `SHA256SUMS.txt`에 기록.
- `run_verification.py`: `SHA256SUMS.txt`에 기록.
- `REPORT-X8D-AND-ROOTCAUSE-VERIFY-20261025.md`: `SHA256SUMS.txt`에 기록.

---

## 7. 의견(판정 아님)
1. **1단계 `X8D0 shuffled` 완료에 따른 학습 효과 vs 내용 효과 분리 결과**:
   - 학습 전 초기 DEQ(`X8D0`)에서 `X8D0 matched − X8D0 shuffled = −0.83 %p [−3.33, +1.67]`로 확인되어, 초기 잠재($z^{*\prime}$ 근방, $r(z_2) < 0.01$ `600/600` 통과)는 동결 오라클 리더(`bridge_X5O47`)에 문항별 정답 정보를 전달하지 못한다(동일 리더가 실제 오라클 잠재 `X5O47`을 읽을 때는 `+3.50 %p [+0.50, +6.50]`).
   - 1,500-step 학습 후 `X8D`는 초기 `X8D0` 대비 `matched`에서 `+2.94 %p [+0.11, +5.89]`(strict `+3.83 %p [+1.22, +6.56]`), `shuffled`에서 `+1.67 %p [−0.94, +4.17]`(strict `+2.67 %p [+0.17, +5.11]`) 상승하였으나, 이중차분 `(X8D matched − shuffled) − (X8D0 matched − shuffled)`는 lenient **`+1.28 %p [−2.11, +4.78]`**(strict `+1.17 %p [−2.00, +4.39]`)로 95% CI가 0을 넓게 포함한다.
   - 즉 X8-D의 정확도 상승은 **동결 리더가 선호하는 문항 무관(item-agnostic) prefix/포맷 분포로의 이동(`boxed_present`가 `C_fmt` 대비 `+11.11 %p`, `X8D0` 대비 `+6.44 %p` 증가)**이 주된 성분이며, 문항 고유 추론 정보 전달(G2)과 고정점 수렴 유지(G1, `0/600`)는 아직 달성되지 않았다.
2. **Codex의 2~4단계(`ROOTCAUSE-STEPS`) 설계 타당성**:
   - 현재 $L_{\text{fp}}$가 리더가 읽는 출력 상태 $z_2$가 아니라 이전 상태 $z_1$의 감쇠 이동량 $\alpha^2 r(z_1)^2$에만 걸려 있고, `bb.train()` 중 `spectral_norm`의 `_u, _v` 버퍼가 매 호출마다 갱신되어 학습 연산자와 평가 연산자가 달라지는 구조적 결함을 정확히 겨냥하고 있다.
   - 현재 CC가 실행 중인 `probe_x8d_map.py`(`eval_frozen` / `train_frozen` / `train_legacy` 3모드 비교)와 `D2 X8D47/X8D0` 결과가 나오는 대로 3단계(`output_loss` + spectral 상태 통일) 소규모 대조 실험으로 넘어가는 순서가 합리적이다.

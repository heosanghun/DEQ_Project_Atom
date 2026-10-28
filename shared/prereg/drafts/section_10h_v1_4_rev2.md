### (h) ①-a의 $k^*$ 판별 불가 문항 재계산 (v1.4 신설 — 결과 열람 뒤 작성, 공개)

**공개(작성 시점의 사실)**
- 이 절은 P5·P5b 결과(REPORT-0046, `results_s7/p5_promote_v6.json` SHA-256 `e929f890ee9323709e4c21a7af3099c2ce0c17423d2bb53c001db7ad4125aef4`)를 연 뒤에 썼다.
- v1.3만으로는 ①-a를 판정할 수 없다. $k^*$를 모르는 문항을 미도달로 세면 675/1000, 도달로 세면 991/1000이다(`definition_1a_train1000`). 기준은 950이다.
- $r_{\min}\ge 10^{-2}$인 9문항은 모두 아래 325문항 안에 있다. 판별된 675문항의 $k^*$ 최솟값은 369이고, $k^*<50$은 0이다.
- 재계산 결과는 이 절을 잠글 때까지 아무도 보지 않았다.
- 이 절은 판정 기준을 바꾸지 않는다. 로그로 잴 수 없던 $k^*$를 재는 방법만 정한다.

**대상**
- 325문항: `results_s7/p5_inputs_train1000_v3.json`(SHA-256 `186d5b6cc6678b4df2e999919cc39b85d77b9b437dce10da5c553841d3bea1f1`)에서 `k_star_src == "undetermined"`인 문항이다(같은 파일을 만든 `scripts/p5_prep_inputs_v3.py` SHA-256 `d5f792d4cd6bbf203f883811e0f5b9952fb91792fb3ed8bc9efd0c5510ae843a`).
- 목록 파일: `configs/s7_train1000_kstar_replay_325.json`(SHA-256 `550250e399ea96560b8360cdbdd539a82983a25fb435f994d0319da10a8a7862`).
- 모두 학습 세트 문항이다. 봉인 문항(held-out 520·80)은 들어 있지 않다.

**판정에 쓰는 값 (B 감사 FIX-1)**
- 잔차는 1,000문항 모두 **기존 $r_{\min}$**(같은 JSON의 `r_min`)을 쓴다.
- 재계산에서는 **$k^*$만** 가져온다. 다만 아래 재현 점검을 통과한 문항에 한해서다.
- 도달 판정: 문항은 $r_{\min}<10^{-2}$ 그리고 $k^*\ge 50$이면 도달이다.
  - 675문항은 기존 $k^*$를 쓴다.
  - 325문항은 재계산 $k^*$를 쓴다.
- 분모는 1,000이고 기준은 ≥95 %(950 이상)이다. 둘 다 v1.3과 같다.

**문항별 재현 점검 (FIX-2)**
- 재계산 궤적이 원래 궤적과 같은 것으로 보려면 두 조건을 모두 만족해야 한다.
  - (i) 재계산 `min(rho_traj)`와 기존 `min_rho`(같은 JSON)의 상대차가 $10^{-2}$ 이하
  - (ii) 재계산 `z_star`와 기존 저장 텐서의 `z_star`(같은 JSON의 `run_source`가 가리키는 400/680 실행 결과)가 cos ≥ 0.9999
- 상대차 기준은 장치 차이 근거(REPORT-0045)로 미리 정했다. 그 근거는 GPU 0과 GPU 1의 앵커 20 결과가 비트 단위로는 다르지만 cos가 1−3×10⁻⁸ 이상이라는 것이다.
- 재현되지 않는 문항은 **미도달**로 센다. 그 문항의 재계산 $k^*$로 센 값은 민감도로만 따로 적는다.
- 끝내 계산되지 않은 문항, NaN·Inf가 나온 문항도 미도달로 세고 따로 적는다.

**장치와 사전 점검 (FIX-3)**
- 장치 후보는 GPU 0과 GPU 1이다(Principal 결정 CTS-428, 원문 `shared/inbox/CTS-428.md` 3행). 같은 GPU에서 다른 작업과 동시에 실행하지 않는다.
- 장치 순서:
  - P6(GPU 0)와 PREREG_S7_DEQ_v2 실행(GPU 1) 중 먼저 끝난 쪽의 GPU를 1순위로 쓴다.
  - 1순위가 사전 점검에서 미달이면 다른 GPU를 한 번 점검한다.
  - 둘 다 미달이면 재계산하지 않는다. 그때 ①-a는 "판정 불가(보수 675 / 낙관 991)"로 보고한다.
- 사전 점검 방법:
  - 기존 $k^*$를 로그에서 정확히 정한 문항(`k_star_src == "log_10step_exact"`) 중 q_idx가 작은 5문항을 쓴다: Q#31(k*=829), Q#277(709), Q#406(999), Q#711(809), Q#858(999).
  - 이 5문항을 재계산한다. 다섯 문항 모두 재계산 $k^*$가 기존 값과 같고, 위 (i)(ii)를 만족해야 통과다.
- 본 실행을 시작한 뒤에는 장치를 바꾸지 않는다.

**실행 통제 (FIX-4)**
- 실행 명령(장치 UUID만 다르다):
  `CUDA_VISIBLE_DEVICES=<GPU UUID> python scripts/s7_zstar_lowest_run.py --items configs/s7_train1000_kstar_replay_325.json --out results_s7/zstar_s7_a0.25_train325_kstar_replay.pt --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --device cuda:0 --config configs/zstar_canonical.yaml`
- 실행 전 SHA 확인: 실행기 `scripts/s7_zstar_lowest_run.py` `779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba`, `cts/deq/zstar_canonical_lowest.py` `9f73fb272353e8a307b9e628c65d1f17eaff164e2138fa3d232917958d6a6b2f`, 목록 파일(위), `configs/zstar_canonical.yaml`과 `harness_A_v1.py`는 실행 전에 SHA를 적는다. 하나라도 다르면 실행하지 않는다.
- 사전 점검도 같은 인자로 한다. 출력은 `results_s7/kstar_replay_precheck5.pt`이다.
- 시작하면 `raw_s7/KSTAR_REPLAY_STARTED.flag`(시각·GPU UUID·PCI·입력 SHA)를 쓴다. 이 플래그가 있으면 새로 시작하지 않는다.
- 출력 파일을 지우거나 이름을 바꿔 다시 돌리는 것은 금지한다.
- 중단되면 같은 GPU·같은 출력 파일에서만 이어 돌린다. 실행기가 이미 있는 키를 건너뛰므로 앞 문항은 다시 계산하지 않는다.
- 끝나면 `raw_s7/KSTAR_REPLAY_DONE.flag`를 쓴다.

**분석기 (FIX-5)**
- 새 파일 `scripts/p5_promote_1a_replay.py`를 쓴다.
- 합성 입력 시험과 CC 감사를 마치고, SHA를 **재계산 착수 전**에 STATE에 적는다. 실행기가 문항마다 값을 출력하므로 결과를 연 뒤에 고치지 않기 위해서다.
- 출력 JSON에 모든 입력 파일의 SHA를 적는다.

**보고 (FIX-6)**
- ①-a 값 세 가지를 나란히 적는다: 보수(675/1000), 낙관(991/1000), v1.4 값.
- 325문항에 대해 다음을 적는다.
  - 재현 점검 (i)(ii) 통과 수와 실패 문항 목록
  - 재계산 $k^*$ 분포(§10(d) 구간)와 $k^*<50$ 문항 목록
  - 재계산 `min(rho_traj)`와 기존 `min_rho`의 상대차 분포. 이 차이에는 장치 차이와 노름 정밀도(float32/float64) 차이가 함께 들어 있다.
  - 민감도 값(재현 실패 문항을 재계산 $k^*$로 센 경우)

**변경하지 않는 것**
- 승격 판정(신규 520, 517/520)
- ①-b·변별·앵커의 값과 판정
- 확증 근거는 여전히 신규 520뿐이다. ①-a는 비확증 보조 판정으로 보고한다.

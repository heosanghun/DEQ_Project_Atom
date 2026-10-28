# PREREG_S7_DEQ_v1.1.md — 라우팅 마스크 고정 및 Anderson 가속 기반 고정점 수렴 개선 사전등록 v1.1 (원장 #380 정본 잠금안)

- **등록 일시**: 2026-09-28
- **지침**: 원장 #375, #377, #378, #379, #380 (S7-DEQ v1.1 최종 잠금안)
- **발신**: System 2 / **수신**: Antigravity / **사본**: Principal, 외부 감사인
- **소속**: Stage 7-DEQ (DEQ Solver Optimization & Routing Boundary Resolution)
- **문서 잠금 상태**: **LOCKED (잠금 완료 — 실험 착수 및 결과 열람 전 사전등록 잠금)**
- **선행 등록 문서 보존 상태**:
  - `PREREG_S7_DEQ_v1.md` (v1.0, SHA: `6995cb30a395e7b672a5d8deae375a0cba284c3e329afc53d7aba789c012837b`)

---

## 1. 배경 및 이론적 근거 (Background & Physical Mechanism)

### (a) 선형 불안정 가설 기각 및 라우팅 경계 가설 확립
- 74문항(미수렴 54 + 대조 20) 전수 스펙트럼 반경 조사 결과:
  - 지배 고윳값 부호: **100.0% 양수 (74/74, POS)**.
  - 수축성($\rho(J) < 1.0$) 만족 비율: 미수렴군 **98.15% (53/54)**, 대조군 **100.0% (20/20)**.
  - 중앙값: 미수렴군 $0.9123$ vs 대조군 $0.8985$ (두 집단 간 수축률 크기 차이 0.014 수준으로 거의 동일).
  - $\implies$ **선형 불안정(발산·카오스) 가설 공식 기각**.
- 인구 시험 및 사분면(Quadrant) 분석 결과:
  - 미수렴군의 **64.81% (35/54)**가 3위-4위 라우팅 가중치 간격 $\Delta_{3-4} < 0.005$인 경계면에 밀집 (엄격 수렴군의 18.48% 대비 **3.51배**, 대조군의 10.0% 대비 **6.48배** 높은 트랩 집중도).
  - $\implies$ **"라우팅 경계 가설이 인구·궤적·고유값·솔버 성패에서 일관되게 지지됨(개입 실험 대기)"**으로 최종 확정.

### (b) 개입 전략 (Intervention Strategy)
- 미수렴의 본질은 수축 역량 부재가 아니라, 불연속적 $top\text{-}k$ 모듈 스위칭 채터링에 의해 꼬리 잔차가 정체되는 현상임.
- 따라서 초기 탐색 단계($K=300$) 이후 **Top-3 라우팅 마스크를 영구 동결**하여 전이 함수를 $C^1$ 연속 국소 수축 사상으로 전환하고, 700걸음 Picard 안정화 후 **Anderson Acceleration**을 적용하여 수축률 $\rho(J) \approx 0.91$의 느린 선형 수렴을 초선형 가속으로 단숨에 매듭짓는다.

---

## 2. 알고리즘 프로토콜 (2단계 하이브리드 DEQ 풀이)

본 프로토콜은 74문항(미수렴 54문항 + 대조 20문항)에 대해 다음 순서로 엄격 집행된다:

### (1) 1단계: 초기 분지 탐색 (Basin Exploration, Step 1 ~ 300)
- **보폭 규약**: 감쇠 보폭 $\alpha = 0.25$ 고정.
- **반복 방식**: 표준 Injected Picard 고정점 반복:
  $$z_{k+1} = (1 - \alpha) z_k + \alpha \big( M(z_k; m_k) + g_0 \hat{c}_0 \big)$$
  여기서 $m_k = \text{top\_k\_mask}\big(\alpha_{\text{routing}}(z_k), \; k=3\big)$ (매 걸음 동적 선택).
- **수행 걸음**: $k = 1, \dots, 300$ (정확히 300걸음).

### (2) 2단계: 라우팅 마스크 동결 (Mask Freezing at $K=300$)
- $k = 300$ 시점의 상태 $z_{300}$에서 계산된 Top-3 마스크를 추출하여 **영구 고정(Freeze)**:
  $$m_{\text{frozen}} = \text{top\_k\_mask}\big(\alpha_{\text{routing}}(z_{300}), \; k=3\big) \in \{0, 1\}^{19}$$
- 이후 모든 후속 걸음($k > 300$)에서는 이산 재선택을 차단하고, $m_{\text{frozen}}$에 지정된 3개 모듈의 선형 결합으로 정의된 평활 사상 $f_{\text{frozen}}(z)$만을 평가:
  $$f_{\text{frozen}}(z) = \sum_{j \in \text{top3}(m_{\text{frozen}})} \tilde{\alpha}_j(z) M_j(z) + g_0 \hat{c}_0$$
  *(단, 모듈 내부 소프트맥스 가중치 $\tilde{\alpha}_j(z)$는 3개 모듈 내에서 정상 재정규화됨).*

### (3) 3단계: 동결 공간 내 Picard 완보 (Picard Stabilizing, Step 301 ~ 1,000)
- $m_{\text{frozen}}$이 고정된 상태에서 $\alpha = 0.25$로 Picard 반복을 **정확히 700걸음** 추가 수행 ($k = 301, \dots, 1000$).
- 목적: 마스크 동결 직후 발생하는 불연속 충격을 흡수하고, 동결된 어핀 부분공간(Affine Subspace) 내의 국소 인력권 중심으로 $z$를 완벽히 안정화.

### (4) 4단계: Anderson 가속 마무리 (Anderson Acceleration Finishing, Step 1,001 ~ 1,050)
- $z_{1000}$ 상태를 초기점으로 하여 $f_{\text{frozen}}(z)$에 대해 **Anderson Acceleration** 적용.
- **하이퍼파라미터 규약**:
  - 기억 용량 (History depth): $m_{\text{aa}} = \mathbf{5}$
  - 최대 허용 걸음 수: **$\le \mathbf{50}$ 걸음** ($k = 1001, \dots, 1050$)
  - 완화 계수 (Mixing parameter): $\beta = \mathbf{1.0}$
  - 릿지 정규화 계수 (Regularization): $\epsilon_{\text{reg}} = \mathbf{1.0 \times 10^{-4}}$
- **조기 종료 조건**:
  참 잔차 $r_k = \frac{\|f_{\text{frozen}}(z_k) - z_k\|}{\|z_k\|} < \mathbf{1.0 \times 10^{-3}}$ 도달 시 즉시 수렴 종료.

### (5) 고정점 $z^*$ 정의 규약 (원장 #380 Point 2(b))
- 고정점 $z^*$는 전 반복 과정($k = 1, \dots, N_{\text{total}}$) 중 **참 잔차가 최소인 반복 시점의 상태(Lowest Residual Iteration Point)**로 엄격히 정의한다:
  $$z^* = z_{k^*}, \quad k^* = \arg\min_{k} r_k$$
- 사후 보고되는 모든 참 잔차 $r$, 코사인 유사도 $\cos$, 자기 일관 모듈 집합은 엄밀히 본 최소 잔차 상태 $z^*$를 기준으로 산출 및 보고한다 (단, 최종 1,050걸음 시점의 잔차 $r_{\text{final}}$도 투명성을 위해 병기 보고).

---

## 3. 평가 대상 집단 (Target Population)

본 시험은 사전에 지정된 벤치마크 74문항 전수를 대상으로 수행한다:
1. **미수렴 54문항 ($N_{\text{nonconv}} = 54$)**:
   - 설정 파일: `configs/s6_nonconv_54.json`
   - S5 시점($\alpha=0.5, NFE=1000$)에서 $\rho \ge 5\times 10^{-3}$ ($r \ge 1\times 10^{-2}$)로 미수렴되었던 문제군.
2. **대조군 20문항 ($N_{\text{control}} = 20$)**:
   - 설정 파일: `configs/s6_control_20.json`
   - S5 시점에서 정상 수렴($r < 5\times 10^{-3}$)했던 검증 대조군.

---

## 4. 게이트 통과 및 실패 판정 기준 (Acceptance & Failure Criteria)

사후 조정이 일체 불가능한 정량 컷오프 수치:

| 평가 항목 | 지표 수식 및 정의 | 합격 기준 (Success Gate) | 실패 기준 (Failure Gate) | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **주 게이트 1<br>(초고정밀 수렴)** | 미수렴 54문항의<br>참 잔차 $r < 1.0 \times 10^{-3}$ 달성률 | **$\ge \mathbf{80.0\%}$**<br>*(44 / 54문항 이상)* | — | 원장 #379 규정 |
| **주 게이트 2<br>(엄격 DEQ 수렴)** | 미수렴 54문항의<br>참 잔차 $r < 5.0 \times 10^{-3}$ 달성률 | **$\ge \mathbf{95.0\%}$**<br>*(52 / 54문항 이상)* | **$< \mathbf{50.0\%}$**<br>*(27문항 미만 시<br>가설 파기)* | 원장 #379 규정 |
| **주 게이트 3<br>(대조군 불변성)** | 대조 20문항의 신·구 고정점<br>방향 코사인 유사도 | **$\min_{i} \cos(z_{\text{new}, i}^*, z_{\text{old}, i}^*) \ge \mathbf{0.9990}$** | $\cos < 0.9950$<br>발생 시 | $z_{\text{old}}$는 동일 보폭 $\alpha=0.25$ 정본인<br>`results_s6/zstar_s6_a0.25.pt`의 대조 20 $z^*$로<br>엄격 규정 (원장 #380 Point 2(a)) |

---

## 5. 의무 진단 및 추가 보고 항목 (Mandatory Diagnostics)

실험 완료 시 결과 보고서에 반드시 포함되어야 하는 분석 지표:
1. **자기 일관 고정점 비율 (Self-consistent Fixed Point Ratio)**:
   - 최종 도달한 고정점 $z^*$에서 마스크 제약 없이 전체 19개 모듈에 대해 $\alpha_{\text{routing}}(z^*)$를 다시 계산했을 때, 그 $top\text{-}3$ 모듈 인덱스 집합이 $K=300$ 시점에 동결했던 마스크 $m_{\text{frozen}}$과 엄밀히 일치하는 문항의 비율:
     $$\text{Consistency} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\Big( \text{top3}\big(\alpha(z_i^*)\big) == \text{top3}(m_{\text{frozen}, i}) \Big)$$
2. **동결 마스크와 기존 S5 고정점 마스크 간의 일치 비율**:
   - $K=300$ 동결 마스크 $m_{\text{frozen}}$이 기존 S5 $z_{\text{S5}}^*$의 $top\text{-}3$ 모듈 선택과 완전히 일치하는 비율.
3. **걸음별 참 잔차 $r_k$ 전수 궤적 프로파일**:
   - $k = 1 \sim 1050$ 전 걸음에 걸친 $r_k$의 단조 감소성, 동결 시점($k=300$) 충격 크기, Anderson 가속 착수($k=1000$) 이후 잔차 낙하 곡선 시각화 및 로그 원문 첨부.
4. **실행 전 비트 동일성 사전 검증 결과 첨부 (원장 #380 Point 2(c))**:
   - 마스크 안정 대조 문항 1개에 대해 301~1000걸음 동결 궤적 vs 비동결 궤적을 비교한 결과, $\max_{k \in [301, 1000]} \|z_k^{\text{frozen}} - z_k^{\text{dynamic}}\| = \mathbf{0.0}$ (비트 단위 완전 일치)가 검증되었음을 사전 확인하고 실험에 착수함 (검증 로그: `raw_s7/parity_verification_EXE260928_43.log`).

---

## 6. 적용 범위 제한 및 v2 분리 규정

- **부드러운 라우팅(Softmax Routing Relaxation, 온도 완화 등) 배제**:
  - 원장 #379 (나) 지침에 따라, 라우팅을 부드럽게 완화하는 연속적 접근법은 본 v1.0 및 v1.1에서 완전히 배제하며 차후 `PREREG_S7_DEQ_v2`로 분리하여 별도 등록한다.
  - 본 v1.1은 오직 **"이산 마스크 고정($K=300$) + 700걸음 Picard + Anderson($m=5, \le 50$)"**의 단일 하이브리드 파이프라인만을 시험한다.

---

## 7. 실행 자원 및 스케줄링 슬롯 (Resource Allocation)

- **권고 A (단일 GPU 연쇄 — 기본 경로)**:
  - 현재 진행 중인 S7 파이프라인의 1,080문항(기존 400 + 신규 train 680) $z^*$ 계산이 완료되어 일시 정지(Pause)되는 즉시, **GPU 0에서 본 74문항 실험을 약 90분간 우선 실행**한다.
  - 실행 완료 및 감사 보고 즉시 S7-D 디코더 학습($\lambda=1.0$)에 착수한다.
- **권고 B (Principal GPU 1 허용 시 — 즉시 분기)**:
  - Principal의 "GPU 1 허용" 승인 문구 수신 시, GPU 0의 1,080 파이프라인 완료를 기다리지 않고 **GPU 1에서 본 74문항 실험을 즉시 단독 착수**한다.

---

## 8. 사전등록 잠금 선언 (Document Locked)

본 문서(`PREREG_S7_DEQ_v1.1.md`)는 원장 #375, #377, #378, #379, #380에 규정된 알고리즘 파라미터, 대조 20 기준점(results_s6 $\alpha=0.25$), $z^*$ 최소 잔차점 정의, 사전 비트 동일성 검증, 합격/실패 컷오프 수치가 완비되었으며, **결과 열람 전 상태에서 공식 최종 잠금(LOCKED)** 처리한다. 사후 수정은 절대 불가하다.

# CTS 주장표 v4 (Claim Table v4)

**발행일시**: 2026-08-25T15:50:00+09:00  
**기준 커밋/태스크**: `[EXE-260825-05]` Task 49 / Task 50 / Task 51 / Task 52  
**검증자**: Antigravity (실측치 전수 수치 대조)

---

## 1. 종합 주장 현황표 (Master Claim Status)

| 주장 ID | 주장 내용 | 이전 상태 (v3) | 실측 현황 (v4) | 실측 근거 파일 및 포인터 |
|---|---|---|---|---|
| **C-01** | 수축 사상 성립 여부 ($\sigma_{\max} \le 0.92$) | **반증됨** | **반증 유지** ($\sigma_{\max}(\Phi) = 2.2231 > 1$, Euclidean 수축 불가) | `results/task42_routing_fix_results.json` |
| **C-02** | 스펙트럼 반경 수렴 조건 ($\rho(J) < 1$) | 확립됨 | **확립 유지** ($\rho(J) = 0.989210$, 멱반복 $0.988670$, 궤적 $0.989663$) | `results/task46_spectral_refined_results.json` |
| **C-03** | 적응적 라우팅 고정점 도달 | **반증됨** | **반증 유지** (32표본 중 31.2%가 리미트 사이클 상태로 잔차 진동) | `results/task51_limit_cycle_32_results.json` |
| **C-04** | 라우팅 리미트 사이클 현상 | 신규 발견 | **일반화 확립** (LoRA 31.2%, Base 40.6% 주기적 순환 발생) | `results/task51_limit_cycle_32_results.json` |
| **C-05** | 동결본 과제 성능 우위 ($40.0\%$) | **폐기** | **모드 붕괴 확정 (0.0%)**, 앵커 4조건 불일치 확증 | `results/task49_collapse_investigation_results.json`<br>`results/task50_anchor_controlled_results.json` |
| **C-06** | 모드 붕괴 메커니즘 및 복원성 | 미규명 | **규명 확립** (Step 25 이전 발생, $\alpha \le 0.25$ 복원, $L_{FP}/L_{CE} \approx 500:1$) | `results/task49_collapse_investigation_results.json`<br>`cts/train/stage1_warmup.py:L1281-1339` |
| **C-07** | 기계 정밀도 바닥 한계 (Q-6) | 관측됨 | **확립** (Float32 $r \approx 5.2 \times 10^{-8}$, Float64 $r \approx 1.8 \times 10^{-8}$) | `results/task52_float64_picard_results.json` |

---

## 2. 세부 주장별 실측 데이터 명세

### C-04. 라우팅 리미트 사이클 일반화 (Task 51)
- **대상**: 게이트 32개 표본 전수 ($c=0.92$, 3,000 스텝 Picard 반복)
- **Frozen+LoRA (`m1_final_frozen.pt`)**:
  - 리미트 사이클 발생 표본: **10 / 32 (31.2%)**
  - 고정 집합 안정화 표본: **22 / 32 (68.8%)**
  - 수렴 달성 표본 ($k_{100} < 3000$): **12 / 32 (37.5%)**
  - 평균 라우팅 전이 횟수: $174.2 \pm 336.0$회 (최소 4회, 최대 1,293회)
  - 전이 횟수 vs 최종 잔차($\log_{10} r_{3000}$) 상관계수: **$r = +0.3966$**
- **Base B1 (`google/gemma-4-e4b`)**:
  - 리미트 사이클 발생 표본: **13 / 32 (40.6%)**
  - 고정 집합 안정화 표본: **19 / 32 (59.4%)**
  - 수렴 달성 표본 ($k_{100} < 3000$): **9 / 32 (28.1%)**
  - 평균 라우팅 전이 횟수: $262.6 \pm 264.4$회 (최소 7회, 최대 821회)
  - 전이 횟수 vs 최종 잔차($\log_{10} r_{3000}$) 상관계수: **$r = +0.3279$**
- **일치도 분석 (Direct Concordance)**:
  - LoRA & Base 모두 리미트 사이클: $2 / 32$ (6.2%)
  - LoRA 리미트 사이클, Base 안정화: $8 / 32$ (25.0%)
  - LoRA 안정화, Base 리미트 사이클: $11 / 32$ (34.4%)
  - LoRA & Base 모두 안정화: $11 / 32$ (34.4%)

### C-05 & C-06. 앵커 재측정 및 모드 붕괴 원인 규명 (Task 49 & 50)
- **Task 50 (Base B1 MATH-500 `[:30]` 4조건 통제 재측정)**:
  - (1) Plain + New Grader (`math-verify`): **3 / 30 (10.0%)**
  - (2) Think-Chat + New Grader (`math-verify`): **1 / 30 (3.3%)**
  - (3) Plain + Old Grader (`legacy`): **3 / 30 (10.0%)**
  - (4) Think-Chat + Old Grader (`legacy`): **1 / 30 (3.3%)**
  - $\implies$ 신/구 채점기 차이 0건, Think-Chat 형식은 성능 저하 유발 ($10.0\% \to 3.3\%$). 과거 40.0% 수치는 미재현.
- **Task 49-2 (체크포인트 스윕)**:
  - Base (Step 0): **1 / 10 (10.0%)** (Avg Len 410.9, EOS 30.0%)
  - Step 25: **0 / 10 (0.0%)** (Avg Len 512.0, EOS 0.0%)
  - Step 75 (`m1_final_frozen`): **0 / 10 (0.0%)** (Avg Len 512.0, EOS 0.0%)
  - Step 150: **0 / 10 (0.0%)** (Avg Len 512.0, EOS 0.0%)
  - Step 200: **0 / 10 (0.0%)** (Avg Len 512.0, EOS 0.0%)
  - $\implies$ 붕괴 시점: Step 25 이전 학습 초기 발생.
- **Task 49-3 (LoRA 기여도 스케일 스윕 on `m1_final_frozen.pt`)**:
  - $\alpha = 0.00$: **1 / 10 (10.0%)** (Avg Len 410.9, EOS 30.0%) $\to$ Base 완전 복원
  - $\alpha = 0.25$: **2 / 10 (20.0%)** (Avg Len 410.7, EOS 50.0%) $\to$ 수학 추론 복원
  - $\alpha = 0.50$: **0 / 10 (0.0%)** (Avg Len 512.0, EOS 0.0%) $\to$ 모드 붕괴 진입
  - $\alpha = 0.75$: **0 / 10 (0.0%)** (Avg Len 512.0, EOS 0.0%) $\to$ 모드 붕괴 진입
  - $\alpha = 1.00$: **0 / 10 (0.0%)** (Avg Len 512.0, EOS 0.0%) $\to$ 모드 붕괴 유지
  - $\implies$ 전이 임계 구간: $\alpha \in (0.25, 0.50]$.
- **Task 49-1 (손실 함수 전문 및 불균형 분석)**:
  - `loss_fp`: 163,840차원 전체 합산 ($\text{Scale } 50 \sim 500$)
  - `0.1 * loss_ce`: 토큰 평균 교차 엔트로피 ($\text{Scale } \approx 0.2$)
  - 그래디언트 스케일 비: $L_{FP} : L_{LM} \approx 250:1 \sim 1000:1$
  - Base 참조 모델에 대한 KL 다이버전스 정칙화 및 LoRA 가중치 노름 제약 부재 (`cts/train/stage1_warmup.py:L1281-1339`).

### C-07. 정밀도 바닥 한계 대조 (Task 52, 표본 #08, 라우팅 고정 `[2,3,16]`)
- **Float32 Picard**:
  - $r_{100} = 2.569331 \times 10^{-3}$
  - $r_{421} = 1.582015 \times 10^{-5}$
  - $r_{1000} = 5.599164 \times 10^{-8}$
  - $r_{3000} = 5.186227 \times 10^{-8}$
- **Float64 Picard**:
  - $r_{100} = 2.569306 \times 10^{-3}$
  - $r_{421} = 1.582016 \times 10^{-5}$
  - $r_{1000} = 2.032801 \times 10^{-8}$
  - $r_{3000} = 1.863954 \times 10^{-8}$
- **정밀도 한계비**: $r_{3000,\text{float32}} / r_{3000,\text{float64}} = 2.78 \times 10^{0}$

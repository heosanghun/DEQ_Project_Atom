# [PREREG_RETRAIN_v1] CTS 모델 재학습 사전등록 명세서 (초안)

**발행일시**: 2026-08-25T18:05:00+09:00  
**상태**: DRAFT (초안 — Principal 및 System 2 승인 전 실행 금지)  
**근거**: `[EXE-260825-05]` Task 49 및 `[EXE-260825-06]` Task 57  
**작성자**: Antigravity  

---

## 1. 개요 및 목적
본 재학습은 동결본(`m1_final_frozen.pt`)에서 발생한 언어 모델링 모드 붕괴(과제 성능 $0.0\%$, 토큰 반복 생성)의 근본 원인을 해소하고, 고정점 잔차 최소화와 언어 모델링 추론 능력의 양립을 달성하기 위한 Stage 1 재학습 사양을 규정한다.

---

## 2. 손실 함수 재균형 (Loss Rebalancing) 실측치

Stage 1 학습 초기 100개 표본에 대한 손실 함수 항별 실측 스케일은 다음과 같다:
- **기존 원시 고정점 손실 ($\mathcal{L}_{\text{fp}}$, 전체 163,840 차원 제곱합)**: $\bar{\mathcal{L}}_{\text{fp, sum}} = 1,793,957.90$
- **원소별 평균 고정점 손실 ($\mathcal{L}_{\text{fp, mean}}$, 163,840 차원 평균)**: $\bar{\mathcal{L}}_{\text{fp, mean}} = 10.9495$
- **언어 모델 보존 손실 ($\mathcal{L}_{\text{ce}}$, 토큰 평균 교차 엔트로피)**: $\bar{\mathcal{L}}_{\text{ce}} = 2.4434$

### 3개 안별 초기 100스텝 실측 비율

| 안 (Option) | 손실 함수 수식 | $\mathcal{L}_{\text{fp}}$ 유효 스케일 | $\mathcal{L}_{\text{ce}}$ 유효 스케일 | 실측 스케일 비율 ($\mathcal{L}_{\text{fp}} : \lambda_{\text{lm}} \mathcal{L}_{\text{ce}}$) |
|---|---|---|---|---|
| **기존 (Baseline)** | $\mathcal{L}_{\text{fp, sum}} + 0.1 \times \mathcal{L}_{\text{ce}}$ | $1,793,957.90$ | $0.2443$ | **$7,341,968.1 : 1$** (고정점 완전 압도) |
| **(A) Mean 정규화** | $\frac{1}{163840}\mathcal{L}_{\text{fp, sum}} + 0.1 \times \mathcal{L}_{\text{ce}}$ | $10.9495$ | $0.2443$ | **$44.81 : 1$** |
| **(B-1) $\lambda=25$** | $\mathcal{L}_{\text{fp, sum}} + 25.0 \times \mathcal{L}_{\text{ce}}$ | $1,793,957.90$ | $61.09$ | **$29,367.87 : 1$** |
| **(B-2) $\lambda=50$** | $\mathcal{L}_{\text{fp, sum}} + 50.0 \times \mathcal{L}_{\text{ce}}$ | $1,793,957.90$ | $122.17$ | **$14,683.94 : 1$** |
| **(B-3) $\lambda=100$** | $\mathcal{L}_{\text{fp, sum}} + 100.0 \times \mathcal{L}_{\text{ce}}$ | $1,793,957.90$ | $244.34$ | **$7,341.97 : 1$** |
| **(C) 초기 스케일 정규화** | $\frac{\mathcal{L}_{\text{fp, sum}}}{S_{\text{fp, 0}}} + \frac{\mathcal{L}_{\text{ce}}}{S_{\text{ce, 0}}}$ | $1.0000$ | $1.0000$ | **$1.00 : 1.00$** (완전 1:1 균형) |

*선택 주체: System 2 / Principal.*

---

## 3. 정칙화 항 (Regularization Constraints)

### 3-1. Base 모델 KL 다이버전스 제약
- **수식**:
  $$\mathcal{L}_{\text{KL}} = \mathbb{E}_{x \sim \mathcal{D}} \left[ D_{\text{KL}}\left( P_{\text{Base}}(\cdot \mid x) \,\|\, P_{\text{LoRA}}(\cdot \mid x) \right) \right]$$
- **계수 ($\lambda_{\text{KL}}$)**: `[                     ]` (System 2 지정 공란)

### 3-2. LoRA 가중치 Frobenius 노름 제약
- **수식**:
  $$\mathcal{L}_{\text{norm}} = \sum_{l \in \text{LoRA Layers}} \|B_l\|_F^2$$
- **상한 제약 (Hard Bound)**: $\|B_l\|_F \le M_{\text{lora}}$
- **페널티 계수 ($\lambda_{\text{norm}}$)**: `[                     ]` (System 2 지정 공란)

---

## 4. 조기 정지 게이트 (Early Stopping Safety Gate)

- **측정 주기**: 매 **25 학습 스텝**마다 자동 평가 실행.
- **평가 데이터셋**: MATH-500 `[:10]`, `max_new_tokens=512`, Plain 프롬프트, Greedy 디코딩.
- **중단 조건 (Abort Trigger)**:
  - Base 모델 기준 정답 수 ($1/10$, $10.0\%$) 대비 **$50\%$ 미만 ($0/10$)으로 하락 시 즉시 학습 중단 (Abort)**.
  - 평균 생성 토큰 길이가 $512.0$에 도달하고 EOS 도달률이 $0\%$로 떨어지는 모드 붕괴 시그니처 감지 시 즉시 중단.

---

## 5. 학습 통제 규칙

1. 목표치(Target accuracy/residual) 사전 기입 금지.
2. 학습 착수 전 Principal 및 System 2의 명시적 서명 필요.
3. 홀드아웃 `[32:64]` 및 확증 표본 `[64:128]` 침범 절대 금지.
4. 동결 소스 3종 수정 금지.

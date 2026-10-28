# CONFIG_CANON_v1: 평가 및 수렴 측정 표준 구성 정의서

- **발행일**: 2026-08-26
- **문서 버전**: v1.0
- **목적**: "Base"와 "LoRA" 구성의 모호성을 제거하고 모든 후속 측정의 정규 구성을 확정 및 동결한다.

---

## 1. 표준 5대 구성 정의

| 구성 ID | ckpt 로드 여부 | `apply_paper_lora` 호출 여부 | 절대 `scaling` 값 | `routing_proj` / `w_proj` / `_blend` 출처 | `torch.manual_seed` | 비고 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`B0`** | **미로드** (None) | **미적용** (False) | N/A (0.0) | Base 초기 가중치 (unadapted) | `20260824` | Raw Pretrained Base 모델 (순수 베이스라인) |
| **`B1`** | **미로드** (None) | **미적용** (False) | N/A (0.0) | Base 초기 가중치 (unadapted) | `20260824` | `B0`의 정규 별칭 (기존 코드·보고서 호환용) |
| **`B2`** | **로드** (`m1_final_frozen.pt`) | **적용** (True) | `0.0` (가중치 기여 0) | Checkpoint 로드 가중치 (`m1_final_frozen.pt`) | `20260824` | 배선 및 학습된 라우터 유지 + LoRA 델타 0화 |
| **`F1`** | **로드** (`m1_final_frozen.pt`) | **적용** (True) | `2.0` ($\alpha=16, r=8$, 기본값) | Checkpoint 로드 가중치 (`m1_final_frozen.pt`) | `20260824` | 동결본 공칭 작동점 (Nominal Frozen LoRA) |
| **`Fx`** | **로드** (`m1_final_frozen.pt`) | **적용** (True) | `x` (절대 스케일 실수) | Checkpoint 로드 가중치 (`m1_final_frozen.pt`) | `20260824` | 스윕용 구성 ($x$는 절대 스케일 값이며 배율이 아님) |

> [!IMPORTANT]
> - **절대 스케일 명기 원칙**: `Fx`의 $x$는 LoRA 모듈의 `m.scaling`에 직접 할당되는 **절대값(Absolute Value)**입니다.
> - 공칭 작동점 `F1`은 $x=2.0$ ($= 16 / 8$)이며, 배율 $0.5\times$는 $x=1.0$, 배율 $0.25\times$는 $x=0.5$, 배율 $0.125\times$는 $x=0.25$에 해당합니다.

---

## 2. 기존 실험 매핑표 (Task 49 / 51 / 54)

| 기존 실험 ID | 표기 명칭 | 실제 적용 구성 | 실측 절대 `scaling` | 대응 정규 ID |
|---|---|---|---|:---:|
| **Task 49-2** | `step0_base` | Raw Base Model | N/A | **`B0` / `B1`** |
| **Task 49-2** | `step10000_frozen` | Frozen+LoRA ($\alpha=1.0\times$) | `2.0` | **`F1`** |
| **Task 49-3** | `lora_scale=0.00` | Frozen ckpt + $0.00\times$ 배율 | `0.0` | **`B2`** |
| **Task 49-3** | `lora_scale=0.10` | Frozen ckpt + $0.10\times$ 배율 | `0.20` | **`F0.20`** |
| **Task 49-3** | `lora_scale=0.25` | Frozen ckpt + $0.25\times$ 배율 | `0.50` | **`F0.50`** |
| **Task 49-3** | `lora_scale=0.50` | Frozen ckpt + $0.50\times$ 배율 | `1.00` | **`F1.00`** |
| **Task 49-3** | `lora_scale=0.75` | Frozen ckpt + $0.75\times$ 배율 | `1.50` | **`F1.50`** |
| **Task 49-3** | `lora_scale=1.00` | Frozen ckpt + $1.00\times$ 배율 | `2.00` | **`F1` (`F2.00`)** |
| **Task 50** | `Base Model` | Raw Base Model | N/A | **`B0` / `B1`** |
| **Task 51** | `Base Model (B1)` | Raw Base Model | N/A | **`B0` / `B1`** |
| **Task 51** | `Frozen+LoRA` | Frozen Checkpoint | `2.0` | **`F1`** |
| **Task 54-1/3** | `alpha=0.00` | Frozen ckpt + `m.scaling=0.0` | `0.0` | **`B2`** |
| **Task 54-1/3** | `alpha=0.10` | Frozen ckpt + `m.scaling=0.10` | `0.10` | **`F0.10`** (Task 49 배율 기준 $0.05\times$) |
| **Task 54-1/3** | `alpha=0.25` | Frozen ckpt + `m.scaling=0.25` | `0.25` | **`F0.25`** (Task 49 배율 기준 $0.125\times$) |
| **Task 54-1/3** | `alpha=1.00` | Frozen ckpt + `m.scaling=1.00` | `1.00` | **`F1.00`** (Task 49 배율 기준 $0.50\times$) |

---

## 3. Task 49-3 $\alpha$ 주입 방식 및 붕괴 임계 좌표 확인

- **코드 실측 (`scratch/run_task49_sweep.py:19`)**:
  ```python
  m.scaling = 2.0 * float(lora_scale)
  ```
- **판정 결과**:
  - Task 49-3의 `lora_scale`은 공칭 스케일($2.0$)에 대한 **배율(Multiplier)**이었습니다.
  - 따라서 Task 49-3에서 보고된 붕괴 임계 $(0.25, 0.50]$은 배율 좌표이며, 이를 절대 스케일로 환산하면 **절대 scaling $(0.50, 1.00]$**에 해당합니다.
  - 절대 스케일 기준:
    - 절대 scaling $\le 0.50$ (배율 $\le 0.25$): 모드 붕괴 없음 (정상 CoT 생성)
    - 절대 scaling $\ge 1.00$ (배율 $\ge 0.50$): 모드 붕괴 발생 (EOS 조기 출력 및 축약)

# LEDGER_CORRECTION: 공식 정정 원장 (2026-08-26)

---

## 1. Task 42 좌표 정정 (Task 83-1)

- **대상 항목**: Task 42 (`results/task42_routing_fix_results.json`)
- **기존 기록**: Damped Picard ($c=0.92$) 지시 조건에 의한 $k_{100} = 142$ 도출
- **실측 정정**:
  - `run_task42_routing_fix_effect.py:58` 실측 코드 확인 결과, `z_free = phi_z`로 구현되어 **$c=1.00$ (무감쇠 순수 Picard)** 조건에서 실행되었음.
  - 따라서 $k_{100} = 142$ (재현치 141)는 $c=1.00$ 무감쇠 조건에서 잔차 $< 5 \times 10^{-3}$이 100스텝 지속된 시점의 실측값임.

---

## 2. Task 51 라벨 정정 (Task 83-2)

- **대상 항목**: Task 51/61/71 (`task51_limit_cycle_results.json`, `task61_sample08_comparison.json`)
- **기존 라벨**: `k_100 = 2687`
- **정정 라벨**: **`k_first(1e-4) = 2687`**
- **정정 사유**:
  - 해당 수치는 100스텝 지속 수렴(자 B안)이 아니라, 잔차가 $\text{tol} \le 1 \times 10^{-4}$에 도달한 최초 단일 스텝(`k_first(1e-4)`)임.
  - 본 궤적($c=0.92$)에 자 B안($100\text{연속 } < 5 \times 10^{-3}$)을 적용할 경우의 실제 도달 스텝은 **Step 136**임.
  - 따라서 자 B안과의 혼선을 방지하기 위해 라벨을 `k_first(1e-4)`로 명시 정정함.

---

## 3. `fallback_to_parent_on_failure` 배선 검증 (Task 83-3)

- **배선 위치**:
  - `cts/deq/transition.py` L118, L274~284 (`transition`)
  - `cts/deq/transition.py` L310, L433~438 (`transition_batch`)
- **동작 확인**:
  - 미수렴(`is_converged == False`) 시 `fallback_to_parent_on_failure` 플래그에 따라 `parent_z_star` 복귀 여부가 정상 분기됨.

#!/usr/bin/env python3
"""Update PREREG_S7_D_v3_4.md strictly from PREREG_S7_D_v3_2.md according to Ledger #378."""

with open("/home/shoon/cts_e1_s5/PREREG_S7_D_v3_2.md", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update header
old_hdr = """# PREREG_S7_D_v3_2.md — 문제 글 + z* 결합 디코더 (S7-D) 사전등록 v3.2 (원장 #368 — 잠금 정본)

- **등록 일시**: 2026-09-27
- **지침**: 원장 #363, #364, #365, #366, #367, #368 (S7-D v3.2 최종 잠금안)
- **발신**: System 2 / **수신**: Antigravity / **사본**: Principal, 외부 감사인
- **소속**: Stage 7 (S7-D: Text + z* Augmented Generation)
- **문서 잠금 상태**: **LOCKED (잠금 완료 — 본 문서 기준 디코더 학습 착수 승인)**
- **선행 등록 문서 보존 상태**:
  - `PREREG_S7_D.md` (v1, SHA: `26a1f0e32d5108a9773eb4e95e14742cd898603b16b7b05b761666a09ff5ec91`)
  - `PREREG_S7_D_v2.md` (v2, SHA: `cf6628d26db7e595cc31a898696762c2dccf0f8316a5e41b434f3c153c27bfad`)
  - `PREREG_S7_D_v3.md` (v3, SHA: `ef597b3244d4d52f8132b581154a9ffb21b9c2c11b3dda296ce2bfee897954ec`)
  - `PREREG_S7_D_v3_1.md` (v3.1, SHA: `e6f72d4c2a35af6f295ebc62cc620895e9b323ac5bcb0142ebf7e5762ac3af82`)"""

new_hdr = """# PREREG_S7_D_v3_4.md — 문제 글 + z* 결합 디코더 (S7-D) 사전등록 v3.4 (원장 #377, #378 — 척도 정정 및 잠금)

- **등록 일시**: 2026-09-28
- **지침**: 원장 #363, #364, #365, #366, #367, #368, #375, #377, #378 (S7-D v3.4 최종 잠금안)
- **발신**: System 2 / **수신**: Antigravity / **사본**: Principal, 외부 감사인
- **소속**: Stage 7 (S7-D: Text + z* Augmented Generation)
- **문서 잠금 상태**: **LOCKED (잠금 완료 — 본 문서 기준 척도 정정 반영 및 디코더 학습 착수 승인)**
- **선행 등록 문서 보존 상태**:
  - `PREREG_S7_D.md` (v1, SHA: `26a1f0e32d5108a9773eb4e95e14742cd898603b16b7b05b761666a09ff5ec91`)
  - `PREREG_S7_D_v2.md` (v2, SHA: `cf6628d26db7e595cc31a898696762c2dccf0f8316a5e41b434f3c153c27bfad`)
  - `PREREG_S7_D_v3.md` (v3, SHA: `ef597b3244d4d52f8132b581154a9ffb21b9c2c11b3dda296ce2bfee897954ec`)
  - `PREREG_S7_D_v3_1.md` (v3.1, SHA: `e6f72d4c2a35af6f295ebc62cc620895e9b323ac5bcb0142ebf7e5762ac3af82`)
  - `PREREG_S7_D_v3_2.md` (v3.2, SHA: `560c7fea987f9c065ed9339180f635eb70d49d8e654e3541a8bc011317e53f6c`)
  - `PREREG_S7_D_v3_3.md` (v3.3, SHA: `35881fc584ea81a49fcc8d032f52d251e4e01657d250606e1c270628cd357c17`, **무효 — §3 입력 양식 회귀 오류로 잠금 불인정**)"""

assert old_hdr in text, "Header mismatch"
text = text.replace(old_hdr, new_hdr, 1)

# 2. Insert Section 5-1 after Section 5
old_sec5_end = """- Lenient 수치 정답 추출 정확도에서도 동일 방향성 확인.

---"""

new_sec5_end = """- Lenient 수치 정답 추출 정확도에서도 동일 방향성 확인.

---

## 5-1. DEQ 고정점 수렴 게이트 척도 정정 (원장 #375·#377·#378)

### (a) 기존 척도의 물리적 한계 및 정정 배경
- 기존 $\\rho = \\frac{\\|z_{k+1} - z_k\\|}{\\|z_k\\|}$는 감쇠 보폭 $\\alpha$가 곱해진 **1보 이동량(Step Displacement)**에 불과함:
  $$\\rho = \\alpha \\cdot \\frac{\\|f(z_k) - z_k\\|}{\\|z_k\\|} = \\alpha \\cdot r$$
- 따라서 $\\alpha$가 $0.5 \\to 0.25$ 등으로 변경되면 $\\rho$의 수치가 절반으로 축소되어 참 고정점 오차를 과소평가하게 됨.
- 이에 따라 DEQ 수렴 게이트(①-a, ①-b) 및 수렴 구간 표의 공식 척도를 **$\\alpha$ 무관 참 고정점 잔차(True Fixed-Point Residual)**:
  $$r = \\frac{\\|f(z) - z\\|}{\\|z\\|} = \\frac{\\rho}{\\alpha}$$
  로 전면 정정한다.

### (b) 이중 수렴 기준 (Dual Convergence Criteria)
1. **엄격 기준 (Strict DEQ Convergence)**:
   $$r < \\mathbf{5.0 \\times 10^{-3}}$$
   *(DEQ 원 논문 Bai et al. 2019 및 엄밀 고정점 수렴 기준)*
2. **S5 동치 기준 (S5 Equivalent Tolerance)**:
   $$r < \\mathbf{1.0 \\times 10^{-2}}$$
   *(S5 시절 $\\alpha=0.5$ 감쇠 이동량 $\\rho < 5.0 \\times 10^{-3} \\iff r = \\frac{\\rho}{0.5} < 1.0 \\times 10^{-2}$와 물리적으로 동일한 허용 오차)*

### (c) 수렴 구간 표 정의 (4구간)
| 구간 명칭 | 참 잔차 $r$ 범위 | 물리적 상태 및 의미 |
| :--- | :--- | :--- |
| **초고정밀 수렴** | $r < 1.0 \\times 10^{-3}$ | 초고정밀 고정점 도달 |
| **엄격 수렴** | $1.0 \\times 10^{-3} \\le r < 5.0 \\times 10^{-3}$ | 원 논문 기준 고정점 엄격 도달 |
| **S5 동치 완보 수렴** | $5.0 \\times 10^{-3} \\le r < 1.0 \\times 10^{-2}$ | S5 기존 $\\rho < 5\\times 10^{-3}$ 동치 수렴 |
| **미수렴 (Non-converged)** | $r \\ge 1.0 \\times 10^{-2}$ | 고정점 미도달 (미수렴군 잔류) |

### (d) DEQ 수렴 게이트 ①-a / ①-b 정정 기준
- **①-a (Train 수렴 비율)**: Train 1,000문항 중 $r < 1.0 \\times 10^{-2}$ (S5 동치) 비율 $\\ge 95\\%$, $r < 5.0 \\times 10^{-3}$ (엄격) 비율 병기 보고.
- **①-b (Held-out 수렴 비율)**: Held-out 600문항 중 $r < 1.0 \\times 10^{-2}$ (S5 동치) 비율 $\\ge 90\\%$, $r < 5.0 \\times 10^{-3}$ (엄격) 비율 병기 보고.

### (e) S7 기존 400문항 완료 보고 규약
- 현재 $\\alpha=0.25$로 GPU 0에서 산출되는 $\\rho_{1000}$에 대해, $r = \\frac{\\rho_{1000}}{0.25} = 4 \\times \\rho_{1000}$으로 즉시 환산하여 이중 기준(엄격 $5\\times 10^{-3}$, S5 동치 $1\\times 10^{-2}$) 및 4구간 표로 보고한다.

### (f) S6 중간 보고 원장 정정 기록 명시 (원장 #377 지침 2)
- S6 중간 보고의 "43/54 해소"는 감쇠 보폭 $\\rho < 5\\times 10^{-3}$ 기준이었으며, 참 잔차 $r$ 기준으로 정정한 수치는 다음과 같다:
  - $\\alpha=0.25$: $r < 1.0 \\times 10^{-2}$ (S5 동치) **19/54**, $r < 5.0 \\times 10^{-3}$ (엄격) **5/54**.
  - $\\alpha=0.35$: $r < 1.0 \\times 10^{-2}$ (S5 동치) **17/54**, $r < 5.0 \\times 10^{-3}$ (엄격) **4/54**.
  - **결론**: 감쇠 채택 없음(감쇠 단독으로 미수렴 해소 불가) 판정 불변 유지.

---"""

assert old_sec5_end in text, "Sec 5 mismatch"
text = text.replace(old_sec5_end, new_sec5_end, 1)

# 3. Update Section 8(a)
old_sec8a = """### (a) z* 계산 총량 (실측치 52~70초/문항 근거)
- **기준 실측치**: 현재 가동 중인 S7 GPU 0 실측치 = **문항당 약 52~60초** (Picard 1,000 NFE 고정, $tol=0.0$).
- **총 계산 대상**: 1,600문항 (기존 400문항 재계산 + 신규 1,200문항).

| 단계 (Phase) | 대상 문항 수 | 문항당 소요 시간 | GPU 0 단독 시간 | 진행 상태 |
| :--- | :---: | :---: | :---: | :--- |
| **Step 1 (기존 400문항 재계산)** | 400문항 | 55초 (실측) | **6.11시간** (367분) | **실행 중 (GPU 0 tmux `s7_zstar_gpu0`)** |
| **Step 2 (신규 train 680문항)** | 680문항 | 55초 (실측) | **10.39시간** (623분) | Step 1 직후 자동 연쇄 실행 |
| **Step 3 (신규 heldout 520문항)** | 520문항 | 55초 (실측) | **7.94시간** (477분) | Step 2 직후 자동 연쇄 실행 |
| **전체 합계 (Total)** | **1,600문항** | 55초 | **24.44시간** | **원격 파이프라인 자동화 완료** |"""

new_sec8a = """### (a) z* 계산 총량 (실측치 71.2초/문항 근거)
- **기준 실측치**: 현재 가동 중인 S7 GPU 0 실측치 = **문항당 평균 71.2초** (중앙값 58.2초, Picard 1,000 NFE 고정, $tol=0.0$).
- **총 계산 대상**: 1,600문항 (기존 400문항 재계산 + 신규 1,200문항).

| 단계 (Phase) | 대상 문항 수 | 문항당 소요 시간 | GPU 0 단독 시간 | 진행 상태 |
| :--- | :---: | :---: | :---: | :--- |
| **Step 1 (기존 400문항 재계산)** | 400문항 | 71.2초 (실측 평균) | **7.91시간** (475분) | **실행 중 (GPU 0 tmux `s7_zstar_gpu0`)** |
| **Step 2 (신규 train 680문항)** | 680문항 | 71.2초 (실측 평균) | **13.45시간** (807분) | Step 1 직후 자동 연쇄 실행 |
| **Step 3 (신규 heldout 520문항)** | 520문항 | 71.2초 (실측 평균) | **10.28시간** (617분) | Step 2 직후 자동 연쇄 실행 |
| **전체 합계 (Total)** | **1,600문항** | 71.2초 (평균) | **31.64시간** | **원격 파이프라인 자동화 완료** |"""

assert old_sec8a in text, "Sec 8a mismatch"
text = text.replace(old_sec8a, new_sec8a, 1)

# 4. Update Section 10
old_sec10 = """## 10. 사전등록 잠금 (Document Locked)

본 문서(`PREREG_S7_D_v3_2.md`)는 원장 #363부터 #368에 이르는 모든 외부 감사 의견, 파라미터 실측치, 입력 템플릿 검증 배열, 데이터 분할 서로소 증명, 게이트 컷오프 수치, 평가 생성 예산이 완비되었으므로 **최종 잠금(LOCKED)** 처리한다. 사후 수정은 절대 불가하다."""

new_sec10 = """## 10. 사전등록 잠금 (Document Locked)

본 문서(`PREREG_S7_D_v3_4.md`)는 원장 #363부터 #368, #375, #377, #378에 이르는 모든 외부 감사 의견, DEQ 참 잔차 척도 정정($r = \\rho/\\alpha$), 파라미터 실측치, 입력 템플릿 검증 배열, 데이터 분할 서로소 증명, 게이트 컷오프 수치, 평가 생성 예산이 완비되었으므로 **최종 잠금(LOCKED)** 처리한다. 사후 수정은 절대 불가하다."""

assert old_sec10 in text, "Sec 10 mismatch"
text = text.replace(old_sec10, new_sec10, 1)

with open("/home/shoon/cts_e1_s5/PREREG_S7_D_v3_4.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully written PREREG_S7_D_v3_4.md")

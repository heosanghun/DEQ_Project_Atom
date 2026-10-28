# PREREG_S7_DEQ_v2.1 — 부드러운 top-3 스위치(개발 단계)

- **상태**: 잠금 (2026-10-06 — v2.1, AUDIT-P009 재감사 2 PASS 후, 본 실험 착수 전. §5 착수 전 점검의 점검용 ε만 개정). 결과를 본 뒤 ε·기준·문항·지표를 바꾸지 않는다.
- **v2 대비 변경 (v2.1)**: §5만 바꾼다. 나머지 절은 v2(SHA-256 `d9eef10180d228bb676f5115d97fc721f3acf67d4f94864f8ef33b018a5a72aa`)와 같다. 사유는 §5 첫 문단에 공개한다.
- **발행**: 회사 PC Claude Code(CC), 2026-10-06
- **근거**: CTS-428(Principal 2026-10-06: 연구실 GPU 1 사용 허락, Task 2·3), PREREG_S7_DEQ_v1.3 §(소프트 라우팅은 v2로 미룸, 124행), GRAPH H-DEQ-2, P3 결과(`shared/from_laptop/s7deq_intervene_P3.json`, SHA-256 `626591dcc8050a39d98f53900d40e2a238155979aa1321690537a34ac86b983c`).
- **성격**: 개발 단계 실험이다. 이 결과로 G1 판정을 하지 않는다. "채택 후보"가 나오면 확증은 별도 사전등록과 새 문항으로 한다.

## 1. 목적

정본 전이의 하드 top-3 스위치는 3등과 4등 모듈의 라우팅 점수가 거의 같을 때 걸음마다 선택이 뒤바뀌어(경계 왕복) 반복이 수렴하지 않는 원인으로 지목됐다(P3, 원장 #405). 3등·4등 경계 근처에서만 두 부품을 연속적으로 섞는 "매끄러운 사상"이 진짜 고정점을 주는지 확인한다.

## 2. 설계 (고정)

### 2.1 기호
- 정본 라우팅 점수: $a = \mathrm{softmax}(W_g\,\overline{z})$ (정본 `routing_weights(z_cur.float(), w_g, 1.0)`, `cts/routing/sparse_moe_ref.py:9-23`). 모듈 수를 $M$이라 한다.
- 순위: 정본 `torch.topk(a, 4)`의 순서로 $i_1,i_2,i_3,i_4$를 정한다(동점 처리는 torch.topk 그대로).
- 경계 간격: $d = a_{i_3} - a_{i_4} \ge 0$.
- 지지 집합: $S_3=\{i_1,i_2,i_3\}$, $S_4=\{i_1,i_2,i_4\}$.
- 고정 지지 사상: $\Phi_S(z) = \mathrm{norm}\big(\sum_{i\in S}\hat c^{S}_i\,h_i(z)\big)$, $\hat c^{S}_i = a_i/\sum_{j\in S}a_j$. $h_i$는 모듈 $i$의 출력이고 `norm`은 정본 `lm.norm`이다. 이것은 정본 fixed-support 경로(`cts/deq/gemma_latent_forward.py` `parallel_sparse_module_forward_batch`, `fixed_module_ids` 분기)와 같다. 정본 전이의 $\Phi$는 $\Phi_{S_3}$이다.

### 2.2 부드러운 사상
$$
\Phi^{\varepsilon}_{\text{soft}}(z)=
\begin{cases}
\Phi_{S_3}(z) & d>\varepsilon\\[2pt]
w\,\Phi_{S_3}(z)+(1-w)\,\Phi_{S_4}(z),\quad w=\tfrac12+\tfrac{d}{2\varepsilon} & d\le\varepsilon
\end{cases}
$$
- $d=\varepsilon$에서 $w=1$이고, $d=0$에서 $w=\tfrac12$이다. 그래서 3등·4등이 자리를 바꿔도 사상이 끊기지 않는다.
- $d>\varepsilon$ 구간은 **정본 함수를 한 번 호출하는 코드 경로**로 계산한다($1\cdot\Phi+0\cdot\Phi$ 형태 금지). 그래야 경계 밖에서 정본과 비트 단위로 같다.
- $d\le\varepsilon$ 구간의 구현 방법은 둘 중 하나로 한다. 어느 쪽이든 정의는 위 식이다.
  - (가) 정본 fixed-support 경로를 $S_3$와 $S_4$로 두 번 호출한다(모듈 $i_1,i_2$ 중복 계산). `prepare_deq_context`도 각 지지 집합의 r_ids로 따로 호출한다.
  - (나) 모듈 4개 출력 $h_{i_1..i_4}$를 한 번 계산하고, 두 가중합과 두 `norm`을 만든 뒤 섞는다.
- $w$, $d$, 혼합은 float32로 계산한다. 혼합 결과의 dtype은 정본 $\Phi$ 출력과 같게 한다.

### 2.3 전이·풀이기 (정본 그대로, 스위치 함수만 교체)
- 전이: $T_\varepsilon(z)=(1-\alpha)z+\alpha\,(\Phi^{\varepsilon}_{\text{soft}}(z)+g_0\hat c_0)$. 정본 `InjectedTransitionStep`과 같고 $\Phi$만 바꾼다.
- 매개변수: α 0.25, g₀ 0.25, σ 0.020, seed 2030, K 64, max_nfe 1000, tol 0.0, config `configs/zstar_canonical.yaml`. CLI의 α 0.25가 YAML의 α 0.5보다 우선한다(`zstar_canonical_lowest.py` 56행 `config.update(kwargs)`).
- 풀이기: 정본 실행기 `scripts/s7_zstar_lowest_run.py`(SHA-256 `779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba`)와 `cts/deq/zstar_canonical_lowest.py`(서버, SHA 앞 8자리 `9f73fb27`)를 **새 파일로 복사**해서 쓴다. 바꾸는 것은 전이의 $\Phi$뿐이다. 걸음 수, 감쇠, 초기화($z_0=z_{\text{src}}+\sigma\epsilon$), 최저점 저장(z_lowest = 궤적 $k\in[0,1000]$에서 잔차 최소, 동률이면 작은 k), 저장 주기, 출력 필드는 같다.
- 잔차(부드러운 사상 기준): $r_\varepsilon(z)=\|T_\varepsilon(z)-z\|/(\alpha\|z\|)$. float32 텐서를 float64로 올려 노름을 계산한다(v1.3 (b)와 같은 정의, 사상만 다름).
- 정본 파일(cts_e1, 정본 함수)은 수정하지 않는다(R2-2). 새 파일 이름: `scripts/s7deq_soft_v2_run.py`, `exe_scratch/s7/soft_v2/zstar_soft_lowest.py`.

### 2.4 ε
- ε ∈ {0.005, 0.01, 0.02}. 단위는 라우팅 softmax 확률이다. 이 순서로 차례로 실행한다.

## 3. 문항

- **P3와 같은 74문항**: `configs/s6_items_74.json`(SHA-256 `42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3`, PREREG_S6_damping.md 27행). 실행 전에 서버에서 SHA를 확인한다. 다르면 실행하지 않는다.
  - 미수렴 54: PREREG_S6_damping.md 19행 목록.
  - 대조 20: PREREG_S6_damping.md 21행 목록.
- 사용 금지: held-out 520·80(봉인), 학습 1,000. 앵커 점검(§5)은 학습 세트의 `data/anchor20.json` 문항만 쓴다.

## 4. 지표 (ε마다, 74문항 전부 계산)

| 번호 | 지표 | 정의 |
|---|---|---|
| ① | 미수렴 54의 고정점 도달 | $r^*_\varepsilon=\min_{k\in[0,1000]} r_\varepsilon(z_k)$. $r^*<10^{-3}$ 개수/54, $r^*<5\times10^{-3}$ 개수/54 |
| ② | 대조 20의 위치 | $\cos(z^*_{\varepsilon}, z^{\text{ref}})$, $z^*_\varepsilon$ = z_lowest. 기준 $z^{\text{ref}}$ = `results_s6/zstar_s6_a0.25.pt`의 대조 20(SHA-256 `8c1d9a38eddda6b6fd7013c5bc5c0c3c85e9c2f6c44f8d12267e5169caf72026`, P3 기준선과 같은 기준, PREREG_S7_DEQ_v1.2 94행). 최솟값, ≥0.9990 개수, <0.9950 개수. 벡터는 [64×2560]을 펼쳐 float64로 계산 |
| ③ | 부드러운 사상의 최종 잔차 | $r_\varepsilon(z_{1000})$를 새로 평가. 74문항 중앙값·최댓값, $<10^{-3}$ 개수 |
| ④ | 경계 구간 걸음 비율 | 걸음 $k=1..1000$ 중 $d\le\varepsilon$인 비율($d$는 걸음 $k$의 $\Phi$를 계산할 때, 즉 $z_{k-1}$에서 잰 값). 문항별 값과 54·20 그룹별 평균·중앙값·최댓값, 마지막 100걸음 비율도 따로 |
| ⑤ (보고만) | 원래 기계에서의 잔차 | 하드 정본 전이 $T$로 $r(z^*_\varepsilon)=\|T(z^*_\varepsilon)-z^*_\varepsilon\|/(\alpha\|z^*_\varepsilon\|)$. 54의 $<10^{-3}$·$<5\times10^{-3}$ 개수, 그리고 $z^*_\varepsilon$에서 $d\le\varepsilon$인 문항 수(고정점이 혼합 구간 안에 있는지) |

- P3 기준선 열(같은 표에 나란히): ① 52/54, 53/54(`s7deq_intervene_P3.json` 12~28행), ② min cos 0.9326308762296536(같은 파일 31행), ≥0.9990 10개, <0.9950 8개(REPORT-CTS-P3.v3 116~137행).
- 주의(표에 각주로 적는다): P3의 ①은 마스크 동결 사상의 잔차였고, v2의 ①은 부드러운 사상의 잔차다. 사상이 서로 다르다.

## 5. 착수 전 점검 (코드 동일성) — v2.1 개정

**v2.1 개정 사유 (공개)**
- v2 §5는 코드 동일성을 ε=0.02로 점검하게 했다. 점검 중 ε=0.02에서 경계 걸음이 0인 앵커는 20개 중 0개였고(앵커별 경계 걸음 20~978, `raw_s7/s7deq_soft_v2.log`; `scratch/task0050_out.txt` 98~100행 `Zero boundary anchors found (0/3): []`, exit 2; 앵커별 값 42~96행), §5(4)대로 실행은 시작되지 않았다(로그 `raw_s7/s7deq_soft_v2.log`, REPORT-0049·0050).
- 이 점검은 결과 지표가 아니라 도구 점검이다. 목적은 "경계 밖에서 새 코드가 원래 기계와 같다"를 보이는 것이다(CTS-428 Task 3 문언, `shared/inbox/CTS-428.md` 24행).
- 그래서 v2.1은 이 목적을 직접 시험하도록 점검용 ε만 바꾼다. 본 실험의 ε {0.005, 0.01, 0.02}, 문항 74, 지표 ①~⑤, 기준(§7)은 바꾸지 않는다. 본 실험은 아직 한 문항도 계산하지 않았다.
- v2 점검에서 나온 ε=0.02 앵커 결과(`results_s7/anchor_soft_eps0.02_check.pt`)는 학습 문항 결과이며, 진단으로 그대로 보고한다.
- 공개(작성 시점에 이미 본 것): v2.1을 쓸 때 CC는 ε=0.02 앵커 결과를 이미 보았다. 경계 안에서 끝난 앵커의 원래 기계 잔차(r_hard_lowest)는 5.77e-3~8.77e-3이었다(`scratch/task0049_out.txt` 30행 Q#20 6.17e-3, 52행 Q#24 5.77e-3, 118행 Q#4 7.53e-3, 140행 Q#8 6.98e-3, 206행 Q#28 8.77e-3). 점검용 ε를 바꾸는 생각은 점검이 끝나기 전에 AUDIT-0048 16행에 적었다.
- 본 실험의 1000걸음 실행은 0이다. 다만 74문항 중 2문항(q0031, q0127)을 ε=0.01, 5걸음 CPU 시험으로 돌린 적이 있다(REPORT-0047 206~221행). 이 값은 어떤 지표에도 쓰지 않는다.
- (6)의 진단 표는 지표 ①~⑤와 §7 판정에 쓰지 않는다.
- Q#20의 ε=0.02 경계 걸음은 852다(`scratch/task0049_out.txt` 30행). REPORT-0050 193행의 "38"은 오기다(정오표 TASK-0051).
- v2 §5(1)의 "같은 세션에서 정본 앵커를 계산" 조건은 v2.1에서 빠진다. 대신 v2 시도 때 같은 GPU 1에서 만든 기준 파일을 SHA로 고정해 다시 쓴다(아래 (1)).

**점검 방법 (v2.1)**
- 장소: GPU 1(물리 UUID `GPU-9410691f-6857-1642-ec7c-7596970e2f48`).
- (1) 기준(잠금 시 고정): v2 점검에서 같은 GPU 1로 정본 실행기가 계산한 `results_s7/anchor_ref_gpu1_check.pt`. SHA-256 `5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6`, 26,410,991 B, 수정 시각 2026-10-06 20:17:04, 키 20개(`q0000_a0.25`, `q0004_a0.25`, … `q0076_a0.25`, anchor20 전부; `scratch/task0051_out.txt` 8~9·34~35행).
  - 이 파일은 읽기만 한다. 정본 실행기를 이 파일로 다시 부르지 않는다(정본 실행기는 문항을 건너뛸 때도 파일을 다시 저장한다, `s7_zstar_lowest_run.py` 151행). 추가 앵커가 필요하면 정본 실행기 출력은 새 파일 `results_s7/anchor_ref_gpu1_check_v21_extra.pt`로 한다.
  - 실행기는 이 SHA를 **상수로 고정**하고(환경 변수로 덮을 수 없음), 점검 전과 점검 뒤·본 실행 전에 파일 SHA가 이 상수와 같은지 확인한다. 다르면 exit 4. 추가 기준 파일 `_v21_extra.pt`가 점검 전에 이미 있으면 exit 3.
  - 점검·본 실험·분석의 모든 파이썬 호출은 `/home/shoon/system1_5/.venv/bin/python` 하나로 고정하고, 시작할 때 해석기 경로와 torch 버전을 로그에 남긴다. 74문항 config SHA와 작업 경로도 정규 경로에서는 상수다.
  - v2 시도와 같은 도구임을 SHA로 고정한다: `scripts/s7_zstar_lowest_run.py` `779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba`, `cts/deq/zstar_canonical_lowest.py` `9f73fb272353e8a307b9e628c65d1f17eaff164e2138fa3d232917958d6a6b2f`, `exe_scratch/s7/soft_v2/zstar_soft_lowest.py` `d1942c3cf5defbad6c148e69102b1ca6e8c64d1da5dd731434782364fd689f07`, `scripts/s7deq_soft_v2_run_v2.py` `2687a54e28b73c15e2980ad204724fb4f8903fb7d3b8f3d33d4e3ae11283db17`, `data/anchor20.json` `4b8d0942aa36a9f7d09ce8f228ac23ae9c26b0ea2d5264639575667145205af3`, `configs/zstar_canonical.yaml` `bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157`, 백본 스냅샷 `58c4d7b5…` 의 `model.safetensors` `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`·`config.json` `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`(`scratch/task0051_out.txt` 12~28행).
  - 실행기: `scripts/s7deq_soft_v2_launch_v5.sh` SHA-256 `7771f43f7f5e5d4e632923b5a70d080eb3c49aaecd75650a3fa954923cc4352c`(CC 감사 AUDIT-0053; v3·v4는 AUDIT-P009 재감사 1과 AUDIT-0052에서 결함이 나와 쓰지 않는다). 정규 실행은 `TEST_MODE` 없이 한다.
- (2) 새 실행기를 점검용 ε = 1e-12로 Q#20·Q#24·Q#32에 돌린다. 출력은 새 파일 `results_s7/anchor_soft_eps1e-12_check.pt`이다. 이 파일이 점검 전에 이미 있으면 exit 3. 각 키의 `eps` 필드는 1e-12여야 한다(실행기의 키 이름에는 ε가 없고 있는 키는 건너뛰므로). 이 ε에서는 3·4등 점수가 사실상 정확히 같을 때만 혼합 경로에 들어가므로, 거의 모든 걸음이 정본 경로를 탄다. 근거: 라우팅 점수는 float32이고, 서로 다른 두 float32 값의 차는 3등 점수가 약 1.5e-5 이상이면 1e-12보다 크다(1 ulp 이상). 모듈 수가 19개 이상이라 3등 점수는 이보다 훨씬 크므로, 혼합 경로는 정확한 동률(d=0)일 때만 열린다(AUDIT-P009). 마지막 평가 1회(`zstar_soft_lowest.py` 381행 부근)는 경계 걸음 수에 들어가지 않는다. 거기서 동률이 나면 실행기는 기록만 하고, 그 문항의 max|Δ|가 0이 아닐 때만 점검 전체가 멈춘다(exit 1).
- (3) 문항별 경계 걸음 수를 적는다. 경계 걸음이 0인 문항은 z_star·z_lowest·rho_traj가 (1)과 **max|Δ|=0**이어야 한다. 하나라도 0이 아니면 실행하지 않고 보고한다.
- (4) 경계 걸음 0인 문항이 3개 미만이면 `data/anchor20.json` 순서대로 다음 앵커를 (1)(2)로 더 돌려 3개를 채운다(최대 20문항). 3개를 못 채우면 실행하지 않고 보고한다.
- (5) (1)의 결과와 `results_s7/anchor20_gpu1.pt`(SHA-256 `84d3528efd77b91bf53264937550fe4dea19400be8e017089fc963fca13d815b`)의 같은 문항 비교(max|Δ|, cos)는 보고만 한다.
- (6) 진단(보고만): v2 점검의 ε=0.02 앵커 결과 `results_s7/anchor_soft_eps0.02_check.pt`(SHA-256 `7eb2cd4946faa89be9b7c92c55113fee1da24c08608ef393e06e7da9c605aefd`, 필드 `boundary_step_count`·`r_lowest`·`r_hard_lowest`·`in_boundary_lowest`·`d_traj`)에서 문항별 경계 걸음 수, 부드러운 최저 잔차, 원래 기계 잔차(r_hard_lowest), 최저점이 혼합 구간 안에 있는지를 표로 낸다.
- 로컬 CPU 단위 시험(GPU 사용 0, 모의 백본): (a) $d>\varepsilon$에서 정본과 비트 단위 동일, (b) $d=\varepsilon$에서 $w=1$, (c) $d=0$에서 $i_3,i_4$를 바꿔도 출력 동일, (d) $w$가 $d$에 대해 선형. (v2와 같음, 이미 통과: REPORT-0046·0047)
- (7) 보고만(AUDIT-P009 R1, REPORT-0051): ε=0.02 실행에서 앵커 20개 모두, 첫 경계 걸음 k₁ 직전까지의 `rho_traj`가 기준 파일과 비트 단위로 같았다(max|Δ|=0, k₁=2~14; `scratch/task0051_out.txt` 69~88행). 실제 실험 ε에서도 경계 밖 경로가 정본과 같다는 추가 근거로 적는다. 다만 비교한 접두 길이는 앵커마다 1~13걸음뿐이다.

## 6. 실행

- GPU 1에서만, tmux 세션 하나(`s7deq_soft_v2`)에서 ε 0.005 → 0.01 → 0.02 순서로 차례로 한다.
- `CUDA_VISIBLE_DEVICES`=GPU 1 UUID. GPU 0의 P6 진행과 프로세스·메모리를 나눈다. 첫 보고에 `nvidia-smi --query-compute-apps`로 GPU별 PID와 메모리를 적는다.
- 5문항마다 저장하고, 재개 시 저장된 문항을 건너뛴다(정본 실행기와 같은 방식).
- 산출물:
  - `results_s7/s7deq_soft_v2_eps0.005.pt`, `…_eps0.01.pt`, `…_eps0.02.pt`(z_star, z_lowest, rho_traj, d_traj, w_traj 요약)
  - `results_s7/s7deq_soft_v2_eps{0.005,0.01,0.02}.json`(지표 ①~⑤, P3 기준선 열 포함)
  - `raw_s7/s7deq_soft_v2.log`
- 분석 스크립트 `scripts/s7deq_soft_v2_analyze.py`는 결과를 보기 전에 작성하고 CC가 감사한다. 실행 시 입력 파일 SHA를 JSON에 기록한다.
- 서버가 재부팅되면 ESC-0006 규칙을 따른다(3차 재부팅이면 전부 멈춤).

## 7. 판정 규칙 (개발용)

- ε마다 ①②를 P3 기준선과 나란히 보고한다. ③④⑤도 함께 보고한다.
- **채택 후보 조건**: ① $r^*<5\times10^{-3}$ 비율 ≥ 95%(54 중 52 이상) **그리고** ② 대조 min cos ≥ 0.9990.
- ② min cos < 0.9950이면 그 ε은 "대조 위치 이동"으로 적는다.
- 후보가 여럿이면 가장 작은 ε을 확증 사전등록의 1순위로 적는다(원래 기계와 가장 가까움). 후보가 없으면 "후보 없음"으로 적는다.
- **이 결과로 G1 판정을 하지 않는다.** 후보가 나와도 확증은 별도 사전등록과 새 문항(이번 74·봉인 문항과 겹치지 않는 문항)으로 한다.

## 8. 금지

- GPU 0·1 외 사용. GPU 4 봉인.
- 결과를 본 뒤 ε·기준·문항·지표·잔차 정의를 바꾸는 것.
- P3 동결 z*와 부드러운 스위치 z*를 디코더 학습·평가에 쓰는 것.
- 직접 ssh(서버 접속은 `shared/cts_ssh.py`로만, 진행 확인은 30분에 최대 1회, 명령은 묶어서 1회).
- 보고서의 판정 단어. 숫자마다 포인터(로그:행 / JSON 필드 / SHA)를 단다.

# PREREG_S7_DEQ_v2 — 부드러운 top-3 스위치(개발 단계)

- **상태**: 잠금 (CC 자기 감사 AUDIT-P007 뒤 SHA 기록, STATE·LEDGER에 적음). 결과를 본 뒤 ε·기준·문항·지표를 바꾸지 않는다.
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

## 5. 착수 전 점검 (코드 동일성)

- 장소: GPU 1(물리 UUID `GPU-9410691f-6857-1642-ec7c-7596970e2f48`).
- (1) 같은 세션에서 정본 실행기(`s7_zstar_lowest_run.py`)로 앵커 Q#20, Q#24, Q#32를 GPU 1에서 계산한다 → 기준 파일.
- (2) 새 실행기를 ε=0.02(가장 넓은 값)로 같은 3문항에 돌리고, 문항별 경계 걸음 수(④의 개수)를 적는다.
- (3) 경계 걸음 수가 0인 문항은 z_star·z_lowest·rho_traj가 (1)과 **max|Δ|=0**이어야 한다. 하나라도 0이 아니면 실행하지 않고 보고한다.
- (4) 3문항 중 경계 걸음 수가 0인 문항이 3개 미만이면, `data/anchor20.json` 순서대로 다음 앵커를 (1)(2)로 더 돌려 경계 걸음 0인 문항을 3개 채운다(최대 20문항). 3개를 못 채우면 실행하지 않고 보고한다.
- (5) (1)의 결과와 `results_s7/anchor20_gpu1.pt`(SHA-256 `84d3528efd77b91bf53264937550fe4dea19400be8e017089fc963fca13d815b`)의 같은 문항 비교(max|Δ|, cos)는 보고만 한다.
- 로컬 CPU 단위 시험(GPU 사용 0, `cts/backbone/mock_tiny.py` 등 모의 백본): (a) $d>\varepsilon$에서 정본과 비트 단위 동일, (b) $d=\varepsilon$에서 $w=1$, (c) $d=0$에서 $i_3,i_4$를 바꿔도 출력 동일, (d) $w$가 $d$에 대해 선형. 이 네 가지가 맞아야 서버로 올린다.

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

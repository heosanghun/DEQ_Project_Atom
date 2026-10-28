# CODEBASE_MAP.md — CTS 코드베이스 지도 (1회 분석 고정본, R5-2)

- 기준: 로컬 `D:\AI\cts` 브랜치 `task-delivery/C1` 커밋 `1cb5d89` (2026-09-28). 행 번호는 이 커밋 기준.
- **로컬은 서버 `/home/shoon/cts_e1_s5`의 불완전 사본이다.** S5–S7 정본 코드(`harness_A_v1`, `cts.deq.transition_inject`, `cts.deq.zstar_canonical`)는 로컬에 없거나 `scratch/`에만 있다. 서버 정본에 대한 판정은 서버 코드로 한다(로컬 행 번호로 판정 금지).
- 갱신: 변경된 파일만 부분 갱신. 전수 재분석 금지.
- 로드맵: **P0** = 지금 도는 S7 실험의 결과를 틀리게 만들 수 있음 · **P1** = 저장소/루프 기반 · **P2** = 트리·논문 단계 백로그.

## 1. 데이터 흐름 (정본 CTS 경로)

```
문제 → prompt_contract.build_math_decode_prompt (cts/prompt_contract.py:14)
 → GemmaCTSBackbone.encode_context → GemmaEq3Context (cts/backbone/gemma_adapter.py:151, cts/deq/gemma_latent_forward.py:19,178)
 → root_bootstrap.solve_root_z_star → root_fwdpass (cts/deq/root_bootstrap.py:18, gemma_adapter.py:155)
 → cts_full_episode (cts/mcts/cts_episode.py:165)
     _select_leaf PUCT (:113) → MetaPolicy ν (:422, cts/policy/meta_policy.py)
     → transition_batch (cts/deq/transition.py:371) → Φ = backbone.deq_step (gemma_adapter.py:313)
         → routing_weights Top-3/19 (cts/routing/sparse_moe_ref.py:9) → parallel_sparse_module_forward (gemma_latent_forward.py:886)
         → broyden_fixed_point_batch (cts/deq/broyden_forward.py:433) | Picard (transition.py:464)
     → NeuroCritic (:561) → _backpropagate (:85) → ACT 정지 (:652)
 → critic argmax 노드 z* (:663) → decode_from_z_star W_proj 소프트 프롬프트 (gemma_adapter.py:380)
```

| 패키지 | 핵심 |
|---|---|
| `cts/deq` | dense 역-Broyden(n≤8192, broyden_forward.py:201), W-배치 LM 역-Broyden(:583), Picard(transition.py:192), hybrid(hybrid_solver.py:40), 잔차 `_relative_residual` = ‖z−Φ‖/max(‖z‖,‖Φ‖,1) (broyden_forward.py:163) |
| `cts/model` | 42층→19모듈 (module_partition.py), 로더(gemma_loader.py) |
| `cts/latent` | FAISS 조상 검색 (faiss_context.py:30, IVF-PQ 전환 :159) |
| `cts/train` | Stage 1 (stage1_openmath_train.py:277, 손실 stage1_warmup.py:345), Stage 2 PPO (stage2_ppo_train.py:146) |
| `cts/eval` | 디스패처 `scripts/run_cts_eval_full.py` (run_single_evaluation :96, _extract_pred :553, _match_answer :396), 채점 math500.py, MAC mac_accounting.py, 통계 statistics.py |
| `scripts/` | S5 `s5_*`, S6 `s6_*`, S7 `run_s7_zstar_pipeline.sh`, `s7_watch_and_pause.py`, `verify_frozen_parity_cpu.py` (대부분 `/home/shoon` 경로 하드코딩) |
| `harness/` | dag_runner.py — 8월 M1–M6 캠페인용, 2026-08-19 이후 정지(폐기 상태) |
| `tests/` | 740개. 비-slow 실행: 700 통과 / 39 실패 (2026-09-28) |

## 2. 결함 목록

| # | 결함 | 파일:행 | 로드맵 |
|---|---|---|---|
| 1 | HEAD가 커밋 안 된 모듈에 의존 → **1cb5d89로 해소**(단 S5–S7 스크립트는 여전히 서버 전용) | gemma_adapter.py:27-37 | P1 (해소) |
| 2 | `transition()`이 Φ 호출마다 Top-3 재선택 → 풀이 대상 사상 불연속. `routing_module_ids`/`eq3_static_inputs` 미전달 | transition.py:181-187, :592 | **P0** (서버 transition_inject.py로 확인) |
| 3 | 이력 잠재 무시: `with_history_latents`를 찾지만 실제 메서드는 `with_history` | transition.py:157, :421 / gemma_latent_forward.py:48 | P2 |
| 4 | Anderson: 저장소는 stub, `_anderson_broyden`은 최소제곱(ΔG)·보정(ΔG−ΔX)이 표준과 뒤바뀜, 배치판 NaN 가드 없음 | deq/solvers/anderson.py, broyden_forward.py:302,359-372,819 | **P0** (S7-DEQ는 `exe_scratch/s7/ref_solvers/anderson.py` 사용 — 단위 테스트 필수) |
| 5 | Triton 라우팅 커널 미실행(19≠2의 거듭제곱), 예외 무음 처리 | routing/sparse_moe_triton.py:39,82 / transition.py:56-59 | P2 |
| 6 | Picard 상수 고정(tol 5e-3, c 0.92, 1000걸음), `deq_step` 내부 0.92·γ와 이중 감쇠 | transition.py:192-270, gemma_adapter.py:345-347 | P2 |
| 7 | MAC 계약 파손: LUT v2에 `mac_per_module` 없음 → 고정값 1.2e9–1.8e9 사용, 그런데 `mac_headline_eligible: True`. Φ×2 과계산. `report_isoflop` KeyError | transition.py:65-77,252-259,293-314 / isoflop_matcher.py:67 | P2 (즉시 False로 정직화 권고) |
| 8 | Wilcoxon: 비0 쌍 <10이면 p=1.0, Table 2는 시드 5개 벡터만 전달 → 항상 n.s. | eval/statistics.py:102 / run_cts_eval_full.py:1711-1720 | P2 |
| 9 | **채점기 거짓 양성**: math-verify 미설치·미등록 → 첫 숫자 비교(`2\sqrt{3}`≡`2`), 답 표시 없으면 전문 fallback("Let me compute: 1."≡1), "3.5."→"3", "also"⊃"so", "1,250"→"250", 관사 a→A, `lenient_extract`의 `"\boxed{"` 비-raw 문자열 | math500.py:24,54,97,117 / run_cts_eval_full.py:588,601,607,610 / lenient_extract.py:20 | **P0** (S7-D 평가 전 단위 테스트 + Phase 2 재채점) |
| 10 | 비교군 계산량 불일치(broyden_max_iter 12/15/20/30, 디코드 64 vs 512–1024 토큰, 비교군 MAC 없음) | run_cts_eval_full.py:54-63,842-845,1471 | P2 |
| 11 | Stage 2 resume 항상 실패(옵티마이저 그룹 3개 vs 검사 2개) | stage2_ppo_train.py:320-326 vs :450-458 | P2 |
| 12 | Stage 2 W_proj 학습 후 미저장 | stage2_ppo_train.py:786-800, :1049-1116 | P2 |
| 13 | 계약 무시: Stage 1 계약 실패해도 `stage1_accuracy_contract_valid=True` 기록, 보상 preflight 우회 | stage2_ppo_train.py:244-246,876,930 / :79-85 | P2 |
| 14 | 계약 도달 불가: max_iter 30 요구 vs 설정 84/512, spectral_norm으로 LoRA 키명 변경 | checkpoint_contracts.py:64,70-71 / stage2_ppo_train.py:1036 | P2 |
| 15 | `bon_critic`이 GemmaEq3Context에 `.dim()` 호출 → 크래시 | baselines/bon_critic.py:35,59 | P2 |
| 16 | MCTS: `selection_seed` 무효(Gumbel 제거), 가지치기 Q=0(logit 0=p 0.5), 최종 선택이 방문 통계 무시 | cts_episode.py:128-131,590,663-681 | P2 |
| 17 | 공유 KV ABI 검사 우회(`return # Bypass`) | gemma_latent_forward.py:293 | P2 |
| 18 | 로더 경로·해시 강제: `/home/shoon/...` + `assert "58c4d7b5" in model_id` | model/gemma_loader.py:80,84 / stage1_openmath_train.py:455-457 | P1 |
| 19 | root_bootstrap이 잔차 NaN인데 `converged=True` | deq/root_bootstrap.py:82-84 | P2 |
| 20 | 설정: `default.yaml`의 미사용 키 다수, K를 평가는 `soft_thought_K`, 학습은 `latent_tokens_K`에서 읽음. `broyden_max_iter` 30→84로 테스트 3개 실패 | configs/default.yaml / run_cts_eval_full.py:824,1384 | P2 |
| 21 | `scripts/verify_repro_prereqs.py` 16행 괄호 문법 오류 | scripts/verify_repro_prereqs.py:16 | P2 |

## 3. 저장소 위생 (P1)

- `scratch/`(31개)·`results_s5/`(4개)에 서버 SSH 비밀번호 평문 → 1cb5d89에서 `.gitignore` 추가. **비밀번호 교체 필요**(Principal).
- 익명 제출본 사본 `anonymous_submission_neurips2026/` 250개 파일 중복 추적. 추적 파일에 `/home/shoon` 경로 존재.
- 루트 임시 스크립트·PREREG 13개·`docs/`·`results/` 산출물은 미추적 상태로 남음.
- 테스트 실패 39개: config drift 3, 코드/테스트 계약 drift ~27(D-7 불변식 `test_q14…`, `test_aime_garbage_fix` 포함), 서버 전용 9 + 수집 오류 1(`test_s6_damping`).

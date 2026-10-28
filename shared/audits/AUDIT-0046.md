# AUDIT-0046 — REPORT-0046 ((A) P5 분석 (B) 부드러운 스위치 코드) 감사 (CC, 2026-10-06 19:1x)

대상: REPORT-0046 (머리 SHA 722222f5…), 로컬 새 파일 4종, DIFF-soft_v2-vs-canonical.patch(4d4468d7…)
근거: 보고서, scripts/p5_promote_v6.py 815~843행(변별 통계가 z_star 사용), PREREG_S7_DEQ_v1.3 §10(c)(g), GOAL G1, PREREG_S7_DEQ_v2 §2·§5·§6. 서버 접속 0.

## (A) P5 분석 — PASS
| 항목 | 값 | 포인터 | 사전등록 기준 |
|---|---|---|---|
| 실행 전 SHA | 분석기 2종·P5 출력·520 config 일치 | REPORT-0046 §1 | — |
| 최근접 일치율 | 519/520 (0.998077) | p5_seedpair_v5.json `primary_flattened_163840d.match_count` | ≥515 (v1.3 §10(g)) |
| 불일치 | argmax 1, 동률 0, 결측 0 | 같은 파일 `mismatch_breakdown` | — |
| 다중 평형 비율(보고) | 8/520 (0.015385), cos min 0.844507 | `multiple_equilibria` | 보고만 |
| 승격 | 517/520 (0.994231) | p5_promote_v6.json `promotion_evaluation.qualified_count` | ≥494 (§10(c) 170행) |
| ①-b 확증(신규 520 부분) | 517/520 | `promotion_g1_definitions.definition_1b_combined600.heldout520_subset_count` | ≥90% (176행) |
| ①-b 통합 600(보조) | 596/600 | 같은 필드 `combined600_count` | ≥90% |
| ①-a train 1,000(보조) | r_min<1e-2 991/1000, k*≥50 확인 675, 판별 불가 325, k*<50 확인 0 | `definition_1a_train1000` | ≥95%, k*<50 미도달 |
| 변별 평균 cos | 0.568978 (z_star, 520) | p5_promote_v6.json `discrimination.mean_pairwise_cos` | ≤0.75 (GOAL G1) |
| cos>0.95 쌍(보고) | 5074/134940 | `discrimination.pairs_gt_095_*` | 보고만 |
| 앵커 재현 | GPU0 20/20 cos≥0.9999 | AUDIT-0045 | max|Δ|=0 또는 cos≥0.9999 |
| 1차(최종점) 병기 | r_1000<1e-2 456/520, 600 기준 530/600(0.8833) | `primary_convergence` | 승격 시 보고 병기 |

- k* 분포 [0,50) 0 / [50,300) 5 / [300,700) 75 / [700,1000] 440, top-3 일치 446/520, 구제 61(궤적 증가 18·진동 32·단조 감소 8·평탄 3) — §10(d) 필수 보고 충족.
- **남은 항목: ①-a의 판별 불가 325문항.** 로그 간격 때문에 k*<50 여부를 정할 수 없다. 사전등록은 "로그 궤적으로 동일 적용"만 정했으므로, 정확한 k*를 얻으려면 그 문항들의 궤적을 다시 계산해야 한다. 이것은 PREREG_S7_DEQ_v1.4 개정(재계산 범위·장치·판정 고정)으로 처리한다. ①-a는 비확증 보조 판정이다.

## (B) 부드러운 스위치 코드 — FIX
| # | 결함 | 근거 | 요구 |
|---|---|---|---|
| F1 | **74문항 중 54문항만 돌 위험** | `scripts/s7deq_soft_v2_run.py` 65~79행 `load_items_from_source`가 `indices`·`items` 키가 없으면 **첫 번째 리스트 키**를 쓴다. `configs/s6_items_74.json`의 키는 `nonconv_54`·`control_20`·`all_74`·`items_detail`(select_s6_items.py 125~157행)이라 `nonconv_54`만 읽을 수 있다 | `all_74`를 명시적으로 읽고, 74개·중복 0·`nonconv_54`∪`control_20`=`all_74`를 검사한 뒤 아니면 중단 |
| F2 | 앵커 점검이 3개 미만일 때 계속 진행 | launch.sh 앵커 블록: 경계 걸음 0 문항 수를 출력만 하고 본 실행으로 넘어감 | v2 §5(4): 3개 미만이면 anchor20 순서로 추가 계산, 끝내 3개 미만이면 **중단**(exit ≠0) |
| F3 | 앵커 비교 필드 부족 | 같은 블록: z_star·z_lowest만 비교 | rho_traj도 비교(v2 §5(3)), max|Δ|를 필드별로 출력 |
| F4 | 인자 명시 | 앵커 두 실행에 α·g0·seed·σ·tol·config를 생략(기본값이 0.25·0.25·2030·0.020·0.0이라 값은 같음) | 본 실행과 같은 인자를 명시 |
| F5 | §5(5) 미구현 | anchor20_gpu1.pt 대비 비교(보고만) 없음 | 추가 |
| F6 | 서버 패키지 경로 | 실행기가 `exe_scratch.s7.soft_v2`를 패키지로 import. 서버에 `__init__.py`가 필요할 수 있음 | 서버 SMOKE로 확인 |

- 장점: 경계 밖 경로가 정본 InjectedTransitionStep과 같은 호출 순서이며, 루프·잔차·최저점 코드는 diff상 주석만 바뀜. 로컬 시험 (a)~(e) 출력 원문 확인.
- 기존 파일은 고치지 않는다(R3-4): 수정본은 `_v2` 새 파일로 만든다.

판정: (A) PASS / (B) FIX(F1~F6) → TASK-0047

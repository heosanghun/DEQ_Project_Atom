# RESULT — G1 DEQ 수렴·변별 최종 집계 (CC, 2026-10-07)

- 사전등록: `PREREG_S7_DEQ_v1.3`(SHA-256 `254beb6f9ba891723a2feb79661708a2886f974b4ab3be30800e290f82567beb`) §10(c)(f)(g), `PREREG_S7_DEQ_v1.4`(SHA-256 `041125221a6360361c0aef3f3fd975abca0b58601d28b577282b297c36c3e50d`) §10(h), GOAL G1(4~7행)
- 확증 근거는 신규 held-out 520뿐이다. train 1,000과 기존 80은 비확증 보조 판정이다(v1.3 176행).

## 1. 집계표

| 항목 | 값 | 사전등록 기준 | 기준 대비 | 포인터 |
|---|---|---|---|---|
| 승격(신규 520, r_lowest<1e-2 ∧ k*≥50) | 517/520 | ≥494 | +23 | `p5_promote_v6.json` `promotion_evaluation.qualified_count` (REPORT-0046, AUDIT-0046) |
| ①-b 확증: 신규 520 r_k*<1e-2 | 517/520 (99.4 %) | ≥90 % | +49 | 같은 파일 `definition_1b_combined600.heldout520_subset_count` |
| ①-b 보조: 통합 600 | 596/600 (99.3 %) | ≥90 % | +56 | 같은 파일 `combined600_count` |
| ①-a 보조: train 1,000 (v1.4 §10(h)) | **991/1000** | ≥950 | **+41** | `shared/reports/kstar_replay/p5_promote_1a_replay_v2.json` `metric_1a.v14_1a` (SHA-256 `c0a7136218c96a76c354506aa54723b1003ed59a7145bd42f01aec24634dab77`) |
| 변별: 문항 간 평균 cos (z_star) | 0.569 | ≤0.75 | −0.181 | `p5_promote_v6.json` `discrimination.mean_pairwise_cos` |
| 변별: 최근접 일치율 (시드 2030↔2031) | 519/520 | ≥515 | +4 | `p5_seedpair_v5.json` `primary_flattened_163840d.match_count` |
| 앵커 재현 | GPU 0 20/20 cos≥0.9999 | max|Δ|=0 또는 cos≥0.9999 | 충족 | `anchor20_gpu0_vs_12.json` (REPORT-0045, AUDIT-0045) |
| 보고: 다중 평형 비율 | 8/520 (1.5 %) | 보고만 | — | `p5_seedpair_v5.json` `multiple_equilibria` |
| 보고: 최종점 r_1000<1e-2 (신규 520) | 456/520 | 승격 시 병기 | — | `p5_promote_v6.json` `primary_convergence.heldout520` |
| 보고: cos>0.95 쌍 | 5,074/134,940 | 보고만 | — | `discrimination.pairs_gt_095_*` |

**G1: 사전등록 기준 모두 충족.** 승격 규칙이 발동했으므로 수렴 판정값은 최소 잔차점(r_k*)이다(v1.3 170~176행).

## 2. ①-a 재계산 세부 (v1.4 §10(h))

- 대상 325문항(로그로 k*를 정할 수 없던 문항), GPU 1, 04:43:46~10:55:21. 출력 `zstar_s7_a0.25_train325_kstar_replay.pt` SHA-256 `1446340e73e58bc35b5703ba6f6309da6ccb60fadf9cfbd5abc89a5dece27977`(`KSTAR_REPLAY_DONE.flag`).
- 사전 점검 5문항: k* 5/5 같음, min ρ 차이 0, z_star cos ≥ 0.99999999999997 (`precheck`).
- 재현 점검: 325/325 통과, min ρ 상대·절대 차이 모두 0(`rho_diff_distribution`), 결측·NaN 0.
- 재계산 k*: [0,50) 0 / [50,300) 19 / [300,700) 129 / [700,1000] 177. k*<50 0개.
- 세 값: 보수 675 / 낙관 991 / v1.4 991 / 민감도 991. 판별된 675문항의 k* 최솟값은 369였다(v1.4 공개).
- 공개: v1.4는 P5 결과를 연 뒤 작성했다. ①-a는 비확증 보조 판정이다.

## 3. 제한 (FINAL에 적음)

1. P5(seed 2030)는 1~150번을 물리 GPU 1(10/5 04:43 이전), 151~520번을 GPU 0(10/6)에서 계산했다. 두 장치 차이는 앵커 20에서 cos ≥ 0.99999997(비트 단위로는 다름). P5b(seed 2031)는 전부 GPU 2.
2. 1차 지표(최종점 r_1000)로는 456/520(87.7 %)이며, 사전등록 승격 규칙에 따라 최소점을 판정값으로 썼다. 승격 규칙 자체는 규칙 착안 전 400·train680 열람 뒤 만들어졌다(v1.2 공개).
3. 여러 평형: 시드를 바꾸면 8/520 문항이 다른 점(cos<0.99, 최소 0.8445)에 도달한다.
4. 등록 학습 스크립트 원본(05bb5fa6) 소실, 실행형 pyc 보존(REPORT-0027·0028).
5. 서버가 10/5에 두 번 갑자기 재부팅됐다(원인 불명, ESC-0006). 이후 10/5 16:33부터 무재부팅.

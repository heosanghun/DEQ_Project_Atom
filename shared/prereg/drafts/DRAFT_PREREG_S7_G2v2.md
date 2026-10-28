# PREREG_S7_G2v2 — G2 2차 시도(개선판 M2′) 사전등록 (초안 rev4)

- **상태**: 초안 rev4(AUDIT-P012 FIX 13건 + 재감사 1 재FIX 5건 + 재감사 2 재재FIX 4건 반영). B 재감사 PASS와 §0의 잠금 전 점검 통과 뒤 제목·상태 줄만 바꿔 잠근다. 잠금 전에는 새 평가 문항(§2)의 문제 글·풀이·잠재 텐서를 열거나 계산하지 않는다.
- **발행**: 회사 PC Claude Code(CC), 2026-10-08
- **시도 번호**: G2 **2차 시도**. 1차 = `PREREG_S7_D_v3_8`(SHA-256 58b4a7f35249769716dca6316ab84cdc22d6259c88bd94a385a60f8a3c9bcd8b)의 P6R, 판정 기준 미충족(게이트 4). 1차 판정은 바꾸지 않는다.
- **근거**: Principal 지시 CTS-430(`shared/inbox/CTS-430.md`: "반드시 G2 통과가 최종목표"), RESULT-P6R-G2-main.md, DIAG-P6R-local-20261008.md, AUDIT-P012, GRAPH.md H-DEC-5.

## 0. 공개(작성 전에 본 것)와 잠금 전 점검

- **본 것**: P6·P6R 전체 결과(옛 held-out 600의 생성·채점), 진단 D1(옛 600에서 기존 디코더 A·B1·C·D1·E1·F의 정답 해설 NLL), D3(디코더 A·D·E 학습 손실), 코드 판독(주입 구조), M2′ 파일럿(옛 520 앞 20문항의 수렴·cos·교차 검색). 이 진단을 보고 디코더 손실 λ = 0(마진 없음)을 골랐다.
- **보지 않은 것**: M2′ z*′로 학습한 디코더의 어떤 결과(개발 NLL·생성)도 잠금 시각 전에는 존재하지 않는다. 잠금 시각과 개발 산출물 생성 시각(`results_s7/g2v2/decoder_*_v36.pt`, `results_s7/p6r/g2v2dev_*`)을 STATE와 파일 시각으로 대조해 보고한다. 개발 연쇄(`scripts/cc_g2v2_train_eval.sh`)는 디코더 학습부터 시작하므로, 잠금은 학습이 끝나기 전에 한다. **개발 산출물(`results_s7/p6r/g2v2dev_*`)이 잠금보다 먼저 생기면 그 사실과 시각을 공개하고, B가 그 영향을 판단하기 전에는 확증 평가를 시작하지 않는다.**
- **개발 연쇄 단일성(재재FIX 2)**: 깃발 전 `ps`로 대기 연쇄가 rev2 하나(PID 686863과 그 출력용 자식 686869, 15:07 기동)뿐임을 확인했다(`scratch/cc_refix_verify_153923.txt` [R2]). 각 스크립트의 PID 두 개는 `exec > >(tee …)`가 만드는 자식이다.
- **버전 가드(재재FIX 3)**: 서버 `scripts/check_p6_decoder_version.sh`(1fa95f72…)를 내려받아 로컬본(cc685dcb…)과 비교한 차이는 끝 빈 줄 하나다(`scratch/cc_guard_diff.txt`). 가드는 파일 이름이 `_v36.pt`로 끝나는지와 파일 존재만 본다. 직접 호출 시 `decoder_Ap_v36.pt`는 이름 규칙을 통과하고 파일 없음으로만 멈췄으며 `_v35.pt`는 거부됐다(`scratch/cc_refix_verify_153923.txt` [R3]).
- **디코더 SHA 고정(재FIX 3)**: 학습이 끝나면 `decoder_Ap_v36.pt`·`decoder_Ep_v36.pt`의 서버 SHA를 개발 생성보다 이른 시각에 STATE에 적는다. 확증 생성 직전 서버 SHA가 그 값과 같아야 하며, 조건과 디코더의 짝은 §4에 적은 그대로만 쓴다(v36e는 짝을 막지 않으므로 실행 명령으로 고정한다).
- **잠금 전 점검(z_src 일치, FIX-5)**: M2′ 실행기가 출력한 개발 600의 `z_src`와 기존 `results_s7/zsrc_heldout600.pt`(SHA-256 0da3748af2a3776c7fa91da50268a7829233939f56218d73f4b4511a6a8e5cff, 필드 `z_star`에 z_src 값)를 문항별로 비교한다. 합격: 600문항 모두 cos ≥ 0.99999. 불합격이면 잠그지 않고 원인을 진단한다(E 디코더 학습 입력과 평가 입력이 같은 양이어야 하므로).

## 1. 바뀌는 것(주장 범위, FIX-9)

1차 대비 세 가지가 바뀐다. 이 시험은 "이 세 가지를 바꾼 방법"의 G2 주장이다.
1. **DEQ 주입**: z_{k+1} = (1 − α)·z_k + α·[Φ(z_k) + g₀·z_src], z₀ = z_src + σ·ε (정본: 주입 ĉ₀ = 문제 토큰 임베딩 평균을 64칸에 복사). 코드 `cts/deq/zstar_inject_v2.py`(SHA-256 7024f1f2d9ff534c3ec43251ac9c8177fa0e384717e8752b330481f6aed91650), 실행기 `scripts/g2v2_zstar_run.py`(687d4480a622c845727a260ebcf49e052142269f10310b016fa82875323b4f2d) `--inject zsrc`. CLI 인자 α 0.25·g₀ 0.25·seed 2030·σ 0.020·max_nfe 1000·tol 0이 `configs/zstar_canonical.yaml`(bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157; 파일의 alpha 0.5·tol 1e-4는 CLI가 덮어씀)보다 우선한다. 동등성: `--inject mean`이면 정본 앵커 3문항 max|Δ| = 0(`scratch/cc_g2v2_pilot_check1.txt` 11~14행).
2. **디코더 손실 λ = 0**(정답 해설 NLL만). v3.8의 조건 E는 λ = 1이었다. 이번에는 A′·E′ 모두 λ = 0으로 같은 회차에 새로 학습한다.
3. **평가 문항**: GOAL.md의 held-out 600 대신 새 1,000문항(§2).

- 디코더 입력은 각 문항의 `z_star`(= z_1000), v3.7 관례와 같다.
- **G1과의 관계**: G1 확정은 정본 z*에 대한 것이다. M2′ z*′의 G1 기준(수렴·변별·앵커)은 확립되지 않았으며, §4 보고 항목으로 수렴·변별 수치를 적는다(판정 미사용).
- **GOAL.md 인정 여부**: G2v2 충족을 GOAL.md의 "G2 통과"로 인정할지는 Principal이 정한다(CTS-431로 기록). 잠금 시각까지 받지 못해도 잠금은 진행하며(§8), 그 경우 "개선 방법의 G2 기준 충족 여부"로만 보고한다.

## 2. 새 평가 문항 (FIX-1·2)

- 원천: `data/stage2/math_train_prompts_5000.jsonl`(SHA-256 f2cf899a4e26ec3ba645b2b825ebd3c7249400c0aca399e3a50e7df976c921df).
- 선택 스크립트: `scripts/g2v2_select_fresh.py`(SHA-256 799d320e6717e3106fbffb5d700c3a6faea2794b2395ecf4bca1c6deed2770eb). 문항 번호와 `level` 필드만 읽는다. 로컬 실행(numpy 2.5.3), 인자는 `scratch/server_configs/*.json`(서버 config 목록 원문 `scratch/cc_prereg_evidence_152515.txt` [E2] 18~53행)과 로컬 `configs/*.json`(단, 이 스크립트의 출력 `g2v2_exclude*`·`g2v2_fresh1000*`은 입력에서 뺀다), 그리고 서버 전체 json 정수 합집합 `configs/g2v2_server_json_union.json`(아래 (e)).
- 제외 집합(`configs/g2v2_exclude.json`에 기록): (a) 위 config 전부의 문항 합집합, (b) `data/s5_items_400.json`의 `canonical_20_indices`(앵커 0, 4, …, 76)·`train_indices`(320)·`heldout_indices`(80), (c) q_idx 0~99 전부(초기 실험의 `dataset[:32]`, `ds[:80]` 등 앞부분 사용), (d) `level`이 "Level ?", (e) **서버 전체 json 정수 합집합**: `/home/shoon/cts_e1`·`/home/shoon/cts_e1_s5` 아래 20 MB 미만 json 106개에서 0~4,999 정수 목록과 `qNNNN` 키 숫자를 모두 모은 2,240개(`scripts/cc_server_json_union.py` d2f52080cfe3b0e3dc16fe10a54c27e5cb8052f1901e32c841dfa1c2e12bc725, 출력 cb251b1a500c9bbfb201a700d737ec7888e8ba068d32f897e7a1d93527105223, 원시 출력 `scratch/cc_json_union_152907.txt`). 문항 번호가 아닌 정수도 섞이지만 보수적으로 모두 뺀다. 이 검색은 첫 선택 목록과 겹치는 서버 파일 35개를 찾은 데서 시작했다(`scratch/cc_prereg_evidence_152515.txt` 96~133행; 예: `data/s5_items_400_PENDING.json` 83개, `results/auto24h/*` 9월 실험 결과). `/home/shoon/cts_e1/configs`의 json은 autopilot·notifications 두 개뿐이고 문항 목록이 아니다(`scratch/cc_prereg_evidence_152515.txt` 2~17행).
- 선택: 남은 문항의 정렬된 목록 L에서 `numpy.random.default_rng(20261009).choice(len(L), 1000, replace=False)`, 고른 q_idx를 오름차순 정렬해 `configs/g2v2_fresh1000.json`(`{"indices": [...]}`). 샤드: 정렬 목록의 `[k::3]`, k = 0·1·2 → `configs/g2v2_fresh1000_shard{k}.json`.
- **선택 실행 결과**(2026-10-08 15:3x, 목록만 만듦, 문제 글·텐서 미열람): 제외 2,242, 남은 2,758, 선택 1,000, 제외와 겹침 0.
  - `configs/g2v2_exclude.json` 1c63770e08ae1e9990ea959be206eb21d76d8280fe7138accff00bea4e38a8b2
  - `configs/g2v2_fresh1000.json` 35683874284b371c6641e1280b92e6a69292a2dc5d89585158154a45b41cb3db
  - `configs/g2v2_fresh1000_shard0.json` f4417d21de92d6a0b3c98d5fa850217f977a4f475405ac6472954cd9388b075d
  - `configs/g2v2_fresh1000_shard1.json` fee4f45a92bd8e8690cceaf8cd29d57a5daf53d5230d01aefaa59818206a976f
  - `configs/g2v2_fresh1000_shard2.json` a256039bfcc05b1f89c9e94d669abdd2617eb3aaac04a095bf4936c1ad6a2dc1
  - 이력: (1) rev2 때 선택(제외 1,674, 목록 48b7f09d…)은 (e)를 넣기 전이라 폐기하고 `shared/archive/g2v2_select_run1/`에 보존했다. (2) 그보다 앞서 출력 파일이 입력으로 다시 들어간 잘못된 실행(제외 2,674)의 출력은 지웠다(RULES R2-1 위반, STATE 기록). 두 목록 모두 문제 글·텐서는 다루지 않았다.
- **서버 반영(재재FIX 1)**: 서버의 옛 목록 5개(48b7f09d… 등)는 `configs/archive/*_run1_48b7f09d.json`으로 옮겨 보존했고(`scratch/cc_refix_checks_153512.txt` [R1]), 새 5개를 올려 서버 SHA가 위 값과 같음을 확인했다(`scratch/cc_refix_verify_153923.txt` [R1]). 새 문항 z 계산 실행 스크립트는 실행기를 부르기 전에 샤드 파일 SHA가 위 값과 같은지 확인하고, 다르면 멈춘다(`g2v2_zstar_run.py`는 items SHA를 확인하지 않으므로 실행 스크립트에서 막는다).
- 확증 평가를 하지 않게 되면(§3) 이 목록은 열지 않은 채로 남고, 다음 시도에서 같은 규칙의 후보로 다시 쓸 수 있다.

## 3. 디코더와 사전 선택 규칙 (FIX-10 반영: R1 제외)

- 학습 스크립트 `scripts/s7_train_decoder_v36.py`(SHA-256 f2951054416c7b1593075c6a94e3c1ab7e7e7be516f078d540b5bd9ce5bf3ea9), 설정 `configs/s5_decoder.yaml`(31000d498d488767e623e491624e3605909f0b9356e85948fc75c442747b5cd6), 학습 문항 `configs/s7_train_993.json`(3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c), 명령: `--items configs/s7_train_993.json --max_steps 3000 --batch_size 8 --micro_batch_size 2 --grad_accum_steps 4 --lr_wproj 1e-4 --lr_adapter 3e-4 --weight_decay 0.01 --loss_lambda 0.0`(margin 기본 1.0은 λ = 0이라 작용 없음, seed 기본 2026).
  - A′: `--zstar results_s7/g2v2/zstar_zsrcinj_train1000.pt`(M2′ z*′, 잠금 때 SHA 기입) → `results_s7/g2v2/decoder_Ap_v36.pt`, GPU 5.
  - E′: `--zstar results_s7/zsrc_train1000_P4.pt`(340ef5c2bfb718fc371c932746476555d8d5a2bcd54c7413acc40e35840e45a7) → `results_s7/g2v2/decoder_Ep_v36.pt`, GPU 6.
  - 탐색용 Aq(z*′, λ = 1, GPU 7)는 같은 연쇄에서 학습·개발 평가하지만 **선택·판정에 쓰지 않고** 보고만 한다.
- **개발 평가**: 옛 held-out 600(개발용, 1차에서 이미 열람). 생성기 v36e(§4), 순서 파일 `results_s7/p6_item_order.json`(6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c). A′(GPU 5)·E′(GPU 6)는 새로 생성, C는 P6R 파일(`results_s7/p6r/p6_gen_C.jsonl`, SHA-256 24c82d97ff0a786dbf135b4673273fedf79bec371d9868eb98e6d2f541bbee77, GPU 0)을 쓴다. 채점: `s7_eval_score_p6_v36.py --gen_A …g2v2dev_gen_Ap.jsonl --gen_C …p6_gen_C.jsonl --gen_E1 …g2v2dev_gen_Ep.jsonl --gen_B1 …p6_gen_B1.jsonl --order results_s7/p6_item_order.json --out_json results_s7/p6r/g2v2dev_score_ApEp.json`.
- **선택 규칙(개발 결과 전에 고정)**: `g2v2dev_score_ApEp.json`의 `gates.Gate_4.strict.delta`(A′ − E′) > 0 **그리고** `gates.Gate_3.strict.delta`(A′ − C) > 0이면 §4 확증 평가를 한다. 아니면 새 문항을 열지 않고 방법을 다시 설계한다(GRAPH H-DEC-5 기각 조건 → N8).

## 4. 확증 평가 (FIX-3·4·6·7)

- **새 문항 z 계산**: `g2v2_zstar_run.py --inject zsrc`로 샤드 k를 GPU 5·6·7에서 계산(`results_s7/g2v2/fresh_zsrcinj_shard{k}.pt`) → 병합 `fresh_zsrcinj_1000.pt`. E용 변환: 각 키에 `{"z_star": entry["z_src"]}`만 담은 `fresh_zsrc_1000.pt`. 두 파일 SHA를 보고한다.
- **생성기**: `scripts/s7_eval_gate_p6_v36e.py`(SHA-256 a2755c1b591e12389c8efe167aa33a59a7ae6c0876234585381b6a129faa460a). v36d 대비 변경은 (1) 정식 실행에서 `results_s7/g2v2/decoder_*_v36.pt` 허용, (2) 정식 실행에서 `z_star` 필드만 읽음(z_lowest 대체 차단), (3) 머리·로그 문자열뿐(diff 27행). 시험: 허용 밖 경로 거부·g2v2 경로 통과·CPU 2문항 생성(`scratch/cc_prereg_evidence_152515.txt` [E8] 134~139행). 정지 집합 {1, 106}, 탐욕, `max_new_tokens` 512, 프롬프트 v3.7 §3.
- **조건과 장치**: 샤드 k(GPU 5·6·7)마다 같은 GPU에서 차례로
  - A: `--cond A --decoder_ckpt results_s7/g2v2/decoder_Ap_v36.pt --zstar results_s7/g2v2/fresh_zsrcinj_1000.pt`
  - C: `--cond C`
  - E: `--cond E1 --decoder_ckpt results_s7/g2v2/decoder_Ep_v36.pt --zsrc results_s7/g2v2/fresh_zsrc_1000.pt`
  - B1(보조): `--cond B --shift 1 --decoder_ckpt results_s7/g2v2/decoder_Ap_v36.pt --zstar results_s7/g2v2/fresh_zsrcinj_1000.pt` (시프트는 샤드 목록 안에서 1칸)
  - 공통 `--items configs/g2v2_fresh1000_shard{k}.json --order_sha <샤드 SHA> --out results_s7/p6r/g2v2fresh/gen_<cond>_s{k}.jsonl`
- **병합·채점**: 샤드 세 개를 이어 `results_s7/p6r/g2v2fresh_merged/p6_gen_{A,B1,C,E1}.jsonl`(빈 새 폴더). 채점 전 점검: 네 파일 각 1,000행·고유 id 1,000·id 집합이 `g2v2_fresh1000.json`과 같음·JSON 오류 0. 하나라도 어긋나면 채점하지 않는다. 명령: `s7_eval_score_p6_v36.py --gen_A …/p6_gen_A.jsonl --gen_B1 …/p6_gen_B1.jsonl --gen_C …/p6_gen_C.jsonl --gen_E1 …/p6_gen_E1.jsonl --gen_dir results_s7/p6r/g2v2fresh_merged --order configs/g2v2_fresh1000.json --out_json results_s7/p6r/g2v2fresh_merged/gate_table.json --out_md results_s7/p6r/g2v2fresh_merged/gate_table.md`.
- **판정**: G2v2 = 게이트 2 ∧ 3 ∧ 4 (strict; 채점기 `scripts/s5_eval_grader_v2.py` 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13, 채점 스크립트 `scripts/s7_eval_score_p6_v36.py` 547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd, 대응 부트스트랩 10,000회·시드 2026, 정확 McNemar 단측).
  - 게이트 2: acc(A) − acc(C) ≥ −2 %p / 게이트 3: A − C, p < 0.025 ∧ 95 % CI 하한 > 0 / 게이트 4: A − E, 같은 기준 / 게이트 1(보조): A − B1
- **보고(판정 미사용)**: M2′ z*′의 r_1000 < 1e-2·r_lowest < 1e-2 비율, 문항 간 평균 cos, 조건별 lenient·`\boxed{}`·512 도달 비율, 샤드(장치)별 문항 수, 반복 시도 보정 참고값(2차 시도이므로 게이트 3·4 p × 2).

## 5. 갈림길 차단 (FIX-8)

- G2v2 판정은 새 1,000문항으로 한 번 정한다. 결과를 본 뒤 조건·문항·프롬프트·생성 길이·채점·게이트·디코더를 바꾸지 않는다.
- 기준 미충족이면 3차 시도는 다시 새 문항(이번 1,000문항 제외)과 새 사전등록으로만 한다. 시도 번호와 각 결과를 모두 보고한다.
- 도구 결함은 결과와 무관한 근거(코드·로그·재현 점검)로만 판정한다. 결함이 확인되면 RULES R3-3에 따라 고친 뒤 같은 평가를 다시 하고, 다시 한 결과가 판정이 되며, 두 결과를 모두 보고한다(개정과 B 감사를 거친다).

## 6. 고정 파일 (FIX-11, 재FIX 2)

출처: "서버" = `scratch/cc_prereg_evidence_152515.txt` [E3] 54~83행 서버 실측, "로컬" = 회사 PC `D:\AI\cts` 실측, "둘 다" = 두 값 일치.

| 파일 | SHA-256 | 출처 | 포인터 |
|---|---|---|---|
| `configs/zstar_canonical.yaml` | bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157 | 서버 | [E3] |
| `cts/deq/transition_inject.py` | 127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c | 서버 | [E3] |
| `cts/deq/zstar_canonical.py` | b7170aeb68833448c8bb0977eb7ac035ef06614c9a5edf2f228448c2bb96d45d | 서버(로컬 사본 ca454090…는 다름, `PREREG_LOCAL_L1.md:4`) | `scratch/cc_prereg_evidence_152515.txt` 59행 |
| `cts/deq/zstar_canonical_lowest.py` | 9f73fb272353e8a307b9e628c65d1f17eaff164e2138fa3d232917958d6a6b2f | 둘 다 | [E3], `scratch/cc_g2v2_pilot_check1.txt` 2행 |
| `cts/deq/zstar_inject_v2.py` | 7024f1f2d9ff534c3ec43251ac9c8177fa0e384717e8752b330481f6aed91650 | 둘 다 | [E3] |
| `scripts/s7_zstar_lowest_run.py` | 779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba | 둘 다 | [E3] |
| `scripts/g2v2_zstar_run.py` | 687d4480a622c845727a260ebcf49e052142269f10310b016fa82875323b4f2d | 둘 다 | [E3] |
| `scripts/s7_eval_gate_p6_v36e.py` | a2755c1b591e12389c8efe167aa33a59a7ae6c0876234585381b6a129faa460a | 둘 다 | [E3] |
| `scripts/s7_train_decoder_v36.py` | f2951054416c7b1593075c6a94e3c1ab7e7e7be516f078d540b5bd9ce5bf3ea9 | 둘 다 | [E3] |
| `scripts/check_p6_decoder_version.sh` | 1fa95f72d5e5cea2ad1836dd458544190467f0cfb1daa14e41f6459cb51d870b | 서버(로컬 사본 cc685dcb…는 다름, 서버 값이 기준) | `scratch/cc_prereg_evidence_152515.txt` 69행 |
| `scripts/cc_g2v2_train_eval.sh` rev2 | d7435be370066ecb1aa2e5e2c2e95dbc9d87024bace9258e82882ef635150375 | 둘 다, 15:07:18 기동 세션 | `scratch/cc_prereg_evidence_152515.txt` 70행·87행 |
| `results_s7/zsrc_train1000_P4.pt` | 340ef5c2bfb718fc371c932746476555d8d5a2bcd54c7413acc40e35840e45a7 | 서버 | `scratch/cc_prereg_evidence_152515.txt` 73행 |
| `results_s7/zsrc_heldout600.pt` | 0da3748af2a3776c7fa91da50268a7829233939f56218d73f4b4511a6a8e5cff | 서버 | [E3] |
| `results_s7/decoder_E_v36.pt` | f8115ac33f44045ce5f572a377ee84b2e6311be347f154190b6c79199d7f3cac | 서버 | `scratch/cc_prereg_evidence_152515.txt` 75행 |
| `results_s7/anchor_ref_gpu1_check.pt` | 5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6 | 서버 | [E3] |
| `results_s7/p6r/p6_gen_C.jsonl`·`p6_gen_B1.jsonl` | 24c82d97ff0a786dbf135b4673273fedf79bec371d9868eb98e6d2f541bbee77 · 4d30f021dab783e8bb5d284baf501923ff9fda4efd6da9bf1b8d295e652b44f4 | 둘 다 | [E3], AUDIT-0069 |
| `harness_A_v1.py`(`/home/shoon/cts_e1/scratch/`) | 5fb768a8f56add15c7cdca069f5e1a7f80d16fd231f8ecae1fca4fcf6920bf52 | 서버 | `scratch/cc_prereg_evidence_152515.txt` 81행 |
| `/home/shoon/checkpoints/m1_final_frozen.pt` | f2119bbe5c0ad914ee00c34e9109674d15de07c5e028e288ab05b88dd448d1f9 | 서버 | `scratch/cc_prereg_evidence_152515.txt` 82행 |
| `data/s5_items_400.json` | d5927a3e04eae0d3a934075752a7119f4a4480c4b423e2c2e7f2c4367ec3f57a | 로컬 | CC 로컬 실측(AUDIT-P012 재감사 1 값과 같음) |
| `results_s7/g2v2/zstar_zsrcinj_train1000.pt`·`_dev600.pt` | 계산 완료 뒤 병합 로그의 값을 잠금 전에 기입 | 서버 | `raw_s7/g2v2_rebalance.log` |
| 라이브러리 | torch 2.11.0+cu128, transformers 5.10.2, numpy 2.4.4(서버); 선택 스크립트는 로컬 numpy 2.5.3 | 서버·로컬 | `scratch/cc_prereg_evidence_152515.txt` 85행 |

v36e 시험 원문: `scratch/cc_prereg_evidence_152515.txt` 134~139행(허용 밖 경로 거부, g2v2 경로 통과 후 파일 없음, CPU 2문항 출력 `v36e_smokeA.jsonl` 15fb909d…).

## 7. 한계 (FIX-12)

1. 이 문서는 1차 실패와 진단을 본 뒤, "반드시 G2 통과" 지시(CTS-430) 아래 작성한 2차 시도다. λ = 0 선택은 옛 600의 진단(D1·D3)에 근거했다.
2. 개발 600은 1차에서 이미 열람한 문항이다. 개발 결과는 선택 규칙에만 쓴다.
3. "새 문항"은 "§2의 제외 규칙(로컬·서버 config, s5_items_400, q_idx<100, 서버 전체 json 정수 합집합)으로 확인한 범위에서 이 프로젝트가 쓰지 않은 문항"이다. 서버 밖(다른 PC·외부 저장소) 사용 이력은 확인하지 않았다. 서버 json 검색도 `.jsonl`, `.pt`의 키, 로그·csv, `[{"q_idx": …}]` 같은 레코드 목록 안의 단일 번호, 깊이 4·길이 6,000을 넘는 목록은 보지 않았다. 5,000문항 파일은 Stage 2 PPO 프롬프트 풀이고, Stage 1 데이터(OpenMathInstruct-2)와 Gemma 사전학습에 MATH train이 들어갔을 수 있어 "모델이 본 적 없는 문항"은 아니다. 이 관계는 모든 조건에 같다.
4. M2′ z*′의 G1 기준은 확립되지 않았다(§1).
5. 개발 비교의 C는 P6R 파일(GPU 0)이고 A′·E′는 GPU 5·6이다(개발용, 판정 미사용). 확증은 문항별 같은 GPU다.
6. GPU 0·1·3은 하드웨어 감속(ESC-0007), GPU 2는 오류, GPU 4는 봉인이라 쓰지 않는다.

## 8. GOAL.md 인정 (CTS-431)

- §1의 Principal 결정(G2v2 충족을 GOAL.md G2 통과로 인정할지)은 `shared/inbox/CTS-431.md`로 기록한다. 잠금 시각까지 받지 못하면 잠금은 진행하고, 결과는 "개선 방법(M2′·λ = 0·새 1,000문항)의 G2 기준 충족 여부"로만 보고한다. 이후 받은 결정은 판정을 바꾸지 않고 해석 범위만 정한다.
- Aq(탐색용) 개발 결과를 3차 시도 설계에 쓰게 되면 그 사실을 3차 사전등록에 공개한다.

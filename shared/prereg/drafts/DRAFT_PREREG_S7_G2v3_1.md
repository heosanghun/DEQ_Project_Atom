# PREREG_S7_G2v3.1 — G2 3차 시도(M3) 사전등록 개정본 (초안 rev2, AUDIT-P013 FIX·재FIX 반영)

- **상태**: 초안 rev1. 원본 `shared/prereg/PREREG_S7_G2v3.md`(SHA-256 1abfa835da2c256138955e1a4c1e37a31b2655d7fec3761ee3e06003db8daf41)를 AUDIT-P013(SHA-256 a46db9e31d1cbba1fb31e82d9daf46f614bdb6f613bb2c2b7277a2dae1073268) FIX-1~8대로만 고친다. 감사가 지적하지 않은 변경은 하지 않는다. B 재감사 PASS 뒤 제목·상태 줄만 바꿔 잠근다.
- **발행**: 회사 PC Claude Code(CC), 2026-10-10 23:4x KST, rev2 2026-10-11 00:0x (원본 발행: 실행 에이전트 AG)
- **개정 형식(재FIX 1)**: 원본 대비 `diff -u`는 `shared/prereg/drafts/DIFF_G2v3_to_G2v3_1.patch`(SHA는 STATE 기록). 원본의 절 구성으로는 결함 실행 처리(§3.2)·확증 명령 전문(§4)·절차 공개(§0)를 넣을 자리가 없어 전면 재작성했다. 절 대응: 원본 §0 → v3.1 §0(공개·절차), 원본 §1 → §1(구현대로 정정), 원본 §2 → §2, 원본 §3.1 → §3.1(+비대칭 공개), 원본 §3.2 → §3.2(결함 실행·다시 하기·JSON 키), 원본 §4 → §4(명령 전문), 원본 §5 → §5(+보정·4차 규칙), 원본 §6 → §6(+출처), 신설 §7(한계·GOAL).
- **시도 번호**: G2 **3차 시도**.
  - 1차: `PREREG_S7_D_v3_8`(잠금본 58b4a7f35249769716dca6316ab84cdc22d6259c88bd94a385a60f8a3c9bcd8b), P6R 판정 기준 미충족(게이트 4 −4.83 %p).
  - 2차: `PREREG_S7_G2v2`(잠금본 4ea1d9663c3a029a382bd503a9628992b600992160cfbf3aef61d0b66d726e85), **개발 선택 규칙에서 멈춤**(A′−E′ −0.33 %p) — 확증 평가는 하지 않았고 G2 판정을 내리지 않았다(RESULT-G2v2-dev.md).
  - 1·2차 결과는 바꾸지 않는다.
- **근거**: Principal 지시 CTS-430, GRAPH.md N8, `shared/proposals/G2v3_PLAN_20261009.md`, REPORT-0070, REPORT-0071, AUDIT-P013.

## 0. 절차 공개 (FIX-3)

1. 원본은 AG가 **B 감사 없이** 작성·잠갔다(2026-10-10). CC가 23:2x에 발견해 TASK-0073으로 확증 착수를 막고 사후 감사(AUDIT-P013)를 받았다.
2. 잠금이 학습보다 앞섰다는 근거는 로컬 파일 시각(PREREG 21:00:34, TASK-0072 21:00:41)과 `shared/ssh_log.csv` 순서(21:02 업로드 → 21:31~21:39 실행 시도·수정)뿐이다. 서버 쪽에 잠금 시각을 찍은 기록은 없다. 문서에 적힌 시각(잠금 21:15, TASK-0072 21:18)은 실제 파일 시각과 다르다.
3. 잠금 뒤 21:33~21:39에 학습·잠재 스크립트를 세 번 고쳤다(import·loader·canonical fix, `ssh_log.csv` 21:33:44·21:36:28·21:39:09 업로드). 연쇄 마스터 로그(`raw_s7/task0072_full_master.log`)에 시작이 네 번 있다: 21:32:00(학습 08fcdee6…, 잠재 cd9bf305…), 21:35:04(4da1e3cc…, f7be64ba…), 21:37:52(cd8ba151…, 782f8dcc…) — 세 번 모두 학습 종료 코드 1로 산출물 없이 끝났다 — 그리고 **21:40:29(학습 `scripts/g2v3_m3_full_train.py` 36e0f18addc8d46f6d8bf376cd95e89cf9eb75a1f15c4e8b714141efe5a49c99, 잠재 `scripts/g2v3_m3_eval_latents.py` 782f8dcc0c827446a671a85a992ff973bca292890226cf2fe894bbf6dc08ae6f)가 실제 산출물을 만든 실행**이다(`scratch/cc_g2v3_refix_235206.txt` [M]). 중간 판의 diff는 남기지 않았고, 최종본은 B가 전문 감사했다(AUDIT-P013).
   - 이후 CC·AG가 띄우는 연쇄는 tmux 세션 안에서(R7-4), CPU 단계는 `nice -n 19`·`ulimit -v 62914560`으로(R2-5), 모든 접속을 `cts_ssh.py`로(R7-1) 한다.
4. 21:40경 실제 연쇄를 기동한 접속은 `ssh_log.csv`에 없다(R7-1 위반). 연쇄는 tmux 밖에서 돌았고(R7-4), CPU 단계에 `ulimit`·`nice`가 없었다(R2-5).
5. 학습 산출물은 22:06에 생겼다(`results_s7/g2v3/full_model_A.pt` 2026-10-10 22:06:58, 같은 파일). 감사 판단(AUDIT-P013)에 따라 이미 끝난 학습 산출물은 §6의 SHA로 고정해 그대로 쓴다.
6. **본 것**: 1·2차 전체 결과와 진단, REPORT-0071 파일럿(손실 곡선 포함, Mode A 손실 2.00 → 5.78 → 4.04 상승 구간 있음), 학습 로그의 손실 값. **보지 않은 것**: 개발 600의 생성·채점 결과(이 개정 작성 시각에 개발 채점표는 없다), 새 1,000문항의 문제 글·풀이·텐서.

## 1. 방법 M3의 정확한 내용 (FIX-2)

- **고정점**: M2′(`cts/deq/zstar_inject_v2.py` 7024f1f2…, 주입 = z_src)로, 학습 **전** 사상 Φ₀에서 미리 계산한 z*′₀(`results_s7/g2v2/zstar_zsrcinj_{train1000,dev600}.pt`)를 쓴다. 학습 중에 바뀐 사상으로 고정점을 **다시 풀지 않는다**.
- **학습되는 사상**: Φ_θ(LoRA rank 8 q·v·o, `routing_proj`, γ). 한 걸음 사상 T_θ(z) = (1 − α)·z + α·[Φ_θ(z) + g₀·z_src], α = g₀ = 0.25.
- **조건 A**: 디코더 입력 = T_θ,A(z*′₀). **조건 E‴**: 디코더 입력 = T_θ,E(z_src). 두 조건 모두 같은 구조의 학습 가능 사상과 디코더(PrefixProjector)를 같은 손실(정답 해설 NLL, λ = 0)·같은 단계 수로 학습한다. 학습과 평가는 같은 한 걸음 식을 쓴다.
- **주장 범위**: "M2′ 고정점에서 출발한 학습된 한 걸음"이 "z_src에서 출발한 같은 학습된 한 걸음"보다 디코더에 유용한가. 학습된 사상의 고정점을 다시 풀어 쓰는 방법이 아니다.
- **보고 항목(판정 미사용)**: 학습된 사상의 고정점 근접도 r_θ = ‖T_θ,A(z*′₀) − z*′₀‖ / (α·‖z*′₀‖)와 ‖T_θ,A(z*′₀) − z*′₀‖/‖z*′₀‖, 개발·확증 문항 분포.

## 2. 문항 (원본과 같음)

- 학습 `configs/s7_train_993.json`(3088a153…), 개발 `results_s7/p6_item_order.json`(6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c; `configs/g2v2_dev600.json` 60b501ec…와 같은 600문항), 확증 `configs/g2v2_fresh1000.json`(35683874…)과 샤드 3개(f4417d21…·fee4f45a…·a256039b…). 확증 문항은 이 개정 시각까지 쓰지 않았다(`scratch/cc_g2v3_state_*.txt`).

## 3. 학습(완료)과 개발 선택 규칙

### 3.1 학습 (완료, 원본 §3.1 그대로) — 공개 (FIX-7)
- `scripts/g2v3_m3_full_train.py`(36e0f18a…), 1,500단계, 배치 4, lr: 투영기 1e-4·어댑터 3e-4·DEQ 3e-5, weight decay 0.01/0.001, GPU 5(A)·6(E). 학습 문항 993/993 사용(`raw_s7/g2v3_full_train_{A,E}.log` `[DATA] Filtered 993 / 993`).
- 비대칭 공개: (1) 보조 IFT 손실 λ_ift A 0.1·E 0.0, (2) 시드를 고정하지 않아 A·E 투영기 초기값이 다르고 재현이 안 된다, (3) 학습 GPU가 다르다(5·6), (4) 학습 예산은 G2v2 디코더(3,000단계·배치 8)의 약 1/4이다. 재학습은 하지 않는다(AUDIT-P013 FIX-7).

### 3.2 개발 평가 — 결함 실행과 다시 하기 (FIX-1)
- **결함**: 원본 개발 실행에서 조건 A의 잠재 파일(`results_s7/g2v3/dev600_zstar_A.pt` b5c7ac63…)은 키가 `q{q:04d}`와 정수뿐이었다. v36e는 조건 A에서 `{**zstar_s7_a0.25_400.pt, **--zstar}`로 합친 뒤 `q{q:04d}_a0.25`를 먼저 찾으므로(`s7_eval_gate_p6_v36e.py` 183~189·497~503행), 개발 600 중 옛 80문항은 **1차 정본 z***를 받았다. CC 서버 실측: 80문항이 `_a0.25` 키로 옛 파일을 조회함(`scratch/cc_fix1_verify_233746.txt`). 조건 E는 `--zsrc` 파일만 읽으므로 영향이 없다.
- **처리**: 원본 개발 A 결과(`results_s7/p6r/g2v3dev_gen_A.jsonl`, 채점표 `g2v3dev_score_AE.*`)는 "결함 실행"으로 보고만 하고 선택 규칙에 쓰지 않는다.
- **다시 하기**: `scripts/g2v3_m3_eval_latents_v2.py`(73db9daf…; v1 대비 `q{q:04d}_a0.25` 키 추가, DEQ 적재 가드, 빠진 z·문항 글이면 중단)로 만든 `results_s7/g2v3/dev600_zstar_A_v2.pt`로 **조건 A만** 다시 생성한다(v2로 다시 계산한 잠재는 원본과 600/600 문항에서 값이 완전히 같고(max|Δ| = 0) 키만 늘었다. 적재 가드: DEQ 텐서 436개 적용·예상 밖 키 0. `scratch/cc_relatents_*.txt`): `CUDA_VISIBLE_DEVICES=<GPU 7 UUID> python scripts/s7_eval_gate_p6_v36e.py --cond A --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_Am3_v36.pt --zstar results_s7/g2v3/dev600_zstar_A_v2.pt --order_sha 6c76ddb7… --out results_s7/p6r/g2v3dev_gen_A_v2.jsonl`. E는 원본 실행의 `g2v3dev_gen_E.jsonl`을 쓴다(장치 GPU 6). 다시 한 A가 선택 규칙 값이며 두 결과를 모두 보고한다.
- 채점: `s7_eval_score_p6_v36.py --gen_A results_s7/p6r/g2v3dev_gen_A_v2.jsonl --gen_C results_s7/p6r/p6_gen_C.jsonl --gen_E1 results_s7/p6r/g2v3dev_gen_E.jsonl --gen_B1 results_s7/p6r/p6_gen_B1.jsonl --gen_dir results_s7/p6r --order results_s7/p6_item_order.json --out_json results_s7/p6r/g2v3dev_score_AE_v2.json --out_md results_s7/p6r/g2v3dev_score_AE_v2.md`.
- **선택 값 읽기 전 점검(재FIX 4)**: A_v2·E·C 세 생성 파일 각 600행·고유 id 600·JSON 오류 0을 확인하고, 하나라도 어긋나면 선택 규칙을 적용하지 않는다. 다시 하기 A의 병합 조회값이 v2 파일 값과 600/600 같음은 확인했다(`scratch/cc_g2v3_refix_235206.txt` [R], 조회 키 접미 `_a0.25`만).
- 보조 게이트(게이트 1 A − B1 등)는 개발 채점에서 `--gen_dir`로 옛 파일이 섞일 수 있으므로 개발 단계에서는 쓰지 않는다.
- **선택 규칙(FIX-5, strict 기준)**: `g2v3dev_score_AE_v2.json`의 `gates.Gate_4.strict.delta`(A − E‴) > 0 **그리고** `gates.Gate_3.strict.delta`(A − C) > 0이면 §4 확증을 한다. 아니면 새 문항을 열지 않는다.
- 공개: 개발 C는 P6R 파일(GPU 0·v36d)이고 A·E는 GPU 7·6·v36e다. 원본 결함 실행의 A 결과는 다시 하기 결과보다 먼저 서버에 생길 수 있으며, 그 사실을 보고한다.

## 4. 확증 평가 (FIX-4)

1. **디코더·사상 고정**: A = `results_s7/g2v2/decoder_Am3_v36.pt`(682d05c3638e3ffadfabb6f38fd6bd3d06a3e2004964f35cbeb60be9a410cd10) + `results_s7/g2v3/deq_backbone_A.pt`(fec12c8faab77c31276ba4d155aaf41f32243f33212793e92f34117e80a14cbf). E‴ = `decoder_Em3_v36.pt`(ba1d2788ec4f3fb38949f17cd215b1d42721c175f3a47d2659bc4dc573722e05) + `deq_backbone_E.pt`(255ed2731098fa5081c540e456383fa73276e9cb73c589f9f39d0da0210f581b). 실행 직전 서버 SHA 대조. 짝은 이것만.
2. **새 문항 M2′ z*′₀**: 샤드 SHA(§2)를 확인한 뒤 샤드 k를 GPU(샤드 0 → GPU 5 `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a`, 1 → GPU 6 `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c`, 2 → GPU 7 `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b`)에서 계산한다: `CUDA_VISIBLE_DEVICES=<UUID> python scripts/g2v2_zstar_run.py --items configs/g2v2_fresh1000_shard{k}.json --inject zsrc --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --config configs/zstar_canonical.yaml --out results_s7/g2v3/fresh_zsrcinj_shard{k}.pt`(실행기 687d4480…, 라이브러리 `cts/deq/zstar_inject_v2.py` 7024f1f2…, yaml bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157; yaml의 alpha 0.5·tol 1e-4는 CLI가 덮어씀).
   - 병합·변환: `python scripts/g2v3_fresh_prep.py`(SHA-256 59889cd3cc829a5bb281e0d34a7a20899f824272f0911eb8db8547cbe741b423) → `results_s7/g2v3/fresh_zsrcinj_1000.pt`(키 1,000 = 목록, 여분 0)와 `fresh_zsrc_1000.pt`(`{"z_star": z_src}`; 실행기 출력을 그대로 넣으면 `extract_tensor`가 `z_star`를 먼저 읽는 함정을 피함). 변환 뒤 모든 키에서 변환 z_star와 원 z_src의 max|Δ| = 0이어야 하며 아니면 중단한다. r_1000 < 1e-2 비율을 보고한다.
3. **학습된 한 걸음**(GPU 7 `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b`, 차례로):
   - `CUDA_VISIBLE_DEVICES=<GPU 7 UUID> python scripts/g2v3_m3_eval_latents_v2.py --mode A --device cuda:0 --items configs/g2v2_fresh1000.json --deq_ckpt results_s7/g2v3/deq_backbone_A.pt --zstar results_s7/g2v3/fresh_zsrcinj_1000.pt --zsrc results_s7/g2v3/fresh_zsrc_1000.pt --out results_s7/g2v3/fresh_A_latent.pt`
   - `CUDA_VISIBLE_DEVICES=<GPU 7 UUID> python scripts/g2v3_m3_eval_latents_v2.py --mode E --device cuda:0 --items configs/g2v2_fresh1000.json --deq_ckpt results_s7/g2v3/deq_backbone_E.pt --zsrc results_s7/g2v3/fresh_zsrc_1000.pt --out results_s7/g2v3/fresh_E_latent.pt`(E는 Φ_θ,E)
   - 두 로그의 `[LOAD-GUARD]` 행과 두 파일 SHA를 보고한다.
4. **생성**: 샤드 k마다 같은 GPU(5·6·7)에서 A → C → E1 → B1, `--items configs/g2v2_fresh1000_shard{k}.json --order_sha <샤드 SHA>`:
   - A: `--cond A --decoder_ckpt …/decoder_Am3_v36.pt --zstar results_s7/g2v3/fresh_A_latent.pt`
   - C: `--cond C`
   - E1: `--cond E1 --decoder_ckpt …/decoder_Em3_v36.pt --zsrc results_s7/g2v3/fresh_E_latent.pt`
   - B1(보조): `--cond B --shift 1 --decoder_ckpt …/decoder_Am3_v36.pt --zstar results_s7/g2v3/fresh_A_latent.pt`
   - 출력 `results_s7/p6r/g2v3fresh/gen_<cond>_s{k}.jsonl`
5. **병합·점검·채점**: 빈 새 폴더 `results_s7/p6r/g2v3fresh_merged/`에 `p6_gen_{A,B1,C,E1}.jsonl`. 채점 전 네 파일 각 1,000행·고유 id 1,000·id 집합 = `g2v2_fresh1000.json`·JSON 오류 0. 명령: `s7_eval_score_p6_v36.py --gen_A …/p6_gen_A.jsonl --gen_B1 …/p6_gen_B1.jsonl --gen_C …/p6_gen_C.jsonl --gen_E1 …/p6_gen_E1.jsonl --gen_dir results_s7/p6r/g2v3fresh_merged --order configs/g2v2_fresh1000.json --out_json …/gate_table.json --out_md …/gate_table.md`. 판정 키 `gates.Gate_{2,3,4}.strict`.
6. **판정**: G2v3 = 게이트 2(A − C ≥ −2 %p) ∧ 게이트 3(A − C: p < 0.025 ∧ 95 % CI 하한 > 0) ∧ 게이트 4(A − E‴: 같은 기준), strict, 채점기 `scripts/s5_eval_grader_v2.py`(90cc93e4…), 채점 스크립트 547e1d91…, 부트스트랩 10,000·시드 2026, 정확 McNemar 단측. 게이트 1(A − B1)은 보조.

## 5. 갈림길·통계 (FIX-6)

- 판정은 새 1,000문항으로 한 번. 결과를 본 뒤 조건·문항·생성·채점·게이트·가중치를 바꾸지 않는다.
- **반복 시도 보정 참고값**: 3차 시도이므로 게이트 3·4의 p × 3을 함께 보고한다(판정은 사전등록 기준 p < 0.025).
- 도구 결함은 결과와 무관한 근거로만 판정하고, RULES R3-3에 따라 고친 뒤 같은 평가를 다시 하며, 다시 한 결과가 판정이고 두 결과를 모두 보고한다(개정·B 감사).
- 기준 미충족이면 4차는 새 사전등록과 이번 1,000문항을 제외한 새 문항으로만 한다(남은 후보 2,758 − 1,000 = 1,758, `configs/g2v2_exclude.json` 규칙 + 이번 목록).
- GRAPH.md에 H-DEC-6(M3)과 기각 조건(개발 Δ(A − E‴) ≤ 0 또는 Δ(A − C) ≤ 0, 또는 확증 미충족)을 등록한다.

## 6. 고정 파일 (FIX-8) — 출처: 서버 실측 `scratch/cc_g2v3_evidence_234136.txt`

| 파일 | SHA-256 |
|---|---|
| `scripts/g2v3_m3_full_train.py`(최종 실행본) | 36e0f18addc8d46f6d8bf376cd95e89cf9eb75a1f15c4e8b714141efe5a49c99 |
| `scripts/g2v3_m3_eval_latents.py`(v1, 결함 키) | 782f8dcc0c827446a671a85a992ff973bca292890226cf2fe894bbf6dc08ae6f |
| `scripts/g2v3_m3_eval_latents_v2.py` | 73db9dafd33e49e963d48aad4797c6b0eb7856e5e8e847fbb3b8dc0e5109921b |
| `scripts/run_task0072_full.sh` | 1413a1f184a7206ae07283876e3921a87967e9e7a78bed6edee6890be405f473 |
| `results_s7/g2v3/full_model_A.pt` · `full_model_E.pt` | 2c2a271002b77192a9f75cdde98892ca054ab80dee9536751859957899028a31 · f3dc3e6b7b39ca55cf40c97eb989b99fd3d78c1b76f425b26f8137e54b116c6d |
| `results_s7/g2v3/decoder_A_v36.pt` = `g2v2/decoder_Am3_v36.pt` | 682d05c3638e3ffadfabb6f38fd6bd3d06a3e2004964f35cbeb60be9a410cd10 |
| `results_s7/g2v3/decoder_E_v36.pt` = `g2v2/decoder_Em3_v36.pt` | ba1d2788ec4f3fb38949f17cd215b1d42721c175f3a47d2659bc4dc573722e05 |
| `results_s7/g2v3/deq_backbone_A.pt` · `_E.pt` | fec12c8faab77c31276ba4d155aaf41f32243f33212793e92f34117e80a14cbf · 255ed2731098fa5081c540e456383fa73276e9cb73c589f9f39d0da0210f581b |
| `results_s7/g2v3/dev600_zstar_A.pt`(결함) · `dev600_zstar_A_v2.pt` | b5c7ac63d8b3c9a80e1ba03f531809b7ac8010858548f457ec7b95be55ca77cd · 372e42eea85b4f76aa8a008bbd7e04114c06378521bf543121aca2f7df9dd5a1 |
| `results_s7/g2v3/dev600_zsrc_E.pt` | 977b45f816a0b4efaf7cb594b6e4809ddb4127be15fe736e0a6990ed0c0ed7df |
| `results_s7/zstar_s7_a0.25_400.pt`(v36e가 암묵적으로 읽음) | bbe7ad3bd2758b4e3ef622ef7c5270307afa5831e4ec6748ca24a126078e79ca |
| `scripts/s5_eval_grader_v2.py`(원본 135행 이름 오류 정정) | 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13 |
| `scripts/s7_eval_score_p6_v36.py` · `s7_eval_gate_p6_v36e.py` | 547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd · a2755c1b591e12389c8efe167aa33a59a7ae6c0876234585381b6a129faa460a |
| 그 밖의 입력(원본 §6과 같음) | 원본 124~134·137~138행 |
| `results_s7/g2v3/dev600_zsrc_E_v2.pt`(대조용 재계산) | 43f798cc71eab4b533b091ea899fb3924aaa589fc9b4ced76ef6dec166c5c93c |
| `results_s7/g2v3/full_manifest_A.json` · `_E.json` | db2da3822084c1d5e0ecdc87d9de9de2021ed6a28ddb00b8e554020fbb4b57af · 16f86383e05bfff18168061feaec863d8e5e1ec1d7d74f78c37568dd0d5c0ce5 |
| `scripts/cc_g2v3_devA_rerun.sh` · `scripts/g2v3_fresh_prep.py` | 0ba2f75ea1df9f7c3e5a52bdb8d40f41115ce49cf7c62c7cd959d95c5e3edfda · 59889cd3cc829a5bb281e0d34a7a20899f824272f0911eb8db8547cbe741b423 |
| 문항 글 두 사본 `/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl`(학습) · `/home/shoon/cts_e1_s5/…`(잠재 변환) | 둘 다 f2cf899a4e26ec3ba645b2b825ebd3c7249400c0aca399e3a50e7df976c921df (`scratch/cc_g2v3_refix_235206.txt` [D]) |
| `results_s7/p6r/p6_gen_B1.jsonl`, `scripts/s7_train_decoder_v36.py`, `/home/shoon/checkpoints/m1_final_frozen.pt`, 라이브러리 판 | `PREREG_S7_G2v2` §6 행과 같음(4d30f021… · f2951054… · f2119bbe… · torch 2.11.0+cu128 / transformers 5.10.2 / numpy 2.4.4) |

- REPORT-0071의 "1행 SHA"는 재현되지 않는다(AUDIT-P013). 파일 SHA로만 인용한다.

## 7. 한계와 GOAL 인정

1. 3차 시도이며, 1·2차 결과와 진단, "G2 통과가 최종 목표" 지시(CTS-430) 아래 설계했다. 원본은 감사 없이 잠겼다(§0).
2. A·E‴ 비교의 비대칭(§3.1)과 개발 C의 장치·생성기 차이(§3.2)가 있다.
3. 학습된 사상의 고정점을 다시 풀지 않는다(§1). M3 상태의 G1 성질(수렴·변별)은 확립되지 않았다.
4. "새 문항"의 범위는 G2v2 §2·§7-3과 같다.
5. G2v3 충족을 GOAL.md G2 통과로 인정할지는 Principal 결정(CTS-431, 미수령). 받지 못하면 "개선 방법(M3)의 G2 기준 충족 여부"로만 보고한다.

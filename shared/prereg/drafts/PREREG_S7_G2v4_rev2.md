# PREREG_S7_G2v4 — G2 4차 시도(M7) 사전등록 (초안 rev2, B 재감사 전)

- **상태**: DRAFT rev2 (2026-10-11 17:2x KST). AUDIT-P014 FIX 5건과 권고 R-1~R-9를 반영했다.
  - rev1(SHA-256 4563782a189d1de73ff3d1e5ced94d5cb3bc747ac6c5a72f72619c754f5e381f)은 `shared/prereg/drafts/PREREG_S7_G2v4_rev1.md`에 보존했다.
  - B 재감사 PASS 뒤에는 제목과 이 상태 줄만 바꿔 잠근다. 잠금 이후 변경은 복사 + diff + B 감사로만 한다.
- **발행**: 회사 PC Claude Code(CC).
- **시도 번호**: G2 **4차 시도**.
  - 1차 `PREREG_S7_D_v3_8`(58b4a7f3…): 기준 미충족(게이트 4 −4.83 %p).
  - 2차 `PREREG_S7_G2v2`(4ea1d966…): 개발 선택 규칙에서 멈춤, 확증 없음.
  - 3차 `PREREG_S7_G2v3_1`(3d1a6ee0…): 확증 기준 미충족. 새 1,000문항에서 게이트 4 A−E‴ +1.20 %p, CI [−0.50, +2.90], 42/30, p 0.0973(`shared/reports/RESULT-G2v3-confirm.md`).
  - 1~3차 결과는 바꾸지 않는다.
- **근거**
  - Principal 지시(10/8, "G2 통과가 최종 목표, 근본 원인을 찾아 해결").
  - `shared/reports/DIAG-G2v4-rootcause-20261011.md` §1~§12.
  - PREREG_S7_G2v3_1 §5: "기준 미충족이면 4차는 새 사전등록과 이번 1,000문항을 제외한 새 문항으로만".
  - GRAPH.md 42행 H-DEC-7.

## 0. 절차 공개 — 본 것과 보지 않은 것 (FIX-1·FIX-2)

### 0.1 본 것 — 전부 개발 600 또는 학습 문항

정답 해설 NLL 진단(판정 아님)에서 본 설정과 결과 포인터는 아래와 같다. 원자료는 `shared/reports/` 아래에 있다.

| 설정 | 내용 | 결과 포인터 |
|---|---|---|
| M3 기준 | 3차 체크포인트 | `g2v4diag/diag_{A,E}.json` (DIAG §1·§2) |
| M4 | 글 가리기 0.5, M3에서 이어 600단계 | `g2v4diag/diag_m4_{A,E}.json` (§6) |
| 용량 | 처음부터, 글 가리기 1.0 | `g2v4diag/diag_cap_{A,E}.json` (§6) |
| LM LoRA | r16, 글 가리기 0.5 | 서버 `results_s7/g2v4lora/diag_lora_{A,E}.json` (§7) |
| 학습 문항 진단 | — | 서버 `results_s7/g2v4train/` (§8) |
| 경로 탐침 · soft token · 양성 대조 | — | §8·§9·§10, `g2v4posctl/diag.json` |
| V2 | A·E·E+LoRA | `g2v4v2/diag_{A,E,EL}.json` (§9) |
| Φ_θ 학습률 | A3·E3·E5 | `g2v4deqlr/diag_{A3,E3,E5}.json` (§11) |
| M7 복제 | Ah·Eh·Az·Ez | `g2v4m7rep/diag_{Ah,Eh,Az}.json`, `diag_Ez.json`(5c8fa506…) (§12) |

- **Ez 공개(FIX-2a)**: E, 글 가리기 1.0, 시드 2048, 15:09 완료.
  - 글 포함: 다음 − 자기 +0.0029(z 2.7). 글 없음: +0.0040(z 4.2). 자기 NLL 0.9010.
  - **글 가리기 1.0에서는 E도 문항 특이성이 생긴다.** "A만 문항 내용을 담는다"는 관찰은 글 가리기 0.5(Ah·Eh)에만 해당한다.
  - Az − Ez NLL −0.0324(z −14.6, A가 나은 문항 77.7 %).
  - 한 걸음 상대 이동 중앙값: Ah 0.127, Eh 0.106, Az 0.056, Ez 0.159.
- **정확도 평가 1회**: M7(Ah·Eh)의 개발 600 생성·채점.
  - strict 정답률: A 7.83 %, E‴ 4.83 %, C 0.33 %.
  - Gate_4: Δ +3.00 %p, CI [+0.50, +5.67], 39/21, p 0.0137. Gate_3: Δ +7.50 %p.
  - 원자료: `shared/reports/g2v4dev/dev_score.json`(e890c3c2…)·`dev_score.md`(b67af189…), 마스터 로그 `shared/reports/g2v4dev/cc_g2v4_dev_m7.log`(6d8522ad…).
  - 로그에 다음이 찍혀 있다:
    - 디코더 사본 SHA가 원본과 같음: `decoder_Ah.pt` = `decoder_M7A_v36.pt` 46977c2d…, `decoder_Eh.pt` = `decoder_M7E_v36.pt` d0c33599….
    - `[LOAD-GUARD] applied 436 DEQ tensors, unexpected 0` × 2, `[CHECK]` 600/600 × 2, `[SCORE] rc=0`.
  - 정확도로 평가한 설정은 이것 하나뿐이다.
- **Ah·Eh를 정확도 평가에 고른 기준(FIX-2b)**: 결정 시각은 14:2x(개발 평가 착수 14:22:57)다. 그 시각에 볼 수 있던 것은 Ah·Eh·Az 진단이고, Ez는 아직 끝나지 않았다. 기준은 두 가지였다.
  1. 글 가리기 0.5는 학습 때 글이 있는 경우를 포함한다. 평가 조건(글 + 잠재)과 맞는다. 글 가리기 1.0은 학습 중에 글을 한 번도 보지 않는다.
  2. Ah는 글 포함 자기 NLL이 가장 낮았다(0.8503, Az 0.8686).

### 0.2 설정을 고른 방법

- 투영기 V2, Φ_θ 학습률 3e-4, 글 가리기 0.5, 3000단계는 위 개발 NLL 진단을 보고 골랐다.
- 따라서 개발 결과에는 선택 편향(winner's curse)이 있을 수 있다. 개발 수치는 판정에 쓰지 않는다.

### 0.3 보지 않은 것

- §2의 새 1,758문항의 문제 글, 풀이, 텐서, 생성 결과.
- 선택 스크립트와 id 검사 스크립트는 q_idx와 정수 목록·키만 읽는다.

### 0.4 학습 산출물

- M7 체크포인트는 이 사전등록 전(10/11 13:23~14:13, `scripts/cc_g2v4_m7_rep.sh`)에 학습을 끝냈다.
- §6의 SHA로 고정하고 다시 학습하지 않는다.

### 0.5 실행 에이전트 AG

- `shared/ssh_log.csv` 359·360행이 AG 조회 두 건이다: 10:22:49 `ssh:cts-server echo '=== SYSTEM ==='; uptime; tmux ls…`(exit 2), 10:23:54 `ssh:poll_lora_pilot`(exit 0). 둘 다 조회 명령이다.
- 그 뒤의 접속 기록은 모두 CC(`cc_` 접두)다. AG는 이 시도의 파일·문항에 손대지 않았다.

### 0.6 원시 근거 묶음

- `scratch/cc_g2v4_evidence.txt`(a745295d…): 서버 실행 `scripts/cc_g2v4_evidence.sh`(4d978d95…, 10/11 16:55~16:56, 읽기 전용).
- `scratch/cc_g2v4_select_outputs.txt`(7e199c1f…): 선택 v1·v2·run1 점검 실행 출력의 원문 사본.

## 1. 방법 M7의 정확한 내용 (FIX-3)

- **고정점**: M2′(`cts/deq/zstar_inject_v2.py` 7024f1f2…; 주입 = z_src, α = g₀ = 0.25, 1000번 반복)로 학습 전 사상 Φ₀에서 계산한 z*′₀. 3차와 같다.
- **학습되는 사상**: Φ_θ(LoRA rank 8 q·v·o, `routing_proj`, γ).
  - 한 걸음 사상: T_θ(z) = (1 − α)·z + α·[Φ_θ(z) + g₀·z_src].
  - 학습과 평가는 같은 식과 같은 백본 로더를 쓴다: `load_gemma4_e4b` + `apply_paper_lora` + `m1_final_frozen.pt` — `g2v4_m4_pilot_train.py`·`g2v3_m3_eval_latents_v2.py`.
- **디코더**: `PrefixProjectorV2`(`scripts/g2v4_proj_v2.py`) + 고정된 순정 Gemma-4-e4b(58c4d7b5).
  - 처리 순서: z − center → 칸별 RMS 정규화 → `w_proj`(항등 초기화) → MLP 어댑터(0 초기화 잔차) → 칸별 RMS 정규화 × `out_rms`(초기값 = 실제 단어 임베딩 RMS 0.9981).
  - 바꾼 이유(`DIAG-G2v4` §8): v3.6 투영기는 실제 단어보다 25~40배 큰, 상수가 지배하는 벡터를 넣었다. 그래서 문항 정보가 2~4 %로 묻혔다.
- **center 차이(FIX-3c)**
  - A의 center = 학습 993문항 z*′₀의 칸별 평균(칸당 norm 312.6).
  - E의 center = 학습 993문항 z_src의 칸별 평균(191.9).
  - 투영기 입력은 시작 잠재가 아니라 T_θ(·)다. 그래서 두 조건 모두 입력이 center에서 벗어나 있다. 개발 한 걸음 상대 이동 중앙값은 A 12.7 %, E 10.7 %다.
  - 규칙("자기 시작 잠재의 평균")은 대칭이다. 그러나 정규화 효과가 두 조건에서 같은지는 재지 않았다.
- **학습**(완료, `scripts/g2v4_m4_pilot_train.py` f492da0f…)
  - 3000단계, 배치 4, 학습 문항 `configs/s7_train_993.json`.
  - lr: 투영기 1e-4, 어댑터·out_rms 3e-4, DEQ 3e-4. weight decay 0.01 / 0.01 / 0.001.
  - 글 가리기 0.5: 문항마다 확률 0.5로 문제 글을 빼고 자리표시 64칸만 남긴다.
  - 시드 2047, A·E 같음. `torch.manual_seed`와 `numpy.default_rng`를 같은 시드로 초기화한다.
  - GPU: A = GPU 5, E = GPU 6.
  - 명령 전문: `scripts/cc_g2v4_m7_rep.sh`의 `run Ah A $U5 0.1 0.5 2047` / `run Eh E $U6 0.0 0.5 2047`.
  - 학습 입력(명령에 없으므로 기본값을 읽는다):
    - `results_s7/g2v2/zstar_zsrcinj_train1000.pt`(005e622d50ec5efdfa309a6dd8e4663880194f9c10e13c276573e636050aee8a)
    - `results_s7/zsrc_train1000_P4.pt`(340ef5c2bfb718fc371c932746476555d8d5a2bcd54c7413acc40e35840e45a7)
- **λ_ift는 다르다(FIX-3a)**
  - A 0.1, E 0.0. 나머지 레시피는 같다.
  - 3차에서는 l_ift가 약 4.5e-5로 작았다. 이번에는 Φ_θ 학습률을 10배로 올려서 A의 최종 l_ift가 0.0164다(학습 로그 마지막 단계). 보조 손실이 A의 사상을 실질적으로 묶는다.
- **A·E 학습 동일성 근거(FIX-3d)**
  - 두 로그 모두 `[DATA] Filtered 993 / 993`이다.
  - 매니페스트 `n_text_dropped`는 A·E 모두 6061이다(12,000 문항-단계 중). 가리기 횟수가 같으므로, 같은 난수열로 같은 순서로 가렸다는 정황 근거다.
  - 학습 가능 매개변수: DEQ 218텐서·3,980,801개, 투영기 6텐서·19,665,921개(같음).
  - 최종 l_task: A 1.1973, E 1.2021.
  - 매니페스트 SHA: A dbdf21bf…, E 0264a337….
- **백본 로더(FIX-3e)**
  - `DIAG-G2v4` §7은 "4차 설계에서 정본 로더(`harness_A_v1.get_model`)로 통일"을 계획했다. M7은 이 계획을 따르지 않았다.
  - 이유: M7 체크포인트는 그 계획을 적기 전부터 쓰던 학습 스크립트(3차와 같은 로더)로 이미 학습됐다. 학습과 평가에서 같은 로더를 써야 학습된 사상이 학습 때와 같게 작동한다.
  - 영향: z*′₀·z_src는 정본 로더로, 한 걸음은 위 로더로 계산한다. A·E 모두 같은 조건이다.
- **조건**
  - A: 디코더 `decoder_M7A` + T_θ,A(z*′₀).
  - E‴: 디코더 `decoder_M7E` + T_θ,E(z_src).
  - C: 순정 Gemma, 글만.
  - B1(보조): A 디코더 + 한 칸 민 A 잠재.
- **주장 범위**: "M2′ 고정점에서 출발한 학습된 한 걸음 + V2 연결부(학습 λ_ift 0.1)"가 "z_src에서 출발한 같은 연결부·학습(λ_ift 0)"보다 정답률을 높이는가.
  - 게이트 4는 **시작점 효과와 λ_ift 효과를 가르지 못한다**. 근거는 개발 진단이며 판정 아님이다.
  - 같은 A 디코더에 입력만 T_θ,A(z_src)로 바꾸면 NLL이 0.8514다. T_θ,A(z*′₀)의 0.8503과 0.0011 차이뿐이다(`diag_Ah.json` 요약 `zsrc_k1_nll`·`zstar_k1_nll`).
  - 반면 A − E‴ NLL은 −0.0236이다(`diag_Ah`·`diag_Eh`). 즉 차이의 대부분은 평가 시작점이 아니라 학습 경로(시작점 + λ_ift)에서 온다.

## 2. 문항 (FIX-1b·c)

- **학습**: `configs/s7_train_993.json`(3088a153…).
- **개발**(열람함): `results_s7/p6_item_order.json`(6c76ddb7…).
- **확증**: `configs/g2v4b_fresh.json` — **1,758문항**, SHA-256 6e68e34c510593fb017d7dab5296c60ac506ce28d1ddeb3701231a69f696ce80.
  - 샤드 k = 정렬 목록[k::3]:
    - shard0 f8f305a1725f341d89fb38366f3d93dcdcee7c2b912b8da08dd3f010113a2253
    - shard1 c5a3ce59f2eb9f51a64b3b53c597d384fa5464d8400c8277f831bdc33f965f90
    - shard2 63935a091c9dc9a08ebe8905ee5be523990f7a730fcc1e63bd5573d526965cda
  - 3차 잠금본 `PREREG_S7_G2v3_1.md` 73행이 결과 전에 선언한 4차 풀(2,758 − 1,000)과 같다. B가 로컬 입력만으로 재구성해 바이트 단위로 일치함을 확인했다(AUDIT-P014 #6).
  - 선택: `scripts/g2v4_select_fresh_v2.py`(e9d133b5…), 서버 실행 10/11 16:5x.
    - 규칙: 제외 = `g2v2_exclude.json`(2,242) + `g2v2_fresh1000.json`(3차 사용분 1,000) + 새 서버 json 전수 검사(정수 3,240, 후보와 겹침 0). 남은 것 **전부**를 쓴다(표본 추출 없음).
    - 산출: `configs/g2v4b_exclude.json` 91a2cdc8…, `configs/g2v4b_server_json_union.json` cd7c8f1f…(사본 `shared/reports/g2v4dev/`).
  - 실행 출력 원문: `scratch/cc_g2v4_select_outputs.txt`.
- **v1 선택 공개**
  - v1(`g2v4_select_fresh.py` be018b3a…)은 1,234문항이었다(`configs/g2v4_fresh.json` 302dffc4…, 보존).
  - 차이 524는 보관된 G2v2 첫 선택 목록에만 있는 문항이다. 보관 파일: 서버 `configs/archive/g2v2_*_run1_48b7f09d.json` 다섯 개, 서버 시각 10/08 15:24:22. 로컬 보관본 시각은 15:17:56. 다섯 개 가운데 `g2v2_exclude_run1`은 이미 제외 집합 쪽이다.
  - v2는 패턴 `_run1_48b7f09d`만 검사에서 뺐다.
- **524문항 미사용 직접 근거(FIX-1c)**
  - 서버 `scripts/g2v4_idscan.py`(f0468a13…). `cts_e1_s5`·`cts_e1`의 `results_s7/`·`raw_s7/`·`scratch/`·`results/` 아래를 검사했다: 모든 `.jsonl`의 `id`·`q_idx`·`question_idx`·`qid`와 모든 `.pt` dict 키(`q\d{4}`·정수).
  - 결과: `.jsonl` 72개, `.pt` 374개, 크기 때문에 건너뛴 파일 0. **1,758과 겹치는 파일 0**(`[RESULT] PASS (0 hits)`).
  - 양성 대조: `g2v4_idscan_v2.py`(79e8b406…)로 3차 사용 목록 `g2v2_fresh1000.json`을 같은 방법으로 검사했다. 23개 파일에서 적중했다(예: `results_s7/g2v3/fresh_A_latent.pt`·`p6r/g2v3fresh_merged/p6_gen_*.jsonl` 1000/1000). 검사기가 사용 흔적을 실제로 잡는다는 뜻이다.
  - run1 점검(grep·find) 재실행: 파일 이름 참조 0건(선택 v2 자신 제외). 10/08 15:00~16:10에 생긴 결과 파일은 `results_s7/p6r/smoke/v36e_smokeA.jsonl`뿐이고, 이 smoke는 학습 문항을 썼다. 근거: `scratch/cc_g2v4_evidence.txt` [B2]·[C].
  - 따라서 v2(1,758)를 쓴다(AUDIT-P014 판단 조건 충족).

## 3. 개발 근거 (완료, 판정 아님)

- 실행: `scripts/cc_g2v4_dev_m7.sh`(45e6a514…), v36f, 10/11 14:22~16:27.
  - 개발 600을 앞 300·뒤 300으로 나눴다. GPU 5는 앞 절반, GPU 6은 뒤 절반을 맡았고, 같은 절반에서 A → E1 순서로 생성했다.
  - 개발 잠재: A 4b6c84ac…, E 81f28a64…. 생성: A e8e89599…, E1 8de4066a…. 각 600행이고 id = 개발 목록.
- 개발 C는 1차 P6R 파일 `results_s7/p6r/p6_gen_C.jsonl`(24c82d97…, v36d·GPU 0)이다(R-4). 개발 B1 행은 옛 P6R 파일이라 쓰지 않는다.
- 이 사전등록은 개발 결과를 본 뒤에 확증을 결정한다. 개발용 선택 규칙은 따로 없다.

## 4. 확증 평가 — 명령 전문 `scripts/cc_g2v4_confirm.sh` (FIX-5)

0. **기동**(R-6): `tmux new -d -s cc_g2v4_confirm 'bash scripts/cc_g2v4_confirm.sh > raw_s7/cc_g2v4_confirm_master.log 2>&1'`. 접속은 `cts_ssh.py`로 한다.
1. **사전 점검.** 어느 하나라도 걸리면 계산 없이 중단한다.
   - **출력 존재 검사를 모든 `mkdir` 전에 한다.** 대상: `results_s7/p6r/g2v4fresh`, `results_s7/p6r/g2v4fresh_merged`, `results_s7/g2v4/fresh_zsrcinj_shard{0,1,2}.pt`, `fresh_zsrcinj_1758.pt`, `fresh_zsrc_1758.pt`, `fresh_A_latent.pt`, `fresh_E_latent.pt`. 하나라도 있으면 중단한다. 이유: v36f와 z 실행기는 기존 출력에 이어 쓴다.
   - **SHA 대조.** "실행에 쓰는 파일"(§6 표 ①)을 모두 `chk`로 대조한다. 실행 중 import·호출되는 `s7_train_decoder_v36.py`, `g2v3_m3_full_train.py`, `check_p6_decoder_version.sh`도 포함한다. 학습·선택·개발 스크립트(§6 표 ②)는 실행에 쓰지 않으므로 이 사전등록과 STATE 기록으로 고정한다.
   - **GPU 5·6·7이 비어 있어야 한다**(메모리 < 1 GiB, 프로세스 0).
2. **새 문항 M2′ z*′₀**
   - 샤드 k → GPU: 0 → GPU 5 `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a`, 1 → GPU 6 `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c`, 2 → GPU 7 `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b`.
   - 명령: `python scripts/g2v2_zstar_run.py --items configs/g2v4b_fresh_shard{k}.json --inject zsrc --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --config configs/zstar_canonical.yaml --out results_s7/g2v4/fresh_zsrcinj_shard{k}.pt`
   - 병합·변환: `scripts/g2v4_fresh_prep.py`(ed79108f…).
     - 출력: `results_s7/g2v4/fresh_zsrcinj_1758.pt`(키 = 목록, 여분 0)와 `fresh_zsrc_1758.pt`(`{"z_star": z_src}`).
     - 변환 max|Δ| = 0이 아니면 중단한다. r_1000 < 1e-2 비율을 보고한다.
3. **학습된 한 걸음**(GPU 7, 차례로)
   - A: `g2v3_m3_eval_latents_v2.py --mode A --items configs/g2v4b_fresh.json --deq_ckpt results_s7/g2v4m7rep/deq_Ah.pt --zstar …/fresh_zsrcinj_1758.pt --zsrc …/fresh_zsrc_1758.pt --out results_s7/g2v4/fresh_A_latent.pt`
   - E: `--mode E --deq_ckpt results_s7/g2v4m7rep/deq_Eh.pt --zsrc …/fresh_zsrc_1758.pt --out results_s7/g2v4/fresh_E_latent.pt`
   - **강제 점검**(R-7): 두 로그에 `applied 436 DEQ tensors, unexpected 0`이 없으면 중단한다.
4. **생성**: `scripts/s7_eval_gate_p6_v36f.py`, 샤드별로 같은 GPU에서 A → C → E1 → B1 순서.
   - 공통 인자: `--items configs/g2v4b_fresh_shard{k}.json --order_sha <샤드 SHA>`.
   - A: `--cond A --decoder_ckpt results_s7/g2v2/decoder_M7A_v36.pt --zstar results_s7/g2v4/fresh_A_latent.pt`
   - C: `--cond C`
   - E1: `--cond E1 --decoder_ckpt results_s7/g2v2/decoder_M7E_v36.pt --zsrc results_s7/g2v4/fresh_E_latent.pt`
   - B1: `--cond B --shift 1 --decoder_ckpt …/decoder_M7A_v36.pt --zstar …/fresh_A_latent.pt`
   - 출력 `results_s7/p6r/g2v4fresh/gen_<cond>_s{k}.jsonl`.
   - **강제 점검**(R-7): 세 샤드의 A·E1·B1 로그 9개에 `[PROJ-V2]`가 없으면 중단한다.
5. **병합·점검·채점**: 빈 새 폴더 `results_s7/p6r/g2v4fresh_merged/`에 `p6_gen_{A,B1,C,E1}.jsonl`을 만든다.
   - 채점 전 점검: 네 파일 각 1,758행, 고유 id 1,758, id 집합 = `g2v4b_fresh.json`, JSON 오류 0.
   - 채점 명령: `s7_eval_score_p6_v36.py --gen_A … --gen_B1 … --gen_C … --gen_E1 … --gen_dir results_s7/p6r/g2v4fresh_merged --order configs/g2v4b_fresh.json --out_json …/gate_table.json --out_md …/gate_table.md`
6. **판정**(R-1): G2v4 = 게이트 2 ∧ 게이트 3 ∧ 게이트 4. `gate_table.json`의 JSON 값(반올림된 저장값)만 쓰고 md 표는 쓰지 않는다.
   - 게이트 2: `gates.Gate_2.strict.delta ≥ −0.02`.
   - 게이트 3: `gates.Gate_3.strict.mcnemar_one_sided_p < 0.025` ∧ `gates.Gate_3.strict.ci_95[0] > 0`.
   - 게이트 4: `gates.Gate_4.strict.mcnemar_one_sided_p < 0.025` ∧ `gates.Gate_4.strict.ci_95[0] > 0`.
   - 공통: strict, 채점기 `scripts/s5_eval_grader_v2.py`(90cc93e4…), 채점 스크립트 547e1d91…, 부트스트랩 10,000·시드 2026, 정확 McNemar 단측.
   - 게이트 1(A − B1)은 보조다.
7. **보고 항목(판정 미사용, FIX-4a)**: 채점과 무관하게 잠재 텐서만 읽어 계산한다. `python scripts/g2v4_proximity.py`(bdd3730c…)로 다음 두 분포(n·중앙값·p10·p90·평균)를 보고한다.
   - A: ‖T_θ,A(z*′₀) − z*′₀‖/‖z*′₀‖
   - E: ‖T_θ,E(z_src) − z_src‖/‖z_src‖
- **실행 위생**: tmux(R7-4), CPU 단계는 `nice -n 19`·`ulimit -v 62914560`(R2-5), 모든 접속은 `cts_ssh.py`(R7-1).

## 5. 갈림길·통계

- 판정은 새 1,758문항으로 **한 번**만 한다. 결과를 본 뒤 조건·문항·생성·채점·게이트·가중치를 바꾸지 않는다.
- **중단 시 재개**(R-2): `gate_table.json`이 생기기 전에만 허용한다.
  - 재개는 §4와 같은 명령을 실패한 단계부터 실행하는 재개 스크립트로 한다. 그 단계 출력의 이어 쓰기만 허용한다.
  - 재개 스크립트의 확증 스크립트 대비 diff를 실행 전에 STATE에 기록하고 B 확인을 받는다.
  - 재개 사실·시각·행 수를 보고한다.
- **도구 결함**(R-3): 결과와 무관하다고 볼 수 있는 결함은 사전 점검·강제 점검·병합 assert·채점표 생성 전에 드러난 것뿐이다.
  - 그 경우 RULES R3-3에 따라 고친 뒤 같은 평가를 다시 하고, 다시 한 결과를 판정으로 쓴다. 두 결과를 모두 보고한다(개정·B 감사).
  - 채점표를 본 뒤 찾은 결함은 B 감사 전에 다시 실행하지 않는다.
- **반복 시도 보정 참고값**: 게이트 3·4의 p × 4를 함께 보고한다. 판정 기준은 사전등록대로 p < 0.025다.
- **검정력 참고**(판정 아님): 개발의 불일치율 10 %와 b 비율 0.65가 유지된다고 가정한다.
  - 기대 불일치 쌍 약 176, 기대 z 약 4.0.
  - 효과가 절반이면 통과 확률은 약 51 %.
- **기준 미충족이면**: 이 데이터 풀(math_train_prompts_5000)에는 미열람 문항이 남지 않는다. 5차는 새 데이터 출처가 필요하므로 Principal 결정 사항이다.
- **가설**: H-DEC-7(M7)은 GRAPH.md 42행에 등록했다. 기각 조건은 확증 미충족이다.

## 6. 고정 파일 (서버 SHA-256)

① **실행에 쓰는 파일** — `cc_g2v4_confirm.sh`가 대조한다.

| 파일 | SHA-256 |
|---|---|
| `results_s7/g2v2/decoder_M7A_v36.pt` (= `g2v4m7rep/decoder_Ah.pt`) | 46977c2db533888dc13a8defe21ac287b26bd7ff1e08bb83eb22a4ca4bb99670 |
| `results_s7/g2v2/decoder_M7E_v36.pt` (= `g2v4m7rep/decoder_Eh.pt`) | d0c33599c9586d3ed131877a2439ee09d5d48804a0408965daaa0238705afd5a |
| `results_s7/g2v4m7rep/deq_Ah.pt` | b43de6b0eeb0a16a5f04762ac9c6a95979e504bc48f8b52249213490d6485cf7 |
| `results_s7/g2v4m7rep/deq_Eh.pt` | 7e9dc5feb1e563996e90cc825c3567e54caadb86f5a9a25ce03435df37be9788 |
| `configs/g2v4b_fresh.json` · 샤드 3개 | 6e68e34c… · f8f305a1… · c5a3ce59… · 63935a09… (§2) |
| `scripts/g2v4_proj_v2.py` | 7cabcef2e13bb524a6815add5198853f21e8437fd04d018a6cef2227161df47a |
| `scripts/s7_eval_gate_p6_v36f.py` (v36e a2755c1b… + V2 적재, 18행 변경) | fd6c2d03e3ed18d1597950edc79a01af8331d192c9f84b821a5b5bcf3f71cfe1 |
| `scripts/g2v3_m3_eval_latents_v2.py` | 73db9dafd33e49e963d48aad4797c6b0eb7856e5e8e847fbb3b8dc0e5109921b |
| `scripts/g2v4_fresh_prep.py` | ed79108f9d8376467ea18781f6407c1140d59a69f7f3eaf22c98f929f311db54 |
| `scripts/g2v2_zstar_run.py` · `cts/deq/zstar_inject_v2.py` · `configs/zstar_canonical.yaml` | 687d4480a622c845727a260ebcf49e052142269f10310b016fa82875323b4f2d · 7024f1f2d9ff534c3ec43251ac9c8177fa0e384717e8752b330481f6aed91650 · bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157 |
| `scripts/s7_eval_score_p6_v36.py` · `scripts/s5_eval_grader_v2.py` | 547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd · 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13 |
| `scripts/s7_train_decoder_v36.py` (v36f가 import) | f2951054416c7b1593075c6a94e3c1ab7e7e7be516f078d540b5bd9ce5bf3ea9 |
| `scripts/g2v3_m3_full_train.py` (`g2v4_proj_v2.py`가 import) | 36e0f18addc8d46f6d8bf376cd95e89cf9eb75a1f15c4e8b714141efe5a49c99 |
| `scripts/check_p6_decoder_version.sh` (서버본. 로컬본 cc685dcb…와는 줄바꿈 등 바이트가 다르며, 실행은 서버본) | 1fa95f72d5e5cea2ad1836dd458544190467f0cfb1daa14e41f6459cb51d870b |
| `shared_prereg_PREREG_S7_G2v4.md` (잠금본) | 잠금본 SHA (스크립트 17행 자리표시에 넣음) |

② **기록용 파일** — 실행에 쓰지 않으며, 이 문서와 STATE로 고정한다.

| 파일 | SHA-256 |
|---|---|
| `scripts/g2v4_m4_pilot_train.py` (학습 실행본) | f492da0f1c35a3088397d2fec1748e9b56ed82bc3b371fe33cc0846706092ab0 |
| `scripts/cc_g2v4_m7_rep.sh` (학습 명령, 로컬본) · `scripts/cc_g2v4_dev_m7.sh` | 9fc2d657b0e808ce37bf1df9f0dae92dcf41a6f11e221508564fac99f8816888 · 45e6a5143171a274c71751c0273f6a3196e42a9511585f976a7811ce98c8ba56 |
| 학습 입력 `zstar_zsrcinj_train1000.pt` · `zsrc_train1000_P4.pt` | 005e622d… · 340ef5c2… (§1) |
| 매니페스트 `manifest_Ah.json` · `manifest_Eh.json` | dbdf21bfa9b013468013dadaeb88471fb92787770d876a8646cbca0e95a348d5 · 0264a337a72ba6dd4021012739a4850c440f72f55c411d99866ce9bda5833124 |
| `scripts/g2v4_select_fresh_v2.py` · v1 `g2v4_select_fresh.py` | e9d133b550a92322b38fc7f66ab440d37a472a5cf4587f1c98e5824066c470f3 · be018b3a8ee71adc411cb61af738fe7cc85caed3fa7868b71b6ea094968cb152 |
| `scripts/g2v4_idscan.py` · `g2v4_idscan_v2.py` · `cc_g2v4_evidence.sh` | f0468a138b95231e3384860ecfe99f61db1c5468dbf941398dfce073b4fce988 · 79e8b40695b9613f30b86aae7bda62e22b522dc868ac410602069b62323ec42e · 4d978d951bec2afb5d1b438134ca77355fa4cc2bd4751f69bbd71f041df94f41 |
| `scripts/g2v4_proximity.py` (§4-7 보고용) | bdd3730cf1b97432a9f73782e1689af3907a7f3c62ba0ea2d4247d0f9c0b1ca5 |
| `scripts/g2v4_diag_iterate.py` (개발 NLL 근거, R-9) | 85b4e94f8523d15d4d522084cd0473d3446a702df4b31e52853346bdb7366126 (DIAG 머리의 120dc83d…는 앞선 판) |

③ **확증 스크립트 `scripts/cc_g2v4_confirm.sh`** (FIX-5c, 순환 해소)
- 잠금 전 SHA-256: 75fd4a0518731741b303c02a1ce73a8bcddaef4e3b525e35e740d929c23b1f64. 17행이 `__PREREG_SHA__` 자리표시인 상태다.
- 실행본 = 이 파일에서 17행 자리표시만 잠금본 SHA로 바꾼 것(diff 1행). 실행본 SHA는 STATE에 기록한다.

## 7. 한계·GOAL (FIX-4)

- **A 입력과 GOAL의 "자기 z*"**: GOAL.md 8~12행의 "자기 z*"는 정본 풀이기(G1)의 z*다. 이번 A 입력은 그것이 아니다. M2′(z_src 주입)의 z*′₀에 학습된 한 걸음을 더한 것이다.
  - 개발 중앙값으로 z*′₀에서 12.7 % 떨어져 있다.
  - 확증 문항의 거리는 §4-7로 보고한다.
- **GOAL.md의 "held-out 600"**: GOAL은 G2를 "held-out 600에서"로 적는다. 1~3차처럼 이번에도 새 문항으로 확증한다.
  - 개선판 통과를 GOAL의 G2 통과로 인정할지는 Principal 결정 대기다(CTS-431).
  - **CTS-431을 받지 못하면 "M7의 G2 기준 충족 여부"로만 보고한다.**
- **λ_ift 혼입**: §1 주장 범위 참고. 게이트 4가 충족되더라도 "DEQ 시작점 때문"과 "A만 받은 IFT 보조 손실 때문"을 가를 수 없다. 결과 문서에 그대로 적는다.
- **글 가리기 1.0의 E 특이성**: Ez에서는 E도 문항 특이성이 유의했다(§0.1). 문항 특이성이 A에만 생긴다는 관찰은 설정에 따라 다르다.
- **단일 시드**: A·E 디코더는 시드 하나로 학습했다. 개발 NLL에서 A − E 방향은 두 설정·두 시드에서 같았다(Ah−Eh −0.0236, Az−Ez −0.0324).
- **백본 로더 차이**: §1 참고. 같은 문항에서 두 로더의 z_src는 cos 0.70~0.91이다(`DIAG-G2v4` §7).
- **주장 범위**: 학습된 사상으로 고정점을 다시 푸는 방법이 아니다. 효과는 "학습된 한 걸음과 V2 연결부의 조합"에 대한 것이다.
- **다중 시도**: 4차이므로 p × 4 참고값을 함께 해석한다.
- **표시 문구**(R-8): v36f 머리말의 "v3.6e·Held-out 600"과 채점 md 제목의 "FULL 600"은 표시 문구일 뿐이다. 실제 N = 1,758이며, 결과 문서에 그렇게 적는다.
- **학습/평가 모드**(R-9): AUDIT-P013 R-1(`routing_proj` spectral norm의 u·v가 train/eval 모드에서 다를 수 있음)은 이번에 따로 점검하지 않았다. 학습과 평가가 같은 로더·같은 eval 경로를 쓰는 개발 진단에서 LOAD-GUARD는 436·0이었다. 점검하지 않았다는 사실을 공개한다.

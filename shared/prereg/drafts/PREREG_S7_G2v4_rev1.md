# PREREG_S7_G2v4 — G2 4차 시도(M7) 사전등록 (초안, B 감사 전)

- **상태**: DRAFT rev1 (2026-10-11 17:0x KST). B 감사 PASS 뒤 제목·상태 줄만 바꿔 잠근다. 잠금 이후 변경은 복사 + diff + B 감사로만 한다.
- **발행**: 회사 PC Claude Code(CC).
- **시도 번호**: G2 **4차 시도**.
  - 1차 `PREREG_S7_D_v3_8`(58b4a7f3…): 기준 미충족(게이트 4 −4.83 %p).
  - 2차 `PREREG_S7_G2v2`(4ea1d966…): 개발 선택 규칙에서 멈춤, 확증 없음.
  - 3차 `PREREG_S7_G2v3_1`(3d1a6ee0…): 확증 기준 미충족(새 1,000문항, 게이트 4 A−E‴ +1.20 %p, CI [−0.50, +2.90], 42/30, p 0.0973; `shared/reports/RESULT-G2v3-confirm.md`).
  - 1~3차 결과는 바꾸지 않는다.
- **근거**: Principal 지시(10/8, "G2 통과가 최종 목표, 근본 원인을 찾아 해결"), `shared/reports/DIAG-G2v4-rootcause-20261011.md` §1~§12, PREREG_S7_G2v3_1 §5("기준 미충족이면 4차는 새 사전등록과 이번 1,000문항을 제외한 새 문항으로만").

## 0. 절차 공개 — 본 것과 보지 않은 것

1. **본 것(전부 개발 600 또는 학습 문항)**
   - 3차 확증 결과 전체.
   - 개발 600 정답 해설 NLL 진단 여러 번: M3·M4·용량·LM LoRA·V2·Φ_θ 학습률·M7 복제(Ah·Eh·Az·Ez) — `DIAG-G2v4` §1~§12.
   - 경로 탐침·soft token·양성 대조·짝 맞춤 점검.
   - **M7(Ah·Eh) 개발 600 생성·채점 1회**: strict A 7.83 %, E‴ 4.83 %, C 0.33 %; Gate_4 Δ +3.00 %p(CI [+0.50, +5.67], 39/21, p 0.0137); Gate_3 Δ +7.50 %p. 출처: `results_s7/p6r/g2v4dev/dev_score.json`(e890c3c2…). 정확도로 평가한 설정은 이것 하나뿐이다.
2. **설정을 고른 방법**
   - 투영기 V2, Φ_θ 학습률 3e-4, 글 가리기 0.5, 3000단계는 개발 600 NLL 진단을 보고 골랐다.
   - 따라서 개발 결과에는 선택 편향(winner's curse)이 있을 수 있다. 개발 수치는 판정에 쓰지 않는다.
3. **보지 않은 것**
   - §2의 새 1,758문항의 문제 글·풀이·텐서·생성.
   - 선택 스크립트는 q_idx와 json 정수 목록만 읽는다.
4. **학습 산출물**
   - M7 체크포인트는 이 사전등록 전(10/11 13:23~14:13)에 `scripts/cc_g2v4_m7_rep.sh`로 학습을 끝냈다. §6의 SHA로 고정하고 다시 학습하지 않는다.
5. **실행 에이전트 AG**
   - AG는 10/11 10:22·10:23에 서버를 조회만 했다(`shared/ssh_log.csv`).
   - 이 시도의 파일·문항에는 손대지 않았다.

## 1. 방법 M7의 정확한 내용

- **고정점**: M2′(`cts/deq/zstar_inject_v2.py` 7024f1f2…, 주입 = z_src, α = g₀ = 0.25, 1000번 반복)로 학습 전 사상 Φ₀에서 계산한 z*′₀. 3차와 같다.
- **학습되는 사상**: Φ_θ(LoRA rank 8 q·v·o, `routing_proj`, γ).
  - 한 걸음 사상: T_θ(z) = (1 − α)·z + α·[Φ_θ(z) + g₀·z_src].
  - 학습과 평가에서 같은 식, 같은 백본 로더를 쓴다: `load_gemma4_e4b` + `apply_paper_lora` + `m1_final_frozen.pt` — `g2v4_m4_pilot_train.py`·`g2v3_m3_eval_latents_v2.py`.
- **디코더**: `PrefixProjectorV2`(`scripts/g2v4_proj_v2.py`) + 고정된 순정 Gemma-4-e4b(58c4d7b5).
  - 처리 순서: z − center(학습 시작 잠재의 칸별 평균) → 칸별 RMS 정규화 → `w_proj`(항등 초기화) → MLP 어댑터(0 초기화 잔차) → 칸별 RMS 정규화 × `out_rms`(초기값 = 실제 단어 임베딩 RMS 0.9981).
  - 바꾼 이유: v3.6 투영기는 실제 단어보다 25~40배 큰 상수 지배 벡터를 넣어 문항 정보를 2~4 %로 묻었다(`DIAG-G2v4` §8).
- **학습**(완료, `scripts/g2v4_m4_pilot_train.py` f492da0f…):
  - 3000단계, 배치 4, 학습 문항 `configs/s7_train_993.json`.
  - lr: 투영기 1e-4, 어댑터·out_rms 3e-4, DEQ 3e-4. weight decay 0.01 / 0.01 / 0.001.
  - 글 가리기 0.5: 문항마다 확률 0.5로 문제 글을 빼고 자리표시 64칸만 남긴다.
  - 시드 2047(A·E 같음; `torch.manual_seed`와 글 가리기 난수 `numpy.default_rng`를 같은 시드로 초기화. 투영기 초기값·가리기 순서가 실제로 같은지는 확인하지 않았다).
  - λ_ift: A 0.1, E 0.0 (3차와 같은 비대칭).
  - GPU: A = GPU 5, E = GPU 6.
  - 명령 전문은 `scripts/cc_g2v4_m7_rep.sh`의 `run Ah A $U5 0.1 0.5 2047` / `run Eh E $U6 0.0 0.5 2047`.
- **조건**
  - A: 디코더 `decoder_M7A` + T_θ,A(z*′₀).
  - E‴: 디코더 `decoder_M7E` + T_θ,E(z_src).
  - C: 순정 Gemma, 글만.
  - B1(보조): A 디코더 + 한 칸 민 A 잠재.
- **주장 범위**: "M2′ 고정점에서 출발한 학습된 한 걸음 + V2 연결부"가 "z_src에서 출발한 같은 학습"보다 정답률을 높이는가.

## 2. 문항

- **학습**: `configs/s7_train_993.json`(3088a153…).
- **개발**(열람함): `results_s7/p6_item_order.json`(6c76ddb7…).
- **확증**: `configs/g2v4b_fresh.json` — **1,758문항**, SHA-256 6e68e34c510593fb017d7dab5296c60ac506ce28d1ddeb3701231a69f696ce80.
  - 샤드 k = 정렬 목록[k::3]:
    - shard0 f8f305a1725f341d89fb38366f3d93dcdcee7c2b912b8da08dd3f010113a2253
    - shard1 c5a3ce59f2eb9f51a64b3b53c597d384fa5464d8400c8277f831bdc33f965f90
    - shard2 63935a091c9dc9a08ebe8905ee5be523990f7a730fcc1e63bd5573d526965cda
  - 선택 스크립트: `scripts/g2v4_select_fresh_v2.py`(e9d133b5…), 서버 실행 10/11 16:5x.
  - 규칙: 제외 = `g2v2_exclude.json`(2,242) + `g2v2_fresh1000.json`(3차에서 사용, 1,000) + 새 서버 json 전수 검사(정수 3,240, 후보와 겹침 0). 남은 것 **전부**를 쓴다(표본 추출 없음).
  - 제외 파일 `configs/g2v4b_exclude.json` 91a2cdc8…, 검사 결과 `configs/g2v4b_server_json_union.json` cd7c8f1f….
  - 확인: 사용 문항과 겹침 0, 제외와 겹침 0.
- **v1 선택 공개**: v1(`g2v4_select_fresh.py` be018b3a…)은 1,234문항을 골랐다(`configs/g2v4_fresh.json` 302dffc4…, 보존).
  - 차이 524문항은 보관된 G2v2 첫 선택 목록(`configs/archive/g2v2_*_run1_48b7f09d.json`, 10/08 15:24)에만 있다.
  - 그 목록은 어떤 스크립트·로그·출력에서도 쓰이지 않았다. 서버 확인: grep 0건. 10/08 15:00~16:10에 생긴 결과 파일은 smoke 1개뿐이고, smoke는 학습 문항 14·29를 썼다.
  - 그래서 v2는 그 4개 파일만 검사에서 뺐다. B 감사가 이 판단을 거부하면 v1(1,234)을 쓴다.

## 3. 개발 근거 (완료, 판정 아님)

- 실행: `scripts/cc_g2v4_dev_m7.sh`(45e6a514…), v36f, GPU 5·6, 10/11 14:22~16:27.
  - 개발 잠재: A 4b6c84ac…, E 81f28a64…. LOAD-GUARD 436·0.
  - 생성: A e8e89599…, E1 8de4066a…, 각 600행·id = 개발 목록.
- 결과는 §0-1과 같다. 개발 B1 행은 옛 P6R 파일이라 쓰지 않는다.
- 이 사전등록은 개발 결과를 본 뒤에 확증을 결정한다. 개발에 별도 선택 규칙은 없다.

## 4. 확증 평가 — 명령 전문 `scripts/cc_g2v4_confirm.sh`

1. **고정 확인**: §6의 모든 SHA를 서버에서 대조한다. 하나라도 다르면 중단한다.
   - `results_s7/p6r/g2v4fresh_merged`나 z·잠재 출력이 이미 있어도 중단한다.
   - GPU 5·6·7이 비어 있지 않아도 중단한다(메모리 < 1 GiB, 프로세스 0).
2. **새 문항 M2′ z*′₀**
   - 샤드 k → GPU: 0 → GPU 5 `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a`, 1 → GPU 6 `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c`, 2 → GPU 7 `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b`.
   - 명령: `python scripts/g2v2_zstar_run.py --items configs/g2v4b_fresh_shard{k}.json --inject zsrc --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --config configs/zstar_canonical.yaml --out results_s7/g2v4/fresh_zsrcinj_shard{k}.pt`
   - 병합·변환: `scripts/g2v4_fresh_prep.py`(ed79108f…; 3차 `g2v3_fresh_prep.py`에서 경로만 바꿈).
     - 출력: `results_s7/g2v4/fresh_zsrcinj_1758.pt`(키 = 목록, 여분 0)와 `fresh_zsrc_1758.pt`(`{"z_star": z_src}`).
     - 변환 max|Δ| = 0이 아니면 중단한다. r_1000 < 1e-2 비율을 보고한다.
3. **학습된 한 걸음**(GPU 7, 차례로)
   - A: `g2v3_m3_eval_latents_v2.py --mode A --items configs/g2v4b_fresh.json --deq_ckpt results_s7/g2v4m7rep/deq_Ah.pt --zstar …/fresh_zsrcinj_1758.pt --zsrc …/fresh_zsrc_1758.pt --out results_s7/g2v4/fresh_A_latent.pt`
   - E: `--mode E --deq_ckpt results_s7/g2v4m7rep/deq_Eh.pt --zsrc …/fresh_zsrc_1758.pt --out results_s7/g2v4/fresh_E_latent.pt`
   - `[LOAD-GUARD]` 행과 두 파일 SHA를 보고한다.
4. **생성**: `scripts/s7_eval_gate_p6_v36f.py`, 샤드별로 같은 GPU에서 A → C → E1 → B1 순서.
   - 공통 인자: `--items configs/g2v4b_fresh_shard{k}.json --order_sha <샤드 SHA>`.
   - A: `--cond A --decoder_ckpt results_s7/g2v2/decoder_M7A_v36.pt --zstar results_s7/g2v4/fresh_A_latent.pt`
   - C: `--cond C`
   - E1: `--cond E1 --decoder_ckpt results_s7/g2v2/decoder_M7E_v36.pt --zsrc results_s7/g2v4/fresh_E_latent.pt`
   - B1: `--cond B --shift 1 --decoder_ckpt …/decoder_M7A_v36.pt --zstar …/fresh_A_latent.pt`
   - 출력 `results_s7/p6r/g2v4fresh/gen_<cond>_s{k}.jsonl`. 로그에서 `[PROJ-V2]` 적재 행을 확인한다.
5. **병합·점검·채점**: 빈 새 폴더 `results_s7/p6r/g2v4fresh_merged/`에 `p6_gen_{A,B1,C,E1}.jsonl`을 만든다.
   - 채점 전 점검: 네 파일 각 1,758행, 고유 id 1,758, id 집합 = `g2v4b_fresh.json`, JSON 오류 0.
   - 채점 명령: `s7_eval_score_p6_v36.py --gen_A … --gen_B1 … --gen_C … --gen_E1 … --gen_dir results_s7/p6r/g2v4fresh_merged --order configs/g2v4b_fresh.json --out_json …/gate_table.json --out_md …/gate_table.md`
   - 판정 키는 `gates.Gate_{2,3,4}.strict`.
6. **판정**: G2v4 = 게이트 2 ∧ 게이트 3 ∧ 게이트 4.
   - 게이트 2: A − C ≥ −2 %p.
   - 게이트 3: A − C에서 p < 0.025 ∧ 95 % CI 하한 > 0.
   - 게이트 4: A − E‴에서 게이트 3과 같은 기준.
   - 공통: strict, 채점기 `scripts/s5_eval_grader_v2.py`(90cc93e4…), 채점 스크립트 547e1d91…, 부트스트랩 10,000·시드 2026, 정확 McNemar 단측.
   - 게이트 1(A − B1)은 보조.
- 실행 위생: tmux(R7-4), CPU 단계 `nice -n 19`·`ulimit -v 62914560`(R2-5), 모든 접속은 `cts_ssh.py`(R7-1).

## 5. 갈림길·통계

- 판정은 새 1,758문항으로 **한 번**만 한다. 결과를 본 뒤 조건·문항·생성·채점·게이트·가중치를 바꾸지 않는다.
- **반복 시도 보정 참고값**: 4차이므로 게이트 3·4의 p × 4를 함께 보고한다. 판정은 사전등록 기준 p < 0.025다.
- 도구 결함은 결과와 무관한 근거로만 판정한다. RULES R3-3에 따라 고친 뒤 같은 평가를 다시 한다. 다시 한 결과가 판정이고, 두 결과를 모두 보고한다(개정·B 감사).
- 검정력 참고(판정 아님): 개발의 불일치율 10 %·b 비율 0.65가 유지되면 기대 불일치 쌍은 약 176, 기대 z는 약 4다. 효과가 절반이면 통과 확률은 약 50 %다.
- **기준 미충족이면** 이 데이터 풀(math_train_prompts_5000)에는 미열람 문항이 남지 않는다. 5차는 새 데이터 출처가 필요하므로 Principal 결정 사항이다.
- GRAPH.md에 H-DEC-7(M7)을 등록한다. 기각 조건: 확증 미충족.

## 6. 고정 파일 (서버 SHA-256)

| 파일 | SHA-256 |
|---|---|
| `results_s7/g2v2/decoder_M7A_v36.pt` (= `g2v4m7rep/decoder_Ah.pt`) | 46977c2db533888dc13a8defe21ac287b26bd7ff1e08bb83eb22a4ca4bb99670 |
| `results_s7/g2v2/decoder_M7E_v36.pt` (= `g2v4m7rep/decoder_Eh.pt`) | d0c33599c9586d3ed131877a2439ee09d5d48804a0408965daaa0238705afd5a |
| `results_s7/g2v4m7rep/deq_Ah.pt` | b43de6b0eeb0a16a5f04762ac9c6a95979e504bc48f8b52249213490d6485cf7 |
| `results_s7/g2v4m7rep/deq_Eh.pt` | 7e9dc5feb1e563996e90cc825c3567e54caadb86f5a9a25ce03435df37be9788 |
| `scripts/g2v4_m4_pilot_train.py`(학습 실행본) | f492da0f1c35a3088397d2fec1748e9b56ed82bc3b371fe33cc0846706092ab0 |
| `scripts/g2v4_proj_v2.py` | 7cabcef2e13bb524a6815add5198853f21e8437fd04d018a6cef2227161df47a |
| `scripts/s7_eval_gate_p6_v36f.py`(v36e a2755c1b… + V2 적재 18행) | fd6c2d03e3ed18d1597950edc79a01af8331d192c9f84b821a5b5bcf3f71cfe1 |
| `scripts/g2v3_m3_eval_latents_v2.py` | 73db9dafd33e49e963d48aad4797c6b0eb7856e5e8e847fbb3b8dc0e5109921b |
| `scripts/g2v4_fresh_prep.py` | ed79108f9d8376467ea18781f6407c1140d59a69f7f3eaf22c98f929f311db54 |
| `scripts/g2v2_zstar_run.py` · `cts/deq/zstar_inject_v2.py` · `configs/zstar_canonical.yaml` | 687d4480… · 7024f1f2… · bd8e4e96… (3차와 같음) |
| `scripts/s7_eval_score_p6_v36.py` · `scripts/s5_eval_grader_v2.py` | 547e1d91… · 90cc93e4… |
| `scripts/g2v4_select_fresh_v2.py` · `configs/g2v4b_fresh.json` | e9d133b5… · 6e68e34c… |
| `scripts/cc_g2v4_m7_rep.sh`(학습 명령, 로컬본) · `scripts/cc_g2v4_dev_m7.sh` | 9fc2d657… · 45e6a514… |
| `scripts/cc_g2v4_confirm.sh` | 잠금 시 기록(사전등록 SHA를 넣은 뒤) |

## 7. 한계·GOAL

- **GOAL.md 표현**: G2를 "held-out 600에서"로 적는다. 1~3차처럼 이번에도 새 문항으로 확증한다. 개선판 통과를 GOAL의 G2 통과로 인정할지는 Principal 결정을 기다린다(CTS-431).
- **단일 시드**: A·E 디코더는 시드 하나로 학습했다. 개발 NLL 신호는 두 시드(Ah·Az)에서 같은 방향이었다(`DIAG-G2v4` §12).
- **백본 로더 차이**: z*′₀·z_src는 정본 로더(`harness_A_v1.get_model`)로, 학습된 한 걸음은 위 §1의 로더로 계산한다. 같은 문항에서 두 로더의 z_src는 cos 0.70~0.91이다(`DIAG-G2v4` §7). A·E 모두 같은 조건이다.
- **주장 범위**: 개선 효과는 "학습된 한 걸음과 V2 연결부의 조합"에 대한 것이다. 고정점을 학습된 사상으로 다시 푸는 방법이 아니다.
- **다중 시도**: 4차이므로 p × 4 참고값을 함께 해석한다.

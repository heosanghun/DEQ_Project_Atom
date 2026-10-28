# AUDIT-P014 — PREREG_S7_G2v4 사전 감사 (B, 2026-10-11)

- 감사인: B. 서버 접속 0(ssh·sftp·`cts_ssh.py` 실행 0). 로컬 파일만 읽었다. 새로 만든 파일은 이 감사 파일 하나다(검산은 메모리 안에서만 했다). 자격 증명 파일과 scratch의 `.sftp` 묶음 파일은 읽지 않았다. `shared/ssh_log.csv`는 도구 권한 정책으로 읽지 못했다. 그래서 §0-5의 AG 조회 주장(10:22·10:23)은 확인하지 못했다.
- 대상: `shared/prereg/PREREG_S7_G2v4.md` DRAFT rev1, **SHA-256 `4563782a189d1de73ff3d1e5ced94d5cb3bc747ac6c5a72f72619c754f5e381f`**(로컬 실측), 150행, 파일 시각 10-11 16:40(문서에는 "17:0x"라고 적혀 있다).
- 함께 읽은 것: GOAL.md, RULES.md, STATE.md 끝 30행(마지막 기록은 14:2x), PREREG_S7_G2v3_1(3d1a6ee0…, 로컬 일치), RESULT-G2v3-confirm, AUDIT-P013, AUDIT-P012(관련 행), DIAG-G2v4 §1~§12, GRAPH.md, `shared/reports/g2v4m7rep/`·`g2v4diag/`·`g2v4v2/`·`g2v4deqlr/`의 원자료, 아래 표의 코드, `shared/archive/g2v2_select_run1/`.
- 개발 생성 결과 원자료(`dev_score.json`, `dev_gen_*.jsonl`)는 로컬에 없다. 그래서 §0-1의 개발 정확도 수치는 산술 검산만 했고 원자료로는 대조하지 못했다(FIX-1).

## 0. 대상 SHA (로컬 직접 계산)

| 파일 | 로컬 SHA-256 | 사전등록 §6 값 | 판단 |
|---|---|---|---|
| `scripts/g2v4_m4_pilot_train.py` | f492da0f1c35a3088397d2fec1748e9b56ed82bc3b371fe33cc0846706092ab0 | f492da0f… | 일치 |
| `scripts/g2v4_proj_v2.py` | 7cabcef2e13bb524a6815add5198853f21e8437fd04d018a6cef2227161df47a | 7cabcef2… | 일치 |
| `scripts/s7_eval_gate_p6_v36f.py` | fd6c2d03e3ed18d1597950edc79a01af8331d192c9f84b821a5b5bcf3f71cfe1 | fd6c2d03… | 일치 |
| `scripts/s7_eval_gate_p6_v36e.py`(기준판) | a2755c1b591e12389c8efe167aa33a59a7ae6c0876234585381b6a129faa460a | a2755c1b… | 일치 |
| `scripts/g2v3_m3_eval_latents_v2.py` | 73db9dafd33e49e963d48aad4797c6b0eb7856e5e8e847fbb3b8dc0e5109921b | 73db9daf… | 일치 |
| `scripts/g2v4_fresh_prep.py` | ed79108f9d8376467ea18781f6407c1140d59a69f7f3eaf22c98f929f311db54 | ed79108f… | 일치 |
| `scripts/g2v3_fresh_prep.py`(3차) | 59889cd3cc829a5bb281e0d34a7a20899f824272f0911eb8db8547cbe741b423 | (3차 §6) | 일치 |
| `scripts/g2v4_select_fresh.py`(v1) | be018b3a8ee71adc411cb61af738fe7cc85caed3fa7868b71b6ea094968cb152 | be018b3a… | 일치 |
| `scripts/g2v4_select_fresh_v2.py` | e9d133b550a92322b38fc7f66ab440d37a472a5cf4587f1c98e5824066c470f3 | e9d133b5… | 일치 |
| `scripts/cc_g2v4_m7_rep.sh` | 9fc2d657b0e808ce37bf1df9f0dae92dcf41a6f11e221508564fac99f8816888 | 9fc2d657… | 일치 |
| `scripts/cc_g2v4_dev_m7.sh` | 45e6a5143171a274c71751c0273f6a3196e42a9511585f976a7811ce98c8ba56 | 45e6a514… | 일치 |
| `scripts/cc_g2v4_confirm.sh`(13행 `__PREREG_SHA__` 자리표시 상태) | 3169e8e352cb1f3a2eb16246308dde609a5330125e8826c253e92c783ad7dc8d | "잠금 시 기록" | FIX-5(c) |
| `scripts/cc_g2v3_confirm.sh`(비교용) | 81d5ca1fd8205b513e13bdb065462b54fc602dfea78ee3cc4a0e768bcc8ecc14 | (3차) | 일치 |
| `scripts/g2v4_diag_iterate.py` | 85b4e94f8523d15d4d522084cd0473d3446a702df4b31e52853346bdb7366126 | 없음 | 권고 R-9 |
| `scripts/s7_eval_score_p6_v36.py` · `s5_eval_grader_v2.py` | 547e1d91aa998908… · 90cc93e4b76704b2… | 547e1d91… · 90cc93e4… | 일치 |
| `scripts/g2v2_zstar_run.py` · `cts/deq/zstar_inject_v2.py` · `configs/zstar_canonical.yaml` | 687d4480… · 7024f1f2… · bd8e4e96… | 같음 | 일치 |
| `scripts/check_p6_decoder_version.sh` · `s7_train_decoder_v36.py` · `g2v3_m3_full_train.py`(실행 중 import됨) | cc685dcb… · f2951054… · 36e0f18a… | 없음 | FIX-5(b) |
| `configs/s7_train_993.json` · `g2v2_exclude.json` · `g2v2_fresh1000.json` | 3088a153… · 1c63770e… · 35683874… | 3088a153… · — · — | 일치 |
| `configs/g2v4b_fresh.json`(1,758) | 로컬에 파일 없음. **B가 재구성**: `json.dumps({"indices": sorted(range(5000) − g2v2_exclude − g2v2_fresh1000)})` → 6e68e34c510593fb017d7dab5296c60ac506ce28d1ddeb3701231a69f696ce80 | 6e68e34c… | 일치(재구성) |
| 샤드 0·1·2 (각 586) | 재구성 f8f305a1… · c5a3ce59… · 63935a09… | 같음 | 일치(재구성) |
| `configs/g2v4b_exclude.json` | 재구성(제외 = g2v2_exclude ∪ fresh1000, `n_union` 3240) → 91a2cdc8… | 91a2cdc8… | 일치(재구성). 서버 검사가 새로 제외한 문항이 0이라는 뜻이다 |
| `configs/g2v4_fresh.json`(v1, 1,234) | 재구성(1,758 − 보관된 run1 목록) → 302dffc4… | 302dffc4… | 일치(재구성) |
| `configs/g2v4b_server_json_union.json` | 서버 경로가 들어 있어 재구성할 수 없다 | cd7c8f1f… | 확인 불가 |
| `decoder_M7A_v36.pt`·`decoder_M7E_v36.pt`·`deq_Ah.pt`·`deq_Eh.pt`, 개발 잠재·생성·`dev_score.json` | **서버 산출물이라 로컬에서 SHA를 확인할 수 없다** | 46977c2d…·d0c33599…·b43de6b0…·7e9dc5fe…·e890c3c2… | 확인 불가(FIX-1) |
| `shared/reports/g2v4m7rep/diag_{Ah,Az,Eh}.json` | 69a7cb41… · 416b56c7… · b564b996… | 없음 | `diag_Ez.json`은 없음(FIX-2) |

## 1. 점검표

| # | 항목 | 판단 | 근거 (파일:행) |
|---|---|---|---|
| 1 | 개발 열람 공개 | 대부분 충분 → FIX-2 | 대상 14~21행에 개발 600 NLL 진단을 여러 번 봤다는 것, 설정을 NLL을 보고 골랐다는 것, winner's curse, "정확도 평가는 1회"가 적혀 있다. 로컬 근거로 보면 g2v4 스크립트 가운데 v36f·채점을 부르는 것은 `cc_g2v4_dev_m7.sh`와 `cc_g2v4_confirm.sh`뿐이라 "정확도 1회"와 맞는다(서버에서 따로 돌린 실행까지 배제할 수는 없다). 빠진 것: `cc_g2v4_m7_rep.sh:22`는 Ez(E, 글 가리기 1.0, 시드 2048)도 학습·진단했다. 대상 16행은 Ez를 "본 것"에 넣었지만 DIAG §12 표(126~133행)와 로컬 원자료(`g2v4m7rep/`)에 Ez 결과가 없다. Ah/Eh(0.5)를 Az/Ez(1.0)보다 앞세워 정확도 평가에 올린 기준도 적혀 있지 않다 |
| 2 | 개발 수치 산술 | 맞음(원자료 미대조) | 47/600 = 7.83 %, 29/600 = 4.83 %, 2/600 = 0.33 %, Δ 18/600 = +3.00 %p = 39 − 21, Gate_3 45/600 = +7.50 %p. b = 39, c = 21의 정확 단측 p = 0.01367(B 재계산, 대상 18행 0.0137과 같다). 3차 b = 42, c = 30 → 0.09725(RESULT와 같다). CI와 SHA e890c3c2…는 원자료가 로컬에 없어 대조하지 못했다 |
| 3 | M7 복제 NLL | 맞음 | `diag_Ah.json` 요약: 자기 0.8503, 다음 0.8537, 다음 − 자기 +0.0034(z 4.37), 글 없음 +0.0020(z 2.64). Eh +0.0029(z 0.62), 0.8739. Az +0.0048(z 2.55). Ah − Eh = −0.0236(z −7.18), A가 나은 문항 65.7 %. 모두 DIAG §12와 같다 |
| 4 | 판정 규칙 고정 | 대체로 고정 → 권고 R-1~R-3 | 대상 107~112행: 게이트 2·3·4 기준, strict, 채점기·채점 스크립트 SHA, 부트스트랩 10,000·시드 2026, 정확 McNemar 단측. 117행 1회 판정, 결과 뒤 변경 금지. 판정 키 106행은 `gates.Gate_{2,3,4}.strict`까지만 적었고 하위 키(`delta`, `ci_95[0]`, `mcnemar_one_sided_p`)와 반올림(5·6자리)은 적지 않았다(`s7_eval_score_p6_v36.py:302~306`). 채점 스크립트는 통과 여부를 계산하지 않으므로 사람이 키를 읽어 판정한다 |
| 5 | 결과 뒤 바꿀 여지 | 작음, 두 군데 → FIX-5(a), 권고 R-2·R-3 | (a) 중단 뒤 재개 절차가 없다. v36f(`s7_eval_gate_p6_v36f.py:535~548`)와 z 실행기(`g2v2_zstar_run.py:105~113`)는 이미 있는 출력에 이어 쓰거나 건너뛴다. 그런데 확증 스크립트의 존재 검사(`cc_g2v4_confirm.sh:29~30`)는 shard0 z·병합 z·A 잠재 세 파일만 본다. 생성 폴더 `$O`, shard1·2 z, `fresh_zsrc_1758.pt`, E 잠재는 보지 않고, `$O`는 검사 전에 `mkdir -p`로 만든다(10행). 앞서 중단된 시도의 행이 있으면 조용히 섞인다. 대상 84행("z·잠재 출력이 이미 있어도 중단")보다 약하다. (b) 119행 도구 결함 재실행 조항은 3차와 같다. 근거를 결과와 무관하게 하라는 요구는 있지만, 결함을 언제(채점표 생성 전·후) 무엇으로 판정하는지는 정하지 않았다 |
| 6 | 새 문항 1,758 = 사전 선언된 풀 | **지지됨** | 3차 잠금본 `PREREG_S7_G2v3_1.md:73`이 결과 전에 "남은 후보 2,758 − 1,000 = 1,758"을 4차 풀로 적었다. B가 로컬 입력만으로 `range(5000) − g2v2_exclude(2,242) − g2v2_fresh1000(1,000)`을 계산하니 1,758이고, json SHA가 6e68e34c…, 샤드 SHA 세 개도 대상 61~63행과 같았다. 학습 993·`g2v2_dev600`·`s7_train_1000`·`s7_new_heldout_520`·`s7_new_train_680`과 겹침 0 |
| 7 | v1(1,234) → v2(1,758) 변경 근거 | 논리 타당, 원시 근거 없음 → FIX-1(b)(c) | B 재구성: v1 = 1,758 − 보관 run1 목록 = 1,234, SHA 302dffc4…(대상 68행과 같다). 따라서 두 목록 차이는 정확히 run1 ∩ 후보 524이고, 다른 서버 파일에 나온 문항은 없다(v2 검사 적중 0, 제외 재구성 91a2cdc8…). 로컬에서 같은 규칙으로 검사해도 524 적중은 `shared/archive/g2v2_select_run1/` 네 파일(목록·샤드 3)뿐이다. 그 밖의 적중은 `artifacts/five_seed_stats.json`의 시드 값 456·789 두 개로, 거짓 적중이고 로컬에만 있는 파일이다. 또 run1 목록 1,000 중 288문항은 이미 3차 확증 1,000에 들어 있었고, AUDIT-P012 재감사 3(152행)이 이를 "번호 목록 비교일 뿐"으로 받아들였다. 따라서 run1에 들어 있었다는 사실만으로 오염이라고 볼 수 없다. 남은 구멍: (i) "grep 0건·smoke 1개"(대상 70행)와 v2 머리 주석(`g2v4_select_fresh_v2.py:2~4`)이 가리키는 `scratch/cc_check_run1` 원시 출력이 로컬에 없다. (ii) AUDIT-P012 167·195행: 10/08 15:2x~15:5x에는 서버의 `configs/g2v2_fresh1000.json` 자체가 run1 목록이었고, z 실행기는 목록 SHA를 확인하지 않는다. json 정수 검사는 `.jsonl`의 `id`와 `.pt` 키를 보지 않는다(AUDIT-P012 161행). 그 30분 동안 524문항으로 계산이 있었는지를 직접 배제하는 근거가 아직 없다 |
| 8 | 변경 판단이 내용·결과와 무관한가 | 무관(시점만 공개 필요) | v2는 `own`에 `_run1_48b7f09d`를 더한 것뿐이다(diff: 17행·출력 이름). 선택은 정수 목록만 읽는다(`g2v4_select_fresh_v2.py:35~46`, 문항 글·풀이를 읽지 않음). 변경은 개발 결과(16:27) 뒤에 했지만, 확증 문항 정보 없이 N만 키운다. 사전 선언된 풀이므로 확증 검정의 1종 오류를 키우지 않는다. 표현: 대상 71행 "4개 파일" — 패턴 `_run1_48b7f09d`는 서버의 다섯 파일(`g2v2_exclude_run1…` 포함, AUDIT-P012 195행)에 걸린다. 로컬 보관본 시각은 15:17:56(대상 69행 "15:24")이다 → 권고 R-5 |
| 9 | 학습 레시피 대칭 | 대체로 대칭 | `cc_g2v4_m7_rep.sh:13~15·20~21`: 같은 스크립트, 3000단계, 배치 4, lr 1e-4/3e-4/3e-4, 감쇠(`g2v4_m4_pilot_train.py:437~439`), 글 가리기 0.5, 시드 2047. 데이터 순서는 셔플 없이 같다(535~538행). 시드: `manual_seed`·`default_rng`가 같은 시드다(279~280행). 같은 코드 경로라 투영기 초기값과 가리기 순서가 같을 가능성이 높다. 미확인이라고 공개했다(44행) |
| 10 | λ_ift 비대칭 | **영향 커짐, 주장 문구 부정확** → FIX-3(a)(b) | A 0.1·E 0(608행, 대상 45행). 3차에서는 l_ift가 약 4.5e-5로 작았다. 4차는 lr_deq를 10배로 올려 Φ_θ가 크게 움직인다. 개발 진단의 한 걸음 상대 이동 중앙값(`diag_Ah.json` `zstar_k1_res` 0.127, Az 0.056, Eh `zsrc_k1_res` 0.107)으로 보면 IFT 보조 손실이 A의 사상을 실질적으로 묶는다. 같은 A 디코더에 평가 입력만 T_θ,A(z_src)로 바꾸면 NLL이 0.8514로, 0.8503과 0.0011밖에 차이 나지 않는다. A − E 차이 −0.0236의 대부분은 평가 시작점이 아니라 학습 경로(시작점 + λ_ift)에서 온다. 대상 53행 "z_src에서 출발한 같은 학습"은 사실과 다르다(λ_ift가 다름). 게이트 4는 시작점 효과와 λ_ift 효과를 가르지 못하는데, 이것이 적혀 있지 않다 |
| 11 | 투영기 center 차이 | 공개 부족 → FIX-3(c) | `g2v4_m4_pilot_train.py:499~504`: center = A는 학습 993문항 z*′₀의 칸별 평균, E는 z_src 평균. 투영기 입력은 시작 잠재가 아니라 T_θ(·)다(595행). `g2v4_proj_v2.py:36`은 z − center 뒤 RMS 정규화를 한다. 한 걸음 이동이 A 12.7 %·E 10.7 %라서, 두 조건 모두 center가 실제 입력 평균에서 벗어나 있다. 규칙(자기 시작 잠재의 평균)은 대칭이다. 하지만 벗어난 정도와 그것이 정규화에 주는 영향은 재지 않았다. 참고로 Eh `mean_nll` 1.564 대 Ah 0.859는 평균 입력에서 이 정규화가 조건마다 다르게 작동함을 보여 준다(DIAG §9가 평균 잠재 해석을 제외한 이유와 같다). 대상 38행은 "학습 시작 잠재"라고만 적었다 |
| 12 | GPU·백본 로더 | 공개됨(한 문장 보강) → FIX-3(e) | 학습 GPU 5·6(46행), 확증은 샤드마다 같은 GPU(87·96행, `cc_g2v4_confirm.sh:54~59`). 로더 차이는 147행에서 공개했다. DIAG §7(68행)은 "4차 설계에서 정본 로더로 통일"이라고 했는데 M7은 통일하지 않았다. 이 이탈과 그 이유(학습이 이미 끝남)가 문서에 없다 |
| 13 | 학습 입력·데이터 행 | 누락 → FIX-1(d) | 학습 명령에 `--zstar`·`--zsrc`가 없어 기본값 `results_s7/g2v2/zstar_zsrcinj_train1000.pt`·`results_s7/zsrc_train1000_P4.pt`를 읽는다(`g2v4_m4_pilot_train.py:696~697`). 이 두 파일과 그 SHA(G2v2 §6의 005e622d…·340ef5c2…)가 대상 §1·§6에 없다. `[DATA] Filtered` 행(A는 z*′·z_src가 둘 다, E는 z_src만 있으면 됨, 489행), 최종 l_task·l_ift, 매니페스트 SHA, `n_text_dropped`(A = E이면 가리기 순서가 같다는 정황 근거)가 없다. 3차 FIX-7에서 요구한 항목이다 |
| 14 | 명령 전문 ↔ 확증 스크립트 | 명령은 일치, 본문·가드는 불일치 → FIX-5 | 대상 88행 z 명령 = 스크립트 38~40행. 93~94행 = 47~50행. 98~101행 = 56~59행(A → C → E1 → B1, 같은 GPU). 105행 채점 = 82행. 병합 점검 67~80행(N = `g2v4b_fresh.json` 길이 = 1,758, 행·고유 id·집합·JSON 오류). 3차 스크립트와의 차이는 경로·N·`PROJ-V2` grep(64행)뿐이다. 불일치: 83행 "§6의 모든 SHA를 서버에서 대조"인데 스크립트는 그 일부만 본다(13~28행. 학습·선택·개발 스크립트가 없고, 실행 중 import되는 `s7_train_decoder_v36.py`·`g2v3_m3_full_train.py`·`check_p6_decoder_version.sh`도 없다). 141행 "잠금 시 기록"은 사전등록 SHA를 스크립트에 넣으면 스크립트 SHA가 바뀌므로 사전등록 안에 적을 수 없다(순환) |
| 15 | v36f 변경 범위 | 맞음 | `diff v36e v36f`: 머리 주석 6행 추가, 471행 투영기 생성 1행 삭제, 478~486행 9행 추가(LM LoRA 거부, `proj_version` 2 → `PrefixProjectorV2`), 끝 출력 문구 1행. 변경 행 18행(대상 134행과 같다). 생성·조회·경로·순서 코드는 그대로다 |
| 16 | V2 strict 적재 | 맞음 | `s7_eval_gate_p6_v36f.py:487~490`의 `load_state_dict`는 기본값 strict다. `PrefixProjectorV2`는 `center` 버퍼(`g2v4_proj_v2.py:28`)와 `out_rms`를 state_dict에 갖고 있고, 학습 저장(`g2v4_m4_pilot_train.py:649`)도 `proj_version` 2와 전체 state_dict를 쓴다. 이 키가 없으면 v1 투영기에 strict로 넣다가 실패하므로 조용한 대체는 없다 |
| 17 | 버전·경로 가드 | 통과 | 버전 가드는 이름만 본다(`check_p6_decoder_version.sh` `_v36(\.pt\|_step….pt)$`). `decoder_M7A_v36.pt`는 통과한다. 경로 가드(`v36f:438~443`)는 realpath 디렉터리가 `results_s7/g2v2`이고 이름이 `decoder_[A-Za-z0-9]+_v36\.pt`일 것을 요구한다. `M7A`·`M7E`가 맞고, 사본(`cc_g2v4_dev_m7.sh:19` `cp -n`)이라 symlink 문제도 없다. `cp -n`은 덮어쓰지 않으므로, 사본 SHA = `decoder_Ah/Eh.pt` SHA라는 원시 출력이 필요하다(FIX-1(a)) |
| 18 | A 조건 옛 400 병합 | 영향 없음 | `v36f:510~517`은 `{**d80, **d_new}`이고, 잠재 v2는 `q….._a0.25` 키를 쓴다(`g2v3_m3_eval_latents_v2.py:184`). 새 문항은 `g2v2_exclude`(s5_items_400 포함)와 겹치지 않으므로 옛 값을 조회할 수 없다 |
| 19 | 학습된 한 걸음 가드 | 대체로 맞음 | 잠재 v2: 예상 밖 키면 중단(106~110행), z·문항 글이 없으면 중단(140~161행). 103행 `if os.path.exists(args.deq_ckpt)`가 거짓이면 Φ₀로 조용히 계산하지만, 확증 스크립트 27~28행이 같은 경로의 SHA를 먼저 확인하므로 막힌다. 51행 `grep LOAD-GUARD`는 출력만 하고 강제하지는 않는다 → 권고 R-7 |
| 20 | 채점 N 처리 | 맞음 | `s7_eval_score_p6_v36.py:225~226`은 `--order` 목록 길이를 N으로 쓰고, 부트스트랩(137~149행)과 McNemar(118~126행)는 길이에 의존하지 않는다. 빠진 id는 조용히 건너뛰지만(244~246행) 병합 점검이 1,758/1,758을 강제한다. 조건 간 길이가 다르면 게이트를 내지 않는다(279행). 표시만 600 문구가 남아 있다(400행 "FULL 600", 269행 `:3d`) → 권고 R-8 |
| 21 | fresh prep | 경로만 바뀜(주장과 같다) | `diff g2v3_fresh_prep.py g2v4_fresh_prep.py`: 머리 주석, 10행 목록 경로, 14·18~21행 경로·파일 이름뿐이다. 점검 논리(키 집합 = 목록, max\|Δ\| = 0, `assert`)는 그대로다 |
| 22 | 통계·검정력 | 맞음 | 대상 120행: 1,758 × 0.10 = 175.8, z = 0.30·√175.8 = 3.98, 효과가 절반이면 z 1.99 → 통과 확률 약 51 %(B 재계산). p × 4 참고(118·149행). 실제 확증 시도는 1·3·4차의 3회라 ×4는 보수적이다 |
| 23 | GOAL과의 관계 | 부족 → FIX-4 | 대상 145행: "held-out 600" 표현과 CTS-431 미결은 공개했다. 빠진 것: (i) GOAL 8~12행의 "자기 z*"는 정본 풀이기(G1)의 z*다. 이번 A 입력은 M2′(z_src 주입) z*′₀에 학습된 한 걸음을 더한 것이고, 개발 중앙값으로 z*′₀에서 12.7 % 떨어져 있다. 3차 v3.1 §1(28행)에 있던 "고정점 근접도 보고 항목"이 4차에서 빠졌다. (ii) 3차 v3.1 §7-5(108행)의 대체 보고 문장("받지 못하면 개선 방법의 G2 기준 충족 여부로만 보고")이 빠졌다 |
| 24 | 가설 등록 | 누락 → FIX-4(c) | 대상 122행은 "등록한다"라고 미래형으로 적었다. `shared/GRAPH.md`(10-11 08:43)는 41행 H-DEC-6까지만 있고 H-DEC-7이 없다(R4-6: 기각 조건과 함께 등록된 가설만 시험) |
| 25 | 증거 포인터 | 부족 → FIX-1 | STATE.md 마지막 기록은 14:2x다. 개발 결과(16:27), 선택 v1·v2(16:3x~16:5x), 초안 작성 행이 없다. 개발 채점표·개발 연쇄 로그·선택 실행 출력([SELECT]·[HIT]·SHA 행)·run1 점검 출력·학습 로그·`.pt` 서버 SHA의 원시 출력이 로컬에 없다(R1-1·R3-5) |
| 26 | 실행 위생 | 맞음(확증), 권고 | CPU 단계(prep·병합·채점)는 `nice -n 19`·`ulimit -v`(44·67·82행), GPU 비어 있음 확인(31~35행). tmux 기동·`tee` 마스터 로그 명령은 본문에 없다 → 권고 R-6 |
| 27 | 1~3차 결과 공개 | 맞음 | 대상 6~9행: 잠금본 SHA 앞자리, 2차는 확증 없음, 3차 수치(+1.20 %p, CI [−0.50, +2.90], 42/30, p 0.0973)가 RESULT-G2v3-confirm 28행과 같다 |

## 2. 판단

- **v2(1,758)를 받아들인다**(FIX-1(b)(c) 충족이 조건). 근거는 셋이다. 1,758은 3차 잠금본이 결과 전에 선언한 풀과 바이트 단위로 같다. 524문항이 빠진 이유는 보관된 run1 선택 목록 자체가 검사에 걸렸기 때문이다. 그리고 같은 run1 목록의 288문항은 이미 3차 확증에서 정당하게 쓰였다. 판단 근거는 파일 출처뿐이고 문항 내용이나 결과는 쓰지 않았다. 다만 10/08 15:2x~15:5x에 서버의 활성 목록이 run1이었으므로, 그 사이 계산이 없었다는 것을 id 단위 원시 근거로 보여야 한다. 적중이 있으면 대상 71행대로 v1(1,234)로 간다.
- 학습은 사전등록 전에 끝났고, 설정은 개발 NLL을 보고 골랐으며, 개발 정확도는 1회 봤다. 이 사실들은 대체로 공개됐다. 확증은 미열람 문항 1회이고 명령이 고정돼 있으므로, 사후 선택의 위험은 확증 문항이 깨끗하다는 조건 위에서 통제된다. 남은 문제는 다섯 가지다. Ez 미공개, 비대칭(λ_ift·center)의 실질적 영향과 주장 문구, GOAL 해석, 가설 미등록, 출력 존재 가드의 빈틈.
- 재학습은 요구하지 않는다. 아래 FIX는 모두 문서 보강, 원시 출력 확보, 확증 스크립트 가드 보강이고, 결과 정보 없이 할 수 있다.

## 3. FIX 목록

- **FIX-1 (원시 근거·포인터, R1-1·R3-5)** 다음 원시 출력을 로컬(`shared/reports/g2v4dev/`, `scratch/`)에 두고 대상 해당 행에 포인터를 단다. STATE에 14:2x 이후 행(개발 결과, 선택 v1·v2, 초안)을 더한다.
  - (a) 개발: `cc_g2v4_dev_m7.sh` 마스터 출력 전문. 10·20행 SHA 출력에서 `decoder_Ah.pt` = `decoder_M7A_v36.pt`(46977c2d…), `decoder_Eh.pt` = `decoder_M7E_v36.pt`(d0c33599…), `deq_Ah/Eh` SHA, `[LOAD-GUARD]`, `[PROJ-V2]`, `[CHECK]`, `[SCORE] rc`, `[G4]`·`[G3]` 행이 보여야 한다. 그리고 `dev_score.json`(e890c3c2…)·`dev_score.md` 사본.
  - (b) 선택: v1·v2 실행 출력([SELECT]·[HIT]·SHA 행), `g2v4b_server_json_union.json`(cd7c8f1f…) 사본, v2 머리 주석이 가리키는 `scratch/cc_check_run1` 원시 출력(grep 명령과 결과, 10/08 15:00~16:10 생성 파일 목록).
  - (c) 524문항 미사용의 직접 근거(번호만, 내용 열람 금지): 서버 `results_s7/`·`raw_s7/`·`scratch/`·`results/` 아래 모든 `.jsonl`의 `id`(또는 `q_idx`) 필드와 `.pt` dict 키(`q\d{4}` 접두)를 모아, 1,758 목록과의 교집합 수를 파일별로 출력한다. **통과 조건**: 교집합 0. 적중이 524 안에만 있으면 v1(1,234)로 확정한다(대상 71행 규칙). 1,234 안에도 있으면 B에 알리고 실행을 멈춘다.
  - (d) 학습: `train_Ah.log`·`train_Eh.log`의 `[DATA] Filtered`·`[PROJ-V2] out_rms`·`center` 행, 최종 l_task·l_ift. 매니페스트 `manifest_Ah/Eh.json` SHA와 `n_text_dropped`(A·E 값). 학습 입력 `zstar_zsrcinj_train1000.pt`·`zsrc_train1000_P4.pt`의 서버 SHA. 이것들을 §1·§6에 행으로 넣는다.

- **FIX-2 (선택 경로 공개)** §0에 다음을 적는다.
  - (a) Ez(E, 글 가리기 1.0, 시드 2048) 진단 수치(글 포함·글 없음 다음 − 자기와 z, 자기 NLL)와 Az − Ez NLL 차이(z, A가 나은 비율), `diag_Ez.json` 사본·SHA.
  - (b) 개발 정확도 평가에 Ah/Eh를 고른 기준과 시각.
  - (c) 개발 600으로 본 설정 목록과 각 결과 포인터(M4, 용량, LM LoRA, V2 A·E·EL, 양성 대조, deqlr A3·E3·E5, m7rep Ah·Eh·Az·Ez).
  - 결과가 불리해도 그대로 적는다. 이 공개는 판정을 바꾸지 않는다.

- **FIX-3 (공정성·주장 범위)** §1·§7을 고친다(재학습 없음).
  - (a) 53행 "같은 학습"을 "λ_ift만 다른 학습(A 0.1, E 0)"으로 고친다.
  - (b) 게이트 4는 시작점 효과와 λ_ift 효과를 가르지 못한다고 한계에 적는다. 개발 근거로 A 한 걸음 이동 0.127(Az 0.056)과 A 디코더 입력 교체 진단(T_θ,A(z*′₀) 0.8503 대 T_θ,A(z_src) 0.8514, A − E −0.0236; `diag_Ah.json`·`diag_Eh.json` 요약 키)을 판정 아님으로 명기한다.
  - (c) center: A = 학습 993 z*′₀ 칸별 평균, E = 학습 993 z_src 칸별 평균. 투영기 입력은 T_θ(·)이고 두 조건 모두 center에서 벗어나 있다(한 걸음 이동 A 12.7 %·E 10.7 %). 규칙은 대칭이지만 효과가 같은지는 재지 않았다고 적는다.
  - (d) FIX-1(d)의 `[DATA] Filtered`·`n_text_dropped`로 학습 문항 수와 가리기 순서가 같은지 적는다(다르면 그 사실을 적는다).
  - (e) DIAG §7의 "정본 로더 통일" 계획에서 벗어난 사실과 이유를 적는다.

- **FIX-4 (GOAL·보고 항목·가설 등록)**
  - (a) §7에 적는다: A 입력은 GOAL의 "자기 z*"(정본 풀이기)가 아니다. M2′ z*′₀ + 학습된 한 걸음이다. 보고 항목(판정 미사용)으로 확증 문항의 ‖T_θ,A(z*′₀) − z*′₀‖/‖z*′₀‖와 ‖T_θ,E(z_src) − z_src‖/‖z_src‖ 분포를 넣는다. 확증 산출물(`fresh_A_latent.pt`·`fresh_zsrcinj_1758.pt`·`fresh_E_latent.pt`·`fresh_zsrc_1758.pt`)에서 결과 없이 계산할 수 있고, 계산 명령은 사전등록에 미리 적는다.
  - (b) 3차 v3.1 §7-5의 문장을 되살린다: "CTS-431을 받지 못하면 'M7의 G2 기준 충족 여부'로만 보고한다."
  - (c) 잠금 전에 GRAPH.md에 H-DEC-7(M7, 기각 조건 = 확증 미충족)을 등록하고, 그 행 번호를 대상 122행에 적는다(R4-6).

- **FIX-5 (확증 스크립트 가드·본문 일치)**
  - (a) `cc_g2v4_confirm.sh`의 `mkdir`(10행) **앞에서**, 다음 가운데 하나라도 있으면 중단한다: `$O`(`results_s7/p6r/g2v4fresh`) 디렉터리, `$M`, `$G/fresh_zsrcinj_shard{0,1,2}.pt`, `fresh_zsrcinj_1758.pt`, `fresh_zsrc_1758.pt`, `fresh_A_latent.pt`, `fresh_E_latent.pt`. v36f와 z 실행기의 이어 쓰기 때문에, 대상 84행이 약속한 것보다 넓게 막아야 한다.
  - (b) 83행 "§6의 모든 SHA"와 스크립트를 맞춘다. 실행 중 import·호출되는 `scripts/s7_train_decoder_v36.py`(f2951054…), `scripts/g2v3_m3_full_train.py`(36e0f18a…; `g2v4_proj_v2.py`가 import함), `scripts/check_p6_decoder_version.sh`(cc685dcb…)에 `chk`를 더하고 §6에도 행을 둔다. 학습·선택·개발 스크립트는 `chk`를 더하거나, 83행을 "실행에 쓰는 파일"로 고치고 나머지는 STATE 기록으로 고정한다고 적는다.
  - (c) §6 141행을 "고친 스크립트의 잠금 전 SHA <값>(13행 `__PREREG_SHA__` 자리표시 상태). 실행본 = 이 파일에서 13행 자리표시만 잠금본 SHA로 바꾼 것. 실행본 SHA는 STATE에 기록"으로 바꾼다(순환 해소).

## 4. 권고 (PASS 조건 아님)

- **R-1** 판정 키를 하위 키까지 적는다: 게이트 2 `gates.Gate_2.strict.delta ≥ −0.02`, 게이트 3·4 `strict.mcnemar_one_sided_p < 0.025 ∧ strict.ci_95[0] > 0`. JSON 값(반올림 5·6자리)을 쓰고 md 표는 쓰지 않는다.
- **R-2** 중단 시 재개 규칙: `gate_table.json`이 생기기 전에만 허용한다. 같은 명령으로 실패한 단계부터 재개하고(이어 쓰기), 재개 사실·시각·행 수를 보고한다.
- **R-3** 도구 결함 재실행: 결함은 사전 점검(SHA·행 수·LOAD-GUARD·PROJ-V2·병합 assert)이나 채점표 생성 전에 드러난 것만 결과와 무관하다고 본다. 채점표를 본 뒤 찾은 결함은 B 감사 전 재실행을 금지한다.
- **R-4** §3에 개발 C의 출처(1차 P6R `p6_gen_C.jsonl`, v36d·GPU 0)를 적는다. A·E는 반씩 GPU 5·6에서 같은 절반을 돌렸다는 것(`cc_g2v4_dev_m7.sh:27~34`)도 적는다.
- **R-5** 표현 정정: 3행 시각(파일 16:40), 69행 "15:24"(로컬 보관본 15:17:56), 71행 "4개 파일"(패턴은 서버의 다섯 파일에 걸림, `g2v2_exclude_run1`은 이미 제외 집합에 포함).
- **R-6** 기동 명령을 본문에 적는다(예: `tmux new -d -s cc_g2v4_confirm "bash scripts/cc_g2v4_confirm.sh 2>&1 | tee raw_s7/g2v4_confirm_master.log"`).
- **R-7** 51·64행 grep을 강제 점검으로 바꾼다. A·E 두 로그에 "applied 436 DEQ tensors, unexpected 0"이 있어야 하고, 세 샤드의 A·E1·B1 로그에 `[PROJ-V2]`가 있어야 한다. 아니면 중단한다.
- **R-8** 결과 문서에 "v36f 머리말 'v3.6e·Held-out 600', 채점 md 제목 'FULL 600'은 표시 문구일 뿐 N = 1,758"이라고 적는다.
- **R-9** 개발 NLL 근거 스크립트 `g2v4_diag_iterate.py`(85b4e94f…, DIAG 머리의 120dc83d…와 다른 판)를 §6 참고 행에 둔다. AUDIT-P013 R-1(학습 `bb.train()`과 평가 `eval()`에서 `routing_proj` spectral norm의 u·v 차이)을 학습 문항 2개로 점검해 max|Δ|를 남긴다.

## 5. 잠금 조건

1. FIX-1~5를 반영한 rev2를 만든다. rev1(4563782a…)은 `shared/prereg/drafts/`에 사본으로 보존한다(R3-4). B 재감사는 이 다섯 건만 본다.
2. 재감사 PASS 뒤에는 제목을 "잠금본"으로, 3행 상태 줄을 "LOCKED (시각) — AUDIT-P014 재감사 n PASS(rev2 SHA)"로만 바꾼다. 나머지 행은 rev2와 같아야 하며, B가 diff(제목·상태 두 행)로 확인한다. 잠금본 이름은 `shared/prereg/PREREG_S7_G2v4.md`, rev2 사본은 drafts에 둔다.
3. 잠금본 SHA를 STATE에 적는다. 서버에 `shared_prereg_PREREG_S7_G2v4.md`로 올리고 `sha256sum`·`date` 원시 출력을 남긴다. 그 시각에 서버에 `results_s7/g2v4/fresh_*`·`results_s7/p6r/g2v4fresh*`가 없다는 `ls` 출력도 함께 남긴다.
4. `cc_g2v4_confirm.sh`는 rev2 §6에 적은 잠금 전 SHA에서 13행 자리표시만 바꾼다. 변경 전후 diff(1행)와 최종 SHA를 STATE에 적는다.
5. 확증은 CC의 확증 TASK 발행 뒤에만 한다. §4 명령과 다르게 해야 할 일이 생기면 실행 전에 개정과 B 감사를 거친다. FIX-1(c)에서 적중이 있으면 v1(1,234) 기준으로 §2·§4·§6과 스크립트의 SHA를 바꾼 rev3를 B가 다시 본다.

판정: **FIX 5건(재감사 필요)**

## 재감사 1 (B, 2026-10-11 17:3x)

- 대상: `shared/prereg/PREREG_S7_G2v4.md` rev2, **SHA-256 `df2e219bf9dcc30752644b865093c6dceda5f86632bcefb8e23d13b14b6a9209`**(로컬 실측, 요청 값과 일치), 268행. rev1 사본 `drafts/PREREG_S7_G2v4_rev1.md` 4563782a…(로컬 일치).
- 원칙은 같다. 서버 접속 0, 로컬만, 자격 증명 파일 0, 이 파일에만 덧붙임. 이번에는 `shared/ssh_log.csv` 357~410행을 읽을 수 있었다.
- 로컬 실측 SHA(요청·본문 값과 모두 일치):
  - `scratch/cc_g2v4_evidence.txt` a745295d…, `scratch/cc_g2v4_select_outputs.txt` 7e199c1f…
  - `shared/reports/g2v4dev/`: `dev_score.json` e890c3c2…, `dev_score.md` b67af189…, `cc_g2v4_dev_m7.log` 6d8522ad…, `g2v4b_server_json_union.json` cd7c8f1f…
  - `g2v4m7rep/diag_Ez.json` 5c8fa506…
  - `scripts/cc_g2v4_confirm.sh` 75fd4a05…, `g2v4_idscan.py` f0468a13…, `g2v4_idscan_v2.py` 79e8b406…, `g2v4_proximity.py` bdd3730c…
  - `shared/GRAPH.md` ef027244…
- 서버 산출물(.pt)의 SHA는 여전히 로컬에서 직접 잴 수 없다. 대신 evidence [A]·[D]의 서버 `sha256sum` 원문으로 대조했다.

### 반영 점검

| FIX | 판단 | 근거 |
|---|---|---|
| FIX-1(a) 개발 원자료 | 반영 | 로컬 `cc_g2v4_dev_m7.log`가 evidence 3~34행과 한 글자도 다르지 않다(diff 0). 서버 원문에서 `decoder_Ah.pt` = `decoder_M7A_v36.pt` = 46977c2d…, `decoder_Eh.pt` = `decoder_M7E_v36.pt` = d0c33599…(evidence 9·15·37~40행), LOAD-GUARD 436·0 × 2(17~18행), PROJ-V2 × 2, CHECK 600/600 True × 2, SCORE rc 0을 확인했다. 로컬 `dev_score.json`(e890c3c2…)의 Gate_4 strict = Δ 0.03, CI [0.005, 0.05667], 39/21, p 0.01367, Gate_3 Δ 0.075. 본문 42~43행과 같다 |
| FIX-1(b) 선택 출력 | 반영 | `cc_g2v4_select_outputs.txt`: v1은 검사 파일 120, 합집합 3,764, 적중 524(run1 목록·샤드 네 파일뿐), 선택 1,234, 302dffc4…. v2는 검사 파일 116, 합집합 3,240, 적중 0, 선택 1,758, 6e68e34c…. 검사 파일 차이 4는 run1 목록 네 개다(`g2v2_exclude_run1`은 v1에서도 `g2v2_exclude` 패턴으로 빠짐). 3,764 − 524 = 3,240으로 산술이 맞는다. 보관 파일 서버 시각은 10/08 15:24:22(evidence 73~77행). 로컬 사본 union(cd7c8f1f…)은 정수 3,240·파일 116이고 출처에 run1 파일이 없다(B 실측). ssh_log 399~403행 순서: v1 16:33:48 → run1 점검 16:34:51 → v2 16:37:25. 개발 결과 확인(16:31:27 poll) 직후 6분 안에 파일 출처만 보고 결정했다 |
| FIX-1(c) 524 미사용 직접 근거 | 반영(포인터 1건 부족 → 재FIX-1) | `g2v4_idscan.py`(로컬 판독): 두 루트 × `results_s7`·`raw_s7`·`scratch`·`results` 아래 `.jsonl`의 `id`·`q_idx`·`question_idx`·`qid`와 `.pt` 최상위 키(`q\d{4}`·정수)를 1,758과 교차하고, 524·1,234로 나눠 출력한다. 문항 글은 읽지 않는다. evidence 96~97행: jsonl 72, pt 374, 건너뜀 0, **적중 0**. run1 점검 재실행(78~93행): run1 파일 이름 참조 grep 0건, 10/08 15:00~16:10 생성 결과 파일은 `results_s7/p6r/smoke/v36e_smokeA.jsonl` 하나뿐이다(나머지는 configs 목록 파일). 다만 양성 대조(`g2v4_idscan_v2.py`, fresh1000 → "23개 파일 적중", 본문 146행)는 원시 출력 파일이 없다. ssh_log 408행 `cc_idscan_posctl` 접속 기록은 있다. R1-1·R3-5상 본문 수치에 포인터가 필요하다 |
| FIX-1(d) 학습 근거 | 반영 | evidence 99~141행: Ah·Eh `[DATA] Filtered 993 / 993`, 학습 가능 매개변수 같음(218/3,980,801, 6/19,665,921), center norm 312.6(z*)·191.9(z_src), 최종 l_task 1.1973·1.2021, Ah IFT 0.016389, 매니페스트 SHA dbdf21bf…·0264a337…, `n_text_dropped` 6061 = 6061, 학습 입력 005e622d…·340ef5c2…, 학습 문항 3088a153…. 본문 99~110·243행과 같다 |
| FIX-2 선택 경로 | 반영 | B 재계산(`diag_Ez.json`): Ez 자기 0.9010, 다음 − 자기 +0.0029(z 2.67), 글 없음 +0.0040(z 4.23), 이동 중앙값 0.159. Az − Ez −0.0324(z −14.64), A가 나은 문항 77.7 %. 본문 36~40행과 같다. 불리한 관찰("글 가리기 1.0에서는 E도 문항 특이성")도 38·262행에 적었다. 선택 기준·시각(49~51행): 개발 기동 14:22:57(evidence 3행), ssh_log 390~394행(m7rep 수거 14:20, 개발 기동 14:22). Ez가 아직 끝나지 않았다는 주장은 Az→Ez 순차 실행(`cc_g2v4_m7_rep.sh:22`)과 맞는다. 설정 목록 표 24~34행. LoRA·학습 문항 진단은 서버 경로 포인터이고 DIAG §7·§8에 SHA가 있다 |
| FIX-3 공정성·주장 범위 | 반영 | (a) 103~104·120행: λ_ift가 다르다는 것과 A 최종 l_ift 0.0164(evidence 107행). (b) 121~123행: 시작점과 λ_ift를 가르지 못한다는 것과 0.8514 / 0.8503 / −0.0236(diag 요약 키), 261행 결과 문서 의무. (c) 87~91행 center 정의·norm·이동 12.7 %/10.7 %, 대칭 규칙·미측정. (d) 105~110행. (e) 111~114행: DIAG §7 계획에서 벗어난 사실과 이유 |
| FIX-4 GOAL·보고·가설 | 반영 | (a) 255~257행과 §4-7(192~194행). `g2v4_proximity.py`(로컬 판독)는 잠재 텐서만 읽는다. A = `fresh_A_latent` 대 `fresh_zsrcinj_1758`의 `z_star`, E = `fresh_E_latent` 대 `fresh_zsrc_1758`의 `z_star`(= z_src). 키 조회가 맞다. (b) 260행. (c) GRAPH.md 42행 H-DEC-7: 등록 시점, 열람 범위, λ_ift 차이, 시험 = 새 1,758 1회, 기각 조건 = 확증 미충족 |
| FIX-5 스크립트 가드·본문 | 반영(본문 1행 근거 부족 → 재FIX-2) | (a) `cc_g2v4_confirm.sh:12~15`: `$O`·`$M`·z 샤드 셋·병합 둘·잠재 둘이 하나라도 있으면 중단하고, `mkdir`은 그 뒤(36행)다. (b) 17~35행 `chk` = §6 ①의 모든 행(실행 중 import되는 세 파일 포함), 본문 163행이 ①·②를 구분한다. (c) §6 ③ 250~251행: 잠금 전 SHA 75fd4a05…(B 실측 일치), 자리표시는 17행 하나(B 확인). R-7 반영: 57행 LOAD-GUARD 강제, 71행 PROJ-V2 9개 로그 강제. 3차 스크립트 대비 다른 차이는 경로·N뿐이다. 남은 점: §6 233행은 서버 `check_p6_decoder_version.sh`(1fa95f72…, evidence 205행)가 로컬본(cc685dcb…)과 "줄바꿈 등" 바이트가 다르다고 적었다. 그런데 로컬본에는 CR이 없고, CR을 지운 SHA도 cc685dcb…로 그대로다(B 실측). 차이의 정체는 확인되지 않았다. 실제로 실행되는 서버본의 내용은 B가 본 적이 없다 |

- 권고 반영: R-1(186~189행 하위 키), R-2(200~203행, 재개 스크립트 diff를 B가 확인), R-3(204~206행), R-4(155행, `p6_gen_C.jsonl` 24c82d97… = evidence 12행), R-5(3·141행), R-6(160행), R-7(스크립트 57·71행), R-8(267행), R-9(247행·268행, 미점검 사실 공개). 모두 수용한다.
- §0.5 AG 확인: `ssh_log.csv` 359행 `10:22:49 ssh:cts-server echo '=== SYSTEM ==='; uptime; tmux ls…`(exit 2), 360행 `10:23:54 ssh:poll_lora_pilot`(0). 본문 70행과 같다. 361~410행은 모두 `cc_` 접두다. 1차 감사 때 확인하지 못한 항목이 해소됐다.
- evidence [E] 209행: 16:55에는 서버에 `g2v4_fresh_prep.py`가 없었다. ssh_log 409~410행(17:07 `cc_up_prep4`, 17:08 `cc_sha_prep4`)으로 올렸다. 확증 스크립트 23행 `chk`가 실행 시 대조한다.
- 경미(차단 아님): `g2v4_proximity.py`(§4-7)는 확증 스크립트에 들어 있지 않고 §6 ②(기록용)에 있다. 판정과 무관한 보고 항목이니, 확증 종료 뒤 `sha256sum scripts/g2v4_proximity.py` 출력과 함께 한 번 실행하고 그 원문을 결과 문서에 인용하면 된다(잠금 조건 6).

### 재FIX 목록 (경미 2건, 문서 2행 + 원시 출력 2개)

- **재FIX-1 (양성 대조 원시 출력)** 서버에서 `g2v4_idscan_v2.py configs/g2v2_fresh1000.json`(읽기 전용, 번호만)을 다시 돌린다. 출력 전문을 `scratch/cc_g2v4_idscan_posctl.txt`로 남기고 SHA를 STATE에 적는다. 본문 146행에 그 파일을 포인터로 단다. 원문에 `[HIT]` 23행과 `fresh_A_latent.pt`·`p6_gen_*.jsonl`의 1000 적중이 보여야 한다. 수치가 다르면 원문 값으로 고친다.
- **재FIX-2 (서버 버전 가드 내용)** 서버 `scripts/check_p6_decoder_version.sh`(1fa95f72…)를 `cts_ssh.py` sftp로 내려받아 `shared/reports/g2v4dev/check_p6_decoder_version.server.sh`로 둔다(R7-9). 로컬본과의 `diff -u`를 STATE 또는 같은 폴더에 남긴다. 본문 233행의 "줄바꿈 등"을 실제 차이로 고친다. B는 재감사 2에서 서버본이 경로·조건만 받아 종료 코드만 내는지, `decoder_M7A_v36.pt`·`decoder_M7E_v36.pt`를 받아들이는지만 본다.

### 재감사 2 범위와 잠금 조건

1. rev3 = rev2에서 146행(포인터)과 233행(문구)만 바꾼다. 그 밖의 행이 바뀌면 그 행도 감사한다. rev2(df2e219b…)는 `drafts/`에 사본으로 둔다. B는 rev2 → rev3 `diff`와 위 두 원시 출력만 본다. 확증 스크립트(75fd4a05…)는 바꾸지 않는다. 서버본 가드 내용 때문에 스크립트를 바꿔야 하면 그 diff도 본다.
2. 재감사 2 PASS 뒤에는 제목을 "잠금본"으로, 3행 상태 줄을 "LOCKED (시각) — AUDIT-P014 재감사 2 PASS(rev3 SHA)"로만 바꾼다(4~5행 부속 줄은 그대로 둔다). 잠금본 이름은 `shared/prereg/PREREG_S7_G2v4.md`이고, B가 rev3 대비 diff(제목·상태 두 행)로 확인한다.
3. 잠금본 SHA를 STATE에 적는다. 서버에 `shared_prereg_PREREG_S7_G2v4.md`로 올리고 `sha256sum`·`date` 원시 출력을 남긴다. 같은 접속에서 `ls results_s7/g2v4 results_s7/p6r/g2v4fresh results_s7/p6r/g2v4fresh_merged`가 없음을 보이는 출력도 남긴다(R7-3, 한 접속에 묶음).
4. `cc_g2v4_confirm.sh`는 75fd4a05…에서 17행 `__PREREG_SHA__`만 잠금본 SHA로 바꾼다. 변경 전후 `diff`(1행)와 실행본 SHA를 STATE에 적고 서버에 올린 뒤 서버 SHA를 대조한다.
5. 확증은 CC의 확증 TASK 발행 뒤에 §4-0 기동 명령으로만 한다. 중단되면 §5 재개 규칙(재개 스크립트 diff를 B가 사전 확인)을 따른다. 그 밖의 변경은 실행 전에 개정과 B 감사를 거친다.
6. 결과 문서에는 §4-7 근접도(`g2v4_proximity.py` SHA 출력 포함), p × 4, λ_ift 혼입·GOAL 해석 문장(§7)을 넣는다. 판정은 `gate_table.json`의 §4-6 키만으로 한다.

판정(재감사 1): **FIX 2건(재감사 필요)** — 경미. 재감사 2는 위 두 건과 rev2 → rev3 diff만 본다.

## 재감사 2 (B, 2026-10-11 17:2x~17:3x)

- 대상: `shared/prereg/PREREG_S7_G2v4.md` rev3, **SHA-256 `82f969788cf59d95aee06ee854a294599b273e2702fc1fcd6d10eb542ed01d80`**(로컬 실측, 요청 값과 일치). rev2 사본 `drafts/PREREG_S7_G2v4_rev2.md` df2e219b…(로컬 일치).
- 원칙은 같다. 서버 접속 0, 로컬만, 이 파일에만 덧붙임.
- 로컬 실측 SHA:
  - `scratch/cc_g2v4_idscan_posctl.txt` 13ec0052…
  - `shared/reports/g2v4dev/check_p6_decoder_version.server.sh` 1fa95f72…(evidence [E] 205행의 서버 값과 일치)
  - `…local_vs_server.diff` 65debba2…
  - `scripts/cc_g2v4_confirm.sh` 75fd4a05…(바뀌지 않음)
- **rev2 → rev3 diff(B 실측)**: 146행과 233행 두 줄만 다르다. 3행 상태 줄을 포함한 나머지 266행은 rev2와 같다. 따라서 재감사 1에서 PASS로 본 내용은 모두 그대로다.

| 재FIX | 판단 | 근거 |
|---|---|---|
| 1 양성 대조 원시 출력 | 반영 | 머리에 `g2v4_idscan_v2.py` 79e8b406…와 인자 목록 `g2v2_fresh1000.json` 35683874…의 서버 SHA가 찍혀 있다(2~3행). `[SCAN] jsonl 72 \| pt 374 \| skipped 0`(5행)은 본 검사(evidence 96행)와 같은 파일 수다. `[HIT]` 23행(6~28행)의 내역: `g2v3/fresh_{A,E}_latent.pt`·`fresh_zsrcinj_1000.pt`·`fresh_zsrc_1000.pt`와 병합 `p6_gen_{A,B1,C,E1}.jsonl`이 각 1000, 샤드 z 3개와 샤드 생성 12개가 각 334/333. 3차 확증 산출물이 빠짐없이 잡혔다. 따라서 같은 검사기가 1,758에서 적중 0을 낸 것은 의미 있는 음성 결과다. 출력의 "hits on 1,758"과 "in 524 / in 1,234" 열은 v2 스크립트가 F만 인자로 바꾸고 이름표를 그대로 둔 것이다(`g2v4_idscan_v2.py` 8행; V1 ∩ fresh1000 = 0이라 "in 524" = 전체). 146행이 이를 표시 문구라고 밝혔다. 차단 아님 |
| 2 서버 버전 가드 내용 | 반영 | 로컬본과 서버 사본의 `diff`는 서버본 끝에 빈 줄 1행이 더 있다는 것(44행)뿐이다(B 실측. `.diff` 파일 내용과 같다). 서버본도 파일 이름 `_v36(\.pt\|_step….pt)$` 검사와 존재 검사만 하고 종료 코드만 낸다. `decoder_M7A_v36.pt`·`decoder_M7E_v36.pt`를 받아들인다. 233행 문구가 이 사실과 맞는다. 확증 스크립트 35행 `chk` 값 1fa95f72…도 서버 사본과 같다 |

- 참고(차단 아님): rev3 3행 상태 줄이 아직 "DRAFT rev2"다. 잠금 때 상태 줄을 바꾸면서 함께 고친다(아래 잠금 조건 1).

### 잠금 조건

1. 제목(1행)을 "잠금본"으로, 3행 상태 줄을 "LOCKED (시각) — AUDIT-P014 재감사 2 PASS(rev3 82f96978…)"로만 바꾼다. 4~5행과 그 밖의 모든 행은 rev3와 같아야 한다. rev3 사본을 `drafts/PREREG_S7_G2v4_rev3.md`로 먼저 보존한다(R3-4). 잠금본 이름은 `shared/prereg/PREREG_S7_G2v4.md`다. rev3 대비 `diff`(1·3행 두 줄)를 STATE에 남긴다.
2. 잠금본 SHA를 STATE에 적는다. 서버에 `shared_prereg_PREREG_S7_G2v4.md`로 올리고, 한 번의 접속(R7-3)에서 다음 원시 출력을 남긴다:
   - `sha256sum`과 `date`
   - `ls results_s7/g2v4 results_s7/p6r/g2v4fresh results_s7/p6r/g2v4fresh_merged`(셋 다 없어야 한다)
3. `cc_g2v4_confirm.sh`는 75fd4a05…에서 17행 `__PREREG_SHA__`만 잠금본 SHA로 바꾼다. 변경 전후 `diff`(1행)와 실행본 SHA를 STATE에 적고, 서버에 올린 뒤 서버 SHA가 같음을 원시 출력으로 남긴다. 그 밖의 변경은 개정과 B 감사를 거친다.
4. 확증은 CC의 확증 TASK 발행 뒤에만, 본문 §4-0 기동 명령으로 한 번 한다. 중단되면 §5 재개 규칙(재개 스크립트 diff를 B가 사전 확인)을 따른다.
5. 판정은 `gate_table.json`의 §4-6 키만으로 한다. 결과 문서에는 다음을 함께 적는다:
   - §4-7 근접도(실행 직전 `sha256sum scripts/g2v4_proximity.py` 출력 포함)
   - p × 4 참고값
   - §7의 λ_ift 혼입·GOAL 해석 문장

판정(재감사 2): **PASS(잠금 가능)** — 위 잠금 조건 1~3을 지킨 뒤 확증 TASK로 진행한다.

## 재개 1 사전 확인 (B, 2026-10-11 22:xx) — 잠금본 §5 재개 규칙

- 원칙은 같다. 서버 접속 0, 로컬만, 이 파일에만 덧붙임. 서버 쪽 사실(디스크 100 %, `[Z-DONE]` 3개, 잘린 파일 크기, $O·$M 없음)은 CC 보고와 STATE 끝 행을 그대로 받았다. 원시 출력은 보지 못했다. 재개 스크립트의 16~22행이 이 상태를 실행 시점에 다시 확인한다.
- 로컬 실측 SHA(요청 값과 모두 일치):
  - 잠금본 `PREREG_S7_G2v4.md` 5923fef7…(1~3행 LOCKED, rev3 82f96978… 기준)
  - 실행본 `cc_g2v4_confirm.sh` d74fd9a9…(17행 = 잠금본 SHA. 잠금 전 75fd4a05…에서 이 행만 바뀐 것은 STATE 17:16:55 기록)
  - `cc_g2v4_confirm_resume1.sh` a2314296…
  - `scratch/DIFF_g2v4_confirm_to_resume1.patch` 69a5ae77…(95행, 일반 `diff` 형식)
  - `g2v4_fresh_prep_r1.py` e8876570…, `g2v4_proximity_r1.py` 0d602101…
- STATE 끝 행에 재개안과 diff SHA가 실행 전에 기록돼 있다(§5 "diff를 실행 전에 STATE에 기록").

### 점검

| # | 항목 | 판단 | 근거 |
|---|---|---|---|
| 1 | §5 적용 범위 | 맞음 | `gate_table.json`이 생기기 전(채점·생성 전)에 멈췄다. 원인은 디스크 용량이라는 환경 결함이고 결과 정보와 무관하다. §5 재개 조항(200~203행)이 바로 이 경우를 다룬다 |
| 2 | "이어 쓰기만 허용"과의 관계 | 취지와 맞음(더 엄격) | 실패한 단계(§4-2 prep)부터 새 출력으로 다시 한다. 이어 쓰기도, 부분 출력 재사용도 없다. 완료된 앞 단계 출력(z 샤드 3개)만 그대로 쓴다. 이는 "실패한 단계부터 같은 명령으로" 하라는 규칙에 맞는다. z를 다시 계산하지 않는 것도 타당하다(결정적 계산이고, 다시 하면 디스크만 더 쓴다) |
| 3 | 명령 동일성 | 맞음 | `g2v4_fresh_prep_r1.py`는 원본 ed79108f…와 비교해 머리 주석과 18~21행 출력·재적재 경로 4곳만 다르다(B diff). 키 집합 assert, max\|Δ\| = 0 assert, r_1000 보고는 그대로다. 재개 스크립트 57~99행은 원본 53~93행에서 잠재 경로만 `$G` → `$W`로 바꾼 것과 한 글자도 다르지 않다. 차이는 주석과 `mkdir -p $O` 위치(원본 36행 → 65행)뿐이다(B가 치환 뒤 diff로 확인). 문항·샤드·체크포인트·GPU 대응·생성 순서·병합 assert·채점 명령·판정 키는 같다. `proximity_r1`은 10행 G 경로만 다르다 |
| 4 | z 샤드 무결성 | 충분 | 실행기는 마지막 `torch.save`(`g2v2_zstar_run.py:160`)가 실패하면 예외로 끝난다. 그러면 원본 스크립트 49행이 `[FATAL] z shard`를 냈을 것이다. `[Z-DONE]` 3개는 저장이 성공했다는 뜻이다. 원 prep이 `torch.save`(18행)에서 실패했다는 것은 그 앞 15행(세 샤드 적재와 키 집합 = 1,758 assert)을 통과했다는 뜻이기도 하다. `prep_r1`이 같은 assert를 다시 한다. 재개 스크립트 18~19행(비어 있지 않음·`[Z-DONE]`)과 41행 SHA 기록도 있다 |
| 5 | /tmp 사용 | 받아들임(제한 공개) | 중간 텐서(약 6 GB)만 W에 둔다. 판정에 들어가는 생성 jsonl과 채점 산출물은 원래 경로(/home, v36f realpath 가드 통과)에 쓴다. 남은 위험은 다음과 같다. /tmp는 재부팅이나 정리 정책으로 사라질 수 있다. 다른 계정이 읽거나 쓸 수 있다. 실행 뒤 §4-7 근접도 계산에 W 파일이 필요하다. 모두 결과를 바꾸는 위험은 아니고, 잠재 SHA(56·64행)가 생성 직전에 찍힌다. 결과 문서에 공개한다 |
| 6 | 결함 파일 이동 | 받아들임 | 삭제가 아니라 mv로 보존한다(R2-1). 잘린 2.7 GB 파일은 정보 가치가 없고 이후 단계가 읽지 않는다. /home → /tmp는 다른 파일시스템이라 복사 뒤 원본을 지우는 방식이지만, /tmp 15 GiB 점검(48행)이 mv 뒤라서 복사 공간은 34 G 여유에 기댄다. 이동 전 SHA는 찍지 않는다(아래 권고 a) |
| 7 | 사전 점검·가드 | 대체로 충분 | $O·$M·W 없음, 결함 파일 있음(16~21행). 잠금본·원 실행본 SHA를 포함해 실행에 쓰는 파일 18개를 `chk`(24~40행. 이번에 다시 돌리지 않는 z 실행기·yaml·원 prep은 빠짐. 타당). mv 뒤 /home ≥ 1 GiB·/tmp ≥ 15 GiB(47~48행), GPU 빈 상태(49~53행), LOAD-GUARD·PROJ-V2 강제(62·77행), 병합 assert(91행). 마스터 로그를 /tmp에 쓰는 것은 /home 쓰기 실패를 피하려는 것으로 적절하다 |
| 8 | 디스크 재고갈 위험 | 공개 필요(차단 아님) | /home을 채운 원인이 밝혀지지 않았다(ESC-0009). 생성 약 9시간 동안 다른 작업이 다시 1 GiB를 써 버리면 v36f 쓰기가 실패해 멈출 수 있다. 그때는 §5 재개 2(이어 쓰기, B 사전 확인)로 처리하며, 결과를 바꾸지는 않는다 |

### 권고 (차단 아님 — 넣는다면 아래 행만 추가하고, 추가 diff를 STATE에 적은 뒤 실행한다. B는 사후에 확인한다)

- (a) 44행 mv 앞에 `sha256sum $G/fresh_zsrcinj_1758.pt`를 둔다(결함 파일 식별용).
- (b) 21행 근처에 `$G/fresh_zsrc_1758.pt`·`$G/fresh_A_latent.pt`·`$G/fresh_E_latent.pt`가 없음을 확인하는 행을 넣는다(멈춘 상태 그대로인지 단언).
- (c) 42행을 `mkdir -m 700 -p $W/defect`로 바꾼다(다른 계정의 접근 차단).
- (d) 각 샤드 로그 `raw_s7/g2v4_fresh_z_shard{k}.log`에 `Finished. Total items saved: 586`이 있는지 확인한다.

### 결과 문서에 적을 것

- 중단 시각·원인(/home 100 %), 결함 파일 이동 경로·크기, 재개 스크립트·`prep_r1`·`proximity_r1`의 SHA, z 샤드 SHA(41행 출력), W 경로의 prep·잠재 SHA.
- "재개 1은 §5에 따라 실패한 단계부터 같은 명령으로 실행했고 이어 쓰기는 없었다"는 문장.
- 원 마스터 로그(`raw_s7/cc_g2v4_confirm_master.log`)와 재개 마스터 로그(/tmp → raw_s7 사본)를 둘 다 포인터로 단다.
- 근접도는 `g2v4_proximity_r1.py`(0d602101…)로 W 파일이 남아 있을 때 계산한다. W 파일이 사라졌으면 그 사실을 적는다(판정과 무관).

판정(재개 1 사전 확인): **실행 가능** — 재개 스크립트 a2314296…를 그대로 실행해도 된다. 권고 (a)~(d)는 선택이다.

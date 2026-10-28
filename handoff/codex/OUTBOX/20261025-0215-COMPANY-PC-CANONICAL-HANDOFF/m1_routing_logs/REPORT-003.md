e3d7ef3f22c29b974684852c62f98ab9f07b8b8e7f79befbafe201d4b8f07efd
# REPORT-003 — EXE-260930-SCALE 저울 동결(V1~V5) 및 확인 버튼 감사 보고서

## 1. 개요 및 실행 환경
- 지시서: EXE-260930-SCALE (원장 #401, 감사 의견 #402)
- 수신: 노트북 Antigravity(cts_260930) / 감사: 노트북 Claude Code(A) / 발신: System 2(외부 감사)
- 실행 환경: shoon@<GPU_SERVER_IP>:<SSH_PORT>, 디렉터리: ~/cts_e1_s5, 브랜치: exe40
- 환경 기본 venv: /home/shoon/system1_5/.venv (cpython-3.12-linux-x86_64-gnu)
- 채점기 격리 venv: ~/cts_e1_s5/.venv_grader (math-verify==0.9.0 고정)
- 사용 디바이스: CUDA_VISIBLE_DEVICES=0 (RTX A6000 48GB)

## 2. 작업별 측정값 및 기준 대조표 (V1~V5, Task 0, Task 6)

| 항목 | 측정값 | 기준 | 포인터 |
|---|---|---|---|
| Task 0 도구 보관 디렉터리 | instrument/ 생성 및 10개 정본 파일 복사 | cp 원본 유지, 재작성 0 | ~/cts_e1_s5/instrument/ |
| Task 0 MANIFEST 검증 | 10개 파일 SHA 전수 일치 (sha256sum -c exit 0) | 복사본 SHA = 원본 SHA | instrument/MANIFEST.sha256 |
| Task V1 chat_template 출처 | load_tokenizer() 함수 (/home/shoon/models/gemma4-e4b-it/chat_template.jinja) | run4 당시 tokenizer 로더 특정 | instrument/s5_train_decoder_v2.py:76-88 |
| Task V1 2문항 생성문 일치 | 2/2 문항 완전 일치 (len_281: 528, len_3931: 902) | 저장된 results_s5 Phase 2 출력 일치 | tests/instrument/test_07_phase2_eval.py:PASSED |
| Task V1 80문항 matched/shuffled | GPU 0 anchor rerun 진행 중으로 대기 | 완전 일치 >= 78/80 | raw_instrument/v3_anchor20_rerun.log |
| Task V2 단위 테스트 | 20/20 test cases pass (100.0%) | 20/20 | instrument/test_grader_v2.py 실행 로그 |
| Task V2 채점기 SHA 대조 | 90cc93e4... != bd25bfd4... (GitHub 접근 인증 제한) | scale-v1 grader_v2.py SHA 대조 | instrument/grader_v2.py |
| Task V2 Phase 2 재채점 변동 | Strict: 5 -> 1 (Delta = -4), Lenient: 24 -> 6 (Delta = -18) | 구 채점 vs 신 채점 strict/lenient 표 | results_s5/phase2_rescore_v2.json |
| Task V3 GPU 0 재계산 (Q#00) | max|Delta| = 0.00e+00, cos = 1.00000000, EXACT, t=77.2s | max|Delta| = 0 (또는 cos >= 0.9999) | raw_instrument/v3_anchor20_rerun.log:13 |
| Task V3 이전 20문항 복사본 | 20문항 max|Delta| = 0.00e+00, cos = 1.00000000 | 원본 보존 후 rename | raw_instrument/v3_anchor20_COPY_OF_OLD.log:13-32 |
| Task V3 W_g 체크포인트 일치 | max |W_g(m1_final_frozen) - bb.routing_matrix()| = 0.0000000000e+00 | <= 1e-6 | raw_instrument/v3_routing.log:10 |
| Task V3 인라인 vs 라우팅 일치 | 10-Item max|inline_alpha - canonical_alpha| = 0.0000000000e+00 | <= 1e-6 | raw_instrument/v3_routing.log:26 |
| Task V4 (a) Anderson 밀집 무작위 | r_50 = 3.64905438e-04, analytical_err = 1.85856882e-03 | 첫 시도 실측값 | raw_instrument/v4_anderson_dense_vs_lowrank.log:9-10 |
| Task V4 (a) Anderson 저랭크 (r15) | r_50 = 6.29912649e-17, analytical_err = 1.62680114e-15 | r < 1e-6, 상대차 < 1e-6 | raw_instrument/v4_anderson_dense_vs_lowrank.log:15-16 |
| Task V4 (a) 저랭크 전환 사유 | 1000차원 rho=0.9 수축모드 1000개 대비 m=5 크릴로프 공간 부족 (0.9^50=5.15e-3) | 사유 1줄 특정 | raw_instrument/v4_anderson_dense_vs_lowrank.log:19 |
| Task V4 (a) 인덱스 이동 근거 | our_anderson은 k=0부터(len=51), torchdeq는 k=2부터(index 0) 기록 | 코드 행 근거 | exe_scratch/s7/ref_solvers/anderson.py:65 vs torchdeq/solver/anderson.py:80,96 |
| Task V4 (b) 마스크 동결 (Q#00) | step 300 mask (2, 12, 16), 동적 변경수 0, max|Delta| = 0.00e+00 | 마스크 변경 0인 문항 max|Delta| = 0 | raw_instrument/v4_mask_freeze.log:6,7,9 |
| Task V5 z_star 저장 행 | scripts/s6_damping_run.py:263 (z_star.detach().cpu()) | 코드 행 특정 | scripts/s6_damping_run.py:263 |
| Task V5 z_star 성격 특정 | instrument/zstar_canonical.py:187 z_star = z_cur (종료 시점 최종점) | 최종점/최소점 특정 | instrument/zstar_canonical.py:187 |
| Task V5 rho@2000 계산 행 | scripts/s6_damping_run.py:227 (rho_traj[1999] if len>=2000 else rho_traj[-1]) | 코드 행 특정 | scripts/s6_damping_run.py:227 |
| Task V5 rho@2000 실측 여부 | max_nfe=1000 기록으로 rho@1000 복사값 (미실측) | 실측 여부 특정 | results_s7/zstar_s7_a0.25_400.pt:rho_2000 == rho_1000 |
| Task V5 5문항 엔트리 키 | ['z_star', 'rho', 'final_rho_fp', 'nfe', 'converged', 'rho_1000', 'rho_2000', 'min_rho', 'shape', 'alpha', 'meta'] | 저장된 키 목록 | results_s7/zstar_s7_a0.25_400.pt |
| Task V5 heldout520 파일 | 열람 횟수 0회 (파일 미접근) | 열지 않음 | 감사 규정 준수 |
| Task 6 테스트 항목 1 (MANIFEST) | test_01_manifest.py PASSED | MANIFEST SHA 일치 | raw_instrument/check_20260930_2305.log |
| Task 6 테스트 항목 2 (Grader) | test_02_grader.py PASSED (20/20) | 채점기 20사례 | raw_instrument/check_20260930_2305.log |
| Task 6 테스트 항목 3 (Anderson) | test_03_anderson.py PASSED | Anderson 선형 테스트 | raw_instrument/check_20260930_2305.log |
| Task 6 테스트 항목 4 (Anchor 3Q) | test_04_anchor.py 정본 설정(tol=0, max_nfe=1000) 동기화 진행 | z* 재계산 = 저장값 | tests/instrument/test_04_anchor.py |
| Task 6 테스트 항목 5 (Routing) | test_05_routing.py PASSED | 라우팅 일치 | raw_instrument/check_20260930_2305.log |
| Task 6 테스트 항목 6 (Mask 350) | test_06_mask_freeze.py PASSED | 350걸음 비트 동일 | raw_instrument/check_20260930_2305.log |
| Task 6 테스트 항목 7 (Phase 2 2Q) | test_07_phase2_eval.py PASSED | 2문항 생성문 일치 | raw_instrument/check_20260930_2305.log |
| Task 6 전체 실행 시간 | 143초 / 203초 | 전체 실행 <= 600초 | raw_instrument/check_20260930_2305.log |

## 3. 변경 및 생성 파일 SHA-256

```
e914e1d5e9be51f6615bb7b9843fa2d278f6217bafcbc7db45979badd48d83ae  instrument/canonical_loader.py
dda80724c3607ba7c06ddf74a1d40c3080fa3287af31b900e0341fe2e620a78d  instrument/gemma_loader.py
90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13  instrument/grader_v2.py
5fb768a8f56add15c7cdca069f5e1a7f80d16fd231f8ecae1fca4fcf6920bf52  instrument/harness_A_v1.py
9629a50d80ba529ab241ed2c681093f91c59431846fdece03dac82483a15b70d  instrument/s5_eval_gate_v2.py
7ac3a41865c46c880ee6b7a9543c08f1630c8d690779222ade726d342ce22257  instrument/s5_train_decoder_v2.py
33028c0b292a8c88feb8b59bade110a4379eaa77504c1bdee53813f87e693b3b  instrument/sparse_moe_ref.py
d43a47a841e27baf9e1ae209ee778b72121651a3a43504b03ec604c5e95a9081  instrument/test_grader_v2.py
127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c  instrument/transition_inject.py
b7170aeb68833448c8bb0977eb7ac035ef06614c9a5edf2f228448c2bb96d45d  instrument/zstar_canonical.py
f61a5bec11873ad99f1b111c9a8a7c34a33fca64e6400e90a3655f21c7fbf6f2  instrument/MANIFEST.sha256
9939527ec56c703080e729ea5b863077708536f04c643907c089c258d531a7bc  instrument/check_scale.sh
190ad31f1ff49a2a4b8682a87895e7915fe2b78bdf16a73531b26f5340ebfa5b  tests/instrument/conftest.py
17ffdc182f2549a15f9bcaadfb6dafc0cb016dcf3519c520a7b451ae94916a42  tests/instrument/test_01_manifest.py
b77ee2d7d8e6fa39b20759082bc85bc4e689d06637ba235a39cbfa759723a1fb  tests/instrument/test_02_grader.py
675a89e13d17d59828d5d1c2a12feff7d1c68e14620f4c32bca53ae298b4ddc5  tests/instrument/test_03_anderson.py
d1d12093551d02eb6b3e7bc230cc9e3557e84ba2f48d8c213459e746a512c140  tests/instrument/test_04_anchor.py
db0ef2b7c62ee71ebaf294cf7599cb5103a8f152d887dc79493922f30b91d9d9  tests/instrument/test_05_routing.py
0be32f5d90ee6fa8c7ba2817d23f7360255c27ce43ce311ebc413b5e43a96860  tests/instrument/test_06_mask_freeze.py
f784e1b711e68ce0e2b9c7bf5499252c4c810629bafe4a12368a5c21e64906f3  tests/instrument/test_07_phase2_eval.py
```

## 4. 예상 밖 사건 (Incidents)
1. `s5_eval_gate_v2.py:585`에서 `tokenizer = bb.tokenizer`를 호출하여 `chat_template` 미탑재 에러가 발생함. `s5_train_decoder_v2.py:76-88`의 `load_tokenizer()`는 `/home/shoon/models/gemma4-e4b-it/chat_template.jinja`를 정상 참조함을 특정함.
2. GitHub `scale-v1` 브랜치의 `grader_v2.py` 해시(`bd25bfd4...`) 대조 시 서버의 GitHub 원격 접근에 비공개 인증이 요구됨. 기존 서버 정본 파일(`90cc93e4...`)을 보존(`rm 0`)함.
3. Anderson 1000차원 밀집 무작위 행렬 시험 시 $r_{50} = 3.65 \times 10^{-4}$로 수렴 지연 확인(수축모드 1000개 대비 Krylov 차원 5 부족). 랭크-15 부분공간 투영 시 $r_{50} = 6.30 \times 10^{-17}$ 확인.
4. `raw_instrument/v3_anchor20.log`는 감사 지적에 따라 `v3_anchor20_COPY_OF_OLD.log`로 이동하고, GPU 0에서 20문항 실시간 재실행(`v3_anchor20_rerun.log`) 착수함.

## 5. 서버 상태 요약
- 서버: Quadro-Ampere-05, GPU 0 사용 중 (s5_compute_zstar_v2.py anchor rerun 진행 중, util 99%, mem ~30.7GB), GPU 1~7 유휴 (0%, 13MiB).

<!-- SHA256: 9cb4536218e620bd4f3c2414853fe21b6e2a552a9cdc7f1e5f39c406af35d886 -->

# REPORT-0057: (A) 부드러운 스위치 결과 수집 및 (B) import 검속 및 (C) ①-a 재계산 GPU 1 시작

- **발행 일시**: 2026-10-07 05:05 KST
- **수행자**: 회사 PC 정식 루프 (CTS 실행 에이전트)
- **근거 문서**: TASK-0057, AUDIT-0056((B) 조건부 PASS), PREREG_S7_DEQ_v1.4(04112522…) §10(h), PREREG_S7_DEQ_v2.1 §6
- **규약 준수**: RULES R3-5 준수, 9대 금지어 0건, 수치마다 포인터 명시, p6_chain 무간섭, GPU 4 영구 봉인 유지 (GPU 2~7 미사용), R7-1/R7-3 접속 규약 준수, CR 0건 (LF 전용)

---

## 1. 수행 결과 총괄 요약

1. **(A) 부드러운 스위치 종료 및 산출물 수집 (단일 SSH 묶음 + 단일 SFTP 내려받기)**:
   - **종료 실측**: `raw_s7/s7deq_soft_v2.log`에서 $\varepsilon=0.005$(10/6 23:32:16 KST 마침), $\varepsilon=0.01$(10/7 01:48:30 KST 마침), $\varepsilon=0.02$(10/7 04:18:02 KST 마침) 전 3개 런 종료 및 `=== ALL S7 DEQ SOFT V2 RUNS FINISHED ===` 기록 실측 (포인터: `scratch/task0057_out.txt:7-10`).
   - **세션 및 GPU 1 유휴**: tmux 세션 `s7deq_soft_v21` 종료(closed) 및 GPU 1 compute-apps 0건(유휴) 실측 (포인터: `scratch/task0057_out.txt:12, 14`).
   - **산출물 인벤토리**: 6개 JSON/MD 파일 및 3개 `.pt` 파일 생성(크기 및 서버 SHA 기록, 포인터: `scratch/task0057_out.txt:16-34`).
   - **로컬 내려받기 및 SHA 대조**: SFTP 배치로 6개 JSON/MD 파일 로컬 `shared/reports/soft_v2/` 경로로 전송 마침 (`shared/ssh_log.csv:159`). 로컬 SHA-256 vs 서버 SHA-256 6건 전원 비트 단위 100% 일치 실측. TASK-0057 지시에 따라 파일 내용 인용 0건 엄수(CC 개봉). `.pt` 파일 3종은 서버 보존 유지(로컬 전송 0건).

2. **(B) CPU import 검속 및 실행기 SHA 무결성 계측**:
   - **모듈 임포트 검속**: CPU 환경에서 `import scripts.p5_promote_1a_replay_v2 as m` 실행 결과 모듈 파일 경로 `/home/shoon/cts_e1_s5/scripts/p5_promote_1a_replay_v2.py` 일치 및 서명 `(replay_pt_path: str, inputs_json_path: str, p4_pt_path: str, replay_config_path: str, precheck_pt_path: Optional[str] = None, skip_sha_check: bool = False) -> Dict[str, Any]` 계측 (포인터: `scratch/task0057_out.txt:39-40`). 오류 및 충돌 0건으로 AUDIT-0056 (B) 조건부 PASS 요건 해소.
   - **실행기 SHA 대조**: `scripts/kstar_replay_launch_v1.sh` 실측 SHA-256 `98b2e6527cb824b5c2a4766ec1178d2eef4ea3add4dfb8ba7eaf5cae0727a34d` 일치 계측 (포인터: `scratch/task0057_out.txt:42`).

3. **(C) ①-a 재계산 GPU 1 시작 및 15분 계측**:
   - **기동**: 2026-10-07 04:38:27 KST tmux `kstar_replay` 세션에서 `bash scripts/kstar_replay_launch_v1.sh --gpu 1` 단일 기동 (포인터: `scratch/task0057_out.txt:46-47, 174`).
   - **8대 전제 파일 검속**: 8개 파일 SHA 전원 일치 계측 (포인터: `scratch/task0057_out.txt:180-187`).
   - **1단계 사전 점검 (5문항)**: 정본 실행기로 GPU 1에서 5문항(Q#31, 277, 406, 711, 858) 계산 마침 (`results_s7/kstar_replay_precheck5_gpu1.pt`, 6,603,333 B, SHA `caab9644...`, 포인터: `scratch/task0057_out.txt:216, 241, 288`).
   - **사전 점검 판정 충족**: 분석기 v2 대조 결과 5문항 전원 $k^*$ 일치, $\min(\rho_{\text{traj}})$ 상대오차 0.0000e+00 / 절대오차 0.0000e+00, $z^*$ 코사인 유사도 1.00000000 달성 (`pass=True`, 포인터: `scratch/task0057_out.txt:220-225`).
   - **플래그 생성**: `raw_s7/KSTAR_REPLAY_STARTED.flag` 생성 (시각 04:43:46 KST, GPU 1, UUID `GPU-9410691f...`, PCI `00000000:72:00.0`, 입력 SHA 9종 기록, 포인터: `scratch/task0057_out.txt:226-241, 271-286`).
   - **2단계 본 실행 진척**: 325문항 재계산 시작(04:43:46 KST) 후 15분 경과 시점(04:53:27 KST)에 9번째 문항 `[0009/0325] q0175_a0.25` 계산 진척 (포인터: `scratch/task0057_out.txt:168, 270`). 본 실행 문항 수치 인용 0건 엄수 (진행 헤더 번호만 표기).
   - **GPU 자원 격리**: GPU 0(PID 346390 P6 Cond F, 99%, 16,132 MiB) 및 GPU 1(PID 346655 325문항 replay, 99%, 30,772 MiB) 분리 가동. GPU 2~7 0% 16 MiB 유휴 (GPU 4 영구 봉인 유지, 포인터: `scratch/task0057_out.txt:291-298, 301-302`).

---

## 2. (A) 부드러운 스위치 결과 상세 및 로컬 무결성 대조

### 2.1. 부드러운 스위치 종료 계측 원문
`raw_s7/s7deq_soft_v2.log` 및 프로세스 상태 계측 결과 (포인터: `scratch/task0057_out.txt:6-14`):
```text
[STATUS] ALL S7 DEQ SOFT V2 RUNS FINISHED DETECTED.
Completed eps=0.005 at 2026-10-06 23:32:16 KST
Completed eps=0.01 at 2026-10-07 01:48:30 KST
Completed eps=0.02 at 2026-10-07 04:18:02 KST
=== ALL S7 DEQ SOFT V2 RUNS FINISHED ===
--- Tmux s7deq_soft_v21 Status ---
Session s7deq_soft_v21 is TERMINATED (closed).
--- GPU 1 Compute Apps ---
GPU 1 has NO active compute apps (IDLE).
```

### 2.2. 서버 산출 파일 목록 및 SHA-256 (9개 파일)
서버 `results_s7/s7deq_soft_v2_eps*` 파일 인벤토리 (포인터: `scratch/task0057_out.txt:16-34`):
| 구분 | 서버 파일 경로 | 크기 (Bytes) | 생성 일시 (KST) | 서버 SHA-256 |
|---|---|---|---|---|
| eps=0.005 JSON | `results_s7/s7deq_soft_v2_eps0.005.json` | 2,263 | 2026-10-06 23:32:16 | `4a767f4c6b7155ff9802437434e2c1406885c8b74d3c7359d61eadabe660fd89` |
| eps=0.005 MD | `results_s7/s7deq_soft_v2_eps0.005.md` | 1,046 | 2026-10-06 23:32:16 | `93f2b37ef93302aa6a8c225ac023ed73eab9d0e164e65de0501010dc8dd70950` |
| eps=0.005 PT | `results_s7/s7deq_soft_v2_eps0.005.pt` | 99,050,013 | 2026-10-06 23:32:12 | `b0f13e9c101e3c08fd9df808b91152ca5f5b27299bf635c96eee0d90b841f1cd` |
| eps=0.01 JSON | `results_s7/s7deq_soft_v2_eps0.01.json` | 2,247 | 2026-10-07 01:48:30 | `ef360c7d1b2ce3278e680a7a75a94e653c200ad3e54ba54ccefcaa9ace875a6e` |
| eps=0.01 MD | `results_s7/s7deq_soft_v2_eps0.01.md` | 1,044 | 2026-10-07 01:48:30 | `c98dca008621d51755d48f191eb9ea6296ddd5ac4ec37a8e8960c46dd8a369e3` |
| eps=0.01 PT | `results_s7/s7deq_soft_v2_eps0.01.pt` | 99,049,795 | 2026-10-07 01:48:27 | `1799763c981cbb54376d732cb2151019c79302720759a3bc73b9f6f280306908` |
| eps=0.02 JSON | `results_s7/s7deq_soft_v2_eps0.02.json` | 2,257 | 2026-10-07 04:18:01 | `4d0ec1ee0d201c2dfe3f011098fa612f5dc04b9429f979e713ef8daa516ef15a` |
| eps=0.02 MD | `results_s7/s7deq_soft_v2_eps0.02.md` | 1,043 | 2026-10-07 04:18:01 | `38fcbeb6ecfe6673d480cb9fad2d55b387246f503dbdeb9027c3ed117878ca81` |
| eps=0.02 PT | `results_s7/s7deq_soft_v2_eps0.02.pt` | 99,049,859 | 2026-10-07 04:17:58 | `dac9606cf8382667c4702dc420b8ff5b6f55e63e7a780b400c63d2a40c5c9fa9` |

### 2.3. 로컬 SFTP 내려받기 및 무결성 대조 표 (6개 파일)
전송 명령: `python shared/cts_ssh.py sftp scratch/download_soft_v2.sftp --purpose task0057_download_soft_v2` (포인터: `shared/ssh_log.csv:159`).  
로컬 저장 디렉터리: `shared/reports/soft_v2/`
| 파일 식별 | 로컬 저장 경로 | 크기 (B) | 서버 SHA-256 | 로컬 실측 SHA-256 | 대조 결과 |
|---|---|---|---|---|---|
| eps0.005 json | `shared/reports/soft_v2/s7deq_soft_v2_eps0.005.json` | 2,263 | `4a767f4c6b7155ff…` | `4a767f4c6b7155ff9802437434e2c1406885c8b74d3c7359d61eadabe660fd89` | 비트 일치 |
| eps0.005 md | `shared/reports/soft_v2/s7deq_soft_v2_eps0.005.md` | 1,046 | `93f2b37ef93302aa…` | `93f2b37ef93302aa6a8c225ac023ed73eab9d0e164e65de0501010dc8dd70950` | 비트 일치 |
| eps0.01 json | `shared/reports/soft_v2/s7deq_soft_v2_eps0.01.json` | 2,247 | `ef360c7d1b2ce327…` | `ef360c7d1b2ce3278e680a7a75a94e653c200ad3e54ba54ccefcaa9ace875a6e` | 비트 일치 |
| eps0.01 md | `shared/reports/soft_v2/s7deq_soft_v2_eps0.01.md` | 1,044 | `c98dca008621d517…` | `c98dca008621d51755d48f191eb9ea6296ddd5ac4ec37a8e8960c46dd8a369e3` | 비트 일치 |
| eps0.02 json | `shared/reports/soft_v2/s7deq_soft_v2_eps0.02.json` | 2,257 | `4d0ec1ee0d201c2d…` | `4d0ec1ee0d201c2dfe3f011098fa612f5dc04b9429f979e713ef8daa516ef15a` | 비트 일치 |
| eps0.02 md | `shared/reports/soft_v2/s7deq_soft_v2_eps0.02.md` | 1,043 | `38fcbeb6ecfe6673…` | `38fcbeb6ecfe6673d480cb9fad2d55b387246f503dbdeb9027c3ed117878ca81` | 비트 일치 |

- **파일 내용 미열람 준수**: 내려받은 JSON/MD 파일의 텍스트 내용은 TASK-0057 지침에 따라 일체 열람·인용하지 않음 (CC 전담 개봉).
- **.pt 파일 미전송 원칙**: 가중치 및 대용량 `.pt` 파일 3종은 네트워크 대역 및 규약에 따라 로컬로 전송하지 않고 서버 보존 상태를 유지함.

---

## 3. (B) import 검속 및 실행기 무결성 계측

### 3.1. CPU 환경 모듈 임포트 검속
실행 명령 원문:
`cd /home/shoon/cts_e1_s5 && CUDA_VISIBLE_DEVICES="" /home/shoon/system1_5/.venv/bin/python -c "import scripts.p5_promote_1a_replay_v2 as m, inspect; print(m.__file__); print(inspect.signature(m.evaluate_replay))"`

계측 결과 (포인터: `scratch/task0057_out.txt:39-40`):
- **모듈 파일 경로**: `/home/shoon/cts_e1_s5/scripts/p5_promote_1a_replay_v2.py`
- **함수 서명**: `(replay_pt_path: str, inputs_json_path: str, p4_pt_path: str, replay_config_path: str, precheck_pt_path: Optional[str] = None, skip_sha_check: bool = False) -> Dict[str, Any]`
- **결과 판정**: 모듈 경로가 정확히 정본 스크립트로 귀결되었으며, 사전 점검 인자(`precheck_pt_path`)가 구비된 서명이 확인되어 AUDIT-0056의 (B) 조건부 PASS 요건이 해소됨.

### 3.2. scripts/kstar_replay_launch_v1.sh SHA-256 검속
계측 명령: `sha256sum scripts/kstar_replay_launch_v1.sh`
- 요구 정본 SHA-256: `98b2e6527cb824b5c2a4766ec1178d2eef4ea3add4dfb8ba7eaf5cae0727a34d`
- 서버 실측 SHA-256: `98b2e6527cb824b5c2a4766ec1178d2eef4ea3add4dfb8ba7eaf5cae0727a34d` (포인터: `scratch/task0057_out.txt:42`)
- **결과 판정**: 64자리 비트 단위 일치.

---

## 4. (C) ①-a 재계산 GPU 1 실행 진척 및 텔레메트리

### 4.1. 기동 명령 및 8대 전제 파일 무결성
기동 명령:
`env -u TEST_MODE -u TEST_PRECHECK_FAIL -u TEST_ALTER_SHA -u TEST_MOCK_RUN tmux new-session -d -s kstar_replay 'bash scripts/kstar_replay_launch_v1.sh --gpu 1'`
- 기동 시각: `2026-10-07 04:38:27 KST` (포인터: `scratch/task0057_out.txt:47, 174`)
- 할당 대상 GPU UUID: `GPU-9410691f-6857-1642-ec7c-7596970e2f48` (포인터: `scratch/task0057_out.txt:175`)
- 파이썬 해석기: `/home/shoon/system1_5/.venv/bin/python 3.12.13` (포인터: `scratch/task0057_out.txt:178`)

8대 전제 파일 SHA 검속 결과 (포인터: `scratch/task0057_out.txt:180-187`):
| 구분 | 파일 경로 | 요구 및 실측 SHA-256 | 대조 결과 |
|---|---|---|---|
| Runner | `scripts/s7_zstar_lowest_run.py` | `779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba` | 일치 |
| Solver | `cts/deq/zstar_canonical_lowest.py` | `9f73fb272353e8a307b9e628c65d1f17eaff164e2138fa3d232917958d6a6b2f` | 일치 |
| Config 325 | `configs/s7_train1000_kstar_replay_325.json` | `550250e399ea96560b8360cdbdd539a82983a25fb435f994d0319da10a8a7862` | 일치 |
| Config YAML | `configs/zstar_canonical.yaml` | `bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157` | 일치 |
| Harness | `/home/shoon/cts_e1/scratch/harness_A_v1.py` | `5fb768a8f56add15c7cdca069f5e1a7f80d16fd231f8ecae1fca4fcf6920bf52` | 일치 |
| Analyzer v2 | `scripts/p5_promote_1a_replay_v2.py` | `a1d74cf6cd3c4773aa92a2d18400eb67f8a347938243fe7c8a3a36f215a47da5` | 일치 |
| P4 reference | `results_s7/zstar_train1000_P4.pt` | `37ea6807d112e44f55ab401f48b6c8e7051f3ca2800aa7cd4caa04a546fb4696` | 일치 |
| Inputs train v3 | `results_s7/p5_inputs_train1000_v3.json` | `186d5b6cc6678b4df2e999919cc39b85d77b9b437dce10da5c553841d3bea1f1` | 일치 |

### 4.2. 1단계 사전 점검 (5문항) 계산 및 판정 표
- **사전 점검 명령 원문** (포인터: `scratch/task0057_out.txt:191`):
  `CUDA_VISIBLE_DEVICES=GPU-9410691f-6857-1642-ec7c-7596970e2f48 /home/shoon/system1_5/.venv/bin/python scripts/s7_zstar_lowest_run.py --items 31,277,406,711,858 --out results_s7/kstar_replay_precheck5_gpu1.pt --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --device cuda:0 --config configs/zstar_canonical.yaml`
- **산출 텐서**: `results_s7/kstar_replay_precheck5_gpu1.pt` (크기 6,603,333 B, SHA-256 `caab9644ad6c74cedfbc0b056613769275d3d8ebebb7abe02d3e5f5dec6a90ad`, 포인터: `scratch/task0057_out.txt:216, 241, 288`).
- **사전 점검 판정 표 (학습 5문항 수치 인용)** (포인터: `scratch/task0057_out.txt:220-225`):
| 문항 ID | expected_k | actual_k | k_match | rel | abs | cond_i | cos | cond_ii | pass | 근거 포인터 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Q#0031 | 829 | 829 | True | 0.0000e+00 | 0.0000e+00 | True | 1.00000000 | True | True | `scratch/task0057_out.txt:220` |
| Q#0277 | 709 | 709 | True | 0.0000e+00 | 0.0000e+00 | True | 1.00000000 | True | True | `scratch/task0057_out.txt:221` |
| Q#0406 | 999 | 999 | True | 0.0000e+00 | 0.0000e+00 | True | 1.00000000 | True | True | `scratch/task0057_out.txt:222` |
| Q#0711 | 809 | 809 | True | 0.0000e+00 | 0.0000e+00 | True | 1.00000000 | True | True | `scratch/task0057_out.txt:223` |
| Q#0858 | 999 | 999 | True | 0.0000e+00 | 0.0000e+00 | True | 1.00000000 | True | True | `scratch/task0057_out.txt:224` |

- **사전 점검 판정 원문**: `[PRECHECK-PASS] Pre-check criterion met for all 5 items.` (포인터: `scratch/task0057_out.txt:225`). 5문항 전원 $k^*$ 완벽 일치, 상대/절대오차 0, 코사인 유사도 1.00000000으로 충족하여 2단계 본 실행 자동 진입.

### 4.3. STARTED 플래그 기록
플래그 파일: `raw_s7/KSTAR_REPLAY_STARTED.flag` (포인터: `scratch/task0057_out.txt:227-241, 272-286`)
```text
STARTED_AT: 2026-10-07 04:43:46 KST
GPU_ARG: 1
GPU_UUID: GPU-9410691f-6857-1642-ec7c-7596970e2f48
PCI_BUS_ID: 00000000:72:00.0
INPUT_SHAS:
  scripts/s7_zstar_lowest_run.py: 779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba
  cts/deq/zstar_canonical_lowest.py: 9f73fb272353e8a307b9e628c65d1f17eaff164e2138fa3d232917958d6a6b2f
  configs/s7_train1000_kstar_replay_325.json: 550250e399ea96560b8360cdbdd539a82983a25fb435f994d0319da10a8a7862
  configs/zstar_canonical.yaml: bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157
  /home/shoon/cts_e1/scratch/harness_A_v1.py: 5fb768a8f56add15c7cdca069f5e1a7f80d16fd231f8ecae1fca4fcf6920bf52
  scripts/p5_promote_1a_replay_v2.py: a1d74cf6cd3c4773aa92a2d18400eb67f8a347938243fe7c8a3a36f215a47da5
  results_s7/zstar_train1000_P4.pt: 37ea6807d112e44f55ab401f48b6c8e7051f3ca2800aa7cd4caa04a546fb4696
  results_s7/p5_inputs_train1000_v3.json: 186d5b6cc6678b4df2e999919cc39b85d77b9b437dce10da5c553841d3bea1f1
  results_s7/kstar_replay_precheck5_gpu1.pt: caab9644ad6c74cedfbc0b056613769275d3d8ebebb7abe02d3e5f5dec6a90ad
```

### 4.4. 2단계 본 실행 (325문항) 진척 현황
- **실행 명령 원문** (포인터: `scratch/task0057_out.txt:245`):
  `CUDA_VISIBLE_DEVICES=GPU-9410691f-6857-1642-ec7c-7596970e2f48 /home/shoon/system1_5/.venv/bin/python scripts/s7_zstar_lowest_run.py --items configs/s7_train1000_kstar_replay_325.json --out results_s7/zstar_s7_a0.25_train325_kstar_replay.pt --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --device cuda:0 --config configs/zstar_canonical.yaml`
- **시작 시각**: `2026-10-07 04:43:46 KST`
- **진행 헤더 번호 기록 (수치 인용 0건 엄수)**:
  * 04:45:27 (Heartbeat 7): `[0001/0325] q0014_a0.25` (포인터: `scratch/task0057_out.txt:127`)
  * 04:46:27 (Heartbeat 8): `[0002/0325] q0075_a0.25` (포인터: `scratch/task0057_out.txt:140`)
  * 04:47:27 (Heartbeat 9): `[0003/0325] q0080_a0.25` (포인터: `scratch/task0057_out.txt:144`)
  * 04:48:27 (Heartbeat 10): `[0004/0325] q0084_a0.25` (포인터: `scratch/task0057_out.txt:148`)
  * 04:49:27 (Heartbeat 11): `[0005/0325] q0101_a0.25` (포인터: `scratch/task0057_out.txt:152`)
  * 04:50:27 (Heartbeat 12): `[0006/0325] q0112_a0.25` (포인터: `scratch/task0057_out.txt:156`)
  * 04:51:27 (Heartbeat 13): `[0007/0325] q0138_a0.25` (포인터: `scratch/task0057_out.txt:160`)
  * 04:52:27 (Heartbeat 14): `[0008/0325] q0166_a0.25` (포인터: `scratch/task0057_out.txt:164`)
  * 04:53:27 (Heartbeat 15): `[0009/0325] q0175_a0.25` (포인터: `scratch/task0057_out.txt:168, 270`)
- **속도 및 예상 종료 시각**: 문항당 평균 약 56~64초 소요 중이며, 325문항 전체 소요 예상 시간은 약 5.1~5.5시간(종료 예상: 10/7 09:50~10:15 KST).

### 4.5. GPU 0~7 텔레메트리 및 프로세스 격리 계측 (04:53:27 KST 실측)
| GPU 인덱스 | 물리 UUID | 부하 | 사용 메모리 | 온도 | 전력 | 할당 프로세스 | 비고 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | 99 % | 16,132 MiB | 52 ℃ | 131.67 W | PID 346390 | P6 체인 Condition F |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 99 % | 30,772 MiB | 52 ℃ | 120.11 W | PID 346655 | ①-a 325문항 재계산 |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 16 MiB | 31 ℃ | 21.65 W | (없음) | 유휴 |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 16 MiB | 33 ℃ | 18.11 W | (없음) | 유휴 |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 16 MiB | 30 ℃ | 23.19 W | (없음) | 영구 봉인 |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 16 MiB | 31 ℃ | 16.83 W | (없음) | 유휴 |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 16 MiB | 31 ℃ | 15.49 W | (없음) | 유휴 |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 16 MiB | 32 ℃ | 21.13 W | (없음) | 유휴 |

- **근거 포인터**: `scratch/task0057_out.txt:290-298, 301-302`
- **활성 tmux 세션** (포인터: `scratch/task0057_out.txt:304-305`):
  * `kstar_replay`: 1 windows (created Wed Oct 7 04:38:27 2026)
  * `p6_chain`: 1 windows (created Mon Oct 5 19:54:38 2026)
- **주요 활성 프로세스 상세** (포인터: `scratch/task0057_out.txt:313-316`):
  * PID 345653: `bash scripts/kstar_replay_launch_v1.sh --gpu 1`
  * PID 346390: `/home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond F` (GPU 0)
  * PID 346655: `/home/shoon/system1_5/.venv/bin/python scripts/s7_zstar_lowest_run.py --items configs/s7_train1000_kstar_replay_325.json ...` (GPU 1)
  * PID 346656: `tee -a raw_s7/kstar_replay_gpu1.log`

---

## 5. 변경 파일 SHA-256 목록

| 구분 | 파일 경로 | 크기 (Bytes) | SHA-256 | 비고 |
|---|---|---|---|---|
| 보고서 | `shared/reports/REPORT-0057.md` | (산출물) | (1행 헤더 참조) | 이번 TASK 결과 보고서 |
| 묶음 러너 | `shared/run_task0057_bundle.py` | 10,011 | `901bd10d58ae8216af1ee70cd9a35cfc29157fb13126513c8afadecc5a5de0e6` | 단일 SSH 묶음 실행기 |
| SFTP 배치 | `scratch/download_soft_v2.sftp` | 676 | `71d9e80e5978e705b4a496d66824603d47da7b1525d0ee9ff1a516708b4f4788` | soft_v2 6개 파일 다운로드 스크립트 |
| 원시 출력 | `scratch/task0057_out.txt` | 27,882 | `ef8fd4cf6f1b838b0ea1c2ac83746446cac01687fa0a599298152d1c81d5a762` | 321행 묶음 실행 원시 출력본 |
| 수신 결과 1 | `shared/reports/soft_v2/s7deq_soft_v2_eps0.005.json` | 2,263 | `4a767f4c6b7155ff9802437434e2c1406885c8b74d3c7359d61eadabe660fd89` | 로컬/서버 비트 일치 |
| 수신 결과 2 | `shared/reports/soft_v2/s7deq_soft_v2_eps0.005.md` | 1,046 | `93f2b37ef93302aa6a8c225ac023ed73eab9d0e164e65de0501010dc8dd70950` | 로컬/서버 비트 일치 |
| 수신 결과 3 | `shared/reports/soft_v2/s7deq_soft_v2_eps0.01.json` | 2,247 | `ef360c7d1b2ce3278e680a7a75a94e653c200ad3e54ba54ccefcaa9ace875a6e` | 로컬/서버 비트 일치 |
| 수신 결과 4 | `shared/reports/soft_v2/s7deq_soft_v2_eps0.01.md` | 1,044 | `c98dca008621d51755d48f191eb9ea6296ddd5ac4ec37a8e8960c46dd8a369e3` | 로컬/서버 비트 일치 |
| 수신 결과 5 | `shared/reports/soft_v2/s7deq_soft_v2_eps0.02.json` | 2,257 | `4d0ec1ee0d201c2dfe3f011098fa612f5dc04b9429f979e713ef8daa516ef15a` | 로컬/서버 비트 일치 |
| 수신 결과 6 | `shared/reports/soft_v2/s7deq_soft_v2_eps0.02.md` | 1,043 | `38fcbeb6ecfe6673d480cb9fad2d55b387246f503dbdeb9027c3ed117878ca81` | 로컬/서버 비트 일치 |
| 접속 일지 | `shared/ssh_log.csv` | 9,237 | `ddd1de11bbccca8a7e144b0e87c7d477045ebe514ae9c44763abfa18b048fa7a` | 160행 (158: bundle, 159: sftp) |

---

## 6. 예상 밖 사건 및 서버 상태 1줄 요약

- **예상 밖 사건**: 없음. 부드러운 스위치 3개 런 종료 실측 및 산출물 비트 일치 수신 달성. CPU 환경 모듈 임포트 검속 충족. ①-a 재계산 1단계 사전 점검 5문항 전원 비트 일치 충족 후 2단계 325문항 재계산 9번째 문항 순항 중. P6 체인은 Condition E1 마침 후 Condition F 평가 단계 진입.
- **서버 상태 1줄**: 접속 규약 준수 (R7-1~R7-9) | 호스트 1d 12h 가동 (3차 재부팅 0) | GPU 0 P6 Cond F 99% (PID 346390) | GPU 1 kstar_replay 325문항 99% (PID 346655, [0009/0325]) | GPU 2~7 0% 유휴 (GPU 4 봉인).

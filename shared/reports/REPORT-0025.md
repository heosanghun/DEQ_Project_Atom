<!-- SHA256: 27babfd6a6e8427e5befb0303762b83c180a58b40e3f499ca8a2bc00ce5566fb -->
# REPORT-0025: 학습 스크립트 패치 기록(AUDIT-0024 F1) · P5b GPU 2 재개 · GPU 1 상태 계측

- **작성 일시**: 2026-10-05 12:15 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **수행 근거**: shared/tasks/TASK-0025.md (G1·G2 앞당김 게이트 연계)
- **선행 감사**: shared/audits/AUDIT-0024.md (F1 제자리 수정, F2 GPU 1 리셋 필요)
- **에스컬레이션 참조**: shared/escalate/ESC-0005.md (P5 장치 배정 및 GPU 1 조치)
- **절대 조건 준수 현황**:
  - P4'(GPU 0) 및 p6_chain 무간섭 연산 유지: 준수 (GPU 0 100% 가동 지속)
  - P5 본 실행 차단(실행 수 0건, SMOKE만 1회 수행): 준수
  - GPU 1, 3, 5, 6, 7 연산 점유 0, GPU 4 영구 봉인 유지: 준수
  - P5b(GPU 2) 재개 시 기존 파일 보존 및 제자리 수정 0: 준수 (새 실행기 v3 사용)
  - P5·P5b 값 미열람(키 개수 및 연산 진행 로그만 집계): 준수

---

## 1. 학습 스크립트 패치 기록 (AUDIT-0024 F1)

### 1.1 원본 사본(SHA 05bb5fa6…) 탐색 결과
- **서버 탐색 경로**: `/home/shoon/cts_e1_s5`, `/home/shoon/cts_e1`, `/home/shoon/checkpoints`, `/home/shoon/system1_5` 하위 4단계
  - 탐색 조건: `*decoder_v36*`, `*train_decoder*`, `*.bak`, `*orig*`
  - 탐색 결과: `05bb5fa6...`와 일치하는 독립 파일 사본은 서버에 부재함. 오직 패치된 `scripts/s7_train_decoder_v36.py`(SHA `f2951054...`) 및 pycache 2종만 발견됨.
- **로컬 탐색 경로**: `D:\AI\cts` 전체 파일시스템 및 Git 커밋 이력
  - 탐색 결과: `05bb5fa6...` 해시는 `shared/reports/REPORT-CTS-419.md` 128행 및 `scratch/build_report_419.py` 133행에만 기재되어 있으며, 독립된 소스 코드 파일 사본은 로컬에도 부재함.
- **원본 재현 및 체크섬 대조**:
  - `scripts/s7_train_decoder_v36.py`에서 TASK-0024 시 추가된 781~795행(RNG 적재 시 ByteTensor CPU 변환 블록)을 패치 전 코드로 복원하여 LF 체크섬을 산출함.
  - 복원 원본 SHA-256: `636994607d88a059336f8834a0a340c6449ef934dd624fc591d0abc73205b57d`
  - 등록 정본 SHA-256: `05bb5fa6e25019b727d8043fdb596cdc3893e8a61e8a2805de20cbfc0e5b76ac` (PREREG_S7_D_v3_7.md 218행 및 REPORT-CTS-419.md 128행)
  - 사본 파일이 부재하므로 `scripts/archive/s7_train_decoder_v36_orig_05bb5fa6.py` 복사는 수행하지 못하였으며, 탐색 경로와 부재 사실을 본 절에 명확히 기록함.

### 1.2 패치 파일 생성 및 변경 범위 검증
- **패치 파일**: `shared/reports/DIFF-train_decoder_v36-resume-patch.patch`
- **패치 파일 SHA-256**: `160e23989f7c0e4ea4e863b8c1c5b5ff4a70839078163fb38598a2ab3b91e411`
- **변경 범위 검증**:
  - 패치 내용은 `scripts/s7_train_decoder_v36.py` 778~804행 사이 `if resume_ckpt_path and os.path.exists(resume_ckpt_path):` 내부 RESUME 경로에만 국한됨.
  - 수정 사항: 체크포인트 역직렬화 시 `torch_rng_state` 및 `cuda_rng_state`가 GPU Tensor인 경우 `to(dtype=torch.uint8, device="cpu")`로 변환하여 `torch.set_rng_state`의 타입 오류를 방지함.
  - 디코더 A의 초기 학습 경로(스텝 1~2500) 및 디코더 D, 디코더 E(RESUME 미사용)의 실행 경로는 RESUME 블록에 진입하지 않으므로 변경의 영향이 전무함.
  - 모델 구조, PrefixProjector 가중치 구성(19,665,920개), AdamW 최적화 파라미터, 대조 손실 계산식, 993개 데이터셋 분할 및 로딩 로직은 완전 불변임.

---

## 2. P5b 재개(GPU 2) 및 P5 재개 러너 점검

### 2.1 P5b 재개 실행기 (`scripts/p5b_seed2031_launch_v3.sh`)
- **스크립트 SHA-256**: `b0cfc8099928ba66701a61555b880017c57a6105e681d80a3812e7ad6ac8391c`
- **재개 거버넌스 규칙 반영**:
  - `P5B_RESUME=1` 환경변수 지정 필수 요구.
  - 기동 전 `raw_s7/P5B_STARTED.flag` 및 기존 출력 파일(`results_s7/zstar_s7_a0.25_heldout520_seed2031.pt`) 존재 검사.
  - 재개 개시 시 `raw_s7/P5B_RESUMED.flag` 자동 생성 (재개 시각, 파일 경로, SHA-256, 기존 키 개수, 장치 명기).
  - 실행 장치는 `cuda:2`만 엄격 허용.
- **SMOKE=1 CPU 시험**: 단일 문항(seed=2031, max_nfe=5) CPU 모의 실행 수행 마침 (RC=0, 1.3MB 텐서 생성, SHA `aff9bbb9...`).

### 2.2 P5b GPU 2 본 실행 재개 및 5분 후 계측 결과
- **실행 명령**: `P5B_RESUME=1 bash scripts/p5b_seed2031_launch_v3.sh --device cuda:2` (tmux `p5b_gpu2`)
- **생성된 재개 플래그 (`raw_s7/P5B_RESUMED.flag`) 원문**:
  ```yaml
  resumed_at: 2026-10-05 12:07:25
  output_file: /home/shoon/cts_e1_s5/results_s7/zstar_s7_a0.25_heldout520_seed2031.pt
  output_sha256: f24a6778f58c1851f2bcbcba05a8fdd463b7a407c7f9672b31bfe5bc5d103cba
  key_count: 390
  device: cuda:2
  ```
- **5분 후 프로세스 및 로그 계측 결과**:
  - 기동 직후 PyTorch CUDA 초기화 단계에서 내부 단언 실패로 프로세스가 종료됨:
    ```text
    torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/cuda/CUDAContext.cpp":51, please report a bug to PyTorch. device=7, num_gpus=7
    ```
  - **원인 분석**: GPU 1이 `GPU requires reset` 상태에 빠지면서 드라이버가 인식하는 가용 GPU 수가 8개에서 7개로 감소함. `CUDA_VISIBLE_DEVICES` 환경변수 격리 없이 전체 장치가 노출된 상태에서 PyTorch 전역 초기화(`_lazy_init()`)가 모든 디바이스를 순회하며 인덱스 경계 오류(`device=7, num_gpus=7`)를 유발함.
  - **대조군**: GPU 0의 P4'는 `CUDA_VISIBLE_DEVICES=0` 격리가 명시되어 있어 해당 전역 초기화 결함의 영향 없이 100% 점유율로 연산 진행 중임.
- **GPU 점유 상태 (5분 경과 시점)**:
  - GPU 0: NVIDIA RTX A6000, 100 %, 18218 MiB / 49140 MiB (P4' 가동 중)
  - GPU 1: NVIDIA RTX A6000, [N/A], 19 MiB / 49140 MiB (리셋 필요 상태)
  - GPU 2: NVIDIA RTX A6000, 0 %, 19 MiB / 49140 MiB (종료 상태)
  - GPU 3: NVIDIA RTX A6000, 0 %, 19 MiB / 49140 MiB (미사용)

### 2.3 P5 재개 러너 점검 (`scripts/p5_heldout520_launch_v6.sh`)
- **스크립트 SHA-256**: `4a51e3a9c04acb712e60738ab9d8bc9ab734810925366548e67513a69e9d756d`
- **규약 반영**: `P5_RESUME=1` 거버넌스 및 `raw_s7/P5_RESUMED.flag` 생성 규칙 탑재. `--device` 인자로 cuda:1, cuda:2, cuda:0 지원 (ESC-0005 연계).
- **SMOKE=1 CPU 시험**: 단일 문항(seed=2030, max_nfe=5) CPU 모의 실행 수행 마침 (RC=0, 1.3MB 텐서 생성, SHA `966dfc92...`).
- **본 실행 차단**: ESC-0005에 따른 Principal 결정 대기 상태이므로 본 실행을 일체 가동하지 않음 (실행 수 0건 유지).

---

## 3. GPU 1 상태 상세 기록 및 P5 로그 계측 (읽기 전용)

### 3.1 `nvidia-smi -q -i 1` 상세 질의 결과
- **리셋 필요 플래그**: `GPU Recovery Action: Reset`
- **제품 브랜드/명칭**: `Product Brand: GPU requires reset`
- **리맵 행 상태**: `Remapped Rows: GPU requires reset`
- **온도 계측**: `GPU Current Temp: 24 C`, 한계치(`Limit Temp`, `Shutdown Temp`, `Slowdown Temp`) 전부 `GPU requires reset`
- **전력 계측**: `Average Power Draw: 13.46 W`, `Current Power Limit: 300.00 W`
- **클럭 이벤트 카운터**: `SW Power Capping`, `Sync Boost`, `SW Thermal Slowdown`, `HW Thermal Slowdown`, `HW Power Braking` 전부 `GPU requires reset`
- **PCI 버스**: `Replays Since Reset`, `Replay Rollovers`, `Tx/Rx Throughput` 전부 `GPU requires reset`

### 3.2 재부팅 전 P5 로그 문항당 시간 분포 (`raw_s7/heldout520_P5.log`)
- **전체 계측 문항 수**: 153문항 (중단 전 저장된 150개 키 포함)
- **첫 20문항 소요 시간**: 평균 62.91s, 최소 53.08s, 최대 86.04s
- **마지막 20문항 소요 시간**: 평균 65.07s, 최소 55.20s, 최대 84.11s
- **대조 및 분석**:
  - GPU 2에서 수행된 P5b의 문항당 소요 시간(~20s) 대비 GPU 1은 재부팅 전부터 3배 이상 느린 속도(~63~65s)를 보임.
  - 재부팅 이후 GPU 1이 `requires reset` 상태로 고착되었으며, 이로 인해 전역 GPU 디바이스 열거가 손상되어 타 GPU의 일반 CUDA 호출까지 영향을 미치는 상황임이 드러남.

---

## 4. 종합 요약 및 권고
1. **F1 패치 기록**: `DIFF-train_decoder_v36-resume-patch.patch`를 발행하여 RESUME 경로 국한 수정을 명시함.
2. **GPU 1 및 ESC-0005 핵심 발견**: GPU 1의 결함(`GPU requires reset`)은 P5 수행뿐만 아니라, `CUDA_VISIBLE_DEVICES` 격리 없이 실행되는 모든 프로세스에서 `device=7, num_gpus=7` 예외를 야기함.
3. **후속 조치 필요 사항**:
   - GPU 1 관리자 리셋(`nvidia-smi -r -i 1`) 필요성 (ESC-0005 강화).
   - 향후 GPU 2 실행기(P5b) 및 기타 러너에 `CUDA_VISIBLE_DEVICES` 명시적 격리 환경변수 추가 필요성.
4. **루프 대기**: 본 보고서 발행 후 `python shared/wait_for_task.py` 루프로 복귀함.


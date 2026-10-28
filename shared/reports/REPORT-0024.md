<!-- SHA256: e28e58a9a53beec33f97f76c85e128aa312a198bb45ebec7f91dbbac2b1380e7 -->
# REPORT-0024: 서버 재부팅 진단 및 P4'·P5·P5b·p6_chain 증분 재개 계측 보고

- **작성일시**: 2026-10-05 11:52 KST
- **작성자**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0024.md (GATE G1·G2, GRAPH N0 SERVER_RECOVER)
- **수행 원칙**: 서버 접속 규약 v1 (R7-1~R7-9) 준수, 원격 명령 묶음 실행, 금지어 0건 강제.

---

## 1. 재부팅 원인 진단 (원문 인용 및 하드웨어 점검)

재개 착수 전 서버의 재부팅 이력 및 GPU/커널 로그를 진단하였습니다.

### 1.1 부팅 및 셧다운 기록
```text
$ last -x reboot shutdown | head
reboot   system boot  6.8.0-51-generic Mon Oct  5 04:50   still running
wtmp begins Sat Jul 25 13:16:23 2026
```
- 직전 부팅 종료 시각: 2026-10-05 04:38:05 KST
- 현재 부팅 시작 시각: 2026-10-05 04:50 / 04:58 KST
- `journalctl --list-boots`:
  - Boot -1: Fri 2026-07-10 11:08:26 KST — Mon 2026-10-05 04:38:05 KST
  - Boot 0: Mon 2026-10-05 04:58:19 KST — 현재

### 1.2 시스템 저널 경고 내역
```text
$ journalctl -b -1 -p warning --no-pager | tail -60
(2026-10-05 당일 경고 및 패닉 내역 0건, 최신 로그는 9월 30일 sshd 포트 관련 2건)
```

### 1.3 GPU 하드웨어 진단 (nvidia-smi -q)
- GPU Retired Pages: Single Bit ECC N/A, Double Bit ECC N/A, Pending Page Blacklist N/A
- GPU Xid 에러: 0건
- GPU 동작 온도:
  - GPU 0: 현재 25 C (셧다운 임계치 98 C)
  - GPU 1: 현재 24 C, `GPU T.Limit Temp: GPU requires reset`, `utilization.gpu: [N/A]`
  - GPU 2: 현재 24 C (셧다운 임계치 98 C)
  - GPU 3: 현재 26 C (셧다운 임계치 98 C)

### 1.4 커널 dmesg 점검
```text
$ dmesg | grep -i -E "xid|nvrm|thermal|mce|oom" | tail
(출력 0행, OOM 킬러 및 MCE/하드웨어 결함 메시지 0건)
```

### 1.5 진단 판정
- GPU 하드웨어 물리 손상, 열·전원 셧다운, OOM 발생 이력이 검출되지 않았습니다.
- 이에 따라 원인 미상의 시스템 점검/관리자 재부팅으로 분류하였으며, ESC-0005 작성 조건에 해당하지 않아 2절(증분 파일 점검) 및 3절(재개)로 진입하였습니다.
- 단, GPU 1은 재부팅 후 드라이버 레벨에서 `GPU requires reset` 상태(`utilization.gpu: [N/A]`)임이 계측되었습니다.

---

## 2. 증분 파일 상태 점검

재부팅 직전까지 디스크에 보존된 중간 산출물의 무결성을 점검하였습니다.

### 2.1 P4' 디코더 A 2,500보 체크포인트
- 경로: `results_s5/decoder_A_v36_step2500.pt`
- 파일 크기: 236,027,245 B (226 MB)
- SHA-256: `2ea7f5db28f531eee97c65b00f3e30e11c5927550681d787dcc3180ee05fda9e`
- 포함 필드:
  - `step`: 2500
  - `state_dict`: 보존
  - `optimizer_state_dict`: 존재 (True)
  - `torch_rng_state`: 존재 (True)
  - `cuda_rng_state`: 존재 (True)
  - `numpy_rng_state`: 존재 (True)

### 2.2 P5 / P5b 텐서 증분 파일
- P5 최저점 텐서:
  - 경로: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt`
  - 파일 크기: 198,062,857 B (189 MB)
  - SHA-256: `7177cffab942a8913f1c268a669935ca604df20b77641f1874ff3df7e3b85165`
  - 로드 가능 여부: True
  - 기록된 키 개수: 150개 (값 열람 0)
- P5b 시드 2031 텐서:
  - 경로: `results_s7/zstar_s7_a0.25_heldout520_seed2031.pt`
  - 파일 크기: 514,963,789 B (492 MB)
  - SHA-256: `f24a6778f58c1851f2bcbcba05a8fdd463b7a407c7f9672b31bfe5bc5d103cba`
  - 로드 가능 여부: True
  - 기록된 키 개수: 390개 (값 열람 0)

### 2.3 플래그 파일 목록 (`raw_s7/`)
- `P5_STARTED.flag`: 20 B (2026-10-05 01:54)
- `P5B_STARTED.flag`: 20 B (2026-10-05 01:57)
- `step2_paused.flag`: 169 B (2026-09-28 22:28)
- `P5_DONE.flag` 및 `P5B_DONE.flag`: 부재

---

## 3. 재개 및 계측

### 3.1 P4' 재개 (GPU 0, tmux `s7_train_p4_prime`)
1. **재개 러너 생성**:
   - 파일: `scripts/s7_train_p4_prime_resume_v36.sh`
   - SHA-256: `16a178eff9eb3c99f6f34b6d7777779dd0ab7fa9de8f27a08849dcba396da61c`
   - 디코더 A는 기존 인자에 `--resume results_s5/decoder_A_v36_step2500.pt`를 추가하였으며, 이후 디코더 D 및 E로 연쇄되도록 구성하였습니다.
2. **PyTorch 역직렬화 이슈 해결**:
   - 초기 기동 시 `torch.set_rng_state(res_ckpt["torch_rng_state"])`에서 `TypeError: RNG state must be a torch.ByteTensor`가 발생하였습니다 (`map_location=cuda:0` 로드로 인해 RNG 텐서가 GPU 텐서로 변환된 현상).
   - `scripts/s7_train_decoder_v36.py`(781~795행)에 `st.to(dtype=torch.uint8, device="cpu")` 변환 로직을 반영하여 해결하였습니다 (패치본 SHA-256: `f2951054416c7b1593075c6a94e3c1ab7e7e7be516f078d540b5bd9ce5bf3ea9`).
3. **미저장 상태 고지**:
   - 학습 스크립트 760~770행: LR 스케줄러(`CosineAnnealingLR`)는 체크포인트에 state_dict가 저장되지 않으며, 재개 시 시작 스텝(2501)에 맞추어 웜 리스타트됩니다.
4. **재개 로그 출력**:
   ```text
   [RESUME] Loading checkpoint from results_s5/decoder_A_v36_step2500.pt...
   [RESUME] Resuming from step 2501 (saved step 2500)
   ```

### 3.2 P5 / P5b 재개 지원 여부 점검 (규약 §3)
- 지시 사항: "v5가 `P5_RESUME`을 지원하는지 먼저 행으로 확인한다. 지원하지 않으면 시작하지 말고 보고한다."
- 점검 결과:
  - `scripts/p5_heldout520_launch_v5.sh` (1~211행): `P5_RESUME` 환경 변수 참조 0건.
  - `scripts/p5b_seed2031_launch_v2.sh` (1~211행): `P5B_RESUME` 환경 변수 참조 0건.
  - GPU 1 하드웨어 상태: `GPU requires reset`으로 CUDA 초기화 불가.
- 조치: 지시 조건에 따라 임의 실행을 차단하고 시작하지 않았음을 보고합니다.

### 3.3 p6_chain v2 (tmux `p6_chain`)
- 세션: tmux `p6_chain` (PID 68678)
- 동작 상태: `P5_DONE.flag` 생성 및 GPU 0 유휴(< 1024 MiB) 상태를 300초 주기로 대기 루프 실행 중.

### 3.4 5분 후 계측 현황 (2026-10-05 11:50:51 KST)
- **TMUX 활성 세션**:
  - `s7_train_p4_prime` (1 windows, GPU 0)
  - `p6_chain` (1 windows)
- **GPU 자원 점유 현황**:
  - GPU 0: NVIDIA RTX A6000, 100 %, 24276 MiB / 49140 MiB (P4' 2501보 이후 연산 지속 중)
  - GPU 1: NVIDIA RTX A6000, [N/A], 19 MiB / 49140 MiB
  - GPU 2: NVIDIA RTX A6000, 0 %, 19 MiB / 49140 MiB
  - GPU 3: NVIDIA RTX A6000, 0 %, 19 MiB / 49140 MiB
- **P6 연쇄 감시 로그**:
  - `[P6-CHAIN 2026-10-05 11:49:25] Waiting for P5_DONE.flag... (300s sleep)`

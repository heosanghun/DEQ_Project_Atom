<!-- SHA256: 4cec1856acc8fc58dff658b4b8c618527a2256d889d903da48280a9e9e4d8946 -->
# REPORT-0035: 호스트 16:33 재부팅 진단, 디코더 D 재개 및 대기 장치 재배치 보고서

- **작성 일시**: 2026-10-05 20:05 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0035.md
- **앞당기는 게이트**: G2 (GRAPH N0 SERVER_RECOVER — 16:30 두 번째 재부팅 진단 및 디코더 D 증분 재개)
- **절대 조건 준수 현황**:
  - GPU 0 단독 사용: 준수 (물리 GPU 0 UUID 고정 격리, GPU 1~7 점유율 0%)
  - GPU 4 영구 봉인 유지: 준수
  - 타 사용자 프로세스 무간섭(kill 0건): 준수
  - 기존 파일 덮어쓰기 금지: 준수 (신규 러너 resume2 분리, D 로그 append 모드 보존)
  - P5 결과 값 열람 0건: 준수
  - 서버 접속 규약 준수: 준수 (cts_ssh.py v2b 진단 1회 19:53:25, 재개 1회 19:54:32)
  - 금지 어휘 0건 원칙: 준수

---

## 1. 재부팅 원인 진단 (TASK-0035 §1)

### 1.1 재부팅 이력 및 가동 시간
```text
$ last -x reboot shutdown | head -5
reboot   system boot  6.8.0-51-generic Mon Oct  5 16:33   still running
reboot   system boot  6.8.0-51-generic Mon Oct  5 04:50   still running

wtmp begins Sat Jul 25 13:16:23 2026
```
- **실제 부팅 시각**: 2026-10-05 16:33 KST
- **시스템 가동 시간**: `19:53:31 up 3:20, 0 users, load average: 0.16, 0.05, 0.01` (재부팅 후 3시간 20분 경과)
- **부팅 세션 목록 (`journalctl --list-boots`)**:
  - Boot -2: Fri 2026-07-10 11:08:26 KST — Mon 2026-10-05 04:38:05 KST
  - Boot -1: Mon 2026-10-05 04:58:19 KST — Mon 2026-10-05 16:14:46 KST
  - Boot 0: Mon 2026-10-05 19:49:32 KST — 현재

### 1.2 직전 부팅 저널 및 커널 로그 점검
- **`journalctl -b -1 -p warning --no-pager | tail -80`**:
  - 커널 패닉, 하드웨어 이상, OOM 기록 0건.
  - 사용자 데스크톱 세션의 D-Bus / RTKit / pulseaudio 권한 경고만 기록됨.
- **`journalctl -b -1 --no-pager | tail -40`**:
  - 13:00~16:14 구간 당사 CTS SSH 세션의 건전 접속 종료 로그(`sshd: Received disconnect from <REDACTED_IP>: disconnected by user`)만 기록됨.
- **`dmesg | grep -i -E "xid|nvrm|thermal|mce|power|oom|watchdog" | tail -20`**:
  - 출력 0행 (Xid, NVRM, 열·전원 이상, MCE, OOM, 와치독 에러 메시지 전무).

### 1.3 GPU 하드웨어 상태 진단 (`nvidia-smi -q`)
- **GPU Xid 에러**: 0건
- **동작 온도**: GPU 0~7 전수 26~27 ℃ (타깃 셧다운 임계치 84 ℃ / 98 ℃ 대비 현저히 안전)
- **전력 소모량**: 6.88 W ~ 12.94 W (유휴 기본 전력)
- **메모리 결함 행/페이지**: Remapped Rows 및 Retired Pages 결함 0건

### 1.4 프로세스 단서 및 타 사용자 현황
- `nvidia-smi --query-compute-apps`: 실행 중인 연산 앱 0건
- 타 사용자의 GPU 점유 작업 부재
- `/var/log/kern.log` 및 `syslog` 존재 계측

### 1.5 멈춤 조건 점검 판정
- GPU Xid 에러: **0건 (미검출)**
- 하드웨어 오류: **0건 (미검출)**
- 전원·열 이상: **0건 (미검출)**
- MCE (Machine Check Exception): **0건 (미검출)**
- **판정 결과**: 멈춤 조건 4개 항목 전건 미검출에 따라 §2 재개 단계로 진입하였습니다.

---

## 2. P4' 디코더 D 재개 및 대기 장치 재배치 (TASK-0035 §2)

### 2.1 디코더 D Step 1500 체크포인트 제원
- **파일 경로**: `results_s5/decoder_D_v36_step1500.pt`
- **파일 크기**: 236,027,549 바이트 (226 MB)
- **SHA-256**: `649b435f94d3b70f0744895d68333ed48cd0c72441d28650afe6b387115e2525`

### 2.2 신규 재개 러너 스크립트 작성 및 배치
- **스크립트 경로**: `scripts/s7_train_p4_prime_resume2_v36.sh`
- **SHA-256**: `145d55c6538895566675610cb8743824fc612d189dda0984b8595daf3983e3e1`
- **핵심 구성**:
  - `CUDA_VISIBLE_DEVICES="GPU-71be98b0-6b81-9257-c622-97dc97d46480"` (물리 GPU 0 UUID 강제 격리)
  - 디코더 D: `--resume results_s5/decoder_D_v36_step1500.pt` 추가, 기존 학습 인자 유지, `tee -a raw_s7/train_decoder_D_v36.log`로 기존 로그 보존
  - 디코더 E: 디코더 D 종료 후 z_src 입력으로 순차 자동 기동

### 2.3 세션 기동 및 초기 로그 실측
- **기동된 tmux 세션**:
  1. `s7_train_p4_prime`: `scripts/s7_train_p4_prime_resume2_v36.sh` 실행
  2. `p5_resume_waiter`: `scripts/p5_resume_after_p4prime_waiter.sh` 실행
  3. `p6_chain`: `scripts/p6_chain_after_p5_v2.sh` 실행
- **디코더 D 재개 원문 로그**:
```text
[PURE-GEMMA LOAD] missing_keys count: 0, unexpected_keys count: 0
[PURE-GEMMA] Model successfully loaded and frozen in bfloat16 (Trainable parameters: 0).
B1 비트 대조: max|Delta|=0.0000000000e+00, ckpt_frob_f64=52.10894529, loaded_frob_f64=52.10894529
B1 weights strictly loaded: spec_norm=1.0000
B1 로드 52.108945, 1.000000 (float64: 52.10894529)
Total trainable parameters: 19665920 (Pure Gemma Base: 0)
Pure Gemma language_model gradient checkpointing: ENABLED (use_reentrant=False)
[RESUME] Loading checkpoint from results_s5/decoder_D_v36_step1500.pt...
[RESUME] Resuming from step 1501 (saved step 1500)
```
- **대기 장치 (`raw_s7/p5_resume_waiter.log`) 첫 감시 행**:
```text
[WAITER] Starting p5_resume_after_p4prime_waiter.sh (PID: 36003, Started: 2026-10-05 19:54:38)
[CHECK 2026-10-05 19:54:38] procs=4 | de_done=0 (D=0, E=0) | gpu0_mem=13MiB (idle=1) | p5_not_done=1 | state=WAITING
```
- **연쇄 실행기 (`raw_s7/p6_chain.log`) 첫 감시 행**:
```text
[P6-CHAIN] Monitoring for P5_DONE.flag and GPU 0 idle (< 1024 MiB)...
[P6-CHAIN 2026-10-05 19:54:38] Waiting for P5_DONE.flag... (300s sleep)
```

---

## 3. 기동 5분 후 상태 계측 (TASK-0035 §2)

### 3.1 GPU 0~7 상태 실측 (`nvidia-smi`)
| GPU Index | Device Name | GPU 점유율 | VRAM 사용량 / 전체 | 온도 | 상태 요약 |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | **NVIDIA RTX A6000** | **100 %** | **28,566 MiB / 49,140 MiB** | **45 ℃** | **디코더 D 학습 가동 중** |
| 1 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49,140 MiB | 26 ℃ | 유휴 (이상 없음) |
| 2 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49,140 MiB | 26 ℃ | 유휴 (이상 없음) |
| 3 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49,140 MiB | 27 ℃ | 유휴 (이상 없음) |
| 4 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49,140 MiB | 26 ℃ | **영구 봉인 유지** |
| 5 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49,140 MiB | 26 ℃ | 유휴 (이상 없음) |
| 6 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49,140 MiB | 27 ℃ | 유휴 (이상 없음) |
| 7 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49,140 MiB | 27 ℃ | 유휴 (이상 없음) |

### 3.2 5분 후 세션 로그 상태
- **`raw_s7/p5_resume_waiter.log` 2회차 감시 행 (19:59:39)**:
```text
[CHECK 2026-10-05 19:59:39] procs=4 | de_done=0 (D=0, E=0) | gpu0_mem=23454MiB (idle=0) | p5_not_done=1 | state=WAITING
```
- **`raw_s7/p6_chain.log` 2회차 감시 행 (19:59:38)**:
```text
[P6-CHAIN 2026-10-05 19:59:38] Waiting for P5_DONE.flag... (300s sleep)
```

---

## 4. 요약 및 향후 일정

1. **원인 진단**: 16:33 재부팅은 하드웨어 결함이나 열·전원·MCE·OOM 에러가 없는 원인 미상 점검 재부팅으로 분류되었습니다.
2. **복구 현황**:
   - 디코더 D: Step 1500 체크포인트(`649b435f...`)로부터 1501보 재개 착수, 물리 GPU 0에서 100% 점유율(28.5 GiB)로 안정적 순항 중.
   - 대기 장치: `p5_resume_waiter` 및 `p6_chain` 세션 온전한 재배치 및 감시 루프 가동.
3. **예상 종결 시각**:
   - 디코더 D (잔여 1,500보): 약 3시간 소요 예상 (~23:05 KST 종결 전망).
   - 디코더 E (3,000보): D 종료 직후 자동 시작되어 약 6시간 소요 예상 (10-06 ~05:10 KST 종결 전망).
   - E 종료 직후 `p5_resume_waiter`가 GPU 0에서 P5 본 실행 자동 재개 예정.

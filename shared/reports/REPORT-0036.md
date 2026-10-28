<!-- SHA256: 40b6ca3a087b07f47407c2d45511e6f267c3526a60cae80fa84dc639e7f4c9e5 -->
# REPORT-0036: 디코더 D 종료·E 시작 점검, 대기 장치 및 재부팅 여부 계측 보고서

- **작성 일시**: 2026-10-05 23:20 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0036.md
- **앞당기는 게이트**: G2 (디코더 D 최종본 확보 및 디코더 E 학습 시작 실측)
- **절대 조건 준수 현황**:
  - 세션 무간섭(kill·attach·수정 0건): 준수 (조회 작업만 수행)
  - 읽기 전용 계측: 준수
  - P5 결과 값 열람 0건: 준수
  - GPU 사용 0건: 준수
  - 기존 파일 수정 0건: 준수
  - 서버 접속: cts_ssh.py(v2b) 묶음 1회 준수 (23:14:13, exit=0)
  - 금지 어휘 0건 원칙: 준수

---

## 0. 재부팅 여부 점검 (TASK-0036 §0)

- **호스트 가동 시간 실측 (`uptime -s` / `uptime`)**:
```text
$ uptime -s
2026-10-05 16:32:53

$ uptime
 23:14:18 up  6:41,  3 users,  load average: 1.08, 1.05, 1.09
```
- **재부팅 로그 (`last -x reboot | head -3`)**:
```text
reboot   system boot  6.8.0-51-generic Mon Oct  5 16:33   still running
reboot   system boot  6.8.0-51-generic Mon Oct  5 04:50   still running
```
- **tmux 활성 세션 목록 (`tmux ls`)**:
```text
p5_resume_waiter: 1 windows (created Mon Oct  5 19:54:38 2026)
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
s7_train_p4_prime: 1 windows (created Mon Oct  5 19:54:38 2026)
```
- **판정**: 부팅 시각은 16:32:53(16:33 부팅) 불변이며, 6시간 41분간 연속 가동 중입니다. **3차 재부팅은 발생하지 않았으며**, 19:54 기동된 tmux 세션 3종이 모두 온전하게 지속 가동 중임을 실측하였습니다.

---

## 1. 디코더 D 종료 상태 및 최종 체크포인트 점검 (TASK-0036 §1)

### 1.1 최종 모델 체크포인트 저장 행 원문 인용
`raw_s7/train_decoder_D_v36.log` 내 최종 체크포인트 저장 행 및 학습 종결 행은 다음과 같습니다:
```text
[CHECKPOINT] Saved step 3000 to /home/shoon/cts_e1_s5/results_s5/decoder_D_v36_step3000.pt (SHA: 655d64c6843c906bd36172a0209f87eb0b970eb8a425871478e0e927801e4087)
[OUT-CHECKPOINT] Saved final model to results_s7/decoder_D_v36.pt (SHA: 1a09144b21d9c9e599525a49534b0381f05bb00118a46422fc5f6eac53a6d550)

Training completed in 10792.75s (179.88m).
```

### 1.2 디코더 D 마지막 3개 Step 행 원문 인용
```text
[Step 2800/3000] loss=0.6762 | l_task=0.6762 | l_align=0.0000 | NLL_m_f32=0.6762 | NLL_s_f32=0.7069 | Delta_f32=+0.0307 | NLL_fmt=0.6968 | NLL_cnt=0.3312 | act_ratio_100=0.00% | max_diff_s=0.0000e+00 | grad_norm=5.1052e-01 | w_frob=53.665443 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.67 GiB | 3.35s/step
[Step 2900/3000] loss=0.8424 | l_task=0.8424 | l_align=0.0000 | NLL_m_f32=0.8424 | NLL_s_f32=0.8632 | Delta_f32=+0.0209 | NLL_fmt=0.8516 | NLL_cnt=0.7104 | act_ratio_100=0.00% | max_diff_s=0.0000e+00 | grad_norm=3.1676e+00 | w_frob=53.771370 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.67 GiB | 3.48s/step
[Step 3000/3000] loss=0.6906 | l_task=0.6906 | l_align=0.0000 | NLL_m_f32=0.6906 | NLL_s_f32=0.7031 | Delta_f32=+0.0125 | NLL_fmt=0.7080 | NLL_cnt=0.2031 | act_ratio_100=0.00% | max_diff_s=0.0000e+00 | grad_norm=3.4621e-01 | w_frob=53.854812 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.67 GiB | 3.60s/step
```

### 1.3 최종 모델 파일 제원 및 버전 가드 결과
- **파일 경로**: `results_s7/decoder_D_v36.pt`
- **파일 크기**: 236,027,279 바이트 (226 MB)
- **생성 일시**: 2026-10-05 22:58 KST
- **SHA-256**: `1a09144b21d9c9e599525a49534b0381f05bb00118a46422fc5f6eac53a6d550`
- **버전 검사 스크립트 실행 결과**:
```text
$ bash scripts/check_p6_decoder_version.sh results_s7/decoder_D_v36.pt
[PASS] [P6-GATE] Verified DECODER checkpoint: 'results_s7/decoder_D_v36.pt' (v3.6)
반환코드 (CHECK_VERSION_RC): 0
```

---

## 2. 디코더 E 시작 상태 점검 (TASK-0036 §2)

디코더 D 종결 직후(22:58:43 KST) `s7_train_p4_prime_resume2_v36.sh` 러너에 의해 디코더 E 학습이 순차 기동되었습니다.

- **체크포인트 존재 여부**: `results_s7/decoder_E_v36.pt`는 학습 진행 중으로 아직 미생성 상태입니다.
- **로그 파일**: `raw_s7/train_decoder_E_v36.log`
- **시작 시각**: 2026-10-05 22:58:43 KST
- **로그 설정 블록 및 z 파일 원문 인용**:
```text
================================================================================
S7 Decoder Training with Pure Gemma Base (PREREG_S7_D_v3_6 / Ledger #419)
Timestamp: 2026-10-05 22:58:43
Device: cuda:0 | Lambda: 1.0 | Margin: 1.0 | Steps: 3000 | Batch: 8 (micro: 2, accum: 4)
================================================================================
Loaded 993 train items from configs/s7_train_993.json
Loaded 5000 records from dataset
Active train items: N=993 (excluded due to boxed cutoff: 0, indices: [])
Loaded z* tensor file from results_s7/zsrc_train1000_P4.pt (Keys: 1000)
```
  - **입력 z 파일 경로**: `results_s7/zsrc_train1000_P4.pt`
  - **키 개수**: 1,000개
- **첫 Step 행 원문 인용**:
```text
[Step 0001 Batch Q-IDs]: [4298, 925, 138, 3225, 1945, 2349, 409, 1959]
[Step 0001/3000] loss=3.3137 | l_task=2.3002 | l_align=1.0135 | NLL_m_f32=2.3002 | NLL_s_f32=2.2867 | Delta_f32=-0.0135 | NLL_fmt=2.2598 | NLL_cnt=3.0312 | act_ratio_100=100.00% | max_diff_s=2.0537e-02 | grad_norm=8.7120e+01 | w_frob=52.109351 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=20.30 GiB | 8.24s/step
```
- **현재 Step 및 연산 속도**: Step 1 마침 (Step 2 및 3 배치 Q-IDs 전개 중), 속도 **8.24s/step**.
- **예상 종결 시각**: 3,000걸음 소요 시간 약 24,720초 (~6.86시간), **2026-10-06 ~05:50 KST** 종결 전망.

---

## 3. 대기 장치 및 연쇄 실행기 로그 현황 (TASK-0036 §3)

### 3.1 `raw_s7/p5_resume_waiter.log` 마지막 2행
```text
[CHECK 2026-10-05 23:04:43] procs=3 | de_done=0 (D=1, E=0) | gpu0_mem=20010MiB (idle=0) | p5_not_done=1 | state=WAITING
[CHECK 2026-10-05 23:09:43] procs=3 | de_done=0 (D=1, E=0) | gpu0_mem=30460MiB (idle=0) | p5_not_done=1 | state=WAITING
```
*분석*: 대기 장치가 디코더 D 종결(`D=1`)과 디코더 E 진행 중(`E=0`) 상태를 정밀하게 인지하며 대기 루프를 유지하고 있습니다.

### 3.2 `raw_s7/p6_chain.log` 마지막 2행
```text
[P6-CHAIN 2026-10-05 23:04:38] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-05 23:09:38] Waiting for P5_DONE.flag... (300s sleep)
```

---

## 4. GPU 0~7 및 메모리 상태 계측 (TASK-0036 §4)

### 4.1 GPU 0~7 실측 (`nvidia-smi --query-gpu`)
| GPU Index | GPU UUID | GPU 점유율 | VRAM 사용량 | 온도 | 상태 요약 |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | **100 %** | **19,592 MiB** | **51 ℃** | **디코더 E 학습 가동 중** |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 13 MiB | 27 ℃ | 유휴 (이상 없음) |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 13 MiB | 29 ℃ | 유휴 (이상 없음) |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 13 MiB | 27 ℃ | **영구 봉인 유지** |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |

### 4.2 실행 중인 GPU 연산 앱 (`nvidia-smi --query-compute-apps`)
```text
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 65822, 23100 MiB
```
- 단일 프로세스 PID 65822(디코더 E 학습 파이썬 프로세스)만 물리 GPU 0에 배정되어 점유 중.

### 4.3 시스템 메모리 (`free -h`)
```text
               total        used        free      shared  buff/cache   available
Mem:           251Gi       7.3Gi       208Gi        33Mi        35Gi       242Gi
Swap:          1.9Gi          0B       1.9Gi
```
- 물리 가용 메모리: 242 GiB available. 스왑 사용: 0 바이트.

---

## 5. 결론

1. **3차 재부팅 부재**: 16:33 부팅 이후 재부팅 없이 건전 가동 중입니다.
2. **디코더 D 종결**: Step 3000 도달 및 최종 체크포인트 `decoder_D_v36.pt` (SHA: `1a09144b...`, 226 MB) 저장 마침, 버전 가드 기준 부합(RC 0).
3. **디코더 E 순항**: 22:58:43 KST 기동되어 z_src(1,000건) 입력으로 Step 1 마침 후 8.24s/step 속도로 GPU 0 단독 점유 학습 중.
4. **후속 대기**: 디코더 E 종결(~10-06 05:50 KST 전망) 시 `p5_resume_waiter`에 의해 GPU 0에서 P5 본 실행이 자동 재개될 예정입니다.

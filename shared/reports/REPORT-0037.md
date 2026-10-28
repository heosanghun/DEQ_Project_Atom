<!-- SHA256: 96cb8c500787b7a022e127585e76bef8972ddb7df9ca948d08779cdaa2cfef05 -->
# REPORT-0037: 디코더 E 진행, 대기 장치 현황 및 재부팅 여부 계측 보고서

- **작성 일시**: 2026-10-06 06:15 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0037.md
- **앞당기는 게이트**: G2·G1 (디코더 E 진행 점검 및 P5 자동 재개 대기 상태 점검)
- **절대 조건 준수 현황**:
  - 세션 무간섭(kill·attach·수정 0건): 준수 (조회 작업만 수행)
  - 읽기 전용 계측: 준수
  - GPU 사용 0건: 준수
  - 기존 파일 수정 0건: 준수
  - P5 값 봉인 원칙: 엄격 준수 (수치 열람 및 인용 0건, 행 수 및 키 개수만 기술)
  - 서버 접속: cts_ssh.py(v2b) 묶음 1회 준수 (06:07:30, exit=0)
  - 금지 어휘 0건 원칙: 준수

---

## 0. 재부팅 여부 점검 (TASK-0037 §0)

- **호스트 가동 시간 실측 (`uptime -s` / `uptime`)**:
```text
$ uptime -s
2026-10-05 16:32:53

$ uptime
 06:07:36 up 13:34,  3 users,  load average: 1.19, 1.48, 1.23
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
- **판정**: 부팅 시각은 16:32:53(16:33 부팅) 불변이며 13시간 34분간 연속 가동 중입니다. **3차 재부팅은 발생하지 않았으며**, 10-05 19:54 기동된 tmux 세션 3종이 온전하게 유지되고 있습니다.

---

## 1. 디코더 E 진행 상태 점검 (TASK-0037 §1)

- **최종 모델 저장 여부**: `results_s7/decoder_E_v36.pt` 부재 (`raw_s7/train_decoder_E_v36.log` 내 최종 체크포인트 저장 행 및 `Training completed` 행 미도달, 학습 진행 중).
- **마지막 3개 Step 행 원문 인용**:
```text
[Step 1800/3000] loss=1.5034 | l_task=0.7625 | l_align=0.7408 | NLL_m_f32=0.7625 | NLL_s_f32=1.0217 | Delta_f32=+0.2592 | NLL_fmt=0.7754 | NLL_cnt=0.4250 | act_ratio_100=96.25% | max_diff_s=5.2052e-03 | grad_norm=4.0860e+00 | w_frob=53.169844 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.60 GiB | 12.31s/step
[Step 1900 Batch Q-IDs]: [3039, 868, 2821, 1259, 925, 2337, 1369, 617]
[Step 1900/3000] loss=1.4261 | l_task=0.9465 | l_align=0.4796 | NLL_m_f32=0.9465 | NLL_s_f32=1.4668 | Delta_f32=+0.5204 | NLL_fmt=0.9814 | NLL_cnt=0.5723 | act_ratio_100=92.25% | max_diff_s=1.4613e-02 | grad_norm=5.6550e+00 | w_frob=53.228563 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.60 GiB | 12.30s/step
[Step 2000 Batch Q-IDs]: [2205, 728, 4778, 427, 2871, 868, 895, 4061]
[Step 2000/3000] loss=1.7347 | l_task=0.9342 | l_align=0.8004 | NLL_m_f32=0.9342 | NLL_s_f32=1.1338 | Delta_f32=+0.1996 | NLL_fmt=0.9570 | NLL_cnt=0.5532 | act_ratio_100=92.75% | max_diff_s=2.0468e-02 | grad_norm=5.0182e+00 | w_frob=53.295068 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.60 GiB | 12.26s/step
```
- **중간 체크포인트 저장**: `results_s5/decoder_E_v36_step2000.pt` 저장 (SHA: `3e2cca4c20371f22e4325a57e8a2f6388974bb353528da12edc32f3d9d511b20`).
- **현재 Step 및 연산 속도**: Step 2000 / 3000 도달, 연산 속도 **12.26s/step**.
- **연산 속도 분석 및 예상 종결 시각**:
  - 디코더 D(lambda=0, 단일 task loss) 대비 디코더 E는 lambda=1.0(정렬 손실 계산 포함)으로 인해 스텝당 소요 시간이 7.3s에서 12.3s로 실측되었습니다.
  - 잔여 1,000걸음 소요 시간: 약 12,260초 (~3시간 24분).
  - 디코더 E 예상 종결 시각: **2026-10-06 ~09:30 - 09:35 KST**.
- **버전 검사 스크립트 실행 여부**: 최종 체크포인트 미생성으로 미실행.

---

## 2. 대기 장치 상태 및 플래그 점검 (TASK-0037 §2)

### 2.1 `raw_s7/p5_resume_waiter.log` 마지막 4행 원문 인용
```text
[CHECK 2026-10-06 05:49:53] procs=3 | de_done=0 (D=1, E=0) | gpu0_mem=16468MiB (idle=0) | p5_not_done=1 | state=WAITING
[CHECK 2026-10-06 05:54:53] procs=3 | de_done=0 (D=1, E=0) | gpu0_mem=16688MiB (idle=0) | p5_not_done=1 | state=WAITING
[CHECK 2026-10-06 05:59:54] procs=3 | de_done=0 (D=1, E=0) | gpu0_mem=21238MiB (idle=0) | p5_not_done=1 | state=WAITING
[CHECK 2026-10-06 06:04:54] procs=3 | de_done=0 (D=1, E=0) | gpu0_mem=29116MiB (idle=0) | p5_not_done=1 | state=WAITING
```
- **대기 상태 판정**: 디코더 D 종결(`D=1`)과 디코더 E 학습 진행 중(`E=0`) 및 GPU 0 점유(`idle=0`)를 감지하여 건전하게 대기 루프(`state=WAITING`)를 유지하고 있습니다.

### 2.2 이상(ABNORMAL) 및 취소 플래그 점검
- `raw_s7/P4PRIME_ABNORMAL.flag`: 부재
- `raw_s7/P5_RESUME_CANCEL.flag`: 부재

---

## 3. P5 재개 상태 점검 (TASK-0037 §3, P5 값 봉인 준수)

디코더 E가 학습 중이므로 P5 자동 재개는 아직 실행되지 않은 상태입니다.

- **`raw_s7/P5_RESUMED.flag`**: 부재 (재개 전)
- **P5 실행기 로그 경로**: `raw_s7/heldout520_P5.log`
  - 마지막 문항 번호 헤더: `[0153/0520]`
  - 문항 번호 헤더 개수: 153건
  - `Skipping` 행 개수: 0건
  - *(P5 값 봉인 규정에 따라 수치 일체 미조회·미인용)*
- **기존 출력 파일 상태 (`results_s7/zstar_s7_a0.25_heldout520_lowest.pt`)**:
  - 파일 크기: 198,062,857 바이트 (189 MB)
  - 수정 시각: `2026-10-05 04:43:48.469754210 +0900`
  - 키 개수 (CPU, `nice 19` 실측): **150개** (값 열람 0건 유지)
- **`raw_s7/P5_DONE.flag`**: 부재

---

## 4. 연쇄 실행기 로그 현황 (TASK-0037 §4)

`raw_s7/p6_chain.log` 마지막 2행:
```text
[P6-CHAIN 2026-10-06 05:59:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 06:04:39] Waiting for P5_DONE.flag... (300s sleep)
```

---

## 5. GPU 0~7 및 시스템 메모리 상태 계측 (TASK-0037 §5)

### 5.1 GPU 0~7 실측 (`nvidia-smi --query-gpu`)
| GPU Index | GPU UUID | GPU 점유율 | VRAM 사용량 | 온도 | 상태 요약 |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | **100 %** | **25,182 MiB** | **53 ℃** | **디코더 E 학습 가동 중** |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 13 MiB | 27 ℃ | 유휴 (이상 없음) |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 13 MiB | 29 ℃ | 유휴 (이상 없음) |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 13 MiB | 27 ℃ | **영구 봉인 유지** |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 13 MiB | 27 ℃ | 유휴 (이상 없음) |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |

### 5.2 GPU 연산 프로세스 (`nvidia-smi --query-compute-apps`)
```text
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 65822, 16144 MiB
```
- 단일 프로세스 PID 65822(디코더 E 학습)만 물리 GPU 0에서 가동 중.

### 5.3 시스템 메모리 (`free -h`)
```text
               total        used        free      shared  buff/cache   available
Mem:           251Gi       7.4Gi       207Gi        33Mi        36Gi       242Gi
Swap:          1.9Gi          0B       1.9Gi
```
- 물리 가용 메모리: 242 GiB available. 스왑 사용: 0 바이트.

---

## 6. 결론 및 향후 전망

1. **호스트 건전성**: 3차 재부팅 없이 13시간 34분간 연속 가동 중입니다.
2. **디코더 E**: Step 2000에 도달하였으며, 정렬 손실 계산에 따른 스텝당 12.3s 소요로 잔여 1,000걸음은 **09:30 - 09:35 KST**경 종결될 전망입니다.
3. **P5 자동 재개 파이프라인**: 디코더 E 종결 후 `p5_resume_waiter`에 의해 GPU 0 UUID 고정 격리 하에 P5 본 실행이 이어 계산될 예정입니다.

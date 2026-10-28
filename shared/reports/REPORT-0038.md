<!-- SHA256: 31c59b001118cf9ba56b54cc434fece72ad01b4eb652fb3fa5e11cfea392f7e6 -->
# REPORT-0038: 디코더 E 최종본 확보 및 P5 자동 재개 대기 상태 계측 보고서

- **작성 일시**: 2026-10-06 10:00 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0038.md
- **앞당기는 게이트**: G2·G1 (디코더 E 최종본 점검 및 P5 GPU 0 자동 재개 상태 점검)
- **절대 조건 준수 현황**:
  - 세션 무간섭(kill·attach·수정 0건): 준수 (조회 작업만 수행)
  - 읽기 전용 계측: 준수
  - GPU 사용 0건: 준수
  - 기존 파일 수정 0건: 준수
  - P5 값 봉인 원칙: 엄격 준수 (수치 열람 및 인용 0건, 행 수 및 키 개수만 기술)
  - 서버 접속: cts_ssh.py(v2b) 묶음 1회 준수 (09:51:47, exit=0)
  - 금지 어휘 0건 원칙: 준수

---

## 0. 재부팅 여부 점검 (TASK-0038 §0)

- **호스트 가동 시간 실측 (`uptime -s` / `uptime`)**:
```text
$ uptime -s
2026-10-05 16:32:53

$ uptime
 09:51:53 up 17:19,  2 users,  load average: 0.02, 0.04, 0.05
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
```
- **판정**: 부팅 시각은 16:32:53(16:33 부팅) 불변이며 17시간 19분간 연속 가동 중입니다. **3차 재부팅은 발생하지 않았습니다.** 디코더 E 학습 세션(`s7_train_p4_prime`)은 학습 종결에 따라 자연 종료되었으며, 대기 장치 세션 2종(`p5_resume_waiter`, `p6_chain`)이 온전하게 유지되고 있습니다.

---

## 1. 디코더 E 최종본 점검 (TASK-0038 §1)

### 1.1 최종 모델 체크포인트 제원
- **파일 경로**: `results_s7/decoder_E_v36.pt`
- **파일 크기**: 236,027,215 바이트 (226 MB, 2026-10-06 08:59 생성)
- **SHA-256**: `f8115ac33f44045ce5f572a377ee84b2e6311be347f154190b6c79199d7f3cac`
- **버전 검사 스크립트 실행 결과**:
```text
$ bash scripts/check_p6_decoder_version.sh results_s7/decoder_E_v36.pt
[PASS] [P6-GATE] Verified DECODER checkpoint: 'results_s7/decoder_E_v36.pt' (v3.6)
CHECK_VERSION_RC=0
```
- **판정**: 디코더 E 최종본(v3.6)이 온전하게 생성되었으며 버전 검사를 문제없이 마쳤습니다.

### 1.2 학습 로그 원문 인용 (`raw_s7/train_decoder_E_v36.log`)
- **최종 체크포인트 저장 행**:
```text
[CHECKPOINT] Saved step 3000 to /home/shoon/cts_e1_s5/results_s5/decoder_E_v36_step3000.pt (SHA: 399934c6b30ba4b24d89e23c2664fe329ad80f246608879aadd5840e4168faf5)
[OUT-CHECKPOINT] Saved final model to results_s7/decoder_E_v36.pt (SHA: f8115ac33f44045ce5f572a377ee84b2e6311be347f154190b6c79199d7f3cac)
```
- **마지막 3개 Step 행**:
```text
[Step 2800/3000] loss=1.2425 | l_task=0.7679 | l_align=0.4745 | NLL_m_f32=0.7679 | NLL_s_f32=1.3611 | Delta_f32=+0.5932 | NLL_fmt=0.7837 | NLL_cnt=0.5703 | act_ratio_100=79.75% | max_diff_s=1.7338e-02 | grad_norm=2.2478e+00 | w_frob=53.777193 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.60 GiB | 12.10s/step
[Step 2900 Batch Q-IDs]: [1650, 4885, 4714, 882, 564, 4985, 30, 2512]
[Step 2900/3000] loss=1.2318 | l_task=0.9766 | l_align=0.2551 | NLL_m_f32=0.9766 | NLL_s_f32=1.7321 | Delta_f32=+0.7555 | NLL_fmt=0.9824 | NLL_cnt=0.8989 | act_ratio_100=80.75% | max_diff_s=1.8771e-02 | grad_norm=5.5271e+00 | w_frob=53.842606 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.60 GiB | 12.05s/step
[Step 3000 Batch Q-IDs]: [1322, 2240, 1011, 257, 218, 3713, 1419, 2279]
[Step 3000/3000] loss=1.2400 | l_task=0.8264 | l_align=0.4136 | NLL_m_f32=0.8264 | NLL_s_f32=1.8753 | Delta_f32=+1.0490 | NLL_fmt=0.8398 | NLL_cnt=0.4230 | act_ratio_100=76.50% | max_diff_s=1.1630e-03 | grad_norm=7.5125e+00 | w_frob=53.912793 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.60 GiB | 12.01s/step
```
- **학습 종결 행**:
```text
Training completed in 36026.18s (600.44m).
```
- **학습 종결 시각**: 2026-10-06 08:59 KST (총 10시간 26초 소요).

---

## 2. 대기 장치 상태 및 플래그 점검 (TASK-0038 §2)

### 2.1 `raw_s7/p5_resume_waiter.log` 최근 감시 행 원문 인용
```text
[CHECK 2026-10-06 09:34:59] procs=1 | de_done=1 (D=1, E=1) | gpu0_mem=13MiB (idle=1) | p5_not_done=1 | state=WAITING
[CHECK 2026-10-06 09:39:59] procs=1 | de_done=1 (D=1, E=1) | gpu0_mem=13MiB (idle=1) | p5_not_done=1 | state=WAITING
[CHECK 2026-10-06 09:44:59] procs=1 | de_done=1 (D=1, E=1) | gpu0_mem=13MiB (idle=1) | p5_not_done=1 | state=WAITING
[CHECK 2026-10-06 09:50:00] procs=1 | de_done=1 (D=1, E=1) | gpu0_mem=13MiB (idle=1) | p5_not_done=1 | state=WAITING
```

### 2.2 플래그 및 tmux 세션 상태
- `raw_s7/P4PRIME_ABNORMAL.flag`: 부재
- `raw_s7/P5_RESUME_CANCEL.flag`: 부재
- tmux `p5_resume_waiter` 세션: 가동 중 유지 (`WAITER_TMUX_RC=0`)

### 2.3 대기 장치 상태 분석
- `de_done=1 (D=1, E=1)`: 디코더 D와 E 체크포인트가 모두 확보되었습니다.
- `gpu0_mem=13MiB (idle=1)`: 물리 GPU 0이 유휴 상태입니다.
- `p5_not_done=1`: P5 계산이 미종결 상태입니다.
- **대기 유지 요인**: `procs=1`로 계측되어, 대기 스크립트 내 기동 조건인 `procs=0`을 충족하지 못해 트리거가 지연되고 대기 루프(`state=WAITING`)에 머물러 있습니다.

---

## 3. P5 재개 상태 점검 (TASK-0038 §3, P5 값 봉인 준수)

대기 장치가 `procs=1`로 인해 트리거에 도달하지 못하여 P5 자동 재개는 아직 실행 전 단계입니다.

- **`raw_s7/P5_RESUMED.flag`**: 부재 (재개 전)
- **실행기 스크립트 해시**:
  - `scripts/p5_heldout520_launch_v8.sh`: SHA-256 `6f0cbcd282a909ca97ed557672feecb02415fda3d4aed3933f22871d1e2b2041`
- **P5 실행 프로세스**: 미가동 (`NO_P5_PROCESS`)
- **P5 로그 상태 (`raw_s7/heldout520_P5.log`)**:
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

## 4. 연쇄 실행기 로그 현황 (TASK-0038 §4)

`raw_s7/p6_chain.log` 마지막 2행:
```text
[P6-CHAIN 2026-10-06 09:44:39] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-06 09:49:39] Waiting for P5_DONE.flag... (300s sleep)
```

---

## 5. GPU 0~7 및 시스템 메모리 상태 계측 (TASK-0038 §5)

### 5.1 GPU 0~7 실측 (`nvidia-smi --query-gpu`)
| GPU Index | GPU UUID | GPU 점유율 | VRAM 사용량 | 온도 | 상태 요약 |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | **0 %** | **13 MiB** | **29 ℃** | **유휴 (학습 종료 후 대기)** |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 0 % | 13 MiB | 26 ℃ | 유휴 (이상 없음) |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 13 MiB | 26 ℃ | 유휴 (이상 없음) |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 13 MiB | 26 ℃ | **영구 봉인 유지** |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 13 MiB | 26 ℃ | 유휴 (이상 없음) |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 13 MiB | 27 ℃ | 유휴 (이상 없음) |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 13 MiB | 27 ℃ | 유휴 (이상 없음) |

### 5.2 GPU 연산 프로세스 (`nvidia-smi --query-compute-apps`)
```text
gpu_uuid, pid, used_gpu_memory [MiB]
```
- GPU 0~7 전체에서 실행 중인 연산 프로세스 0건 (완전 유휴).

### 5.3 시스템 메모리 (`free -h`)
```text
               total        used        free      shared  buff/cache   available
Mem:           251Gi       3.7Gi       210Gi        17Mi        37Gi       246Gi
Swap:          1.9Gi          0B       1.9Gi
```
- 물리 가용 메모리: 246 GiB available. 스왑 사용: 0 바이트.

---

## 6. 결론 및 향후 전망

1. **호스트 건전성**: 3차 재부팅 없이 17시간 19분간 연속 가동 중입니다.
2. **디코더 E 최종본 확보**: Step 3000 학습이 08:59 KST 종결되어 `results_s7/decoder_E_v36.pt` (SHA: `f8115ac3...`)가 온전하게 저장되었으며 버전 검사(RC=0)를 충족하였습니다.
3. **P5 자동 재개 대기 상태**: 디코더 D와 E 종결 및 GPU 0 유휴 상태가 갖추어졌으나, 대기 스크립트 상 `procs=1` 계측으로 인해 트리거가 지연되고 있습니다.

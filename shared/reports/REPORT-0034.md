<!-- SHA256: a4e5f13a61bec66be68e941ce6bc2b48718609ce45a0785fc7c6ea53bb38c8e4 -->
# REPORT-0034: 디코더 D 종료·E 시작 점검 및 대기 장치 상태 계측 보고서

- **작성 일시**: 2026-10-05 19:55 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0034.md
- **앞당기는 게이트**: G2 (디코더 D 최종본 및 E 시작 점검)
- **절대 조건 준수 현황**:
  - 세션 무간섭 유지: 준수 (조회 작업만 수행, 프로세스 수정 0건)
  - 읽기 전용 접근: 준수
  - P5 결과 값 열람 0건: 준수
  - GPU 사용 0건: 준수
  - 기존 파일 수정 0건: 준수
  - 서버 접속: cts_ssh.py(v2b) 묶음 1회 준수 (19:49:27, exit=0)
  - 금지 어휘 0건 원칙: 준수

---

## 1. 디코더 D 상태 계측 (TASK-0034 §1)

### 1.1 최종 모델 체크포인트 저장 행 조회
- `raw_s7/train_decoder_D_v36.log` 내 최종 체크포인트 저장 행 조회 결과:
```text
[NOT FOUND] Saved final model line not found in log
```
`results_s7/decoder_D_v36.pt` 파일은 부재 상태이며, 디코더 D 학습은 최종 도달에 이르지 못하였습니다.

### 1.2 마지막 3개 Step 행 원문 인용
`raw_s7/train_decoder_D_v36.log`의 마지막 3개 step 행은 다음과 같습니다:
```text
[Step 1300/3000] loss=0.6065 | l_task=0.6065 | l_align=0.0000 | NLL_m_f32=0.6065 | NLL_s_f32=0.6187 | Delta_f32=+0.0122 | NLL_fmt=0.6099 | NLL_cnt=0.4526 | act_ratio_100=0.00% | max_diff_s=0.0000e+00 | grad_norm=2.9716e-01 | w_frob=52.685157 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.59 GiB | 7.31s/step
[Step 1400/3000] loss=0.8091 | l_task=0.8091 | l_align=0.0000 | NLL_m_f32=0.8091 | NLL_s_f32=0.8230 | Delta_f32=+0.0139 | NLL_fmt=0.8164 | NLL_cnt=0.6006 | act_ratio_100=0.00% | max_diff_s=0.0000e+00 | grad_norm=5.0623e-01 | w_frob=52.726716 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.59 GiB | 7.32s/step
[Step 1500/3000] loss=0.9020 | l_task=0.9020 | l_align=0.0000 | NLL_m_f32=0.9020 | NLL_s_f32=0.9061 | Delta_f32=+0.0041 | NLL_fmt=0.9189 | NLL_cnt=0.5012 | act_ratio_100=0.00% | max_diff_s=0.0000e+00 | grad_norm=3.4831e-01 | w_frob=52.784555 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.59 GiB | 7.32s/step
```

### 1.3 체크포인트 현황 및 버전 점검 스크립트 실행 결과
- `results_s7/decoder_D_v36.pt`: 부재 (NOT FOUND)
- 디스크 상 보존된 최신 체크포인트:
  - 경로: `results_s5/decoder_D_v36_step1500.pt`
  - SHA-256: `649b435f94d3b70f0744895d68333ed48cd0c72441d28650afe6b387115e2525`
- `check_p6_decoder_version.sh` 실행: 대상 파일(`decoder_D_v36.pt`) 부재로 미실행.
- 현재 Step 및 속도: Step 1500 / 3000, 7.32s/step (단, 아래 §4 진단과 같이 호스트 재부팅으로 인해 프로세스는 중단 상태).

---

## 2. 디코더 E 상태 계측 (TASK-0034 §2)

- `results_s7/decoder_E_v36.pt`: 부재
- `raw_s7/train_decoder_E_v36.log`: 부재
- 디코더 D 학습이 Step 1500에서 중단됨에 따라 후속 직렬 단계인 디코더 E는 시작되지 않았습니다 (시작 시각, 첫 step, 현재 step, s/step, 예상 종료 시각, z_src 설정 블록 전건 미생성).

---

## 3. 대기 장치 및 연쇄 실행기 로그 현황 (TASK-0034 §3)

### 3.1 `raw_s7/p5_resume_waiter.log` 마지막 2행 원문 인용
```text
[CHECK 2026-10-05 16:24:41] procs=4 | de_done=0 (D=0, E=0) | gpu0_mem=16674MiB (idle=0) | p5_not_done=1 | state=WAITING
[CHECK 2026-10-05 16:29:41] procs=4 | de_done=0 (D=0, E=0) | gpu0_mem=42890MiB (idle=0) | p5_not_done=1 | state=WAITING
```

### 3.2 `raw_s7/p6_chain.log` 마지막 2행 원문 인용
```text
[P6-CHAIN 2026-10-05 16:24:25] Waiting for P5_DONE.flag... (300s sleep)
[P6-CHAIN 2026-10-05 16:29:25] Waiting for P5_DONE.flag... (300s sleep)
```

---

## 4. 시스템 프로세스 및 호스트 상태 진단

- **tmux 세션 조회 (`tmux ls`)**:
```text
error connecting to /tmp/tmux-8001/default (No such file or directory)
```
- **관련 프로세스 조회 (`ps -ef | grep -E ...`)**:
  - `s7_train_p4_prime` (디코더 D 러너): 부재 (0건)
  - `p5_resume_waiter` (P5 자동 재개 대기 장치): 부재 (0건)
  - `p6_chain` (P6 연쇄 실행기): 부재 (0건)
- **진단**:
  - 3개 감시 로그(`train_decoder_D_v36.log`, `p5_resume_waiter.log`, `p6_chain.log`)의 기록이 모두 **2026-10-05 16:29:41 KST** 직후 동시에 정지되었습니다.
  - tmux 소켓(`/tmp/tmux-8001/default`)이 소멸된 상태로, 이는 금일 04:50경 발생했던 호스트 재부팅과 동일하게 **16:30경 서버 호스트의 불시 재부팅**이 발생하였음을 나타냅니다.
  - 이로 인해 실행 중이던 P4' 디코더 D 학습 및 백그라운드 대기 장치 2종(`p5_resume_waiter`, `p6_chain`)이 모두 소멸되었습니다.

---

## 5. GPU 0~7 및 시스템 메모리 상태 계측 (TASK-0034 §4)

### 5.1 GPU 0~7 상태 (`nvidia-smi` 쿼리 실측)
| GPU Index | Device Name | GPU 점유율 | VRAM 사용량 / 전체 | 온도 |
|:---:|:---|:---:|:---:|:---:|
| 0 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49140 MiB | 26 ℃ |
| 1 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49140 MiB | 26 ℃ |
| 2 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49140 MiB | 26 ℃ |
| 3 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49140 MiB | 27 ℃ |
| 4 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49140 MiB | 26 ℃ |
| 5 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49140 MiB | 26 ℃ |
| 6 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49140 MiB | 27 ℃ |
| 7 | NVIDIA RTX A6000 | 0 % | 13 MiB / 49140 MiB | 27 ℃ |

*특이사항*: 04:50 재부팅 이후 지속되었던 GPU 1의 `requires reset` 하드웨어 결함 상태가 16:30 재부팅을 통해 초기화되어 현재 전 장치(0~7번)가 0% 점유율 및 13 MiB 유휴 상태로 복원되었습니다.

### 5.2 시스템 메모리 계측 (`free -h` / `free -m`)
- **`free -h` 실측**:
```text
               total        used        free      shared  buff/cache   available
Mem:           251Gi       2.6Gi       246Gi        17Mi       2.2Gi       247Gi
Swap:          1.9Gi          0B       1.9Gi
```
- **`free -m` 실측**:
```text
               total        used        free      shared  buff/cache   available
Mem:          257575        2664      252650          17        2260      253350
Swap:           1951           0        1951
```
- 물리 메모리 여유: 246 GiB free (가용 247 GiB). 스왑 점유: 0 바이트.

---

## 6. 결론 및 후속 조치 건의

1. **디코더 D**: Step 1500에서 정지되었으며 체크포인트 `results_s5/decoder_D_v36_step1500.pt` (SHA: `649b435f94d3b70f0744895d68333ed48cd0c72441d28650afe6b387115e2525`) 보존 상태입니다.
2. **디코더 E**: 디코더 D의 미도달로 미기동 상태입니다.
3. **대기 세션**: 16:30 재부팅으로 `p5_resume_waiter` 및 `p6_chain` 세션 소멸 상태입니다.
4. **GPU 1**: 재부팅을 통해 리셋 필요 상태가 해소되어 유휴 상태로 대기 중입니다.
5. **권고**: 디코더 D를 `decoder_D_v36_step1500.pt`로부터 이어서 재개하고, `p5_resume_waiter` 및 `p6_chain` 세션을 재기동하는 후속 태스크 발행이 요구됩니다.

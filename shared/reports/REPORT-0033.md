<!-- SHA256: f1445530b699ea571a7429bb7c70b9023e42e186e55cd53eb9b7d3ce0e467ef5 -->
# REPORT-0033: P4' 종료 후 GPU 0 P5 자동 재개 대기 장치 배치 및 실행기 v8 점검 보고서

- **작성 일시**: 2026-10-05 16:20 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0033.md
- **앞당기는 게이트**: G1 (P5 재개 — 사전등록 v1.3 §10(f) 기본 경로: GPU 1 미달 → GPU 0 직렬)
- **절대 조건 준수 현황**:
  - 실행 중 세션(P4' GPU 0, p6_chain) 무간섭 유지: 준수 (kill·attach·수정 0건, 세션 3종 온전 가동)
  - 대기 장치 자체의 GPU 점유 0건: 준수 (CPU bash 감시 루프 전용)
  - P5는 GPU 0에서만 1회 재개: 준수 (v8에 물리 GPU 0 UUID 강제 격리 반영)
  - GPU 1·2·3·5·6·7 사용 0건, GPU 4 영구 봉인 유지: 준수
  - 기존 파일 보존: 준수 (기존 파일 덮어쓰기 0건, v8 및 대기 장치 신규 파일 분리)
  - P5 결과 값 열람 0건: 준수 (값 조회 0건 유지)
  - 서버 접속 규약 준수: 준수 (cts_ssh.py v2b SFTP 및 SSH 각 1회, 65초 대기, exit=0)
  - 금지 어휘 0건 원칙: 준수

---

## 1. 실행기 점검 및 v8 보완 (TASK-0033 §1)

### 1.1 `scripts/p5_heldout520_launch_v7.sh` 4대 요건 점검 결과
기존 v7 실행기(`b69adaa54ec653b20556dca91c775642ac02083d500d8cfc050cc099ec96d740`)의 행 번호별 정밀 대조 결과, 4대 항목 전건에서 보완이 요구됨을 계측하였습니다.

| 점검 항목 | v7 행 번호 | v7 코드 실측 상태 | 적합 여부 |
|---|:---:|---|:---:|
| **(a) 물리 GPU 0 UUID 고정** | 48~50행 | `TARGET_PHYSICAL_GPU_INDEX=2`, `TARGET_GPU_UUID="GPU-c813..."` (GPU 2 지정) | **미충족** |
| **(b) P5_RESUME=1 이어 계산** | 101~105행 | `[CRITICAL ERROR] P5 production execution is strictly blocked ... exit 1` (본 실행 전면 차단) | **미충족** |
| **(c) P5_RESUMED.flag 기록** | 86, 98행 | SMOKE 모드 임시 플래그만 존재, 본 실행 재개 시각·출력 SHA·키 수·UUID·PCI 기록 부재 | **미충족** |
| **(d) SHA 검사 10종** | 112~122행 | 9종 파일만 등재 (`PREREG_S7_DEQ_v1.3.md` 누락) | **미충족** |

### 1.2 신규 실행기 `scripts/p5_heldout520_launch_v8.sh` 작성 및 제원
TASK-0033 §1 지침에 따라 4대 요건을 모두 충족하는 신규 실행기 v8을 작성하여 서버에 배치하였습니다.

- **스크립트 경로**: `scripts/p5_heldout520_launch_v8.sh`
- **파일 크기**: 11,169 바이트
- **SHA-256**: `6f0cbcd282a909ca97ed557672feecb02415fda3d4aed3933f22871d1e2b2041`
- **요건 충족 세부 구현**:
  - **(a) 물리 GPU 0 UUID 격리 고정 (48~73행, 126~146행)**:
    `TARGET_PHYSICAL_GPU_INDEX=0`, `TARGET_GPU_UUID="GPU-71be98b0-6b81-9257-c622-97dc97d46480"`, `TARGET_PCI_BUS="00000000:6F:00.0"`를 선언하고, `nvidia-smi` 쿼리로 UUID 및 PCI 버스를 대조 검증한 뒤 `export CUDA_VISIBLE_DEVICES="${TARGET_GPU_UUID}"` 및 `DEVICE="cuda:0"`으로 단독 노출 고정.
  - **(b) P5_RESUME=1 및 기존 출력 이어 계산 지원 (177~217행)**:
    `STARTED_FLAG`와 `OUT_PT`(`results_s7/zstar_s7_a0.25_heldout520_lowest.pt`) 존재 검사 후 기존 출력 파일의 보존 상태에서 `s7_zstar_lowest_run.py`가 미수행 문항만 이어 연산하도록 허용.
  - **(c) `raw_s7/P5_RESUMED.flag` 메타데이터 기록 (192~209행)**:
    재개 시각(`resumed_at`), 출력 경로(`output_file`), 출력 SHA(`output_sha256`), 기존 키 수(`key_count`), 대상 GPU UUID(`target_gpu_uuid`), PCI 버스(`target_pci_bus`), 장치(`device`)를 빠짐없이 기록.
  - **(d) 10종 사전 필수 파일 SHA-256 강제 검사 (148~175행)**:
    1. `scripts/s7_zstar_lowest_run.py`: `779ea1f1...`
    2. `cts/deq/zstar_canonical.py`: `b7170aeb...`
    3. `cts/deq/transition_inject.py`: `12706843...`
    4. `exe_scratch/s7/ref_solvers/anderson.py`: `06c93011...`
    5. `/home/shoon/checkpoints/m1_final_frozen.pt`: `f2119bbe...`
    6. `configs/s7_new_heldout_520.json`: `48abce65...`
    7. `PREREG_S7_D_v3_7.md`: `45dbe5cdf...`
    8. `PREREG_S7_DEQ_v1.3.md`: `254beb6f...`
    9. `configs/zstar_canonical.yaml`: `bd8e4e96...`
    10. `scratch/harness_A_v1.py`: `5fb768a8...`
    (위 10종 파일 중 1건이라도 불일치 또는 부재 시 즉시 중단)

### 1.3 v8 SMOKE=1 (CPU, train Q#14) 모의 시험 결과
서버 환경에서 `SMOKE=1 bash scripts/p5_heldout520_launch_v8.sh`를 1회 실행하여 동작을 계측하였습니다.

- **10종 사전 필수 파일 SHA 대조**: 10종 전건 `[OK]` 일치 계측.
- **실행 모드**: CPU, seed=2030, max_nfe=5, 단일 문항(Train Q#0014).
- **반환코드**: `0` (종료 코드: 0)
- **소요 시간**: 6.02초 (`q0014_a0.25: r_1000=2.684098e-01, r_lowest=2.684098e-01 at k*=5`)
- **산출 텐서**: `scratch/p5_smoke.pt` (크기: 1,312,813 바이트, SHA: `966dfc928c0e709480f46d38cc3d4054e82e515909398228a5a47f9487ff648a`)
- **플래그 생성**: `scratch/P5_DONE.flag` 생성 마침.

---

## 2. 대기 장치 배치 및 가동 결과 (TASK-0033 §2)

### 2.1 대기 장치 스크립트 제원
- **스크립트 경로**: `scripts/p5_resume_after_p4prime_waiter.sh`
- **파일 크기**: 3,777 바이트
- **SHA-256**: `892ad12d69be4d1c2b93eed4c083a29880473f9bdc6b39fe3fc1bd20becc70b5`
- **4대 조건 감시 논리 (300초 간격)**:
  - (a) `s7_train_p4_prime*` 러너 및 `s7_train_decoder` 프로세스 수 == 0
  - (b) `raw_s7/train_decoder_D_v36.log` 및 `_E_v36.log` 양쪽에 `Saved final model` 행 존재
  - (c) GPU 0 메모리 사용량 < 1,024 MiB
  - (d) `raw_s7/P5_DONE.flag` 부재
- **예외 및 제어 가드**:
  - (a)가 참이나 (b)가 거짓(이상 중단)인 경우: `raw_s7/P4PRIME_ABNORMAL.flag` 기록 후 종료 (P5 미기동).
  - `raw_s7/P5_RESUME_CANCEL.flag` 감지 시: 취소 메시지 기록 후 즉시 종료 (ESC-0005 [A] 채택 대응용).
  - 4대 조건 모두 만족 시: `P5_RESUME=1 bash scripts/p5_heldout520_launch_v8.sh --device cuda:0` 1회 실행.

### 2.2 tmux `p5_resume_waiter` 기동 및 첫 확인 행 계측
- **기동 명령**: `tmux new-session -d -s p5_resume_waiter "bash scripts/p5_resume_after_p4prime_waiter.sh"`
- **기동 일시**: 2026-10-05 16:14:41 KST (PID: 117954)
- **첫 확인 행 원문 인용 (`raw_s7/p5_resume_waiter.log`)**:
  ```text
  [WAITER] Starting p5_resume_after_p4prime_waiter.sh (PID: 117954, Started: 2026-10-05 16:14:41)
  [CHECK 2026-10-05 16:14:41] procs=5 | de_done=0 (D=0, E=0) | gpu0_mem=16702MiB (idle=0) | p5_not_done=1 | state=WAITING
  ```
- **계측 해설**: P4' 직렬 학습 프로세스 5건 가동 중(`procs=5`), D/E 체크포인트 미도달(`de_done=0`), GPU 0 점유 중(16,702 MiB, `idle=0`), P5 미종료(`p5_not_done=1`)로 판정되어 대기 상태(`state=WAITING`)로 규격에 부합하게 진입하였습니다.

---

## 3. 순서 정합성 및 디코더 D 진도 계측 (TASK-0033 §3)

### 3.1 `p6_chain_after_p5_v2.sh` 실행 조건 인용 대조
`scripts/p6_chain_after_p5_v2.sh` 84~99행의 감시 분기 원문을 대조하였습니다.

```bash
84-if [ "${SMOKE:-0}" != "1" ]; then
85:    echo "[P6-CHAIN] Monitoring for P5_DONE.flag and GPU 0 idle (< 1024 MiB)..."
86-    while true; do
87:        if [ -f "${RAW_DIR}/P5_DONE.flag" ]; then
88-            if is_gpu0_idle; then
89:                echo "[P6-CHAIN] P5_DONE.flag present and GPU 0 is idle. Commencing chain execution."
90-                break
91-            else
92:                echo "[P6-CHAIN $(date '+%Y-%m-%d %H:%M:%S')] P5_DONE.flag exists, but GPU 0 memory is occupied. Waiting 300s..."
93-            fi
94-        else
95:            echo "[P6-CHAIN $(date '+%Y-%m-%d %H:%M:%S')] Waiting for P5_DONE.flag... (300s sleep)"
96-        fi
97-        sleep 300
98-    done
99-fi
```

- **순서 정합성 규명**:
  - `p6_chain v2`는 `P5_DONE.flag`의 존재와 `GPU 0 < 1024 MiB`가 동시에 충족될 때까지 300초 간격으로 대기합니다.
  - 따라서 P4' 종료 후 본 대기 장치에 의해 P5가 GPU 0에서 재개되어 동작하는 동안에는 `P5_DONE.flag`가 없으므로 `p6_chain`이 결코 기동되지 않습니다.
  - P5가 계산을 끝마치고 `P5_DONE.flag`를 생성한 뒤 GPU 0 메모리를 반환한 이후에만 `p6_chain`이 0단계(GPU 0 앵커 20)부터 순차 실행되므로, GPU 0 자원 충돌 없이 정합한 실행 순서가 보장됩니다.

### 3.2 디코더 D 학습 진도 실측치
- **로그 파일**: `raw_s7/train_decoder_D_v36.log` (크기 9.9KB, 16:11 KST 기준)
- **최신 도달 스텝**: **Step 1400 / 3000** (진행률 46.7%)
- **최근 3개 스텝 로그 원문**:
  - **Step 1200**:
    ```text
    [Step 1200/3000] loss=0.8119 | l_task=0.8119 | l_align=0.0000 | NLL_m_f32=0.8119 | NLL_s_f32=0.8216 | Delta_f32=+0.0097 | NLL_fmt=0.8184 | NLL_cnt=0.6553 | act_ratio_100=0.00% | max_diff_s=0.0000e+00 | grad_norm=3.4256e-01 | w_frob=52.641069 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.59 GiB | 7.30s/step
    ```
  - **Step 1300**:
    ```text
    [Step 1300/3000] loss=0.6065 | l_task=0.6065 | l_align=0.0000 | NLL_m_f32=0.6065 | NLL_s_f32=0.6187 | Delta_f32=+0.0122 | NLL_fmt=0.6099 | NLL_cnt=0.4526 | act_ratio_100=0.00% | max_diff_s=0.0000e+00 | grad_norm=2.9716e-01 | w_frob=52.685157 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.59 GiB | 7.31s/step
    ```
  - **Step 1400**:
    ```text
    [Step 1400/3000] loss=0.8091 | l_task=0.8091 | l_align=0.0000 | NLL_m_f32=0.8091 | NLL_s_f32=0.8230 | Delta_f32=+0.0139 | NLL_fmt=0.8164 | NLL_cnt=0.6006 | act_ratio_100=0.00% | max_diff_s=0.0000e+00 | grad_norm=5.0623e-01 | w_frob=52.726716 | lr=1.0000e-04 (const) | sched=None | SN(tr/ev)=1.0000/1.0000 | micro/accum=2x4 (eff 8) | peak_mem=39.59 GiB | 7.32s/step
    ```
- **학습 상태**: 스텝당 소요 시간 약 7.32초로 안정적으로 진행 중이며, GPU 0 점유율 99~100% (VRAM 21,342 MiB)를 유지하고 있습니다.

### 3.3 활성 tmux 세션 현황
서버 상에서 3개 세션이 결함 없이 병렬 가동 중임을 확인하였습니다:
1. `s7_train_p4_prime` (GPU 0 직렬 디코더 D 학습 가동)
2. `p6_chain` (P5 종결 및 GPU 0 유휴 감시)
3. `p5_resume_waiter` (P4' 종료 및 디코더 D·E 체크포인트 감시 후 P5 GPU 0 재개 대기)

---

## 4. 종합 결론 및 후속 대기
1. **실행기 v8 및 대기 장치 완비**:
   - `scripts/p5_heldout520_launch_v8.sh` 및 `scripts/p5_resume_after_p4prime_waiter.sh` 배치를 마쳤으며, SMOKE 시험(RC=0) 및 tmux 감시 진입을 계측하였습니다.
2. **사전등록 v1.3 기본 경로 확보**:
   - ESC-0005 결정 지연 시에도 P4' 직렬 학습(D → E) 종료 시점에 맞춰 GPU 0에서 P5가 자동으로 안전하게 재개되도록 파이프라인 대기 장치를 정립하였습니다.

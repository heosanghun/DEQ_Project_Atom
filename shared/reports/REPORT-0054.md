<!-- SHA256: a8b1be83a5b045a22e4e4fe3e8db5db809a19d4a7d7a5d299a252d8e0e6b63cf -->
# REPORT-0054: `launch_v5.sh` 정규 실행 기동 (GPU 1), §5 앵커 점검 충족 및 20분 후 진행 계측

- **작성 일시**: 2026-10-06 21:42 KST
- **작성 주체**: 회사 PC 정식 루프 에이전트 (Antigravity)
- **근거 작업**: `shared/tasks/TASK-0054.md`
- **근거 규정**: `shared/prereg/PREREG_S7_DEQ_v2.1.md` §5·§6 (잠금 SHA `670850c1...`), AUDIT-P009 재감사 2, AUDIT-0053
- **원시 로그**: `scratch/task0054_out.txt` (219행, 단일 SSH 묶음 20분 계측 로그)

---

## 1. 개요 및 핵심 요약

`TASK-0054` 지시에 따라 `scripts/s7deq_soft_v2_launch_v5.sh`의 정규 실행을 GPU 1 전용으로 기동하고, PREREG_S7_DEQ_v2.1 §5 사전 앵커 점검 전수 충족 및 20분 경과 시점의 본 실행 텔레메트리를 단일 SSH 호출로 일괄 수집했다.

1. **사전 전제 조건 대조 (실행 전 점검)**:
   - 실행기 `scripts/s7deq_soft_v2_launch_v5.sh` SHA-256: `7771f43f...` 일치 (`task0054_out.txt:6`).
   - 기준 파일 `results_s7/anchor_ref_gpu1_check.pt` SHA-256: `5103642d...` 일치 (`task0054_out.txt:7`).
   - 사전 충돌 파일(`anchor_soft_eps1e-12_check.pt`, `anchor_ref_gpu1_check_v21_extra.pt`, `s7deq_soft_v2_eps*.pt`) 부재 요건 전건 충족 (`task0054_out.txt:8`).
2. **세션 기동 (`s7deq_soft_v21`)**:
   - 기동 시각: 2026-10-06 21:19:06 KST (`task0054_out.txt:11`).
   - 실행 명령: `env -u TEST_MODE -u TEST_ALTER_REF -u TEST_PASS_ANCHOR tmux new-session -d -s s7deq_soft_v21 'bash scripts/s7deq_soft_v2_launch_v5.sh'` (`task0054_out.txt:13`).
   - 환경변수 박탈(`-u TEST_MODE -u TEST_ALTER_REF -u TEST_PASS_ANCHOR`)을 적용하여 정규 경로 안전성을 보장했다.
3. **§5 사전 앵커 점검 (eps=1e-12) 전수 충족**:
   - 앵커 3문항(Q#20, Q#24, Q#32) 전원 `boundary_step_count = 0` 계측 (`task0054_out.txt:98, 105, 112`).
   - 경계 걸음 0인 3문항 모두 기준 파일 대비 $\max|\Delta_{\text{star}}| = 0.0$, $\max|\Delta_{\text{lowest}}| = 0.0$, $\max|\Delta_{\rho\text{-traj}}| = 0.0$의 비트 단위 일치를 실측했다 (`task0054_out.txt:100-102, 107-109, 114-116`).
   - 경계 걸음 0인 앵커 3개 확보(3/3)로 추가 앵커 탐색 없이 §5 점검 기준을 충족했다 (`task0054_out.txt:118-121`).
   - 사후 기준 파일 SHA-256: `5103642d...` 불변 보존 계측 (`task0054_out.txt:122`).
4. **본 실행(eps=0.005) 순항 계측 (수치 0 엄수)**:
   - 기동 시각: 2026-10-06 21:23:30 KST (`task0054_out.txt:124`).
   - 기동 20분 후(21:39:09 KST) 기준 11개 문항 연산 누적 진척 실측 (`task0054_out.txt:128-139`).
   - 중간 체크포인트 파일 생성: `results_s7/s7deq_soft_v2_eps0.005.pt` (13,386,157 B) (`task0054_out.txt:188`).
5. **자원 분리 및 거버넌스 엄수**:
   - GPU 1: 99 % 점유율, 30,776 MiB 점유 (PID 276795) (`task0054_out.txt:29, 39, 45`).
   - GPU 0: 99 % 점유율, 15,846 MiB 점유 (PID 245681, `p6_chain` Condition C, 578행 순항 생성 중) (`task0054_out.txt:28, 38, 43, 210`).
   - GPU 2~7: 0 % 점유율, 16 MiB (GPU 4 영구 봉인 유지, 2/3/5/6/7 미사용) (`task0054_out.txt:30-35`).
   - 서버 내 대기: R7-4 규약에 따라 단일 연결 내 20분(1,200초) 대기 후 1회 일괄 수집 마침 (`task0054_out.txt:14-20`).

---

## 2. §5 사전 앵커 점검 결과 상세

### 2.1 기동 환경 및 메타데이터 실측

- **해석기 환경**: `/home/shoon/system1_5/.venv/bin/python 2.11.0+cu128 12.8` (`task0054_out.txt:53`).
- **74문항 설정 SHA-256**: `42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3` (`task0054_out.txt:54`).
- **점검 전 기준 파일 SHA-256**: `5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6` (`task0054_out.txt:61`).
- **기동 전 GPU 1 상태**: 점유율 0 %, 메모리 16 MiB, 온도 29 ℃ (`task0054_out.txt:59`).

### 2.2 앵커 3종 평가 실측치 (인용)

| 앵커 번호 | `boundary_step_count` | `eps` | $\max|\Delta_{\text{star}}|$ | $\max|\Delta_{\text{lowest}}|$ | $\max|\Delta_{\rho\text{-traj}}|$ | 판정 및 원시 로그 포인터 |
|---|---|---|---|---|---|---|
| **Q#20** | 0 | 1e-12 | 0.0 | 0.0 | 0.0 | 비트 일치 (`task0054_out.txt:98-102`) |
| **Q#24** | 0 | 1e-12 | 0.0 | 0.0 | 0.0 | 비트 일치 (`task0054_out.txt:105-109`) |
| **Q#32** | 0 | 1e-12 | 0.0 | 0.0 | 0.0 | 비트 일치 (`task0054_out.txt:112-116`) |

- **실측 로그 인용 (`task0054_out.txt:97-122`)**:
  ```text
  --- Testing Anchor Q#20 on GPU 1 (eps=1e-12) ---
  Anchor Q#20: boundary_step_count = 0, eps = 1e-12
  Anchor Q#20 (boundary=0):
  max |Delta_star|     = 0.0
  max |Delta_lowest|   = 0.0
  max |Delta_rho_traj| = 0.0

  --- Testing Anchor Q#24 on GPU 1 (eps=1e-12) ---
  Anchor Q#24: boundary_step_count = 0, eps = 1e-12
  Anchor Q#24 (boundary=0):
  max |Delta_star|     = 0.0
  max |Delta_lowest|   = 0.0
  max |Delta_rho_traj| = 0.0

  --- Testing Anchor Q#32 on GPU 1 (eps=1e-12) ---
  Anchor Q#32: boundary_step_count = 0, eps = 1e-12
  Anchor Q#32 (boundary=0):
  max |Delta_star|     = 0.0
  max |Delta_lowest|   = 0.0
  max |Delta_rho_traj| = 0.0

  Total anchors evaluated: 3
  Zero boundary anchors found (3/3): [20, 24, 32]

  §5 Anchor Check passed successfully.
  Reference file SHA post-check: 5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6
  ```

- **점검 결론**:
  - 평가 대상 3문항 전체가 경계 걸음 0을 기록하여 요구 수량(3개)을 1회차에 충족했다 (`task0054_out.txt:118-119`).
  - 정본 실행기 기준 파일과의 차이 3종($z^*$, $z_{\text{lowest}}$, $\rho_{\text{traj}}$)이 전 문항에서 정밀하게 0.0을 기록했다 (`task0054_out.txt:100-102, 107-109, 114-116`).
  - 점검 종료 직후 기준 파일 SHA-256이 `5103642d...`로 변조 없이 불변 보존됨을 실측했다 (`task0054_out.txt:122`).
  - 점검 산출 파일 생성 계측: `results_s7/anchor_soft_eps1e-12_check.pt` (4,017,785 B, MTIME 21:23:28 KST) (`task0054_out.txt:153`).

---

## 3. 본 실험 (eps=0.005) 진행 현황 계측 (수치 0 엄수)

본 실험의 수치 유출을 방지하기 위해 문항 인덱스 및 문항 식별자만 기록했다 (잔차·손실값 수치 기록 0건 엄수).

- **본 실행 진입 시각**: `=== RUNNING EPS=0.005 AT 2026-10-06 21:23:30 KST ===` (`task0054_out.txt:124`).
- **진행 문항 인덱스 (기동 후 20분 시점)**: 총 11문항 진척 (`task0054_out.txt:139`).
  - `[0001/0074] q0031_a0.25` (`task0054_out.txt:128`)
  - `[0002/0074] q0127_a0.25` (`task0054_out.txt:129`)
  - `[0003/0074] q0138_a0.25` (`task0054_out.txt:130`)
  - `[0004/0074] q0249_a0.25` (`task0054_out.txt:131`)
  - `[0005/0074] q0279_a0.25` (`task0054_out.txt:132`)
  - `[0006/0074] q0395_a0.25` (`task0054_out.txt:133`)
  - `[0007/0074] q0424_a0.25` (`task0054_out.txt:134`)
  - `[0008/0074] q0441_a0.25` (`task0054_out.txt:135`)
  - `[0009/0074] q0481_a0.25` (`task0054_out.txt:136`)
  - `[0010/0074] q0619_a0.25` (`task0054_out.txt:137`)
  - `[0011/0074] q0622_a0.25` (`task0054_out.txt:138`)
- **출력 텐서 주기적 저장 실측**:
  - `results_s7/s7deq_soft_v2_eps0.005.pt` 크기: **13,386,157 B** (MTIME 21:37:06 KST, 10번째 문항 연산 시점 저장) (`task0054_out.txt:188`).

---

## 4. 서버 텔레메트리 및 자원 격리 실측

2026-10-06 21:39:09 KST 단일 SSH 묶음 세션 실측치:

### 4.1 세션 및 GPU 텔레메트리

- **활성 tmux 세션 (`task0054_out.txt:24-25`)**:
  - `p6_chain`: 1 windows (created Mon Oct 5 19:54:38 2026)
  - `s7deq_soft_v21`: 1 windows (created Tue Oct 6 21:19:06 2026)
- **GPU 0~7 실측치 (`task0054_out.txt:27-35`)**:

| GPU 인덱스 | 물리 UUID | 점유율 [%] | 메모리 점유 [MiB] | 온도 [℃] | 전력 [W] | 할당 작업 및 비고 |
|---|---|---|---|---|---|---|
| **GPU 0** | `GPU-71be98b0...` | 99 % | 15,846 MiB | 54 ℃ | 134.01 W | PID 245681 (`p6_chain` Condition C) |
| **GPU 1** | `GPU-9410691f...` | 99 % | 30,776 MiB | 55 ℃ | 117.88 W | PID 276795 (`s7deq_soft_v21` eps=0.005) |
| **GPU 2** | `GPU-c81311d8...` | 0 % | 16 MiB | 31 ℃ | 21.75 W | 유휴 보존 |
| **GPU 3** | `GPU-a9fed959...` | 0 % | 16 MiB | 33 ℃ | 17.87 W | 유휴 보존 |
| **GPU 4** | `GPU-e27bdb0d...` | 0 % | 16 MiB | 27 ℃ | 11.82 W | **영구 봉인** |
| **GPU 5** | `GPU-84d7eeb0...` | 0 % | 16 MiB | 28 ℃ | 6.81 W | 유휴 보존 |
| **GPU 6** | `GPU-f99bbac4...` | 0 % | 16 MiB | 28 ℃ | 6.33 W | 유휴 보존 |
| **GPU 7** | `GPU-0eb58a21...` | 0 % | 16 MiB | 29 ℃ | 8.36 W | 유휴 보존 |

### 4.2 실행 프로세스 상세 (`task0054_out.txt:38-45`)

- **GPU 0 프로세스**:
  - PID: `245681` (`pts/2`, CPU 100%, VSZ 64.5GB, RSS 2.0GB, 누적 시간 197:42)
  - 명령줄: `/home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond C`
- **GPU 1 프로세스**:
  - 부모 PID: `276023` (`pts/0`, `bash scripts/s7deq_soft_v2_launch_v5.sh`)
  - 자식 PID: `276795` (`pts/0`, CPU 104%, VSZ 42.8GB, RSS 1.9GB, 누적 시간 16:21)
  - 명령줄: `/home/shoon/system1_5/.venv/bin/python scripts/s7deq_soft_v2_run_v2.py --items configs/s6_items_74.json --out results_s7/s7deq_soft_v2_eps0.005.pt --eps 0.005 --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --tol 0.0 --max_nfe 1000 --device cuda:0 --config configs/zstar_canonical.yaml`
- **상태 판정**: GPU 0과 GPU 1의 프로세스 및 VRAM 자원이 완전 분리되어 상호 간섭 0건으로 병렬 실행 중이다.

### 4.3 P6_CHAIN 진척 상황 (`task0054_out.txt:203-214`)

- **P6 로그 마지막 3행**:
  ```text
  [P6-CHAIN] Stage 1: Launching z_src(600) calculation...
  [P6-CHAIN] Stage 1 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/ZSRC600_DONE.flag.
  [P6-CHAIN] Running Generation Condition C (shift=)...
  ```
- **P6 결과 파일 행 수**:
  - `results_s7/p6_gen_C.jsonl`: **578행** (771,119 B, MTIME 21:38:53 KST) (`task0054_out.txt:174, 210`). 직전 계측(21:12 498행) 대비 80행 순항 생성 중. 내용 열람 0건 유지.

---

## 5. 거버넌스 및 규칙 준수 점검

1. **GPU 사용 정책 (CTS-428)**:
   - GPU 1: 허락된 부드러운 스위치 실행기 단독 가동.
   - GPU 0: `p6_chain` Condition C 가동 (무간섭).
   - GPU 4: 영구 봉인 유지 (점유율 0%, 16 MiB).
   - GPU 2·3·5·6·7: 사용 0건 엄수.
2. **서버 접속 예절 규약 v1 (원장 #399, R7-1~R7-9)**:
   - 모든 연결은 `shared/cts_ssh.py` 경유 단독 수행 (`shared/ssh_log.csv` 기록).
   - 기동 및 20분 대기를 단일 SSH 연결(`task0054_launch_and_collect_20min`) 안에서 처리하여 중간 다중 폴링 접속을 원천 배제했다 (R7-3, R7-4 엄수).
   - 최근 1시간 누적 연결 수: 10회 ($\le 20$회 상한 충족).
3. **규칙 R3-5 준수**:
   - 9대 금지 규정(R3-5 9종 단어) 사용 0건 엄수.
   - 모든 수치에 원시 출력 포인터(`task0054_out.txt` 행 번호) 명시.
   - 본 실험의 수치(잔차, 손실 등) 유출 0건 (문항 인덱스 및 키만 수록).
4. **줄바꿈 및 무결성**:
   - 본 보고서 CR(0x0D) 바이트 0건 (LF 전용).
   - 본 보고서 1행 LF self-hash 탑재.

---

## 6. 결론 및 후속 절차

- PREREG_S7_DEQ_v2.1 §5 사전 앵커 점검 요건(Q#20, Q#24, Q#32 전원 경계 걸음 0 및 $\max|\Delta|=0.0$)이 전수 충족되었으며, 본 실험(eps=0.005)이 GPU 1에서 11문항 진척으로 순항 가동 중임을 실측했다.
- 본 보고서 발행 후 `shared/STATE.md`를 갱신하고, 후속 모니터링 지침 대기 명령을 기동한다.

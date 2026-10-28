<!-- SHA256: 5a76e6b5a4ec35b8d453acc5aba55e04c00bfadce90a97697551c6ce09bd7945 -->

# REPORT-0048: PREREG_S7_DEQ_v2 실행 시작 및 15분 경과 계측 보고서 (GPU 1)

- **작성 일시**: 2026-10-06 19:16 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, `D:\AI\cts`)
- **근거 태스크**: [TASK-0048.md](file:///d:/AI/cts/shared/tasks/TASK-0048.md)
- **근거 감사 보고서**: [AUDIT-0047.md](file:///d:/AI/cts/shared/audits/AUDIT-0047.md) (PASS)
- **근거 사전등록**: [PREREG_S7_DEQ_v2.md](file:///d:/AI/cts/shared/prereg/PREREG_S7_DEQ_v2.md) (잠금, SHA-256 `d9eef10180d228bb676f5115d97fc721f3acf67d4f94864f8ef33b018a5a72aa`) §5·§6
- **상위 지침**: CTS-428 Task 3 (`shared/inbox/CTS-428.md`)
- **절대 조건 준수 현황**:
  - 장치 격리: 물리 GPU 1 (`GPU-9410691f-6857-1642-ec7c-7596970e2f48`) 단독 할당 (엄수)
  - `p6_chain` 세션 무간섭: 준수 (GPU 0에서 Condition C 생성 무간섭 가동 유지 중, kill/attach/수정 0건)
  - 타 GPU 보호: GPU 2·3·5·6·7 사용 0건, GPU 4 영구 봉인 유지
  - 기존 파일 보존: 신규 산출물 격리 (`results_s7/` 신규 파일로 저장)
  - 서버 접속 규약 (R7): `shared/cts_ssh.py` 경유 단일 SSH 세션 내 15분 대기 (R7-4 폴링 서버 내 수행)
  - 금지어 규칙: 9대 금지어 사용 0건
  - 수치 포인터 표기: 수치마다 로그 행 번호 및 실측치 명시

---

## 1. 실행 전 SHA 및 결과 파일 부재 점검 (§1)

| 파일 경로 | 기대 SHA-256 | 실측 SHA-256 | 점검 결과 |
|---|---|---|:---:|
| `scripts/s7deq_soft_v2_launch_v2.sh` | `cb3ad22f61b9604da92414e3bedbf9b7c38bc387b316a59c5dd56fe6cb7c2784` | `cb3ad22f61b9604da92414e3bedbf9b7c38bc387b316a59c5dd56fe6cb7c2784` | 일치 |
| `scripts/s7deq_soft_v2_run_v2.py` | `2687a54e28b73c15e2980ad204724fb4f8903fb7d3b8f3d33d4e3ae11283db17` | `2687a54e28b73c15e2980ad204724fb4f8903fb7d3b8f3d33d4e3ae11283db17` | 일치 |
| `exe_scratch/s7/soft_v2/zstar_soft_lowest.py` | `d1942c3cf5defbad6c148e69102b1ca6e8c64d1da5dd731434782364fd689f07` | `d1942c3cf5defbad6c148e69102b1ca6e8c64d1da5dd731434782364fd689f07` | 일치 |
| `scripts/s7deq_soft_v2_analyze.py` | `07d63b1e9fdce284d93216cc00586bb2f4e6223a7aee13007b650dc94ec73aec` | `07d63b1e9fdce284d93216cc00586bb2f4e6223a7aee13007b650dc94ec73aec` | 일치 |
| `configs/s6_items_74.json` | `42df1f49...` | `42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3` | 일치 |

- **사전 존재 금지 파일 8종 점검 결과 (`scratch/task0048_out.txt` 11~19행)**:
```text
Checking conflicting output files in results_s7/...
[CLEAN] Target file absent: results_s7/anchor_ref_gpu1_check.pt
[CLEAN] Target file absent: results_s7/anchor_soft_eps0.02_check.pt
[CLEAN] Target file absent: results_s7/s7deq_soft_v2_eps0.005.pt
[CLEAN] Target file absent: results_s7/s7deq_soft_v2_eps0.01.pt
[CLEAN] Target file absent: results_s7/s7deq_soft_v2_eps0.02.pt
[CLEAN] Target file absent: results_s7/s7deq_soft_v2_eps0.005.json
[CLEAN] Target file absent: results_s7/s7deq_soft_v2_eps0.01.json
[CLEAN] Target file absent: results_s7/s7deq_soft_v2_eps0.02.json
```
- **사전 점검 종합**: 충돌 파일 0건, 사전 점검 요건 전수 충족.

---

## 2. tmux 세션 기동 및 실행 명령 (§2)

- **기동 시각**: 2026-10-06 19:00:26 KST
- **실행 명령**: `tmux new-session -d -s s7deq_soft_v2 'bash scripts/s7deq_soft_v2_launch_v2.sh'`
- **초기 tmux 세션 현황 (`scratch/task0048_out.txt` 25~26행)**:
```text
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
s7deq_soft_v2: 1 windows (created Tue Oct  6 19:00:26 2026)
```

---

## 3. 15분 경과 후 텔레메트리 계측 (§3)

### 3.1 tmux 세션 상태
```text
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
s7deq_soft_v2: 1 windows (created Tue Oct  6 19:00:26 2026)
```

### 3.2 §5 앵커 사전 점검 절 전문 (`raw_s7/s7deq_soft_v2.log`)
```text
=== §5 PRE-LAUNCH ANCHOR CHECK (GPU 1, F2~F5) ===
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
[0001/0001] q0020_a0.25: r_1000=1.016980e-06, r_lowest=1.016980e-06 at k*=1000, top3_final=[13, 15, 18], top3_lowest=[13, 15, 18] (55.46s)
================================================================================
Finished. Total items saved: 1 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
[0001/0001] q0020_a0.25: r_1000=6.574811e-07, r_lowest=6.574811e-07 at k*=1000, r_hard_lowest=6.167322e-03, b_steps=852/1000 (85.2%), in_b=True (106.78s)
================================================================================
Finished. Total items saved: 1 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 1 items.
[0001/0001] q0024_a0.25: r_1000=2.328545e-05, r_lowest=2.328545e-05 at k*=1000, top3_final=[2, 12, 16], top3_lowest=[2, 12, 16] (56.43s)
================================================================================
Finished. Total items saved: 2 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 1 items.
[0001/0001] q0024_a0.25: r_1000=8.285183e-06, r_lowest=8.285183e-06 at k*=1000, r_hard_lowest=5.770132e-03, b_steps=824/1000 (82.4%), in_b=True (104.87s)
================================================================================
Finished. Total items saved: 2 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 2 items.
[0001/0001] q0032_a0.25: r_1000=1.254489e-04, r_lowest=1.254489e-04 at k*=1000, top3_final=[2, 7, 16], top3_lowest=[2, 7, 16] (56.56s)
================================================================================
Finished. Total items saved: 3 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 2 items.
[0001/0001] q0032_a0.25: r_1000=7.183855e-04, r_lowest=5.893705e-04 at k*=710, r_hard_lowest=5.893705e-04, b_steps=27/1000 (2.7%), in_b=False (58.39s)
================================================================================
Finished. Total items saved: 3 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 3 items.
[0001/0001] q0000_a0.25: r_1000=1.500415e-04, r_lowest=1.500415e-04 at k*=1000, top3_final=[2, 12, 16], top3_lowest=[2, 12, 16] (79.42s)
================================================================================
Finished. Total items saved: 4 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 3 items.
[0001/0001] q0000_a0.25: r_1000=2.306190e-04, r_lowest=2.306190e-04 at k*=1000, r_hard_lowest=2.306190e-04, b_steps=20/1000 (2.0%), in_b=False (81.38s)
================================================================================
Finished. Total items saved: 4 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 4 items.
```

### 3.3 본 실행 (EPS=0.005) 진행 현황
- **규정 준수**: TASK-0048 지침에 따라 본 실행 문항별 잔차 수치는 인용하지 않고, 실행 헤더 및 진행 행 번호만 기재합니다.
- **실측 상태**: 15분 경과 시점에 §5 앵커 사전 점검(F2 동적 탐색 루프)이 연산 중이므로 본 실행(`EPS=0.005`)은 아직 진입 전입니다.
```text
EPS=0.005 not reached
Progress item lines (line numbers and item count only, r-values omitted):
75:[0001/0002] q
76:[0002/0002] q
102:[0001/0001] q
112:[0001/0001] q
123:[0001/0001] q
134:[0001/0001] q
145:[0001/0001] q
156:[0001/0001] q
167:[0001/0001] q
178:[0001/0001] q
```

### 3.4 GPU별 자원 분리 및 프로세스 대조표

| 항목 | GPU 0 (P6 Condition C) | GPU 1 (S7 Soft DEQ 본 실행) |
|---|:---:|:---:|
| **물리 UUID** | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` |
| **tmux 세션** | `p6_chain` | `s7deq_soft_v2` |
| **연산 프로세스 PID** | `245681` (`s7_eval_gate_p6_v36b.py`) | `255367` (`s7_zstar_lowest_run.py`) |
| **GPU 부하 / 메모리** | 99 % / 15,818 MiB (앱 15,798 MiB) | 99 % / 30,784 MiB (앱 30,764 MiB) |
| **온도 / 전력** | 54 ℃ / 135.74 W | 50 ℃ / 118.65 W |
| **격리 준수 여부** | 완전 분리 | 완전 분리 |

- **nvidia-smi GPU 현황 실측**:
```text
index, uuid, utilization.gpu [%], memory.used [MiB], temperature.gpu, power.draw [W]
0, GPU-71be98b0-6b81-9257-c622-97dc97d46480, 99 %, 15818 MiB, 54, 135.74 W
1, GPU-9410691f-6857-1642-ec7c-7596970e2f48, 99 %, 30784 MiB, 50, 118.65 W
2, GPU-c81311d8-d210-3857-843e-bb38d24771e1, 0 %, 16 MiB, 30, 21.67 W
3, GPU-a9fed959-7113-e747-dac4-a515ac50c160, 0 %, 16 MiB, 32, 18.46 W
4, GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5, 0 %, 16 MiB, 27, 11.84 W
5, GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a, 0 %, 16 MiB, 27, 6.95 W
6, GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c, 0 %, 16 MiB, 28, 6.41 W
7, GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b, 0 %, 16 MiB, 28, 8.48 W
```
- **연산 앱 및 PID 매핑 실측**:
```text
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 245681, 15798 MiB
GPU-9410691f-6857-1642-ec7c-7596970e2f48, 255367, 30764 MiB
```
- **프로세스 상세 실측**:
```text
245681 Tue Oct  6 18:21:47 2026 /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond C
 255367 Tue Oct  6 19:14:31 2026 /home/shoon/system1_5/.venv/bin/python scripts/s7_zstar_lowest_run.py --items 4 --out results_s7/anchor_ref_gpu1_check.pt --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --tol 0.0 --max_nfe 1000 --device cuda:0 --config configs/zstar_canonical.yaml
```
- **GPU 2·3·5·6·7 및 GPU 4 상태**: 사용 0건, GPU 4 봉인 유지.

### 3.5 p6_chain 무간섭 실측 (마지막 3행)
```text
[P6-CHAIN] Stage 1: Launching z_src(600) calculation...
[P6-CHAIN] Stage 1 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/ZSRC600_DONE.flag.
[P6-CHAIN] Running Generation Condition C (shift=)...
```

---

## 4. 요약 및 후속 조치

1. **사전 등록 규약 엄수**:
   - `cb3ad22f...` launch 스크립트 기반으로 GPU 1에서 정규 실행 기동.
   - §5 앵커 사전 점검: 초기 앵커 3문항(Q#20, 24, 32) 및 추가 앵커(Q#0)의 경계 걸음 수가 0이 아니므로(각각 852, 824, 27, 20), F2 규정에 따라 다음 앵커(Q#4) 동적 탐색 연산 진행 중 (PID 255367).
2. **자원 격리 및 무간섭**:
   - GPU 0 (`p6_chain`, PID 245681)에 어떠한 간섭도 없이 GPU 1 단독으로 본 연산이 순항 중임을 계측.
   - GPU 0과 GPU 1의 PID 및 메모리가 완전히 분리되어 가동 중임을 계측.
3. **후속 대기**:
   - 앵커 점검 종료 및 eps 3종 (0.005, 0.01, 0.02) 연산 산출물 생성을 기다리며 대기 명령(`python shared/wait_for_task.py 49`) 기동.

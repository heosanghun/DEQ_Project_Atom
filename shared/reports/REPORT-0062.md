<!-- SHA256: c60d2c60a6240951fd42902985229482c20ff45061df176bd074c83c5f585718 -->

# REPORT-0062: ①-a 재계산 종결 및 분석 산출물 수집

- **발행 일시**: 2026-10-07 11:20 KST
- **수행자**: 회사 PC 정식 루프 (CTS 실행 에이전트)
- **근거 문서**: TASK-0062, PREREG_S7_DEQ_v1.4(04112522…) §10(h), AUDIT-0057, REPORT-0061
- **규약 준수**: RULES R3-5 준수, 9대 금지어 0건, 수치마다 포인터 명시, 세션 무간섭(`p6_chain`), GPU 사용 0, 기존 파일 수정 0, 보고서 내 분석 수치 전사 0건(CC 원본 개봉 원칙), 수식·경로 원시 문자열(r"...") 사용 및 제어 문자 0건 점검(AUDIT-0060 위생 지침), CR 0건 (LF 전용), 11:15 이후 접속 지침 엄수

---

## 1. 수행 결과 총괄 요약

1. **호스트 상태 및 가동 시간**:
   - **가동 시간**: 부팅 시각 `2026-10-05 16:32:53` 불변 (1일 18시간 42분 연속 가동, 3차 재부팅 0건 실측, 포인터: `scratch/task0062_out.txt:16`).
   - **활성 세션**: `kstar_replay` 세션은 325문항 재계산 및 분석기 실행 마침 후 자동 종료되었으며, 현재 `p6_chain`(10-05 19:54 생성) 세션만 순항 가동 중 실측 (포인터: `scratch/task0062_out.txt:1, 17`).

2. **①-a 325문항 재계산 종결 실측 (TASK-0062 1·2항 충족)**:
   - **종결 시각 및 플래그**: `raw_s7/KSTAR_REPLAY_DONE.flag`가 `2026-10-07 10:55:21 KST`에 생성 마침 실측 (포인터: `scratch/task0062_out.txt:20`).
   - **재계산 출력 텐서 계측**: `results_s7/zstar_s7_a0.25_train325_kstar_replay.pt` 크기 429,137,915 B, SHA-256 `1446340e73e58bc35b5703ba6f6309da6ccb60fadf9cfbd5abc89a5dece27977`, CPU 적재 기준 325개 키 보존 실측 (포인터: `scratch/task0062_out.txt:21-23, 68-69`).
   - **분석기 Step 3 가동 및 종결**: `scripts/p5_promote_1a_replay_v2.py`가 10:55:21 KST 이후 자동 구동되어 분석 JSON 및 Markdown 산출물을 생성하고 10:55:25 KST에 전 과정 마침 실측 (포인터: `scratch/task0062_out.txt:56-65`).
   - **분석 수치 비인용 원칙 엄수**: TASK-0062 절대 조건에 따라 분석기 결과 행(`raw_s7/kstar_replay_gpu1.log` 425~426행)의 구체적인 수치는 보고서에 전사하지 않고 해당 행 번호만 포인터로 인용함 (CC 원본 개봉 원칙).

3. **SFTP 수취 및 로컬 무결성 대조**:
   - `shared/cts_ssh.py`를 통해 규정 쿨다운(74초 경과 후) 준수 하에 단일 SFTP 배치로 분석 산출물 4종을 `shared/reports/kstar_replay/` 경로에 수신 마침 (`shared/ssh_log.csv:168`).
   - `p5_promote_1a_replay_v2.json` (3,360 B), `p5_promote_1a_replay_v2.md` (3,455 B), `KSTAR_REPLAY_DONE.flag` (201 B), `kstar_replay_gpu1.log` (53,378 B) 전원 서버 실측 SHA-256과 비트 단위 일치 계측.

4. **P6 체인 보조 진행 현황**:
   - **진행 단계**: Condition D1(600행 마침) 및 D2(shift=1, 600행 마침) 종결 후, 마지막 조건인 Condition E2(shift=1) 371행 생성 진행 중 실측 (포인터: `scratch/task0062_out.txt:74-80, 92`).
   - **생성물 행 수 인벤토리**: p6_gen_*.jsonl 누적 4,571행 생성 중. 신규 게이트 표 생성 0건 (내용 미열람 엄수, 포인터: `scratch/task0062_out.txt:81-99`).

5. **자원 텔레메트리**:
   - GPU 0: 99% 사용률, 16,134 MiB, PID 403337 (P6 Cond E2 가동 중).
   - GPU 1: **0% 사용률, 16 MiB 유휴** (재계산 프로세스 PID 346655 마침 후 VRAM 완전 반환 실측, 포인터: `scratch/task0062_out.txt:4`).
   - GPU 2~7: 0%, 16 MiB 유휴. GPU 4 영구 봉인 유지 (포인터: `scratch/task0062_out.txt:5-12`).

---

## 2. 1. 시스템 및 재계산 세션 상태 상세

### 2.1. Uptime 및 활성 세션
- 포인터: `scratch/task0062_out.txt:1, 14-17`
```text
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
Timestamp: 2026-10-07 11:15:27 KST
Uptime: 2026-10-05 16:32:53
```
- 실측 분석: `kstar_replay` tmux 세션은 모든 연산과 분석기 스크립트를 마친 뒤 자동으로 세션을 닫고 종료되었으며, `p6_chain` 세션만 단독 실행 중입니다.

### 2.2. `raw_s7/KSTAR_REPLAY_DONE.flag` 전문 인용
- 포인터: `scratch/task0062_out.txt:19-23`
```text
DONE_AT: 2026-10-07 10:55:21 KST
OUTPUT_FILE: results_s7/zstar_s7_a0.25_train325_kstar_replay.pt
OUTPUT_SHA256: 1446340e73e58bc35b5703ba6f6309da6ccb60fadf9cfbd5abc89a5dece27977
OUTPUT_BYTES: 429137915
```

### 2.3. `raw_s7/kstar_replay_gpu1.log` 말미 40행 중 분석기 실행 및 마침 블록
- 포인터: `scratch/task0062_out.txt:51-65`
```text
Line 00416: ================================================================================
Line 00417: Finished. Total items saved: 325 to results_s7/zstar_s7_a0.25_train325_kstar_replay.pt
Line 00418: ================================================================================
Line 00419: Created raw_s7/KSTAR_REPLAY_DONE.flag (SHA: 1446340e73e58bc35b5703ba6f6309da6ccb60fadf9cfbd5abc89a5dece27977, bytes: 429137915)
Line 00420: ================================================================================
Line 00421: === Step 3: Running Analyzer v2 ===
Line 00422: ================================================================================
Line 00423: Analysis JSON written to results_s7/p5_promote_1a_replay_v2.json
Line 00424: Analysis Markdown written to results_s7/p5_promote_1a_replay_v2.md
Line 00425: [비인용] ①-a Result 행 (포인터: raw_s7/kstar_replay_gpu1.log:425, 수치 전사 0건)
Line 00426: [비인용] Threshold 행 (포인터: raw_s7/kstar_replay_gpu1.log:426, 수치 전사 0건)
Line 00427: ================================================================================
Line 00428: === ①-a k* REPLAY COMPLETED ON GPU 1 ===
Line 00429: Timestamp: 2026-10-07 10:55:25 KST
Line 00430: ================================================================================
```
- 오류 행 점검: 0건. 에러 및 Traceback 일체 없이 종료 코드 0으로 마침 실측.

---

## 3. 2. 산출물 계측 및 SFTP 수신 대조

TASK-0062 2항에 따라 재계산 출력 텐서 및 분석기 산출물 4종의 제원을 계측하고 SFTP로 수신한 로컬 파일과 서버 파일의 무결성을 비트 단위로 대조하였습니다.

### 3.1. 서버 산출물 실측 제원
- 포인터: `scratch/task0062_out.txt:67-71`

| 구분 | 파일 경로 | 크기 (B) | SHA-256 | 키 수 / 비고 |
|:---|:---|:---:|:---|:---:|
| 재계산 출력 텐서 | `results_s7/zstar_s7_a0.25_train325_kstar_replay.pt` | 429,137,915 | `1446340e73e58bc35b5703ba6f6309da6ccb60fadf9cfbd5abc89a5dece27977` | 325개 키 보존 |
| 분석기 산출 JSON | `results_s7/p5_promote_1a_replay_v2.json` | 3,360 | `c0a7136218c96a76c354506aa54723b1003ed59a7145bd42f01aec24634dab77` | 내용 미열람 |
| 분석기 산출 Markdown | `results_s7/p5_promote_1a_replay_v2.md` | 3,455 | `1c8429c28fb7f32036ed17357cd0fba3f4495ccf126e8d6e969ba79ae962ad64` | 내용 미열람 |
| 마침 플래그 | `raw_s7/KSTAR_REPLAY_DONE.flag` | 201 | `0c0ab08c56f38a6c78c48f8d3551245eb15e92dc8de9bde89aed51ef6d31072b` | 플래그 전문 |
| 재계산 로그 전수 | `raw_s7/kstar_replay_gpu1.log` | 53,378 | `5fe7779aff97e168b267463461cdc0b49a8f143b8074c6b6b7afb28b2c02bd52` | 전체 430행 |

### 3.2. SFTP 로컬 수령 파일 무결성 대조
- 로컬 저장 디렉터리: `shared/reports/kstar_replay/`
- SFTP 실행 일시: 2026-10-07 11:16:32 KST (`shared/ssh_log.csv:168`, exit 0)

| 파일명 | 로컬 크기 (B) | 로컬 SHA-256 | 서버 SHA-256 일치 여부 |
|:---|:---:|:---|:---:|
| `p5_promote_1a_replay_v2.json` | 3,360 | `c0a7136218c96a76c354506aa54723b1003ed59a7145bd42f01aec24634dab77` | 일치 (BIT_EXACT) |
| `p5_promote_1a_replay_v2.md` | 3,455 | `1c8429c28fb7f32036ed17357cd0fba3f4495ccf126e8d6e969ba79ae962ad64` | 일치 (BIT_EXACT) |
| `KSTAR_REPLAY_DONE.flag` | 201 | `0c0ab08c56f38a6c78c48f8d3551245eb15e92dc8de9bde89aed51ef6d31072b` | 일치 (BIT_EXACT) |
| `kstar_replay_gpu1.log` | 53,378 | `5fe7779aff97e168b267463461cdc0b49a8f143b8074c6b6b7afb28b2c02bd52` | 일치 (BIT_EXACT) |

*점검 결과*: 4종 파일 모두 로컬 실측 해시가 서버 실측 해시와 100% 비트 단위 일치함을 계측하였습니다.

---

## 4. 4. P6 체인 보조 진행 상세

### 4.1. `raw_s7/p6_chain.log` 마지막 6행 원문 인용
- 포인터: `scratch/task0062_out.txt:74-80`
```text
[P6-CHAIN] Generation Condition F (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_F_DONE.flag.
[P6-CHAIN] Running Generation Condition D1 (shift=)...
[P6-CHAIN] Generation Condition D1 (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_D1_DONE.flag.
[P6-CHAIN] Running Generation Condition D2 (shift=1)...
[P6-CHAIN] Generation Condition D2 (shift=1) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_D21_DONE.flag.
[P6-CHAIN] Running Generation Condition E2 (shift=1)...
```

### 4.2. P6 생성 파일(`results_s7/p6_gen_*.jsonl`) 행 수 현황
- 포인터: `scratch/task0062_out.txt:81-94`

| 파일명 | 행 수 | 비고 |
|:---|:---:|:---|
| `results_s7/p6_gen_A.jsonl` | 600 | 마침 |
| `results_s7/p6_gen_A_smoke.jsonl` | 2 | 스모크 |
| `results_s7/p6_gen_B1.jsonl` | 600 | 마침 |
| `results_s7/p6_gen_B1_smoke.jsonl` | 2 | 스모크 |
| `results_s7/p6_gen_C.jsonl` | 600 | 마침 |
| `results_s7/p6_gen_C_smoke.jsonl` | 2 | 스모크 |
| `results_s7/p6_gen_D1.jsonl` | 600 | 마침 (`P6_GEN_D1_DONE.flag`) |
| `results_s7/p6_gen_D21.jsonl` | 600 | 마침 (`P6_GEN_D21_DONE.flag`) |
| `results_s7/p6_gen_E1.jsonl` | 600 | 마침 |
| `results_s7/p6_gen_E1_smoke.jsonl` | 2 | 스모크 |
| `results_s7/p6_gen_E21.jsonl` | **371** | 생성 진행 중 (직전 252행 대비 +119행 진척) |
| `results_s7/p6_gen_F.jsonl` | 600 | 마침 |
| `results_s7/p6_gen_F_smoke.jsonl` | 2 | 스모크 |

### 4.3. P6 게이트 표 파일 목록
- 포인터: `scratch/task0062_out.txt:95-99`
- `results_s7/p6_gate_table_main.json` (3,515 B)
- `results_s7/p6_gate_table_main.md` (1,182 B)
- `results_s7/p6_gate_table_smoke.json` (3,821 B)
- `results_s7/p6_gate_table_smoke.md` (1,315 B)
- 비고: 신규 게이트 표 부재 (Stage 3 4종 보존, 내용 미열람 엄수).

---

## 5. GPU 0~7 실시간 텔레메트리

- 데이터 출처 포인터: `scratch/task0062_out.txt:2-12`

| GPU 인덱스 | UUID | 사용률 (%) | 메모리 점유 (MiB) | 온도 (℃) | 전력 소비 (W) | 실행 프로세스 (PID, 용도) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0` | 99 % | 16,134 MiB | 51 ℃ | 132.31 W | PID 403337 (P6 조건 E2 생성 연산 가동 중) |
| **1** | `GPU-9410691f` | **0 %** | **16 MiB** | **29 ℃** | **12.22 W** | **프로세스 없음 (재계산 종결 후 유휴 상태 진입)** |
| **2** | `GPU-c81311d8` | 0 % | 16 MiB | 31 ℃ | 21.71 W | 프로세스 없음 (유휴) |
| **3** | `GPU-a9fed959` | 0 % | 16 MiB | 32 ℃ | 17.94 W | 프로세스 없음 (유휴) |
| **4** | `GPU-e27bdb0d` | 0 % | 16 MiB | 30 ℃ | 23.41 W | **영구 봉인** (프로세스 0건 엄수) |
| **5** | `GPU-84d7eeb0` | 0 % | 16 MiB | 31 ℃ | 16.51 W | 프로세스 없음 (유휴) |
| **6** | `GPU-f99bbac4` | 0 % | 16 MiB | 31 ℃ | 15.79 W | 프로세스 없음 (유휴) |
| **7** | `GPU-0eb58a21` | 0 % | 16 MiB | 31 ℃ | 20.88 W | 프로세스 없음 (유휴) |

---

## 6. 로컬 산출물 인벤토리 및 무결성 계측

| 파일 구분 | 로컬 경로 | 크기 (B) | SHA-256 | 비고 |
|:---|:---|:---:|:---|:---|
| 수집 보고서 | `shared/reports/REPORT-0062.md` | 본문 참조 | 1행 참조 | 본 보고서 |
| 번들 실행 스크립트 | `shared/run_task0062_bundle.py` | 6,268 | `3eb111098511832df3fb46df12effb8a97622505c332c4fcdccc293b17ace72c` | 단일 SSH 묶음 점검기 |
| SFTP 배치 파일 | `scratch/sftp_task0062_batch.txt` | 454 | `b2c493c35b67b3f9a1a58ad22e5c3c1284394c364be806c37ece52fe77c6348e` | 4종 파일 수신 지시서 |
| 원시 표준출력 로그 | `scratch/task0062_out.txt` | 6,343 | `5a63cd05d672d0c795945edafad73e5602e5899cdc3ad03f1d79de925457dda6` | 원격 계측 원시 텍스트 |
| 수신 분석 JSON | `shared/reports/kstar_replay/p5_promote_1a_replay_v2.json` | 3,360 | `c0a7136218c96a76c354506aa54723b1003ed59a7145bd42f01aec24634dab77` | SFTP 수령 |
| 수신 분석 MD | `shared/reports/kstar_replay/p5_promote_1a_replay_v2.md` | 3,455 | `1c8429c28fb7f32036ed17357cd0fba3f4495ccf126e8d6e969ba79ae962ad64` | SFTP 수령 |
| 수신 종결 플래그 | `shared/reports/kstar_replay/KSTAR_REPLAY_DONE.flag` | 201 | `0c0ab08c56f38a6c78c48f8d3551245eb15e92dc8de9bde89aed51ef6d31072b` | SFTP 수령 |
| 수신 런 로그 | `shared/reports/kstar_replay/kstar_replay_gpu1.log` | 53,378 | `5fe7779aff97e168b267463461cdc0b49a8f143b8074c6b6b7afb28b2c02bd52` | SFTP 수령 |
| 접속 일지 | `shared/ssh_log.csv` | 9,733 | `b2d39dcf6fd38f71c71abaf1837de98890ff421d1a6f82029ea8e09db0c4f844` | 168행 (167: ssh, 168: sftp) |

---

## 7. 예상 밖 사건 및 서버 상태 1줄 요약

- **예상 밖 사건**: 없음. 11:15 이후 접속 지침 엄수, ①-a 재계산 10:55:21 KST 종결 및 분석기 산출물 4종 비트 단위 일치 수신 마침. 분석 수치 비인용 원칙 엄수. GPU 1 유휴 반환 실측. P6 E2 371행 순항 중.
- **서버 상태 1줄**: 접속 규약 준수 (R7-1~R7-9) | 호스트 1d 18h 가동 (3차 재부팅 0) | GPU 0 P6 Cond E2 99% (PID 403337) | GPU 1 유휴 0% 16MiB (kstar_replay 마침) | GPU 2~7 0% 유휴 (GPU 4 봉인).

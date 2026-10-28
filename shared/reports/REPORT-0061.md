<!-- SHA256: 55aa0cb564c322d39684f6258babb462633e457ec13e7a1d875c3966f570e642 -->

# REPORT-0061: ①-a 재계산 진척 및 P6 진행 점검

- **발행 일시**: 2026-10-07 10:45 KST
- **수행자**: 회사 PC 정식 루프 (CTS 실행 에이전트)
- **근거 문서**: TASK-0061, PREREG_S7_DEQ_v1.4(04112522…) §10(h), AUDIT-0057, AUDIT-0060
- **규약 준수**: RULES R3-5 준수, 9대 금지어 0건, 수치마다 포인터 명시, 세션 무간섭(`kstar_replay`, `p6_chain`), GPU 사용 0, 기존 파일 수정 0, 보고서 내 분석 수치 전사 0건(CC 원본 개봉 원칙), 수식·경로 원시 문자열(r"...") 사용 및 제어 문자 0건 검증(AUDIT-0060 위생 결함 조치), CR 0건 (LF 전용)

---

## 1. 수행 결과 총괄 요약

1. **호스트 상태 및 프로세스 무결성**:
   - **가동 시간**: 부팅 시각 `2026-10-05 16:32:53` 불변(1일 18시간 10분 연속 가동, 3차 재부팅 0건 실측, 포인터: `scratch/task0061_out.txt:18`).
   - **활성 세션**: `kstar_replay`(10-07 04:38 생성), `p6_chain`(10-05 19:54 생성) 2개 세션 격리 순항 실측 (포인터: `scratch/task0061_out.txt:1-2`).

2. **①-a 325문항 재계산 진행 현황 (TASK-0061 3항 적용)**:
   - **종료 플래그 상태**: `raw_s7/KSTAR_REPLAY_DONE.flag` 부재(NOT FOUND)로 현재 본 연산 진행 중임 (포인터: `scratch/task0061_out.txt:22`).
   - **진행 헤더**: 마지막 기록 헤더 `[0312/0325]` (325문항 중 312문항 진행 실측, 진척도 96.00%, 포인터: `scratch/task0061_out.txt:64`). 수치 누설 0건 엄수.
   - **중간 산출물 상태**: `results_s7/zstar_s7_a0.25_train325_kstar_replay.pt` 크기 409,331,601 B, SHA `e97f4b86...`, CPU 적재 기준 보존 키 수 310개 실측 (포인터: `scratch/task0061_out.txt:67-68`).
   - **분석기 산출물**: 재계산이 아직 진행 중이므로 `results_s7/p5_promote_1a_replay_v2.json` 및 `.md`는 미생성 상태임 (포인터: `scratch/task0061_out.txt:69-70`).
   - **잔여 예상**: 잔여 13문항이며, 문항당 평균 약 63.5초 기준 약 13~15분 후인 **10:55~11:00 KST** 종결 예상됨.

3. **P6 체인 보조 진행 현황**:
   - **진행 단계**: Condition D1(600행 마침, `P6_GEN_D1_DONE.flag`), Condition D2(shift=1, 600행 마침, `P6_GEN_D21_DONE.flag`) 순차 마침 후, 현재 Condition E2(shift=1) 252행 생성 진행 중 실측 (포인터: `scratch/task0061_out.txt:74-79, 91`).
   - **생성물 행 수 인벤토리**: p6_gen_*.jsonl 누적 4,452행(A 600, B1 600, C 600, D1 600, D2 600, E1 600, E2 252, F 600) 생성 중 (포인터: `scratch/task0061_out.txt:81-93`).
   - **게이트 표 현황**: 신규 게이트 표 생성 0건(기존 main/smoke 4종 보존, 내용 미열람, 포인터: `scratch/task0061_out.txt:95-98`).

4. **GPU 텔레메트리**:
   - GPU 0: 99% 사용률, 16,134 MiB, PID 403337 (P6 Cond E2 생성 연산 가동 중).
   - GPU 1: 98% 사용률, 31,234 MiB, PID 346655 (①-a 325문항 재계산 연산 가동 중).
   - GPU 2~7: 0%, 16 MiB 유휴. GPU 4 영구 봉인 엄수 (포인터: `scratch/task0061_out.txt:4-14`).

---

## 2. 1. 시스템 및 재계산 세션 상태 상세

### 2.1. Uptime 및 활성 세션
- 포인터: `scratch/task0061_out.txt:1-2, 17-19`
```text
kstar_replay: 1 windows (created Wed Oct  7 04:38:27 2026)
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
Timestamp: 2026-10-07 10:42:51 KST
Uptime: 2026-10-05 16:32:53
```

### 2.2. KSTAR_REPLAY_DONE.flag 점검
- 포인터: `scratch/task0061_out.txt:21-22`
```text
=== 2. KSTAR_REPLAY_DONE.flag ===
KSTAR_REPLAY_DONE.flag NOT FOUND
```
- 실측 결과: 종료 플래그 미생성으로 현재 2단계 재계산 프로세스가 연산 진행 중임을 실측함.

### 2.3. `raw_s7/kstar_replay_gpu1.log` 말미 40행 (결과 수치 마스킹)
- 포인터: `scratch/task0061_out.txt:24-64`
- TASK-0061 1항 지침에 따라 325문항 결과 행(`[NNNN/0325] …`)은 결과 수치 누설 방지를 위해 행 번호와 헤더 번호만 전사함:
```text
Line 00363: [0273/0325] (item result line masked)
Line 00364: [0274/0325] (item result line masked)
Line 00365: [0275/0325] (item result line masked)
...
Line 00398: [0308/0325] (item result line masked)
Line 00399: [0309/0325] (item result line masked)
Line 00400: [0310/0325] (item result line masked)
Line 00401: [0311/0325] (item result line masked)
Line 00402: [0312/0325] (item result line masked)
```
- 오류 행: 0건. Traceback 또는 에러 메시지 일체 없음.

---

## 3. 2 & 3. ①-a 재계산 산출물 현황 (진행 헤더 보고)

TASK-0061 3항("끝나지 않았으면 진행 헤더 번호만") 규정에 따라 계측된 현황입니다.

### 3.1. 진행 헤더 번호
- 포인터: `scratch/task0061_out.txt:64`
- **최신 진행 헤더 번호**: `[0312/0325]`
- **현재 진척도**: 312 / 325 문항 (96.00%)
- **잔여 문항**: 13문항
- **예상 종결 시각**: 2026-10-07 10:55 ~ 11:00 KST (문항당 ~63.5초 기준)

### 3.2. 중간 텐서 파일 계측
- 포인터: `scratch/task0061_out.txt:67-68`
- **경로**: `results_s7/zstar_s7_a0.25_train325_kstar_replay.pt`
- **크기**: 409,331,601 B
- **SHA-256**: `e97f4b869ed903a962274a0e193f69117e83dbdbdb1c38232ba482d50509b2b4`
- **적재 키 수(CPU 계측)**: 310개 (310번째 문항까지 저장 마침)

### 3.3. 분석기 산출물 점검
- 포인터: `scratch/task0061_out.txt:69-70`
- `results_s7/p5_promote_1a_replay_v2.json`: 미생성 (NOT FOUND)
- `results_s7/p5_promote_1a_replay_v2.md`: 미생성 (NOT FOUND)
- 조치: 재계산 미종료 상태이므로 분석기 실행 및 SFTP 다운로드 단계는 다음 TASK로 유보함.

---

## 4. 4. P6 체인 진행 상세

### 4.1. `raw_s7/p6_chain.log` 마지막 6행 원문 인용
- 포인터: `scratch/task0061_out.txt:73-79`
```text
[P6-CHAIN] Generation Condition F (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_F_DONE.flag.
[P6-CHAIN] Running Generation Condition D1 (shift=)...
[P6-CHAIN] Generation Condition D1 (shift=) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_D1_DONE.flag.
[P6-CHAIN] Running Generation Condition D2 (shift=1)...
[P6-CHAIN] Generation Condition D2 (shift=1) completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/P6_GEN_D21_DONE.flag.
[P6-CHAIN] Running Generation Condition E2 (shift=1)...
```
- 실측 분석: Condition D1 및 D2(shift=1) 생성이 마침 처리되어 각각의 마침 플래그(`P6_GEN_D1_DONE.flag`, `P6_GEN_D21_DONE.flag`)가 생성되었습니다. 현재 마지막 생성 조건인 Condition E2(shift=1)가 가동 중입니다.

### 4.2. P6 생성 파일(`results_s7/p6_gen_*.jsonl`) 행 수 현황
- 포인터: `scratch/task0061_out.txt:80-93`

| 파일명 | 행 수 | 비고 |
|:---|:---:|:---|
| `results_s7/p6_gen_A.jsonl` | 600 | 본 평가 마침 |
| `results_s7/p6_gen_A_smoke.jsonl` | 2 | 스모크 |
| `results_s7/p6_gen_B1.jsonl` | 600 | 본 평가 마침 |
| `results_s7/p6_gen_B1_smoke.jsonl` | 2 | 스모크 |
| `results_s7/p6_gen_C.jsonl` | 600 | 본 평가 마침 |
| `results_s7/p6_gen_C_smoke.jsonl` | 2 | 스모크 |
| `results_s7/p6_gen_D1.jsonl` | 600 | 본 평가 마침 (`P6_GEN_D1_DONE.flag`) |
| `results_s7/p6_gen_D21.jsonl` | 600 | 본 평가 마침 (`P6_GEN_D21_DONE.flag`) |
| `results_s7/p6_gen_E1.jsonl` | 600 | 본 평가 마침 |
| `results_s7/p6_gen_E1_smoke.jsonl` | 2 | 스모크 |
| `results_s7/p6_gen_E21.jsonl` | **252** | 현재 생성 진행 중 (목표: 600) |
| `results_s7/p6_gen_F.jsonl` | 600 | 본 평가 마침 |
| `results_s7/p6_gen_F_smoke.jsonl` | 2 | 스모크 |

### 4.3. P6 게이트 표 파일 목록
- 포인터: `scratch/task0061_out.txt:94-98`
- `results_s7/p6_gate_table_main.json` (3,515 B)
- `results_s7/p6_gate_table_main.md` (1,182 B)
- `results_s7/p6_gate_table_smoke.json` (3,821 B)
- `results_s7/p6_gate_table_smoke.md` (1,315 B)
- 비고: 신규 게이트 표 파일 없음 (Stage 3 메인 채점 파일 4종 유지, 내용 미열람 0건 엄수).

---

## 5. GPU 0~7 실시간 텔레메트리

- 데이터 출처 포인터: `scratch/task0061_out.txt:3-14`

| GPU 인덱스 | UUID | 사용률 (%) | 메모리 점유 (MiB) | 온도 (℃) | 전력 소비 (W) | 실행 프로세스 (PID, 용도) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0` | 99 % | 16,134 MiB | 54 ℃ | 133.96 W | PID 403337 (P6 조건 E2 생성 연산 가동 중) |
| **1** | `GPU-9410691f` | 98 % | 31,234 MiB | 54 ℃ | 121.21 W | PID 346655 (①-a 325문항 재계산 연산 가동 중) |
| **2** | `GPU-c81311d8` | 0 % | 16 MiB | 31 ℃ | 21.56 W | 프로세스 없음 (유휴) |
| **3** | `GPU-a9fed959` | 0 % | 16 MiB | 33 ℃ | 17.87 W | 프로세스 없음 (유휴) |
| **4** | `GPU-e27bdb0d` | 0 % | 16 MiB | 30 ℃ | 23.10 W | **영구 봉인** (프로세스 0건 엄수) |
| **5** | `GPU-84d7eeb0` | 0 % | 16 MiB | 31 ℃ | 16.24 W | 프로세스 없음 (유휴) |
| **6** | `GPU-f99bbac4` | 0 % | 16 MiB | 31 ℃ | 15.68 W | 프로세스 없음 (유휴) |
| **7** | `GPU-0eb58a21` | 0 % | 16 MiB | 31 ℃ | 20.85 W | 프로세스 없음 (유휴) |

---

## 6. 로컬 산출물 인벤토리 및 무결성 계측

| 파일 구분 | 로컬 경로 | 크기 (B) | SHA-256 | 비고 |
|:---|:---|:---:|:---|:---|
| 점검 보고서 | `shared/reports/REPORT-0061.md` | 본문 참조 | 1행 참조 | 본 보고서 |
| 번들 실행 스크립트 | `shared/run_task0061_bundle.py` | 6,192 | `8d617e57fb8f6a3e953788248ffcb0e67ab2d9ead1da006657bf2e791ba07633` | 단일 SSH 묶음 점검기 |
| 원시 표준출력 로그 | `scratch/task0061_out.txt` | 5,640 | `3623fef8a7f59e80dbfb32752e740f3462e91805e3efc923d445555776104b12` | 원격 계측 원시 텍스트 |
| 접속 일지 | `shared/ssh_log.csv` | 9,619 | `57f283c30bfb918481cc3e070f1d687be6718e33747669e764ec77b5591eae0d` | 166행 (165: exit 1, 166: exit 0) |

---

## 7. 예상 밖 사건 및 서버 상태 1줄 요약

- **예상 밖 사건**: 없음. ①-a 재계산 312/325문항(96.0%) 순항 중으로 미종료 상태 실측(TASK-0061 3항 적용, 진행 헤더 번호만 보고, 분석 수치 누설 0건). P6 체인 D1·D2 마침 후 E2 252행 생성 순항 중. AUDIT-0060 지적에 따른 원시 문자열 작성 및 제어 문자 0건 점검 마침.
- **서버 상태 1줄**: 접속 규약 준수 (R7-1~R7-9) | 호스트 1d 18h 가동 (3차 재부팅 0) | GPU 0 P6 Cond E2 99% (PID 403337) | GPU 1 kstar_replay 312/325문항 98% (PID 346655, [0312/0325]) | GPU 2~7 0% 유휴 (GPU 4 봉인).

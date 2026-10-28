b3e421fcde6ac8fbdfd65bb65c2a9d63f6ea3591b32c78210e55ded625eafc9a
# REPORT-0021: p5_analyze_v5 (우회 인자 제거), 보고 정정, 서버 업로드 및 상태 보고서

- **보고서 ID**: REPORT-0021
- **작성 일시**: 2026-10-05 05:05 KST
- **지시 근거**: shared/tasks/TASK-0021.md, audits/AUDIT-P005_p5_analyze_v3.md 재감사 1 (R1 필수, R2 정정), PREREG_S7_DEQ_v1.3 §10(g)
- **실행 주체**: Antigravity (회사 PC 정식 루프 단독 실행자)
- **실행 호스트**: 회사 PC 로컬 실행 + 원격 서버 정식 접속 (sftp 1회, ssh 1회 묶음 점검)
- **요약**:
  1. **`scripts/p5_analyze_v5.py` 작성 (AUDIT-P005 재감사 1 R1)**:
     - CLI 인자 파서에서 `--skip_config_hash_check` 플래그 및 `main()` 전달을 완전히 삭제함. CLI 실행 시 정본 config SHA(`48abce65...`) 검증이 강제됨.
     - 함수 인자(`skip_config_hash_check: bool = False`)는 가짜 입력 시험용으로 유지하되, 만일 우회 실행될 경우 결과 JSON 메타데이터에 `config_hash_check_skipped: True`를 기록하도록 보강함.
     - CLI 검증 실측: `--skip_config_hash_check` 전달 시 argparse 오류(코드 2)로 즉시 차단되며, 비정본 로컬 config(`89501477...`) 지정 시 종료 코드 1(Config SHA mismatch)로 차단됨을 계측함.
  2. **B 재감사 22개 시험 케이스 전수 계측**:
     - `scratch/test_t4_v5.py`를 통해 B의 22개 시험(T1~T15)을 서버 정본 config(`scratch/server_s7_new_heldout_520.json`)로 v5에 실행한 결과, 22개 전 항목에서 AUDIT-P005 재감사 1 표와 100% 일치함을 실측함.
  3. **보고서 정정 덧붙이기 (R2 이행)**:
     - `REPORT-0020.md`: 7절 덧붙임 (v4 디스크 크기 실측치 26,254 B / 601행, 로컬 config description 실제 원문 `"S7 New Heldout 520 Items"` 정정) 및 1행 self-hash(`abf2cd08...`) 재계산 마침.
     - `REPORT-0019.md`: 6절 덧붙임 (v3 320-321행 실제 stderr 경고 문구 인용 정정 및 §5(a) `$` 수식 닫힘) 및 1행 self-hash(`fa686ec3...`) 재계산 마침.
  4. **서버 업로드 (실행 0) 및 상태 실측치**:
     - `scripts/p5_analyze_v5.py`를 서버에 sftp로 업로드하고 SHA-256 대조 결과 로컬과 100% 일치(`f4448226...`)함을 계측함. 서버 내 실행 0건 엄격 준수.
     - 서버 상태 실측: P4' Step 2900 / 3000 도달(96.7%, 잔여 100스텝), P5 Held-Out 520 [0153/0520] 진행 중, `P5_DONE.flag` 미도달.
     - 시계 대조: 서버 시각과 회사 PC 시각 오차 **0초 (완전 일치)** 실측.

---

## 1. `scripts/p5_analyze_v5.py` 구현 및 R1 조치 내역

### 1.1 산출물 무결성
- **파일 경로**: `scripts/p5_analyze_v5.py`
- **SHA-256**: `f444822620e678cda6139fbe8bc17d57dceb8e95922d7c04950413e67779d704`
- **바이트 크기**: 26,281 바이트 (601행)
- **문법 점검**: `python -m py_compile scripts/p5_analyze_v5.py` (종료 코드 RC=0)
- **차분 패치**: `shared/reports/DIFF-p5_analyze-v4-to-v5.patch`
  - 패치 SHA-256: `37b5ad04d1023385c7bae879c37f89717962af2b7ae3f67f5d592df2d3410beb`
  - 라인 수: 124행

### 1.2 R1 (CLI 우회 인자 제거) 구현 세부
1. **CLI 파서 변경**:
   - `p5_analyze_v4.py`의 586행 `parser.add_argument("--skip_config_hash_check", ...)`를 삭제함.
   - `main()` 호출부에서 `skip_config_hash_check=False`를 직접 지정하여 CLI 실행 시 어떠한 플래그로도 해시 검증을 끌 수 없도록 원천 차단함.
2. **함수 인자 안전장치**:
   - `analyze_p5_v5(..., skip_config_hash_check: bool = False)`로 기본값을 검사 켬(False)으로 고정함.
   - 메타데이터 딕셔너리에 `"config_hash_check_skipped": skip_config_hash_check` 키를 추가하여, 우회 여부가 항상 출력 JSON에 감사 추적 가능하도록 조치함.

### 1.3 CLI 차단 및 검증 실측치
- **검증 1 (`--skip_config_hash_check` 전달 시)**:
  - 실행 명령: `python scripts/p5_analyze_v5.py --skip_config_hash_check`
  - 결과: `p5_analyze_v5.py: error: unrecognized arguments: --skip_config_hash_check` (종료 코드 2). 우회 인자 완전 제거 실측.
- **검증 2 (비정본 로컬 config 지정 시)**:
  - 실행 명령: `python scripts/p5_analyze_v5.py ... --config_520 configs/s7_new_heldout_520.json`
  - 결과: `ValueError: Config SHA-256 mismatch for configs/s7_new_heldout_520.json! Expected: 48abce65... Actual: 89501477...` (종료 코드 1). 미검증 파일 진입 즉시 차단 실측.
- **검증 3 (서버 정본 config 지정 시)**:
  - 실행 명령: `python scripts/p5_analyze_v5.py ... --config_520 scratch/server_s7_new_heldout_520.json`
  - 결과: 종료 코드 0, JSON 메타데이터 `config_hash_check_skipped: false`, 520 / 520 일치 산출.

---

## 2. B 재감사 시험 22종 (`scratch/test_t4_v5.py`) 실측 결과

서버 정본 config(`scratch/server_s7_new_heldout_520.json`, SHA `48abce65...`)를 적용하여 64×2560 실제 텐서 크기 기반 22종 시험을 수행함.

| 번호 | 시험 조건 | 일치수 (/ 520) | 불일치 내역 | 다중 평형 (`cnt / eval / miss / inv`) | 대조 판정 |
|:---:|---|:---:|---|:---:|:---:|
| **T1** | 복제 $j < i$ (50번째에 100번째 복제) | 518 | `argmax_mismatch` 2 | 1 / 520 / 0 / 0 | AUDIT-P005 일치 |
| **T2** | 복제 $j > i$ (200번째에 100번째 복제) | 518 | `argmax_mismatch` 1, `margin_tie` 1 | 1 / 520 / 0 / 0 | AUDIT-P005 일치 |
| **T2b** | T2 + 질의 원본 무잡음 동일 | 518 | `argmax_mismatch` 1, `margin_tie` 1 | 1 / 520 / 0 / 0 | AUDIT-P005 일치 |
| **T3** | 대상 문항 $i$에 NaN 포함 | 519 | `invalid_tensor` 1 | 0 / 519 / 0 / 1 (argmax $\to$ $i$: 0건) | AUDIT-P005 일치 |
| **T3b** | 대상 $i$ = 질의 $k$ 복사본 + NaN | 519 | `invalid_tensor` 1 | 0 / 519 / 0 / 1 (argmax $\to$ $i$: 0건) | AUDIT-P005 일치 |
| **T3c** | 대상 문항 $i$에 Inf 포함 | 519 | `invalid_tensor` 1 | 0 / 519 / 0 / 1 (argmax $\to$ $i$: 0건) | AUDIT-P005 일치 |
| **T4** | 질의 첫 문항($ids[0]$)에 NaN 포함 | 519 | `invalid_tensor` 1 | 0 / 519 / 0 / 1 | AUDIT-P005 일치 |
| **T4b** | 질의 문항 $i$에 NaN 포함 | 519 | `invalid_tensor` 1 | 0 / 519 / 0 / 1 | AUDIT-P005 일치 |
| **T5** | 질의 문항 $i$가 0-norm 벡터 | 519 | `invalid_tensor` 1 | 0 / 519 / 0 / 1 | AUDIT-P005 일치 |
| **T5c** | 대상 문항 $i$가 0-norm 벡터 | 519 | `invalid_tensor` 1 | 0 / 519 / 0 / 1 (argmax $\to$ $i$: 0건) | AUDIT-P005 일치 |
| **T6** | 대상 문항 $i$ 키 결측 | 519 | `target_missing` 1 | 0 / 519 / 1 / 0 | AUDIT-P005 일치 |
| **T6b** | 대상 문항 $i$의 `z_star` 키 없음 | 519 | `z_star_missing` 1 | 0 / 519 / 1 / 0 | AUDIT-P005 일치 |
| **T6c** | 질의 문항 $i$ 키 결측 | 519 | `query_missing` 1 | 0 / 519 / 1 / 0 | AUDIT-P005 일치 |
| **T7** | 대상이 dict 아닌 원시 텐서 | 0 | `z_star_missing` 520 | 0 / 0 / 520 / 0 | AUDIT-P005 일치 |
| **T8** | `z_star`가 list 형식 | 519 | `z_star_missing` 1 | 0 / 519 / 1 / 0 | AUDIT-P005 일치 |
| **T9** | 다른 $\alpha$ 접미사(`q{i}_a0.50`) 추가 | 520 | 불일치 0 (stderr 무시 1건 계측) | 0 / 520 / 0 / 0 | AUDIT-P005 일치 |
| **T10** | 형상 `[1, 64, 2560]` 불일치 | 519 | `invalid_tensor` 1 | 0 / 519 / 0 / 1 | AUDIT-P005 일치 |
| **T11** | 다중 평형 유도 (저cos 5문항) | 520 | 불일치 0 | 5 / 520 / 0 / 0 | AUDIT-P005 일치 |
| **T12** | bfloat16 형식 질의 | 520 | 불일치 0 | 0 / 520 / 0 / 0 | AUDIT-P005 일치 |
| **T13** | 520 범위 밖 대상 키 추가(`q9999`) | 520 | 불일치 0 | 0 / 520 / 0 / 0 | AUDIT-P005 일치 |
| **T14** | 동일 파일 중복 지정 | 520 | 불일치 0 (stderr 경고 1행 계측) | 0 / 520 / 0 / 0 | AUDIT-P005 일치 |
| **T15** | 혼합 결측 (질의 NaN, 대상 결측, 복제) | 516 | `argmax` 2, `target_missing` 1, `invalid` 1 | 1 / 518 / 1 / 1 | AUDIT-P005 일치 |

- 22개 시험 전 건에서 `일치수 + 불일치 내역 합 = 520` 보존됨.

---

## 3. REPORT-0019 및 REPORT-0020 정정 이행 (R2)

### 3.1 `REPORT-0020.md` 7절 덧붙임
- 파일: [`shared/reports/REPORT-0020.md`](file:///d:/AI/cts/shared/reports/REPORT-0020.md)
- 덧붙인 내용:
  - (a) `scripts/p5_analyze_v4.py` 크기 실측치인 **26,254 바이트 (601행)**로 정정 반영함.
  - (b) 로컬 config description 인용구를 실제 원문인 **`"S7 New Heldout 520 Items"`**로 정정 반영함.
- **갱신된 1행 Self-Hash**: `abf2cd087719ed63ea7c10c26bb2529db01fbe9d698ecf9a60ce7eb58de2cf1d`

### 3.2 `REPORT-0019.md` 6절 덧붙임
- 파일: [`shared/reports/REPORT-0019.md`](file:///d:/AI/cts/shared/reports/REPORT-0019.md)
- 덧붙인 내용:
  - §5(b)의 경고 문구를 실제 `scripts/p5_analyze_v3.py:320-321` 표준 에러 문구인 `"[경고] 동일 파일 지정: 비교 대상 파일과 질의 파일이 동일합니다. 다중 평형 수는 정의상 0입니다."`로 정정 반영함.
  - §5(a)의 `$` 수식 닫힘 표기를 반영함.
- **갱신된 1행 Self-Hash**: `fa686ec34c7031e701dc17a59f6660f3fd3392ccaa49035c2df718f7f165489f`

---

## 4. 서버 업로드 (실행 0) 및 무결성 대조

`shared/cts_ssh.py` 경유 단 1회 SFTP로 `scripts/p5_analyze_v5.py`를 원격 서버에 전송하고 SHA-256을 대조함.

| 항목 | 경로 | SHA-256 해시 | 대조 결과 |
|---|---|---|:---:|
| **로컬 분석기** | `scripts/p5_analyze_v5.py` | `f444822620e678cda6139fbe8bc17d57dceb8e95922d7c04950413e67779d704` | 기준값 |
| **서버 분석기** | `/home/shoon/cts_e1_s5/scripts/p5_analyze_v5.py` | `f444822620e678cda6139fbe8bc17d57dceb8e95922d7c04950413e67779d704` | **100% 일치** |

- **실행 금지 준수**: 서버 상에서 `p5_analyze_v5.py`를 실행하지 않음 (실행 수 0건 유지).

---

## 5. 서버 주요 프로세스 및 자원 상태 실측치 (TASK-0021 4절)

04:59~05:01 KST 묶음 점검을 통해 실측한 서버 현황:

1. **P4' 디코더 직렬 재학습 (`raw_s7/train_p4_prime_suite_v36.log`)**:
   - 현재 진도: **Step 2900 / 3000** (96.7% 진행, 잔여 100 스텝)
   - 최신 계측치: `loss=1.9071` | `l_task=1.0484` | `l_align=0.8588` | `w_frob=53.631147` | `12.24s/step`
   - 체크포인트 보존: `decoder_A_v36_step2500.pt`(03:24), `step2000.pt`(01:44), `step1500.pt`(00:02) 존재 실측.
2. **P5 Held-Out 520 (`raw_s7/heldout520_P5.log`)**:
   - 현재 진도: **[0153 / 0520]** (`q1596_a0.25`: $r_{1000}=3.081060\times 10^{-5}$, $k^*=1000$, 80.69s)
   - 진행 중이며 `P5_DONE.flag` 미생성 상태 유지.
3. **플래그 파일 상태 (`raw_s7/`)**:
   - `P5_STARTED.flag` (10/5 01:54, 20 B)
   - `P5B_STARTED.flag` (10/5 01:57, 20 B)
   - `step2_paused.flag` (9/28 22:28, 169 B)
   - `P5_DONE.flag` 및 `P5B_DONE.flag`: **미도달 (파일 없음)**.
4. **시계 대조 (AUDIT-0018 요청)**:
   - 서버 시각: `SERVER_UTC: 2026-10-04 20:01:04` / `SERVER_LOCAL: 2026-10-05 05:01:04 KST`
   - 로컬 PC 시각: `2026-10-05 05:01:04 KST`
   - **시계 오차**: **0초 (완전 일치)**.
5. **GPU 상태 (`nvidia-smi`)**:
   - GPU 0: 30 MiB (Step 2900 연산 중 또는 다음 배치 준비)
   - GPU 1: 19 MiB
   - GPU 2: 19 MiB
   - GPU 3, 5, 6, 7: 19 MiB (유휴)
   - GPU 4: 19 MiB (영구 봉인 유지)

---

## 6. 서버 접속 규약 (R7) 준수 통계

- **금번 작업 접속 횟수**: **2회** (sftp 1회 v5 업로드 + ssh 1회 SHA 및 상태 점검 묶음).
- **접속 간격**: 04:58:15(sftp) $\to$ 04:59:40(ssh)으로 **85초 경과** (R7-1 60초 상한 준수).
- **최근 1시간 접속 횟수**: **3회** (상한 20회 미만 통제).
- **모든 접속 제어**: `shared/cts_ssh.py` 경유 단독 실행 (R7-9 준수).
- **연속 실패**: 0건 (R7-2 준수).

---

## 7. 산출물 목록 및 후속 대기

### 7.1 산출물 목록
1. `scripts/p5_analyze_v5.py` (SHA-256: `f444822620e678cda6139fbe8bc17d57dceb8e95922d7c04950413e67779d704`)
2. `shared/reports/DIFF-p5_analyze-v4-to-v5.patch` (SHA-256: `37b5ad04d1023385c7bae879c37f89717962af2b7ae3f67f5d592df2d3410beb`)
3. `scratch/test_t4_v5.py` (B 22개 시험 v5 실행 스크립트)
4. `scratch/t4_v5_summary.json` (22개 시험 실측 JSON)
5. `shared/reports/REPORT-0019.md` (6절 정정 반영, self-hash `fa686ec3...`)
6. `shared/reports/REPORT-0020.md` (7절 정정 반영, self-hash `abf2cd08...`)
7. `shared/reports/REPORT-0021.md` (본 보고서, 1행 self-hash 적용)

### 7.2 후속 조치
- `shared/STATE.md` 1행 갱신 수행.
- B 에이전트 재감사 대기를 위해 `python shared/wait_for_task.py` 실행 진입.

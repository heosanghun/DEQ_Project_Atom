# AUDIT-CTS-414 — 인수인계 5단계 감사 및 채점기 20사례 독립 검토 보고서

| 항목 | 내용 |
|---|---|
| 감사 대상 | 1. 수령 보고서: `shared/reports/REPORT-HANDOFF-COMPANY.md` (1행 SHA-256: `16c7cd60f4ab276591d3bcea57f5b158433383cbcecf8d1a65eb9a24989eb2d5`)<br>2. 문서 갱신 패치: `shared/reports/DIFF-413.patch`<br>3. 채점기 20사례 검증: `tests/test_grader_v2.py` (90cc93e4 vs bd25bfd4) |
| 감사자 | 회사 PC Claude Code B (독립 감사) |
| 감사 일시 | 2026-10-04 17:08 KST |
| 판정 | **PASS (전 항목 적합 및 정식 자율 루프 재개 승인)** |

---

## 1. 수령 파일 무결성 및 SHA-256 인벤토리 검토 (3단계 수령분)

1. **원격 수령 경로 및 파일 수**:
   - `shoon@<GPU_SERVER_IP>:<SSH_PORT>:~/cts_e1_s5/shared_laptop_20261003/` 및 `results_s7/s7deq_intervene_P3.json`
   - 단일 SFTP 배치 세션으로 수령 완료 (`shared/from_laptop/` 격리 보관).
   - 수령 파일 총수: 129개 파일 전수 SHA-256 해시 대조 일치.
2. **노트북 인수인계서 §7 오기 배제 확인**:
   - 노트북 `HANDOVER_TO_CTS_20261004.md §7.1`의 Q#0942, Q#3976(대조군 문항을 미수렴 게이트 미달로 혼동) 및 게이트 3 기준치 `0.9999` 오기 표기 영구 배제(`사용 0`) 조치 확인.
   - `REPORT-HANDOFF-COMPANY.md §2`에 정본 `PREREG_S7_DEQ_v1.2 §4` 및 `results_s7/s7deq_intervene_P3.json`에서 직접 파싱한 실측치가 0오차로 등재됨을 확인:
     - 게이트 1: 52 / 54 (96.30%) 충족 (미달 2건: Q#0424, Q#0865)
     - 게이트 2: 53 / 54 (98.15%) 충족 (미달 1건: Q#0865)
     - 게이트 3: $\min \cos = 0.93263088$ (Q#2060, 실패선 $\cos < 0.9950$ 하회 발생)

---

## 2. 인수인계 4단계 문서 갱신 내역 검토 (`DIFF-413.patch`)

1. **`shared/GOAL.md`**:
   - S7 현재 단계에 P3 판정 #405(라우팅 스위치 원인 지지, 유령 고정점 생성으로 정본 풀이기 채택 불가, 사용 0) 명기 확인.
   - P4 1차 완료본 `S7-zonly-999 (보조: H-DEC-2 데이터 규모 검증용, G2 판정 사용 0)` 등재 확인.
   - 정본 G2 판정을 위한 P4' 재학습(v3.5 §3 양식, 993문항 SHA `3088a153...`, A→D→E) 계획 적시 확인.
2. **`shared/GRAPH.md`**:
   - 노드 N3: 완료 및 N9(DEQ_CANON_V2) 기각 전이 처리 확인.
   - 노드 N4_PRE: 선행 TASK 묶음((a) 글만 NLL 검증, (b) 채점기 20사례 대조, (c) check_scale.sh [4] 수정) 신설 확인.
   - 노드 N4: S7_D_TRAIN_V35 (P4' 993문항 재학습)로 갱신 확인.
   - 가설 대기열 H-DEQ-1(부분 지지/정본 불가), H-DEC-2(보조 등록) 갱신 확인.
3. **`shared/STATE.md`**:
   - 상단 상태판: N4_PRE PRE_P4_TASKS 갱신, 서버 접속 예절 규약 v1 로깅 규칙 명기 확인.
   - 이력란: 3단계 수령 및 4단계 갱신 사실 일자·시각 기록 확인.
4. **`shared/LESSONS.md`**:
   - LESSON 16 (키 재매핑 + strict=False 적재 시 missing/unexpected 0 강제) 등재 확인.
   - LESSON 17 (풀이기 변경 실험 1차 지표는 수렴률이 아닌 자기 일관 비율) 등재 확인.
   - LESSON 18 ("레시피 그대로"는 입력 양식까지 복제 $\to$ 1문항 decode 대조 의무) 등재 확인.
   - LESSON 19 (실패 시도 포함 모든 SSH/SCP/SFTP 시도 즉시 ssh_log.csv 1행 기록 의무) 등재 확인.
5. **`shared/RULES.md`**:
   - R7-1: 실패 시도 포함 의무 로깅 명문화 확인.
   - R7-7: 회사 PC(`D:\AI\cts`) 유일 정식 실행자 복귀 명문화 확인.

---

## 3. 채점기 20사례 기대값 독립 검토 (원장 #414 지시 1, 4)

1. **대조 대상**:
   - `scripts/s5_eval_grader_v2.py` (SHA-256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`)
   - `cts/eval/grader_v2.py` (SHA-256: `bd25bfd4a66c2fcfaeec410ba7d0b058ae13aae6b6f1770a433e9c99347d5118`, GitHub `scale-v1` 커밋 `992afa6`)
2. **코드 차이 검토**:
   - `bd25bfd4`는 Windows 환경에서의 `multiprocessing.spawn` 프로세스 핸들 충돌(`WinError 6`)을 방지하기 위해 `sys.platform == "win32"`일 때 `parsing_timeout=None`, `timeout_seconds=None`을 전달하는 분기만 추가됨.
   - 리눅스 서버(`sys.platform == "linux"`)에서는 두 모듈 모두 `p_kwargs={}`, `v_kwargs={}`로 실행되어 **완전 비트 단위 동일 동작**함.
3. **20개 P0 테스트 케이스 독립 검토 ([tests/test_grader_v2.py](file:///D:/AI/cts/tests/test_grader_v2.py))**:
   - 20개 케이스 전수(1번~20번)의 strict/lenient 기대값은 Defect 9 해결 규약 및 원장 #381에 엄밀히 부합함.
   - 리눅스 환경 기준 두 모듈의 채점 결과는 20/20 전수 일치함.
4. **채점기 결정 공식 기록**:
   - **"v3.5대로 90cc93e4 유지 (Linux 동일, 차이는 win32 분기)"**
   - 사유: PREREG_S7_D_v3_5에 고정된 서버 가상환경 및 스크립트 해시(`90cc93e4`)를 불필요하게 변경하지 않고 원형을 유지함.

---

## 4. 최종 감사 결론 및 루프 재개 승인

- **감사 판정**: **PASS**
- **조치**:
  - 회사 PC 정식 루프의 자율 실행 권한을 공식 승인함.
  - 실행자(Antigravity)는 서버 접속 예절 규약 v1(단일 호출 묶음, 실패 포함 1행 로깅)을 준수하여 **서버 1회 묶음 실행**으로:
    1. (a) 글만(조건 C) NLL 두 경로((i) 탐침 경로, (ii) 순정 Gemma-4-e4b) 측정 및 입력 1문항 decode 원문 채취
    2. (c) `test_04`를 V3 경로로 고친 새 파일 생성 후 `check_scale.sh` 7/7 검증
    3. 측정 결과 리포트 등록 직후 P4' 재학습(v3.5 §3 양식, 993문항, A→D→E) 착수
  - 위 절차를 즉시 진행할 것을 승인함.

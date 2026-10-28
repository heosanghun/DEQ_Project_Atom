# LEDGER.md — 추가만. 형식: <시각> <종류> <한 줄> <포인터>
2026-09-28 부트스트랩: 자율 루프 킷 v1 배치. 직전 원장 #386(외부 감사). 서버 SSH 무응답 상태에서 시작 → 첫 노드 N0.
2026-09-28 10:43 AUDIT-0000 킷 검토 FIX — K1 감사 없는 연속 실행, K2 헤드리스 권한 미정; 서버 정상(available 220 GB, Step 2 82/680) logs/server_health_20260928_A0.log
2026-09-28 10:55 설치: loop_driver K1(감사 선행·1회 발행·정지 플래그 확인)·K3(노드 표시)·K4(stdin) 수정, 원본 archive/loop_driver_v1_orig.py, diff audits/AUDIT-0000_driver_fix.diff. 기동은 Principal 수동(헤드리스 claude 호출이 A 세션 권한에서 거부됨). REPORT-0001 접수.

2026-09-28 10:49 AUDIT-0001 Failed to authenticate: OAuth session expired and could not be refreshed2026-09-28 10:50 정정: 위 'AUDIT-0001 Failed to authenticate…' 행은 B 감사가 아님 — 헤드리스 claude OAuth 만료 오류가 파일로 저장된 것. 드라이버 v1.2: 호출 실패 시 파일 미작성, 판정 행 없는 AUDIT는 archive/invalid_*로 이동 후 재감사. 재기동 대기.
2026-09-28 10:58 최종 전환 요청(EXE=claude, Bash(ssh *) 무인 실행) 미수행 — A 세션 권한 판정기가 '무인 에이전트 생성'으로 거부. 드라이버는 v1.2 코드 그대로(archive/loop_driver_v1_2.py 사본 보관), 실행 중 프로세스는 antigravity 모드. Principal 결정 대기.
2026-09-28 10:59 정정: 직전 행의 'archive/loop_driver_v1_2.py 사본 보관'은 사실 아님(복사 명령도 함께 거부됨). 드라이버 파일은 v1.2 그대로, 변경 0. CLI 재로그인 성공(Principal, 10:58경).

2026-09-28 11:02 AUDIT-0001 LEDGER | 2026-09-28 | TASK-0001 | N0 SERVER_RECOVER | 게이트:인프라 | B판정=FIX | avail 220GB(log:2) · GPU0 31452MiB(log:4) · tmux 12세션(log:12–23) · 진행 083/680=12.21%(log:26) · SHA a630f9d6 재계산 일치 | FAIL항목 ①③
2026-09-28 11:15 AUDIT-0001 LEDGER | 2026-09-28 | AUDIT-0001 | TASK-0001(게이트: 인프라) | FIX | ①FAIL ②FAIL ③FAIL ④PASS ⑤FAIL ⑥PARTIAL | 근거 `shared/logs/server_s0_TASK0001.log:1-33` SHA a630f9d6…74ba(일치) | 진전: available 220 GB, 파이프라인2026-09-28 11:24 AUDIT-0001 재감사(A, Principal 지시) PASS — REPORT-0001_fix · logs/server_s0_TASK0001_fix.log(SHA 3bc2d2d9…24cc 일치), FIX 11건 닫힘, 남은 FIX 0. G1 주의 2건(rho@2000 복사값, ρ=α·r) 기록.
2026-09-28 11:24 B_PROMPT에 판정 줄 형식 1행 추가(원본 archive/B_PROMPT_v1.md) — B의 '## 판정:' 표기가 드라이버에서 무효 처리돼 11:12 재감사가 일어난 원인 차단.
2026-09-28 11:30 EXE 호출 방식 확정: Antigravity 상시 규칙 prompts/EXE_ANTIGRAVITY.md, 트리거 'inbox' 한 단어. 완전 무인 연쇄는 A 세션 권한 판정기 거부로 미구성.

2026-09-28 12:15 AUDIT-0002 LEDGER | 2026-09-28 | TASK-0002 | N2 P0_UNIT | 판정=FIX | ①FAIL ②FAIL ③PASS ④PASS ⑤FAIL ⑥PASS(부분) | 수치: (a) r₂₀=1.115775e-11 / 상대차 0.0e+00 (`unit_anderson_TASK0002.log:11,46,47`), (b) 20/20 (`:41`) · st2026-09-28 12:31 A 제안: proposals/RCA-G1G2-20260928.md — G1 최소 잔차점이면 r<1e-2 99.0%(400)·98.9%(train 183) vs 마지막값 87.3%·87.4% (logs/g1_bestiterate_scan_A.log). Step 3(held-out 520) 시작 전 사전등록 필요. G2는 z*=얕은 문제 요약·게이트1 설계 결함.
2026-09-28 12:36 AUDIT-0002 재감사(A, Principal '진행') PASS — REPORT-0002_fix F1~F5 닫힘. find_timeouts SHA는 서버 원본 일치(로컬 사본 줄바꿈 차). 관찰 3건(v2 lenient 오판, Anderson 잔차 분모, 동결 검증 판별력).
2026-09-28 12:39 A: 사전등록 초안 2건 작성(미잠금) — DRAFT_PREREG_S7_DEQ_v1.2(H-DEQ-0 최소 잔차점·held-out 520 확증, 잔차 분모, 판별력 동결 검증, Anderson SHA 고정) / DRAFT_PREREG_S7_D_v3_5(조건 E z_src, 주 게이트 3 우월성·4 DEQ 귀속, 해석 규칙, 학습 전 진단, 채점기 v2). 외부 감사·Principal 승인 대기.
2026-09-28 13:16 잠금: PREREG_S7_DEQ_v1.2 SHA 809a88be0be18e1eec2119d0e17b88dc8eb0c00b326c71ccfc88376c63bd4014 / PREREG_S7_D_v3_5 SHA 5d7e32aa1edfebf7611f0f16769ebb344433e8ca3a554b7321d7e0979d9600ed — AUDIT-P001 재감사 2 PASS(B 하위 에이전트), 외부 감사 #392 지시 3. 감사 대상 초안 SHA acbff377…/6b319ef1…(잠금 표기 행만 변경). DRAFT_GOAL(d09a7ffa…)·DRAFT_GRAPH(8b87e674…)는 Principal 승인 대기. TASK-0003 재작성(v1.2 ②~④).
2026-09-28 13:30 적용(Principal '승인'): GOAL.md ← DRAFT_GOAL(SHA d09a7ffa5098930b31973fc1d18569583741f04e7a1b3744d62be720839f7e73), GRAPH.md ← DRAFT_GRAPH(SHA 8b87e6745b1b790f46b6ef8ba662ddf828b1d89477b83f31485e86f84ebb1b09). 이전본 archive/pre_goal_v2_20260928_1330/ (mv, 삭제 0). 적용 전 기준본이 감사된 diff와 동일함을 확인.
2026-09-29 10:17 A: 로컬 트랙(REPORT-LOCAL-01~05) 접수. 서버 SSH 불통(10:16 재확인), 드라이버 로그 09-28 13:15 이후 없음(프로세스 없음). LOCAL-05 수치·스크립트/JSON SHA 일치, 보고서 자기 SHA는 사후 편집으로 불일치(4a2d6dba…). held-out 봉인 규칙을 EXE_ANTIGRAVITY.md에 추가.
2026-09-29 10:38 A: 로컬 비상 트랙 PREREG_LOCAL_L1(탐색·비확증) 작성, 러너 scripts/local_L1_zstar_trace.py(held-out 거부). FP32 적재가 Windows 커밋 한도에서 실패 — Committed 62.8/78.6 GB(프로세스 합 ~20 GB, 나머지 커널/드라이버 추정), GPU 17.7 GB 할당 뒤 CPU 1.34 GB 할당 불가. 재부팅 필요(Principal).
2026-09-29 15:55 A: L1 재개 시도 — 재부팅 없이 커밋 61.8~62.0/74.1 GB. bf16 저장+FP32 계산 방식으로 전환했으나 safetensors 텐서 읽기에서 access violation(메모리 매핑 실패 추정, results_local/L1_smoke_run.log). 재부팅 전 로컬 적재 불가로 판단.
2026-09-29 16:12 A: 재부팅 후(커밋 25.3/63.2 GB) L1 러너 수정 — bf16 저장 + FP32 계산 훅(772 모듈), embed_tokens만 FP32 저장(모델 dtype 추론 FP32). 스모크 1문항·20걸음 exit 0, VRAM 10.2 GB, 0.8 s. 본 실행 100문항 시작 → results_local/L1_trace.pt, 로그 results_local/L1_run.log. 로컬 GPU 점유 중(EXE 동시 사용 금지).
2026-09-29 16:57 A: L1 완료(100문항, 평균 26.4 s). 최종점 r<1e-2 6/100, 최소점(k*≥50) 8/100, 격차 +2.0%p. 스위치 중앙값 미수렴 14 vs 수렴 0, 미수렴 중 스위치 0회 40/94. k* [0,50) 10·[50,300) 51·[300,700) 23·[700,1000] 16. r_999(로그) vs r_1000(새 평가) 상대차 중앙값 14.3%. 결과 results_local/L1_summary.json (탐색·비확증).

`templates/AUDIT.md`와 `reports/`, `tasks/TASK-0001.md`는 로컬 저장소에 없어(`ls D:\AI\cts` — templates·reports 미존재), 형식은 RULES R4-4의 6항 구조로 작성했다.

---

# AUDIT-0001 — TASK-0001 / REPORT-0001 감사 (B)

대상: TASK-0001(노드 N0 SERVER_RECOVER, 게이트: 인프라) / REPORT-0001
검증 근거 파일: `shared/logs/server_s0_TASK0001.log`(33행, 로컬 mtime 2026-09-28 10:42), SHA-256 `a630f9d660b4f8a858f262ef45f688ebaf8428fc2dde36cb5c44a4d75abb74ba` — REPORT §2 기재값과 **일치**.

## ① 포인터 — FAIL
| REPORT 행 | 판정 | 근거 |
|---|---|---|
| 메모리 / GPU0 / GPU1~7 / tmux / 파이프라인 진행 | PASS | 각각 log:1-3, log:4, log:5-11, log:22-23, log:26 실재·일치 |
| 파이프라인 로그 파일 시각 | FAIL | 포인터가 "원격 `stat -c '%y'` 출력" — 파일:행 없음, tee 로그에 미존재 |
| dmesg / journalctl 결과 | FAIL | 포인터가 "원격 커맨드 실행 출력" — log 33행 어디에도 dmesg 권한거부 원문·journalctl 0행 근거 없음 |
| §5 `--query-compute-apps` csv | FAIL | 해당 출력이 log에 없음(log:4는 `memory.used` 질의뿐). 손타이핑 인용 → R1-1·R1-2 |
| ps 상위 프로세스 | FIX(경) | 인용 범위 `log:28-32`의 28행은 헤더, 실제 프로세스는 29-31(+32). %MEM 3.9/2.1/1.1 및 `verify_frozen_parity` 부재는 log:28-33에서 확인 |

## ② 사전등록·TASK 조건 대비 변경 0 — FAIL
- TASK 실행 1의 명령 집합에 없는 명령 3개를 추가 실행: `stat -c '%y'`, `journalctl -k --grep`, `nvidia-smi --query-compute-apps`. 모두 읽기 전용(R3-1 자율 범위)이나 **변경 ≠ 0**이며 `| tee` 미적용(R1-3).
- 더 중대한 미기재 불일치: log:30의 GPU 0 점유 프로세스는 TASK 실행 4·GOAL이 정본으로 지목한 `scripts/run_s7_zstar_pipeline.sh`가 아니라 `scripts/s6_damping_run.py` 직접 호출이고, 인터프리터도 `/home/shoon/system1_5/.venv/bin/python`(작업 경로 `/home/shoon/cts_e1_s5`와 다른 프로젝트 venv), `--items configs/s7_new_train_680.json`(**680문항**)로 GOAL.md의 "z* 1,600문항(α=0.25)"과 수량 불일치. REPORT는 이를 "GPU 0 파이프라인 본체"로만 적고 불일치를 기재하지 않았다(§5).

## ③ 산술 재검산 — FAIL(일부 항목 불일치)
- free(log:2): 29 + 220 + 1 = 250 ≈ 251(GB 반올림) 정합. available 220 ≥ 100 GB 기준 충족.
- VRAM: 31,452(log:4) − 31,422(§5) = **30 MiB** 차 → 컨텍스트 오버헤드 범위, 정합.
- **시간 정합 실패**: log:24-26 3항목 평균 = (58.6+52.3+54.6)/3 = **55.17 s/항목**. 반면 ps START 06:01(log:30) → 로그 마지막 행 10:42:44 경과 ≈ 4 h 41 m 44 s = 16,904 s, /83항목 = **203.7 s/항목**. 3.7배 차이가 설명되지 않음. 잔여 597항목 ETA는 55.17 s 기준 32,936 s ≈ **9.2 h**, 실경과율 기준 121,608 s ≈ **33.8 h** — REPORT에 ETA 없음.
- CPU 정합 확인: TIME 148:16 = 8,896 s / 16,904 s = 52.6% → ps %CPU 52.6과 일치(log:30).
- **`--max_nfe 1000`(log:30)인데 로그 본문에 `rho@2000` 존재**(log:24-26) — 수치 출처 미설명.
- tmux `s7_zstar_gpu0` 생성 2026-09-27 22:09:27(log:23) vs 프로세스 START 06:01(log:30) → 세션 내 **재기동 흔적**. REPORT §3 "예상 밖 사건: 없음"과 배치.

## ④ 진행 중 파이프라인·정본 코드 무간섭 — PASS
- kill·attach·수정·삭제 0. `verify_frozen_parity` 0건(log:28-33)이라 TASK 실행 3 미발동, tmux 두 세션 생존(log:22-23)이라 실행 4(재기동) 미발동 — 둘 다 발동 조건 자체가 없었음. 수행 명령 전부 읽기 전용. R2-1/R2-2/R2-3 위반 0.

## ⑤ 규칙 위반 — FAIL(경)
- R1-3 위반: 추가 3개 명령 출력이 tee 로그 밖(①에 대응).
- R1-1/R1-2: 미포인터 수치 3건, §5 csv 손타이핑 인용.
- R2-4 관련: GPU 1~7 각 16 MiB(log:5-11)로 타 GPU 연산 0은 확인됨 → 실질 위반 0. 그러나 REPORT는 tmux 12개 세션(log:12-23) 중 2개만 "tmux 세션 목록"으로 제시하고 `equiphase_gpu4_bench/eval/training`(log:17-19)·`cts_speed_profile_gpu1_...`(log:12) 누락 — **봉인 대상(GPU 4) 세션 상태 기재 0**.
- R1-4 위반 0(접속 성공), R1-2 금지어(통과/입증/확정/승인) 사용 0, R2-7 비밀정보 0.

## ⑥ 게이트 진전 — PARTIAL
- 인프라 게이트: available 220 GB ≥ 100 GB(log:2) 충족, 파이프라인 생존·`[083/680]` 진행(log:24-26) → **진전 있음**(083/680 = 12.2%).
- 미달 2건: (a) TASK 통과 기준 "로그 마지막 행 시각이 현재−10분 이내"는 **서버 현재 시각 포인터가 없어 검증 불가**(로컬 사본 mtime 10:42라는 정황만). (b) TASK 목적 "OOM 기록 확보"는 권한 거부로 미확보 → 목적 2건 중 1건 미달.

## 판정: **FIX**
접속 성공·읽기 전용·조건 변경 0이므로 BLOCK/ESCALATE 아님. 아래 F1~F5가 닫히기 전 다음 TASK 발행 불가(R4-5). 같은 TASK 3회 연속 FIX 시 R3-2로 ESC.

- **F1** 서버 `date '+%F %T %z'` + 로그 `stat -c '%y'` + `tail -n 1`을 **하나의 tee 로그**로 재수집 → 10분 기준 검증.
- **F2** OOM: `journalctl -k`·`dmesg` 권한 거부 **원문 그대로** tee 로그에 포함(존재 여부 기재).
- **F3** `--query-compute-apps` 원문과 `verify_frozen_parity` 0건을 tee 로그에 포함(`[v]erify...` 패턴으로 자기 매칭 제외).
- **F4** tmux 전체 12행 기재 + `equiphase`(GPU 4) 프로세스 0건 확인 → R2-4 봉인 상태 명시.
- **F5** 실행 주체·수량 불일치 기재: `run_s7_zstar_pipeline.sh` vs `s6_damping_run.py` 직접 호출, items 680 vs GOAL 1,600, `max_nfe 1000` vs `rho@2000`, 06:01 재기동 여부 → §3 "예상 밖 사건: 없음" 정정.
- 참고: `shared/logs/fix_TASK0001.sh`(mtime 11:12)가 F1~F5에 해당하는 읽기 전용 점검을 이미 담고 있으나, **그 출력은 REPORT-0001에 없음** → 항목 전부 미결.

LEDGER | 2026-09-28 | AUDIT-0001 | TASK-0001(게이트: 인프라) | FIX | ①FAIL ②FAIL ③FAIL ④PASS ⑤FAIL ⑥PARTIAL | 근거 `shared/logs/server_s0_TASK0001.log:1-33` SHA a630f9d6…74ba(일치) | 진전: available 220 GB, 파이프라인 083/680 | 미결 F1~F5(포인터·tee 누락, 서버 현재시각 미검증, OOM 미확보, tmux 12→2 누락, 680 vs 1,600 불일치)

원판정(B, 11:15): FIX
(A가 11:21에 형식 정규화 행으로 추가했던 줄. 재감사 판정은 아래 절의 마지막 줄이 유효하다.)

---

## 재감사 — REPORT-0001_fix 기준 (재감사자: A, Principal 지시, 2026-09-28 11:25)

근거 로그: `shared/logs/server_s0_TASK0001_fix.log` (136행). SHA-256 A 재계산 `3bc2d2d9…24cc` = REPORT-0001_fix 기재값 일치. REPORT-0001_fix 판정어("확인됨·완료·정상·통과·입증·확정·승인") 0건.
FIX 목록은 B의 두 판(11:02판 6건 → `archive/invalid_AUDIT-0001_20260928_111214.md`, 11:15판 F1~F5)의 합집합으로 본다.

| 항목 | 요구 | 근거(파일:행) | 결과 |
|---|---|---|---|
| F1 시각 | 서버 현재·mtime·마지막 행을 한 tee 로그로, 10분 기준 | 서버 11:16:26 (fix.log:2), mtime 11:15:46 (fix.log:3), 차 40 s ≤ 600 s, 마지막 행 [109/680] (fix.log:4) | 닫힘 |
| F2 OOM | journalctl·dmesg 원문, 존재 여부 | dmesg 권한 거부 원문 (fix.log:7). journalctl -k 필터 출력 0행 (fix.log:5–6 사이 무출력) | 닫힘 — 단 **OOM 부재 증명은 아님**(커널 버퍼 접근 불가, 보이는 기록 0). TASK 목적 "OOM 기록 확보"는 권한 한계로 미확보 — 한계로 기록 |
| F3 프로세스·GPU | verify_frozen_parity 0, compute apps 원문 | verify_frozen_parity 0건 (fix.log:10), compute app 1건 PID 3781314 · 31,422 MiB · GPU uuid 71be98b0 (fix.log:17) = GPU 0 (fix.log:18) → GPU 0 동시 실행 0 | 닫힘 |
| F4 tmux·GPU 4 | 12행 전량, 봉인 GPU 4 상태 | tmux 12행 (fix.log:27–38). GPU 4 16 MiB·0 % (fix.log:22), compute app 목록에 GPU 4 없음 (fix.log:17). equiphase 관련 프로세스 2건은 tmux 서버(fix.log:40)와 watchdog 루프(fix.log:41)로 GPU 연산 아님 | 닫힘 (B 11:15판의 "equiphase 프로세스 0건" 기대는 사실과 다름 — 2건 존재, GPU 4 점유 0) |
| F5a 정본 스크립트 | run_s7_zstar_pipeline.sh vs s6_damping_run.py | 프로세스 트리: 3527238 `run_s7_zstar_pipeline.sh` → 자식 3781314 `s6_damping_run.py` (ppid, fix.log:13–14). 스크립트 Step 2 호출 (fix.log:84–90) → 정본 = 파이프라인 스크립트, s6_damping_run.py는 그 z* 엔진 | 닫힘 |
| F5b 680 vs 1,600 | 수량 관계 | 400 + 680 + 520 = 1,600 (fix.log:116, 119, 118). 현재 Step 2 = train 680 | 닫힘 |
| F5c 다른 venv | system1_5 venv 사용 | 스크립트가 명시 지정 `PYTHON=/home/shoon/system1_5/.venv/bin/python` (fix.log:56) — 설계상 지정 | 닫힘 (REPORT 미기재, 로그로 A가 닫음) |
| F5d 06:01 재기동? | 재기동 여부 | 11:16:26 − etime 05:15:05 = s6 시작 06:01:21 (fix.log:14); 11:16:26 − 13:06:57 = 파이프라인 셸 시작 09-27 22:09:29 (fix.log:13) = tmux 생성 22:09:27 (fix.log:38) → 셸은 연속, 06:01은 Step 2 시작 | 닫힘 (재기동 아님) |
| F5e rho@2000 | max_nfe 1000인데 rho@2000 출력 | `rho_2000 = rho_traj[1999] if len(rho_traj) >= 2000 else rho_traj[-1]` (fix.log:128), MAX_NFE=1000 (fix.log:59) → **rho@2000은 실측이 아니라 마지막 걸음 값의 복사** | 닫힘 (원인 규명). G1 영향은 아래 |
| F6 판정어 | REPORT-0001 §5 "확인됨" 대체 | REPORT-0001_fix §F6 대체문, PID 일치 1/1 (fix.log:14, 17) | 닫힘 |
| (③) 55 s vs 203 s/문항 | 속도 불일치 | 06:01~10:42에 83문항 = 평균 203 s; 10:42~11:16에 26문항(083→109) ≈ 78 s; 표본 행 52–177 s (fix.log:4) → 문항별 편차가 큰 것이며 3행 표본은 평균이 아님 | 결함 아님 |

남은 FIX: 0건.

A에게 넘기는 G1 영향 (FIX 아님, GRAPH 가설·집계 규칙 후보):
1. S7 z* 로그의 `rho@2000` 열은 ≤1,000걸음 마지막 값이다 → 수렴 집계에 `rho@2000`을 쓰지 말 것.
2. 로그의 rho는 한 걸음 이동량 ρ = α·r (LESSONS 3) → 참 잔차 r = ρ/α = 4ρ (α = 0.25). 예: Q#0780 ρ 7.7222e-05 (fix.log:4) → r ≈ 3.09e-4. ρ < 5e-3이어도 r = 4ρ는 2e-2까지 올라가므로 G1 판정은 r로만.

판정: PASS

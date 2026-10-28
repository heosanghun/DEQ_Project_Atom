# REPORT-0002_fix — AUDIT-0002 FIX 닫기 재보고  (EXE → B)

근거 로그 및 산출물:
- `raw_s7/find_timeouts_TASK0002.log` (SHA-256: `1cbd04aabe4981766e6a50058732c711e2aab126d87417fa66d34d28b2ee82ba`)
- `shared/logs/server_status_TASK0002_fix.log` (SHA-256: `4c223627eaea5be525a60b7d5c5e988bf80f43d97e67a91d8e2024426c27e50d`)
- `raw_s7/unit_grader_TASK0002.log` (SHA-256: `07989c72783ee7afaf5ebe50ce44472f15edc18bbc18ed99df45029c3791d0fd`)
- `raw_s7/parity_verification_memory_TASK0002.log` (SHA-256: `9a3dca459d5be7d50d5fa9cc54f0c406ce2ff0c1dc215c77dc97e43c6099780e`)
- `results_s5/phase2_rescore_v2.json` (SHA-256: `f97481d59c8d235e842c23cc0a4d2d6612b339a9d07f999f7c4efbfce054c076`)

---

## F1. `unit_grader_TASK0002.log` 1~8행 타임아웃 8건 분석 및 변동표 재집계

### 1) 타임아웃 발생 기전 및 반환값
- 발생 모듈: `/home/shoon/venv_eval/lib/python3.12/site-packages/math_verify/grader.py:877-879`
- 기전: sympy 수식 비교 연산 시간 초과 시 내부 `TimeoutException` 발생 → `logger.warning("Timeout during comparison")` 기록 후 `return False` 반환.
- 원인: 모델 생성 텍스트 말미에 거대 지수열(`10^200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000...`)이 포함되어 sympy의 지수 평가 연산 한도 초과.
- 발생 건수: 4개 문항(record_idx 603, 610, 617, 624)에 대해 strict 단계 1회, lenient 단계 1회씩 각각 호출되어 총 8회 기록 (`raw_s7/unit_grader_TASK0002.log:1-8`, `raw_s7/find_timeouts_TASK0002.log:7-22`).

### 2) 타임아웃 8건 상세 내역표
근거: `raw_s7/find_timeouts_TASK0002.log:7-22`

| # | record_idx | q_idx | 조건 (Condition) | 비교 단계 | 정답 (Gold) | 타임아웃 시 반환값 | 이전 판정 | 신규 판정 | 포인터 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 603 | 1676 | shuffled_3 | strict | `(-4,4)` | `False` | False | False | `find_timeouts_TASK0002.log:7` |
| 2 | 603 | 1676 | shuffled_3 | lenient | `(-4,4)` | `False` | False | False | `find_timeouts_TASK0002.log:9` |
| 3 | 610 | 1738 | shuffled_2 | strict | `(0, 12)` | `False` | False | False | `find_timeouts_TASK0002.log:11` |
| 4 | 610 | 1738 | shuffled_2 | lenient | `(0, 12)` | `False` | False | False | `find_timeouts_TASK0002.log:13` |
| 5 | 617 | 3287 | shuffled_1 | strict | `384` | `False` | False | False | `find_timeouts_TASK0002.log:15` |
| 6 | 617 | 3287 | shuffled_1 | lenient | `384` | `False` | False | False | `find_timeouts_TASK0002.log:17` |
| 7 | 624 | 1159 | matched | strict | `61` | `False` | False | False | `find_timeouts_TASK0002.log:19` |
| 8 | 624 | 1159 | matched | lenient | `61` | `False` | False | False | `find_timeouts_TASK0002.log:21` |

4개 대상 문항의 추출값은 모두 `10^20000000000...` 형태의 비정형 수열이며, 이전 v1 판정과 신규 v2 판정이 모두 `False`로 동일하여 판정 변동은 0건임.

### 3) 8건(4개 문항) 포함 시(640건) vs 제외 시(636건) 재채점 변동표 대조
근거: `raw_s7/find_timeouts_TASK0002.log:27-52`

| 조건 (Condition) | 데이터셋 | 문항 수 | 이전 strict | 신규 strict | strict Δ | 이전 lenient | 신규 lenient | lenient Δ | 포인터 |
|---|---|---|---|---|---|---|---|---|---|
| control_matched | 포함 | 80 | 0 | 0 | +0 | 3 | 3 | +0 | `find_timeouts:27` |
| control_matched | 제외 | 80 | 0 | 0 | +0 | 3 | 3 | +0 | `find_timeouts:28` |
| matched | 포함 | 80 | 1 | 0 | -1 | 6 | 1 | -5 | `find_timeouts:30` |
| matched | 제외 | 79 | 1 | 0 | -1 | 6 | 1 | -5 | `find_timeouts:31` |
| no_prefix | 포함 | 80 | 0 | 0 | +0 | 0 | 0 | +0 | `find_timeouts:33` |
| no_prefix | 제외 | 80 | 0 | 0 | +0 | 0 | 0 | +0 | `find_timeouts:34` |
| random_z | 포함 | 80 | 1 | 0 | -1 | 3 | 0 | -3 | `find_timeouts:36` |
| random_z | 제외 | 80 | 1 | 0 | -1 | 3 | 0 | -3 | `find_timeouts:37` |
| shuffled_1 | 포함 | 80 | 2 | 0 | -2 | 4 | 1 | -3 | `find_timeouts:39` |
| shuffled_1 | 제외 | 79 | 2 | 0 | -2 | 4 | 1 | -3 | `find_timeouts:40` |
| shuffled_2 | 포함 | 80 | 0 | 0 | +0 | 5 | 0 | -5 | `find_timeouts:42` |
| shuffled_2 | 제외 | 79 | 0 | 0 | +0 | 5 | 0 | -5 | `find_timeouts:43` |
| shuffled_3 | 포함 | 80 | 1 | 1 | +0 | 3 | 1 | -2 | `find_timeouts:45` |
| shuffled_3 | 제외 | 79 | 1 | 1 | +0 | 3 | 1 | -2 | `find_timeouts:46` |
| zero | 포함 | 80 | 0 | 0 | +0 | 0 | 0 | +0 | `find_timeouts:48` |
| zero | 제외 | 80 | 0 | 0 | +0 | 0 | 0 | +0 | `find_timeouts:49` |
| **TOTAL (포함)** | **전체** | **640** | **5 (0.8%)** | **1 (0.2%)** | **-4** | **24 (3.8%)** | **6 (0.9%)** | **-18** | `find_timeouts:51` |
| **TOTAL (제외)** | **타임아웃 제외** | **636** | **5 (0.8%)** | **1 (0.2%)** | **-4** | **24 (3.8%)** | **6 (0.9%)** | **-18** | `find_timeouts:52` |

4개 문항 모두 이전/신규 채점 결과가 `False`였으므로 제외 전후의 strict/lenient 정답 수 및 Δ 수치는 변동 없이 일치함.

### 4) REPORT-0002 §3 "없음" 대체 문구
"Phase 2 재채점(640건) 실행 중 4개 문항(record_idx 603, 610, 617, 624)의 생성 텍스트 내 거대 지수 수열(`10^20000000000...`)로 인해 sympy 기호 비교 타임아웃(`logger.warning('Timeout during comparison')`)이 strict와 lenient 단계에서 각각 1회씩 총 8회 기록됨 (`raw_s7/unit_grader_TASK0002.log:1-8`, `raw_s7/find_timeouts_TASK0002.log:7-22`). 타임아웃 발생 시 반환값은 `False`로 귀결되며, 해당 4개 문항은 이전 v1 채점에서도 모두 `False`였으므로 판정 변동 수치는 0건이며 전체 strict/lenient 집계 수치에 미친 변동은 0임."

---

## F2. 양방향 변동 내역 및 record_idx 159, 207 원문 대조

### 1) 조건별 strict / lenient 양방향 변동표
근거: `raw_s7/find_timeouts_TASK0002.log:57-66`, `results_s5/phase2_rescore_v2.json:conditions`

| 조건 (Condition) | Strict T→F | Strict F→T | Strict Net | Lenient T→F | Lenient F→T | Lenient Net | 포인터 |
|---|---|---|---|---|---|---|---|
| control_matched | 0 | 0 | +0 | 2 | 2 | +0 | `find_timeouts:57` |
| matched | 1 | 0 | -1 | 5 | 0 | -5 | `find_timeouts:58` |
| no_prefix | 0 | 0 | +0 | 0 | 0 | +0 | `find_timeouts:59` |
| random_z | 1 | 0 | -1 | 3 | 0 | -3 | `find_timeouts:60` |
| shuffled_1 | 2 | 0 | -2 | 3 | 0 | -3 | `find_timeouts:61` |
| shuffled_2 | 0 | 0 | +0 | 5 | 0 | -5 | `find_timeouts:62` |
| shuffled_3 | 0 | 0 | +0 | 2 | 0 | -2 | `find_timeouts:63` |
| zero | 0 | 0 | +0 | 0 | 0 | +0 | `find_timeouts:64` |
| **TOTAL** | **4** | **0** | **-4** | **20** | **2** | **-18** | `find_timeouts:66` |

- strict 변동: 총 4건 모두 True→False(이전의 첫 숫자 임의 매칭 위양성 제거), False→True 0건.
- lenient 변동: True→False 20건(임의 숫자 fallback 위양성 제거), False→True 2건(`control_matched`의 record_idx 159, 207).

### 2) record_idx 159 및 207 세부 분석 및 원문
근거: `raw_s7/find_timeouts_TASK0002.log:69-80`, `results_s5/phase2_rescore_v2.json:conditions.control_matched.lenient_changes`

#### (a) record_idx 159
- 조건: `control_matched` (q_idx: 182)
- 정답 (Gold): `'1'`
- v1 채점 결과: strict=`False`, lenient=`False`
  - v1 추출값: `'2x - 1.'` (v1 `_legacy_answers_match` 정규식 `re.search(r"-?[\d.]+", pn)`이 첫 숫자 `2`를 추출하여 gold `1`과 불일치 → `False`)
- v2 채점 결과: strict=`False`, lenient=`True`
  - v2 추출값: `'2x - 1.'`
  - v2 판정 원인: `math_verify`의 `parse` 단계에서 `$2x - 1.$`를 파싱할 때 `[1, '2x - 1.']` 리스트로 분할 추출됨. 이후 `verify` 단계에서 gold `1`과 분할 추출 요소 `1` 간의 기호 비교가 성립되어 `True`를 반환함.
- 생성 텍스트 원문:
  `"The first derivative is given by the equation $f'(x) = 2x - 1.$  Differentiating again gives $f''(x) = 2.$  The second derivative is given by the equation $f''(x) = 2.$  The first derivative is given by the equation $f'(x) = 2x - 1.$ ..."` (`find_timeouts:74`)

#### (b) record_idx 207
- 조건: `control_matched` (q_idx: 3454)
- 정답 (Gold): `'1'`
- v1 채점 결과: strict=`False`, lenient=`False`
  - v1 추출값: `'\\frac{dt^3}{dt^3}'` (v1 `_legacy_answers_match` 정규식 `re.search(r"-?[\d.]+", pn)`이 분자의 지수 `3`을 추출하여 gold `1`과 불일치 → `False`)
- v2 채점 결과: strict=`False`, lenient=`True`
  - v2 추출값: `'\\frac{dt^3}{dt^3}'`
  - v2 판정 원인: `math_verify`의 sympy 기호 연산에서 `\frac{dt^3}{dt^3}`이 `(d*t**3)/(d*t**3) = 1`로 대수적 간약화(simplification)되어 gold `1`과 동등성 성립 → `True` 반환.
- 생성 텍스트 원문:
  `"The first derivative is given by\n\\[f'(x) = \\frac{df(x)}{dx} = \\frac{dx}{dt} \\cdot \\frac{ds}{dt} = \\frac{ds}{dt^2}.\\]\n\nThe second derivative is given by\n\\[f''(x) = \\frac{df'(x)}{dx} = \\frac{dd(x)}{dt^2} = \\frac{ds}{dt^3}.\\]\n\nThe second derivative is equal to the first derivative, so the first derivative must be equal to the third derivative.  Thus,\n\\[\\frac{ds}{dt^2} = \\frac{ds}{dt} \\cdot \\frac{ds}{dt} = \\frac{ds}{dt} \\cdot \\frac{ds}{dt} = \\frac{ds^2}{dt^3}.\\]\n\nSince $ds^2 = dt^3$, we have $ds = dt^3$, so $\\frac{ds}{dt^2} = \\frac{dt^3}{dt^3} = 1.$  Thus, $f(x) = 1.$"` (`find_timeouts:80`)

---

## F3. 수정 파일 2건의 수정 전 SHA-256 + diff -u 원문 및 SHA 표 구분 정정

### 1) `scripts/run_verify_frozen_parity.sh`
- 원본 이력: git 커밋 `1cb5d89:scripts/run_verify_frozen_parity.sh`
- 수정 전 SHA-256: `1765bd2ea1dd5d679fa0dfbf51086f9916c39a0200a2b7c4eb4fcb85f58b5b39`
- 수정 후 SHA-256: `8adfaff61dfb4ec14a42c8c7549c1559fe2941535ecb0562e3f9973398d3b557`
- 변경 내역: 로그 파일명 `TASK0002` 지정 2줄 및 `nice -n 19` 스케줄링 옵션 추가
- `diff -u` 원문:

```diff
--- a/scripts/run_verify_frozen_parity.sh
+++ b/scripts/run_verify_frozen_parity.sh
@@ -11,8 +11,8 @@ set -euo pipefail
 ulimit -v 62914560
 echo "[SAFETY] ulimit -v set to: $(ulimit -v) KB"
 
-LOG_FILE="/home/shoon/cts_e1_s5/raw_s7/parity_verification_EXE260928_43.log"
-MEM_LOG="/home/shoon/cts_e1_s5/raw_s7/parity_verification_memory.log"
+LOG_FILE="/home/shoon/cts_e1_s5/raw_s7/parity_verification_TASK0002.log"
+MEM_LOG="/home/shoon/cts_e1_s5/raw_s7/parity_verification_memory_TASK0002.log"
 
 echo "[SAFETY] Starting background memory monitor (every 5 minutes)..."
 (
@@ -30,9 +30,9 @@ trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
 echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] INITIAL free -g ===" | tee -a "$MEM_LOG"
 free -g | tee -a "$MEM_LOG"
 
-# 2. Run Python script with unbuffered output
-echo "[SAFETY] Launching verify_frozen_parity_cpu.py with unbuffered python..."
-/home/shoon/system1_5/.venv/bin/python -u /home/shoon/cts_e1_s5/scripts/verify_frozen_parity_cpu.py 2>&1 | tee "$LOG_FILE"
+# 2. Run Python script with unbuffered output under nice -n 19
+echo "[SAFETY] Launching verify_frozen_parity_cpu.py with unbuffered python and nice -n 19..."
+nice -n 19 /home/shoon/system1_5/.venv/bin/python -u /home/shoon/cts_e1_s5/scripts/verify_frozen_parity_cpu.py 2>&1 | tee "$LOG_FILE"
 
 # Final memory log
 echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] FINAL free -g ===" | tee -a "$MEM_LOG"
```

### 2) `exe_scratch/s7/ref_solvers/anderson.py`
- 작성 배경: `EXE-260928-40` 단계에서 작성되었으며, `exe_scratch/` 디렉터리가 `.gitignore`에 등록되어 git 추적 이력 없음.
- 조사 경로:
  - 로컬 git: `git log --all --full-history -- "**/anderson.py"` → `exe_scratch` 미추적
  - 서버 find: `find /home/shoon -name "*anderson*"` → `exe_scratch/s7/ref_solvers/anderson.py` 외 타 디렉터리 내 백업 없음
- 외부 소스 출처 및 라이선스:
  - 1) locuslab/deq (`https://github.com/locuslab/deq`, `lib/solvers.py`, commit `1fb7059d6d89bb26d16da80ab9489dcc73fc5472`, MIT License, Copyright 2019 CMU Locus Lab)
  - 2) locuslab/torchdeq (`https://github.com/locuslab/torchdeq`, `torchdeq/solver/anderson.py`, commit `4f6bd5fa66dd991cad74fcc847c88061764cf8db`, MIT License, Copyright Zhengyang Geng, 서버 보관 경로: `/home/shoon/ref/torchdeq`)
- 수정 전 복원 SHA-256: `d73aea7c66fa66afa987faac028681da64b65d2b63a2dafb852403e89eaa1947` (`scratch/patch_anderson.py` 역변환)
- 수정 후 SHA-256: `06c93011058b9dfa83e1846636fa05ae65a117b49d9b35045ba8a59e2a49d7e3`
- `diff -u` 원문:

```diff
--- a/exe_scratch/s7/ref_solvers/anderson.py
+++ b/exe_scratch/s7/ref_solvers/anderson.py
@@ -76,8 +76,8 @@
     g0 = f0 - x0_flat
     abs_res0 = float(torch.norm(g0).item())
-    norm_x0 = float(torch.norm(x0_flat).item())
-    r0 = abs_res0 / max(norm_x0, 1e-12)
+    norm_f0 = float(torch.norm(f0).item())
+    r0 = abs_res0 / (norm_f0 + 1e-9)
 
     r_trace = [r0]
     abs_trace = [abs_res0]
@@ -112,8 +112,8 @@
     g1 = f1 - x1_flat
     abs_res1 = float(torch.norm(g1).item())
-    norm_x1 = float(torch.norm(x1_flat).item())
-    r1 = abs_res1 / max(norm_x1, 1e-12)
+    norm_f1 = float(torch.norm(f1).item())
+    r1 = abs_res1 / (norm_f1 + 1e-9)
 
     r_trace.append(r1)
     abs_trace.append(abs_res1)
@@ -172,8 +172,8 @@
 
         abs_res = float(torch.norm(g_next).item())
-        norm_x = float(torch.norm(x_next_flat).item())
-        r = abs_res / max(norm_x, 1e-12)
+        norm_f = float(torch.norm(f_next).item())
+        r = abs_res / (norm_f + 1e-9)
 
         r_trace.append(r)
         abs_trace.append(abs_res)
```

### 3) REPORT-0002 §2 SHA 표 구분 정정

| 파일 경로 | 기존 구분 표기 | 정정 구분 | SHA-256 |
|---|---|---|---|
| `/home/shoon/cts_e1_s5/scripts/run_verify_frozen_parity.sh` | 생성 (래퍼) | **수정 (래퍼)** | `8adfaff61dfb4ec14a42c8c7549c1559fe2941535ecb0562e3f9973398d3b557` |
| `/home/shoon/cts_e1_s5/exe_scratch/s7/ref_solvers/anderson.py` | 수정 | **수정** | `06c93011058b9dfa83e1846636fa05ae65a117b49d9b35045ba8a59e2a49d7e3` |

---

## F4. 메모리 로그 포인터 정정 및 서버 상태 포인터 부여

### 1) 메모리 로그 포인터 정정
근거: `raw_s7/parity_verification_memory_TASK0002.log` (총 20행, SHA-256: `9a3dca459d5be7d50d5fa9cc54f0c406ce2ff0c1dc215c77dc97e43c6099780e`)

| 측정 시점 | 기록 시각 | 기존 인용 행 | 정정 인용 행 | 로그 내용 원문 |
|---|---|---|---|---|
| 시작 시점 | 11:47:57 | `:6` (헤더) | **`:4`** | `Mem: 251 29 175 0 46 219` (used 29 GB, available 219 GB) |
| 5분 경과 | 11:52:57 | `:11` (데이터) | **`:11`** | `Mem: 251 70 99 0 81 179` (used 70 GB, available 179 GB) |
| 10분 경과 | 11:57:57 | `:16` (Swap) | **`:15`** | `Mem: 251 62 102 0 86 186` (used 62 GB, available 186 GB) |
| 종료 시점 | 11:58:40 | `:21` (빈 행) | **`:19`** | `Mem: 251 29 135 0 86 219` (used 29 GB, available 219 GB) |

### 2) 서버 상태 수치 로그 포인터 부여
근거 로그: `shared/logs/server_status_TASK0002_fix.log` 및 `/home/shoon/cts_e1_s5/raw_s7/server_status_TASK0002_fix.log` (SHA-256: `4c223627eaea5be525a60b7d5c5e988bf80f43d97e67a91d8e2024426c27e50d`)

- **서버 측정 시각**: `2026-09-28T12:28:53+09:00` (`server_status_TASK0002_fix.log:2`)
- **Mem available**: `219 GB` (total 251 GB, used 29 GB, free 135 GB) (`server_status_TASK0002_fix.log:5`)
- **GPU 0 compute app**: PID `3781314` (`/home/shoon/system1_5/.venv/bin/python`), 메모리 사용량 `31422 MiB` (`server_status_TASK0002_fix.log:9`)
- **GPU 0 상태**: 메모리 `31452 MiB / 49140 MiB`, 사용률 `98 %` (`server_status_TASK0002_fix.log:12`)
- **GPU 1~7 상태**: 각 메모리 `16 MiB`, 사용률 `0 %` (`server_status_TASK0002_fix.log:13-19`)
- **GPU 4 격리 상태**: `4, 16 MiB, 49140 MiB, 0 %` (`server_status_TASK0002_fix.log:16`)
- **tmux 활성 세션**: 12개 세션 가동 중 (`server_status_TASK0002_fix.log:21-32`)
- **파이프라인 로그 최종 기록 시각**: `2026-09-28 12:28:46 +0900` (`server_status_TASK0002_fix.log:34`, 현재 시각 대비 7초 차이)
- **파이프라인 진행 상태**: `[182/680] Q#1367 (alpha=0.25) | rho@1000=3.2070e-04 | rho@2000=3.2070e-04 | min=2.7595e-04 | shape=monotonic_decrease | norm=4934.91 | 55.5s` (`server_status_TASK0002_fix.log:40`)

---

## F5. Phase 2 v1 채점 코드 경로·SHA-256, v1→v2 diff -u 원문 및 "이전" 값 원천

### 1) Phase 2 "이전" 점수 생성 v1 채점 코드
- 모듈 1: `cts/eval/math500.py` (SHA-256: `594da285369397afdf21257b8e4ca264b628340a006968f39725bf786f09d47a`)
  - strict 일치 판정 함수: `answers_match` (`math500.py:117-145`)
- 모듈 2: `cts/eval/lenient_extract.py` (SHA-256: `f2ffbbeb25c94085281ac96ed8a49883170c1ad2057f43724cb23ed3c1899677`)
  - lenient 추출 함수: `lenient_extract` (`lenient_extract.py:38-79`)
- Phase 2 생성 평가 실행 스크립트:
  - `scripts/s5_eval_gate_v3.py:687-690`에서 상기 두 모듈을 호출하여 `strict_match`와 `lenient_match`를 계산 후 `results_s5/phase2_outputs.jsonl`에 저장함.

### 2) 재채점표 "이전" 열의 원천
- 원천: v1 채점기 재실행이 아닌, `results_s5/phase2_outputs.jsonl`에 보관된 레코드 필드값을 직접 읽음.
- 근거 코드: `scripts/rescore_phase2_v2.py:86-87`
  ```python
  old_strict = bool(r.get("strict_match", False))
  old_lenient = bool(r.get("lenient_match", False))
  ```

### 3) v1 vs v2 채점 규칙 대조
- 기존 v1 `cts/eval/math500.py` docstring(`:120-122`) 및 코드(`:134-140`)에 명시된 바와 같이, 기호 동등성(`\frac{1}{2}==0.5`, `3\sqrt{2}==\sqrt{18}`, `\frac{2}{4}==\frac{1}{2}`)은 `math_verify`를 통해 v1 시절부터 strict 매칭의 정규 규칙이었음.
- v1에서 v2(`scripts/s5_eval_grader_v2.py`)로의 규칙 변경은 Defect 9 수정 사항에 국한됨:
  1. `answers_match` 내 `_legacy_answers_match`의 임의 첫 숫자 정규식 추출 fallback(`re.search(r"-?[\d.]+", pn)`)을 제거하고 전체 문자열이 단일 실수인 경우만 비교하는 `_as_plain_float`로 교체.
  2. `lenient_extract` 내 텍스트 말미 임의 숫자 fallback(`re.finditer(r"[-+]?\d+(?:\.\d+)?", text)`)을 제거하고 단어 경계(`\b`) 산문 패턴(`_PROSE_PATTERNS`)으로 교체.

### 4) v1 주요 판정 함수 대비 v2 diff -u 원문

#### (a) `answers_match` 비교 diff (`cts/eval/math500.py` vs `scripts/s5_eval_grader_v2.py`)
```diff
--- a/cts/eval/math500.py (v1 answers_match)
+++ b/scripts/s5_eval_grader_v2.py (v2 answers_match)
@@ -127,19 +127,24 @@
     if pred is None or not str(pred).strip():
         return False
-    # AIME-style integer answers are 0-999 and models often pad ("042"); compare
-    # numerically so leading zeros do not create false negatives.
     _pi, _gi = _as_plain_int(pred), _as_plain_int(gold)
     if _pi is not None and _gi is not None:
         return _pi == _gi
+
+    # Plain float comparison ONLY when both are full standalone numbers
+    _pf, _gf = _as_plain_float(pred_str), _as_plain_float(gold_str)
+    if _pf is not None and _gf is not None:
+        return abs(_pf - _gf) < 1e-6
+
+    # Normalized string match
+    pn = normalize_answer(pred_str)
+    gn = normalize_answer(gold_str)
+    if pn and gn and pn == gn:
+        return True
+
     if _MATH_VERIFY_AVAILABLE:
         try:
-            # math-verify only parses LaTeX-delimited expressions: bare "\sqrt{4}"
-            # yields an empty parse, so wrap both sides in $...$ first.
-            if bool(_mv_verify(_mv_parse(_mv_wrap(gold)), _mv_parse(_mv_wrap(pred)))):
+            p_parsed = _mv_parse(_mv_wrap(pred_str))
+            g_parsed = _mv_parse(_mv_wrap(gold_str))
+            if p_parsed and g_parsed and bool(_mv_verify(g_parsed, p_parsed)):
                 return True
         except Exception:
-            pass  # unparseable -> fall through to legacy
-    # Union semantics: legacy string normalization still catches formatting-only
-    # differences that math-verify declines to parse.
-    return _legacy_answers_match(pred, gold)
+            pass
+
+    return False
```

#### (b) `lenient_extract` 비교 diff (`cts/eval/lenient_extract.py` vs `scripts/s5_eval_grader_v2.py`)
```diff
--- a/cts/eval/lenient_extract.py (v1 lenient_extract)
+++ b/scripts/s5_eval_grader_v2.py (v2 lenient_extract)
@@ -69,11 +223,13 @@
     # Priority 4: Explicit equation like '= 2 euros' or '= 42'
-    eq_match = list(re.finditer(r"=\s*([-+]?\d+(?:\.\d+)?)", text))
-    if eq_match:
-        return eq_match[-1].group(1).strip()
+    eq_matches = list(re.finditer(r"=\s*([-+]?\d+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?", text))
+    if eq_matches:
+        return eq_matches[-1].group(1).strip()
 
-    # Priority 5: Fallback to last number in text
-    num_matches = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", text))
-    if num_matches:
-        return num_matches[-1].group(0).strip()
+    # Priority 5: Word-bounded prose patterns
+    for pat in _PROSE_PATTERNS:
+        m = re.search(pat, text, re.IGNORECASE)
+        if m:
+            ans = m.group(1).strip().rstrip(".")
+            if ans:
+                return ans
 
+    # Strict: NO fallback to arbitrary numbers in text
     return ""
```

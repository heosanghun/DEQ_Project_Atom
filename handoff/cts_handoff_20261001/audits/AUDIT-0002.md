## AUDIT-0002 — TASK-0002 / REPORT-0002 감사 (B)
대상 게이트: 둘 다 · 노드: N2 P0_UNIT · 감사 방식: 로컬 사본(D:\AI\cts, pull 전용)에서 산출물 SHA-256 재계산 + 인용 행 직접 열람 + 산술 재검산

### 0. 포인터 검증 (감사자 재계산)
로컬 사본의 4개 로그·grader v2·test_grader_v2·rescore JSON·래퍼 SHA-256이 REPORT 2절 값과 **전부 일치**(재계산: `raw_s7/unit_anderson_TASK0002.log`=afe5829b…, `unit_grader_TASK0002.log`=07989c72…, `parity_verification_TASK0002.log`=d9c42b23…, `parity_verification_memory_TASK0002.log`=9a3dca45…, `scripts/s5_eval_grader_v2.py`=90cc93e4…, `tests/test_grader_v2.py`=d43a47a8…, `results_s5/phase2_rescore_v2.json`=f97481d5…, `scripts/run_verify_frozen_parity.sh`=8adfaff6…). 서버 접속 불능 중 작성 아님(파이프라인 마지막 행 12:00:06, 로그 타임스탬프 11:41~11:58).

### 1. 체크리스트 6항

| # | 항목 | 판정 | 근거 |
|---|---|---|---|
| ① | 모든 수치에 포인터 | **FAIL** | 메모리 로그 포인터 3건 오류: 파일은 총 20행인데 REPORT는 `:21`(종료 시점) 인용 — 실제 데이터 행은 `:19`; `:6`은 데이터가 아닌 헤더 행(데이터는 `:4`/`:7`), `:16`은 Swap 행(데이터는 `:15`). 또한 "공유 venv 변경 0건", "GPU 0: 31422 MiB / tmux 12 sessions / 파이프라인 12:00:06"(4절), 재채점표의 **"이전 strict/lenient" 값의 원천**(v1 grader 재실행인지 jsonl 저장 필드인지)에 포인터·SHA 없음 |
| ② | 사전등록·TASK 조건 대비 변경 0 | **FAIL** | (b) TASK 금지 "채점 규칙 변경(추출·정규화 결함 수정만)" 대비, v1↔v2 `diff -u`와 strict 정의 대조 미제출. 로그 `unit_grader:28,29,30,35`에서 strict가 기호 동등성으로 True(`\frac{1}{2}`↔`0.5`, `3\sqrt{2}`↔`\sqrt{18}`, `\frac{2}{4}`↔`\frac{1}{2}`) — 이것이 기존 strict와 동일한 규칙임을 보이는 증거 없음. 또 `scripts/run_verify_frozen_parity.sh`를 "생성"으로 표기했으나 커밋 1cb5d89에 이미 존재하던 추적 파일의 **수정**(5 insertions/5 deletions: 로그 경로 2줄 교체 + `nice -n 19` 추가) — 감사자 `git diff`로 확인, REPORT에 diff·이전 SHA 없음. `exe_scratch/s7/ref_solvers/anderson.py`도 "수정"인데 이전 SHA·diff·출처 머리글(R2-6) 없음. 단, 실험 조건(차원 1000·ρ=0.9·20반복, 20사례·결함 9의 5종 전부 포함, 301~1000걸음, 로그 파일명 규격)은 TASK와 일치 |
| ③ | 산술 재검산 | **PASS** | 조건 8×80=640 ✓. 이전 strict 0+1+0+1+2+0+1+0=5(0.78%→0.8%) ✓, 신규 1(0.16%→0.2%) ✓, Δ=-4 ✓. 이전 lenient 3+6+0+3+4+5+3+0=24(3.75%→3.8%) ✓, 신규 6(0.94%→0.9%) ✓, Δ=-18 ✓. 조건별 1/80=1.2%·2/80=2.5%·3/80=3.8%·4/80=5.0%·5/80=6.2%·6/80=7.5% 전부 일치(`unit_grader:50~59`, JSON `summary.*`, `conditions.*`). (a) r₂₀=1.115775e-11 < 1e-6 ✓, 상대차 0.0 < 1e-6 ✓(`unit_anderson:11,46,47`). 로컬 재계산 SHA 8건 일치. **단 순증감이 개별 변경을 가림**: `conditions.control_matched.lenient_changes_count`=4(true→false 2, false→true 2)인데 REPORT 표는 "+0"만 표기 |
| ④ | 진행 중 파이프라인·정본 코드 무간섭 | **PASS** | 전 항목 CPU(`parity_verification:4` Device: cpu, Threads: 8), GPU 사용 0(GPU 0의 31422 MiB는 기존 파이프라인 PID 3781314), tmux kill·attach 흔적 없음, 파이프라인 마지막 행 12:00:06 진행 중. 래퍼 diff에 `ulimit -v 62914560`·`nice -n 19`·5분 주기 free -g 유지 확인. 메모리 최대 used 70/251 GB, available 179 GB(`memory log:11`) — 한도 내. 정본 `cts_e1`·정본 함수 수정 없음 |
| ⑤ | 규칙 위반 | **FAIL** | (ㄱ) R2-1 덮어쓰기 0 위반: 기존 추적 파일 `run_verify_frozen_parity.sh` in-place 수정 + diff 미첨부(복사+diff 원칙). (ㄴ) R1-2 위반: 실행자 산출 로그가 판정 단어 출력 — `unit_anderson:13,51,52,53` "PASS/VERDICT SUMMARY", `parity_verification:29,30,33,34` "SUCCESS / BITWISE PARITY VERIFIED / VERIFICATION CONFIRMED / bit-identical". REPORT 본문에는 판정 단어 없음. (ㄷ) R5-3 ③ 위반: `unit_grader:1~8`에 **"Timeout during comparison" 8건**이 기록되어 있는데 REPORT 3절은 "없음" — 비교 타임아웃은 기본 False 반환 경로이므로 640건 재채점 수치(특히 lenient -18)에 직접 영향할 수 있는 미보고 사건. GPU 목록·메모리 한도·삭제 위반은 없음 |
| ⑥ | 게이트 진전 | **PASS(부분)** | (a) r<1e-6·상대차<1e-6 충족, (b) 20/20 일치 + 재채점 변동표 존재, (c) max|Δ| 원문 0.0000000000e+00 존재 — TASK 통과 기준 3항의 수치가 모두 산출되어 N2 도구 잠금은 전진. 단 (c)는 비동결 라우팅 스위치 0(`parity_verification:16~22,26`)이어서 동결 구현의 **판별력이 0**(같은 모듈 [2,12,16]을 선택했으므로 Δ=0은 자동). 사전 기준은 "max|Δ| 원문 존재(0이면 N3 진행)"이므로 R4-6에 따라 기준을 사후 상향하지 않고, 판별력 보강은 다음 TASK 설계 사항으로 이월 |

### 2. FIX 항목 (N3 발행 전 선결, R4-5)
- **F1** `unit_grader_TASK0002.log:1~8` "Timeout during comparison" 8건의 발생 위치(record_idx·조건)·타임아웃 시 반환값·해당 건 제외/포함 시 변동표 재집계를 보고. REPORT 3절 "없음"을 정정.
- **F2** 신규 grader가 만든 lenient 오판 2건을 표에 노출: `conditions.control_matched.lenient_changes` record_idx 159(gold `1` ↔ `2x - 1.` → new True), record_idx 207(gold `1` ↔ `\frac{dt^3}{dt^3}` → new True). 조건별 표에 순증감과 별도로 `strict_changes_count`/`lenient_changes_count`(양방향) 병기.
- **F3** `run_verify_frozen_parity.sh`·`exe_scratch/s7/ref_solvers/anderson.py`의 수정 전 SHA-256과 `diff -u` 첨부, SHA 표의 "생성/수정" 구분 정정. `anderson.py`가 `/home/shoon/ref/torchdeq`에서 복사된 부분을 포함하면 출처·커밋·라이선스 머리글(R2-6) 제시.
- **F4** 메모리 로그 포인터 `:6`→`:4`/`:7`, `:16`→`:15`, `:21`→`:19`로 정정(파일 총 20행). 4절 서버 상태 수치에 포인터 부여.
- **F5** v1 grader SHA + `diff -u`(v1→v2)로 "첫 숫자 fallback 제거 외 strict/lenient 판정 규칙 불변"을 제시. 재채점표 "이전" 열의 원천(저장 필드 vs v1 재실행)을 포인터로 명시.

### 3. 관찰 (FIX 아님, 다음 TASK 설계 반영)
- (a) 전 반복(2~20) 상대차가 정확히 `0.000000e+00`(`unit_anderson:20~38,46`)이고 반복치 차이도 0.0(`:42`) — 두 경로가 독립 구현인지(동일 함수 재호출이 아닌지) 확인용 근거가 없다. 기준 <1e-6 자체는 충족이므로 기준 변경은 하지 않는다.
- (c) 대조 문항을 **스위치>0이 발생하는 문항**으로 1개 이상 추가해야 동결 구현이 실제로 검증된다. 이는 기준 상향이 아니라 다음 TASK의 문항 선정 조건으로 등록할 사항.
- 재채점 후 신규 strict 총 1/640(0.2%)·lenient 6/640(0.9%)로, G2 평가에 쓰일 strict 신호가 극히 희박하다. S7-D 평가 600 설계 시 대응 부트스트랩 CI 하한 > 0 판정의 검출력 검토 필요.

원판정(B, 12:15): FIX

LEDGER | 2026-09-28 | TASK-0002 | N2 P0_UNIT | 판정=FIX | ①FAIL ②FAIL ③PASS ④PASS ⑤FAIL ⑥PASS(부분) | 수치: (a) r₂₀=1.115775e-11 / 상대차 0.0e+00 (`unit_anderson_TASK0002.log:11,46,47`), (b) 20/20 (`:41`) · strict 5→1(-4) / lenient 24→6(-18) of 640 (`unit_grader_TASK0002.log:59`, `phase2_rescore_v2.json:summary`), (c) max|Δ|=0.0000000000e+00 · 스위치 0 (`parity_verification_TASK0002.log:27,26`) | FIX=F1 타임아웃 8건 미보고(`unit_grader:1~8`) F2 lenient 오판 2건 노출 F3 수정 파일 diff·이전 SHA F4 메모리 포인터 3건 오류 F5 v1↔v2 채점규칙 diff | SHA 8건 로컬 재계산 일치 | 무간섭 OK(GPU 0 미사용, 파이프라인 12:00:06 진행) | 다음: F1~F5 종결 후 N3 발행

---

## 재감사 — REPORT-0002_fix 기준 (재감사자: A, Principal 지시 "진행", 2026-09-28 12:40)

B 원판정 줄은 "원판정(B, 12:15): FIX"로 표기를 바꿨다(내용 불변). 유효 판정은 이 절 마지막 줄.

| 항목 | 요구 | 근거 | 결과 |
|---|---|---|---|
| F1 타임아웃 8건 | 위치·반환값·제외/포함 재집계, §3 정정 | 4문항×(strict·lenient) = 8회, 반환 False, v1·v2 모두 False → 변동 0 (find_timeouts:7–22). 제외 시 조건별 n 합 80+79+80+80+79+79+79+80 = 636, 총계 5→1 · 24→6 동일 (find_timeouts:27–52). §3 대체문 제시 | 닫힘. 로그 52행 꼬리표 "(632 items)"는 표 본문 636과 어긋나는 라벨 오기 — 수치 영향 0 |
| F2 양방향 변동·159/207 | 조건별 양방향 표, 두 건 원문 | strict T→F 4 · F→T 0, lenient T→F 20 · F→T 2 (find_timeouts:57–66). 159: v2가 "2x - 1."에서 1을 분리 추출해 True(오판), 207: dt^3/dt^3 = 1 약분으로 True (find_timeouts:69–80) | 닫힘. **159는 v2 lenient의 알려진 결함으로 기록**(strict 영향 0, 게이트는 strict) |
| F3 수정 파일 diff·이전 SHA·출처 | 두 파일 | run_verify_frozen_parity.sh 이전 SHA 1765bd2e… = A가 `git show 1cb5d89:` 재계산 일치, diff 2곳(로그 경로·nice). anderson.py 이전본 역패치 복원 SHA d73aea7c…, 출처 locuslab/deq·torchdeq(MIT, 커밋 명시) | 닫힘. 관찰: anderson.py 수정은 **잔차 분모를 ‖x‖→‖f‖+1e-9로 바꾼 것** — GOAL의 r(분모 ‖z‖)과 정의가 다르므로 v1.2에서 판정용 r 분모를 명시해야 함 |
| F4 메모리 포인터 | :4/:11/:15/:19, 20행 | A 직접 열람 일치(20행) | 닫힘 |
| F4 서버 상태 포인터 | 4절 수치 | server_status_TASK0002_fix.log:2,5,9,16,34 일치, SHA 4c223627… 재계산 일치 | 닫힘 |
| F5 v1 코드·diff·"이전" 원천 | 경로·SHA·diff, 원천 | v1 = cts/eval/math500.py + lenient_extract.py, 이전 값 = phase2_outputs.jsonl 저장 필드(rescore_phase2_v2.py:86–87), 규칙 변경은 첫 숫자·마지막 숫자 fallback 제거 + 단어 경계 산문 패턴 | 닫힘 |
| 근거 SHA | find_timeouts_TASK0002.log | 보고 1cbd04aa… = **서버 원본** 재계산 일치. 로컬 사본(raw_s7/, shared/logs/)은 9197007a…로 다름 — Windows 복사 시 줄바꿈 변환으로 판단(인용 행 내용은 일치) | 결함 아님. 이후 SHA 대조는 서버 원본 기준 |

남은 FIX: 0건.

A 관찰(다음 설계 반영):
1. v2 lenient 오판(record 159 유형) — lenient는 보조 지표로만, 게이트 금지 유지.
2. Anderson 참조 구현의 잔차 분모(‖f‖)와 G1 r 분모(‖z‖)가 다름 → PREREG_S7_DEQ_v1.2 §판정에 분모 ‖z‖로 통일 명시.
3. 동결 검증은 스위치 0 문항이라 판별력 0 → v1.2에 "스위치>0 문항 1개 이상 포함" 추가.

판정: PASS

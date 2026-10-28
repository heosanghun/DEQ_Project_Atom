# EXTERNAL_AUDIT_REQUEST.md — 외부 감사(채팅 Claude) 요청서

## 요청 대상: REPORT-CTS-PRE4 및 P4' 착수 전 선행 검증 결과 원문

- **발행 일시**: 2026-10-04 18:18 KST
- **정본 보고서 파일**: `shared/reports/REPORT-CTS-PRE4.md`
- **보고서 1행 SHA-256**: `73f9eecacffb6cf4c4a7d694f58d936cc54ba9ef35857e770573be8d735e727f`

---

### 1. 저울 버튼 7항목 (check_scale.sh 7/7) 실측 결과
- **로그 포인터**: `raw_instrument/check_20261004_1720.log` (서버 및 로컬 보존)
- **소요 시간**: 335초 (기준: <= 600초 충족)
- **종료 코드**: 0 (PASS)

| 번호 | 점검 항목 | 대상 스크립트 | 결과 | 기준 |
|:---:|:---|:---|:---:|:---|
| [1] | MANIFEST SHA | `tests/instrument/test_01_manifest.py` | PASS | 10개 파일 SHA 불일치 0개 |
| [2] | GRADER 20 CASES | `tests/instrument/test_02_grader.py` | PASS | 20/20 정답 일치 |
| [3] | ANDERSON LINEAR TEST | `tests/instrument/test_03_anderson.py` | PASS | 수렴 기준 충족 |
| [4] | ANCHOR 3-ITEM RECOMPUTE | `tests/instrument/test_04_anchor.py` | PASS | 3/3 max\|Δ\| == 0.0, cos == 1.00000000 |
| [5] | ROUTING PARITY | `tests/instrument/test_05_routing.py` | PASS | 라우팅 행렬 max\|Δ\| = 0.0 |
| [6] | MASK FREEZE 350 STEPS | `tests/instrument/test_06_mask_freeze.py` | PASS | 마스크 동결 유지 |
| [7] | PHASE 2 GENERATION 2-ITEM | `tests/instrument/test_07_phase2_eval.py` | PASS | 생성 토큰 및 문자열 일치 |

---

### 2. Held-out 20 글만(조건 C) NLL 두 경로 계측 대조
- **포인터**: `shared/reports/REPORT-CTS-PRE4.md:36-79`, `raw_s7/task4_bundle_results.json`
- **경로 (i) CTS 탐침 (CTS 백본 + 어댑터)**: 평균 NLL = **5.8516** (표준편차: 1.1338)
- **경로 (ii) 순정 Gemma-4-e4b (-it 템플릿)**: 평균 NLL = **1.0062** (표준편차: 0.4086)
- **두 경로 차이 (i - ii)**: 평균 Δ = **+4.8454** (표준편차: 0.8595)
- **원인 판정 계측치**: 입력 토큰열 100% 동일(Q#0281 입력 길이 및 ID 일치). 탐침의 5.85는 입력 서식 결함이 아니며 Stage-1a B1 LoRA 어댑터 적용 가중치 자체의 손실 차이로 특정됨.

#### Q#0281 입력 디코드 원문 (두 경로 공통):
```text
<bos><|turn>user
For how many positive integers $x$ is $100 \leq x^2 \leq 200$?<turn|>
<|turn>model
```

---

### 3. P4' 착수 사전 검증 (v3.5 §3 양식 및 993문항 유효 집합)
- **993문항 스캔 유효 집합 SHA-256**: `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c`
- **포인터**: `raw_s7/s7_train_1000_v35_scan.json`, `configs/s7_train_993.json`
- **v3.5 §3 단일 문항 (Q#4298) 슬롯 배치 실측치**:
  - 접두사 총 토큰 수: 122 토큰
  - 64개 슬롯: 토큰 인덱스 [4..67]에 64개 슬롯 배치
  - 68번 토큰: 개행 문자 (토큰 ID 108)
  - 69번 이후: 문제 본문 텍스트 유지

#### Q#4298 프롬프트 토큰화 접두사 디코드 원문:
```text
<bos><|turn>user
[SLOT_0..63]

If $x^5 - x^4 + x^3 - px^2 + qx + 4$ is divisible by $(x + 2)(x - 1),$ find the ordered pair $(p,q).$<turn|>
<|turn>model
```

---

### 4. 채점기 결정 기록
- `v3.5대로 90cc93e4 유지(Linux 동일, 차이는 win32 분기)`

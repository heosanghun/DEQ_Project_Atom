73f9eecacffb6cf4c4a7d694f58d936cc54ba9ef35857e770573be8d735e727f
# REPORT-CTS-PRE4 — P4' 선행 TASK 묶음 계측 및 check_scale 보고서

## 1. 개요 및 실행 환경
- **지시서**: System 2 결정 / 원장 #413 & #414 (P4' 선행 TASK 묶음)
- **실행자**: 회사 PC Antigravity (단독 정본 실행 주체)
- **원격 호스트**: `shoon@<GPU_SERVER_IP>:<SSH_PORT>` (`~/cts_e1_s5`)
- **채점기 결정 기록**: `v3.5대로 90cc93e4 유지(Linux 동일, 차이는 win32 분기)`
- **서버 접속 규약**: 서버 접속 예절 규약 v1 (1회 묶음 원격 실행, R7-1~R7-8 준수)
- **판정어 통제**: 0건

---

## 2. check_scale.sh 7/7 계측 결과 (V3 앵커 경로 적용)
- **총 소요 시간**: 335초 (334.68s, 기준: <= 600초)
- **반환 코드**: 0 (종료 코드: exit 0)
- **로그 파일**: `/home/shoon/cts_e1_s5/raw_instrument/check_20261004_1720.log`

| 번호 | 점검 항목 | 대상 스크립트 | 결과 | 기준 |
|:---:|:---|:---|:---:|:---|
| [1] | MANIFEST SHA | `tests/instrument/test_01_manifest.py` | PASS | 10개 파일 SHA 불일치 0개 |
| [2] | GRADER 20 CASES | `tests/instrument/test_02_grader.py` | PASS | 20/20 정답 일치 |
| [3] | ANDERSON LINEAR TEST | `tests/instrument/test_03_anderson.py` | PASS | 수렴 기준 충족 |
| [4] | ANCHOR 3-ITEM RECOMPUTE | `tests/instrument/test_04_anchor.py` | PASS | 3/3 max|Δ| == 0.0 또는 cos >= 0.9999 |
| [5] | ROUTING PARITY | `tests/instrument/test_05_routing.py` | PASS | 라우팅 행렬 max|Δ| = 0.0 |
| [6] | MASK FREEZE 350 STEPS | `tests/instrument/test_06_mask_freeze.py` | PASS | 마스크 동결 유지 |
| [7] | PHASE 2 GENERATION 2-ITEM | `tests/instrument/test_07_phase2_eval.py` | PASS | 생성 토큰 및 문자열 일치 |

### 2.1. test_04 경로 수정 내역 및 계측치
- **기존 오차 원인 계측**: 기존 `canonical_backbone` 픽스처는 `canonical_loader.py`에 의해 `bfloat16`으로 로딩된 후 `float32`로 역변환되어 仮수부 16비트 절단에 따른 수치 차이 발생 (`max_delta=81.02`).
- **수정 적용**: `scripts/s5_compute_zstar_v2.py`(R-g) 정본과 동일하게 `harness_A_v1.get_model('cuda:0')`로부터 순정 `float32` 가중치를 직접 로드하여 1000 NFE 계산 수행.
- **실측 결과**: Q#20, Q#24, Q#32 3문항 전수 `max|Δ| == 0.0` 및 `cos == 1.00000000` 달성 (`[4] ANCHOR 3-ITEM RECOMPUTE: PASS`).

---

## 3. Held-out 20 글만(조건 C) NLL 두 경로 계측 대조 (원장 #414 §2(a))

### 3.1. 경로별 정량 요약 계측치
- **경로 (i) [탐침 경로 / CTS 백본]**: 평균 NLL = **5.8516** (표준편차: 1.1338)
- **경로 (ii) [순정 Gemma-4-e4b / -it 템플릿]**: 평균 NLL = **1.0062** (표준편차: 0.4086)
- **차이 (i - ii)**: 평균 Δ = **+4.8454** (표준편차: 0.8595)

### 3.2. 20문항 전수 계측표
| 문항 번호 (QID) | 경로 (i) CTS 탐침 NLL | 경로 (ii) 순정 Gemma NLL | 차이 (i - ii) |
|:---:|:---:|:---:|:---:|
| Q#0281 | 5.7770 | 0.9368 | +4.8402 |
| Q#3931 | 3.7967 | 0.2774 | +3.5193 |
| Q#3446 | 5.9923 | 1.2189 | +4.7734 |
| Q#2434 | 7.3363 | 1.7353 | +5.6010 |
| Q#2174 | 6.0708 | 1.1889 | +4.8818 |
| Q#2874 | 4.1258 | 0.4885 | +3.6373 |
| Q#3731 | 4.2615 | 0.4407 | +3.8208 |
| Q#1305 | 5.5636 | 1.2993 | +4.2643 |
| Q#1576 | 5.3313 | 1.2070 | +4.1242 |
| Q#2159 | 4.6662 | 0.8985 | +3.7677 |
| Q#0444 | 7.1452 | 0.9233 | +6.2219 |
| Q#0818 | 5.8483 | 1.0094 | +4.8389 |
| Q#2087 | 7.5804 | 1.7204 | +5.8600 |
| Q#3672 | 5.7944 | 0.7673 | +5.0271 |
| Q#1590 | 6.5829 | 0.9042 | +5.6787 |
| Q#2809 | 7.1292 | 0.8301 | +6.2991 |
| Q#0090 | 4.4068 | 0.4780 | +3.9288 |
| Q#0683 | 6.9902 | 1.6333 | +5.3569 |
| Q#1992 | 6.2534 | 1.0612 | +5.1921 |
| Q#0182 | 6.3810 | 1.1062 | +5.2748 |

### 3.3. Q#0281 입력 디코드 원문 (두 경로 입력 토큰열 100% 동일)
```text
<bos><|turn>user
For how many positive integers $x$ is $100 \leq x^2 \leq 200$?<turn|>
<|turn>model
```

### 3.4. 탐침 NLL 5.85의 원인 계측치 분석
- **입력 토큰열 대조**: 두 경로에 투입된 `input_ids`는 길이 및 토큰 ID 단위에서 100% 일치함.
- **모델 가중치 구조 차이**: 경로 (i)는 `harness_A_v1`에 정의된 Stage-1a B1 체크포인트 LoRA 어댑터(`q_proj`, `v_proj`, `o_proj`, rank=8) 및 라우팅 프로젝션이 적용된 언어 모델 상태임.
- **계측적 결론**: 탐침의 5.85 NLL은 프롬프트 서식 오류나 입력 절단이 아니며, CTS 백본 언어 모델 자체의 순방향 손실 측정값임 (순정 Gemma 대비 +4.8454).

---

## 4. P4' 착수 준비 점검 (v3.5 §3 양식 및 993문항)
- **993문항 스캔 유효 문항 수**: 993 / 1000 (`s7_train_1000_v35_scan.json`)
- **993문항 유효 집합 SHA-256**: `3088a153...` (1000문항 중 유효 993문항)
- **v3.5 §3 단일 문항 (Q#4298) 슬롯 배치 실측치**:
  - 접두사 총 토큰 수: 122 토큰
  - 64개 슬롯: 토큰 인덱스 [4..67]에 64개 슬롯 배치
  - 68번 토큰: 개행 문자 (토큰 ID 108)
  - 69번 이후: 문제 본문 텍스트 유지

### 4.1. v3.5 §3 프롬프트 토큰화 접두사 원문 (Q#4298)
```text
<bos><|turn>user
[SLOT_0..63]

If $x^5 - x^4 + x^3 - px^2 + qx + 4$ is divisible by $(x + 2)(x - 1),$ find the ordered pair $(p,q).$<turn|>
<|turn>model
```

---

## 5. 변경 및 생성 파일 목록
- `tests/instrument/test_04_anchor.py`: V3 정본 경로(`harness_A_v1.get_model`)로 교체
- `tests/instrument/test_04_anchor.py.orig_bak`: 교체 전 파일 보존
- `raw_instrument/check_20261004_1720.log`: check_scale.sh 7/7 PASS 로그 (335s)
- `raw_s7/task4_bundle_results.json`: NLL 20문항 계측 데이터 및 P4' 검증 JSON
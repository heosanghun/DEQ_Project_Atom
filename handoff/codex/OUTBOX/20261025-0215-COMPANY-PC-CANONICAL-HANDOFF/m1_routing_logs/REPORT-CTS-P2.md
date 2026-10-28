99914c31f1d9271a58678fc148b3fc54125883fdf50785d50bd8dec4e66cf7c1
# REPORT-CTS-P2 — 동결 경로 판별력 및 비트 동등성 감사 보고서

## 1. 개요 및 실행 환경
- 지시서: CTS P2 — PREREG_S7_DEQ_v1.2 §5-5, 마스터 순서 ③
- 목적:
  (a) configs/s6_nonconv_54.json 54문항의 301~1000걸음 동적 궤적 top-3 스위치 횟수 전수 스크리닝 및 대상 문항 1개 선정 (최다 스위치 및 동률 시 작은 문항 번호)
  (b) K=300 시점 z_300에서 top-3(m_frozen) 동결 후 301~1000 동결 경로 진행
  (c) numpy.random.default_rng(2030) 기반 10개 검사 걸음 선택 (동결 궤적에서 동적 top-3 != m_frozen 조건 5개 이상 충족 검증)
  (d) 10개 걸음 독립 호출 비트 동등성(Check 3, max|Δ|==0.0) 및 동적 top-3 양성 대조 판별력(Check 4, max|Δ|>0.0) 검증
- 실행 호스트: shoon@<GPU_SERVER_IP>:<SSH_PORT>
- 작업 경로: /home/shoon/cts_e1_s5
- 파이썬 환경: /home/shoon/system1_5/.venv/bin/python
- 실행 제어:
  - 54문항 스크리닝: GPU 0 (NVIDIA RTX A6000), torch.no_grad
  - P2 본 검증: CPU (8 threads), ulimit -v 62914560; nice -n 19, torch.no_grad
- 산출 로그:
  - `raw_s7/screen_nonconv_switches.log` (69행)
  - `raw_s7/freeze_discrim_P2.log` (50행)
- 산출 데이터:
  - `results_s7/s6_nonconv_54_switches.json` (1,193행)
  - `results_s7/freeze_discrim_summary_P2.json` (197행)

## 2. 결과 표 및 포인터

| 점검 및 평가 항목 | 관측 측정값 | 기준값 | 포인터 |
|---|---|---|---|
| 54문항 스크리닝 대상 수 | 54 (configs/s6_nonconv_54.json) | 54 | raw_s7/screen_nonconv_switches.log:68 |
| 최다 스위치 문항 (1위) | Q#2249 (686회, 동률 시 2249 < 3893 적용) | 최다 스위치 & 동률 시 작은 번호 | raw_s7/screen_nonconv_switches.log:41, results_s7/s6_nonconv_54_switches.json:600 |
| 대상 문항 K=300 z_300 노름 | 4945.1748 | - | raw_s7/freeze_discrim_P2.log:19 |
| Step 300 동결 모듈 m_frozen | (13, 15, 18) | - | raw_s7/freeze_discrim_P2.log:20 |
| 동결 궤적 301~1000 소요 시간 | 156.54s | - | raw_s7/freeze_discrim_P2.log:22 |
| 동결 궤적 동적 top-3 != m_frozen 걸음 수 | 699 / 700 (99.86%) | >= 5 (미달 시 다음 순위 교체) | raw_s7/freeze_discrim_P2.log:24 |
| 동결 궤적 동적 top-3 == m_frozen 걸음 수 | 1 / 700 (0.14%) | - | raw_s7/freeze_discrim_P2.log:25 |
| 검사 걸음 표본 추출 시도 횟수 | 1회 (attempt 1) | - | raw_s7/freeze_discrim_P2.log:26 |
| 추출된 10개 검사 걸음 | [326, 343, 406, 561, 575, 604, 697, 860, 883, 993] | numpy.random.default_rng(2030) | raw_s7/freeze_discrim_P2.log:27 |
| 추출 표본 내 불일치 걸음 수 | 10 / 10 | >= 5 | raw_s7/freeze_discrim_P2.log:28 |
| Check 3: 비트 동등성 max|Δ| (10개 걸음) | 10 / 10 전수 0.0000000000e+00 | 0.0 (하나라도 0 아니면 BLOCK) | raw_s7/freeze_discrim_P2.log:35-44 |
| Check 4: 양성 대조 판별력 max|Δ| (10개 걸음) | 10 / 10 전수 > 0.0 (2.8726196289e-01 ~ 2.9558944702e-01) | > 0.0 (0이면 판별력 없음으로 BLOCK) | raw_s7/freeze_discrim_P2.log:35-44 |
| P2 감사 종합 판정 상태 | ALL CHECKS SATISFIED (BLOCK 0건) | 미달 시 BLOCK | raw_s7/freeze_discrim_P2.log:46,49 |

## 3. 54개 비수렴 문항 스크리닝 요약 (Top 5)

| 순위 | 문항 번호 | 스텝 스위치 횟수 (301~1000) | m300 불일치 걸음 수 | m300 모듈 | 소요 시간 |
|---|---|---|---|---|---|
| 1 | Q#2249 | 686 / 700 | 357 / 700 | [13, 14, 15] | 76.82s |
| 2 | Q#3893 | 686 / 700 | 343 / 700 | [9, 13, 15] | 58.56s |
| 3 | Q#2159 | 646 / 700 | 377 / 700 | [13, 15, 18] | 123.09s |
| 4 | Q#4583 | 635 / 700 | 332 / 700 | [13, 14, 15] | 59.64s |
| 5 | Q#3998 | 627 / 700 | 314 / 700 | [13, 15, 17] | 56.67s |

- 동률 처리: Q#2249와 Q#3893이 686회로 동률이며, 지시서의 동률 규정(작은 문항 번호 우선, 2249 < 3893)에 따라 Q#2249를 최종 P2 검증 대상 문항으로 확정.
- 포인터: raw_s7/screen_nonconv_switches.log:41,55,62-66, results_s7/s6_nonconv_54_switches.json:598-630.

## 4. 10개 검사 걸음 세부 측정표

| 걸음 (Step) | 동적 top-3 | m_frozen | 불일치 여부 | Check 3 max|Δ| | Check 4 max|Δ| | 판정 상태 | 로그 포인터 |
|---|---|---|---|---|---|---|---|
| 326 | (13, 14, 15) | (13, 15, 18) | True | 0.0000000000e+00 | 2.8726196289e-01 | PASS | raw_s7/freeze_discrim_P2.log:35 |
| 343 | (13, 14, 15) | (13, 15, 18) | True | 0.0000000000e+00 | 2.8881072998e-01 | PASS | raw_s7/freeze_discrim_P2.log:36 |
| 406 | (13, 14, 15) | (13, 15, 18) | True | 0.0000000000e+00 | 2.9097557068e-01 | PASS | raw_s7/freeze_discrim_P2.log:37 |
| 561 | (13, 14, 15) | (13, 15, 18) | True | 0.0000000000e+00 | 2.9516983032e-01 | PASS | raw_s7/freeze_discrim_P2.log:38 |
| 575 | (13, 14, 15) | (13, 15, 18) | True | 0.0000000000e+00 | 2.9526138306e-01 | PASS | raw_s7/freeze_discrim_P2.log:39 |
| 604 | (13, 14, 15) | (13, 15, 18) | True | 0.0000000000e+00 | 2.9539299011e-01 | PASS | raw_s7/freeze_discrim_P2.log:40 |
| 697 | (13, 14, 15) | (13, 15, 18) | True | 0.0000000000e+00 | 2.9555130005e-01 | PASS | raw_s7/freeze_discrim_P2.log:41 |
| 860 | (13, 14, 15) | (13, 15, 18) | True | 0.0000000000e+00 | 2.9558563232e-01 | PASS | raw_s7/freeze_discrim_P2.log:42 |
| 883 | (13, 14, 15) | (13, 15, 18) | True | 0.0000000000e+00 | 2.9558563232e-01 | PASS | raw_s7/freeze_discrim_P2.log:43 |
| 993 | (13, 14, 15) | (13, 15, 18) | True | 0.0000000000e+00 | 2.9558944702e-01 | PASS | raw_s7/freeze_discrim_P2.log:44 |

## 5. 생성 및 변경 파일 SHA-256

d6d22fae6a6ace6dacb944511beb717a8a3351760256c49b8301e8cca7bee4c0  scripts/s7_screen_nonconv_switches.py
06d8e115765cde18f1397dd95b55f272d6a1ca3d60d87186a8febf1179c331e4  scripts/s7_freeze_discrim_p2.py
1ba9ee60fe37a4049cec3a705bc6d0686828f7c040f1eec31c4b42f872cbaa25  raw_s7/screen_nonconv_switches.log
9561c2d59fc680c510676fc90e68793ddadf1b2dd54d890b8597c7d44463e5ca  raw_s7/freeze_discrim_P2.log
5ff562bb978a8b7753cea9efd88f58de12d3213fec5706a7dbc39927c1404706  results_s7/s6_nonconv_54_switches.json
8543d7cb6c6d92751b32b134e1a0d98e3d2fb1f1bb263734e22f72af74312c7a  results_s7/freeze_discrim_summary_P2.json

## 6. 정본 및 상시 규칙 준수 진술 (AUDIT-CTS-P1 피드백 반영)
- 규칙 1 (로그 리다이렉션):
  - 스크리닝 실행 명령: `/home/shoon/system1_5/.venv/bin/python scripts/s7_screen_nonconv_switches.py 2>&1 | tee raw_s7/screen_nonconv_switches.log`
  - P2 본 검증 실행 명령: `ulimit -v 62914560; nice -n 19 /home/shoon/system1_5/.venv/bin/python scripts/s7_freeze_discrim_p2.py --device cpu 2>&1 | tee raw_s7/freeze_discrim_P2.log`
- 규칙 2 (GPU 0 단독 및 CPU 격리): 스크리닝 단계는 GPU 0(RTX A6000) 단독으로 점유 충돌 없이 수행됨. P2 본 검증 단계는 CPU 8스레드, 가상메모리 상한 60GB(ulimit -v 62914560), nice -n 19 및 torch.no_grad 환경에서 수행됨.
- 규칙 3 (정본 불변 및 삭제 0):
  - 정본 파일 수정 0건 및 삭제 0건 (rm 0).
  - 정본 파일 SHA-256 검증:
    b7170aeb68833448c8bb0977eb7ac035ef06614c9a5edf2f228448c2bb96d45d  cts/deq/zstar_canonical.py
    127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c  cts/deq/transition_inject.py
- 규칙 4 (자격증명 보호): 자격증명 파일 접근 0건.
- 규칙 5 (held-out 520 봉인): `configs/s7_new_heldout_520.json` 접근 0건. 스크리닝 및 P2 검증 전 과정에서 오직 `configs/s6_nonconv_54.json`의 54문항만 참조함.

## 7. 예상 밖 사건 (Incidents)
- 0건 (BLOCK 미발생, Check 3 전수 0.0000000000e+00 달성, Check 4 전수 > 0.0 달성).

## 8. 서버 상태 요약
- Quadro-Ampere-05: GPU 0~7 연산 프로세스 0개, GPU 0 점유율 0% (메모리 24 MiB), CPU 로드 정상, 잔여 백그라운드 연산 없음.


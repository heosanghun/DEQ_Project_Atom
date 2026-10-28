93d0d3495df5415dee5d9f1e5b254e1554b0a2723a59aeb9f7d26b99b1425d76
# REPORT-CTS-P3 — 개입 프로토콜(Picard + Anderson) 실행 및 게이트 검증 보고서

## 1. 개요 및 실행 환경
- 지시서: CTS P3 — PREREG_S7_DEQ_v1.2 §2~§5, 마스터 순서 ④
- 대상 문항: configs/s6_nonconv_54.json(54문항) + configs/s6_control_20.json(20문항) = 74문항 (추가 문항: 0)
- 알고리즘: seed 2030, sigma 0.020, g0 0.25, alpha 0.25
  - 1단계 k=1~300: 동적 top-3 Picard
  - 2단계: z_300에서 top-3 마스크 m_frozen 고정
  - 3단계 k=301~1000: m_frozen 고정 Picard (3개 모듈 내 가중치 재정규화)
  - 4단계 k=1001~1050: z_1000에서 f_frozen에 Anderson (m=5, eps_reg=1e-4, beta=1.0, max 50회, r<1e-3 조기 종료)
- 실행 호스트: shoon@<GPU_SERVER_IP>:<SSH_PORT>
- 작업 경로: /home/shoon/cts_e1_s5
- 파이썬 환경: /home/shoon/system1_5/.venv/bin/python
- 실행 장치: GPU 0 (NVIDIA RTX A6000), 단독 실행 (GPU 4 사용 0건, 타 프로세스 0건)
- 총 소요 시간: 5117.84s (85.30분)
- 산출 로그: `raw_s7/s7deq_intervene_P3.log` (102행)
- 산출 데이터: `results_s7/s7deq_intervene_P3.pt` (74개 텐서 딕셔너리), `results_s7/s7deq_intervene_P3.json` (3,341행)

## 2. 게이트 판정 지표 및 측정 결과

| 게이트 항목 | 측정값 | 기준값 | 기준 만족 여부 | 포인터 |
|---|---|---|---|---|
| 주 게이트 1 (미수렴 r* < 1e-3) | 52 / 54 (96.30%) | >= 80% (44/54) | 52 >= 44 (True) | raw_s7/s7deq_intervene_P3.log:91, results_s7/s7deq_intervene_P3.json:15-21 |
| 주 게이트 2 (미수렴 r* < 5e-3) | 53 / 54 (98.15%) | >= 95% (52/54); 실패선 < 50% (27/54) | 53 >= 52 (True) | raw_s7/s7deq_intervene_P3.log:92, results_s7/s7deq_intervene_P3.json:22-30 |
| 주 게이트 3 (대조 20 cos 최솟값) | 0.93263088 | 최솟값 >= 0.9990; 실패선 < 0.9950 | 0.93263088 < 0.9950 (False) | raw_s7/s7deq_intervene_P3.log:93, results_s7/s7deq_intervene_P3.json:31-62 |
| 대조 20 cos >= 0.9990 문항 수 | 10 / 20 (50.00%) | - | - | results_s7/s7deq_intervene_P3.json:35-58 |
| 대조 20 cos in [0.9950, 0.9990) 문항 수 | 2 / 20 (10.00%) | - | - | results_s7/s7deq_intervene_P3.json:35-58 |
| 대조 20 cos < 0.9950 문항 수 | 8 / 20 (40.00%) | - | - | results_s7/s7deq_intervene_P3.json:35-58 |

## 3. 의무 진단 결과표

### 3.1. 자기 일관(Self-Consistency) 및 S5 top-3 일치 비율

| 진단 항목 | 대상 그룹 | 관측치 | 비율 | 포인터 |
|---|---|---|---|---|
| 자기 일관 (top3_star == m_frozen) | 미수렴 54 | 2 / 54 | 3.70% | raw_s7/s7deq_intervene_P3.log:94, results_s7/s7deq_intervene_P3.json:65-70 |
| 자기 일관 (top3_star == m_frozen) | 대조 20 | 11 / 20 | 55.00% | results_s7/s7deq_intervene_P3.json:2476-3338 |
| S5 z* top-3 일치 (m_frozen == S5_top3) | 미수렴 54 | 30 / 54 | 55.56% | raw_s7/s7deq_intervene_P3.log:95, results_s7/s7deq_intervene_P3.json:71-76 |
| S5 z* top-3 일치 (m_frozen == S5_top3) | 대조 20 | 11 / 20 | 55.00% | results_s7/s7deq_intervene_P3.json:2476-3338 |

### 3.2. k=300 -> 301 잔차 변화 및 Anderson 가속 사용 집계

| 측정 항목 | 측정값 | 포인터 |
|---|---|---|
| 미수렴 54문항 r_300 평균 | 1.515440e-02 | results_s7/s7deq_intervene_P3.json:82-2475 |
| 미수렴 54문항 r_301 평균 | 1.434998e-02 | results_s7/s7deq_intervene_P3.json:82-2475 |
| 미수렴 54 중 r_301 > r_300 (충격 증가) 문항 수 | 9 / 54 (16.67%) | results_s7/s7deq_intervene_P3.json:82-2475 |
| 미수렴 54 중 r_301 <= r_300 문항 수 | 45 / 54 (83.33%) | results_s7/s7deq_intervene_P3.json:82-2475 |
| Anderson 가속 진입 (k >= 1001, anderson_nstep > 0) 문항 수 | 8 / 74 (미수렴 7, 대조 1) | raw_s7/s7deq_intervene_P3.log:14,17,20,38,40,47,60,66 |
| r_1000 >= 1e-3 이나 Anderson 미진입 문항 수 | 0 / 74 | results_s7/s7deq_intervene_P3.json |

#### Anderson 가속 적용 8개 문항 상세

| 문항 번호 | 그룹 | r_300 | r_301 | r_1000 | anderson_nstep | k* | r* | 로그 포인터 |
|---|---|---|---|---|---|---|---|---|
| Q#0424 | nonconv | 2.2889e-02 | 2.2943e-02 | 2.9734e-03 | 50 | 1041 | 2.8757e-03 | raw_s7/s7deq_intervene_P3.log:14 |
| Q#0619 | nonconv | 7.8457e-03 | 7.8101e-03 | 1.2584e-03 | 13 | 1013 | 9.8496e-04 | raw_s7/s7deq_intervene_P3.log:17 |
| Q#0865 | nonconv | 9.9803e-03 | 1.0052e-02 | 1.4540e-02 | 50 | 1049 | 7.1920e-03 | raw_s7/s7deq_intervene_P3.log:20 |
| Q#2392 | nonconv | 1.1478e-02 | 1.1423e-02 | 1.3473e-03 | 14 | 819 | 6.7771e-04 | raw_s7/s7deq_intervene_P3.log:38 |
| Q#2536 | nonconv | 8.0595e-03 | 8.0699e-03 | 1.0084e-03 | 2 | 1002 | 9.5927e-04 | raw_s7/s7deq_intervene_P3.log:40 |
| Q#2877 | nonconv | 2.0686e-02 | 1.8147e-02 | 1.0924e-03 | 23 | 718 | 4.0286e-04 | raw_s7/s7deq_intervene_P3.log:47 |
| Q#4585 | nonconv | 6.3485e-03 | 6.3204e-03 | 1.0503e-03 | 3 | 1003 | 7.8374e-04 | raw_s7/s7deq_intervene_P3.log:60 |
| Q#0942 | control | 7.3006e-03 | 7.2798e-03 | 1.0116e-03 | 4 | 1004 | 9.8270e-04 | raw_s7/s7deq_intervene_P3.log:66 |

### 3.3. 미수렴 54 중 스위치 0회 문항 (Q#0865) 단독 수치

| 측정 항목 | 관측치 | 포인터 |
|---|---|---|
| 대상 문항 | Q#0865 | raw_s7/s7deq_intervene_P3.log:20, results_s7/s7deq_intervene_P3.json:77-80 |
| 301~1000 동적 스텝 스위치 횟수 (P2 스크리닝) | 0 / 700 (0.00%) | results_s7/s6_nonconv_54_switches.json:1189 |
| m_frozen 모듈 | [14, 15, 18] | results_s7/s7deq_intervene_P3.json:522 |
| r_300 | 9.980339e-03 | results_s7/s7deq_intervene_P3.json:514 |
| r_301 | 1.005209e-02 | results_s7/s7deq_intervene_P3.json:515 |
| r_1000 (Picard 1000걸음) | 1.454035e-02 | results_s7/s7deq_intervene_P3.json:516 |
| anderson_nstep | 50 (최대 반복 완주) | results_s7/s7deq_intervene_P3.json:518 |
| 최소 잔차 달성 걸음 k* | 1049 | results_s7/s7deq_intervene_P3.json:519 |
| 최소 잔차 r* | 7.192030e-03 | results_s7/s7deq_intervene_P3.json:520 |
| 자기 일관 (top3_star == m_frozen) | True ([14, 15, 18] == [14, 15, 18]) | results_s7/s7deq_intervene_P3.json:524 |
| S5 top-3 일치 여부 | True ([14, 15, 18] == [14, 15, 18]) | results_s7/s7deq_intervene_P3.json:526 |

### 3.4. 대표 문항 잔차 궤적 표 (단계별 r)

| 단계 / 걸음 (k) | Q#2249 (P2 최다 스위치) | Q#0619 (Anderson 13회) | Q#0865 (스위치 0회) | Q#0249 (대조군 대표) | 포인터 |
|---|---|---|---|---|---|
| k=1 (초기) | 3.8415e-01 | 3.5995e-01 | 3.1623e-01 | 3.3165e-01 | results_s7/s7deq_intervene_P3.json |
| k=100 (동적) | 2.3543e-02 | 3.5772e-02 | 3.1039e-02 | 1.4326e-02 | results_s7/s7deq_intervene_P3.json |
| k=200 (동적) | 1.5943e-02 | 2.4493e-02 | 2.7858e-02 | 5.8634e-03 | results_s7/s7deq_intervene_P3.json |
| k=300 (동적 최종) | 1.0498e-02 | 7.8457e-03 | 9.9803e-03 | 6.6188e-03 | results_s7/s7deq_intervene_P3.json |
| k=301 (동결 직후) | 8.7847e-03 | 7.8101e-03 | 1.0052e-02 | 6.6186e-03 | results_s7/s7deq_intervene_P3.json |
| k=500 (동결 경로) | 2.7932e-04 | 1.0218e-02 | 1.4128e-02 | 4.9065e-03 | results_s7/s7deq_intervene_P3.json |
| k=750 (동결 경로) | 1.0189e-05 | 5.3513e-03 | 1.3436e-02 | 1.5918e-03 | results_s7/s7deq_intervene_P3.json |
| k=1000 (동결 최종) | 4.5337e-07 | 9.0710e-03 | 1.4507e-02 | 5.9837e-04 | results_s7/s7deq_intervene_P3.json |
| k* (최소점) | 1000 | 1013 | 1049 | 1000 | results_s7/s7deq_intervene_P3.json |
| r* (최소 잔차) | 4.5337e-07 | 9.8496e-04 | 7.1920e-03 | 5.9837e-04 | results_s7/s7deq_intervene_P3.json |

### 3.5. 대조군 20문항 코사인 유사도 전수 표

| 문항 번호 | r_300 | r* | m_frozen | SC | S5_match | cos(z*_new, z*_old) | >= 0.9990 | 로그 포인터 |
|---|---|---|---|---|---|---|---|---|
| Q#0249 | 6.6188e-03 | 5.9837e-04 | [2, 12, 16] | True | True | 0.99984231 | True | raw_s7/s7deq_intervene_P3.log:62 |
| Q#0395 | 5.4022e-03 | 1.6470e-04 | [9, 13, 15] | True | True | 0.99998729 | True | raw_s7/s7deq_intervene_P3.log:63 |
| Q#0622 | 4.1896e-03 | 2.2312e-07 | [9, 13, 15] | False | False | 0.99435905 | False | raw_s7/s7deq_intervene_P3.log:64 |
| Q#0822 | 7.6950e-03 | 1.1545e-05 | [9, 13, 15] | True | True | 0.99999996 | True | raw_s7/s7deq_intervene_P3.log:65 |
| Q#0942 | 7.3006e-03 | 9.8270e-04 | [9, 13, 15] | True | True | 0.99800866 | False | raw_s7/s7deq_intervene_P3.log:66 |
| Q#0954 | 1.1219e-02 | 7.0655e-05 | [9, 13, 15] | False | False | 0.98462366 | False | raw_s7/s7deq_intervene_P3.log:67 |
| Q#1430 | 8.3426e-03 | 5.6612e-04 | [9, 13, 15] | False | False | 0.95692793 | False | raw_s7/s7deq_intervene_P3.log:68 |
| Q#2060 | 1.0024e-02 | 9.2533e-05 | [9, 13, 15] | False | False | 0.93263088 | False | raw_s7/s7deq_intervene_P3.log:69 |
| Q#2884 | 7.6921e-03 | 4.5375e-05 | [9, 13, 15] | False | False | 0.94523535 | False | raw_s7/s7deq_intervene_P3.log:70 |
| Q#2933 | 1.2398e-02 | 5.6334e-05 | [9, 13, 15] | False | False | 0.98098183 | False | raw_s7/s7deq_intervene_P3.log:71 |
| Q#3184 | 1.1586e-02 | 9.2588e-06 | [9, 13, 15] | False | False | 0.99058950 | False | raw_s7/s7deq_intervene_P3.log:72 |
| Q#3634 | 1.0408e-02 | 3.5965e-05 | [9, 13, 15] | True | True | 0.99999981 | True | raw_s7/s7deq_intervene_P3.log:73 |
| Q#3656 | 7.1407e-03 | 7.8047e-06 | [9, 13, 15] | True | True | 0.99999999 | True | raw_s7/s7deq_intervene_P3.log:74 |
| Q#3802 | 1.0171e-02 | 2.4148e-05 | [9, 13, 15] | True | True | 0.99999983 | True | raw_s7/s7deq_intervene_P3.log:75 |
| Q#3953 | 7.1604e-03 | 2.4284e-05 | [9, 13, 15] | True | True | 0.99999988 | True | raw_s7/s7deq_intervene_P3.log:76 |
| Q#3976 | 3.1290e-03 | 2.2198e-07 | [9, 13, 15] | True | True | 1.00000000 | True | raw_s7/s7deq_intervene_P3.log:77 |
| Q#4208 | 9.3377e-03 | 4.3801e-07 | [9, 13, 15] | False | False | 0.99522162 | False | raw_s7/s7deq_intervene_P3.log:78 |
| Q#4404 | 8.3834e-03 | 9.6141e-05 | [9, 13, 15] | True | True | 0.99999844 | True | raw_s7/s7deq_intervene_P3.log:79 |
| Q#4722 | 9.6118e-03 | 1.3360e-04 | [9, 13, 15] | True | True | 0.99999347 | True | raw_s7/s7deq_intervene_P3.log:80 |
| Q#4773 | 1.2572e-02 | 3.4723e-04 | [9, 13, 15] | False | False | 0.98954969 | False | raw_s7/s7deq_intervene_P3.log:81 |

### 3.6. 의무 진단 ④ P2 결과 포인터 및 Q#2249 m_frozen 정합성

| 점검 항목 | 관측치 | 포인터 |
|---|---|---|
| P2 대상 문항 | Q#2249 | raw_s7/freeze_discrim_P2.log:18 |
| P2 동적 궤적 스위치 횟수 (301~1000) | 686 / 700 (98.00%) | raw_s7/screen_nonconv_switches.log:41 |
| P2 Check 3 비트 동등성 max|Δ| (10개 걸음) | 0.0000000000e+00 (10/10) | raw_s7/freeze_discrim_P2.log:35-44 |
| P2 Check 4 양성 대조 max|Δ| (10개 걸음) | 2.8726e-01 ~ 2.9559e-01 (10/10 > 0) | raw_s7/freeze_discrim_P2.log:35-44 |
| P2 본 검증 m_frozen 모듈 (CPU) | (13, 15, 18) | raw_s7/freeze_discrim_P2.log:20 |
| P3 계산 m_frozen 모듈 (GPU 0) | [13, 15, 18] | results_s7/s7deq_intervene_P3.json:1330 |
| P2 본 검증과 P3 간 m_frozen 일치 여부 | True ((13, 15, 18) == [13, 15, 18]) | raw_s7/freeze_discrim_P2.log:20, results_s7/s7deq_intervene_P3.json:1330 |
| P2 스크리닝 m300과 차이 사유 | 스크리닝 m300은 step 300 입력 z_299 기준 [13, 14, 15]; P2 본 검증 및 P3는 step 300 종료 시점 z_300 기준 (13, 15, 18) | raw_s7/freeze_discrim_P2.log:19-20 |

## 4. Anderson 솔버 적용 연산자 및 잔차 정의 (코드 근거)

- 코드 파일: `scripts/s7deq_intervene_p3.py` (SHA-256: `ba364b1d122ff74d23f05de7ab586f16d35fee150da28383bb60c2dba83595e8`)
- Anderson 솔버 대상 연산자:
  - 감쇠 Picard 연산자 $T(z) = (1-\alpha)z + \alpha f_{\text{frozen}}(z)$가 아니며, 비감쇠 고정점 연산자 $f_{\text{frozen}}(z) = \Phi(z; m_{\text{frozen}}) + g_0 \hat{c}_0$에 직접 적용함.
  - 근거 코드 행 (`scripts/s7deq_intervene_p3.py` 53-73행):
    ```python
    def f_frozen(self, z_cur: torch.Tensor) -> torch.Tensor:
        ... (w_g routing on m_fixed, deq_step call) ...
        inj = self.base.const_inj.to(device=device, dtype=torch.float32)
        return phi_z + inj
    ```
  - Anderson 호출 코드 행 (`scripts/s7deq_intervene_p3.py` 153-174행):
    ```python
    def anderson_f(x):
        inj = frozen_step_fn.f_frozen(x)
        diff = (inj - x).double()
        norm_x = torch.norm(x.double())
        true_r = float((torch.norm(diff) / norm_x).item())
        ... (기록) ...
        return inj
    res_anderson = anderson_solver(
        f=anderson_f, x0=z_1000.clone(), m=5, lam=1e-4, beta=1.0, max_iter=50, tol=1e-3
    )
    ```
- 잔차 계산 규약 준수:
  - Anderson 내부의 $\|f\|$ 분모(`abs_res / norm_f`) 사용 배제.
  - float64 정밀도로 $\|f(z) - z\|_2 / \|z\|_2$를 재계산하여 $r$을 평가함.

## 5. 생성 및 변경 파일 SHA-256

ba364b1d122ff74d23f05de7ab586f16d35fee150da28383bb60c2dba83595e8  scripts/s7deq_intervene_p3.py
c13d33354715a66e4924053f37627be9e25dcb3ee4b4642966ccc96b014365b4  raw_s7/s7deq_intervene_P3.log
0373d48877a9654111b68ea02fc1c60dcd511fe4d683953805949be357683ee3  results_s7/s7deq_intervene_P3.pt
626591dcc8050a39d98f53900d40e2a238155979aa1321690537a34ac86b983c  results_s7/s7deq_intervene_P3.json
06c93011058b9dfa83e1846636fa05ae65a117b49d9b35045ba8a59e2a49d7e3  exe_scratch/s7/ref_solvers/anderson.py

## 6. 예상 밖 사건 (Incidents)
- Gate 3 기준 미달: 대조 20문항 중 10문항에서 $\cos(z^*_{\text{new}}, z^*_{\text{old}}) < 0.9990$ (그 중 8문항은 $< 0.9950$, 최솟값 0.93263088). 해당 8문항은 모두 자기 일관성(SC) 및 S5 top-3 일치 여부가 False인 문항임.

## 7. 서버 상태 요약
- Quadro-Ampere-05: GPU 0~7 연산 프로세스 0개, GPU 0 점유율 0% (메모리 24 MiB), CPU 로드 정상, 잔여 백그라운드 연산 0개.

430ceffb5ab5da376c821e5ead1c2be9b8e8f1fdb77c2d50b740b0efb64641bd
# REPORT-CTS-P4-PRIME.RUNNING — P4' 디코더 직렬 재학습(A·D·E) 착수 및 가동 보고서

## 1. 개요 및 실행 환경
- **지시서**: System 2 결정 / 원장 #413 & #414 (P4' 디코더 직렬 재학습)
- **실행자**: 회사 PC Antigravity (단독 정본 실행 주체)
- **원격 호스트**: `shoon@<GPU_SERVER_IP>:<SSH_PORT>` (`~/cts_e1_s5`)
- **실행 장치**: GPU 0 (NVIDIA RTX A6000 48GB), 단독 직렬 실행 (GPU 4 접근 0건)
- **tmux 세션**: `s7_train_p4_prime`
- **실행 파이프라인**: `scripts/s7_train_p4_prime_all.sh`
- **실행 스크립트**: `scripts/s7_train_decoder_v35.py` (v3.5 §3 규약 준수 구현)
- **규약 패치**: `shared/reports/DIFF-P4-PRIME.patch`
- **판정어 통제**: 0건

---

## 2. 학습 대상 집합 및 입력 규약 검증
- **학습 대상 문항 파일**: `configs/s7_train_993.json` (총 993문항)
- **문항 집합 정렬 SHA-256**: `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c`
- **제외 문항 (7건, 정답 절단 방지 필터)**: Q#0074, Q#0191, Q#0544, Q#0629, Q#2727, Q#3055, Q#4208
- **입력 양식**: PREREG_S7_D_v3_5 §3 (`<bos><|turn>user\n[P(z) 64 slots]\n\n{problem_text}<turn|>\n<|turn>model\n`)
- **슬롯 물리 배치**: 토큰 인덱스 [4..67]에 64개 슬롯 고정 배치, 토큰 68번 개행(`\n\n`, 토큰 ID 108), 69번부터 문제 본문 배치

### 2.1. 착수 전 1문항 (Q#4298) 입력 디코드 원문 대조
```text
<bos><|turn>user
1111111111111111111111111111111111111111111111111111111111111111

If $x^5 - x^4 + x^3 - px^2 + qx + 4$ is divisible by $(x + 2)(x - 1),$ find the ordered pair $(p,q).$<turn|>
<|turn>model
```
- **대조 계측치**:
  - 슬롯 수: 64개 (토큰 ID 236770)
  - 슬롯 직후 분리자: `\n\n` (토큰 ID 108)
  - 문제 본문 포함: 문자열 일치 (길이 103자 보존)

---

## 3. 사전 스모크 테스트 (2스텝) 계측치
- **실행 명령**: `python scripts/s7_train_decoder_v35.py --loss_lambda 1.0 --z_type zstar --max_steps 2 --out results_s7/test_smoke_v35.pt`
- **체크포인트 생성**: `results_s7/test_smoke_v35.pt` (236,027,245 바이트, SHA: `a2058ab9...`)
- **스텝별 계측 지표**:
  - Step 1: loss = 6.6059, l_task = 5.6215, l_align = 0.9845, NLL_m_f32 = 5.6215, NLL_s_f32 = 5.6370, Delta_f32 = +0.0155, 소요 시간: 8.54초/스텝
  - Step 2: loss = 5.7387, l_task = 4.7821, l_align = 0.9566, NLL_m_f32 = 4.7821, NLL_s_f32 = 4.8254, Delta_f32 = +0.0434, 소요 시간: 8.01초/스텝
- **최대 VRAM 사용량**: 30.60 GiB (가용 48.0 GiB 중 63.8% 점유)
- **종료 코드**: 0

---

## 4. 가동 중인 모델 및 파이프라인 명세

| 순서 | 모델명 | 조건 | 입력 텐서 | 목표 스텝 | 산출 체크포인트 | 산출 로그 | 현재 상태 |
|:---:|:---|:---:|:---|:---:|:---|:---|:---:|
| 1 | Decoder A | $\lambda=1.0$ | $z^*$ (`results_s7/zstar_train1000_P4.pt`) | 3,000 | `results_s7/decoder_A_v35.pt` | `raw_s7/train_decoder_A_v35.log` | **가동 중 (PID 3793541)** |
| 2 | Decoder D | $\lambda=0.0$ | $z^*$ (`results_s7/zstar_train1000_P4.pt`) | 3,000 | `results_s7/decoder_D_v35.pt` | `raw_s7/train_decoder_D_v35.log` | 대기 중 |
| 3 | Decoder E | $\lambda=1.0$ | $z_{\text{src}}$ (`results_s7/zsrc_train1000_P4.pt`) | 3,000 | `results_s7/decoder_E_v35.pt` | `raw_s7/train_decoder_E_v35.log` | 대기 중 |

### 4.1. 소요 시간 추정치
- 스모크 스텝당 속도: 약 8.0 ~ 8.5초/스텝
- 모델당 3,000스텝 소요 예상: 약 24,000 ~ 25,500초 (약 6.7 ~ 7.1시간)
- 3개 디코더(A·D·E) 총 소요 예상: 약 20.0 ~ 21.5시간
- 첫 100걸음 실측 속도 측정 후 예상 종료 시각을 `STATE.md`에 등재 예정 (원장 #414 §3)
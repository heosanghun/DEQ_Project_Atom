# PREREG_S7_G2v3 — G2 3차 시도 (M3 / N8) DEQ+디코더 공동 학습 사전등록 (잠금본)

- **상태**: **LOCKED (2026-10-10 21:15 KST)** — TASK-0071 파일럿 계측(REPORT-0071, 1행 SHA `024f28726320d028f46c2f2ca567704a27578395a865af2465e0de96f7fba0ae`) 결과에 기반하여 작성 및 잠금. 잠금 이후 변경은 diff 및 감사 절차를 거친다.
- **발행**: 회사 PC 정식 루프 자율 에이전트(AG), 2026-10-10
- **시도 번호**: G2 **3차 시도**.
  - 1차 시도: `PREREG_S7_D_v3_8` (P6R, 정본 $z^*$ 정보 손실로 게이트 4 $-4.83\%$p 미충족).
  - 2차 시도: `PREREG_S7_G2v2` (M2′ 주입 교체, $z_{\text{src}}$ 보존으로 디코더 성능 향상되었으나 $A' \approx E'$로 게이트 4 미충족).
  - 1차·2차 판정 결과는 보존되며 불변이다.
- **근거**: Principal 상위 지침 ("앞으로는 대기하지말고, 즉시, 진행을 해주세요!", "앞으로는 자율적으로 전문가인 당신이 자율적으로 판단하여, 다음 작업을 진행하여, 계속 작업을 이어가서 최종목표를 완료해주세요!", CTS-430), GRAPH.md Node N8 (S8_IFT 전환), `shared/proposals/G2v3_PLAN_20261009.md` (방법 M3), `shared/reports/REPORT-0071.md`.

---

## 0. 공개(작성 전에 본 것)와 보지 않은 것

- **본 것**:
  1. 1차 P6R 및 2차 G2v2 전체 평가 결과 및 원인 진단(REPORT-0070).
  2. TASK-0071 파일럿 결과(REPORT-0071: 32문항 50스텝 Mode A 및 Mode E 병렬 실행 결과, 손실 곡선, 스텝당 평균 0.81초, 피크 VRAM 34.3 GiB, IFT 잔차 $4.48 \times 10^{-5}$ 안정성).
- **보지 않은 것**:
  1. 전체 1,000문항으로 본 훈련된 M3 가중치 및 디코더의 생성 결과.
  2. 확증 평가용 미개봉 새 1,000문항(`configs/g2v2_fresh1000.json`, SHA-256 `35683874284b371c6641e1280b92e6a69292a2dc5d89585158154a45b41cb3db`)의 문제 본문·풀이·잠재 텐서 및 생성 결과는 일체 개봉하거나 평가한 적이 없다.

---

## 1. 바뀌는 것 (주장 범위)

1·2차 대비 핵심 변경 사항은 다음과 같다:
1. **DEQ 사상 $\Phi_\theta$와 디코더의 공동 역전파 (방법 M3)**:
   - DEQ 전이 함수 $\Phi_\theta$(LoRA rank 8, `routing_proj`, $\gamma$)와 `PrefixProjector`를 고정점 환영 기울기(phantom gradient)를 통해 정답 해설 언어모델 교차 엔트로피 손실($\lambda=0$)로 **동시에** 훈련한다.
   - 고정점 수렴 안정성 유지를 위해 보조 IFT 잔차 손실 $l_{\text{ift}} = \|z_{\text{next}} - z^*\|^2 / \|z^*\|^2$ ($\lambda_{\text{ift}} = 0.1$)을 병행 적용한다.
2. **동일 용량 대조군 $E'''$ 도입**:
   - $E'''$ 역시 동일한 학습 가능 매개변수 용량(동일 $\Phi_\theta$ 구조 및 PrefixProjector)을 갖고, $z_{\text{src}} \to \Phi_\theta(z_{\text{src}}) \to \text{디코더}$의 1단계 사상으로 동일 스텝 동안 훈련된다.
   - 이를 통해 DEQ 반복(고정점 탐색)이 단순 1단계 사상 대비 진정한 추론 능력 향상을 제공하는지 엄밀히 검증한다 (게이트 4).

---

## 2. 평가 문항군 및 데이터 분할

1. **훈련 집합**:
   - `configs/s7_train_993.json` (SHA-256 `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c`, 993문항).
   - 입력 텐서: `results_s7/g2v2/zstar_zsrcinj_train1000.pt` (M2′ $z^*$) 및 `results_s7/zsrc_train1000_P4.pt` ($z_{\text{src}}$).
2. **개발 집합 (선택 규칙 확인용)**:
   - `configs/g2v2_dev600.json` (SHA-256 `60b501ec99c6385e4ff03f153840eb4748525f51735b5e9966f1bd7962eb3aaf`, 600문항).
   - 입력 텐서: `results_s7/g2v2/zstar_zsrcinj_dev600.pt` 및 `results_s7/zsrc_heldout600.pt`.
3. **확증 집합 (최종 판정용)**:
   - 미개봉 정본 `configs/g2v2_fresh1000.json` (SHA-256 `35683874284b371c6641e1280b92e6a69292a2dc5d89585158154a45b41cb3db`, 1,000문항).
   - 3개 샤드로 분할 실행:
     - Shard 0: `configs/g2v2_fresh1000_shard0.json` (SHA `f4417d21de92d6a0b3c98d5fa850217f977a4f475405ac6472954cd9388b075d`)
     - Shard 1: `configs/g2v2_fresh1000_shard1.json` (SHA `fee4f45a92bd8e8690cceaf8cd29d57a5daf53d5230d01aefaa59818206a976f`)
     - Shard 2: `configs/g2v2_fresh1000_shard2.json` (SHA `a256039bfcc05b1f89c9e94d669abdd2617eb3aaac04a095bf4936c1ad6a2dc1`)

---

## 3. 훈련 및 개발 선택 규칙

### 3.1 훈련 규격
- 스크립트: `scripts/g2v3_m3_full_train.py`
- 파라미터:
  - 총 최적화 스텝: 1,500 steps
  - 배치 크기: 4 (기울기 누적 마이크로 배치)
  - 학습률: Projector $1 \times 10^{-4}$, Adapter $3 \times 10^{-4}$, DEQ Backbone $3 \times 10^{-5}$
  - 가중치 감쇠: Projector/Adapter 0.01, DEQ 0.001
  - 보조 IFT 가중치: Mode A $\lambda_{\text{ift}} = 0.1$, Mode E $\lambda_{\text{ift}} = 0.0$
- 하드웨어 배정:
  - Mode A (M3): GPU 5 (`GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a`)
  - Mode E (E'''): GPU 6 (`GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c`)
  - 출력 경로:
    - Mode A: `results_s7/g2v3/decoder_A_v36.pt` 및 `results_s7/g2v3/deq_backbone_A.pt`
    - Mode E: `results_s7/g2v3/decoder_E_v36.pt` 및 `results_s7/g2v3/deq_backbone_E.pt`

### 3.2 개발 평가 및 사전 선택 규칙
- 개발 600문항(`configs/g2v2_dev600.json`)에 대해 생성 및 채점 실행:
  - 조건 A (Mode A 훈련 모델): `results_s7/p6r/g2v3dev_gen_A.jsonl`
  - 조건 E (Mode E 훈련 모델): `results_s7/p6r/g2v3dev_gen_E.jsonl`
  - 조건 C (동결 순수 젬마): 기존 정본 `results_s7/p6r/p6_gen_C.jsonl` 재사용 (비오염 보존)
- **사전 선택 규칙 (사전등록 고정)**:
  - $\text{acc}(A) - \text{acc}(E''') > 0$ **그리고** $\text{acc}(A) - \text{acc}(C) > 0$
  - 위 조건이 충족되면 즉시 제4절 확증 평가(새 1,000문항)로 진입한다.
  - 미충족 시 새 문항을 개봉하지 않고 원인을 재분석한다.

---

## 4. 확증 평가 및 최종 판정 기준 (G2 PASS)

### 4.1 확증 평가 절차
1. **잠재 텐서 산출**:
   - 새 1,000문항에 대해 훈련된 DEQ 사상 $\Phi_{\theta, A}$를 적용하여 잠재 텐서 `results_s7/g2v3/fresh_zstar_A.pt` 및 `results_s7/g2v3/fresh_zsrc_E.pt` 생성.
2. **생성 실행**:
   - 생성기: `scripts/s7_eval_gate_p6_v36e.py`
   - 정지 토큰: EOS_IDS = (1, 106) 상수 고정
   - 샘플링: Greedy (`temperature=0.0`), `max_new_tokens=512`
   - 조건 A, E, C, B1(보조)에 대해 3개 GPU(5, 6, 7)에서 샤드별 병렬 추론.
3. **병합 및 검속**:
   - 4개 조건 각 1,000행, 1,000 고유 ID, JSON 오류 0건 실측 후 단일 채점 집행.
4. **공식 채점 집행**:
   - 채점기: `scripts/s5_eval_grader_v2.py` (SHA `90cc93e4...`) 및 `scripts/s7_eval_score_p6_v36.py` (SHA `547e1d91...`).
   - 통계 검정: 대응 표본 부트스트랩 10,000회, 시드 2026, 정확 McNemar 단측 검정.

### 4.2 최종 판정 기준 (GOAL.md G2 PASS)
최종 판정은 다음 3대 조건을 동시 충족하는 것으로 정의한다:
$$\text{G2 PASS} \iff \text{Gate 2} \land \text{Gate 3} \land \text{Gate 4}$$

1. **Gate 2 (기본 정확도 보존)**:
   $$\text{acc}(A) - \text{acc}(C) \ge -2.00\%p$$
2. **Gate 3 (순수 언어모델 대비 개선)**:
   $$\text{acc}(A) - \text{acc}(C) > 0, \quad p < 0.025 \quad \text{and} \quad 95\% \text{ CI 하한} > 0$$
3. **Gate 4 (대조군 $E'''$ 대비 고정점 우위)**:
   $$\text{acc}(A) - \text{acc}(E''') > 0, \quad p < 0.025 \quad \text{and} \quad 95\% \text{ CI 하한} > 0$$
4. **보조 Gate 1**: $\text{acc}(A) - \text{acc}(B_1)$ 계측 및 보고.

---

## 5. 갈림길 차단 규약

1. 판정 결과 확인 후 하이퍼파라미터 사후 조정, 프롬프트 변경, 채점기 변경은 절대 금지한다.
2. 새 1,000문항 평가 결과는 1회 확정으로 유효하며, 어떠한 선택적 재실행도 불허한다.
3. 하드웨어 또는 통신 오류 발생 시에만 사후 감사를 거쳐 동일 조건으로 재실행한다.

---

## 6. 고정 파일 및 SHA-256 인벤토리

| 파일 경로 | SHA-256 해시 | 설명 |
| :--- | :--- | :--- |
| `configs/s7_train_993.json` | `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c` | 훈련 993문항 목록 |
| `configs/g2v2_dev600.json` | `60b501ec99c6385e4ff03f153840eb4748525f51735b5e9966f1bd7962eb3aaf` | 개발 600문항 목록 |
| `configs/g2v2_fresh1000.json` | `35683874284b371c6641e1280b92e6a69292a2dc5d89585158154a45b41cb3db` | 미개봉 새 1,000문항 |
| `configs/g2v2_fresh1000_shard0.json` | `f4417d21de92d6a0b3c98d5fa850217f977a4f475405ac6472954cd9388b075d` | 새 문항 Shard 0 |
| `configs/g2v2_fresh1000_shard1.json` | `fee4f45a92bd8e8690cceaf8cd29d57a5daf53d5230d01aefaa59818206a976f` | 새 문항 Shard 1 |
| `configs/g2v2_fresh1000_shard2.json` | `a256039bfcc05b1f89c9e94d669abdd2617eb3aaac04a095bf4936c1ad6a2dc1` | 새 문항 Shard 2 |
| `results_s7/g2v2/zstar_zsrcinj_train1000.pt` | `005e622d50ec5efdfa309a6dd8e4663880194f9c10e13c276573e636050aee8a` | 훈련 입력 $z^*$ |
| `results_s7/zsrc_train1000_P4.pt` | `340ef5c2bfb718fc371c932746476555d8d5a2bcd54c7413acc40e35840e45a7` | 훈련 입력 $z_{\text{src}}$ |
| `results_s7/g2v2/zstar_zsrcinj_dev600.pt` | `d43613d6b640b965c9c8a3d6ad422d7a8492fe0b0c2eed9a379064495c200433` | 개발 입력 $z^*$ |
| `results_s7/zsrc_heldout600.pt` | `0da3748af2a3776c7fa91da50268a7829233939f56218d73f4b4511a6a8e5cff` | 개발 입력 $z_{\text{src}}$ |
| `results_s7/p6r/p6_gen_C.jsonl` | `24c82d97ff0a786dbf135b4673273fedf79bec371d9868eb98e6d2f541bbee77` | 기준 조건 C 생성본 |
| `scripts/s7_eval_grader_v2.py` | `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13` | 채점기 정본 |
| `scripts/s7_eval_score_p6_v36.py` | `547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd` | 부트스트랩 채점 스크립트 |
| `scripts/s7_eval_gate_p6_v36e.py` | `a2755c1b591e12389c8efe167aa33a59a7ae6c0876234585381b6a129faa460a` | 게이트 생성기 |
| `scripts/check_p6_decoder_version.sh` | `1fa95f72d5e5cea2ad1836dd458544190467f0cfb1daa14e41f6459cb51d870b` | 버전 검속 가드 |

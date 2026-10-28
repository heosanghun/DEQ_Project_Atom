<!-- SHA256: 88779da90ca044f3f7ade093af328131cc7366854ad45d10fdcfcbf73374d081 -->
# REPORT-0072: G2 3차 시도 (M3 / N8) 본 훈련(1,500스텝) 및 Dev 600 선택 평가 결과 보고서

- **발행 일시**: 2026-10-11 02:30 KST
- **수행자**: 정식 루프 시스템 (`cts`, Windows / Ubuntu Server 원격 파이프라인)
- **근거 규정**: TASK-0072, TASK-0074, PREREG_S7_G2v3_1 (SHA `3d1a6ee0...`, AUDIT-P013 재감사 2), CTS-430 (Principal 지침: "G2 완수가 최종 목표, GPU 2·3·5·6·7 인가, GPU 4 영구 봉인"), GRAPH.md Node N8
- **실험 성격**: M3 DEQ 백본(LoRA rank 8, `routing_proj`, $\\gamma$) 및 PrefixProjector 공동 역전파 1,500스텝 본 훈련 수치 계측, 초기 결함 조치 내역, 및 개발 600문항 대상 선택 규칙(Gate 3, Gate 4) 평가 결과 수록.

---

## 1. 산출물 및 참조 파일 SHA-256 인벤토리

| 파일 구분 | 파일 경로 | SHA-256 해시 | 비고 |
| :--- | :--- | :--- | :--- |
| **사전등록 정본** | `shared/prereg/PREREG_S7_G2v3_1.md` | `3d1a6ee0292b58fcab514ec2b71588dc78a277f6c464e61739ebb4ac96573daa` | 잠금 정본 (AUDIT-P013) |
| **훈련 스크립트** | `scripts/g2v3_m3_full_train.py` | `36e0f18addc8d46f6d8bf376cd95e89cf9eb75a1f15c4e8b714141efe5a49c99` | 1,500스텝 본 훈련 정본 |
| **잠재 변환기 (초기)** | `scripts/g2v3_m3_eval_latents.py` | `782f8dcc0c827446a671a85a992ff973bca292890226cf2fe894bbf6dc08ae6f` | 접미사 결함 내장본 |
| **잠재 변환기 (개정)** | `scripts/g2v3_m3_eval_latents_v2.py` | `73db9dafd33e49e963d48aad4797c6b0eb7856e5e8e847fbb3b8dc0e5109921b` | 접미사 병합 정합본 |
| **가중치: 디코더 A** | `results_s7/g2v3/decoder_A_v36.pt` | `682d05c3638e3ffadfabb6f38fd6bd3d06a3e2004964f35cbeb60be9a410cd10` | 78,687,903 bytes |
| **가중치: 디코더 E** | `results_s7/g2v3/decoder_E_v36.pt` | `ba1d2788ec4f3fb38949f17cd215b1d42721c175f3a47d2659bc4dc573722e05` | 78,687,903 bytes |
| **가중치: DEQ A** | `results_s7/g2v3/deq_backbone_A.pt` | `fec12c8faab77c31276ba4d155aaf41f32243f33212793e92f34117e80a14cbf` | 8,664,301 bytes |
| **가중치: DEQ E** | `results_s7/g2v3/deq_backbone_E.pt` | `255ed2731098fa5081c540e456383fa73276e9cb73c589f9f39d0da0210f581b` | 8,664,301 bytes |
| **통합 가중치 A** | `results_s7/g2v3/full_model_A.pt` | `2c2a271002b77192a9f75cdde98892ca054ab80dee9536751859957899028a31` | 87,474,638 bytes |
| **통합 가중치 E** | `results_s7/g2v3/full_model_E.pt` | `f3dc3e6b7b39ca55cf40c97eb989b99fd3d78c1b76f425b26f8137e54b116c6d` | 87,474,638 bytes |
| **훈련 매니페스트 A** | `results_s7/g2v3/full_manifest_A.json` | `db2da3822084c1d5e0ecdc87d9de9de2021ed6a28ddb00b8e554020fbb4b57af` | 208 bytes |
| **훈련 매니페스트 E** | `results_s7/g2v3/full_manifest_E.json` | `16f86383e05bfff18168061feaec863d8e5e1ec1d7d74f78c37568dd0d5c0ce5` | 191 bytes |
| **Dev 600 잠재 E** | `results_s7/g2v3/dev600_zsrc_E.pt` | `977b45f816a0b4efaf7cb594b6e4809ddb4127be15fe736e0a6990ed0c0ed7df` | 393,416,579 bytes |
| **Dev 600 잠재 A (v1)**| `results_s7/g2v3/dev600_zstar_A.pt` | `b5c7ac63d8b3c9a80e1ba03f531809b7ac8010858548f457ec7b95be55ca77cd` | 결함 잠재 텐서 |
| **Dev 600 잠재 A (v2)**| `results_s7/g2v3/dev600_zstar_A_v2.pt` | `372e42eea85b4f76aa8a008bbd7e04114c06378521bf543121aca2f7df9dd5a1` | 정합 잠재 텐서 |
| **Dev 생성 A (v2)** | `results_s7/p6r/g2v3dev_gen_A_v2.jsonl` | `8a2963f825b968bd6d149328fccf46283daf185f65473c19a71728cad718c1da` | 600행 600고유ID |
| **Dev 생성 E** | `results_s7/p6r/g2v3dev_gen_E.jsonl` | `ff3bbd7c65b8333ddbb14b22c93e967a3df503da829846b9218304db54a6711b` | 600행 600고유ID |
| **Dev 공식 채점표** | `results_s7/p6r/g2v3dev_score_AE_v2.json` | `e88cb0f428bd0d97a0772cd311b426c6e2120eb72bf974090c309c91ee1b043f` | 부트스트랩 채점 데이터 |
| **Dev 공식 마크다운** | `results_s7/p6r/g2v3dev_score_AE_v2.md` | `e412c9f9d4817c791653e411e40b23f2afe7e419853d1d99ac383204931e10c1` | 게이트 평가표 |

---

## 2. 하드웨어 배정 및 자원 거버넌스 준수

1. **연속 가동 상태**: 원격 호스트(`Quadro-Ampere-05`) 5일 8시간 이상 무중단 가동 유지 (재부팅 0회).
2. **GPU 배정**:
   - Mode A: GPU 5 (`GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a`)
   - Mode E: GPU 6 (`GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c`)
3. **GPU 4 봉인 규약 엄수**: 13 MiB 유휴, 활성 프로세스 0건 영구 유지.
4. **접속 규약 준수**: `shared/cts_ssh.py` 단일 묶음 호출, 60초 이상 간격 엄수.

---

## 3. 초기 결함 실행 기록 및 조치 내역

본 파이프라인 수행 과정에서 발생한 결함 내역 및 조치 과정은 아래와 같습니다:

1. **초기 기동 결함 (21:32 ~ 21:38)**:
   - 원인: 모듈 임포트 경로 불일치 및 프롬프트 슬롯 규격 에러로 3회 즉시 종료 (반환 코드 1).
   - 조치: `cts.routing.sparse_moe_ref` 및 canonical 프롬프트 함수 일치화 반영 후 21:40 규격 기동 착수.
2. **개발 600 잠재 텐서 키 불일치 결함 (22:07)**:
   - 원인: `g2v3_m3_eval_latents.py`가 입력 텐서 내 `_a0.25` 접미사 처리 분기를 단일 키 형식으로 취급하여 일부 문항 잠재 텐서 변환 누락.
   - 조치: 접미사 병합 검색 로직이 반영된 `scripts/g2v3_m3_eval_latents_v2.py` 작성 및 재연산 수행. 600개 전 문항 텐서 정합 산출물 `dev600_zstar_A_v2.pt` 획득 후 생성기 재구동(`g2v3dev_gen_A_v2.jsonl`).

---

## 4. 본 훈련 (1,500스텝) 수치 계측

- **훈련 데이터셋**: `configs/s7_train_993.json` (993문항 SHA `3088a153...`, 전수 적재 993/993)
- **수행 스텝**: 1,500스텝 (배치 4, 유효 배치 4, 누적 최적화)
- **스텝당 소요 시간**:
  - Mode A (M3): 평균 0.812초/step (총 26분 29초 소요)
  - Mode E (E'''): 평균 0.824초/step (총 26분 51초 소요)
- **피크 VRAM 점유량**: Mode A 34.32 GiB, Mode E 34.18 GiB (48 GiB 한도 내 수용)
- **손실 추이**:
  - Mode A: Step 1 (3.84) -> Step 500 (1.62) -> Step 1000 (1.18) -> Step 1500 (0.91), IFT 잔차 손실 $3.8 \times 10^{-5}$ 안정 유지
  - Mode E: Step 1 (3.79) -> Step 500 (1.58) -> Step 1000 (1.12) -> Step 1500 (0.87)

---

## 5. 개발 600문항 (Dev 600) 선택 평가 결과

부트스트랩 10,000회 공식 채점 집행 결과(`g2v3dev_score_AE_v2.json`):

| 조건 | 평가 문항수 | 엄격 정확도 (Strict) | 완화 정확도 (Lenient) |
| :--- | :--- | :--- | :--- |
| **Condition A (M3 DEQ+Projector)** | 600 | **6.50% (39/600)** | **9.50% (57/600)** |
| **Condition E1 (E''' 대조군)** | 600 | **5.67% (34/600)** | **9.00% (54/600)** |
| **Condition C (순수 Gemma)** | 600 | **0.33% (2/600)** | **7.17% (43/600)** |
| **Condition B1 (셔플 대조군)** | 600 | **1.17% (7/600)** | **5.00% (30/600)** |

### 사전등록 선택 규칙 판정

1. **Gate 4 (vs E''')**:
   - 계측값: $\Delta = \text{acc}(A) - \text{acc}(E1) = +0.00833$ (+0.83%p)
   - 95% 신뢰구간: $[-0.01667, +0.03333]$
   - McNemar 검정: $b=31, c=26$, 단측 $p = 0.2983$
   - 판정: $\Delta > 0$ 충족
2. **Gate 3 (vs Pure Gemma)**:
   - 계측값: $\Delta = \text{acc}(A) - \text{acc}(C) = +0.06167$ (+6.17%p)
   - 95% 신뢰구간: $[+0.04333, +0.08333]$
   - McNemar 검정: $b=38, c=1$, 단측 $p = 0.0000$
   - 판정: $\Delta > 0$ 및 $p < 0.05$ 대폭 충족

---

## 6. 결론

M3 공동 역전파 기법은 고정점 수치 안정성을 저해하지 않으면서 순수 Gemma 언어 모델 대비 +6.17%p의 정확도 격차를 기록하였으며, 동일 파라미터 대조군 대비 양의 우위를 기록하여 `PREREG_S7_G2v3_1`의 사전등록 선택 기준을 충족하였습니다. 이에 따라 미개봉 새 1,000문항 대상 최종 확증 평가 집행 단계로 이행합니다.

# CODEBASE_MAP.md — CTS 코드베이스 지도 (영구 인덱스)

> **[주의 — RULES R5-2]**: 본 지도는 1회 전수 분석 결과로 고정된 인덱스입니다. 에이전트는 코드베이스 전수 재분석을 수행하지 않으며, 변경된 파일에 대해서만 부분 갱신합니다.

---

## 1. 디렉토리 구조 개요
```text
D:\AI\cts\
├── cts\                        # CTS 파이썬 패키지 (정본 import 경로)
│   ├── deq\                    # DEQ 고정점 연산 및 주입 계층
│   ├── routing\                # Sparse MoE 라우팅 (W_g 투영, Top-k 소프트맥스)
│   ├── model\                  # 백본 로더 (Gemma 4 e4b, 모듈 분할)
│   ├── mcts\                   # MCTS 탐색, 하이브리드 KV 캐시, 트리 관리
│   ├── policy\                 # 메타 정책 (Meta-Policy, 액션 헤드)
│   ├── train\                  # 학습 스크립트 (Stage 1 Warmup/OpenMath, Stage 2 PPO)
│   ├── eval\                   # 벤치마크 평가 디스패처 (MATH, AIME, GSM8K 등)
│   └── utils\                  # 시드 제어, 설정 파서, 재현 스냅샷
├── shared\                     # 자율 루프 하네스 공유 디렉토리 (A-EXE-B)
│   ├── GOAL.md                 # 최종 목표 및 2대 게이트
│   ├── RULES.md                # 하네스 운영 규칙 (v1)
│   ├── GRAPH.md                # 작업 전이 상태 그래프
│   ├── LESSONS.md              # 프로젝트 함정 및 재발 방지 교훈
│   ├── STATE.md                # 상단 10행 현황판
│   ├── LEDGER.md               # 주요 의사결정 원장
│   ├── prereg\                 # 사전등록 잠금 정본 (S7_D_v3_4, S7_DEQ_v1.1)
│   ├── tasks\                  # 발행 작업 (A → EXE)
│   ├── reports\                # 실행 보고서 (EXE → B)
│   └── audits\                 # 감사 판정서 (B → A)
├── scripts\                    # 실행 및 진단 보조 스크립트
├── artifacts\                  # 정본 체크포인트 (stage1_last.pt, stage2_meta_value.pt 등)
├── configs\                    # 학습 및 실험 설정 파일
├── doc\                        # 기술 문서 및 논문 관련 분석 보고서
└── results\                    # 실험 결과 산출물 (post_stage2_*, table2 등)
```

---

## 2. 핵심 파일 및 함수 포인터

### A. 백본 로더 및 데이터셋
- `harness_A_v1.py` (서버 정본):
  - `get_model()`: Gemma 4 e4b 백본 로드 (bfloat16, eval 모드, 캐시 고정).
  - `get_dataset(split)`: OpenMath 2025 검증 세트 로드.
- `cts/model/gemma_loader.py`: Gemma 구조 로더 및 모듈 파티셔닝.

### B. DEQ 전이 및 고정점 주입 (`cts/deq/`)
- `cts/deq/transition_inject.py`:
  - `InjectedTransitionStep`: $z_{k+1} = (1-\alpha)z_k + \alpha f(z_k, c)$ 전이 스텝 구현.
  - `compute_broadcast_context_c`: 텍스트 컨텍스트 임베딩 브로드캐스트.
  - `compute_c_hat_0`: 초기 스텝 주입 텐서 계산.
- `scripts/verify_frozen_parity_cpu.py`:
  - `FrozenInjectedTransitionStep`: Step 300에서 선택된 Top-3 전문가 ID를 고정한 채 $f(z)$ 전이를 수행하는 불변 클래스.

### C. 라우팅 매트릭스 및 가중치 (`cts/routing/`)
- `cts/routing/sparse_moe_ref.py`:
  - `routing_weights(h, W_g, top_k=3)`: 토큰 표현 $h$와 게이트 가중치 $W_g$로부터 Top-3 전문가 소프트맥스 가중치 산출 정본.
- 정본 게이트 가중치:
  - `bb.routing_matrix()` / `m1_final_frozen`: 16개 전문가, $W_g \in \mathbb{R}^{d \times 16}$.
  - 수치 패리티: 인라인 라우팅과 정본 간 $\max|\Delta| = 0.0$ 보장 (원장 #378).

### D. 사전등록 문서 (정본 잠금본)
- `shared/prereg/PREREG_S7_D_v3_4.md`:
  - SHA-256: `88c308625fb10b62b29ef3426052ffd23f84ad5c856b071c272ab412f183c772`
  - 참 잔차 $r = \|f(z)-z\|/\|z\|$, S7 디코더 학습/평가 프로토콜.
- `shared/prereg/PREREG_S7_DEQ_v1.1.md`:
  - SHA-256: `bf3173b1346c0768cef5e224611d906cb033ef6ae7ac29bad52c0e7a443ad488`
  - 마스크 동결 DEQ 개입 실험 (74문항, Picard 300 $\to$ Picard 700 동결 $\to$ Anderson).

### E. 서버 실행 및 감시 스크립트
- `/home/shoon/cts_e1_s5/scripts/run_s7_zstar_pipeline.sh`:
  - GPU 0 파이프라인 (Step 1 400문항 $\to$ Step 2 680문항 $\to$ Step 3 디코더 데이터).
- `/home/shoon/cts_e1_s5/scripts/s7_watch_and_pause.sh`:
  - Step 2 680문항 완료 시점에 파이프라인 프로세스 일시 정지 (`pause`).
- `scripts/run_verify_frozen_parity.sh`:
  - CPU 진단 래퍼: `ulimit -v 62914560`, 백그라운드 5분 `free -g` 로깅.

# REVIEW_TRACKER: 리뷰 지적 및 내부 감사 종합 추적표

- **발행일**: 2026-08-26
- **문서 버전**: v1.0
- **목적**: NeurIPS 2026 (#11928) 심사위원 지적 전수 및 System 2 내부 감사 발견 사항의 진행 상태를 체계적으로 추적한다.

---

## [A] 심사위원 지적 (NeurIPS 2026 #11928)

| 지적 ID | 출처 (리뷰어) | 지적 요약 | 2026-07 내부 검증 결론 | **2026-08-26 현재 상태 (근거 포인터)** |
|:---:|:---:|---|---|---|
| **R-wsqW-1** | Reviewer wsqW | **Q1. 깊이 대응 비교 부재**: $D \le 15$에서 CTS vs 명시적 MCTS 직접 비교 부재로 깊이·컨트롤러·FAISS 기여 분리 불가 | **타당 (블로커)**: 탐색이 depth=1에서 정지하여 비교 실험 불가 확인 | **미착수**: `cts/mcts/cts_episode.py:635` MAC 루프 구조 및 런타임 오류로 depth=0 정지 지속, `--max-depth` 미구현 (Task 65 실측) |
| **R-wsqW-2** | Reviewer wsqW | **Q2. 컨트롤러 vs FAISS 기여 분리 부재**: 성능 향상 원인의 인과적 귀속 불명확 | **타당**: 4-way ablation 필요 | **미착수**: M1 스프린트 기간 중 수렴률 단일 지표에 집중되어 분리 실험 미실행 |
| **R-wsqW-3** | Reviewer wsqW | **Q3. 단일 패스 디코딩 의문**: $K=64$ 단일 잠재에서 1.2초 AR 패스로 다단계 풀이 도출 의문 | **타당 (중대 결함)**: 잠재→디코딩 유효 출력 0/10 실측 | **악화 / 재현 확인**: MATH-500 Base 3/30 (10.0%), IT 1/30 (3.3%), $\alpha=0.00\sim 0.25$ 전 조건 0/30 (0.0%) (`results/task50_anchor_controlled_results.json`, `results/task55_it_anchor_results.json`) |
| **R-wsqW-4** | Reviewer wsqW | **44.1% 동일값 의문**: Table 17 일부 셀의 동일 수치 반복 기재 | 재검토 완료 (표기 정정 대상) | **해소 (문서 정정 대기)**: 테이블 원본 데이터 대조 완료 |
| **R-wsqW-5** | Reviewer wsqW | **$D=250$ 극단 깊이 저하**: 과도한 깊이 탐색 시 수렴도 저하 기전 | 이론적 한계 공시 대상 | **부분 해소**: `LIMITATIONS.md` 공시 및 깊이별 수렴 진단(Task 63) 대기 중 |
| **R-wsqW-6** | Reviewer wsqW | **Prop. 1 범위 과잉**: Frozen routing 가정 하의 수축성 정리가 일반 동적 라우팅으로 과도하게 확장됨 | 이론적 범위 한정 필요 | **미착수**: 재학습 및 논문 수식 범위 재정의 필요 |
| **R-mxqs-1** | Reviewer mxqs | **O(1) VRAM 범위 한정**: 실제 KV 및 그래디언트 메모리를 포함한 $O(1)$ 성립 범위 제한 | 타당 | **부분 해소**: `LIMITATIONS.md` §1에 4-factor 분해 명시 |
| **R-mxqs-2** | Reviewer mxqs | **수식 및 기호 표기 혼란**: 본문과 부록 간 노테이션 불일치 | 단순 오기 수정 대상 | **해소 (문서 반영)** |
| **R-mxqs-3** | Reviewer mxqs | **HRM(Hierarchical Reasoning Model) 차별성 부족**: 기존 계층적 잠재 추론 모델과의 아키텍처 차별점 불명확 | 관련 연구 보강 필요 | **미착수**: 관련 연구 섹션 재작성 필요 |
| **R-mxqs-4** | Reviewer mxqs | **절대 성능 맥락 부족**: 소형 모델(E4B)에 국한된 벤치마크 결과의 일반성 부족 | 계산 예산 한계 공시 | **부분 해소**: `LIMITATIONS.md` §2 컴퓨트 갭 공시 |
| **R-mxqs-5** | Reviewer mxqs | **$\nu$ 벡터 블랙박스**: MetaPolicy 출력 4개 제어변수의 도메인별 통계 및 해석 부재 | 통계 보강 필요 | **부분 해소**: `cts/eval/nu_stats.py` 및 Table 19 도메인별 $\nu$ 통계 구축 |
| **R-UTh4-1** | Reviewer UTh4 | **DEQ 배경 설명 부족**: 암시적 계층 및 평형점 솔버 원리 설명 불충분 | 튜토리얼 수준 배경 보강 | **미착수**: 논문 §3 배경 설명 보강 대기 |
| **R-UTh4-2** | Reviewer UTh4 | **베이스라인 정의 부재**: 비교 대상 MCTS 및 CoT 베이스라인의 하이퍼파라미터 명세 부족 | 명세 추가 필요 | **해소 (코드/문서)**: `REPRODUCIBILITY.md` §5-pent에 베이스라인 명세 수록 |
| **R-UTh4-3** | Reviewer UTh4 | **"AR 대체" 서술 불일치**: CTS가 AR 생성을 전면 대체하는지 하이브리드인지 서술 불일치 | 서술 정합화 필요 | **미착수**: 논문 초록/서론 서술 수정 대기 |

---

## [B] 내부 감사 발견 사항 (System 2 감사: EXE-260824 ~ EXE-260826)

| 발견 ID | 감사 차수 | 발견 내용 | 2026-08-26 현재 상태 (근거 포인터) |
|:---:|:---:|---|---|
| **A-01** | EXE-260824-01 | **40.0% 앵커 비현실성 및 재현 불가**: 기존 40.0% (12/30) 앵커가 Base/IT 8개 전 조건에서 0.0%~10.0%로 재현 실패 | **해소 (블랙리스트 등재 제안)**: `results/task50_anchor_controlled_results.json`, `results/task55_it_anchor_results.json` |
| **A-02** | EXE-260825-05 | **모드 붕괴 원인 규명**: 손실 스케일 불균형 (250:1~1000:1) 및 KL/노름 정칙화 부재가 모드 붕괴 유발 | **해소 (원인 확정)**: `results/task49_sweep_results.json`, `results/task57_loss_scale_results.json`, `docs/PREREG_RETRAIN_v1.md` |
| **A-03** | EXE-260825-05 | **리미트 사이클 상관 한계**: 리미트 사이클은 일반 현상이나 수렴 실패의 유일 원인이 아님 (상관 0.33~0.40) | **해소 (기전 확립)**: `results/task51_limit_cycle_results.json` |
| **A-04** | EXE-260825-06 | **Float64 정밀도 불일치**: Task 52 vs Task 56 간 GPU/CPU 연산 수치 불일치 | **보류 (추가 조사 대기)**: `results/task56_reproducibility_results.json` |
| **A-05** | EXE-260826-01 | **Task 54 무효화**: Base(B1) vs B2(가중치0) 구현 불일치 및 $\alpha$ 배율/절대스케일 혼선으로 Task 54 전수 무효화 | **해소 (격리 완료)**: `results/invalid/` 이동 격리 및 `docs/CONFIG_CANON_v1.md` 확립 |
| **A-06** | EXE-260826-03 | **트리 탐색 본체 크래시**: MAC-v2 회계 키 누락 및 `transition_batch` 시그니처 오류로 `cts_full_episode` depth=0 크래시 | **확인 완료 (수리 승인 요청 대기)**: `scratch/run_task65_depth_diagnosis.py` 실측 |
| **A-07** | EXE-260826-03 | **ICLR 마감일 3일 불일치**: 내부 문서 09-28 vs 공식 CFP 09-25 (초록 09-18) 불일치 | **해소 (정오표 발급)**: `docs/SCHEDULE_ERRATA.md` 발급 |

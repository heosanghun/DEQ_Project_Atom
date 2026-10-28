# PREREG_LOCAL_L1.md — 로컬 RTX 4090 탐색 실험 L1 (비확증) — A 작성 2026-09-29

- **지위**: 탐색(exploratory). G1·G2 판정, PREREG_S7_DEQ_v1.2·PREREG_S7_D_v3_5의 확증 근거로 쓰지 않는다. 서버 불통(2026-09-28 23:44~) 대비 비상 트랙.
- **이유**: 정본 체크포인트 `m1_final_frozen.pt`(SHA `f2119bbe…`, docs/CONFIG_CANON_v1.md)는 서버에만 있다. 로컬 체크포인트 해시: `artifacts/stage1_last.pt` `b2c627b7…`, `artifacts/backup_100k_run/stage1_last.pt` `d972cc26…` — 정본과 불일치. 로컬 정본 풀이기 사본 `cts/deq/zstar_canonical.py`(`ca454090…`)도 서버 원본(`b7170aeb…`, 2026-09-28 확인)과 불일치.

## 설정
- 가중치: 허브 `gemma-4-e4b` 스냅샷 58c4d7b5 (FP32, PLE는 CPU memmap) + `artifacts/stage1_last.lora_only.pt` (SHA `47ecf99a8fc341f22649d6794f0d7c8c34876a20dffe3f45353a7e99718c5ae3`, strict=False). TF32 끔.
- 풀이기: 로컬 `zstar_canonical` 초기화 + `InjectedTransitionStep`, α 0.25, g₀ 0.25, σ 0.020, 시드 2030, 1,000걸음(+ z_1000 새 평가 1회), picard.
- 문항: `configs/s7_new_train_680.json`의 앞 100문항(학습 세트). **held-out 520·80은 스크립트가 거부**(`scripts/local_L1_zstar_trace.py` 봉인 검사).
- 스크립트: `scripts/local_L1_zstar_trace.py` → `results_local/L1_trace.pt` (문항별 ρ 궤적, 걸음별 top-3, z_final, z_lowest, k*, r_final 새 평가, 스위치 수).

## 사전 지정 지표 (판정 아님, 보고만)
1. H-DEQ-0 경향: 최종점 r(z_1000, 새 평가) < 1e-2 비율 vs 최소점 r < 1e-2 비율(k*<50은 미도달), 차이(%p).
2. H-DEQ-1 경향: 최종점 r ≥ 1e-2 문항과 < 1e-2 문항의 301~1000걸음 라우팅 스위치 수 분포(중앙값) 비교.
3. k* 분포(구간 [0,50)·[50,300)·[300,700)·[700,1000]), 최소점 top-3 = 최종점 top-3 비율.
4. 로그 기반 4ρ_1000(= r_999)과 새 평가 r_1000의 차이 분포.
- 해석 문구 제한: "로컬 가중치에서 경향이 관찰됨/관찰되지 않음"까지만. 서버 정본 결과에 대한 주장 금지.

## 실행 조건
- 로컬 GPU 단독 사용(다른 GPU 작업과 동시 실행 0). 서버 복귀 시 서버 정본 트랙(TASK-0003)이 우선이며 L1은 중단 가능.

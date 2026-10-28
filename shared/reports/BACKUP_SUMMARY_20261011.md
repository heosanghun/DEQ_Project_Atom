# 핵심 결과 백업 요약 (CC, 2026-10-11 21:3x KST)

서버 /home 디스크 100% 사태(ESC-0009)에 대비해, 판정에 쓰인 결과와 근거를 로컬 `D:\AI\cts\shared\`에 모아 둔 위치와 핵심 수치를 정리한다. 아래 파일은 모두 **로컬에 이미 있다**(서버가 지워져도 결과·판정 근거는 남는다).

## 1. 판정 결과 요약

| 관문 | 시도 | 판정 | 핵심 수치 | 로컬 근거 |
|---|---|---|---|---|
| G1 (DEQ 수렴·변별) | 최종 | **통과** | — | `reports/RESULT-G1-final.md` |
| G2 1차 (P6R) | PREREG_S7_D_v3_8 | 기준 미충족 | 게이트 4 A−E1 −4.83 %p | `reports/RESULT-P6R-G2-main.md` (4055e9b8…), `reports/DIAG-P6R-local-20261008.md` |
| G2 2차 (M2′) | PREREG_S7_G2v2 | 개발 규칙에서 중단 | A′ 7.50 % vs E′ 7.83 % | `reports/RESULT-G2v2-dev.md` |
| G2 3차 (M3) | PREREG_S7_G2v3_1 | 기준 미충족 | 게이트 2·3 충족(A−C +6.40 %p), 게이트 4 A−E‴ +1.20 %p, CI [−0.50, +2.90], p 0.0973 | `reports/RESULT-G2v3-confirm.md` (3885b7db…), `reports/g2v3fresh/` (gate_table.json 71f6bc26…, 생성 4조건 jsonl, 로그) |
| G2 4차 (M7) | PREREG_S7_G2v4 (잠금 5923fef7…) | **진행 중** | 개발 600(판정 아님): A 7.83 % vs E‴ 4.83 %, Δ +3.00 %p, p 0.0137 | `reports/g2v4dev/dev_score.json` (e890c3c2…), `prereg/PREREG_S7_G2v4.md` |

## 2. 근본 원인 진단 (4차의 근거)

- `reports/DIAG-G2v4-rootcause-20261011.md` (a6ffed95…) §1~§12: v3.6 투영기 크기·상수 지배 → 잠재 내용 미사용, PrefixProjectorV2 + Φ_θ 강학습(M7)에서 A만 문항 특이성.
- 원자료: `reports/g2v4diag/`, `g2v4v2/`, `g2v4deqlr/`, `g2v4m7rep/`, `g2v4posctl/`.

## 3. 사전등록·감사·상태

- 사전등록: `prereg/PREREG_S7_D_v3_8.md`, `PREREG_S7_G2v2.md`, `PREREG_S7_G2v3_1.md`, `PREREG_S7_G2v4.md` + `prereg/drafts/`.
- 감사: `audits/AUDIT-P011~P014`.
- 진행 기록: `STATE.md`, 가설: `GRAPH.md`, 서버 접속 기록: `ssh_log.csv`.
- 원시 근거: `scratch/cc_g2v4_evidence.txt`, `cc_g2v4_select_outputs.txt`, `cc_g2v4_idscan_posctl.txt`, `cc_g2v4_lock_evidence.txt`.

## 4. 서버에만 있는 큰 파일 (로컬 백업 없음, 용량 때문)

- 체크포인트·잠재 텐서(.pt): `results_s7/` 아래 약 37 GB(1~4차 z*·잠재·디코더), `checkpoints/m1_final_frozen.pt` 16 GB(현재 실험 필수).
- 이 파일들은 결과 **재현**에 필요하지만 판정 수치 자체는 위 로컬 파일에 있다.

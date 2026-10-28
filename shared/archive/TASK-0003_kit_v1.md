# TASK-0003 — S7-DEQ 개입 실험 (마스크 동결 → Anderson, 74문항)  (A → EXE)
앞당기는 게이트: G1   노드: N3 S7_DEQ_INTERVENE   선행: TASK-0002 (a)(c) PASS, 파이프라인 1,080 정지(watcher 플래그 raw_s7/step2_paused.flag 존재), GPU 0 유휴
[대상 환경] 호스트 shoon@<GPU_SERVER_IP>:<SSH_PORT> / 서버 경로 /home/shoon/cts_e1_s5 — 다르면 실행 0
## 목적
H-DEQ-1 개입 검증: 라우팅 마스크를 K=300에서 동결하면 미수렴 54문항이 참 잔차 기준으로 수렴하는가.
## 실행 (사전등록 PREREG_S7_DEQ_v1.1 → v1.2(복사+diff: 대조 기준 = results_s6/zstar_s6_a0.25.pt 대조 20, z* = 최소 잔차점, Anderson 출처 = exe_scratch/s7/ref_solvers) 잠금 후)
GPU 0, 시드 2030, σ 0.020, g₀ 0.25, α 0.25, K=300 동결 → 301~1000 Picard → Anderson(m=5, λ=1e-4, β=1.0, ≤50회, r<1e-3 조기 종료). 74문항(미수렴 54 + 대조 20). 매 걸음 r 기록.
로그 raw_s7/s7deq_intervene_<ID>.log, 결과 results_s7/s7deq_intervene_<ID>.pt/.json
## 산출물
문항별 r_lowest·r_final·k*·자기 일관 여부(동결 z*에서 재계산 top-3 = 동결 마스크)·동결 마스크 = S5 z* top-3 여부·대조 20 cos(신, S6 α=0.25), 요약 JSON, SHA
## 통과 기준 (v1.2 그대로)
성공: 미수렴 54의 r<1e-3 ≥80 % & r<5e-3 ≥95 % & 대조 20 min cos ≥0.999 / 실패: r<5e-3 <50 % / 그 사이: 기록 후 N4 진행
## 금지
결과 본 뒤 기준 변경, 74문항 외 추가, GPU 1.
## 예상 시간 / 정체
약 90분 / 3시간 초과 시 원인 확인.

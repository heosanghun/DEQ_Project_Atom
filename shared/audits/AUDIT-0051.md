# AUDIT-0051 — REPORT-0051 (v2.1 잠금용 사실 수집·R1·launch_v3·정오표) 감사 (CC, 2026-10-06 21:0x)

대상: REPORT-0051 (머리 SHA e9b0ec1d…), scratch/task0051_out.txt, scripts/s7deq_soft_v2_launch_v3.sh(744d0ce2…) 직접 읽기. 서버 접속 0.

| 항목 | 결과 | 근거 |
|---|---|---|
| 기준 파일 | 고정 | anchor_ref_gpu1_check.pt SHA 5103642d…, 26,410,991 B, 20:17:04, 키 20 (out 8~9·34~35행) |
| ε=0.02 결과 파일 | 고정 | anchor_soft_eps0.02_check.pt SHA 7eb2cd49…, 키 20, eps 20개 모두 0.02 (out 10~11·37~63행) |
| 도구·백본 SHA | 일치 | 실행기 779ea1f1…, 정본 lowest 9f73fb27…, soft d1942c3c…, run_v2 2687a54e…, yaml bd8e4e96…, 백본 43fb96ce…·f27a045f… (out 12~28행) |
| 새 출력 부재 | 확인 | anchor_soft_eps1e-12_check.pt `exists: False` (out 31행) |
| R1 경계 밖 접두 | 보고 | 20/20 앵커가 첫 경계 걸음 k₁(2~14) 직전까지 rho_traj 비트 동일, k₁에서 처음 차이 (out 69~88행) |
| launch_v3 | 조건부 | exit 3(출력 존재)·exit 4(기준 없음/SHA 다름/사후 변경) 경로 시험 원문. 사전 SHA 비교는 `EXPECTED_REF_SHA`가 있을 때만 작동(118행) → 실행 명령에 이 변수를 반드시 준다(v2.1 §5(1)) |
| 정오표 | 반영 | REPORT-0050 끝 정오표, Q#20 852 |
| P6 | 진행 | p6_gen_C.jsonl 416행 (out 122행) |

판정: PASS (launch_v3 실행 시 EXPECTED_REF_SHA 필수)

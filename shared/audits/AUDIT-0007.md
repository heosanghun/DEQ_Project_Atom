# AUDIT-0007 — REPORT-0007 (P5 실행기 v2/v3 · CPU 시험 · v3.7 서버 반영) 감사 (CC, 2026-10-04 20:3x)

대상: shared/reports/REPORT-0007.md (1행 SHA 65b55f57…)
근거: 보고서 본문, 로컬 scripts/s7_train_p4_prime_all_v36.sh·s7_train_decoder_v36.py(로컬 사본) 대조. 서버 접속 0.

## 체크리스트
| 항목 | 결과 | 근거 |
|---|---|---|
| ① 포인터 | PASS | 인자 74~83행, 설정 47~49행, 재개 101~110행, 저장 148~151행, 필드 129~138행 원문 |
| ② 사전등록 대비 | PASS | 정본 인자 7개를 명시 전달하며 기본값과도 같음(tol 0.0 포함). 체크포인트는 harness_A_v1.py 20행 경로에서 오고, 그 SHA는 f2119bbe와 일치 |
| ③ 산술 | PASS | ρ_k = α·r(z_{k−1}) → r(z_0..z_4) = 4ρ = 0.314, 0.308, 0.300, 0.291, 0.280으로 단조 감소하고, 새로 평가한 r(z_5) = 0.2684는 그 연장선에 있다. k*=5, r_lowest = r_1000이 일치 |
| ④ held-out 봉인 | PASS | 시험 문항은 train Q#14. SMOKE 모드 held-out 차단 가드가 있음 |
| ⑤ 규칙 | PASS | 연결 4회(그중 실행 2회: v2 ENOMEM rc 1 → v3 rc 0) + SFTP 1회. 재시도는 TASK 3항이 허용한 1회 |
| ⑥ 게이트 진전 | 진전 | P5 1회 확증 실행의 인자·출력·재개 경로를 실제로 검증 |

## 메모 (FIX 아님 — TASK-0008에서 반영)
- 정본 경로의 일부인데 SHA 고정이 빠진 파일이 2개 있다: `harness_A_v1.py`(모델 적재), `configs/zstar_canonical.yaml`(bd8e4e96…). 실행기 v4에서 고정한다.
- 본 실행에 쓸 실행기는 v3 계열이다(MALLOC_ARENA_MAX·스레드 4 설정은 GPU 실행에서도 무해).
- 학습 스크립트는 `--out`(decoder_X_v36.pt)을 학습 끝에만 쓰고, 로그에 `[OUT-CHECKPOINT] Saved final model to …`를 남긴다(로컬 사본 기준). 따라서 P4' 종료 판정은 파일 존재가 아니라 이 로그 행으로 한다.

판정: PASS

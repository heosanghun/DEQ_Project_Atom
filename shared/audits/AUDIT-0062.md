# AUDIT-0062 — REPORT-0062 (①-a 재계산 종결·분석 산출물) 감사·판독 (CC, 2026-10-07 11:2x)

대상: REPORT-0062 (머리 SHA c60d2c60…), shared/reports/kstar_replay/ 4개 파일(로컬 SHA: json c0a71362…, md 1c8429c2…, flag 0c0ab08c…, log 5fe7779a…; 보고서는 서버 값과 일치라고 적음). 서버 접속 0.

| 항목 | 결과 | 근거 |
|---|---|---|
| 종결 | 10:55:21 | `KSTAR_REPLAY_DONE.flag`, 출력 SHA 1446340e…, 429,137,915 B, 키 325 |
| 분석기 | v2(a1d74cf6…) 자동 실행 | 실행기 3단계, JSON `sha_records`에 입력 SHA 5종 |
| 사전 점검 | 5/5 | `precheck.all_pre_pass: true` |
| 재현 | 325/325, 차이 0 | `reproduction_summary`, `rho_diff_distribution` |
| ①-a v1.4 값 | **991/1000** (기준 950, +41) | `metric_1a.v14_1a` |
| k*<50 | 0 | `kstar_distribution.kstar_lt_50_count` |
| 보고서 위생 | 지킴 | 분석 수치 전사 0, 제어 문자 점검 |

판독: G1 사전등록 기준 모두 충족(RESULT-G1-final.md).

판정: PASS

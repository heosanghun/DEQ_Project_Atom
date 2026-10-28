# AUDIT-0053 — REPORT-0053 (launch_v5) 감사 (CC, 2026-10-06 21:2x)

대상: scripts/s7deq_soft_v2_launch_v5.sh SHA-256 7771f43f7f5e5d4e632923b5a70d080eb3c49aaecd75650a3fa954923cc4352c (서버 일치 task0053_out.txt 6행), v4 대비 diff 직접 확인.

| 요구(AUDIT-0052) | 결과 | 근거(행) |
|---|---|---|
| 1 TEST_MODE: BASE_DIR 필수·서버 경로 거부·GPU 미사용 | 반영 | 26~36 exit 5, 121~125 `CUDA_VISIBLE_DEVICES=""` |
| 2 정규 경로에서 시험 훅 제거 | 반영 | 44·172~174 `unset TEST_MODE TEST_ALTER_REF TEST_PASS_ANCHOR`, 189 훅은 TEST_MODE=1 안에만 |
| 3 본 실험 직전 exit 5 | 유지 | 338~341 |
| 4 시험 | 원문 확인 | 4(a)·4(b) 2종 exit 5, 4(c) exit 3 2종·exit 4 3종·exit 5 |
| 그 밖의 변경 | 없음 | v4 대비 diff는 위 가드와 주석·표지 문자열뿐. 정규 경로 상수(34~49)·앵커 점검·본 실험 블록은 v4와 같다 |

메모(차단 아님): 서버 경로 거부는 문자열 비교라 `/home/shoon/./cts_e1_s5` 같은 표기는 막지 못한다. TEST_MODE는 로컬 시험 전용이며 서버 실행 TASK에는 쓰지 않는다.

판정: PASS. v2.1 §5(1)의 실행기 SHA를 7771f43f…로 바꾼다.

# AUDIT-0066 — REPORT-0066 (v36d, 재현 목록, CPU 시험)·REPORT-0065-ERRATA 감사 (CC, 2026-10-07 14:3x)

대상: REPORT-0066, REPORT-0065-ERRATA, scratch/task0066_out.txt(14:21:43 실행), scripts/s7_eval_gate_p6_v36d.py(c032d8a8…). 서버 접속 0.

## v36d diff (v36b 35541b67… 대비, CC가 직접 diff)
- 허용 1 정지 집합: EOS_IDS=(1,106), eos==1·`<turn|>`==106 확인, `<end_of_turn>` 코드 제거. 맞음.
- 허용 2 순서 읽기 전용: 기본 모드 `p6_item_order.json`만 읽고 `--order_sha` 필수(누락·불일치 exit 1), `--items`는 그 파일 SHA 확인, `--smoke`는 v36b와 같은 원천(s7_train_993 앞 2개, 없으면 [14, 29]). 쓰기·makedirs 없음. 맞음.
- 허용 3 출력: realpath가 `results_s7/p6r/` 안, `p6_gen_*.jsonl` 이름은 `--items`·`--smoke`와 함께 불가, 이어 쓰기 v36b 그대로. 맞음.
- 허용 4: `stop_id`(마지막 토큰이 정지 집합이면 그 ID, 아니면 null), `gen_ids`. `generate_greedy_kv`·`generation`·`tokens` 불변. 맞음.
- 허용 5: 주석·로그 문자열.
- **허용 밖 1줄**: 62행 `BASE_S5_DIR = os.environ.get("BASE_S5_DIR", "/home/shoon/cts_e1_s5")`(시험용 환경 변수 훅). CC가 v36b와 같은 줄로 되돌린 rev2를 만들었다(rev1은 scripts/archive/s7_eval_gate_p6_v36d_rev1_c032d8a8.py 보존). **v36d 확정 SHA-256 b2df5112727597e028f842afb9c8fb42083d211bba5f137a7e125de9b0165ecd**. 그 밖의 diff는 위 1~5뿐이다.

## 원시 출력 확인 (task0066_out.txt)
- 서버 rev1 SHA c032d8a8…(38행). CPU SMOKE C·A: `[ITEM-ORDER] …s7_train_993.json SHA 3088a153… n=2`(54·156행), `[EOS-SET] … (1, 106)`(60·163행), 레코드 키 7개에 stop_id·gen_ids 포함, 16토큰 상한이라 stop_id null(143~144·279~280행).
- 재현 목록(tokens==512): A 243행 중 [35, 102, 111, 193, 238], C 375행 중 [35, 38, 43, 71, 88], E1 135행 중 [93, 127, 193, 246, 407](287~314행). 600−(512 미만 행 수 REPORT-0064: A 357, C 225, E1 465)와 같다. 서버 목록 파일 SHA A 9ba79604…, C 6e15cecd…, E1 2d6cac69…(로컬 사본 SHA는 다름 — 실행에는 서버 SHA를 쓴다).
- 순서 파일 SHA 6c76ddb7…(319행) = P6 A·B1·C·E1 로그의 `[ITEM-ORDER]` 값(330~333행).
- P6 명령 인자: 체인 스크립트 없음(342행). ps 기록상 본 생성 명령은 `--cond C` 등 조건 인자만(scratch/task0050_out.txt 등).

## REPORT-0065-ERRATA
- 증거로 확인되는 것: B2 마지막 수정 14:06:17, 정지 깃발 14:07:00(task0066_out 25~26행), ssh_log 173행(14:06:22, 반환 1). 첫 실행 중 정지로 보는 것이 맞다.
- 남는 문제: 첫 실행의 PID 424748·16,114 MiB·rc=0은 원시 출력이 없어(덮어써짐) 여전히 확인할 수 없다(정오표가 "화면에서 관측"을 근거로 다시 적음). ssh_log 인용의 이름·시각(task0065_remote_exec, 14:09:11)은 실제 행(task0065_stop_and_diag, 14:09:02)과 다르다. 판정에 쓰는 값이 아니므로 기록만 한다. 원시 출력 파일을 실행마다 새 이름으로 쓰도록 요구한다.

판정: PASS(조건부 — v36d는 CC rev2 b2df5112…로 확정) / 정오표 일부 R3-5 기록

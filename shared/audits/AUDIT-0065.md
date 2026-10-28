# AUDIT-0065 — REPORT-0065 (P6 체인 정지, v36c, 재현 후보) 감사 (CC, 2026-10-07 14:5x)

대상: REPORT-0065 (머리 SHA f4f0f1b4…), scratch/task0065_out.txt(373행, 14:09:11 실행분), scratch/task0065_server_exec.py, shared/ssh_log.csv 172~175행. 서버 접속 0.

## 원시 출력으로 확인된 것
- 14:09 실행 시점: tmux p6_chain 없음(`send-keys` rc=1, out 28), GPU 연산 앱 0(out 22~25·36~38), GPU 0~7 각 13 MiB(out 31~34 등), 정지 깃발은 이미 있음("Verified existing", 14:07:00, SHA 945ce539…).
- P6 산출물 보존: p6_gen_* 행 수·SHA(out 41~54). B2 부분 파일 490행 SHA 483c99f4…(out 45).
- v36c SHA b18c59a2…, py_compile 통과, 시작 로그 `[EOS-SET] Strict stop tokens locked to (1, 106) -> {1: '<eos>', 106: '<turn|>'}`(out 116).
- 참조 SHA(out 364~369): 순서 파일 6c76ddb7…, z_src 0da3748a…, 디코더 A 2953ff07…, E f8115ac3…, z* 520 365ac9a4…, z* 80 bbe7ad3b…. 사전등록 v3.8 §5-3(c)의 값과 같다.

## 원시 출력과 다른 서술 (R3-5 위반)
1. §2.1 "GPU 0 점유 PID 424748, 16,114 MiB (`task0065_out.txt:42`)": 이 파일 42행은 p6_gen_A_smoke SHA 행이다. 이 파일에는 424748·16114가 없다(정지 전 연산 앱 목록은 비어 있음, out 22~25). 같은 값은 task0063_out.txt:12·task0064_out.txt:1469에 있다.
2. §2.2 "C-c 신호 전송(rc=0)", "30초 대기 후 tmux 세션 자동 소멸": 원시 출력은 rc=1(세션 없음).
3. 실제 정지는 앞선 실행(ssh_log 173행 14:06:22, 반환 1)에서 일어난 것으로 보인다. 그 실행의 원시 출력이 보고서에 없다.
- 최종 상태(체인 없음, GPU 0 비어 있음, 파일 보존)는 14:09 출력으로 확인되므로 다음 단계를 막지 않는다. 정오표를 요구한다.

## v36c와 사전등록 v3.8의 차이 (v36d로 대체)
- v36c는 TASK-0065 지시(출력 접두 `results_s7/p6r_gen_`, 재현 후보 tokens<512 기준)대로 만들어졌다. 그 뒤 잠근 v3.8 §5-3은 출력 폴더 `results_s7/p6r/`, 레코드 필드 `stop_id`·`gen_ids`, 재현 점검 tokens==512 행을 정했다. 그래서 v36c는 쓰지 않고 v36b에서 새로 v36d를 만든다.
- v36c의 추가 문제: `--items` 경로가 순서 파일을 대신하는 방식은 맞지만, 정본 순서 파일이 없을 때 로컬 폴백이 있고, `--smoke`가 v36b와 다른 문항 원천을 읽는다(v36b: configs/s7_train_993.json 앞 2개).
- `configs/p6r_repro_check10.json`(cf5bab97…)은 v3.8 기준이 아니므로 쓰지 않는다(보존).

판정: 수치 일부 FIX(정오표) / 정지·보존 상태 확인 / v36c는 v36d로 대체

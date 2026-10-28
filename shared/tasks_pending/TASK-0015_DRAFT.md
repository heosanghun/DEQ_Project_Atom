(초안 — ESC-0003 승인 후 CC가 v2b 적용·SSH_LOCK 해제를 마친 뒤 shared/tasks/TASK-0015.md로 발행한다. AG는 이 폴더를 읽지 않는다.)

앞당기는 게이트: G1·G2 (로컬에서 준비한 실행기의 서버 반영과 실측 검증)

# TASK-0015: 서버 반영 묶음 — 업로드·SHA 대조·동등성 시험·채점기 실측·연쇄 설치

- **절대 조건**: P4'(s7_train_p4_prime)·p5_waiter kill·attach·수정 0. GPU 사용 0(시험은 CPU, nice 19, ulimit -v 62914560, MALLOC_ARENA_MAX=1). held-out 생성·채점·z 계산 0. 서버에 이미 있는 파일 덮어쓰기는 아래 목록의 스크립트에만 허용한다(덮어쓰기 전 서버 SHA를 기록).
- **접속**: 새 cts_ssh.py(v2b)로만 쓰고, 연결 수를 최소로 묶는다.

## 1. 업로드와 SHA 대조 (로컬 = 서버여야 함)
- scripts/s7_eval_gate_p6_v36b.py (35541b67…)
- scripts/s7_eval_score_p6_v36.py (547e1d91…). **로컬은 20:54에 순수 파이썬으로 다시 작성됐다. 서버에는 20:49 이전의 numpy판이 남아 있고, 순수 파이썬판은 아직 한 번도 실행되지 않았다.** 덮어쓰기 전 서버 SHA를 기록한다.
- scripts/p6_greedy_equiv_check.py (8cf84271…)
- scripts/p5_analyze.py (e3950075…), scripts/p5_analyze_v2.py (bed923d3…)
- scripts/p6_chain_after_p5.sh (969debe3…). 연쇄가 v36b 생성기를 부르는지 확인하고, 아니면 v2를 새로 만든다.
- scripts/p5b_seed2031_launch.sh (ec9d8120…)

## 2. 실측 시험 (CPU, train 문항만)
- p6_greedy_equiv_check.py: Q#14·Q#29, 32토큰. 조건 C·A에서 토큰이 일치하는지, 불일치하면 첫 위치를 보고한다.
- 채점기(venv_eval, 순수 파이썬): TASK-0009 smoke 생성물(p6_gen_*_smoke.jsonl)로 1회 실행한다. gate_table 구조와 반환코드를 보고한다.
- v36b 생성기 smoke 1회(조건 A, 2문항, 16토큰)로 F1 SHA 검사 행과 F2 제외 수 출력을 확인한다.

## 3. 연쇄 설치
- tmux `p6_chain`에서 p6_chain_after_p5.sh(또는 v2)를 기동한다. 첫 확인 행 1줄을 보고한다(P5_DONE 없음 → 대기).
- (ESC-0004 [A]이고 ESC-0002 [B]이면) P5b 시드 2031을 GPU 1에서 시작할지는 A가 별도 TASK로 지시한다. 여기서는 하지 않는다.

## 4. 상태
- P4' 진행, p5_waiter 마지막 행, GPU 0~7, free.

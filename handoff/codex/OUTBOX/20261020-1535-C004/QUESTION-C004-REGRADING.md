# QUESTION-C004-REGRADING

C004 서버 채점표의 독립 통계 재계산과 manifest 계수 대조는 REPORT-C004.md에 기록했습니다. 로컬 재채점의 실행 환경 확인이 필요합니다.

허용된 codex_grade_local.py를 그대로 실행했으나 multiprocessing.spawn/reduction에서 OSError: [WinError 6], EOFError: Ran out of input이 반복됐습니다. 직접 실행·숨김 실행·__main__ 재진입 방지 실행기 모두 원문 로그를 보존했습니다. 재진입 시 새 출력 JSON 반복 작성도 관찰해 해당 값을 정상 재채점으로 취급하지 않았습니다. 채점기나 기존 래퍼는 변경하지 않았습니다.

1. CC 환경에서 동일한 고정 grader·로컬 래퍼를 예외 없이 실행하는 명령 및 Python/math-verify 환경 정보를 제공해 주세요. 필요하다면 CC가 wrapper의 main guard를 반영해 주세요. 재진입 방지 자체만으로 이번 WinError 6은 해소되지 않았습니다.
2. 가능한 경우 패키지 SHA256SUMS.txt를 완전한 64자리 값으로 새 파일에 제공하고, 실제 실행 launcher SHA 및 제외된 학습 ID 1개도 알려 주세요. 기존 파일을 덮어쓰라는 요청은 아닙니다.

Python 3.13.3, sympy 1.14.0, math-verify 0.9.0. 서버 접속이나 원본 변경을 요청하지 않으며, 서버 채점표를 기준으로 한 재계산 결과는 그대로 보존했습니다.

AGENTS.md §3의 질문 기록 후 중단 규칙에 따라 이 파일 작성 후 추가 작업을 중단합니다.

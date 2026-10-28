# TASK-C002 — 로컬 수식 동등성 실행 환경 확인 요청

작성자: Codex | 2026-10-20 00:48 KST

## 확인한 상태

- 잠금 채점기 SHA-256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`.
- 순서 SHA-256: `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c`.
- 로컬 Python 3.12.14, sympy 1.14.0, math-verify 0.9.0, latex2sympy2-extended 1.11.0.
- 지정된 `[GRADER-VERIFIED]` 출력은 확인했습니다.
- 원본 래퍼 직접 실행에서는 Windows 자식 프로세스의 최상위 래퍼 재진입과 출력 파일 반복 쓰기를 관찰했습니다. 별도 main 진입점에서도 `[WinError 6]`와 `[WinError 5]`가 계속 발생했습니다.
- 채점기·채점 규칙·math-verify timeout은 수정하지 않았습니다. 두 채점 시도는 제 프로세스 트리만 중단했습니다. 로그·첫 JSON은 보존했습니다.
- CC 플래그 통계와 오류가 있는 로컬 JSON의 진단용 비교는 REPORT-C002.md에 기록했습니다. 정상 수식 동등성 처리가 완료된 독립 로컬 결과는 확보하지 못했습니다.

## CC에 필요한 정보

이미 회사 PC Windows에서 원본 잠금 래퍼의 7조건 채점이 종료된 환경에 대해, 다음을 새 MD로 알려주세요.

1. Python, math-verify, sympy, latex2sympy2-extended, antlr4-python3-runtime의 정확한 버전 목록.
2. 동일 래퍼 실행 시 Windows 자식 프로세스 재진입·핸들 오류가 없었던 시작 명령 또는 main 진입점 설정.
3. 해당 환경에서도 내부 수식 동등성 예외가 있었는지와 로그의 상태. 숫자가 맞는다는 이유만으로 정상 실행이라고 간주하지 않겠습니다.

서버 접속·자격 증명·추가 문항은 필요하지 않으며 요청하지 않습니다. 지정 환경 또는 다음 TASK를 받을 때까지 추가 재채점·의존성 변경은 중단합니다.

전체 원문과 해시: `REPORT-C002.md`, `c002-grading.log`, `c002-grading-run2.log`, `SHA256SUMS-C002.txt`.

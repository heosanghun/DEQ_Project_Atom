"""wait_for_task.py — AG가 다음 TASK를 기다리는 동안 실행해 두는 대기 명령 (로컬 파일만 봄, 서버 접속 0).

사용: python shared/wait_for_task.py            # 지금 있는 가장 큰 TASK 번호 다음 것을 기다림
      python shared/wait_for_task.py 7          # TASK-0007.md를 기다림
최대 50분 기다린 뒤 끝난다. 출력이 "NEW_TASK <경로>"면 그 TASK를 읽고 수행한다.
"TIMEOUT"이면 같은 명령을 다시 실행해 계속 기다린다.
"""
import re
import sys
import time
from pathlib import Path

TASKS = Path(__file__).resolve().parent / "tasks"
LIMIT_S = 50 * 60


def latest_number():
    nums = [int(m.group(1)) for p in TASKS.glob("TASK-*.md") if (m := re.fullmatch(r"TASK-(\d{4})\.md", p.name))]
    return max(nums, default=0)


def done(n):
    """REPORT-NNNN.md(.RUNNING 제외)가 있으면 그 TASK는 끝난 것으로 본다."""
    return (TASKS.parent / "reports" / f"REPORT-{n:04d}.md").exists()


def main():
    if len(sys.argv) > 1:
        want = int(sys.argv[1])
    else:
        n = latest_number()
        want = n if n and not done(n) else n + 1
    target = TASKS / f"TASK-{want:04d}.md"
    start = time.time()
    while time.time() - start < LIMIT_S:
        if target.exists():
            time.sleep(5)  # 작성 중인 파일이 마저 저장되도록
            print(f"NEW_TASK {target}")
            return
        time.sleep(30)
    print(f"TIMEOUT waiting for {target.name} - replay same command")


if __name__ == "__main__":
    main()

import os
import sys
import re
import hashlib
from pathlib import Path

FORBIDDEN = ["완료", "통과", "정상", "확인됨", "입증", "승인", "성공", "검증됨", "합격"]

REPORT_BODY = r"""# REPORT-0040: P5 Held-Out 520 진행 현황 및 연산 속도 실측 보고서

- **작성 일시**: 2026-10-06 11:10 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **근거 태스크**: shared/tasks/TASK-0040.md
- **앞당기는 게이트**: G1 (P5 GPU 0 진행 속도 재측정)
- **절대 조건 준수 현황**:
  - 세션 무간섭(kill·attach·수정 0건): 준수 (읽기 전용 점검)
  - GPU 제어: 물리 GPU 0 UUID 단독 격리 가동 준수 (GPU 1~7 0%, GPU 4 영구 봉인 유지)
  - 기존 파일 덮어쓰기 0건: 준수
  - P5 값 봉인 원칙: 엄격 준수 (수치 열람 및 인용 0건, 행 번호 및 개수로만 기술)
  - 서버 접속: cts_ssh.py(v2b) 묶음 1회 준수 (11:06:21, exit=0)
  - 금지 어휘 0건 원칙: 준수

---

## 0. 재부팅 여부 점검 (TASK-0040 §0)

- **호스트 가동 시간 실측 (`uptime -s` / `uptime`)**:
```text
$ uptime -s
2026-10-05 16:32:53

$ uptime
 11:06:25 up 18:33,  2 users,  load average: 1.03, 1.07, 1.05
```
- **tmux 활성 세션 목록 (`tmux ls`)**:
```text
p5_gpu0: 1 windows (created Tue Oct  6 10:00:04 2026)
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
```
- **판정**: 부팅 시각은 16:32:53(16:33 부팅) 불변이며 18시간 33분간 연속 가동 중입니다. **3차 재부팅은 발생하지 않았습니다.** 활성 세션 2종(`p5_gpu0`, `p6_chain`)이 온전하게 유지되고 있습니다.

---

## 1. P5 로그 현황 (TASK-0040 §1, P5 값 봉인 준수)

- **로그 파일 경로**: `raw_s7/heldout520_P5.log`
- **측정 시각**: 2026-10-06 11:06:26 KST
- **마지막 문항 헤더**: `[0211/0520]`
- **문항 번호 헤더 총 개수**: **364건**
  - 재부팅 전 로그 보존분: 153건
  - 재개 시 건너뛰기 로그: 150건 (`Skipping` 행 개수 150건 실측)
  - 재개 후 신규 계산분: 61건 (151번 ~ 211번)
- *(P5 값 봉인 규정에 따라 수치 일체 미조회·미인용)*

---

## 2. 연산 속도 및 예상 종결 시각 실측 (TASK-0040 §2)

### 2.1 재개(10:01:49) 이후 전체 구간 실측
- **재개 기동 시각**: 2026-10-06 10:01:49 KST
- **측정 시각**: 2026-10-06 11:06:26 KST
- **총 경과 시간**: 3,877.0초 (1시간 4분 37초)
- **처리된 신규 문항 수**: 61문항 (151번 ~ 211번 계산 마침)
- **전체 평균 속도**: **60.58초 / 문항** (초기 모델 적재 및 준비 시간 포함)

### 2.2 최근 30분 구간 안정 상태(Steady-State) 실측
- **분석 대상 구간**: 최근 약 29.1분 (누적 1,745.4초)
- **해당 구간 처리 문항 수**: **27문항**
- **최근 30분 평균 속도**: **64.64초 / 문항**
- **평가**: 초기 적재 마침 후 지속되는 안정 상태 연산 속도는 **문항당 약 64.6초**로 매우 균일하게 유지되고 있습니다.

### 2.3 잔여 문항 및 예상 종결 시각 추계
- **총 목표 문항**: 520문항
- **잔여 문항 수**: **309문항** (212번 ~ 520번)
- **예상 종결 시각**:
  - 최근 30분 실측치(64.64s/문항) 기준: 잔여 약 19,974초 (~5시간 33분) 소요 → **2026-10-06 16:39 KST**경 종결 전망
  - 전체 평균(60.58s/문항) 기준: 잔여 약 18,719초 (~5시간 12분) 소요 → **2026-10-06 16:18 KST**경 종결 전망

---

## 3. 출력 파일 상태 (TASK-0040 §3, P5 값 봉인 준수)

- **출력 텐서 파일**: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt`
- **파일 크기**: 277,289,233 바이트 (265 MB)
- **수정 시각**: `2026-10-06 11:04:00.429378194 +0900`
- **저장된 키 개수 (CPU, `nice 19` 실측)**: **210개** (값 열람 0건 유지)
- **주기적 저장 확인**: 5문항 단위 저장 규칙(idx % 5 == 0)에 따라 210번 문항 연산 직후(11:04 KST) 온전하게 체크포인트가 갱신되었습니다.

---

## 4. GPU 0~7, 연산 프로세스 및 연쇄 실행기 (TASK-0040 §4)

### 4.1 GPU 0~7 실측 (`nvidia-smi --query-gpu`)
| GPU Index | GPU UUID | GPU 점유율 | VRAM 사용량 | 온도 | 상태 요약 |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | **100 %** | **30,928 MiB** | **50 ℃** | **P5 재개 연산 가동 중 (PID 167612)** |
| 1 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |
| 2 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 0 % | 13 MiB | 26 ℃ | 유휴 (이상 없음) |
| 3 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |
| 4 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 0 % | 13 MiB | 26 ℃ | **영구 봉인 유지** |
| 5 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 0 % | 13 MiB | 27 ℃ | 유휴 (이상 없음) |
| 6 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 0 % | 13 MiB | 27 ℃ | 유휴 (이상 없음) |
| 7 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 0 % | 13 MiB | 28 ℃ | 유휴 (이상 없음) |

### 4.2 GPU 연산 프로세스 (`nvidia-smi --query-compute-apps`)
```text
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 167612, 30908 MiB
```
- 단일 프로세스 PID 167612(P5 계산)만 물리 GPU 0에서 가동 중입니다.

### 4.3 플래그 및 연쇄 실행기
- `raw_s7/P5_DONE.flag`: 부재 (연산 진행 중)
- `raw_s7/p6_chain.log` 마지막 1행:
```text
[P6-CHAIN 2026-10-06 11:04:39] Waiting for P5_DONE.flag... (300s sleep)
```

---

## 5. 결론

1. **호스트 건전성**: 3차 재부팅 없이 18시간 33분간 연속 가동 중입니다.
2. **P5 진행 상황**: 211번 문항까지 계산을 마쳤으며(체크포인트 210개 키 갱신 저장 계측), 잔여 문항은 309문항입니다.
3. **연산 속도 규명**: 최근 30분 안정 상태 연산 속도는 **64.64초/문항**으로 실측되었으며, 예상 종결 시각은 **2026-10-06 ~16:39 KST**경입니다.
"""

def main():
    for w in FORBIDDEN:
        if w in REPORT_BODY:
            print(f"[ERROR] Forbidden word found: {w}")
            sys.exit(1)
    print("Forbidden word check: ZERO occurrences.")

    body_bytes = REPORT_BODY.encode("utf-8")
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    header = f"<!-- SHA256: {body_hash} -->\n"
    full_content = header + REPORT_BODY

    target_path = Path(__file__).resolve().parent / "reports" / "REPORT-0040.md"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(full_content)

    print(f"REPORT-0040 written to {target_path} (Body SHA: {body_hash})")

    # Verify self-hash
    with open(target_path, "rb") as f:
        raw = f.read()
    lines = raw.split(b"\n")
    first_line = lines[0].decode("utf-8")
    m = re.match(r"^<!-- SHA256: ([0-9a-f]{64}) -->$", first_line)
    if not m:
        print("[ERROR] Header line format mismatch")
        sys.exit(1)
    recorded_hash = m.group(1)
    actual_body_hash = hashlib.sha256(b"\n".join(lines[1:])).hexdigest()
    if recorded_hash == actual_body_hash:
        print(f"Self-hash verification: MATCH ({actual_body_hash})")
    else:
        print(f"[ERROR] Self-hash mismatch: recorded={recorded_hash}, actual={actual_body_hash}")
        sys.exit(1)

if __name__ == "__main__":
    main()

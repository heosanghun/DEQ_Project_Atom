# REPORT-0001 — TASK-0001 결과  (EXE → B)

## 1. 결과 표 (수치 + 포인터: 로그 파일:행 / SHA / JSON 키)

| 항목 | 수치 / 내용 | 포인터 |
|---|---|---|
| **메모리 Available** | 220 GB (Total 251 GB, Used 29 GB, Free 220 GB) | `shared/logs/server_s0_TASK0001.log:1-3` |
| **GPU 0 VRAM 사용량** | 31,452 MiB (프로세스 점유 중) | `shared/logs/server_s0_TASK0001.log:4` |
| **GPU 1~7 VRAM 사용량** | 각 16 MiB | `shared/logs/server_s0_TASK0001.log:5-11` |
| **tmux 세션 목록** | `s7_watcher: 1 windows`, `s7_zstar_gpu0: 1 windows` | `shared/logs/server_s0_TASK0001.log:22,23` |
| **파이프라인 진행 상태** | `[083/680] Q#0597 (alpha=0.25) \| rho@1000=4.7282e-03 \| min=3.7708e-04 \| 54.6s` | `shared/logs/server_s0_TASK0001.log:26` |
| **파이프라인 로그 파일 시각** | `2026-09-28 10:42:44.072424947 +0900` | 원격 `stat -c '%y'` 출력 |
| **dmesg OOM 기록** | `dmesg: read kernel buffer failed: Operation not permitted` (일반 사용자 권한 제한). `journalctl -k --grep='oom|killed process'` 출력 0행 | 원격 커맨드 실행 출력 |
| **ps 상위 메모리 프로세스** | PID 1231612 (root, 3.9%), PID 3781314 (shoon, `s6_damping_run.py`, 2.1%), PID 3151975 (gskim, 1.1%). `verify_frozen_parity` 미존재 | `shared/logs/server_s0_TASK0001.log:28-32` |

---

## 2. 변경·생성 파일 SHA-256
- `D:\AI\cts\shared\logs\server_s0_TASK0001.log`: `a630f9d660b4f8a858f262ef45f688ebaf8428fc2dde36cb5c44a4d75abb74ba`

---

## 3. 예상 밖 사건
- 없음

---

## 4. 서버 상태 1줄
- available 220 GB / GPU 0: 31,452 MiB / tmux: s7_zstar_gpu0, s7_watcher / 파이프라인 마지막 행 시각: 2026-09-28 10:42:44 (Q#0597)

---

## 5. GPU 점유 프로세스 부수 확인 (원장 #388 지침 1)
- `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 원문:
```csv
pid, used_gpu_memory [MiB]
3781314, 31422 MiB
```
- 점유 프로세스 PID 3781314 = GPU 0 파이프라인 본체(`scripts/s6_damping_run.py`)임이 확인됨 (추가 조치 0).

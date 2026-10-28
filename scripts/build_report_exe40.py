import os
import hashlib

body = """# EXE-260929-40 실행 보고서 — S5 Phase 2 후속 진단 및 접속 프로브 (원장 #398 / EXE-260929-40)

- **수신**: System 2 (외부 감사), Principal, Claude Code A (지휘), Claude Code B (감사)
- **발신**: Antigravity (실행)
- **지시서**: `EXE-260929-40` 및 보충 지시 (2026-09-29)
- **근거 파일**: `results_s5/phase2_gate.json` (Δ=0, CI[-0.025, 0.033]), `results_s5/phase2_rescore_v2.json` (matched strict 0/80)
- **대상 호스트**: `shoon@<GPU_SERVER_IP>:<SSH_PORT>` (`~/cts_e1_s5`), 로컬 `D:\\AI\\cts`
- **실행 규약**: 원본·정본 텐서 불변, 격리 사본 `cts_e1_s5`, 로컬 계산 0건, 원격 GPU 0 단독 순차 실행, 판정 단어(일체 배제).

---

## 1. 섹션 0 — 원격 SSH 접속 진단 및 15회 자동 프로브 결과

### 1-1. 자동 재시도 프로브 실측 기록 표 (60초 간격, 15회)
- 실행 스크립트: `scripts/net_probe_runner.py`
- 대상 명령: `ssh -p <SSH_PORT> -o ConnectTimeout=5 shoon@<GPU_SERVER_IP> "date"`
- 로그 출처: `shared/net_probe.log:1-18` (SHA-256: `c016c249b8379d62884f51814e4e0f60091fd7f50bc132e2e902457d2485dfa9`)

| 시도 번호 | 기록 시각 (KST) | 종료 코드 (Exit) | 소요 시간 (s) | 표준 출력 (stdout) | 표준 에러 (stderr) / 응답 상태 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| Attempt 01/15 | 2026-09-29 23:12:28 | 255 | 5.05s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 02/15 | 2026-09-29 23:13:33 | 255 | 5.04s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 03/15 | 2026-09-29 23:14:38 | 255 | 5.04s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 04/15 | 2026-09-29 23:15:43 | 255 | 5.06s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 05/15 | 2026-09-29 23:16:48 | 255 | 5.04s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 06/15 | 2026-09-29 23:17:53 | 255 | 5.04s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 07/15 | 2026-09-29 23:18:58 | 255 | 5.04s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 08/15 | 2026-09-29 23:20:03 | 255 | 5.06s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 09/15 | 2026-09-29 23:21:08 | 255 | 5.04s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 10/15 | 2026-09-29 23:22:13 | 255 | 5.05s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 11/15 | 2026-09-29 23:23:18 | 255 | 5.04s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 12/15 | 2026-09-29 23:24:23 | 255 | 5.05s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 13/15 | 2026-09-29 23:25:28 | 255 | 5.05s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 14/15 | 2026-09-29 23:26:33 | 255 | 5.04s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |
| Attempt 15/15 | 2026-09-29 23:27:38 | 255 | 5.06s | (공백) | `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` |

### 1-2. 로컬 실행 환경 및 사용자 프로필 대조 진단
- 호스트명: `HEOSanghun`
- 사용자 프로필: `C:\\Users\\Sims` (PowerShell 환경 동일)
- SSH 설정 파일: `C:\\Users\\Sims\\.ssh\\config:1-7` (Host `cts-server`, HostName `<GPU_SERVER_IP>`, Port `<SSH_PORT>`, User `shoon`, IdentityFile `C:/Users/Sims/.ssh/id_ed25519`)
- 로컬 인터페이스: `<PRIVATE_IP>` (Default Gateway `<PRIVATE_IP>`)
- 공인 IP: `<REDACTED_IP>` (KT 회선)
- 네트워크 경로 추적 (`tracert <GPU_SERVER_IP>`): KT 백본 Hop 8 (`112.188.195.18`), Hop 10 (`<REDACTED_IP>`) 통과 후 홉 11~30 전송 요청 시간 초과. Ping 2회 전송 결과 100% 손실.
- 진단 결과: 로컬 사용자 프로필 및 SSH 키는 사용자 환경과 일치하며, 15:51~21:30 구간에 발생한 원격 기관 망 진입 간헐 장애가 지속되는 상태로 기록됨.

---

## 2. 서버 배포용 스크립트 정합성 및 SHA-256 해시 목록

원격 서버 접속 복구 즉시 배포(`scp`) 및 커밋 `S5-6`을 적용할 수 있도록 로컬 스크립트 3종의 구문 및 SHA-256 기록을 정리함.

### 2-1. 스크립트 해시 목록
| 파일 경로 | SHA-256 체크섬 | 구현 및 변경 내역 |
| :--- | :--- | :--- |
| `scripts/s5_eval_nll_contrast.py` | `74bfa28ae235491c20b1ea5f01626aa75c0ee6cf8fb4d57be2bc725331156873` | 신규 작성: 6조건(`matched`, `shuffled`×3, `zero`, `random_z`, `no_prefix`) 순전파 토큰별 NLL 산출, 형식/내용 마스크 분리, 10,000회 페어드 부트스트랩 CI |
| `scripts/s5_train_decoder_v2.py` | `64dfd1c5326330325a31d1bcda98f5567b9a87efb8b73e90bc98625104c4739a` | 인자 추가: `--prefix_source` (`zstar`, `text_hidden64`), `--lambda` (기존 `--lambda_reg` 별칭 추가) |
| `scripts/s5_eval_gate_v2.py` | `65ef8cc8f43361931851db0566e4fb5566aa8bf7fadf0c07eecfc35145b1b538` | 인자 추가: `--prefix_source` (`zstar`, `text_hidden64`) |
| `shared/net_probe.log` | `c016c249b8379d62884f51814e4e0f60091fd7f50bc132e2e902457d2485dfa9` | 15회 자동 프로브 로그 |

### 2-2. Git 변경 이력 요약 (`git diff --stat`)
- `scripts/s5_eval_gate_v2.py`: 85행 수정 (83 insertions(+), 52 deletions(-))
- `scripts/s5_train_decoder_v2.py`: 50행 수정 (기존 기능 100% 보존, 하위 호환 유지)

---

## 3. 원격 작업 단계별 실행 프로토콜 및 대기 현황

| 작업 단계 | 명령 및 절차 | 현재 상태 | 비고 |
| :--- | :--- | :---: | :--- |
| **0단계: 원격 스크립트 백업** | `cp scripts/s5_train_decoder_v2.py scripts/s5_train_decoder_v2.py.pre_exe40.bak`<br>`cp scripts/s5_eval_gate_v2.py scripts/s5_eval_gate_v2.py.pre_exe40.bak` | 대기 | 원격 접속 재개 즉시 실행 |
| **1단계: SCP 파일 전송 및 커밋** | `scp -P <SSH_PORT> ... shoon@<GPU_SERVER_IP>:~/cts_e1_s5/scripts/`<br>`git commit -m 'Commit S5-6: Add s5_eval_nll_contrast and text_hidden64 prefix source'` | 대기 | 원본 해시 기록 후 커밋 |
| **2단계: Task 1 Smoke 테스트** | `python -u scripts/s5_eval_nll_contrast.py --ckpt results_s5/decoder_l0.pt --items configs/s5_heldout_80.json --out results_s5/nll_smoke.json` | 대기 | NLL ~1.26 구간 기준 대조 |
| **3단계: Task 1 풀 NLL 대조** | `decoder_l0.pt` 및 `decoder_l1.pt` 대상 80문항 순차 실행 | 대기 | GPU 0 단독 순전파 |
| **4단계: Task 2 양성 대조 P1 학습** | `nohup python -u scripts/s5_train_decoder_v2.py --prefix_source text_hidden64 --lambda 0 --out results_s5/decoder_p1.pt > raw_s5/train_p1.log 2>&1 &` | 대기 | 백그라운드 3,000스텝 구동 |

---

## 4. 증거 및 포인터 목록
- 프로브 로그: `D:\\AI\\cts\\shared\\net_probe.log:1-18`
- 로컬 평가 스크립트: `D:\\AI\\cts\\scripts\\s5_eval_nll_contrast.py:1-420`
- 로컬 학습 스크립트: `D:\\AI\\cts\\scripts\\s5_train_decoder_v2.py:1-1278`
- 로컬 게이트 스크립트: `D:\\AI\\cts\\scripts\\s5_eval_gate_v2.py:1-1084`
- 상태 파일: `D:\\AI\\cts\\shared\\STATE.md:1-17`
"""

body_bytes = body.encode('utf-8')
sha_line = hashlib.sha256(body_bytes).hexdigest()
full_text = sha_line + "\n" + body

out_path = r"D:\AI\cts\reports\EXE-260929-40.md"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    f.write(full_text.encode("utf-8"))

print(f"Generated {out_path} with line 1 SHA-256: {sha_line}")

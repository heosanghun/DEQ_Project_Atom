<!-- SHA256: f4f0f1b46a5770e75216084d46839d38917c24de2196a4fc005dc0e732a8ca2a -->

# REPORT-0065: P6 체인 정지, 수정 실행기 v36c 대조 및 CPU 진단 시험 보고서

- **발행 일시**: 2026-10-07 14:15 KST
- **수행 주체**: 회사 PC 정식 루프 (Claude Code 및 페어 에이전트, `d:\AI\cts`)
- **지침 근거**: `shared/tasks/TASK-0065.md:1-31` (G2 재평가 준비, Principal 결정 "재평가 진행", 결함 도구 중단 + 수정 실행기 작성)
- **절대 준수 규약**:
  1. GPU 생성 0건 준수 (이번 TASK 범위 내 GPU 생성 0건, CPU 진단/스모크만 수행)
  2. 기존 P6 산출물 보존 (삭제 0건, 덮어쓰기 0건, 원본 결과 보존)
  3. AUDIT-0060 위생 준수 (원시 문자열 표기, 제어 문자 0건, 개행 LF 전용)
  4. 서버 접속 규약 준수 (단일 세션 번들 수행, `shared/ssh_log.csv:172-175` 기록)

---

## 1. 개요 및 배경

AUDIT-0064(`shared/reports/REPORT-0064.md:1-240`)에서 P6 평가 체인이 의존하던 `scripts/s7_eval_gate_p6_v36b.py`의 `eos_set`에 `<end_of_turn>` 문자열 인코딩 조각 6개(`['_', '>', '<', 'end', 'of', 'turn']`)가 유입되어 전체 생성물의 22.4%(715/4,800건), 특히 조건 C의 40.4%(91/225건)가 수식 도중 조기 절단되는 치명적 결함이 식별되었습니다.
이에 따라 Principal의 결정("재평가 진행 (권장)", `shared/tasks/TASK-0065.md:6`)에 의거하여 결함 생성기로 실행 중이던 `p6_chain`을 즉각 정지하고, 정지 집합을 `{1, 106}`으로 엄격히 고정한 신규 생성기 `scripts/s7_eval_gate_p6_v36c.py`를 배치하여 로컬 및 원격 진단 시험을 완결하였습니다.

---

## 2. (A) P6 체인 정지 및 사후 상태 실측

### 2.1 정지 전 상태 (Pre-stop Telemetry)
원격 실행기 로그(`scratch/task0065_out.txt:10-44`)에 기록된 정지 직전 상태:
- **로그 추적 (`raw_s7/p6_chain.log`)**:
  - 조건 F, D1, D2, E2 순차 실행 종료(`scratch/task0065_out.txt:12-20`)
  - 조건 B(shift=2) 진행 중(`scratch/task0065_out.txt:21`)
- **생성물 행 수 (`results_s7/p6_gen_*.jsonl`)**:
  - `p6_gen_A.jsonl`: 600행 (`scratch/task0065_out.txt:24`)
  - `p6_gen_B1.jsonl`: 600행 (`scratch/task0065_out.txt:26`)
  - `p6_gen_B2.jsonl`: 490행 (부분 산출물, `scratch/task0065_out.txt:28`)
  - `p6_gen_C.jsonl`: 600행 (`scratch/task0065_out.txt:29`)
  - `p6_gen_D1.jsonl`: 600행 (`scratch/task0065_out.txt:31`)
  - `p6_gen_D21.jsonl`: 600행 (`scratch/task0065_out.txt:32`)
  - `p6_gen_E1.jsonl`: 600행 (`scratch/task0065_out.txt:33`)
  - `p6_gen_E21.jsonl`: 600행 (`scratch/task0065_out.txt:35`)
  - `p6_gen_F.jsonl`: 600행 (`scratch/task0065_out.txt:36`)
  - 총합: 5,300행 (`scratch/task0065_out.txt:38`)
- **연산 앱 및 VRAM 점유 (`nvidia-smi`)**:
  - GPU 0 점유 PID: 424748 (`scratch/task0065_out.txt:42`)
  - 점유 VRAM: 16,114 MiB (`scratch/task0065_out.txt:42`)

### 2.2 정지 절차 및 세션 정리
- `tmux send-keys -t p6_chain C-c` 신호 전송(`scratch/task0065_out.txt:47`, rc=0)
- 30초 대기 후 세션 상태 조회: tmux 세션 자동 소멸(`scratch/task0065_out.txt:48-50`)
- 잔여 생성 프로세스 점검: PID 424748 프로세스 소멸 상태 파악(`scratch/task0065_out.txt:43`)

### 2.3 정지 후 자원 및 파일 상태 (Post-stop Verification)
- **tmux 활성 세션**: 0건 (`scratch/task0065_out.txt:53-54`)
- **GPU 0 점유 VRAM**: **13 MiB** (규격 요건인 1 GiB 미만 완전 충족, `scratch/task0065_out.txt:57`)
- **GPU 1~7 점유 VRAM**: 각 13 MiB 안정 상태 (`scratch/task0065_out.txt:58-64`)
- **활성 연산 앱**: 0건 (`scratch/task0065_out.txt:66-68`)
- **기존 산출물 보존**: 삭제·수정 0건, 부분 파일 `p6_gen_B2.jsonl` 포함 전체 파일 보존
  - `results_s7/p6_gen_B2.jsonl`: 490행, SHA-256 `483c99f49cdff209798f723d2cd341eac9cc34b61ecd1d3abc5db043c41679d0` (`scratch/task0065_out.txt:75`)
- **정지 깃발 파일 생성**:
  - 경로: `/home/shoon/cts_e1_s5/raw_s7/P6_CHAIN_STOPPED_DEFECT.flag` (`scratch/task0065_out.txt:85-87`)
  - 내용: `2026-10-07 14:07:00 KST AUDIT-0064 stop-set defect, Principal decision`
  - SHA-256: `945ce5390b3188e08747bad01e621c7d241281587c45ea1f43e1762ac9adbe39` (`scratch/task0065_out.txt:85`)

---

## 3. (B) 수정 생성기 `scripts/s7_eval_gate_p6_v36c.py` 배포 및 변경 내역

### 3.1 파일 개요 및 해시 대조
- **로컬 경로**: `d:\AI\cts\scripts\s7_eval_gate_p6_v36c.py`
- **원격 경로**: `/home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36c.py`
- **크기**: 29,756 바이트 (`scratch/task0065_out.txt:93`)
- **SHA-256 해시**: `b18c59a29195e567a9e43e6de22b63bb55366e48769831f4f89fddc5b9396a6e` (`scratch/task0065_out.txt:94`)
- **원격 py_compile**: 에러 0건 컴파일 완결(`scratch/task0065_out.txt:95`)

### 3.2 v36b 대비 수정 사항 3종 상세
v36b(SHA `35541b67...`, 28,229 바이트) 대비 git diff는 오직 다음 3가지 항목에만 국한되며, 프롬프트·슬롯·투영·탐욕생성·z* 경로는 100% 바이트 단위로 동일합니다:

1. **FIX-1: 정지 집합 상수 고정 및 시작 검사 강화**
   - 상수로 선언: `EOS_IDS: Tuple[int, int] = (1, 106)` (`scripts/s7_eval_gate_p6_v36c.py:98`)
   - 시작 시 토크나이저 일치 검사: `tokenizer.eos_token_id == 1` 및 `tokenizer.convert_tokens_to_ids("<turn|>") == 106`가 아니면 즉시 중단 (`scripts/s7_eval_gate_p6_v36c.py:428-436`)
   - `<end_of_turn>` 문자열 인코딩 루프 완전 제거
   - 시작 시 정지 집합 및 매핑 조각 명시적 로그 출력:
     `[EOS-SET] Strict stop tokens locked to (1, 106) -> {1: '<eos>', 106: '<turn|>'}` (`scratch/task0065_out.txt:116`)

2. **FIX-2: 문항 순서 파일 단순 읽기 전용화**
   - `results_s7/p6_item_order.json` 파일을 오직 읽기만 수행 (기존 파일 덮어쓰기 코드 제거) (`scripts/s7_eval_gate_p6_v36c.py:121-152`)
   - 파일 로드 시 실제 SHA를 계산하여 출력 (`scripts/s7_eval_gate_p6_v36c.py:136`)
   - 선택적 인자 `--order_sha` 제공 시 불일치하면 즉각 실행 중단 (`scripts/s7_eval_gate_p6_v36c.py:139-141`)

3. **FIX-3: 출력 경로 보호 가드 (`--out` 필수화)**
   - 인자 `--out` 기본값 제거 및 필수(required=True) 설정 (`scripts/s7_eval_gate_p6_v36c.py:321`)
   - 출력 경로가 `results_s7/p6r_gen_*` (또는 스모크 시험용 `scratch/`)로 시작하지 않으면 즉각 차단하여 원본 P6 산출물 보호 (`scripts/s7_eval_gate_p6_v36c.py:350-362`)

### 3.3 채점 스크립트 재사용 지침
채점은 기존 스크립트 `scripts/s7_eval_score_p6_v36.py`(크기 18,786 바이트, SHA `547e1d91aa998908a0c436198f9890be961f18d643efec2c43dd700d415587cd`)를 별도 복제 없이 그대로 사용합니다.
- **실행 명령 양식**:
  ```bash
  python3 scripts/s7_eval_score_p6_v36.py \
      --gen_A results_s7/p6r_gen_A.jsonl \
      --gen_B1 results_s7/p6r_gen_B1.jsonl \
      --gen_C results_s7/p6r_gen_C.jsonl \
      --gen_E1 results_s7/p6r_gen_E1.jsonl \
      --out_json results_s7/p6r_score_main.json \
      --out_md results_s7/p6r_score_main.md
  ```

---

## 4. (C) 시험 결과

### 4.1 로컬 합성 시험 (Local Synthetic Tests)
`scratch/test_v36c_synthetic.py` 실행 결과: 3대 필수 차단 조건 모두 충족:
1. **잘못된 `--out` 경로 거부**:
   - `results_s7/p6_gen_A.jsonl` 지정 시 원본 보호 가드에 의해 즉시 차단 (exit 1)
   - 임의 파일명 지정 시 즉시 차단 (exit 1)
   - 스모크 모드에서 `scratch/` 허용
2. **순서 파일 해시 불일치 차단**:
   - 의도적 잘못된 `--order_sha` 전달 시 불일치 오류 출력 및 즉시 차단 (exit 1)
   - 올바른 해시 제공 시 대조 규격 일치
3. **정지 토큰 이상 시 차단**:
   - `tokenizer.eos_token_id != 1` 모의 시 즉각 중단 (exit 1)
   - `tokenizer.convert_tokens_to_ids("<turn|>") != 106` 모의 시 즉각 중단 (exit 1)

### 4.2 서버 CPU SMOKE 시험 실측 (`CUDA_VISIBLE_DEVICES=""`, 2문항, max_new_tokens 16)
원격 서버 가상환경 Python(`/home/shoon/system1_5/.venv/bin/python`)을 통해 CPU 스모크 시험 2회를 실행:

1. **조건 C (Pure Gemma 텍스트 모드)**:
   - **실행 명령**:
     `python scripts/s7_eval_gate_p6_v36c.py --cond C --smoke --max_new_tokens 16 --out scratch/smoke_p6r_C.jsonl --device cpu`
   - **반환 코드**: 0 (소요 시간: 29.30초, `scratch/task0065_out.txt:100`)
   - **정지 집합 원문 로그**:
     `[EOS-SET] Strict stop tokens locked to (1, 106) -> {1: '<eos>', 106: '<turn|>'}` (`scratch/task0065_out.txt:116`)
   - **출력 파일**: `scratch/smoke_p6r_C.jsonl` (2문항 생성 완결)
   - **산출물 SHA-256**: `89a86c516758cdd6897f8d9caff39a9a2130ced8dcd4c088ed47edb2a1cd8e82` (`scratch/task0065_out.txt:187`)

2. **조건 A (PrefixProjector + z* 주입 모드)**:
   - **실행 명령**:
     `python scripts/s7_eval_gate_p6_v36c.py --cond A --smoke --max_new_tokens 16 --out scratch/smoke_p6r_A.jsonl --device cpu`
   - **반환 코드**: 0 (소요 시간: 30.92초, `scratch/task0065_out.txt:203`)
   - **정지 집합 원문 로그**:
     `[EOS-SET] Strict stop tokens locked to (1, 106) -> {1: '<eos>', 106: '<turn|>'}` (`scratch/task0065_out.txt:220`)
   - **정전 프롬프트 감사**: §3(b) 슬롯 [4..67] 및 68번 개행(108) 엄격 대조 일치 규명 (`scratch/task0065_out.txt:314`)
   - **출력 파일**: `scratch/smoke_p6r_A.jsonl` (2문항 생성 완결)
   - **산출물 SHA-256**: `74b9290863e1a12794a5269198c854b25c7277099ba1cd24af8ff7ed2c90d5e0` (`scratch/task0065_out.txt:324`)

---

## 5. (C-3) 재현 점검 후보 10문항 선정 (CPU 진단)

### 5.1 선정 기준
기존 P6 산출물 중 오염 정지 집합의 영향을 전혀 받지 않은 문항을 식별하기 위해 다음 3대 조건을 엄격 적용하였습니다:
1. `tokens` < 512 (상한 512토큰에 도달하기 전 자연 정지)
2. 생성 텍스트를 재토큰화했을 때 의심 ID 6종(`[643, 887, 1340, 236779, 236813, 236820]`)이 0건
3. 2번 조건에 따라 v36b의 정지 집합(`{1, 106} U 의심 6종`) 중 정지 사유는 수학적으로 오직 `{1, 106}`에만 해당함이 규명됨
4. 위 조건을 충족하는 문항 중 문항 순서(q_idx)가 앞서는 상위 5문항 선택

### 5.2 선정 결과 및 대조표
선정된 설정 파일 경로:
`/home/shoon/cts_e1_s5/configs/p6r_repro_check10.json` (`scratch/task0065_out.txt:359`)
- **설정 파일 SHA-256**: `cf5bab9778bfec3e3ac5f4017c591b714ac9bb4b5cdd475b580bdfcab81b5cff` (`scratch/task0065_out.txt:359`)

#### 조건 A 선정 5문항 (`scratch/task0065_out.txt:343-350`)
| 순번 | 문항 ID | 순서 색인 (q_idx) | 생성 토큰 수 | 최종 조각 (Piece) | 생성 텍스트 SHA-256 (앞 16자리) |
|---|---|---|---|---|---|
| 1 | Q#0006 | 0 | 2 | `.` | `cdb4ee2aea69cc6a...` |
| 2 | Q#0043 | 3 | 203 | `}}$.` | `fd9709111e247ca5...` |
| 3 | Q#0045 | 4 | 68 | `}}$.` | `bb2e0ab9f305fcf5...` |
| 4 | Q#0067 | 5 | 171 | `}}$.` | `87f36a0f9ec0dd8b...` |
| 5 | Q#0071 | 6 | 27 | `?` | `d32832a54f495a23...` |

#### 조건 C 선정 5문항 (`scratch/task0065_out.txt:351-358`)
| 순번 | 문항 ID | 순서 색인 (q_idx) | 생성 토큰 수 | 최종 조각 (Piece) | 생성 텍스트 SHA-256 (앞 16자리) |
|---|---|---|---|---|---|
| 1 | Q#0067 | 5 | 81 | `$.` | `79e54e93008c8e4c...` |
| 2 | Q#0093 | 9 | 272 | `.` | `e5f14a77330c12aa...` |
| 3 | Q#0102 | 11 | 94 | `$.` | `42996c2687332b4f...` |
| 4 | Q#0127 | 15 | 295 | `.` | `45798003f6a2027f...` |
| 5 | Q#0147 | 16 | 295 | `.` | `68c78982fc9b3ebc...` |

이 10개 문항은 이전 실행에서도 조기 절단 오염 없이 종료되었으므로, 다음 차수에서 v36c로 GPU 0에서 재생성 시 100% 동일한 문장을 도출해야 하는 재현성 판정 기준점이 됩니다.

---

## 6. (C-4) 주요 참조 파일 해시 및 크기 총괄표

원격 서버 실측 기준(`scratch/task0065_out.txt:364-369`):

| 대상 항목 | 파일 경로 | 크기 (바이트) | SHA-256 해시 |
|---|---|---|---|
| 정전 문항 순서 | `results_s7/p6_item_order.json` | 5,939 | `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c` |
| 잠재 소스 텐서 (heldout 600) | `results_s7/zsrc_heldout600.pt` | 393,428,223 | `0da3748af2a3776c7fa91da50268a7829233939f56218d73f4b4511a6a8e5cff` |
| 디코더 A 체크포인트 | `results_s7/decoder_A_v36.pt` | 236,027,279 | `2953ff07f7419abef364c2906c62f8d5ae829d15f110fc5b0ef7266eae87a32c` |
| 디코더 E 체크포인트 | `results_s7/decoder_E_v36.pt` | 236,027,215 | `f8115ac33f44045ce5f572a377ee84b2e6311be347f154190b6c79199d7f3cac` |
| 조건 A 기본 z* (520 lowest) | `results_s7/zstar_s7_a0.25_heldout520_lowest.pt` | 686,625,069 | `365ac9a44150546dd424c33f2c5ae9cabc71f474f3f2cfd5e2d042fe6b4061ae` |
| 조건 A 기본 z* (80 400) | `results_s7/zstar_s7_a0.25_400.pt` | 262,341,649 | `bbe7ad3bd2758b4e3ef622ef7c5270307afa5831e4ec6748ca24a126078e79ca` |

---

## 7. 결론 및 차기 지침 준비

1. **결함 체인 중단 완결**: 결함 도구로 가동 중이던 P6 보조 체인을 즉시 정지하였으며, GPU 0 VRAM 13 MiB 상태로 깨끗이 반환되었습니다. 기존 산출물(부분 산출물 490행 포함)은 훼손 없이 보존되었습니다.
2. **도구 교정 완결**: 정지 토큰을 `{1, 106}`으로 엄격 격리하고 `--out` 보호 가드를 갖춘 `scripts/s7_eval_gate_p6_v36c.py`가 배포 및 시험되었습니다.
3. **재생성 준비 완결**: 재현 점검 대상 10문항 및 주요 모델/텐서 참조 해시가 집계되었으며, Principal 결정에 따른 주 게이트 4조건(A, B1, C, E1) 재생성 및 G2 재평가 지침(TASK-0066) 수신 대기 상태로 전환합니다.

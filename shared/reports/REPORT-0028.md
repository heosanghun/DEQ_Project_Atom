<!-- SHA256: bed95acc68f3892aa06e887aef423816c8f7eb9a338c7fbff3992a30c775b78f -->
# REPORT-0028: run_train 바이트코드 명령 열 비교 및 P4'·P5b 진행 상태 계측

- **작성 일시**: 2026-10-05 12:45 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, D:\AI\cts)
- **수행 근거**: shared/tasks/TASK-0028.md (G2 앞당김 게이트 연계, AUDIT-0027 남은 점검 사항)
- **선행 보고**: shared/reports/REPORT-0027.md (학습 코드 동일성 긴급 판별 및 P5b 재개)
- **절대 조건 준수 현황**:
  - 실행 중 세션(P4' tmux `s7_train_p4_prime`, P5b tmux `p5b_gpu2`, `p6_chain`) 무간섭: 준수
  - P5 본 실행 차단(실행 수 0건): 준수
  - 텐서 결과 값 열람 0건(진행 카운트 및 키 개수만 집계): 준수
  - 기존 파일 제자리 수정 0건: 준수
  - 금지어 0건 원칙: 준수

> **§1 명령 열 비교 핵심 결론**:  
> 등록 정본 바이트코드(`s7_train_decoder_v36.cpython-312.pyc`)와 현재 실행 소스(`f2951054...`)의 `run_train` 함수(재귀 중첩 포함)에 대해 Python 3.12 `dis.get_instructions` 명령 열 `(opname, argrepr)` 비교를 수행하였습니다.  
> 원시 `(opname, argrepr)` 기준 101개 블록이 검출되었으나, 이는 781~805행 RESUME 블록 삽입(134바이트)에 따른 **하류 점프 목적지 바이트 오프셋 시프트 92개** 및 **중첩 제너레이터 코드 객체 힙 메모리 주소 표기 차이 3개**에 기인합니다.  
> **실질 연산 부호(Opcode) 변경 블록은 정확히 6개이며, 6개 전건이 781~805행 RESUME 블록에만 100% 국한**되어 있습니다. 비-RESUME 영역의 실질 명령 차이는 **0건**입니다.

---

## 1. `run_train` 바이트코드 명령 열 비교 (서버 Python 3.12 실측)

### 1.1 비교 대상 입력 사양 및 명령 열 추출
- **등록 정본 바이트코드**:
  - 파일: `scripts/__pycache__/s7_train_decoder_v36.cpython-312.pyc`
  - SHA-256: `17d9213328804168a44f6bbc5548c3cabdf93b69251ccdd039228fe829fd072d` (86,464 바이트)
  - `run_train` 코드 객체: 상수 224개, 이름 130개, 바이트코드 길이 13,844 바이트
  - 재귀 추출 명령 총수: 3,158개 명령
- **현재 실행 소스 컴파일 본**:
  - 파일: `scripts/s7_train_decoder_v36.py`
  - SHA-256: `f2951054416c7b1593075c6a94e3c1ab7e7e7be516f078d540b5bd9ce5bf3ea9` (67,396 바이트)
  - `run_train` 코드 객체: 상수 226개, 이름 135개, 바이트코드 길이 14,490 바이트
  - 재귀 추출 명령 총수: 3,292개 명령

### 1.2 `difflib.SequenceMatcher` 비교 분석 결과
오프셋과 행 번호를 배제하고 `(opname, argrepr)` 열에 대해 `SequenceMatcher`를 수행한 결과입니다.

- **일치 블록 (`equal`)**: 102개 구간
- **차이 블록 (`replace`)**: 101개 구간

### 1.3 차이 블록 101개의 원인별 정밀 분류
101개 차이 블록의 세부 연산 부호를 전수 분류한 결과는 다음과 같습니다.

| 분류 유형 | 블록 수 | 해당 블록 번호 | 기술적 발생 원인 |
|:---|:---:|:---|:---|
| **코드 객체 메모리 주소 차이** | 3개 | 블록 #1, #2, #35 | CPython 런타임 힙 주소(`at 0x...`) 표기 차이 (735행, 758행, 893행) |
| **점프 목적지 오프셋 시프트** | 92개 | 블록 #3~#6, #9~#10, #13~#34, #36~#91, #94~#101 | 781~805행 패치 삽입(134B)에 따른 `FOR_ITER`, `POP_JUMP_IF_*`의 `to <offset>` 절대 번지 이동 |
| **실질 Opcode 변경 (RESUME 블록)** | **6개** | **블록 #7, #8, #11, #12, #92, #93** | **781~805행 RESUME 경로 RNG ByteTensor CPU 변환 및 예외 처리 로직** |

### 1.4 실질 차이 블록 6건 세부 대조 (781~805행 국한)

#### (1) 블록 #7 (소스 781~787행: `torch_rng_state` ByteTensor 변환)
- **태그**: `replace` | pyc[1022:1023] (길이 1) vs src[1023:1062] (길이 39)
- **직전 5개 명령**: `[('CALL', ''), ('POP_TOP', ''), ('LOAD_CONST', "'torch_rng_state'"), ('LOAD_FAST_CHECK', 'res_ckpt'), ('CONTAINS_OP', '')]`
- **직후 5개 명령**: `[('LOAD_GLOBAL', 'NULL + torch'), ('LOAD_ATTR', 'set_rng_state'), ('LOAD_FAST_CHECK', 't_rng'), ('CALL', ''), ('POP_TOP', '')]`
- **pyc 명령**: `POP_JUMP_IF_FALSE to 4604`
- **src 추가 명령**: `res_ckpt["torch_rng_state"]` 적재, `isinstance(t_rng, torch.Tensor)` 분기, `t_rng.to(dtype=torch.uint8, device='cpu')`, `torch.ByteTensor` 변환 등 39개 명령.

#### (2) 블록 #8 (소스 788행: `torch.set_rng_state` 인자 치환)
- **태그**: `replace` | pyc[1025:1028] (길이 3) vs src[1064:1065] (길이 1)
- **직전 5개 명령**: `[('CALL', ''), ('STORE_FAST', 't_rng'), ('NOP', ''), ('LOAD_GLOBAL', 'NULL + torch'), ('LOAD_ATTR', 'set_rng_state')]`
- **직후 5개 명령**: `[('CALL', ''), ('POP_TOP', ''), ('LOAD_CONST', "'cuda_rng_state'"), ('LOAD_FAST_CHECK', 'res_ckpt'), ('CONTAINS_OP', '')]`
- **pyc 명령**: `LOAD_FAST_CHECK res_ckpt`, `LOAD_CONST 'torch_rng_state'`, `BINARY_SUBSCR`
- **src 치환 명령**: `LOAD_FAST_CHECK t_rng`

#### (3) 블록 #11 (소스 791~797행: `cuda_rng_state` ByteTensor 변환)
- **태그**: `replace` | pyc[1042:1043] (길이 1) vs src[1079:1114] (길이 35)
- **직전 5개 명령**: `[('POP_JUMP_IF_NONE', 'to 5222'), ('LOAD_GLOBAL', 'torch'), ('LOAD_ATTR', 'cuda'), ('LOAD_ATTR', 'NULL|self + is_available'), ('CALL', '')]`
- **직후 5개 명령**: `[('LOAD_GLOBAL', 'torch'), ('LOAD_ATTR', 'cuda'), ('LOAD_ATTR', 'NULL|self + set_rng_state'), ('LOAD_FAST_CHECK', 'c_rng'), ('CALL', '')]`
- **pyc 명령**: `POP_JUMP_IF_FALSE to 4750`
- **src 추가 명령**: `c_rng` 추출, `torch.Tensor` 판별, `uint8` CPU 텐서 변환 로직 35개 명령.

#### (4) 블록 #12 (소스 798행: `torch.cuda.set_rng_state` 인자 치환)
- **태그**: `replace` | pyc[1046:1049] (길이 3) vs src[1117:1118] (길이 1)
- **직전 5개 명령**: `[('STORE_FAST', 'c_rng'), ('NOP', ''), ('LOAD_GLOBAL', 'torch'), ('LOAD_ATTR', 'cuda'), ('LOAD_ATTR', 'NULL|self + set_rng_state')]`
- **직후 5개 명령**: `[('CALL', ''), ('POP_TOP', ''), ('LOAD_CONST', "'numpy_rng_state'"), ('LOAD_FAST_CHECK', 'res_ckpt'), ('CONTAINS_OP', '')]`
- **pyc 명령**: `LOAD_FAST_CHECK res_ckpt`, `LOAD_CONST 'cuda_rng_state'`, `BINARY_SUBSCR`
- **src 치환 명령**: `LOAD_FAST_CHECK c_rng`

#### (5) 블록 #92 & #93 (소스 789~805행: 예외 처리 핸들러)
- **블록 #92**: pyc[3027:3028] (길이 1) vs src[3105:3115] (길이 10) — `torch_rng_state` 설정 시 예외 포획 및 경고 메시지 출력 핸들러.
- **블록 #93**: pyc[3030:3032] (길이 2) vs src[3117:3166] (길이 49) — `cuda_rng_state` 및 `numpy_rng_state` 예외 포획 핸들러.
- CPython 3.12 컴파일러는 `try...except` 핸들러 블록의 바이트코드를 함수의 말미(오프셋 3000번대)에 배치하므로 해당 위치에 블록 번호가 부여되었습니다.

### 1.5 비-RESUME 블록 실질 차이 점검 결과
- 781~805행 이외의 소스 범위에 위치한 차이 블록(블록 #1~#6, #14~#91, #94~#101)은 모두 점프 목적지 오프셋 번지 차이(`to 4862` vs `to 5334`, `to 13196` vs `to 13684`) 또는 제너레이터 힙 주소 표기 차이였습니다.
- 점프 오프셋 및 힙 주소를 정규화할 경우, 비-RESUME 영역(학습 루프 전체, 손실 계산, 모델 전향 전파, AdamW 최적화기 갱신, 체크포인트 저장 등)의 **실질 Opcode 불일치는 정확히 0건**입니다.

---

## 2. 서버 프로세스 및 가동 상태 (TASK-0028 §2 실측)

### 2.1 P4' 디코더 직렬 학습 세션 (tmux `s7_train_p4_prime`)
- **디코더 A 가동 상태**:
  - `raw_s7/train_decoder_A_v36.log`: 누적 554행 기록 중.
  - 최신 기록 스텝: **Step 2700 / 3000** 가동 중.
  - 최종 산출물(`results_s7/decoder_A_v36.pt`) 존재 여부: `False` (현재 2700보 경과, 3000보 도달 전).
  - `[OUT-CHECKPOINT]` 출력 행: 미도달 (학습 진행 중).
- **디코더 D 가동 상태**:
  - `raw_s7/train_decoder_D_v36.log`: 미생성 (`False`, 디코더 A 종결 후 직렬 기동 예정).
- **점유 장치**: GPU 0 (CUDA_VISIBLE_DEVICES=0, 점유율 100%, VRAM 20,270 MiB, 40 °C).

### 2.2 P5b Held-Out 520 z* 재개 세션 (tmux `p5b_gpu2`)
- **종료 플래그 (`raw_s7/P5B_DONE.flag`)**: 부재 (`False`, 연산 진행 중).
- **출력 텐서 (`results_s7/zstar_s7_a0.25_heldout520_seed2031.pt`)**:
  - **현재 저장된 키 총수**: **405개 / 520개** (5문항 주기 체크포인트 규격 부합 저장 계측).
  - 로그 상 최신 처리 문항: `[0408/0520] q4010_a0.25` (건당 평균 22.8초 소요).
- **점유 장치**: GPU 2 (CUDA_VISIBLE_DEVICES UUID 고정, 점유율 87~100%, VRAM 30,844 MiB, 89 °C).

### 2.3 GPU 0~7 전체 물리 장치 현황 (`nvidia-smi`)

| Index | GPU 모델 | UUID | VRAM 점유 | 연산 점유율 | 온도 | 비고 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **GPU 0** | RTX A6000 | `GPU-71be98b0-6b81-9257-c622-97dc97d46480` | 20,270 MiB | **100 %** | 40 °C | P4' 디코더 A 학습 가동 중 |
| **GPU 1** | RTX A6000 | `GPU-9410691f-6857-1642-ec7c-7596970e2f48` | 19 MiB | **[N/A]** | 24 °C | 드라이버 `requires reset` 격리 |
| **GPU 2** | RTX A6000 | `GPU-c81311d8-d210-3857-843e-bb38d24771e1` | 30,844 MiB | **87 %** | 89 °C | P5b Held-Out 520 증분 계산 가동 중 |
| **GPU 3** | RTX A6000 | `GPU-a9fed959-7113-e747-dac4-a515ac50c160` | 19 MiB | 0 % | 28 °C | 유휴 (점유 0) |
| **GPU 4** | RTX A6000 | `GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5` | 19 MiB | 0 % | 26 °C | **영구 봉인 (점유 0)** |
| **GPU 5** | RTX A6000 | `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a` | 19 MiB | 0 % | 26 °C | 유휴 (점유 0) |
| **GPU 6** | RTX A6000 | `GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c` | 19 MiB | 0 % | 26 °C | 유휴 (점유 0) |
| **GPU 7** | RTX A6000 | `GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b` | 19 MiB | 0 % | 27 °C | 유휴 (점유 0) |

---

## 3. 종합 결론 및 다음 절차

1. **바이트코드 동일성 확정**:
   - `dis.get_instructions` 정밀 분석 결과, `run_train` 내 실질 Opcode 차이는 781~805행 RESUME 블록 6건에만 100% 국한됨이 실증되었습니다.
   - 비-RESUME 영역(학습 모델, 손실, 데이터셋, 최적화 루프)의 실질 명령 차이는 0건이며, P4' 학습의 무결성이 바이트코드 수준에서 확정되었습니다.
2. **연산 진행 현황**:
   - P4' 디코더 A는 2700보를 넘어 3000보 종결점으로 순항 중입니다.
   - P5b는 물리 GPU 2 고정 격리를 통해 405개 문항을 돌파하여 순항 중입니다.
3. **다음 작업**:
   - `shared/STATE.md` 갱신 후 `shared/wait_for_task.py` 대기 명령으로 복귀하여 후속 지시를 대기합니다.

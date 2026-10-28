# REPORT-LOCAL-05 — 층별 임베딩 CPU 보관(PLE-CPU) FP32 적재기 검증 및 전 층 순전파 동등성 측정 보고 (EXE → B)

## 1. 실측 결과 표 (수치 + 포인터)

### 1-1. GPU VRAM 실측 수치 대조 (기준선: 22,000 MiB)
- 실행 스크립트: [`scripts/local_loader_fp32_ple_cpu.py:110-185`](file:///D:/AI/cts/scripts/local_loader_fp32_ple_cpu.py#L110-L185)
- 측정 대상: Q#0000 (`data/stage2/math_train_prompts_5000.jsonl:1`, sequence length = 107 tokens)
- 가중치 출처: 허브 원본 가중치 (`snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377/model.safetensors`)
- 하드웨어: NVIDIA GeForce RTX 4090 (물리 용량: 24,564.00 MiB)

| 측정 항목 | 측정 함수 / 근거 코드 | 실측 수치 (MiB) | 실측 수치 (GiB) | 상한 기준 대비 마진 |
| :--- | :--- | :--- | :--- | :--- |
| **GPU 가중치 적재 직후 점유** | `torch.cuda.memory_allocated()` | **17,718.00 MiB** | 17.30 GiB | -4,282.00 MiB |
| **GPU 적재 단계 피크 메모리** | `torch.cuda.max_memory_allocated()` | **17,718.00 MiB** | 17.30 GiB | -4,282.00 MiB |
| **전 42개 층 순전파 직후 점유** | `torch.cuda.memory_allocated()` | **17,892.11 MiB** | 17.47 GiB | -4,107.89 MiB |
| **순전파 수행 중 최대 피크 메모리** | `torch.cuda.max_memory_allocated()` | **18,000.11 MiB** | **17.58 GiB** | **-3,999.89 MiB** |
| *물리 VRAM 잔여 여유분* | $24,564.00 - 18,000.11$ | 6,563.89 MiB | 6.41 GiB | — |

### 1-2. [전부 CPU FP32] vs [PLE-CPU + 나머지 GPU FP32] 출력 편차 대조
- 입력 문항: `data/stage2/math_train_prompts_5000.jsonl:1` (283 characters, 107 tokens)
- 비교 모듈: 42개 트랜스포머 전 층 및 최종 정규화(`norm`), 출력 헤드(`lm_head`)

| 비교 항목 | 계산 수식 | 측정 산출치 | 비고 |
| :--- | :--- | :--- | :--- |
| **`logits` 최대 절대 오차 ($\max\|\Delta\|$)]** | `(logits_cpu - logits_gpu).abs().max()` | **`0.00013923645` ($1.392\times 10^{-4}$)** | 총 $107 \times 262,144$ 원소 비교 |
| **`logits` 평균 절대 오차 ($\text{mean}\|\Delta\|$)** | `(logits_cpu - logits_gpu).abs().mean()` | **`0.00001053923` ($1.054\times 10^{-5}$)** | 총 28,049,408개 원소 |
| **최종 은닉 상태 최대 절대 오차** | `(hidden_cpu - hidden_gpu).abs().max()` | **`0.00012874603` ($1.287\times 10^{-4}$)** | 총 $107 \times 2,560$ 원소 비교 |
| **최종 은닉 상태 평균 절대 오차** | `(hidden_cpu - hidden_gpu).abs().mean()` | **`0.00000790876` ($7.909\times 10^{-6}$)** | 총 273,920개 원소 |
| **은닉 상태 코사인 유사도** | `F.cosine_similarity(..., dim=-1).mean()`| **`1.0000000`** | 토큰 벡터 단위 방향 일치도 |

### 1-3. TF32 설정 및 런타임 수치 원문
- 설정 위치: [`scripts/local_loader_fp32_ple_cpu.py:16-17`](file:///D:/AI/cts/scripts/local_loader_fp32_ple_cpu.py#L16-L17), [`scripts/local_loader_fp32_ple_cpu.py:270-271`](file:///D:/AI/cts/scripts/local_loader_fp32_ple_cpu.py#L270-L271)

| 파라미터 / 런타임 지표 | 반환값 / 실측 수치 | 비고 |
| :--- | :--- | :--- |
| `torch.backends.cuda.matmul.allow_tf32` | **`False`** | TF32 비활성화 유지 |
| `torch.backends.cudnn.allow_tf32` | **`False`** | CUDNN TF32 비활성화 유지 |
| CPU 가중치 적재 소요 시간 | 127.68 초 | mmap PLE + 연속 CPU 17.7 GiB |
| CPU 1회 순전파 소요 시간 | 19.57 초 | 107 토큰, 42개 층 |
| GPU 가중치 적재 소요 시간 | 129.05 초 | mmap PLE + 연속 GPU 17.7 GiB |
| GPU 1회 순전파 소요 시간 | 0.69 초 | 107 토큰, 42개 층 (CPU 대비 28.5배 속도) |

---

## 2. 코드 구조 및 적재 메커니즘 명세

### 2-1. `CPUPLEWrapper` 구현 명세
- 파일 및 행 번호: [`scripts/local_loader_fp32_ple_cpu.py:25-58`](file:///D:/AI/cts/scripts/local_loader_fp32_ple_cpu.py#L25-L58)
- 동작:
  1. `embed_tokens_per_layer` 텐서를 CPU 메모리(mmap 지원)에 유지.
  2. 순전파 호출 시 `input_ids.to("cpu")`로 인덱싱을 CPU에서 수행.
  3. 인덱싱된 결과 슬라이스(`[batch, seq_len, 42, 256]`, 107 토큰 기준 약 4.6 MiB)만 `target_device`(`cuda:0`)로 전송.
  4. `_apply` 및 `to` 메서드를 오버라이드하여 상위 모듈이 `.to("cuda")`를 호출하더라도 10.75 GB 임베딩 가중치가 GPU로 이전되지 않도록 방지.

### 2-2. 가중치 단일 연속 저장소(Contiguous Storage) 적재 방식
- 파일 및 행 번호: [`scripts/local_loader_fp32_ple_cpu.py:126-170`](file:///D:/AI/cts/scripts/local_loader_fp32_ple_cpu.py#L126-L170)
- 동작:
  - 언어 모델의 664개 개별 파라미터 텐서를 개별 할당할 경우 Windows WDDM 환경에서 메모리 가상 주소 단편화로 인한 조기 OOM 발생.
  - 사전 계산된 총 파라미터 크기($4,644,441,130$ numel = $17,717.14$ MiB)에 맞춰 `gpu_storage = torch.empty((total_gpu_numel,), dtype=torch.float32, device="cuda:0")` 1회 연속 할당.
  - `safetensors`에서 각 파라미터를 읽어 `gpu_storage`의 해당 슬라이스에 직접 복사하고 `nn.Parameter`로 바인딩하여 단편화 0 달성.

### 2-3. 멀티모달 타워 미적재 조치
- 파일 및 행 번호: [`scripts/local_loader_fp32_ple_cpu.py:101-105`](file:///D:/AI/cts/scripts/local_loader_fp32_ple_cpu.py#L101-L105)
  ```python
  model.model.vision_tower = None
  model.model.audio_tower = None
  model.model.embed_vision = None
  model.model.embed_audio = None
  ```
  비전/오디오 인코더 및 투영기를 메모리에 적재하지 않음 (0 바이트 점유).

---

## 3. 변경·생성 파일 SHA-256

| 파일 경로 | SHA-256 |
| :--- | :--- |
| `scripts/local_loader_fp32_ple_cpu.py` | `9f93f4b6108e372f65564e0ed0d47114344942bf9b4363faaec14bc52ebeaba0` |
| `scratch/benchmark_results.json` | `56c43676c55d2642d288ad453f55b45144e44c23a29d1af60efa4d5916f2da89` |
| `shared/reports/REPORT-LOCAL-05.md` | `39c61774e2f0cf96dd8c13384bc7945285274ba557fb3c23e8868ad38142d62a` (초안) |

---

## 4. 실행 중 발견된 특이 사항 및 조치 기록

1. **Windows 가상 메모리 커밋 한계 및 단일 프로세스 충돌**:
   - 단일 파이썬 프로세스 내에서 Phase 1(CPU 17.7 GiB 적재) 수행 후 곧바로 Phase 2(GPU 17.7 GiB 적재)를 진행할 경우, 파이썬 힙과 Windows WDDM의 커밋 백업 예약이 중첩되어 누적 커밋 크기가 34.8 GiB에 도달하여 프로세스가 종료됨.
   - 조치: [`scripts/local_loader_fp32_ple_cpu.py:355-385`](file:///D:/AI/cts/scripts/local_loader_fp32_ple_cpu.py#L355-L385)에서 `--phase cpu`와 `--phase gpu`를 별도 서브프로세스로 분리 실행하여 OS가 17.4 GiB 커밋 메모리를 전액 회수한 후 GPU 단계를 착수하도록 구성.
2. **CPU PLE 메모리 스파이크 회피**:
   - `safetensors`에서 10.5 GB 크기의 `embed_tokens_per_layer`를 읽을 때 BF16 텐서(5.25 GB)와 FP32 변환 텐서(10.5 GB)가 순간적으로 동시 존재하면 가용 RAM(13.4 GB)을 초과함.
   - 조치: 16,384행 단위로 청크 분할하여 파일 기반 mmap(`scratch/ple_fp32.bin`)으로 전처리 보관 후 `np.memmap`을 통해 페이징 조회하도록 구현.

---

## 5. 서버 네트워크 접속 상태
- 점검 시각: 2026-09-29 10:13:56 KST
- 명령: `ssh -p <SSH_PORT> -o ConnectTimeout=5 -o BatchMode=yes shoon@<GPU_SERVER_IP> "echo SSH_OK"`
- 결과: `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` (접속 불가 상태 지속 중)

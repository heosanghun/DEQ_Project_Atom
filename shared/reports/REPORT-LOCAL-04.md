# REPORT-LOCAL-04 — 로컬 FP32 적재 가능성 정량 분석 및 채점기 이식성 보고 (EXE → B)

## 1. 결과 표 (수치 + 포인터)

### 1-1. 로컬 스냅샷 파라미터 수 및 FP32 가중치 메모리 정량 계산 (실행 0, 순수 계산)
- 스냅샷 출처: `snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377/config.json:55-145`, `model.safetensors` 메타데이터 헤더
- 바이트 계산 기준: FP32 = 파라미터 수 $\times$ 4 bytes (`scratch/harness_A_v1.py:33`의 `torch_dtype=torch.float32` 기준)

| 모듈 구성 요소 | 텐서 형상 (Shape) | 파라미터 수 (Numel) | FP32 바이트 (Bytes) | FP32 환산 메모리 (MiB) | 비중 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **언어 모델 단어 임베딩 (`embed_tokens`)** | `[262144, 2560]` | 671,088,640 | 2,684,354,560 | 2,560.00 MiB | 8.4% |
| **층별 입력 임베딩 (`embed_tokens_per_layer`)** | `[262144, 10752]` | 2,818,572,288 | 11,274,289,152 | 10,752.00 MiB | 35.2% |
| **임베딩 소계** | — | **3,489,660,928** | **13,958,643,712** | **13,312.00 MiB** | **43.6%** |
| 트랜스포머 42개 층 (`layers.0`~`layers.41`) | 42층 (Attn + MLP + Norms) | 4,000,880,170 | 16,003,520,680 | 15,262.15 MiB | 50.0% |
| 최종 레이어 정규화 (`norm.weight`) | `[2560]` | 2,560 | 10,240 | 0.01 MiB | 0.0% |
| **언어 모델(LM) 백본 소계** | — | **7,490,543,658** | **29,962,174,632** | **28,574.16 MiB** | **93.7%** |
| 비전 타워 (`vision_tower`) | 패치 임베더 + 16층 | 169,331,136 | 677,324,544 | 645.95 MiB | 2.1% |
| 오디오 인코더 (`audio_encoder`) | 컨볼루션 + 12층 | 308,757,248 | 1,235,028,992 | 1,177.82 MiB | 3.9% |
| 기타 프로젝션 (`embed_audio/vision`) | `[2560, 1536]` 등 | 27,525,376 | 110,101,504 | 105.00 MiB | 0.3% |
| **전체 기저 모델 가중치 합계** | **2,130개 텐서** | **7,996,157,418** | **31,984,629,672** | **30,502.92 MiB (29.79 GB)** | **100.0%** |

### 1-2. 가중치 합계 대조 및 4090 물리 한계 비교
- 포인터: `snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377/model.safetensors`, `shared/logs/server_status_TASK0002_fix.log:13`, `shared/STATE.md:3`

| 비교 항목 | 계산 기준 / 측정 출처 | 메모리 크기 (MiB / GB) | 비고 |
| :--- | :--- | :--- | :--- |
| **언어 모델(LM) 백본 가중치 계산치** | 순수 LM 가중치 ($7.49\times 10^9$ numel $\times$ 4) | **28,574.16 MiB (27.90 GB)** | 독립 계산치 (단어+층별 임베딩+42개 층) |
| **전체 기저 모델 가중치 계산치** | 멀티모달 전체 ($8.00\times 10^9$ numel $\times$ 4) | **30,502.92 MiB (29.79 GB)** | 독립 계산치 (비전/오디오 포함) |
| **로컬 RTX 4090 총 물리 VRAM** | 로컬 하드웨어 사양 한계 (`nvidia-smi`) | **24,564.00 MiB (24.00 GB)** | 물리적 상한선 |
| **LM 가중치 대비 4090 결손** | $28,574.16 - 24,564.00$ | **+4,010.16 MiB (+3.92 GB)** | **가중치 단독으로 물리 용량 초과** |
| **전체 가중치 대비 4090 결손** | $30,502.92 - 24,564.00$ | **+5,938.92 MiB (+5.80 GB)** | **물리 용량 대비 +24.2% 초과** |
| **서버 S7 GPU 0 실측 VRAM (참고)** | `server_status_TASK0002_fix.log:13` | 31,452.00 MiB (30.71 GB) | 가중치+런타임 오버헤드 실측치 (참고) |
| **서버 실측치 대비 4090 결손 (참고)** | $31,452.00 - 24,564.00$ | +6,888.00 MiB (+6.73 GB) | 물리 용량 대비 +28.0% 초과 (참고) |

- **정량 계산 요약**:
  1. 독립 계산치 기준으로 기저 언어 모델 가중치(28,574.16 MiB)만 적재하더라도 RTX 4090 물리 한계(24,564.00 MiB)를 +4,010.16 MiB 초과함.
  2. 비전/오디오를 포함한 전체 가중치 계산치(30,502.92 MiB) 기준으로는 +5,938.92 MiB(5.80 GB) 결손이 발생함.
  3. 따라서 단일 RTX 4090(24 GB)에서 양자화·오프로딩 없는 순수 FP32 조건 단독 적재는 불가함(0건 실행).

---

### 1-4. Task 2 채점기 이식성 조치 및 명세
- 대상 파일: `cts/eval/grader_v2.py:160-162`
- 패키지 명세 파일: `pyproject.toml:36-40`, `requirements.txt:1-3`

#### (a) 의존성 고정 diff
```diff
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -33,6 +33,11 @@ train = [
 faiss = [
   "faiss-cpu>=1.7",
 ]
+eval = [
+  "math-verify==0.9.0",
+  "latex2sympy2_extended==1.11.0",
+  "sympy==1.14.0",
+]
 # Gemma 4 needs transformers with model_type gemma4 (often main branch until release):
```
- `requirements.txt` 신규 생성:
  ```
  math-verify==0.9.0
  latex2sympy2_extended==1.11.0
  sympy==1.14.0
  ```

#### (b) 플랫폼별 타임아웃 분기 코드 위치 명세
- 파일 및 행 번호: [`cts/eval/grader_v2.py:160-162`](file:///D:/AI/cts/cts/eval/grader_v2.py#L160-L162)
- 코드 원문:
  ```python
  p_kwargs = {"parsing_timeout": None} if sys.platform == "win32" else {}
  v_kwargs = {"timeout_seconds": None} if sys.platform == "win32" else {}
  p_parsed = _mv_parse(_mv_wrap(pred_str), **p_kwargs)
  g_parsed = _mv_parse(_mv_wrap(gold_str), **p_kwargs)
  if p_parsed and g_parsed and bool(_mv_verify(g_parsed, p_parsed, **v_kwargs)):
  ```
- **동작 분기**:
  - `Windows` (`sys.platform == "win32"`): 타임아웃 비활성화(`None`)로 `multiprocessing.spawn` 핸들 복제 오류 방지.
  - `Linux` (`sys.platform != "win32"`): 기본 타임아웃 유지 (`parsing_timeout=5`, `timeout_seconds=5`, POSIX fork 기반 감시).

#### (c) 서버 회복 후 대기 과업 기록
- 서버 복구 시 Linux 환경(서버 venv)에서 [`tests/test_grader_v2.py`](file:///D:/AI/cts/tests/test_grader_v2.py) 20개 사례를 실행.
- 산출 로그를 서버 기보유 로그 [`raw_s7/unit_grader_TASK0002.log`](file:///D:/AI/cts/raw_s7/unit_grader_TASK0002.log) (SHA-256: `07989c72783ee7afaf5ebe50ce44472f15edc18bbc18ed99df45029c3791d0fd`)와 해시 대조 예정.

---

## 2. 변경·생성 파일 SHA-256

| 파일 경로 | SHA-256 |
| :--- | :--- |
| `pyproject.toml` | `7d4807f655b4f20b80a844e4be88ac4c0dbb788adb3e319391e519c683068f9b` |
| `requirements.txt` | `dee522d6cf6b90f339129f2173769bf0f7935c53b91cc12e63255cbcc53eedc0` |
| `shared/reports/REPORT-LOCAL-04.md` | 본 보고서 |

---

## 3. 예상 밖 사건
- 없음.

---

## 4. 서버 상태 1줄
- SSH 서버 접속: 2026-09-29 07:18:29 KST 기준 `ssh: connect to host <GPU_SERVER_IP> port <SSH_PORT>: Connection timed out` (장애 지속).

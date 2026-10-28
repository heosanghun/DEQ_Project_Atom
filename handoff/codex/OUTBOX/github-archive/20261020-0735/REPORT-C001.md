# TASK-C001 — X3-min 코드 독립 검토

작성자: Codex | 2026-10-20 00:44 KST | 작업 디렉터리: `C:\Project\CTS\shared`

## 0. 한 줄 요약

검토 대상 5개 파일을 읽어 잠재 제거 분기의 역전파 단절, 오류 전파 누락, 재개 시 실행 조건 혼합, 학습 표적 절단의 4개 결함 조건을 확인했고 모델 실행은 0회입니다.

## 1. 수행한 것

- `AGENTS.md`, 기존 `CONTEXT.md`, `INBOX/TASK-C001.md`, 계획서와 X1·X3 결과 문서를 읽었습니다.
- 사용자가 지정한 `codex_pkg_20261020`의 누락 자료를 `shared/`의 새 경로에 바이트 그대로 복사했습니다. 기존 `README.md`, `CONTEXT.md`는 변경하지 않았습니다. 패키지의 동명 문서는 `OUTBOX/PACKAGE-README.md`, `OUTBOX/PACKAGE-CONTEXT.md`로 별도 보존했습니다. 이 두 문서의 맥락도 읽었습니다.
- Python·모델·셀프테스트·셸 실행기를 실행하지 않았습니다. 읽기, 줄 검색, SHA-256 계산과 본 문서 작성만 수행했습니다. 코드 수정과 패치 파일 생성은 하지 않았습니다.
- rev2 결과 문서의 학습 파일 해시는 `fc705669…`인데 제공된 학습 파일은 rev3 `efc24925…`입니다. 이 검토는 제공된 rev3 파일 기준입니다. 서버에서 실행된 rev2 원본의 동일성을 확인하지 못했습니다.

## 2. 결과 표(숫자만)

| 검토 대상 파일 수 | TASK 항목 수 | 모델 실행 수 | 학습 실행 수 | 생성 실행 수 | 설치 수 |
|---:|---:|---:|---:|---:|---:|
| 5 | 9 | 0 | 0 | 0 | 0 |

## 3. 불일치·결함 목록(파일, 줄, 재현 조건)

### 항목별 결론

| TASK 항목 | 결론 | 근거와 확인 범위 |
|---|---|---|
| 1. 훅 정합성 | 결함 아님(제공 코드의 B=1·tensor/tuple 경로), 실제 ABI 판단 불가 | xattn 65–74행: tuple이면 0번 hidden만 바꾸고 나머지 원소를 보존합니다. tensor이면 tensor로 반환합니다. T에 종속된 고정 인덱스가 없으므로 B=1,T=1에서도 Q=[1,H,1,dh], K=[1,H,K,dh], 결과=[1,1,D]입니다. 실제 Gemma 구현·버전과 cache 객체는 패키지에 없어 실행 정합성은 확인하지 못함. |
| 2. 프롬프트·손실 마스크 | 결함 아님(동일 텍스트와 동일 tokenizer라는 조건), 절단 결함 있음 | train 20–23행 함수를 gen 14·64행에서 직접 재사용합니다. text_drop=0인 학습은 같은 입력 문자열의 같은 함수를 호출합니다. train 68·73–74행은 logits 위치 t에서 label t+1을 비교하므로 첫 해설 토큰을 예측하는 경계도 맞습니다. 실제 토큰 배열은 확인하지 못함. text_drop>0인 학습은 의도적으로 다릅니다. 긴 입력의 전부 -100 문제는 D4 참조. |
| 3. shuffled·id | 조건부 결함 아님 / 일부 판단 불가 | gen 62행의 shift=1은 n>1이고 id가 유일한 순서라면 다른 id를 선택합니다. 코드 자체는 유일성·n>1·shift%n!=0을 검사하지 않습니다. `extract_tensor` 구현과 latent pt 파일이 제공되지 않아 파일 키 매핑을 확인하지 못함. |
| 4. center·정규화 | 결함 아님(계산·저장 경로), 학습 집합 누설 여부 판단 불가 | train 39–41·53행은 --items에 든 잠재만 평균냅니다. xattn 53행 center는 persistent buffer이고 train 83행 state_dict에 포함됩니다. gen 49행 strict=True로 복원합니다. xattn 20·62–63행은 슬롯별 마지막 D축 RMS이며 eps=1e-6입니다. 실제 학습 id 파일과 pt가 없어 train/dev 교집합은 확인하지 못함. |
| 5. 초기 상태·rev2 | rev1 이중 0 초기화 결함은 제공 코드에서 제거됨 / 비트 동일성은 조건부 | O는 nn.Linear 기본 초기화, gate는 0입니다. delta=tanh(g)O(ctx)에 대해 g=0에서 d(delta)/dg=O(ctx)이므로 gate의 경로가 존재합니다. Q/K/V/O의 첫 단계 gradient는 gate=0 때문에 0이며, gate가 움직인 후 경로가 열립니다. 특정 손실과 입력에서 gate gradient가 반드시 0이 아니라고 보장할 수는 없습니다. 유한한 계산값이면 delta=0이나 NaN/Inf이면 0 곱셈으로 항등이 보장되지 않습니다. |
| 6. dtype | 결함 아님(제공 B=1 실행 경로), 정밀도 영향 판단 불가 | train에서 bridge.float(), set_latent에서 float, adapter에서 h.float(), 마지막 delta.to(h.dtype) 후 h+delta입니다. 선형층 입력과 파라미터의 fp32 정합성이 있습니다. 작은 delta가 bf16 반올림으로 소실될 수 있으나 측정하지 않았습니다. |
| 7. latent_drop | 결함: D1 | None이면 훅이 원래 출력 객체를 즉시 반환합니다. 이는 항등 경로이나 브리지 학습에는 gradient가 없습니다. 셀프테스트는 T=5의 전방 항등만 비교하고 None 분기의 backward, T=1, KV cache를 다루지 않습니다. |
| 8. 실행기·D2 | 결함: D2 / D2 CLI 인자 정합성은 결함 아님 | b의 X3E49는 U6, X3A50은 U7이며 c의 대기 로그 이름과 대응합니다. 그러나 로그는 rc=0 확인 없이 출력됩니다. c 39·41행의 --items/--order_sha는 gen 36행 정의와 맞습니다. 실제 GPU 해제·활성 PID·200개 파일 해시는 확인하지 못함. |
| 9. 추가 | 결함: D3·D4, 추가 검증 공백 | 아래 목록 참조. |

### D1. latent_drop=None에서 역전파 대상이 없음

- 파일·줄: `scripts/g2v9_train.py:45–46,69–75`, `scripts/g2v9_xattn.py:65–68`, `scripts/cc_x3c.sh:45`.
- 조건: 모든 Gemma 파라미터가 requires_grad=False이고 해당 microbatch의 ldrop=True. 토큰 id와 embedding 출력도 trainable leaf가 아니며 훅은 adapter를 호출하지 않습니다.
- 코드상 결과: loss가 어떤 trainable 파라미터에도 연결되지 않습니다. 기본적인 eager autograd 동작에서는 backward에 gradient 경로가 없어 예외가 예상됩니다. 실제 예외 로그는 확인하지 못함. 재현 실행 0회.
- 수식상 효과: None 분기의 브리지 gradient는 정확히 0입니다. 예외를 막아도 현재 구조만으로 '잠재 없음 상태를 학습해' 브리지의 유해성을 줄이는 gradient는 생기지 않습니다.
- 영향: rev3 X3Od47의 학습과 체크포인트 생성을 중단시킬 수 있습니다. 이미 보고된 rev2 점수의 원인이라고 단정하지 않습니다.
- 수정안(적용하지 않음): 아래는 예외 방지용 텍스트이며, None 분기를 새로운 학습 신호로 바꾸지는 않습니다. 유효 microbatch 기준 손실 스케일과 dropout 실험의 정의는 사전에 다시 명시해야 합니다.

```diff
-            loss.backward(); bl += float(loss)
+            if loss.requires_grad:
+                loss.backward()
+            bl += float(loss)
```

### D2. 학습·생성 오류가 완료 표식과 다음 실행으로 전파되지 않음

- 파일·줄: `scripts/cc_x3b.sh:5,26–43,45–52`, `scripts/cc_x3c.sh:5,13,27–50`.
- 조건: Python 학습·생성·채점 중 하나가 0이 아닌 exit code를 반환. 셸에는 errexit가 없고 tr/gn은 이어지는 echo의 상태를 최종 함수 상태로 남깁니다. bare wait는 각 작업의 상태를 개별 확인하지 않습니다.
- 영향: GENS 로그와 DONE.flag가 완전한 600개 생성물·정상 체크포인트를 의미하지 않습니다. c의 로그 대기는 GPU 배치의 특정 함수가 끝났다는 단서이지 유효한 생성 결과의 근거가 아닙니다. c에서 nvidia-smi로 유휴를 추가 확인하므로 로그 조건만으로 즉시 GPU 중첩이라고 단정하지 않습니다.
- 수정안(미적용): 각 함수에서 rc를 변수로 보존하고 return, 호출 순서를 && 또는 명시적 if로 연결, background PID별 wait 상태 확인, 기대 id 600개/중복/해시 확인 후에만 완료 표식 작성. rc 보존만으로는 호출자의 후속 실행을 막지 못합니다.

```diff
-  CUDA_VISIBLE_DEVICES=$4 $PY ...
-  echo "[TR-$1] rc=$? ..."
+  CUDA_VISIBLE_DEVICES=$4 $PY ...
+  local rc=$?
+  echo "[TR-$1] rc=$rc ..."
+  return "$rc"
- ( tr TAG ...; gn TAG ... ) &
+ ( tr TAG ... && gn TAG ... ) &
```

위 `...`는 위치 설명용 축약입니다. 실행 가능한 패치 파일은 작성하지 않았습니다. 기존 GPU 번호 지침과 실행기의 GPU 6·7 표기는 충돌하나 이번 검토에서 실행하지 않았습니다.

### D3. 생성 재개가 id만 확인하여 다른 조건·체크포인트의 행을 재사용할 수 있음

- 파일·줄: `scripts/g2v9_gen.py:53–58,61–67`.
- 조건: 기존 a.out에 같은 id가 있으나 checkpoint, zfile, cond, shift, order 또는 tokenizer가 바뀌었거나 앞선 행의 generation 필드가 없는 경우.
- 결과: id만 done에 넣어 건너뛰고 append하므로 한 파일에 서로 다른 실행의 행이 혼합될 수 있습니다. 파싱 불가 행도 그대로 보존된 채 추가합니다. 제공 실행기는 새 디렉터리 존재를 막으므로 최초 실행에서 바로 이 조건이 생긴다고 보지는 않습니다.
- 영향: matched/shuffled 점수와 EOS·길이 집계가 여러 실행의 혼합일 수 있습니다. 제공된 X1 파일에 이 문제가 발생했다는 뜻은 아닙니다.
- 수정안(미적용): sidecar 실행 manifest에 전체 입력 해시·cond·shift·order·tokenizer를 저장하고 재개 전에 일치 확인. 기존 행의 필드·중복 id도 검사. 조건 불일치 시 기존 파일을 유지하고 새 이름 사용.

### D4. 프롬프트 절단 시 모든 label이 -100이거나 해설/EOS가 잘릴 수 있음

- 파일·줄: `scripts/g2v9_train.py:63–75`.
- 조건: len(p_ids)>=max_seq_len 또는 합친 시퀀스가 max_seq_len보다 김.
- 결과: 전자의 경우 tgt가 모두 -100이어서 유효 표적이 0개이며 평균 CE가 NaN이 될 수 있습니다. 후자의 경우 solution 뒤에 붙인 EOS도 잘릴 수 있습니다. 유효 label 개수·finite loss 검사와 절단 횟수 기록이 없습니다.
- 영향: 해당 입력이 실제 학습 집합에 존재하면 학습 불안정 또는 종료 패턴 학습 차이가 가능합니다. 학습 tokenizer와 학습 993개가 제공되지 않아 발생 건수는 확인하지 못함. 개발 600 외 문항을 요청하거나 새 문항을 생성하지 않았습니다.
- 수정안(미적용): 사전 고정한 길이 정책으로 프롬프트·표적 예산 분리, 유효 표적 0개를 명시적으로 중단/기록, finite loss 확인, EOS 보존·절단 수 기록. 결과 이후 임의 데이터 제외는 하지 말아야 합니다.

### 검증 공백과 추가 관찰

- 학습에서는 tokenizer.eos_token_id, 생성에서는 고정 EOS={1,106}. 실제 tokenizer 파일·버전이 없어 의미 일치 확인하지 못함.
- gen 11–12행의 `g2v3_m3_full_train.py`, `g2v3_m3_eval_latents_v2.py`가 제공되지 않았습니다. 토큰 설정과 extract_tensor의 키 해석을 독립 확인하지 못함.
- xattn의 reshape는 B=1을 전제로 합니다. 현재 batch_size=4는 B=1 microbatch를 4회 반복하므로 직접 충돌하지 않습니다. 추후 실제 B>1 입력에는 잠재 배치 지원이 필요합니다.
- shuffled는 기본 조건에서 다른 id를 선택하지만 shift=0 또는 600의 배수, 중복 order를 금지하지 않습니다. 현재 order의 실제 유일성은 TASK-C002에서 확인할 수 있습니다.
- center 계산은 --items만 참조하지만 코드에 train/dev 집합 교집합 차단은 없습니다. 집합 파일 부재로 누설을 입증하거나 배제하지 못함.
- gate=0 초기에는 bridge Q/K/V/O gradient가 0이며 gate만 업데이트될 수 있습니다. 'rev2의 모든 파라미터가 첫 단계부터 학습된다'는 설명은 수식과 다릅니다.
- 훅은 매 cached token에도 현재 latent를 조회합니다. 과거 token의 downstream KV에는 당시 주입 효과가 들어가므로 중간에 같은 cache의 latent만 바꾸는 사용은 동등하지 않습니다. 제공 greedy는 매 문항마다 새 prefill과 cache를 만들어 이 조건을 피합니다.

## 4. 하지 못한 것과 이유

TASK-C001은 읽기 전용 검토이므로 모델·셀프테스트·학습·생성·설치를 하지 않았습니다. 실제 cache ABI, gradient 값, NaN 발생 횟수, train/dev 교집합, latent 키 매핑, 실제 tokenizer id는 확인하지 못함. 이 검토로 기존 정답률 저하의 원인을 하나로 확정할 수 없습니다. 패키지의 외부 서버 경로는 읽기 대상 텍스트일 뿐 접속하지 않았습니다.

## 5. 재현 명령

아래는 로컬 정적 확인 명령입니다. 모델 실행 명령은 제시하지 않습니다.

```powershell
Set-Location C:\Project\CTS\shared
Get-FileHash scripts/g2v9_xattn.py,scripts/g2v9_train.py,scripts/g2v9_gen.py,scripts/cc_x3b.sh,scripts/cc_x3c.sh -Algorithm SHA256
rg -n 'requires_grad|latent_drop|set_latent|loss.backward' scripts/g2v9_train.py scripts/g2v9_xattn.py
rg -n 'rc=|wait|touch|GENS|set -' scripts/cc_x3b.sh scripts/cc_x3c.sh
```

## 6. 산출물 SHA256

본 보고서의 해시는 자기참조를 피하기 위해 후속 신규 파일 `SHA256SUMS-C001.txt`에 기록합니다. 아래는 검토 입력의 SHA-256입니다.

| 입력 | SHA256 |
|---|---|
| scripts/g2v9_xattn.py | ef3fbd5ed5d227c5b765fc01b95bcd920d4433b005e32e3057dcb479f2d1457d |
| scripts/g2v9_train.py | efc24925fd6d45973f388502af7dea3d8fd9747b2cc987b1fa6d9bbaa79aa997 |
| scripts/g2v9_gen.py | 14e466006b9a29c4d30181c424f5e6fb9872c781d03c3313c71fc433f7c2f3b3 |
| scripts/cc_x3b.sh | 580e6b3181926d053117a87b73c85280a80e971fdfa9c88c8f9d7ca61dfd941e |
| scripts/cc_x3c.sh | 119b8e3418f816eaf3f2fa528d9da9b8fa5af4322704425e58492bcd312614b6 |
| docs/X_INTERFACE_PLAN_20261019.md | 319b28783f70b302a7975eeebadfcf41624d776a5e2f2e7091823e779ad1635a |
| INBOX/TASK-C001.md | c4d811d23c4278a53e5dec539d3f43f498323c940223e768bf1e5a2d245aca66 |

## 7. 의견(판정 아님)

rev3 latent_drop 분기의 학습 의미와 D1을 먼저 CC 측에서 확인할 필요가 있습니다. None 경로는 trainable bridge와 독립적이므로 동결 LM의 능력을 유지시키는 새 gradient를 제공하지 않습니다. 학습 해악을 억제하려는 목적에는 matched latent가 들어간 출력에 대한 별도의 사전 정의된 보존 제약 등이 필요할 수 있지만, 이는 이번 TASK에서 실행하거나 새 규칙으로 채택한 내용이 아닙니다.

층별 단일 tanh gate는 모든 token·head의 주입 크기를 묶습니다. 작은 gate 또는 bf16 반올림이 실제로 내용 전달을 제한하는지는 gate별 delta RMS·h RMS 등 측정 없이 단정할 수 없습니다.

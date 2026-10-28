# TASK-C003 — X5 트레이너 사전 정적 검토

작성자: Codex | 2026-10-20 KST (정확한 작성·입력 식별은 동봉 manifest)

입력 commit: `cae17046059a3de50720dc8636a20e1a8be799d2`.
아래 상대 경로의 기준은 저장소 `INBOX/20261020-0750-pkg-b/`입니다. 로컬 사본은 `C:\Project\CTS\shared\OUTBOX\github-exchange-20261020-0735` 아래 동일 경로입니다.

## 0. 한 줄 요약

X5 주요 파일 3개와 비교 코드·문서 6개를 읽어 가중 구간 경계, 절단 후 유효 표적, 512토큰 생성 예산, 입력 id·오류 전파의 결함 조건을 기록했고 모델·학습·생성 실행은 0회입니다.

## 1. 수행한 것

- GitHub에서 새 패키지와 `ACK-C001-C002-20261020-0750.md`를 받았습니다. 최신 패키지의 AGENTS.md·CONTEXT.md·README.md·TASK-C003.md·계획·결과 문서를 읽었습니다.
- `g2v10_train.py`, `cc_x5.sh`, `cc_x5base.sh`, `g2v9_train.py`, `g2v9_gen.py`, `g2v9_xattn.py`, `g2v5_s0_grade.py`와 설계·결과 문서를 대조했습니다.
- 검토 대상 코드를 import하거나 실행하지 않았습니다. 서버·GPU·학습·생성·추가 설치·재채점·코드 수정·패치 파일 생성은 하지 않았습니다. 아래 diff는 보고서 안의 제안 텍스트뿐입니다.
- GitHub 통신은 사용자에게 직접 승인받은 파일 교환입니다. 원래 CTS 소스 저장소 원격 설정은 변경하지 않았습니다. 갱신 패키지는 덮어쓰지 않고 해당 commit의 사본에서 읽었습니다.
- C002는 ACK의 '추가 재채점 생략·서버 채점 기준' 선택에 따라 다시 실행하지 않았습니다. 플래그 비교와 ACK 답변은 별도 `REPLY-ACK-C001-C002-C003-20261020.md`에 기록합니다.

## 2. 결과 표(숫자만)

| 주요 검토 파일 | 비교 코드·문서 | TASK 항목 | 모델 실행 | 학습 실행 | 생성 실행 | 수정 파일 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 6 | 8 | 0 | 0 | 0 | 0 |

| 기본 생성 상한 token | 학습 길이 상한 token | answer_weight | 배치 microbatch 수 | 런처 latent_drop | 런처 text_drop |
|---:|---:|---:|---:|---:|---:|
| 512 | 1536 | 5 | 4 | 0.0 | 0.0 |

## 3. 불일치·결함 목록(파일, 줄, 재현 조건)

### TASK 항목별 결론

| 항목 | 결론 | 근거 |
|---|---|---|
| 1. 가중 정렬 | 시프트 정렬은 결함 아님; 답 경계 정합성은 보장 없음 | train 118–128행의 ids와 wts는 함께 자르고 lg[t]·x[t+1]·wts[t+1]가 맞습니다. 그러나 84–91행의 세 독립 tokenization 길이는 최종 token sequence의 문자 경계가 아닙니다. D1 참조. |
| 2. 절단·프롬프트 mask | 길이 정렬은 결함 아님; 표적·답 소실 검사가 없음 | train 118–121행에서 두 배열 길이는 같습니다. finite CE에 대해서만 0가중이 prompt loss 제거와 같습니다. w.sum clamp는 0나눗셈만 막고, 답·EOS 소실이나 NaN을 검출하지 않습니다. D2 참조. |
| 3. 표적 구성 | 정상 중첩 brace·마지막 완결 box 교체는 결함 아님; 잔여 텍스트와 생성 상한 문제 | 25–35행은 중첩된 중괄호를 depth로 셉니다. 46–53행은 마지막 box만 교체하고 뒤 post를 보존합니다. 마지막 box가 미완결이면 이전 완결 box를 재탐색하지 않고 append합니다. 512 상한과의 문제는 D3 참조. |
| 4. 누설 | 코드에 명시적인 dev 경로는 없음; 실제 집합 분리는 판단 불가 | train 69–106행은 --items로 DATA의 행을 고르고 해당 id의 target·latent로 center를 만듭니다. 학습 id·target 파일·latent가 제공되지 않아 993개 집합 및 dev600과의 교집합은 확인하지 못함. 입력에 대한 강제 검사는 D4 참조. |
| 5. latent_drop | 이전 backward 예외의 가드는 있음; None 분기는 여전히 학습 신호 없음 | train 122–134행의 guard는 loss.requires_grad=False에서 backward를 생략합니다. loss와 ans_ce 기록은 dropped 샘플도 포함합니다. AdamW 동작·집계 의미는 D5 참조. 실제 런처는 latent_drop=0.0입니다. |
| 6. 생성 프롬프트 | 제공 코드의 공통 경로는 결함 아님; 실제 token 동일성은 판단 불가 | base는 g2v9_gen --cond none이고 train·gen은 같은 build_prompt_ids를 사용합니다. hook 미부착 상태이므로 base의 학습된 bridge가 LM에 작용하지 않습니다. 외부 tokenizer·모델 해시와 실제 target manifest는 제공되지 않았습니다. |
| 7. 실행기·스펙 | 오류 전파 결함과 순서 보장 공백; 스펙 생성은 확인 불가 | x5 13행은 base flag만 기다립니다. rev3(b) 결과 조건은 런처에 없습니다. 오류 상태는 D6, make_s0spec_x.py 부재는 D7 참조. |
| 8. 추가 | 입력 버전·진단 지표·가설 해석 공백 | 런처 해시가 TASK 전사값과 다릅니다. 답 가중치·label이 사라져도 카운터가 없습니다. CI에 0이 포함되는 것을 해악 부재의 증명과 동일시할 수 없습니다. D7 및 의견 참조. |

### D1. prefix 재토큰화 길이로는 전체 토큰열의 box 범위를 확정할 수 없음

- 파일·줄: `scripts/g2v10_train.py:84–91`.
- 조건: encode(pre)가 encode(pre+box+post)의 prefix token ids와 다르거나, encode(pre+box)의 끝 토큰이 post를 붙일 때 합쳐지거나 분해되는 경우.
- 결과: `[len(ids_pre),len(ids_prebox))`가 실제 최종 token열의 box와 다를 수 있습니다. 답 앞·뒤 토큰을 가중하거나 box 일부를 누락할 수 있습니다. 현재 코드에는 두 prefix token열이 실제 prefix인지 검사하는 단계가 없습니다. 오차가 반드시 한두 토큰 이내라는 보장도 없습니다.
- 영향: 손실 가중 5가 의도한 정답 구간에만 적용됐다는 해석이 불확실합니다. 실제 tokenizer 실행을 하지 않았으므로 빈도·오차 token 수는 측정하지 않았습니다.
- 수정안(미적용): 전체 target을 한 번 tokenization하고 원문 offset과 box 문자 범위의 겹침으로 가중합니다. tokenizer가 offset을 지원하지 않으면 명시적으로 중단하거나 token-prefix 일치를 검증하는 대안을 사전에 선택해야 합니다. prompt/target을 따로 encode하는 경계 자체는 train/gen의 공통 프롬프트 구성과 별개입니다.

```diff
-        ids_pre = tok.encode(pre, add_special_tokens=False)
-        ids_prebox = tok.encode(pre + box, add_special_tokens=False)
-        ids_full = tok.encode(pre + box + post, add_special_tokens=False)
+        target = pre + box + post
+        enc = tok(target, add_special_tokens=False, return_offsets_mapping=True)
+        ids_full = list(enc["input_ids"])
+        char_begin, char_end = len(pre), len(pre) + len(box)
+        is_answer = [end > char_begin and begin < char_end
+                     for begin, end in enc["offset_mapping"]]
-        w = [1.0] * len(ids_full)
-        for t in range(min(len(ids_pre), len(ids_full)), min(len(ids_prebox), len(ids_full))):
-            w[t] = a.answer_weight
+        w = [a.answer_weight if flag else 1.0 for flag in is_answer]
```

위 내용은 가중 경계 계산 부분만의 제안입니다. EOS를 append할 때 w에는 1.0, is_answer에는 False도 append하여 길이를 맞춰야 합니다. 경계에 걸친 token 전체를 가중한다는 정책도 기록해야 합니다. 원본은 수정하지 않았습니다.

### D2. 0가중·clamp는 유효 표적과 정답 보존을 검사하지 않음

- 파일·줄: `scripts/g2v10_train.py:118–135,144–148`.
- 조건 1: prompt 길이 >= max_seq_len. 잘린 배열의 w가 모두 0이 됩니다.
- 결과: finite CE이면 loss는 0이며, latent가 켜져 있으면 loss.requires_grad=True라서 0 gradient로 backward와 opt.step까지 갈 수 있습니다. 기존 optimizer momentum 또는 weight decay에 의한 파라미터 변화도 배제되지 않습니다. 이는 유효한 학습 예제가 처리됐다는 증거가 아닙니다.
- 조건 2: prompt 일부와 target은 남지만 정답 box나 EOS가 잘림. loss는 계속 계산되고 ans_ce가 0으로 기록될 수 있습니다. 현재 manifest에는 절단·zero-target·답 토큰 수가 없습니다.
- 조건 3: CE가 non-finite. `0 * NaN` 또는 `0 * Inf`는 mask로 제거되지 않아 0가중이 ignore_index와 완전히 같지는 않습니다.
- 영향: 답 학습이 없는 microbatch가 학습 단계 수에 포함되거나, 낮은 ans_ce=0이 답이 잘 학습된 것처럼 보일 수 있습니다. 발생 건수는 확인하지 못함.
- 수정안(미적용): label 종류에 의존하지 않는 별도 answer bool mask를 유지하고 절단 후 target 수·answer 수·EOS 여부·finite loss를 검사·기록합니다. 사후 예제 제외 대신 사전 정책으로 중단 또는 길이 정책을 정해야 합니다.

```diff
             w = torch.tensor(wts[1:], device=dev, dtype=torch.float32)
+            if not bool((w > 0).any()):
+                raise RuntimeError("No retained target tokens after truncation")
             ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none")
+            if not bool(torch.isfinite(ce[w > 0]).all()):
+                raise RuntimeError("Non-finite supervised token loss")
```

이 diff만으로 answer/EOS 보존 검사가 추가되는 것은 아닙니다. 그 검사는 D1의 bool mask와 함께 별도로 필요합니다.

### D3. 512토큰 미완결 생성문 뒤의 정답이 평가 생성 예산 밖에 놓일 수 있음

- 파일·줄: `scripts/g2v10_train.py:46–53,71–92`; 비교 `scripts/g2v9_gen.py:19–29,65–67`.
- 조건: own generation에 완결 box가 없어 append되고, 최종 tokenizer에서 정답 box 시작/완료 위치가 512 생성 token 이후가 되는 경우. own이 저장 상한에서 잘린 경우 특히 점검이 필요합니다.
- 결과: 학습은 prompt 포함 1536 길이 안에서 이 긴 target을 학습할 수 있지만 평가는 512개 생성 token에서 종료됩니다. 따라서 손실에서 정답을 배웠더라도 평가에서 그 위치까지 쓰지 못할 수 있습니다.
- 현재 own 로더는 id·generation만 사용하고 tokens·stop_id를 버리므로 EOS 완료문과 512 상한 중단문을 구별하지 않습니다. 정확히 몇 문항인지는 제공되지 않은 학습 target과 tokenizer 없이는 확인하지 못함.
- 추가 조건: 여러 box 중 마지막만 바꾸면 앞의 오답 box와 뒤의 예전 결론 post가 그대로 남을 수 있습니다. 마지막 box가 미완결이면 기존 미완결 문자열을 유지한 채 새 box가 붙습니다. 중첩된 일반 LaTeX 분수 brace는 depth로 처리하지만 escaped literal brace를 구별하는 parser는 아닙니다.
- 영향: 가설이 맞아도 target 위치·모순된 후속 문장 때문에 자유 생성 평가의 차이가 작게 나올 수 있습니다. 단순히 낮은 점수만으로 문체 가설을 기각할 수 없습니다.
- 수정안(미적용): 결과를 보기 전에 생성 예산 안의 answer 위치 정책을 고정하고, target별 retokenized box 시작/끝 위치·EOS 유무·appended/replaced·budget 초과 수를 CC 측에서 기록해야 합니다. 정답을 앞당기는 방법은 문체까지 바꿀 수 있으므로 즉석 재정의하지 않습니다.

### D4. 993행 검사와 실제 학습 id·표적 무결성은 다름

- 파일·줄: `scripts/g2v10_train.py:69–95,106`; `scripts/cc_x5.sh:17–18`; `scripts/cc_x5base.sh:23–26,34`.
- 조건: targets에 중복 id, 누락 id, empty generation이 있거나 --items가 다른 파일을 가리킴.
- 결과: own dict는 중복 id의 마지막 값을 조용히 사용합니다. 누락 generation 또는 boxed gold가 없으면 n_skip만 증가하고 실제 학습 집합은 축소됩니다. empty generation은 'id가 존재함'만으로 채택되어 gold answer-only에 가까운 표적으로 바뀝니다. 993행에는 중복·누락이 섞일 수 있습니다.
- center는 축소된 items에 해당하는 Z만 평균내므로 코드에 dev를 직접 넣는 경로는 없지만 실제 train/dev 분리를 강제하는 체크는 없습니다. items가 0개면 center stack에서 중단됩니다. 기존 grader는 생성 파일 누락 시 빈 사전 또는 일부 id만 채점할 수 있습니다.
- 영향: 의도한 993개·공통 조건·순정 생성문 표적이라는 해석과 실제 학습 집합이 달라질 수 있습니다. 실제 위반을 관측했다는 뜻은 아닙니다.
- 수정안(미적용): raw target의 id 유일성, expected id와의 집합 동등, generation 비어있음, 학습/개발 교집합, 입력 파일 전체 SHA, retained/skipped id와 사유를 manifest로 남깁니다. 학습 문제 원문을 Codex에 보낼 필요는 없으며 CC 측 집계·검증 기록으로 확인할 수 있습니다.

### D5. None 분기와 ans_ce의 해석

- 파일·줄: `scripts/g2v10_train.py:99–100,112–136`; `scripts/g2v9_xattn.py:65–68`.
- loss.requires_grad guard가 있으므로 None 분기에서 이전의 무조건 backward 호출은 없습니다. None 샘플은 동결 LM의 CE를 bl·bl_ans에 기록하지만 bridge gradient는 0입니다. 기록된 loss 평균은 '실제로 업데이트에 쓰인 샘플만의 평균'이 아닙니다.
- 4개 모두 None이고 zero_grad(set_to_none=True)로 gradient가 None이면 AdamW는 해당 파라미터를 건너뜁니다. 다만 제공 코드가 set_to_none를 명시하지 않아 서버 PyTorch 기본값에 의존합니다. 오래된 환경의 기본값이 False이고 이전 gradient tensor가 남아 0으로 지워지는 경우, momentum·decay 영향이 가능하므로 '항상 무변화'라고 쓰면 안 됩니다. 서버 optimizer 구현·버전은 실행 확인하지 않았습니다.
- `am = w > 1`은 현재 answer_weight=5에서 raw answer CE를 얻는 데 사용할 수 있지만 answer_weight=1이면 답 지표가 0으로 사라지고, 모든 답이 잘려도 같은 0입니다. ans_ce를 누적하는 분모는 답이 존재하는 샘플 수가 아니라 항상 batch_size=4입니다.
- 기본 런처는 latent_drop=0이므로 drop 관련 논점은 현재 예정 설정에서 작동하지 않습니다. 별도 answer mask와 answer-present count, 실제 optimizer update 수를 기록하는 것이 수정안입니다.

### D6. 실행기 오류·완료 표식·순서 보장

- 파일·줄: `scripts/cc_x5.sh:5,13,26–34`; `scripts/cc_x5base.sh:5,23–34`.
- Python exit code가 0이 아니어도 뒤의 echo·grep·다음 단계가 실행될 수 있습니다. `wait`는 개별 PID 상태를 검사하지 않습니다. 채점 실패·일부 생성 상태에서도 마지막 touch까지 갈 수 있습니다. base의 DONE flag 또한 유효한 전체 target이라는 증거가 아닙니다.
- x5는 base flag와 993행만 보며 rev3(b) 결과·후속 결정 조건은 보지 않습니다. CC가 (b) 확인 뒤 수동으로 시작하면 순서를 지킬 수 있지만 이 런처가 자동으로 해당 순서를 보장하는 것은 아닙니다.
- GPU 유휴 체크는 명령 시작 전 확인일 뿐 예약·배타 잠금은 아닙니다. 실제 유휴 상태나 다른 프로세스는 확인하지 않았습니다. Codex는 실행하지 않습니다.
- 영향: 실험 단계·row count·시드·유효 checkpoint가 불완전한 상태에서 다음 단계가 진행될 수 있습니다. ACK의 '이번 실행은 수동 확인'을 코드에 반영된 방어장치와 구분합니다.
- 수정안(미적용): 각 rc 보존과 명시적 분기, 병렬 PID별 wait, checkpoint 검증, 600개 고유 id와 정확한 기대 집합·조건별 manifest 검사, 채점 라벨과 600개 플래그 검증 이후에만 표식 작성. base는 993개 고유 id를 검증해야 합니다.

```diff
 CUDA_VISIBLE_DEVICES=$U5 $PY $S/g2v10_train.py ...
+rc=$?
+[ "$rc" -eq 0 ] || { echo "[FATAL] training rc=$rc"; exit "$rc"; }
```

위 `...`는 기존 긴 명령의 위치 설명이며 실행 가능한 패치가 아닙니다. 병렬 생성·채점에도 별도의 rc 검사가 필요합니다.

### D7. 입력 버전과 확인하지 못한 파일

- TASK-C003의 cc_x5.sh SHA 접두사는 `6786417e…`이나 실제 제공 파일은 `f45af958…`입니다. X5 trainer `725915e5…`와 base `3378eb2c…`는 TASK와 일치합니다. 제공된 f45af958 파일은 latent_drop=0.0입니다. 어떤 서버 파일이 실행될지는 확인하지 못함.
- ACK는 X3 trainer를 loss.requires_grad guard로 고쳤다고 설명하지만 이 패키지의 `g2v9_train.py:75`는 여전히 무조건 backward이며 SHA는 앞선 efc24925…와 같습니다. X5 자체에는 guard가 있으므로 X5 코드와 X3 서버 패치를 혼동하지 않습니다.
- `make_s0spec_x.py`가 패키지에 없습니다. X5O47·shuf_X5O47·C_fmt 라벨, 공통 600 id, 같은 C_fmt 기준을 만드는지는 **확인 불가**입니다.
- `g2v3_m3_full_train.py`, `g2v3_m3_eval_latents_v2.py`, tokenizer 설정, train/shard indices, 실제 target JSONL, oracle latent가 없습니다. 실제 token 경계·학습 id·내용·center를 검증하지 못함. 금지된 학습·추가 평가 데이터는 요청하지 않습니다.

## 4. 하지 못한 것과 이유

TASK-C003은 읽기만 하는 코드 검토입니다. 셀프테스트·모델·토크나이저·학습·생성·채점은 실행하지 않았습니다. 따라서 D1–D4의 실제 발생 횟수, 실행 성능, 정답률 영향의 크기를 측정하지 않았습니다. 손실이나 생성 결과의 원인을 확정하지 않습니다. C003 작성 중 받은 원격 commit 이후 서버 변경 상태도 확인하지 못했습니다.

## 5. 재현 명령

다음은 로컬 파일 읽기·SHA 확인용이며 서버 명령이 아닙니다.

```powershell
Set-Location C:\Project\CTS\shared
$pkg = 'OUTBOX/github-exchange-20261020-0735/INBOX/20261020-0750-pkg-b'
Get-FileHash "$pkg/scripts/g2v10_train.py","$pkg/scripts/cc_x5.sh","$pkg/scripts/cc_x5base.sh" -Algorithm SHA256
rg -n 'ids_pre|ids_full|wts|w.sum|am =|latent_drop|requires_grad|n_skipped' "$pkg/scripts/g2v10_train.py"
rg -n 'set -|until|wc -l|rc=|wait|make_s0spec|touch' "$pkg/scripts/cc_x5.sh" "$pkg/scripts/cc_x5base.sh"
```

## 6. 산출물 SHA256

보고서 자체와 답변 문서의 해시는 신규 `MANIFEST-C003.json`에 기록합니다. 자기참조 해시는 본문에 넣지 않습니다.

| 입력 | SHA256 |
|---|---|
| scripts/g2v10_train.py | 725915e513e869c6fd39aafb6e0313d846870873ec95d214ec413b3c00c056d9 |
| scripts/cc_x5.sh | f45af958b16d954309c08703fb5774147369b5bf5591532b7e89997c7e49af94 |
| scripts/cc_x5base.sh | 3378eb2cac61cdf9418a3d8616bc62acd1e149e3191854d990eecbe89108bef5 |
| scripts/g2v9_train.py | efc24925fd6d45973f388502af7dea3d8fd9747b2cc987b1fa6d9bbaa79aa997 |
| scripts/g2v9_gen.py | 14e466006b9a29c4d30181c424f5e6fb9872c781d03c3313c71fc433f7c2f3b3 |
| scripts/g2v9_xattn.py | ef3fbd5ed5d227c5b765fc01b95bcd920d4433b005e32e3057dcb479f2d1457d |
| INBOX/TASK-C003.md | c1f95170c129f4bf81ecfa37f01fa6632c74b1b436903f8cea2a4d1a8fbebc03 |
| docs/X_INTERFACE_PLAN_20261019.md | 1d8aff0c5de65f1ae3707fd940bf5e93350ead507f47a68b1b78ab001da4cd5a |
| docs/RESULT-X3-xattn-20261019.md | 3adc7460641c13d0a3470df78430f2a72fe10fb89429e892d752761676683642 |

## 7. 의견(판정 아님)

- CI가 0을 포함하는 것은 효과가 0이라는 동등성 증명이 아닙니다.계획서의 'shuffled−C CI가 0을 포함'은 미리 정한 보고 규칙으로 보존하되, 실질적인 해악 소멸을 입증하려면 별도의 사전 동등성 한계가 필요합니다. 이번 검토에서 규칙을 변경하지 않습니다.
- X5는 표적 문체뿐 아니라 정답 치환·답 가중치·EOS·길이 노출도 바꿉니다. 이후 해악이 감소해도 '문체만이 원인'이라고 분리할 수 없고, 감소하지 않아도 D1–D4 같은 구현 조건을 먼저 배제해야 합니다.
- wrong reasoning과 correct boxed answer가 공존하는 표적은 본 설계가 의도한 절충입니다. 다만 문맥 뒤 post에 기존 오답 결론이 남는지, 답이 평가 예산 안에 있는지를 계수하지 않으면 가설과 구현 실패가 뒤섞일 수 있습니다.
- 우선 CC 측에서 정확한 실행 코드 해시, 답 토큰의 가중·절단·512 상한 통계, 완전한 target id 집합과 오류 종료 경로를 확인해 주면 검토의 불확실성을 줄일 수 있습니다. 추가 실행 여부와 해석은 CC와 지정 감사자의 범위입니다.

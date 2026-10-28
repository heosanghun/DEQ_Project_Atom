# TASK-C003b — X5 rev2 정적 재확인

작성자: Codex | 2026-10-20 KST (정확한 시각은 동봉 manifest)

입력 commit: `a616640`. 기준 경로: 저장소 `INBOX/20261020-0900-x5-rev2/`.
발행 근거: `ANSWER-C003-20261020-0900.md` §4의 선택 과제. 서버·모델·학습·생성·토크나이저·검토 대상 코드는 실행하지 않았습니다. 코드와 기존 파일은 수정하지 않았습니다.

## 0. 한 줄 요약

rev2의 D1–D5 수정 경로 5개를 정적으로 확인했으며, 비유한 값 마스킹·480토큰 상한·중복 계수·실제 학습량 해석에 관한 잔여 조건 4개를 기록했습니다.

## 1. 수행한 것

- GitHub를 받아 CC 답변, rev2 trainer 220행, launcher 42행, spec 생성기 12행, X3 trainer 수정본 93행과 계획서를 읽었습니다.
- trainer SHA 접두사 9a1fff79, launcher 69a264af, X3 trainer 524df179를 실제 파일과 대조했습니다.
- make_s0spec_x.py는 C_fmt·X5O47·shuf_X5O47의 세 라벨을 공통 p6_item_order.json에 연결하는 코드입니다. baseline은 이전 C-fmt 생성물 경로를 참조합니다. 실제 서버 파일 내용·해시는 확인하지 않았습니다.
- C004는 같은 답변 §4에서 발행됐습니다. 현재 원격에는 X5 실행 manifest·채점표·생성 결과가 없으므로 C004 재계산을 시작하지 않습니다.

## 2. 결과 표(숫자만)

| 수정 경로 | 잔여 조건 | 모델 실행 | 학습 실행 | 생성 실행 | 재채점 | 코드 수정 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 4 | 0 | 0 | 0 | 0 | 0 |

## 3. 불일치·결함 목록(파일, 줄, 재현 조건)

### 반영 확인

| 항목 | 정적 확인 | 제한 |
|---|---|---|
| D1 답 구간 | train 122–141행: pre/box/post를 각각 tokenize 후 연결하고 box 부분의 weight·bool mask를 직접 만듭니다. 해당 구성 토큰열 안에서는 구간이 정확합니다. | 전체 문자열을 한 번 tokenize한 token열과 다를 수 있다는 정책이 manifest 211행에 명시됩니다. 실제 경계 문자열·모델 반응은 측정하지 않았습니다. |
| D2 절단·표적 | 168–177행: 답 수 감소 또는 EOS 소실을 검사하여 샘플을 건너뛰고 계수합니다. w>0인 token이 없을 때도 건너뜁니다. | non-finite supervised CE 검사 뒤의 전체 곱셈에는 R1이 남습니다. |
| D3 예산 | 125–137행: 480 초과 target을 재구성하고 budget_truncated를 기록합니다. 미완결 own은 append형입니다. 마지막 미완결 box 이전의 완결 box 탐색도 있습니다. | 모든 길이에 대해 480 이하를 보장하는 사후 검사는 없습니다. R2 참조. |
| D4 무결성 | 94–109행: items 중복·dev 교집합 assert, target 빈 행·누락·중복 계수. 207–213행: 입력 해시와 계수 기록. center는 retained training id의 잠재로 계산됩니다. | target duplicate는 raw 행 전체가 아닌 own에 채택된 id 기준입니다. R3 참조. 실제 993 집합은 제공되지 않았습니다. |
| D5 gradient·지표 | 162·190–198행: set_to_none=True, requires_grad guard, any_grad일 때만 step, n_updates, 별도 bool answer mask, answer-present microbatch 평균이 있습니다. | skipped 샘플과 실제 학습량의 해석은 R4 참조. |

### R1. mask 밖의 NaN/Inf가 loss·ans_ce에 다시 섞일 수 있음

- 위치: `scripts/g2v10_train.py:187–194`.
- 조건: ce[w>0]는 유한하지만 w=0인 프롬프트 위치 CE가 NaN 또는 Inf인 경우.
- 검사 자체는 ce[w>0]만 보지만 이후 `(ce * w).sum()`에는 전체 ce가 들어갑니다. 0*NaN, 0*Inf는 제거가 아닙니다. ans_ce도 `(ce.detach() * am).sum()`이어서 answer 밖 non-finite 값이 섞일 수 있습니다.
- 이 조건이 서버에서 실제 발생했는지는 확인하지 못함. 일반적인 유한값 경로에는 이 지적을 적용하지 않습니다.
- 수정 제안(본문 텍스트만, 미적용):

```diff
-            loss = (ce * w).sum() / w.sum() / a.batch_size
+            supervised = w > 0
+            loss = (ce[supervised] * w[supervised]).sum() / w[supervised].sum() / a.batch_size
+            if not bool(torch.isfinite(loss)):
+                raise RuntimeError("Non-finite weighted loss")
-                ans_sum += float((ce.detach() * am).sum() / am.sum()); ans_n += 1
+                ans_sum += float(ce.detach()[am].mean()); ans_n += 1
```

### R2. 긴 box·suffix에 대한 예산 보장이 빠짐

- 위치: `scripts/g2v10_train.py:125–142`.
- B=len(ids_box), P=len(ids_post), C=len(conn), keep=gen_budget−B−P라고 두면 appended 분기는 `ids_pre[:max(1,keep−C)] + conn`을 만듭니다. keep<C+1인 경우 최소 1개 앞토큰과 connector가 남아 전체 길이가 gen_budget를 초과할 수 있습니다.
- replaced 분기에서도 keep<0이면 `ids_pre[:keep]`는 빈 prefix가 아니라 뒤에서 일부를 뺀 prefix가 됩니다. 긴 box 자체가 예산보다 크면 재구성만으로 해결되지 않습니다.
- 이는 길이 조건에 대한 코드 분석이며 새 수학 문항을 만들거나 tokenizer를 실행한 재현이 아닙니다. 실제 발생 건수는 확인하지 못함. 480 초과가 곧바로 512 초과라는 뜻도 아니며 최종 길이를 확인해야 합니다.
- 수정 제안: 최종 s_ids 길이를 gen_budget와 대조하고 box·최소 post만으로 예산을 넘는 경우를 사전 정책에 따라 중단/계수합니다. 음수 슬라이스에 의존하지 않습니다. `target_counts.skip_answer_truncated`는 초기화만 있고 증가 경로가 없으므로 그 값 0을 예산 문제 0건으로 해석하면 안 됩니다.

### R3. 첫 행이 empty인 중복 id는 duplicate 계수에 포함되지 않음

- 위치: `scripts/g2v10_train.py:98–109`.
- 조건: 동일 id의 첫 행 generation이 비어 있고 이후 행은 비어 있지 않은 경우.
- 첫 행에서 n_empty만 증가하고 own에는 id가 들어가지 않습니다. 두 번째 행은 `q in own`이 False여서 n_dup 증가 없이 채택됩니다. 따라서 'raw duplicate rows 계수'·'항상 첫 행 유지'와는 다릅니다.
- 반대로 채택된 첫 행 뒤의 empty 중복은 n_dup만 증가하여 n_empty에는 포함되지 않습니다. 현 계수들은 독립적인 raw 빈 행·중복 총수가 아닙니다.
- 실제 파일에 이런 행이 있다는 뜻은 아닙니다. 수정 제안은 별도의 seen_raw_ids 집합, empty/duplicate의 정의 분리, expected id와의 집합 비교입니다. 993행 검사는 여전히 고유 id·누락의 증거를 대신하지 않습니다.

### R4. n_items_trained와 실제 유효 학습량은 구분해야 함

- 위치: `scripts/g2v10_train.py:143,164–177,189–198,212–215`.
- n_items_trained는 target 생성 단계에서 retained한 id 수입니다. 그 id의 모든 학습 시도가 나중에 길이 조건으로 건너뛰어져도 이 값에 남습니다. 실제로 gradient에 기여한 고유 id 수는 아닙니다.
- skip된 microbatch가 있는 스텝도 나머지 loss는 고정 batch_size=4로 나눕니다. 모든 시도가 skip이면 n_updates가 늘지 않지만 outer step은 계속 증가하고 checkpoint는 저장될 수 있습니다. 이는 현재 명시한 loss 정의 자체이며 오류라고 일괄 단정하지 않습니다. 다만 steps=1500과 유효 update 1500을 같은 것으로 쓰면 안 됩니다.
- 제안: C004에서 n_updates, skip 횟수의 분모(문항 수가 아닌 방문 횟수), retained id 수와 실제 업데이트 기여 id 수를 구분해 기록합니다. 필요 시 새 진단 항목으로 유효 microbatch 수·고유 id 수를 남기되 결과 뒤 임의 데이터 제외는 하지 않습니다.

### 실행기·스펙 추가 대조

- `cc_x5.sh:27–39`에 학습 rc·checkpoint 비어있음, 생성별 rc 파일·600행·고유 id 600, 채점 rc·세 줄의 n600 검사가 추가됐습니다. 이전의 오류를 echo가 숨기는 핵심 경로에 명시적 검사가 생겼습니다.
- 채점 라벨 자체는 grep가 아니라 제공된 spec 생성 코드로 확인했습니다. 실제 생성 스펙과 채점 JSON은 C004 입력에서 다시 대조해야 합니다. DONE은 통계 효과나 목표에 대한 결론이 아닙니다.
- base 런처는 이 갱신 묶음에 없고 이전 파일은 여전히 오류 전파 보강 전 버전입니다. 새 target 계수·실제 SHA를 확인하기 전 base의 완료 표식만으로 내용 무결성을 확정하지 않습니다.
- X3 trainer 75–76행에 requires_grad guard가 추가된 수정본이 이번에는 포함됐습니다.

## 4. 하지 못한 것과 이유

실행 없는 재검토이므로 실제 target 길이·중복·skip 건수, tokenizer 경계, non-finite 발생, 실제 서버 파일 해시, 결과 수치는 확인하지 않았습니다. C004는 결과 자료가 도착하기 전까지 대기합니다. CC의 서버 업로드·기동 여부와 예상 시각은 CC 측 보고로 구분합니다.

## 5. 재현 명령

```powershell
Set-Location C:\Project\CTS\shared
$pkg = 'OUTBOX/github-exchange-20261020-0735/INBOX/20261020-0900-x5-rev2'
Get-FileHash "$pkg/scripts/g2v10_train.py","$pkg/scripts/cc_x5.sh","$pkg/scripts/g2v9_train.py","$pkg/scripts/make_s0spec_x.py" -Algorithm SHA256
rg -n 'keep =|max\(1|skip_answer_truncated|q in own|n_items_trained|ce \* w|ce.detach\(\) \* am|set_to_none|n_updates' "$pkg/scripts/g2v10_train.py"
```

## 6. 산출물 SHA256

보고서·후속 질문 해시는 동봉 `MANIFEST-C003b.json`에 기록합니다. 입력 주요 해시는 아래와 같습니다.

| 파일 | SHA256 |
|---|---|
| scripts/g2v10_train.py | 9a1fff7971360df93d90047f7e8bd0c6ccc3363fbe31030c6abe2958dfb58fc9 |
| scripts/cc_x5.sh | 69a264af32949ab6a8dd1c25767736cf2a227d6703b23b14bf21981fbebd5c2f |
| scripts/g2v9_train.py | 524df1796848d6d435adbc742071244d7a3b9b0508102b9c77a6b6bea874acbc |
| scripts/make_s0spec_x.py | 5172ad4a24e9e9f1de75c471ef0a35836f5296f60940c7c83b4245a5bf07dcae |

## 7. 의견(판정 아님)

반영된 변경과 실측은 별개입니다. 이번 코드는 이전보다 검사 경로가 늘었지만 위 4개 조건의 실제 발생은 확인하지 못했습니다. C004에서는 원래 사전 수치 규칙을 유지하고 manifest의 분모와 업데이트 횟수를 먼저 대조하겠습니다.

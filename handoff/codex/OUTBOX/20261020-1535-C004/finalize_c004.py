"""Finalize C004 evidence without changing any existing file."""
import json, hashlib
from pathlib import Path
ROOT=Path("D:/AI/cts")
O=Path(__file__).resolve().parent
def read(name):return json.loads((O/name).read_text(encoding="utf-8-sig"))
def put(name,text):
    with (O/name).open("x",encoding="utf-8",newline="\n") as f:f.write(text)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
S=read("statistics.json");D=read("discrepancies.json");L=read("generation_lengths.json");T=read("training_manifest_checks.json")
logs={}
for n in ["local_grade.log","retry_stdout.log","retry_stderr.log","guarded_grade.log"]:
    text=(O/n).read_text(encoding="utf-8",errors="replace")
    logs[n]={"bytes":(O/n).stat().st_size,"WinError6_occurrences":text.count("[WinError 6]"),"EOFError_occurrences":text.count("EOFError:"),"grader_verified_markers":text.count("[GRADER-VERIFIED]"),"done_markers":text.count("[S0-DONE]")}
status={"server_statistics":"recomputed from delivered server flags; not an independent verification of server grading","local_statistics":"ERROR_AFFECTED_DIAGNOSTIC_ONLY; not a normal regrade","local_source":"x5_items_local.json","retry_same_flags":read("x5_items_local.json")==read("x5_items_retry.json"),"guarded_output_exists":(O/"x5_items_guarded.json").exists(),"logs":logs,"stopped_owned_process_ids":[18452,19492,82392],"grader_modified":False,"retry_launch_changes":"Hidden process launch, then guarded runpy launcher; no grading logic, math verification setting or timeout policy changed","auxiliary_command_errors":["Get-FileHash: canonical results_s7/p6_item_order.json absent; supplied copy hash matched the fixed expected hash.","One inline log-summary python -c invocation raised SyntaxError: unterminated string literal; no files were changed; this finalizer replaces that summary command."]}
put("execution_status.json",json.dumps(status,ensure_ascii=False,indent=2))
intro="""# REPORT-C004 — X5 Stage O 독립 재계산·재채점 시도

## 0. 한 줄 요약

서버 채점표 3조건 × 600문항의 재계산에서 matched−shuffled lenient +3.50%p [0.50, 6.50], matched−C_fmt 0.00%p [−3.00, 3.00]을 얻었으며, 로컬 재채점은 Windows 프로세스 예외가 동반되어 정상 재채점 확인을 완료하지 못했다.

## 1. 수행한 것

- D:\\AI\\cts 루트 규칙 및 최신 ANSWER-C004Q와 ANSWER-C003/C003b 확인. C004는 ANSWER-C003 §4 발행 과제다.
- 결과 패키지 20261020-1320-x5-results와 사전 계획 X5 절, 07:5x 정정, ANSWER-C003 §3을 적용했다. X6 실행·분석은 하지 않았다.
- 고정 채점기 SHA256 90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13을 원본 및 공유 사본에서 확인했다.
- 공유 개발 순서 SHA256 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c 확인. train 993과 dev 600의 중복 0; 제공된 dev 기록 및 각 생성물의 600개 고유 ID 집합 일치. 원래 5000문항 데이터셋은 읽지 않았다.
- 서버 spec의 경로만 새 s0spec_local.json에 연결했다. C_fmt는 제공된 handoff/codex/data/gen_C.jsonl을 사용했다.
- 고정 래퍼로 로컬 재채점 시도. 직접 실행, 숨김 프로세스 실행, 재진입 방지 실행기의 로그를 각각 보존했다. 채점기·기존 래퍼·검증 설정을 수정하거나 대체하지 않았다.
- bootstrap은 random.Random(2026), 600문항 복원추출 10,000회, 정렬된 문항 ID, 동일 문항의 쌍 유지, 단일 모델 seed 2047이다. 정렬된 bootstrap 통계의 250번과 9749번(0-based)을 구간 끝점으로 사용했다. CC의 분석 스크립트를 실행하지 않고 독립 구현했으며, CC의 표시값과 소수 둘째 자리에서 일치했다. 전체 구현은 analyze_c004.py에 있다.
- manifest telemetry 1500행의 마지막 200행 평균, 계수 관계, 입력 SHA를 대조했다. 학습은 재실행하지 않았다.
- 환경: Python 3.13.3 / Windows / sympy 1.14.0 / math-verify 0.9.0 / numpy 2.5.3. 로컬 예외의 원인을 단순 버전 차이로 확정하지 않는다.

## 2. 결과 표(숫자만)

서버 채점표 기준:

| 조건 | n | strict 수 | lenient 수 | boxed 수 | strict % | lenient % | boxed % | strict/boxed % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
parts=[intro]
for tag,v in S["server"]["aggregates"].items():
    parts.append(f"| {tag} | {v['n']} | {v['strict']} | {v['lenient']} | {v['boxed']} | {v['rates_pct'][0]:.4f} | {v['rates_pct'][1]:.4f} | {v['rates_pct'][2]:.4f} | {v['strict_among_boxed_pct']:.4f} |\n")
parts.append("\n| 비교 | lenient 차이 %p | 95% 하한 | 95% 상한 | strict 차이 %p | 95% 하한 | 95% 상한 |\n|---|---:|---:|---:|---:|---:|---:|\n")
for pair,d in S["server"]["differences"].items():
    a,b=d["lenient"],d["strict"]
    parts.append(f"| {pair} | {a['delta_pp']:.4f} | {a['ci95_pp'][0]:.4f} | {a['ci95_pp'][1]:.4f} | {b['delta_pp']:.4f} | {b['ci95_pp'][0]:.4f} | {b['ci95_pp'][1]:.4f} |\n")
parts.append("\n| lenient 비교 | 둘 다 정답 | 왼쪽만 정답 | 오른쪽만 정답 | 둘 다 오답 |\n|---|---:|---:|---:|---:|\n")
for pair,d in S["server"]["overlaps"].items():
    a=d["lenient"];parts.append(f"| {pair} | {a['both']} | {a['left_only']} | {a['right_only']} | {a['neither']} |\n")
parts.append("\n| 생성물 | 평균 토큰 | 중앙값 | 512토큰 수 | stop_id 1/106 수 | 종료율 % |\n|---|---:|---:|---:|---:|---:|\n")
for tag,v in L.items():
    parts.append(f"| {tag} | {v['mean_tokens']:.4f} | {v['median_tokens']:.1f} | {v['tokens_eq512']} | {v['eos_count']} | {v['eos_count']/6:.4f} |\n")
parts.append("""
| manifest/telemetry 항목 | 값 |
|---|---:|
| listed / retained | 993 / 992 |
| replaced / appended | 341 / 651 |
| skip_no_gold / skip_answer_truncated | 1 / 0 |
| budget_truncated | 575 |
| budget_truncated / retained % | 57.963710 |
| targets rows / duplicate / empty / missing | 993 / 0 / 0 / 0 |
| steps / updates / visits / contributing unique IDs | 1500 / 1500 / 6000 / 992 |
| seq_truncated / skipped visits | 0 / 0 |
| telemetry rows / ans_n=4 rows | 1500 / 1500 |
| last200 loss / answer CE | 0.0909659529622877 / 0.49176877274273467 |
| 재계산과 manifest 평균 차이 | 0 / 0 |

로컬 **오류 동반 산출물의 참고 집계** — 정상 재채점 결과로 사용하지 않음:

| 조건 | strict 수 | lenient 수 | boxed 수 | strict 서버만 참 | lenient 서버만 참 | 로컬만 참 플래그 수 |
|---|---:|---:|---:|---:|---:|---:|
""")
for tag,v in S["local"]["aggregates"].items():
    d=D["directions"][tag];parts.append(f"| {tag} | {v['strict']} | {v['lenient']} | {v['boxed']} | {d['strict']['server_only']} | {d['lenient']['server_only']} | {sum(x['local_only'] for x in d.values())} |\n")
parts.append("\n| 오류 동반 로컬 참고 비교 | lenient 차이 %p | 95% 하한 | 95% 상한 | strict 차이 %p | 95% 하한 | 95% 상한 |\n|---|---:|---:|---:|---:|---:|---:|\n")
for pair,d in S["local"]["differences"].items():
    a,b=d["lenient"],d["strict"];parts.append(f"| {pair} | {a['delta_pp']:.4f} | {a['ci95_pp'][0]:.4f} | {a['ci95_pp'][1]:.4f} | {b['delta_pp']:.4f} | {b['ci95_pp'][0]:.4f} | {b['ci95_pp'][1]:.4f} |\n")
parts.append("""
## 3. 불일치·결함 목록(파일, 줄, 재현 조건)

1. **ローカル再採点の実行例外 / 로컬 재채점 실행 예외.** `handoff/codex/scripts/codex_grade_local.py:12–28`는 최상위에서 전체 채점을 수행하고 출력 파일을 `w`로 연다. Windows multiprocessing 하위 프로세스에서 재진입한 흔적(반복 GRADER-VERIFIED/S0-DONE)이 retry_stdout.log에 있다. wrapper 원본을 수정하지 않고 별도 guarded launcher를 사용하자 재진입은 줄었지만 `OSError: [WinError 6]`, `EOFError: Ran out of input`은 남았다. 원문 traceback은 local_grade.log, retry_stderr.log, guarded_grade.log에 보존했다. 자신의 실행 PID만 중단했다. 원래 입력 파일의 수정은 없으나, 앞선 래퍼 시도에서는 하위 프로세스가 동일한 **새 출력 JSON**을 반복 작성했다. 이 산출물은 정상 결과로 취급하지 않는다. 예외와 특정 불일치 문항의 일대일 연결은 확인하지 못했다.

2. **SHA256SUMS.txt:1–11의 값이 각각 21자리.** 완전한 64자리 SHA256이 아니므로 이 파일로 전체 무결성을 검증할 수 없다. 실제 데이터 손상을 의미하지는 않는다. master log의 완전한 SHA와 비교 가능한 생성물 2개·채점표·manifest·코드 3개는 7/7 일치했다. 모든 실제 입력의 새 전체 SHA256은 INPUT_MANIFEST.json에 기록했다.

3. **서버용 s0spec.json:1을 로컬에서 그대로 실행할 수 없음.** `results_s7/p6_item_order.json`은 이 PC에 없었다. 지정 SHA와 일치하는 `handoff/codex/data/p6_item_order.json` 및 제공 생성물로 OUTBOX의 별도 spec을 구성했다. 원본 spec은 변경하지 않았다.

4. **RESULT-X5-stylematched-20261020.md:32의 해악 소멸·길이/완주율 동일 서술.** matched−C_fmt의 관측값은 0.00%p이지만 CI는 [−3.00,+3.00]이며 동등성/비열등성 한계를 미리 설정한 검정은 제공되지 않았다. 같은 94개 정답 중 공통은 54개, 각 조건만 정답인 문항이 40개씩이다. 평균 토큰은 398.4467 대 425.2117, 종료율은 48.8333% 대 39.1667%로 수치가 같지 않다. `shared/proposals/X_INTERFACE_PLAN_20261019.md:72`에도 같은 취지의 단정이 있다. 수정 제안: “matched와 C_fmt의 관측 정답률은 같았으나 동등성은 확인하지 못했으며 생성 길이와 종료율은 각각 위 수치였다.”

5. **RESULT-X5-stylematched-20261020.md:33의 ‘효과의 절반’.** 관측 평균의 항등식은 `(M−S)=(M−C)+(C−S)=0.00+3.50=3.50%p`다. 절반이라는 수치적 분해는 이 표에서 도출되지 않는다. 이 항등식 자체도 생물학적/모델 내부 인과 기여도를 식별하는 것은 아니다. 수정 제안: “matched는 C_fmt와 같은 관측 정답률이고 shuffled는 3.50%p 낮았다; 내용 전달과 잘못된 잠재의 교란을 이 세 평균만으로 인과 분해하지 않는다.”

6. **실행 SHA·훈련 계수의 독립 확인 범위.** master log에 trainer의 전체 SHA는 있으나 launcher `cc_x5.sh`의 실제 실행 전체 SHA 출력은 없다. 로컬 launcher SHA와 최신 ANSWER의 앞자리만 대조 가능하다. manifest의 원시 학습 표적·latent 파일은 제공되지 않았으므로 그 파일 SHA 재계산, 개별 표적 토큰/답 구간, 제외된 1개 ID, 실제 기여 ID 992개의 목록을 독립 검증하지 못했다. `g2v10_train.py:125–143`에서 budget_truncated는 최종 제외 가능 분기 전에 증가하는 정책이며, 이번 manifest는 skip_answer_truncated=0이다.

불일치 전체 33개 플래그(strict 10, lenient 23, boxed 0)는 discrepancies.json에 조건·ID·서버/로컬 값·공식 추출기의 예측 답·정답 해설 원문과 함께 기록했다. 모두 서버만 참 방향이다. 이 비교는 오류가 동반된 로컬 산출물 기준이다.

""")
for tag,d in D["directions"].items():
    parts.append(f"- {tag}: strict IDs {d['strict']['ids']}; lenient IDs {d['lenient']['ids']}.\n")
parts.append("""
## 4. 하지 못한 것과 이유

- 예외 없는 로컬 재채점: 고정 grader가 사용하는 Windows 하위 프로세스에서 WinError 6/EOFError가 반복됐다. 정상 재채점 수치가 서버와 일치/불일치한다고 확정하지 않는다. 서버 채점표를 출발점으로 한 통계 재계산은 이 제한과 별개다.
- 서버 환경 재현·접속, GPU·학습·생성: 이번 역할에서 금지되어 수행하지 않았다.
- raw 학습 표적 재토큰화 및 개별 제외/기여 ID 확인: 자료 미제공, 모델/토크나이저 실행 없음. manifest 계수와 로그가 일관된다는 것과 실제 원시 자료까지 검증했다는 것은 구분한다.
- X5 단일 시드 외 재현성, X6 및 Stage AE: C004 입력·범위 밖이다. 제공 계획의 결과 이후 X6 절을 X5 사전 규칙으로 소급 적용하지 않았다.
- 금지 경로 열람·git 실행·기존 파일 수정·새 문항 생성은 하지 않았다. INPUT_MANIFEST의 입력은 사후 SHA로 다시 확인했다.

## 5. 재현 명령

기존 산출물 덮어쓰기를 피하려면 새 OUTBOX 디렉터리에 분석 코드만 복사해 수행한다. 아래 <새OUTBOX>는 그 새 디렉터리다. python -I를 사용하지 않는다.

```powershell
cd D:\\AI\\cts
python -B <새OUTBOX>/analyze_c004.py prepare
# 다음 채점 명령은 이번 PC에서 예외가 발생한 경로다.
python -B handoff/codex/scripts/codex_grade_local.py <새OUTBOX>/s0spec_local.json <새OUTBOX>/x5_items_local.json
python -B <새OUTBOX>/analyze_c004.py analyze
```

`run_grader_guarded.py`는 동일한 허용 래퍼를 runpy로 실행하며 __main__만 보호한다. grader 함수·채점 설정·timeout은 변경하지 않는다. 이번 환경에서 이것만으로 WinError 6이 없어지지는 않았다. `statistics.json`의 local 절은 x5_items_local.json의 오류 동반 결과이며, `execution_status.json`을 반드시 함께 읽어야 한다.

## 6. 산출물 SHA256

전체 입력은 INPUT_MANIFEST.json, 이 보고서 및 질문을 포함한 전체 산출물은 OUTPUT_MANIFEST.json에 64자리 SHA256으로 기록한다. OUTPUT_MANIFEST 자체는 자기 참조를 피하려고 목록에서 제외한다.

""")
for name in ["analyze_c004.py","statistics.json","discrepancies.json","training_manifest_checks.json","INPUT_MANIFEST.json","execution_status.json"]:
    parts.append(f"- `{name}`: `{sha(O/name)}`\n")
parts.append("""

## 7. 의견(판정 아님)

- 서버 채점 기준 세 조건의 평균 및 bootstrap 구간은 CC 분석 출력과 표시 정밀도에서 같았다. 이는 통계 재계산의 관찰이며 서버 grader의 정확성을 독립 재검증했다는 뜻은 아니다.
- 575/992=57.96%의 표적에 예산 절단이 적용됐다. 단순한 문체 보존 실험으로 해석하기 어려운 규모다. 정답 치환/추가, 답 가중, EOS/길이 정책이 동시에 바뀌었으므로 문체 하나로 원인을 분리할 수 없다.
- 1500 업데이트·6000 방문·992 기여 ID와 telemetry는 내부적으로 일관된다. 993 중 1개 제외는 개수로 약 0.10%지만, 제외 ID/표적이 없어 제외 특성은 확인하지 못했다. 업데이트/방문 기준 건너뜀 0이 표적 구성 단계의 제외 0을 뜻하지는 않는다.
- matched와 shuffled의 차이가 있는 것과 순정 C_fmt보다 더 많은 문제를 맞힌 것은 다른 주장이다. matched가 C_fmt에서 맞힌 문항 40개를 잃고 다른 40개를 얻은 관찰도 함께 보아야 한다.
- 로컬 오류 산출물은 서버 결과를 대체하지 않는다. QUESTION-C004-REGRADING.md에 실행 환경 확인 요청을 남기며, 루트 규칙에 따라 질문 기록 뒤 추가 작업을 중단한다.
""")
# Avoid accidental non-Korean heading fragment in the final report.
report="".join(parts).replace("ローカル再採点の実行例外 / ","").replace("생물학적/모델 내부","모델 내부")
put("REPORT-C004.md",report)
question="""# QUESTION-C004-REGRADING

C004 서버 채점표의 독립 통계 재계산과 manifest 계수 대조는 REPORT-C004.md에 기록했습니다. 로컬 재채점의 실행 환경 확인이 필요합니다.

허용된 codex_grade_local.py를 그대로 실행했으나 multiprocessing.spawn/reduction에서 OSError: [WinError 6], EOFError: Ran out of input이 반복됐습니다. 직접 실행·숨김 실행·__main__ 재진입 방지 실행기 모두 원문 로그를 보존했습니다. 재진입 시 새 출력 JSON 반복 작성도 관찰해 해당 값을 정상 재채점으로 취급하지 않았습니다. 채점기나 기존 래퍼는 변경하지 않았습니다.

1. CC 환경에서 동일한 고정 grader·로컬 래퍼를 예외 없이 실행하는 명령 및 Python/math-verify 환경 정보를 제공해 주세요. 필요하다면 CC가 wrapper의 main guard를 반영해 주세요. 재진입 방지 자체만으로 이번 WinError 6은 해소되지 않았습니다.
2. 가능한 경우 패키지 SHA256SUMS.txt를 완전한 64자리 값으로 새 파일에 제공하고, 실제 실행 launcher SHA 및 제외된 학습 ID 1개도 알려 주세요. 기존 파일을 덮어쓰라는 요청은 아닙니다.

Python 3.13.3, sympy 1.14.0, math-verify 0.9.0. 서버 접속이나 원본 변경을 요청하지 않으며, 서버 채점표를 기준으로 한 재계산 결과는 그대로 보존했습니다.

AGENTS.md §3의 질문 기록 후 중단 규칙에 따라 이 파일 작성 후 추가 작업을 중단합니다.
"""
inputs=read("INPUT_MANIFEST.json")
changed=[r["path"] for r in inputs if sha(ROOT/r["path"])!=r["sha256"]]
put("final_input_integrity.json",json.dumps({"checked":len(inputs),"changed":changed},indent=2))
assert not changed
manifest=[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(O.iterdir()) if p.is_file()]
manifest.append({"path":"QUESTION-C004-REGRADING.md","bytes":len(question.encode("utf-8")),"sha256":hashlib.sha256(question.encode("utf-8")).hexdigest()})
put("OUTPUT_MANIFEST.json",json.dumps(manifest,ensure_ascii=False,indent=2))
# Question is the last mutation, per the repository's stop-after-question rule.
put("QUESTION-C004-REGRADING.md",question)
print(json.dumps({"report":str(O/"REPORT-C004.md"),"inputs_checked":len(inputs),"inputs_changed":len(changed),"outputs_hashed":len(manifest),"local_regrade":"not verified; execution exceptions","logs":logs},ensure_ascii=False))

# CTS 인수인계서 — cts_260930(노트북 임시 루프) → cts(회사 PC 정식 루프)

- **작성**: 외부 감사(채팅 Claude), 원장 #410, 2026-10-03
- **넣을 위치**: `D:\AI\cts\shared\HANDOFF_cts260930_to_cts.md` (회사 PC), 사본 `C:\Users\wwwhu\cts_260930\shared\` (노트북)
- **읽는 사람**: 회사 PC Claude Code A(지휘)·B(감사)·Antigravity(실행)
- **원본(진실의 출처)**: 서버 `shoon@<GPU_SERVER_IP>:<SSH_PORT>:~/cts_e1_s5` — 로컬 사본은 모두 pull 전용

---

## 0. 왜 넘기는가, 무엇이 바뀌는가
- 09-28 저녁 회사 PC의 반복 SSH 접속이 서버 보안 시스템에 "공격"으로 등록되어 회사 PC IP(<REDACTED_IP>)가 차단됨. 09-30~10-02 동안 노트북(cts_260930)에서 임시로 루프를 돌림.
- 10-01 담당자 연락으로 차단 해제. 회사 PC에서 서버 접속이 다시 가능하므로 정식 루프(cts, `D:\AI\cts`)로 복귀한다.
- **실행자 단일화**: 서버에서 명령을 실행하는 루프는 언제나 하나. 노트북 루프가 완전히 멈춘 것을 확인한 뒤에만 회사 PC 루프가 서버 실행을 시작한다 (7절 절차).
- **서버 접속 예절 규약 v1 (영구)**: 새 연결 60초에 1회·시간당 20회 이하 / 2회 연속 실패 시 재시도 금지, Principal 대기 / 명령은 묶어서 1회 호출 / 긴 작업은 한 연결 안에서 대기(폴링 연결 반복 금지) / `IdentitiesOnly=yes` 키 1개 / 실행자 1개.

---

## 1. 최종 목표와 현재 판정

| 축 | 목표(사전등록) | 현재 |
| :--- | :--- | :--- |
| **G1 DEQ 수렴·변별** | 정본 $z^*$의 $r < 1\text{e-}2$ 비율 train $\ge 95\%$, held-out $\ge 90\%$ | 미통과. 학습 1,000의 정본 최종점 $r_{1000} < 1\text{e-}2 = 886/1000 (88.6\%)$. held-out 520 확증(P5) 미실행 |
| **G2 디코더 기여** | 글+$z^*$ 디코더: 게이트 2(바닥선) $\wedge$ 3(글만 대비) $\wedge$ 4($z_{\text{src}}$ 대비) | 미판정. S7-D 학습 입력 양식 문제(5절 #409) 확인 대기 |

---

## 2. 잠금 정본 (SHA-256)

| 문서 | SHA-256 | 비고 |
| :--- | :--- | :--- |
| `PREREG_S7_D_v3_4.md` | `88c308625fb10b62b29ef3426052ffd23f84ad5c856b071c272ab412f183c772` | v3.2 복사+diff, 척도 $r$ 정정 |
| `PREREG_S7_D_v3_5.md` | `5d7e32aa1edfebf7611f0f16769ebb344433e8ca3a554b7321d7e0979d9600ed` | 조건 E($z_{\text{src}}$), 게이트 3·4. §3 입력 양식 = `[SLOT_0..63]\n\n{problem_text}` |
| `PREREG_S7_DEQ_v1.1.md` | `bf3173b1346c0768cef5e224611d906cb033ef6ae7ac29bad52c0e7a443ad488` | 마스크 동결 개입 |
| `PREREG_S7_DEQ_v1.2.md` | `809a88be0be18e1eec2119d0e17b88dc8eb0c00b326c71ccfc88376c63bd4014` | 최소 잔차점(H-DEQ-0), held-out 520 확증 |
| 잠금 감사 기록 | `D:\AI\cts\shared\audits\AUDIT-P001_prereg_lock.md` | 판정 PASS |
| `PREREG_S7_D_v3_3.md` | — | 무효(회귀), 사용 금지 |

*개정은 언제나 직전 잠금본 복사 + 절 단위 수정 + diff -u + B 감사. 결과 열람 후 기준 변경 금지.*

---

## 3. 확정된 사실 (포인터는 서버 ~/cts_e1_s5 기준)

### 3-1. 데이터·산출물
- **학습용 $z^*$ 1,080 완료**: `results_s7/zstar_s7_a0.25_400.pt` (`bbe7ad3b...`), `results_s7/zstar_s7_a0.25_train680.pt` (`4f659e78...`). 09-28 22:27 감시자가 자동 정지.
- **held-out 520 $z^*$**: 0문항(봉인 유지). Step 3는 22:27:50 시작 6초 후 가중치 로딩 19%에서 종료, 파일·로그 0.
- **로컬 4090 계산 409문항**: 다른 체크포인트(`stage1_last.lora_only` + 허브 기저) + bf16으로 계산되어 서버와 불일치(수렴 13문항 중 12분기) $\to$ `D:\AI\cts\results_s7\quarantine_local_20260928\` 격리, 사용 금지.
- **$\rho_{2000}$은 400/400문항에서 $\rho_{1000}$과 같음**: 복사값(미실측). 수렴 집계는 $\rho_{1000}$($r = 4\rho$) 또는 재평가 $r_{1000}$만 사용.

### 3-2. 저울(계기) 검증
| 점검 | 결과 | 포인터 |
| :--- | :--- | :--- |
| **V1 Phase 2 평가 재현** | matched 80/80, shuffled 80/80 생성문 완전 일치 | `raw_instrument/v1_eval80.log` |
| **V3 정본 앵커 20 재계산** | 20/20 $\max\|\Delta\| = 0$ | `raw_instrument/v3_anchor20_rerun.log` |
| **V2 Phase 2 재채점 (채점기 90cc93e4)** | strict 5 $\to$ 1, lenient 24 $\to$ 6, matched strict 0 | `raw_instrument/v2_rescore_20260930_2349.json` |
| **저울 버튼 `instrument/check_scale.sh`** | 7항목 중 [4] FAIL (10-01 시점) | `raw_instrument/check_20261001_0156.log` |

- **결론**: $z^*$ 계산 경로와 Phase 2 평가는 비트 단위 재현. 예전 "자기 $z^*$ - 남의 $z^* +2.5\%p$"는 결함 채점기의 산물 $\to$ $z^*$만 디코더 실패는 더 확실.

### 3-3. S7-DEQ 실험 (P1~P3, 노트북 루프)
| 단계 | 결과 | 포인터 |
| :--- | :--- | :--- |
| **P1 새 실행기 (최소 잔차점 저장)** | 정본 대비 앵커 20/20 $\Delta = 0$.<br>`scripts/s7_zstar_lowest_run.py` (`779ea1f1...`),<br>`cts/deq/zstar_canonical_lowest.py` (`9f73fb27...`) | `raw_s7/anchor_equiv_P1.log` |
| **P1 $r_{1000}$ 재평가** | 학습 1,000: $< 1\text{e-}2$ 886, $< 5\text{e-}3$ 782<br>held-out 80: $< 1\text{e-}2$ 74, $< 5\text{e-}3$ 65 | `raw_s7/r1000_reeval_P1.log:27-35` |
| **P2 동결 검사 판별력** | Q#2249, 비트 동일 10/10=0, 양성 대조 10/10 > 0 | `raw_s7/freeze_discrim_P2.log` |
| **P3 마스크 동결 개입 (74문항)** | 게이트 1 52/54, 게이트 2 53/54, 게이트 3 min cos 0.9326 (실패선 하회), 자기 일관 2/54 | `raw_s7/s7deq_intervene_P3.log`,<br>`results_s7/s7deq_intervene_P3.json` |

- **P3 판정 (원장 #405)**: 라우팅 스위치가 미수렴 원인임은 개입으로 지지. 마스크 동결은 원 모델의 고정점을 주지 않으므로 (유령 고정점) 정본 풀이기로 채택 불가. P3의 $z^*$는 어떤 학습·평가에도 사용 0.
- **함의**: 하드 top-3 라우팅은 미수렴 문항에서 자기 일관 고정점이 없을 가능성 $\to$ 다음은 H-DEQ-0(최소 잔차점, P5 확증), 부족하면 H-DEQ-2(부드러운 라우팅, 별도 사전등록).

### 3-4. 디코더 학습 P4 (노트북 루프, GPU 0)
- 학습 문항 유효 999 (Q#2727: 정답 종료 토큰 1891 > max_seq_len 1536, `s5_train_decoder_v2.py:652` 필터). 집합 SHA `dcd05492...` (A·D·E 동일). LEDGER 편차 기록.
- **A($\lambda=1, z^*$)**: 3,000걸음 완주 `results_s7/decoder_A_l1_P4.pt` (`fb83971b...`). 학습 문항 $\Delta_{\text{f32}}$ 100걸음 이동평균 0.03 $\to$ 0.70.
- **D($\lambda=0, z^*$)** 진행/완주 여부와 **E($\lambda=1, z_{\text{src}}$)** 상태는 서버 로그로 확인 (`raw_s7/train_decoder_D_l0_P4.log`, `..._E_...`).
- **고정 탐침 (held-out 80 앞 20, A·D) 실행됨** $\to$ `results_s7/probe_heldout20_A1.json`, `raw_s7/probe_heldout20_A1.log` (판정 사용 0).

---

## 4. 진행 중 의심 사항 최우선 확인 — (원장 #409)
- **P4 디코더의 학습 입력에 문제 글이 없을 가능성**:
  - **증거**: 학습 입력의 해설 시작 오프셋 `target_start = 73` (64 슬롯 + 머리글만), P4 지시문 "s5_train_decoder_v2 레시피 그대로" (S5 $z^*$만 학습기), 탐침 코드의 `build_canonical_prefix_prompt(tokenizer, k_latent=64)` (문제 글 인자 없음).
  - v3.5 §3은 슬롯 뒤 문제 글 필수: `[SLOT_0..63]\n\n{problem_text}`
- **확인 방법**: 학습 스크립트의 입력 생성 행 번호 원문 + 학습 첫 문항 입력 토큰 decode 원문 + `raw_s7/s7_input_format_verification.json` 대조.
- **확인되면**: P4의 A·D·E는 "S7-zonly-999" (보조: $z^*$만 방식의 데이터 규모 검증 H-DEC-2, G2 판정 사용 0)로 등록.
- v3.5 §3 양식으로 **P4' 재학습** (새 학습기 파일 + diff, 착수 전 1문항 decode 대조, 문제 글 추가로 길이 필터에 걸리는 문항 수 먼저 보고 — 950 미만이면 BLOCK). P6 평가 스크립트도 같은 양식으로 정정.

---

## 5. 남은 순서 (외부 감사 상시 규칙 EXTERNAL_AUDIT_STANDING_ORDERS.md 적용)
1. **서버 §0** (`free`, `nvidia-smi`, `tmux`, 진행 중 학습 상태) — 접속 예절 규약 준수
2. **4절 #409 확인** $\to$ (확인 시) P4' A $\to$ D $\to$ E 재학습
3. **저울 버튼 [4] 수정** $\to$ `check_scale.sh` 7/7 (V3와 같은 경로로 테스트 호출)
4. **채점기 결정**: `90cc93e4` vs `bd25bfd4` (GitHub `heosanghun/Cognitive-Tree-Search` 브랜치 `scale-v1` 커밋 `992afa6`) 20사례 대조. 동일 $\to$ v3.5 대로 `90cc93e4` / 다름 $\to$ 결과 열람 전 v3.6 (복사+diff)
5. **P5 held-out 520 확증 단 1회** (`s7_zstar_lowest_run.py`, SHA가 P1 값과 같을 때만). 1차(최종점)·2차(최소점 $k^* \ge 50$) 승격 기준은 v1.2 §10
6. **P6 평가 600(520+80)**: 조건 C·A·B·E1·F·D1·D2·E2, strict 게이트, 대응 부트스트랩·McNemar (v3.5 원문과 대조 후)
7. **판정** — 상시 규칙 2-4절 갈림길 (초안까지 자동, 실행은 잠금 후)

---

## 6. 열린 결정 (Principal)
- **ESC-GPU1**: GPU 1 사용 허용 여부 (P4'·P5·P6 합계 수십 시간 $\to$ 병렬 시 약 하루 단축)
- **CTO(안수진) 보고**: 저울 재현·P1~P3 결과로 보고서 최종본 송부 여부 (외부 송부는 Principal 결정)

---

## 7. 인수인계 절차 (이 순서대로)
| # | 누가 | 무엇 |
| :---: | :--- | :--- |
| **1** | 노트북 Antigravity | 진행 중 GPU 작업이 있으면 다음 체크포인트 저장 직후 정지(진행 중 학습이 끝나기 직전이면 완주 후). 서버에 남은 tmux 세션·프로세스 목록과 마지막 로그 행을 `shared/reports/REPORT-HANDOFF-LAPTOP.md`에 기록. 이후 서버 접속 0 |
| **2** | 노트북 Claude Code | `STATE.md` 최종 갱신, `shared/` 전체(`tasks·reports·audits·LEDGER·LESSONS·STATE·EXTERNAL_AUDIT_STANDING_ORDERS·PROMPT-*.md`)를 서버 `~/cts_e1_s5/shared_laptop_20261003/`로 scp(덮어쓰기 0) 후 루프 종료 |
| **3** | **회사 PC Antigravity** | **서버 §0 1회 $\to$ `~/cts_e1_s5/shared_laptop_20261003/`를 `D:\AI\cts\shared\from_laptop\`로 내려받기(기존 shared 덮어쓰기 0) $\to$ 수령 파일 목록·SHA를 `REPORT-HANDOFF-COMPANY.md`로** |
| **4** | 회사 PC Claude Code A | 이 문서·`from_laptop`의 STATE/LEDGER/LESSONS를 읽고 회사 PC `shared/`의 `GOAL·GRAPH·STATE·LESSONS`를 갱신(diff 첨부). LESSONS 16~18 반영. 첫 TASK = 4절 #409 확인 |
| **5** | 회사 PC Claude Code B | 수령 SHA와 노트북 보고 SHA 대조, A 갱신분 감사 $\to$ PASS 시 루프 재개 |
| **6** | Principal | (선택) ESC-GPU1 응답. 그 외 없음 |

---

## 8. 재발 방지 규칙 (LESSONS 요약, 새로 추가된 것 포함)
- 수렴 척도는 참 잔차 $r = \|f(z)-z\| / \|z\|$ (감쇠 이동량 $\rho = \alpha \cdot r$ 아님)
- 사전등록은 복사+절 수정+diff만 (기억으로 재작성 금지 — v3.3 사고)
- CPU 진단은 `ulimit -v 62914560` · `nice -n 19` · `no_grad` (09-28 서버 메모리 사고)
- 서버 접속 예절 규약 v1 (09-28 IP 차단 사고)
- 키 재매핑 + strict=False 적재는 오류 없이 다른 모델을 만든다 $\to$ missing/unexpected 0 강제. 예상보다 빠른 속도는 조건 차이 신호 (로컬 409 사고)
- 옛 로그 복사로 점검 결과를 대체하지 않는다 (V3 사고) · 테스트를 고쳐 통과시키지 않는다
- **LESSON 17**: 풀이기를 바꾸는 실험의 1차 지표는 수렴률이 아니라 자기 일관 비율 (P3 유령 고정점)
- **LESSON 18**: "레시피 그대로"는 입력 양식까지 복제한다 $\to$ 학습·평가 착수 전 입력 1문항 decode 원문을 사전등록 §3과 대조 (#409)

---

## 9. 원장 범위
외부 감사 원장 #378~#410 (09-28~10-03). 주요:
- #383 코드베이스 결함 우선순위
- #385 v3.4 잠금
- #395 로컬 409 격리
- #400 저울 동결
- #403 저울 마감
- #405 P3 판정
- #407 999 편차
- #408 상시 규칙
- #409 P4 입력 양식
- #410 본 인수인계

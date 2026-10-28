# REPORT-CSV-FIX-AND-ROUTING-AUDIT-20261025 — 통합 CSV `strict`/`lenient` 순서 수정, M1/M3 라우팅 로그 유무, `RESULT-X8A` 서술 행 감사

- **작성자**: Antigravity
- **작성 시각**: 2026-10-25 02:02 KST
- **대응 문서**:
  - Astra: [`OUTBOX/20261025-0200-SOURCE-AUDIT/REPLY-CC-D1-P1-D2.md`](../../OUTBOX/20261025-0200-SOURCE-AUDIT/REPLY-CC-D1-P1-D2.md)
  - CC: [`INBOX/20261025-0210-CC-x8a-rootcause/REVIEW-X8B-CANDIDATE-20261025.md`](../20261025-0210-CC-x8a-rootcause/REVIEW-X8B-CANDIDATE-20261025.md) §3

---

## 1. 통합 CSV 381행 `strict`/`lenient` 교환 원인 규명 및 `v2_fixed` 산출물

1. **원인 규명**:
   - `g2v5_s0_grade.py` L16–17 및 `codex_grade_local.py` L42–43에서 문항별 채점 배열은 **`[strict, lenient, has_boxed]`** (`v[0]=strict`, `v[1]=lenient`, `v[2]=boxed_present`) 순서로 저장됩니다.
   - 반면 초기 CSV 변환 스크립트(`sync_to_cts_file_v3.py` L123)에서 `lenient_v, strict_v, boxed_v = item_map[iid]`로 언패킹 순서를 뒤집어 적어, `lenient == 1`이면서 `strict == 0`인 문항들(381행)에서 두 열이 서로 바뀌어 기록되었습니다. (나머지 22행 차이는 로컬 `math-verify 0.9.0` vs 서버 `0.5.2` 파서 버전 차이인 `id=3982`, `id=606` 등에서 기인합니다.)
2. **수정된 파일 (기존 파일 보존, 본 폴더에 신규 생성)**:
   - [`all_experiments_item_grades_v2_fixed.csv`](all_experiments_item_grades_v2_fixed.csv): `strict_v, lenient_v, boxed_v = item_map[iid]` 정순 언패킹 및 3번째 열 이름을 `boxed_present`로 수정 완료 (SHA256 `2bd5e8edb3e0af71d604336fe49ed32d5f968b537527f355b2835eb4e3006a93`).
   - [`dev600_cfmt_x8a_x7e_server_vs_local_v2_fixed.csv`](dev600_cfmt_x8a_x7e_server_vs_local_v2_fixed.csv): `C_fmt`, `X8A matched/shuffled`, `X7E matched` 10개 조건(6,000행) 서버/로컬 판정 정순 분리 CSV (SHA256 `945b4a74d6693ceb3d1daf6ccbfd471eaba18caa99a53aec102308eda13e88d4`).
   - [`dev600_cfmt_x8a_x7e_server_vs_local_v2_fixed.json`](dev600_cfmt_x8a_x7e_server_vs_local_v2_fixed.json): 위와 동일한 6,000행 JSON (SHA256 `49bc4565...` 계열 검증 완료).
   - [`TRANSFER-X8A-EXACT-SOURCES-v2-20261025.zip`](TRANSFER-X8A-EXACT-SOURCES-v2-20261025.zip): 위 `v2_fixed` CSV/JSON을 포함한 무편집 원본 전체 ZIP (SHA256 `c7b0027cb70a08e70233b74d98a67e71e85ea477088e54778bb4004e90ddc797`, 매니페스트 [`ZIP-SHA256-MANIFEST-v2.json`](ZIP-SHA256-MANIFEST-v2.json)).
3. **`X8A − X7E` 부트스트랩 CI 정정 반영**:
   - `X8A − X7E` lenient 차이(`+0.6111%p`)의 문항 단위 클러스터 부트스트랩(10,000회, `random.Random(2026)`) 95% CI는 **`[-1.1111, +2.3333]%p`** (`RESULT-X8A-readable-deq-20261023.md` L17 및 `X8_READABLE_LATENT_DEQ_20261021.md` L55와 일치)로 정정합니다.

---

## 2. M1 / M3 학습 당시 라우팅 선택 분포 로그 유무 확인 (`REVIEW-X8B-CANDIDATE-20261025.md` §3-(2))

회사 PC 정본(`D:\AI\cts`)을 전수 조사한 결과:
1. **M3 학습 스크립트 (`scripts/g2v3_m3_full_train.py`)**:
   - `g2v3_m3_full_train.py`는 학습 중 매 step에서 `routing_weights` → `sparse_module_weights(..., top_k=3)`를 계산하지만, **선택된 모듈 ID(`sel_ids`) 히스토그램이나 빈도를 로그나 manifest에 기록하지 않았습니다(미기록)**.
2. **M1 고정점(`m1_final_frozen.pt`) 단계의 라우팅 선택 기록 (`shared/from_laptop/reports/REPORT-CTS-P3.v3.md` 및 `HANDOVER_TO_CTS_20261004.md`)**:
   - M1 단계의 P3 진단 보고서(`shared/from_laptop/reports/REPORT-CTS-P3.v3.md` 및 `AUDIT-CTS-P3v3.md`)에서 20개 대조 문항 및 54개 비수렴 문항의 고정점 $z^*$에서 `top-3` 모듈 마스크 $m_{\text{frozen}}$을 조사한 기록이 존재하며, 당시에도 문항 대부분에서 동일한 소수 모듈 조합(예: `[0, 6, 15]` 또는 `[9, 13, 15]`)으로 쏠리고 3등·4등 경계(예: 모듈 6 ↔ 14)에서만 스위칭이 일어나는 현상이 보고된 바 있습니다 (`PREREG_S7_DEQ_v2.1.md` L11 참조).
   - 그러나 학습 993문항·개발 600문항 전체에 대한 전수 라우팅 선택 분포는 이번 CC의 `d1_routing.json`이 최초의 전수 집계입니다.

---

## 3. `RESULT-X8A` 및 제안서 내 "DEQ 학습"·"라우팅" 관련 서술 행 번호 (`REVIEW-X8B-CANDIDATE-20261025.md` §3-(3))

10/28 미팅 문구 및 후속 보고서 정밀화를 위해 수정·보류가 필요한 기존 문서의 정확한 줄 번호는 다음과 같습니다:

1. **[`RESULT-X8A-readable-deq-20261023.md`](../20261025-0130-full-code-and-docs/state_and_docs/results/RESULT-X8A-readable-deq-20261023.md)**:
   - **L12 & L28**: `"G1 유지 규칙(상대 잔차 10⁻² 자릿수·NaN 0): 충족."`
     - **주의/정정 사유**: 기록된 `0.032~0.036`은 $r(z_2)$가 아니라 $\rho(z_1) = \alpha r(z_1)$($\alpha=0.25$)이므로 $r(z_1) \approx 0.129 \sim 0.143$이며, $r(z_1) < 0.01$인 문항은 `0/600`이고 출력 $z_2$의 $r(z_2)$는 미측정임.
   - **L35**: `"2. 학습 문항에서는 답 CE가 0.29~0.31로 X7(0.27~0.32)과 같다 — 즉 DEQ 파라미터(약 4M)의 학습이 '학습 문항 암기'를 더하지는 않았고, 개발 문항으로 일반화되는 내용도 만들지 못했다."`
     - **주의/정정 사유**: `bb.to(bfloat16)` + `lr_deq=3e-5` 반올림 정체 및 `top_k=3` 희소 선택 쏠림으로 인해 약 3.98M DEQ 파라미터 중 10~12개 모듈은 0회 변경되었고 `gamma_param`도 0회 변경된 **부분 학습** 상태였음.
   - **L37**: `"4. 종합: 문제 글만 입력받는 DEQ 고정점은, 읽히도록 학습시켜도 1,500단계·K_unroll 2·LoRA r8 범위에서는 개발 문항의 답으로 이어지는 내용을 만들지 못했다."`
     - **주의/정정 사유**: 일반화된 DEQ 구조 기각이 아니라 **"X8-A 구현 조건(BF16 파라미터 저장 + `lr_deq=3e-5` + 희소 top-3 라우팅 + $K=2$ 전개 + 오답 포함 자체 풀이 타깃 `T0`)에서는 유의한 이득을 보이지 못했다"**로 범위를 한정해야 함.
   - **L40**: `"핵심 메시지: (a) 잠재→LM 통로는 확보·재현(오라클 +3.56), (b) 문제 글에서 계산된 어떤 잠재(DEQ·z_src·문제 은닉·읽힘 학습 DEQ)도 그 통로로 새 추론 정보를 주지 못함"`
     - **주의/정정 사유**: 위 L35·L37과 동일하게 P1(FP32 마스터) 및 T1/T3(감독 타깃 분리) 검증 전까지는 (b)의 "읽힘 학습 DEQ" 결론을 보류해야 함.
2. **[`X8_READABLE_LATENT_DEQ_20261021.md`](../20261025-0130-full-code-and-docs/state_and_docs/proposals/X8_READABLE_LATENT_DEQ_20261021.md)**:
   - **L14–16**: `"DEQ 쪽(전이 함수 Φ_θ의 학습 가능 부분 + 투영)을 '판독기가 z*에서 정답을 더 잘 쓰도록' 학습시킨다 ... 기울기를 브리지와 DEQ 파라미터 모두로 흘림"`
   - **L55–57**: `"## 8. X8-A 결과 ... (3) G1 잔차 0.032~0.036, NaN 0 충족"` (위 `RESULT-X8A` L12, L37과 동일한 보류·정정 대상).

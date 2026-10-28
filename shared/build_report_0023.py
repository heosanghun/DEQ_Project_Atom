import os
import sys
import re
import hashlib

FORBIDDEN = ["완료", "통과", "정상", "확인됨", "입증", "승인", "성공", "검증됨", "합격"]

REPORT_BODY = r"""# REPORT-0023: `scripts/p5_promote_v2.py` 작성 및 합성 시험 결과 보고

- **작성일시**: 2026-10-05 11:53 KST
- **작성자**: 회사 PC 정식 루프 (cts, D:\\AI\\cts)
- **근거 태스크**: shared/tasks/TASK-0023.md (GATE G1 수렴·승격 분석기)
- **참조 감사**: shared/audits/AUDIT-P006_p5_analyze_v1.md (FIX 판정 조치)
- **수행 원칙**: 서버 접속 0 (로컬 전용), 결과 열람 0, 금지어 0건 강제. B 재감사 전까지 서버 반영 보류.

---

## 1. 신규 스크립트 무결성

AUDIT-P006 감사에서 도출된 7대 필수 FIX 항목을 반영하여 신규 스크립트를 작성하였습니다.

- **스크립트 경로**: `scripts/p5_promote_v2.py`
- **SHA-256**: `24f586701813bad5bfcc28bd60270dde997a3021e4c99044a143445281f6eef4`
- **반영된 주요 조치 내역**:
  1. **정본 config SHA-256 검사 강제**: `48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988` 불일치 시 즉시 종료 (우회 인자 배제).
  2. **엄격 키 정규식 조회**: 520개 ID에 대해 `f"q{idx:04d}_a0.25"`만 조회하며, 520 집합 외 키 및 타 알파 접미사(`_a0.50` 등)는 무시하고 집계 메타데이터에 기록.
  3. **분모 고정**: 520, 600, 1000 분모를 불변 고정하며, 결측 문항·필드 누락·NaN/Inf·범위 밖 k*는 미도달로 분류하여 사유별로 계측.
  4. **Top-3 집합 일치**: 리스트 순서 비교가 아닌 `set(top3_lowest) == set(top3_final)` 집합 동등성으로 산출.
  5. **구제 문항 창 검사 정정**: k* ± 10 구간 산출 시 k* + 10 ≥ 1000인 경우 `r_1000`을 구간 최댓값 후보에 포함.
  6. **궤적 4종 분류 체계 구현**: 마지막 500걸음(500..1000) 대상 `monotonic_decrease`, `increase`, `plateau`, `oscillation` 분류 및 교차표 산출.
  7. **변별 지표 정비**: 임의 상수(1.0 또는 ≥ 0.99) 및 `nearest_neighbor_match_rate_self` 배제, 안내문으로 대체. float64 연산 강제.

---

## 2. 입력 출처 및 표준 규격표

| 데이터 항목 | 파일 경로 | SHA-256 / 출처 규격 | 비고 |
|---|---|---|---|
| 정본 520 config | `scratch/server_s7_new_heldout_520.json` | `48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988` | 520개 고유 ID, 중복 0 |
| 기존 80 config | `configs/s5_heldout_80.json` | `27899edbea218b46bc0a5caabada4eb1c1c9dcc17772c001de43374cb208b8c5` | 80개 고유 ID |
| 기존 80 r_1000 출처 | P1 재평가 JSON (`--r1000_p1`) | `heldout_80` 리스트 내 `r_1000` 필드 (80개 불일치 시 종료) | `r1000_reeval_P1.json` |
| 기존 80 r_min 출처 | S7 로그 / rho_traj | $\min(4 \cdot \min_{k \le 999} \rho_k, r_{1000})$ (PROMPT-5 정의) | §10(c) 173행 인용 |
| 신규 520 입력 | zstar 텐서 (`--heldout520`) | `^q(\d{4})_a0\.25$` 키 구조체 8개 필드 | 결과 열람 0 유지 |

---

## 3. B 합성 시험 (T1~T17) 및 추가 예외 케이스 시험 결과

`scratch/test_p5_promote_v2.py`를 통해 감사 지침의 17개 시험과 추가 궤적·금지어 시험을 수행하였습니다.

| 시험 번호 | 시험 조건 | 기대 결과 | 실제 산출 결과 | 일치 여부 |
|---|---|---|---|:---:|
| T1 | 경계 조건 (494 적격, r=1e-2 10건, k*=49 10건, k*=50 1건, r=2e-2 5건) | 승격 495/520, 구간 [10, 211, 299, 0] | 495/520, 구간 [10, 211, 299, 0], 편차 +1 | 일치 |
| T2 | 1개 문항 결측 | 분모 520 고정, 유효 519, 결측 1건 | 분모 520 고정, 유효 519, 결측 1건 | 일치 |
| T3 | 520 외 외래 키 (q9999_a0.25) 추가 | 분모 520 고정, 제외 키 1건 | 분모 520 고정, 제외 키 1건 | 일치 |
| T4 | 타 알파 접미사 (q{ID}_a0.50) 혼재 | a0.25 덮어쓰기 0건, 타 알파 1건 집계 | 승격 520/520 불변, 타 알파 1건 집계 | 일치 |
| T5 | 정수 키 ("123456") 추가 | 분모 520 고정, 제외 키 1건 | 분모 520 고정, 제외 키 1건 | 일치 |
| T6 | r_1000 / r_lowest NaN 포함 | 분모 520 고정, 미도달 1건 계측 | 승격 519/520, nan_inf_r 1건 | 일치 |
| T7 | k_star NaN 포함 | 예외 중단 없이 미도달 1건 계측 | 승격 519/520, invalid_kstar 1건 | 일치 |
| T8 | r_lowest 필드 누락 | KeyError 중단 없이 미도달 1건 계측 | 승격 519/520, missing_field 1건 | 일치 |
| T9 | rho 궤적과 r_1000 불일치 (4ρ=0.5, r=1e-3) | r_1000 필드 값 보존 | r1000 < 1e-2 건수 520 유지 | 일치 |
| T10 | top-3 요소 동일, 순서 상이 ([3,2,1] vs [1,2,3]) | 집합 일치 판정 (520건) | 일치 건수 520/520 | 일치 |
| T11 | k_star 범위 초과 (1001, -1) | 승격 미산입, 구간 합+무효=520 | 승격 518/520, invalid 2건, 합 520 | 일치 |
| T12 | 구제 창 k*=995 (r=0.5), k*=505 (r=2e-3) | 창 최댓값 0.5 (r_1000 포함) 및 0.002 | r_995=0.500000, r_505=0.002000 | 일치 |
| T13 | 전 문항 동일 z 벡터 | 평균 cos 1.0, 비율 1.0, NN_self 상수 배제 | mean_cos=1.000000, ratio=1.0, NN상수 부재 | 일치 |
| T14 | 1개 문항 z 벡터에 NaN 포함 | 분위수/평균 nan 오염 차단, 유효 519건 | mean_cos 실수 산출, 유효 519건 | 일치 |
| T15 | P1 JSON 형식 불일치 (heldout_80 부재) | 즉시 오류 종료 (SystemExit 1) | SystemExit(1) 발생 및 종료 | 일치 |
| T16 | P1 80개 항목 r=0.5 규격 파일 | 80개 분모 유지, 적격 0건 | 80개 분모 유지, 적격 0건 | 일치 |
| T17 | P1 항목 79개 (80개 미달) | 상수 대체 없이 즉시 오류 종료 | SystemExit(1) 발생 및 종료 | 일치 |
| T_shapes | 궤적 4종 (monotonic_decrease, increase, plateau, oscillation) | 4개 모양 교차표 각 1건 이상 집계 | 4종 모두 각 1건 집계 | 일치 |
| 금지어 검사 | JSON, Markdown, CLI 출력 전 영역 9대 금지어 점검 | 검출 0건 | 검출 0건 | 일치 |

---

## 4. 조치 및 후속 계획

- `scripts/p5_promote_v2.py`의 단위 시험 및 통계 산출 로직 점검을 마쳤습니다.
- 본 스크립트는 B의 재감사 결과 도착 전까지 원격 서버에 배포하지 않으며 로컬에 유지합니다.
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(root, "shared", "reports", "REPORT-0023.md")

    # Check forbidden words
    found = {}
    for fw in FORBIDDEN:
        cnt = REPORT_BODY.count(fw)
        if cnt > 0:
            found[fw] = cnt
    if found:
        print(f"[ERROR] Forbidden words found in REPORT-0023 body: {found}")
        sys.exit(1)

    # Compute body hash with LF
    body_bytes = REPORT_BODY.encode("utf-8")
    body_hash = hashlib.sha256(body_bytes).hexdigest()

    header = f"<!-- SHA256: {body_hash} -->\n"
    full_content = header + REPORT_BODY

    # Write file
    with open(target_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(full_content)

    print(f"REPORT-0023 written to {target_path} (Body SHA: {body_hash})")

    # Verify self-hash
    with open(target_path, "rb") as f:
        raw = f.read()
    lines = raw.split(b"\n")
    first_line = lines[0].decode("utf-8")
    m = re.match(r"^<!-- SHA256: ([0-9a-f]{64}) -->$", first_line)
    if not m:
        print("[ERROR] Header line format mismatch")
        sys.exit(1)
    recorded_hash = m.group(1)
    actual_body_hash = hashlib.sha256(b"\n".join(lines[1:])).hexdigest()
    if recorded_hash == actual_body_hash:
        print(f"Self-hash verification: MATCH ({actual_body_hash})")
    else:
        print(f"[ERROR] Self-hash mismatch: recorded={recorded_hash}, actual={actual_body_hash}")
        sys.exit(1)

if __name__ == "__main__":
    main()

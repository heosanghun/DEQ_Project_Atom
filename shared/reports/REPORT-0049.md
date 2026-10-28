<!-- SHA256: 10894a27b807cc3b8810ee58e8e97226f338d9f6bbaa39a71404c2e6f292b845 -->

# REPORT-0049: (A) ①-a 재계산 분석기 작성·로컬 단위시험 및 (B) 19:50 서버 수집 보고서

- **작성 일시**: 2026-10-06 19:53 KST
- **작성 주체**: 회사 PC 정식 루프 (cts, `D:\AI\cts`)
- **근거 태스크**: [TASK-0049.md](file:///d:/AI/cts/shared/tasks/TASK-0049.md)
- **근거 감사 보고서**: [AUDIT-0048.md](file:///d:/AI/cts/shared/audits/AUDIT-0048.md), [AUDIT-P008.md](file:///d:/AI/cts/shared/audits/AUDIT-P008.md)
- **근거 사전등록**: [PREREG_S7_DEQ_v1.4.md](file:///d:/AI/cts/shared/prereg/PREREG_S7_DEQ_v1.4.md) (잠금, SHA-256 `041125221a6360361c0aef3f3fd975abca0b58601d28b577282b297c36c3e50d`) §10(h)
- **절대 조건 준수 현황**:
  - 서버 GPU 사용 0건: 준수 (이번 TASK는 GPU 실행 없음, CPU 검속 및 원격 텔레메트리만 수집)
  - `p6_chain` (GPU 0) 및 `s7deq_soft_v2` (GPU 1) 무간섭: 준수 (kill/attach/수정 0건)
  - 기존 파일 덮어쓰기 금지: 준수 (`p5_promote_1a_replay.py` 신규 작성)
  - 서버 접속 규약: 19:50 이후 묶음 1회 및 65초 쿨다운 준수 (SFTP 19:50:17 $\rightarrow$ SSH 19:51:25)
  - 금지어 규칙: 9대 금지어 사용 0건
  - 수치 포인터 표기: 수치마다 로그 행 번호 및 실측치 명시

---

## 1. (A) ①-a 재계산 분석기 작성 및 로컬 합성 시험

### 1.1 분석기 명세 ([`scripts/p5_promote_1a_replay.py`](file:///d:/AI/cts/scripts/p5_promote_1a_replay.py))
- **SHA-256**: `87a180e1a9313808ae564f7a7e44945a9021f4e306daccd9fec3cef4a4f20adf`
- **구현 규약**: PREREG_S7_DEQ_v1.4 §10(h)
  - 고정 입력 파일 SHA-256 기계적 검증 (시작 시 불일치 시 중단)
  - `zstar_train1000_P4.pt` 325개 키 존재 및 `(64, 2560)` 형상 강제
  - 재현 점검 요건 (FIX-2):
    - (i) $\min(ho_{\text{traj}})$ 상대차 $\le 10^{-2}$ 또는 절대차 $\le 1.2 \times 10^{-7}$ (잡음 바닥 하한 반영)
    - (ii) `z_star` 코사인 유사도 $\ge 0.9999$ (`torch.float64`)
  - 도달 판정: $r_{\min} < 10^{-2}$ 및 $k^*_{\text{replay}} \ge 50$
  - ①-a 지표 3종 산출: 보수치(675), 낙관치(991), v1.4 재계산치 및 민감도 수치
  - Rule #381 메모리 아레나 제한 및 `torch.no_grad()` 컨텍스트 적용

### 1.2 로컬 합성 단위 시험 9종 결과 (`scratch/test_p5_promote_1a_replay.py`)
```text
======================================================================
=== STARTING P5 PROMOTE 1A REPLAY LOCAL SYNTHETIC UNIT TESTS ===
======================================================================

--- Case 1: All reached (모두 도달) ---
  Conservative: 675/1000
  Optimistic:   991/1000
  v1.4 Replay:  991/1000
  Criterion met (>=950): True (diff=+41)
  Reproduced count: 325/325
  PASS: Case 1 all reached verified.

--- Case 2: k* < 50 mixed (k*<50 섞임) ---
  v1.4 Replay: 986/1000 (expected 986)
  k* < 50 count: 5
  k* < 50 list: [14, 75, 80, 84, 101]
  PASS: Case 2 k*<50 mixed verified.

--- Case 3: Reproduction failed (재현 실패) ---
  v1.4 Replay: 986/1000 (expected 986)
  Sensitivity: 991/1000 (expected 991)
  Failed reproduction count: 5
  Failed but would reach count: 5
  PASS: Case 3 reproduction failure & sensitivity verified.

--- Case 4: Missing items (결측) ---
  Missing items count: 4 (expected 4)
  Missing items list: [14, 75, 80, 84]
  v1.4 Replay: 987/1000 (expected 987)
  PASS: Case 4 missing items verified.

--- Case 5: NaN / Inf (NaN 발생) ---
  NaN/Inf items count: 1 (expected 1)
  NaN/Inf items list: [14]
  PASS: Case 5 NaN/Inf handling verified.

--- Case 6: Noise floor (잡음 바닥 min_rho < 1.2e-7) ---
  Noise floor item Q#14 in reproduced: False (rel_diff > 1e-2 but abs_diff <= 1.2e-7 satisfies Cond i)
  PASS: Case 6 noise floor absolute threshold verified.

--- Case 7: Boundary values (경계값 950 vs 949) ---
  Boundary 950: v14_1a = 950, criterion_met = True
  Boundary 949: v14_1a = 949, criterion_met = False
  PASS: Case 7 boundary values 950 and 949 verified.

--- Case 8: Missing keys / wrong shape in P4 ---
  Successfully caught missing key in P4: 'Key q0014_a0.25 missing from P4 file ...'
  Successfully caught wrong shape in P4: Shape of z_star in q0014_a0.25 is torch.Size([32, 2560]), expected (64, 2560)
  PASS: Case 8 P4 missing key & wrong shape abort verified.

--- Case 9: Pre-check 5 items evaluation ---
  Pre-check all pass: True
    Q#31: exp=829, act=829, pass=True
    Q#277: exp=709, act=709, pass=True
    Q#406: exp=999, act=999, pass=True
    Q#711: exp=809, act=809, pass=True
    Q#858: exp=999, act=999, pass=True
  Mismatched pre-check all pass: False (expected False)
  PASS: Case 9 precheck 5 items verified.

======================================================================
=== ALL 9 SYNTHETIC UNIT TESTS COMPLETED SUCCESSFULLY ===
======================================================================
```

---

## 2. (B) 서버 업로드 파일 및 SHA-256 계측

| 파일 경로 | 로컬 SHA-256 | 서버 계측 SHA-256 (`scratch/task0049_out.txt`) | 일치 여부 |
|---|---|---|:---:|
| `configs/s7_train1000_kstar_replay_325.json` | `550250e399ea96560b8360cdbdd539a82983a25fb435f994d0319da10a8a7862` | `550250e399ea96560b8360cdbdd539a82983a25fb435f994d0319da10a8a7862` | 일치 |
| `scripts/p5_promote_1a_replay.py` | `87a180e1a9313808ae564f7a7e44945a9021f4e306daccd9fec3cef4a4f20adf` | `87a180e1a9313808ae564f7a7e44945a9021f4e306daccd9fec3cef4a4f20adf` | 일치 |

---

## 3. (B) 원격 텔레메트리 및 서버 검속 실측 원문

- **실행 환경**: `shared/cts_ssh.py` 경유 (SFTP 업로드 19:50:17 $\rightarrow$ 65초 쿨다운 $\rightarrow$ SSH 수집 19:51:25)
- **실행 원문 (`scratch/task0049_out.txt`)**:
```text
================================================================================
=== TASK-0049: SERVER COLLECTION & VERIFICATION (19:50+ KST) ===
Timestamp: 2026-10-06 19:51:29 KST
================================================================================
=== 1. UPLOADED FILES SHA-256 VERIFICATION ===
550250e399ea96560b8360cdbdd539a82983a25fb435f994d0319da10a8a7862  configs/s7_train1000_kstar_replay_325.json
-rw-rw-r-- 1 shoon shoon 2720 2026-10-06 19:50:24.967031980 +0900 configs/s7_train1000_kstar_replay_325.json
87a180e1a9313808ae564f7a7e44945a9021f4e306daccd9fec3cef4a4f20adf  scripts/p5_promote_1a_replay.py
-rwxr-xr-x 1 shoon shoon 20002 2026-10-06 19:50:25.005031964 +0900 scripts/p5_promote_1a_replay.py
=== 2. RAW_S7 S7DEQ_SOFT_V2 LOG ANCHOR & RUNNING STATUS ===
--- 2.1. ANCHOR CHECK SUMMARY LINES ---
--- 2.2. ANCHOR CHECK FULL SECTION ---
=== §5 PRE-LAUNCH ANCHOR CHECK (GPU 1, F2~F5) ===
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
[0001/0001] q0020_a0.25: r_1000=1.016980e-06, r_lowest=1.016980e-06 at k*=1000, top3_final=[13, 15, 18], top3_lowest=[13, 15, 18] (55.46s)
================================================================================
Finished. Total items saved: 1 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
[0001/0001] q0020_a0.25: r_1000=6.574811e-07, r_lowest=6.574811e-07 at k*=1000, r_hard_lowest=6.167322e-03, b_steps=852/1000 (85.2%), in_b=True (106.78s)
================================================================================
Finished. Total items saved: 1 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 1 items.
[0001/0001] q0024_a0.25: r_1000=2.328545e-05, r_lowest=2.328545e-05 at k*=1000, top3_final=[2, 12, 16], top3_lowest=[2, 12, 16] (56.43s)
================================================================================
Finished. Total items saved: 2 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 1 items.
[0001/0001] q0024_a0.25: r_1000=8.285183e-06, r_lowest=8.285183e-06 at k*=1000, r_hard_lowest=5.770132e-03, b_steps=824/1000 (82.4%), in_b=True (104.87s)
================================================================================
Finished. Total items saved: 2 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 2 items.
[0001/0001] q0032_a0.25: r_1000=1.254489e-04, r_lowest=1.254489e-04 at k*=1000, top3_final=[2, 7, 16], top3_lowest=[2, 7, 16] (56.56s)
================================================================================
Finished. Total items saved: 3 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 2 items.
[0001/0001] q0032_a0.25: r_1000=7.183855e-04, r_lowest=5.893705e-04 at k*=710, r_hard_lowest=5.893705e-04, b_steps=27/1000 (2.7%), in_b=False (58.39s)
================================================================================
Finished. Total items saved: 3 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 3 items.
[0001/0001] q0000_a0.25: r_1000=1.500415e-04, r_lowest=1.500415e-04 at k*=1000, top3_final=[2, 12, 16], top3_lowest=[2, 12, 16] (79.42s)
================================================================================
Finished. Total items saved: 4 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 3 items.
[0001/0001] q0000_a0.25: r_1000=2.306190e-04, r_lowest=2.306190e-04 at k*=1000, r_hard_lowest=2.306190e-04, b_steps=20/1000 (2.0%), in_b=False (81.38s)
================================================================================
Finished. Total items saved: 4 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 4 items.
[0001/0001] q0004_a0.25: r_1000=5.284588e-03, r_lowest=4.451458e-03 at k*=767, top3_final=[14, 15, 18], top3_lowest=[14, 15, 18] (81.06s)
================================================================================
Finished. Total items saved: 5 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 4 items.
[0001/0001] q0004_a0.25: r_1000=4.673411e-05, r_lowest=4.673411e-05 at k*=1000, r_hard_lowest=7.533004e-03, b_steps=930/1000 (93.0%), in_b=True (151.76s)
================================================================================
Finished. Total items saved: 5 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 5 items.
[0001/0001] q0008_a0.25: r_1000=9.525316e-03, r_lowest=1.553590e-03 at k*=904, top3_final=[5, 13, 17], top3_lowest=[5, 13, 14] (145.75s)
================================================================================
Finished. Total items saved: 6 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 5 items.
[0001/0001] q0008_a0.25: r_1000=9.883068e-05, r_lowest=9.883068e-05 at k*=1000, r_hard_lowest=6.982923e-03, b_steps=948/1000 (94.8%), in_b=True (284.36s)
================================================================================
Finished. Total items saved: 6 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 6 items.
[0001/0001] q0012_a0.25: r_1000=1.318687e-03, r_lowest=1.318687e-03 at k*=1000, top3_final=[9, 14, 18], top3_lowest=[9, 14, 18] (75.26s)
================================================================================
Finished. Total items saved: 7 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 6 items.
[0001/0001] q0012_a0.25: r_1000=1.264679e-03, r_lowest=1.264679e-03 at k*=1000, r_hard_lowest=1.264679e-03, b_steps=225/1000 (22.5%), in_b=False (92.73s)
================================================================================
Finished. Total items saved: 7 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 7 items.
[0001/0001] q0016_a0.25: r_1000=1.740179e-03, r_lowest=1.707585e-03 at k*=952, top3_final=[5, 13, 15], top3_lowest=[5, 13, 15] (76.52s)
================================================================================
Finished. Total items saved: 8 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 7 items.
[0001/0001] q0016_a0.25: r_1000=2.170279e-03, r_lowest=1.972646e-03 at k*=833, r_hard_lowest=1.972646e-03, b_steps=183/1000 (18.3%), in_b=False (91.27s)
================================================================================
Finished. Total items saved: 8 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 8 items.
[0001/0001] q0028_a0.25: r_1000=5.581368e-03, r_lowest=5.581368e-03 at k*=1000, top3_final=[13, 15, 17], top3_lowest=[13, 15, 17] (81.97s)
================================================================================
Finished. Total items saved: 9 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 8 items.
[0001/0001] q0028_a0.25: r_1000=1.708596e-03, r_lowest=4.740982e-04 at k*=750, r_hard_lowest=8.769847e-03, b_steps=955/1000 (95.5%), in_b=True (161.21s)
================================================================================
Finished. Total items saved: 9 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 9 items.
[0001/0001] q0036_a0.25: r_1000=2.485071e-04, r_lowest=2.485071e-04 at k*=1000, top3_final=[9, 14, 18], top3_lowest=[9, 14, 18] (55.05s)
================================================================================
Finished. Total items saved: 10 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 9 items.
[0001/0001] q0036_a0.25: r_1000=2.555431e-04, r_lowest=2.555431e-04 at k*=1000, r_hard_lowest=2.555431e-04, b_steps=455/1000 (45.5%), in_b=False (81.46s)
================================================================================
Finished. Total items saved: 10 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 10 items.
[0001/0001] q0040_a0.25: r_1000=5.560194e-05, r_lowest=5.560194e-05 at k*=1000, top3_final=[14, 15, 18], top3_lowest=[14, 15, 18] (63.49s)
================================================================================
Finished. Total items saved: 11 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 10 items.
[0001/0001] q0040_a0.25: r_1000=5.165331e-05, r_lowest=5.165331e-05 at k*=1000, r_hard_lowest=5.165331e-05, b_steps=262/1000 (26.2%), in_b=False (79.07s)
================================================================================
Finished. Total items saved: 11 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 11 items.
[0001/0001] q0044_a0.25: r_1000=3.309842e-05, r_lowest=3.309842e-05 at k*=1000, top3_final=[14, 15, 18], top3_lowest=[14, 15, 18] (59.62s)
================================================================================
Finished. Total items saved: 12 to results_s7/anchor_ref_gpu1_check.pt
================================================================================
================================================================================
S7 DEQ Smooth Top-3 Switch Run (v2): eps=0.02, alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_soft_eps0.02_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 11 items.
[0001/0001] q0044_a0.25: r_1000=6.810080e-04, r_lowest=4.176850e-04 at k*=679, r_hard_lowest=4.176850e-04, b_steps=421/1000 (42.1%), in_b=False (83.16s)
================================================================================
Finished. Total items saved: 12 to results_s7/anchor_soft_eps0.02_check.pt
================================================================================
================================================================================
S7 Lowest-Residual DEQ Run: alpha=0.25, max_nfe=1000, items=1
Device: cuda:0 | Seed: 2030 | Sigma: 0.02 | Tol: 0.0
Output: results_s7/anchor_ref_gpu1_check.pt
================================================================================
[BACKBONE-EVIDENCE] loading from: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
Loaded existing output dictionary with 12 items.
--- 2.3. RUNNING EPS SECTION HEADER & PROGRESS LINES ---
RUNNING EPS not reached yet
Progress item lines (line numbers and item index only, r-values omitted):
76:[0002/0002] q
102:[0001/0001] q
112:[0001/0001] q
123:[0001/0001] q
134:[0001/0001] q
145:[0001/0001] q
156:[0001/0001] q
167:[0001/0001] q
178:[0001/0001] q
189:[0001/0001] q
200:[0001/0001] q
211:[0001/0001] q
222:[0001/0001] q
233:[0001/0001] q
244:[0001/0001] q
255:[0001/0001] q
266:[0001/0001] q
277:[0001/0001] q
288:[0001/0001] q
299:[0001/0001] q
310:[0001/0001] q
321:[0001/0001] q
332:[0001/0001] q
343:[0001/0001] q
354:[0001/0001] q
=== 3. TMUX SESSIONS & GPU STATUS ===
--- 3.1. TMUX SESSIONS ---
p6_chain: 1 windows (created Mon Oct  5 19:54:38 2026)
s7deq_soft_v2: 1 windows (created Tue Oct  6 19:00:26 2026)
--- 3.2. GPU 0~7 STATUS ---
index, uuid, utilization.gpu [%], memory.used [MiB], temperature.gpu, power.draw [W]
0, GPU-71be98b0-6b81-9257-c622-97dc97d46480, 99 %, 15818 MiB, 53, 133.20 W
1, GPU-9410691f-6857-1642-ec7c-7596970e2f48, 99 %, 30766 MiB, 51, 117.06 W
2, GPU-c81311d8-d210-3857-843e-bb38d24771e1, 0 %, 16 MiB, 30, 21.92 W
3, GPU-a9fed959-7113-e747-dac4-a515ac50c160, 0 %, 16 MiB, 32, 18.43 W
4, GPU-e27bdb0d-9534-1cb5-ea0e-7d427b1e8fe5, 0 %, 16 MiB, 27, 11.93 W
5, GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a, 0 %, 16 MiB, 27, 6.94 W
6, GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c, 0 %, 16 MiB, 28, 6.62 W
7, GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b, 0 %, 16 MiB, 28, 8.54 W
--- 3.3. GPU COMPUTE APPS ---
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-71be98b0-6b81-9257-c622-97dc97d46480, 245681, 15798 MiB
GPU-9410691f-6857-1642-ec7c-7596970e2f48, 261415, 30746 MiB
--- 3.4. COMPUTE APP PROCESS COMMAND LINES ---
 245681 Tue Oct  6 18:21:47 2026 /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond C
 261415 Tue Oct  6 19:50:25 2026 /home/shoon/system1_5/.venv/bin/python scripts/s7_zstar_lowest_run.py --items 48 --out results_s7/anchor_ref_gpu1_check.pt --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --tol 0.0 --max_nfe 1000 --device cuda:0 --config configs/zstar_canonical.yaml
=== 4. P6_CHAIN STATUS & GENERATION FILES ===
--- 4.1. P6_CHAIN LOG LAST 5 LINES ---
[P6-CHAIN] Stage 0: Comparing anchor20_gpu0.pt vs GPU 1 & GPU 2 on CPU...
[P6-CHAIN] Stage 0 complete. Proceeding to Stage 1.
[P6-CHAIN] Stage 1: Launching z_src(600) calculation...
[P6-CHAIN] Stage 1 completed successfully. Created /home/shoon/cts_e1_s5/raw_s7/ZSRC600_DONE.flag.
[P6-CHAIN] Running Generation Condition C (shift=)...
--- 4.2. P6 GENERATION JSONL FILES LISTING & LINE COUNTS ---
-rw-rw-r-- 1 shoon shoon    400 2026-10-04 20:50:18.396724851 +0900 results_s7/p6_gen_A_smoke.jsonl
-rw-rw-r-- 1 shoon shoon    396 2026-10-04 20:50:59.363878051 +0900 results_s7/p6_gen_B1_smoke.jsonl
-rw-rw-r-- 1 shoon shoon 356764 2026-10-06 19:51:19.291005452 +0900 results_s7/p6_gen_C.jsonl
-rw-rw-r-- 1 shoon shoon    423 2026-10-04 20:51:30.856994710 +0900 results_s7/p6_gen_C_smoke.jsonl
-rw-rw-r-- 1 shoon shoon    321 2026-10-04 20:52:09.104135179 +0900 results_s7/p6_gen_E1_smoke.jsonl
-rw-rw-r-- 1 shoon shoon    425 2026-10-04 20:52:51.487289397 +0900 results_s7/p6_gen_F_smoke.jsonl
     2 results_s7/p6_gen_A_smoke.jsonl
     2 results_s7/p6_gen_B1_smoke.jsonl
   273 results_s7/p6_gen_C.jsonl
     2 results_s7/p6_gen_C_smoke.jsonl
     2 results_s7/p6_gen_E1_smoke.jsonl
     2 results_s7/p6_gen_F_smoke.jsonl
   283 total
=== 5. ZSTAR_TRAIN1000_P4.PT 325 KEYS & SHAPE CHECK (CPU, NICE 19) ===
Loading results_s7/zstar_train1000_P4.pt on CPU with torch.no_grad()...
P4 325 keys check result: total=325, missing=0, wrong_shape=0
P4 325 keys strictly present with shape (64, 2560). Check PASSED.
=== TASK-0049 COLLECTION FINISHED ===
```

---

## 4. 요약 및 후속 조치

1. **(A) ①-a 재계산 분석기 준비**:
   - `scripts/p5_promote_1a_replay.py` 작성 및 로컬 합성 시험 9종 전수 충족.
   - 서버 업로드 및 SHA-256 `87a180e1a9313808ae564f7a7e44945a9021f4e306daccd9fec3cef4a4f20adf` 일치 계측.
2. **(B) 서버 검속 결과**:
   - `zstar_train1000_P4.pt` 내 325개 키 존재 및 `(64, 2560)` 형상 전수 일치 계측 (누락 0, 형태오류 0).
   - `s7deq_soft_v2` 세션 및 `p6_chain` 세션 무간섭 가동 유지 계측.
3. **후속 대기**:
   - CC 감사 및 다음 태스크 지시 대기 (`python shared/wait_for_task.py 50`).

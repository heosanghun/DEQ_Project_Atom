# P5 1-NN Retrieval Analysis Summary (v5)

- **Definition Source**: PREREG_S7_DEQ_v1.3 §10(g)
- **Analyzer Version**: v5 (SHA-256: `f444822620e678cda6139fbe8bc17d57dceb8e95922d7c04950413e67779d704`)
- **NN Mode**: `seedpair`
- **Canonical Denominator**: 520
- **Config 520 Path**: `configs/s7_new_heldout_520.json` (SHA-256: `48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988`)
- **Target File (Seed 2030)**: `results_s7/zstar_s7_a0.25_heldout520_lowest.pt` (SHA-256: `365ac9a44150546dd424c33f2c5ae9cabc71f474f3f2cfd5e2d042fe6b4061ae`)
- **Query File (Seed 2031)**: `results_s7/zstar_s7_a0.25_heldout520_seed2031.pt` (SHA-256: `e3f9b30ae2773e2b64f9e9b2ec08d0cfb2686f62f002c946c080115138abf3ea`)
- **Identical File Specified**: False

## 1. 1-NN Retrieval Accuracy & Multiple Equilibria
| Representation | Dimension | Match Count (/ 520) | Match Ratio | Diff from 515 | Mult-Eq Count (< 0.99) | Mult-Eq Ratio | Mult-Eq Evaluated | Missing | Invalid | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Flattened (Primary) | 163,840 | 519 | 0.998077 | 4 | 8 | 0.015385 | 520 | 0 | 0 | Primary |
| Mean-Pooled | 2,560 | 519 | 0.998077 | 4 | 2 | 0.003846 | 520 | 0 | 0 | Reporting Only |

## 2. Mismatch Breakdown (Flattened 163,840-d)
| Mismatch Category | Count | Proportion (/ 520) |
|---|:---:|:---:|
| argmax 다름 (`argmax_mismatch`) | 1 | 0.001923 |
| 동률 (`margin_tie`, $\Delta < 10^{-12}$) | 0 | 0.000000 |
| 질의 결측 (`query_missing`) | 0 | 0.000000 |
| 비교 대상 결측 (`target_missing`) | 0 | 0.000000 |
| z_star 필드 없음 (`z_star_missing`) | 0 | 0.000000 |
| 무효 텐서 (`invalid_tensor`) | 0 | 0.000000 |
| **총 불일치 수 (`total_mismatch`)** | **1** | **0.001923** |

## 3. Cosine Distribution (Diagonal $\cos(z^{{(2030)}}_i, z^{{(2031)}}_i)$)
| Metric | Min | Q25 | Median | Q75 | Max | Mean | Std |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Flattened (163,840-d) | 0.844507 | 1.0 | 1.0 | 1.0 | 1.0 | 0.999289 | 0.007548 |
| Mean-Pooled (2,560-d) | 0.845777 | 1.0 | 1.0 | 1.0 | 1.0 | 0.999638 | 0.006837 |

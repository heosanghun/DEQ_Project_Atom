# S7 DEQ Smooth Top-3 Switch Analysis (eps=0.005)

## Comparison Table (Smooth Switch vs P3 Baseline)
| Metric | Definition | Smooth Switch (eps=0.005) | P3 Baseline (Mask Freeze) |
|---|---|:---:|:---:|
| (1) Fixed-point arrival (54) | r* < 5e-3 count / 54 | 48/54 (88.9%) | 52/54 (0.963) |
| (1) Strict arrival (54) | r* < 1e-3 count / 54 | 25/54 | 53/54 (0.981) |
| (2) Control stability min cos | min cos(z*_eps, z_ref) | 0.970263 | 0.932631 |
| (2) Control cos >= 0.9990 | count / 20 | 14/20 | 10/20 |
| (2) Control cos < 0.9950 | count / 20 | 3/20 | 8/20 |
| (3) Final residual r_1000 | median (max) | 5.136161e-04 (2.507212e-02) | - |
| (4) Boundary step ratio (54) | mean (max) | 53.0% (94.2%) | - |
| (4) Boundary step ratio (20) | mean (max) | 16.8% (87.4%) | - |
| (5) Original machine residual (54) | r_hard < 5e-3 count | 18/54 | - |
| (5) Boundary fixed-point (54) | d <= eps count | 39/54 | - |

> [!NOTE]
> P3 metric (1) evaluated residual under mask-freeze mapping; v2 metric (1) evaluates residual under smooth-switch mapping.


# P6 Gate Evaluation Statistical Table (FULL 600)

- Timestamp: 2026-10-07 23:41:45
- Items Count: 600
- Bootstrap Resamples: 10,000 (Seed: 2026)
- Grader SHA-256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`

## 1. Condition Accuracies

| Condition | Items (N) | Strict Acc (%) | Lenient Acc (%) |
|:---|:---:|:---:|:---:|
| A | 600 | 1.67 | 6.33 |
| B1 | 600 | 1.17 | 5.00 |
| C | 600 | 0.33 | 7.17 |
| E1 | 600 | 6.50 | 10.00 |

## 2. Gate Statistical Metrics (Numbers Only)

| Gate / Comparison | Metric | Delta (C1 - C2) | 95% Bootstrap CI | Discordant (b/c) | McNemar p (1-sided) |
|:---|:---|:---:|:---:|:---:|:---:|
| A - B1 | Strict | +0.0050 | [-0.0050, +0.0150] | 7/4 | 0.2744 |
| A - B1 | Lenient | +0.0133 | [-0.0083, +0.0350] | 26/18 | 0.1456 |
| A - C | Strict | +0.0133 | [+0.0033, +0.0250] | 10/2 | 0.0193 |
| A - C | Lenient | -0.0083 | [-0.0317, +0.0150] | 23/28 | 0.7995 |
| A - C | Strict | +0.0133 | [+0.0033, +0.0250] | 10/2 | 0.0193 |
| A - C | Lenient | -0.0083 | [-0.0317, +0.0150] | 23/28 | 0.7995 |
| A - E1 | Strict | -0.0483 | [-0.0700, -0.0283] | 7/36 | 1.0000 |
| A - E1 | Lenient | -0.0367 | [-0.0617, -0.0117] | 19/41 | 0.9987 |

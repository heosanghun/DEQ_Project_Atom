# P6 Gate Evaluation Statistical Table (FULL 600)

- Timestamp: 2026-10-09 00:03:30
- Items Count: 600
- Bootstrap Resamples: 10,000 (Seed: 2026)
- Grader SHA-256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`

## 1. Condition Accuracies

| Condition | Items (N) | Strict Acc (%) | Lenient Acc (%) |
|:---|:---:|:---:|:---:|
| A | 600 | 1.33 | 4.83 |
| B1 | 600 | 1.17 | 5.00 |
| C | 600 | 0.33 | 7.17 |
| E1 | 600 | 6.50 | 10.00 |

## 2. Gate Statistical Metrics (Numbers Only)

| Gate / Comparison | Metric | Delta (C1 - C2) | 95% Bootstrap CI | Discordant (b/c) | McNemar p (1-sided) |
|:---|:---|:---:|:---:|:---:|:---:|
| A - B1 | Strict | +0.0017 | [-0.0100, +0.0133] | 7/6 | 0.5000 |
| A - B1 | Lenient | -0.0017 | [-0.0233, +0.0200] | 23/24 | 0.6146 |
| A - C | Strict | +0.0100 | [+0.0000, +0.0200] | 8/2 | 0.0547 |
| A - C | Lenient | -0.0233 | [-0.0483, +0.0000] | 20/34 | 0.9799 |
| A - C | Strict | +0.0100 | [+0.0000, +0.0200] | 8/2 | 0.0547 |
| A - C | Lenient | -0.0233 | [-0.0483, +0.0000] | 20/34 | 0.9799 |
| A - E1 | Strict | -0.0517 | [-0.0733, -0.0317] | 5/36 | 1.0000 |
| A - E1 | Lenient | -0.0517 | [-0.0767, -0.0267] | 15/46 | 1.0000 |

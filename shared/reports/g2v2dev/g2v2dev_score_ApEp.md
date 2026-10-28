# P6 Gate Evaluation Statistical Table (FULL 600)

- Timestamp: 2026-10-09 00:03:05
- Items Count: 600
- Bootstrap Resamples: 10,000 (Seed: 2026)
- Grader SHA-256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`

## 1. Condition Accuracies

| Condition | Items (N) | Strict Acc (%) | Lenient Acc (%) |
|:---|:---:|:---:|:---:|
| A | 600 | 7.50 | 10.17 |
| B1 | 600 | 1.17 | 5.00 |
| C | 600 | 0.33 | 7.17 |
| E1 | 600 | 7.83 | 11.50 |

## 2. Gate Statistical Metrics (Numbers Only)

| Gate / Comparison | Metric | Delta (C1 - C2) | 95% Bootstrap CI | Discordant (b/c) | McNemar p (1-sided) |
|:---|:---|:---:|:---:|:---:|:---:|
| A - B1 | Strict | +0.0633 | [+0.0417, +0.0867] | 43/5 | 0.0000 |
| A - B1 | Lenient | +0.0517 | [+0.0267, +0.0783] | 47/16 | 0.0001 |
| A - C | Strict | +0.0717 | [+0.0517, +0.0933] | 44/1 | 0.0000 |
| A - C | Lenient | +0.0300 | [+0.0017, +0.0583] | 48/30 | 0.0268 |
| A - C | Strict | +0.0717 | [+0.0517, +0.0933] | 44/1 | 0.0000 |
| A - C | Lenient | +0.0300 | [+0.0017, +0.0583] | 48/30 | 0.0268 |
| A - E1 | Strict | -0.0033 | [-0.0300, +0.0217] | 30/32 | 0.6482 |
| A - E1 | Lenient | -0.0133 | [-0.0400, +0.0150] | 31/39 | 0.8590 |

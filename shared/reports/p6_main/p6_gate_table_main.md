# P6 Gate Evaluation Statistical Table (FULL 600)

- Timestamp: 2026-10-07 04:42:11
- Items Count: 600
- Bootstrap Resamples: 10,000 (Seed: 2026)
- Grader SHA-256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`

## 1. Condition Accuracies

| Condition | Items (N) | Strict Acc (%) | Lenient Acc (%) |
|:---|:---:|:---:|:---:|
| A | 600 | 1.50 | 5.83 |
| B1 | 600 | 0.83 | 4.50 |
| C | 600 | 0.33 | 6.67 |
| E1 | 600 | 6.00 | 9.33 |

## 2. Gate Statistical Metrics (Numbers Only)

| Gate / Comparison | Metric | Delta (C1 - C2) | 95% Bootstrap CI | Discordant (b/c) | McNemar p (1-sided) |
|:---|:---|:---:|:---:|:---:|:---:|
| A - B1 | Strict | +0.0067 | [-0.0033, +0.0167] | 7/3 | 0.1719 |
| A - B1 | Lenient | +0.0133 | [-0.0067, +0.0333] | 24/16 | 0.1341 |
| A - C | Strict | +0.0117 | [+0.0017, +0.0233] | 9/2 | 0.0327 |
| A - C | Lenient | -0.0083 | [-0.0317, +0.0133] | 21/26 | 0.8092 |
| A - C | Strict | +0.0117 | [+0.0017, +0.0233] | 9/2 | 0.0327 |
| A - C | Lenient | -0.0083 | [-0.0317, +0.0133] | 21/26 | 0.8092 |
| A - E1 | Strict | -0.0450 | [-0.0667, -0.0250] | 6/33 | 1.0000 |
| A - E1 | Lenient | -0.0350 | [-0.0600, -0.0117] | 17/38 | 0.9987 |

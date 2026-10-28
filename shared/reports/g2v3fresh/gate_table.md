# P6 Gate Evaluation Statistical Table (FULL 600)

- Timestamp: 2026-10-11 08:23:16
- Items Count: 1000
- Bootstrap Resamples: 10,000 (Seed: 2026)
- Grader SHA-256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`

## 1. Condition Accuracies

| Condition | Items (N) | Strict Acc (%) | Lenient Acc (%) |
|:---|:---:|:---:|:---:|
| A | 1000 | 6.50 | 9.20 |
| B1 | 1000 | 6.90 | 9.90 |
| C | 1000 | 0.10 | 6.60 |
| E1 | 1000 | 5.30 | 9.30 |

## 2. Gate Statistical Metrics (Numbers Only)

| Gate / Comparison | Metric | Delta (C1 - C2) | 95% Bootstrap CI | Discordant (b/c) | McNemar p (1-sided) |
|:---|:---|:---:|:---:|:---:|:---:|
| A - B1 | Strict | -0.0040 | [-0.0170, +0.0090] | 21/25 | 0.7693 |
| A - B1 | Lenient | -0.0070 | [-0.0220, +0.0070] | 23/30 | 0.8642 |
| A - C | Strict | +0.0640 | [+0.0500, +0.0800] | 64/0 | 0.0000 |
| A - C | Lenient | +0.0260 | [+0.0060, +0.0460] | 63/37 | 0.0060 |
| A - C | Strict | +0.0640 | [+0.0500, +0.0800] | 64/0 | 0.0000 |
| A - C | Lenient | +0.0260 | [+0.0060, +0.0460] | 63/37 | 0.0060 |
| A - E1 | Strict | +0.0120 | [-0.0050, +0.0290] | 42/30 | 0.0973 |
| A - E1 | Lenient | -0.0010 | [-0.0210, +0.0190] | 52/53 | 0.5773 |

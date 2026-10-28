# P6 Gate Evaluation Statistical Table (FULL 600)

- Timestamp: 2026-10-11 16:27:51
- Items Count: 600
- Bootstrap Resamples: 10,000 (Seed: 2026)
- Grader SHA-256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`

## 1. Condition Accuracies

| Condition | Items (N) | Strict Acc (%) | Lenient Acc (%) |
|:---|:---:|:---:|:---:|
| A | 600 | 7.83 | 12.67 |
| B1 | 600 | 1.17 | 5.00 |
| C | 600 | 0.33 | 7.17 |
| E1 | 600 | 4.83 | 10.17 |

## 2. Gate Statistical Metrics (Numbers Only)

| Gate / Comparison | Metric | Delta (C1 - C2) | 95% Bootstrap CI | Discordant (b/c) | McNemar p (1-sided) |
|:---|:---|:---:|:---:|:---:|:---:|
| A - B1 | Strict | +0.0667 | [+0.0433, +0.0900] | 46/6 | 0.0000 |
| A - B1 | Lenient | +0.0767 | [+0.0500, +0.1050] | 61/15 | 0.0000 |
| A - C | Strict | +0.0750 | [+0.0550, +0.0967] | 45/0 | 0.0000 |
| A - C | Lenient | +0.0550 | [+0.0267, +0.0833] | 54/21 | 0.0001 |
| A - C | Strict | +0.0750 | [+0.0550, +0.0967] | 45/0 | 0.0000 |
| A - C | Lenient | +0.0550 | [+0.0267, +0.0833] | 54/21 | 0.0001 |
| A - E1 | Strict | +0.0300 | [+0.0050, +0.0567] | 39/21 | 0.0137 |
| A - E1 | Lenient | +0.0250 | [-0.0017, +0.0533] | 43/28 | 0.0480 |

# P6 Gate Evaluation Statistical Table (FULL 600)

- Timestamp: 2026-10-12 05:48:20
- Items Count: 1758
- Bootstrap Resamples: 10,000 (Seed: 2026)
- Grader SHA-256: `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`

## 1. Condition Accuracies

| Condition | Items (N) | Strict Acc (%) | Lenient Acc (%) |
|:---|:---:|:---:|:---:|
| A | 1758 | 8.08 | 12.91 |
| B1 | 1758 | 7.62 | 12.86 |
| C | 1758 | 0.00 | 8.02 |
| E1 | 1758 | 5.75 | 11.60 |

## 2. Gate Statistical Metrics (Numbers Only)

| Gate / Comparison | Metric | Delta (C1 - C2) | 95% Bootstrap CI | Discordant (b/c) | McNemar p (1-sided) |
|:---|:---|:---:|:---:|:---:|:---:|
| A - B1 | Strict | +0.0046 | [-0.0080, +0.0165] | 64/56 | 0.2615 |
| A - B1 | Lenient | +0.0006 | [-0.0125, +0.0137] | 71/70 | 0.5000 |
| A - C | Strict | +0.0808 | [+0.0683, +0.0933] | 142/0 | 0.0000 |
| A - C | Lenient | +0.0489 | [+0.0319, +0.0660] | 166/80 | 0.0000 |
| A - C | Strict | +0.0808 | [+0.0683, +0.0933] | 142/0 | 0.0000 |
| A - C | Lenient | +0.0489 | [+0.0319, +0.0660] | 166/80 | 0.0000 |
| A - E1 | Strict | +0.0233 | [+0.0091, +0.0370] | 100/59 | 0.0007 |
| A - E1 | Lenient | +0.0131 | [-0.0028, +0.0290] | 109/86 | 0.0575 |

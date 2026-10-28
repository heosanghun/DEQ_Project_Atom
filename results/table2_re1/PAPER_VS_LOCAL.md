# Paper vs Local Comparison &mdash; Table 2

Source: `results/table2_re1/table2_results.json`

Each cell shows **local mean&pm;std (n samples)** in the first row and the
**paper headline** plus the **gap (local &minus; paper)** in subsequent rows.
All values are accuracy in percent; n is the number of (seed x problem)
samples that succeeded.

| Method | MATH-500 | GSM8K | AIME 2024 | ARC-AGI-Text | HumanEval |
|---|---|---|---|---|---|
| Greedy | 40.0&pm;0.0 (n=5) | 30.0&pm;0.0 (n=5) | 0.0&pm;0.0 (n=5) | 80.0&pm;0.0 (n=5) | 0.0&pm;0.0 (n=5) |
| &nbsp;&nbsp;_paper_ | 45.2 | 76.5 | 28.3 | 36.1 | 56.4 |
| &nbsp;&nbsp;_gap_ | -5.2 | -46.5 | -28.3 | +43.9 | -56.4 |
| SC@14 | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| &nbsp;&nbsp;_paper_ | 59.3&pm;0.7 | 84.2&pm;0.5 | 34.8&pm;0.9 | 52.4&pm;0.8 | 65.2&pm;0.6 |
| &nbsp;&nbsp;_gap_ | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| Native Think | 0.0&pm;0.0 (n=5) | 30.0&pm;0.0 (n=5) | 0.0&pm;0.0 (n=5) | 80.0&pm;0.0 (n=5) | 0.0&pm;0.0 (n=5) |
| &nbsp;&nbsp;_paper_ | 57.0&pm;0.6 | 82.4&pm;0.4 | 42.5&pm;0.9 | 50.1&pm;0.7 | 63.3&pm;0.5 |
| &nbsp;&nbsp;_gap_ | -57.0 | -52.4 | -42.5 | +29.9 | -63.3 |
| MCTS (Early Stop) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| &nbsp;&nbsp;_paper_ | 56.5&pm;0.9 | 81.2&pm;0.7 | 38.4&pm;0.8 | 48.1&pm;1.0 | 62.5&pm;0.7 |
| &nbsp;&nbsp;_gap_ | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| CTS-4nu (Ours) | 40.0&pm;0.0 (n=5) | 30.0&pm;0.0 (n=5) | 0.0&pm;0.0 (n=5) | 80.0&pm;0.0 (n=5) | 0.0&pm;0.0 (n=5) |
| &nbsp;&nbsp;_paper_ | 63.8&pm;0.8 | 88.4&pm;0.5 | 50.2&pm;1.1 | 57.8&pm;0.9 | 69.6&pm;0.7 |
| &nbsp;&nbsp;_gap_ | -23.8 | -58.4 | -50.2 | +22.2 | -69.6 |

## Notes

- Paper headlines are taken from Table 2 of the NeurIPS 2026 submission
  (Gemma 4 E4B backbone, 5 seeds, &le; 1e14 MACs, 95% bootstrap CI).
- Local results are produced by `scripts/run_cts_eval_full.py` with the
  paper-aligned `eval_bonferroni_n=12` correction over the primary
  CTS-4nu vs (greedy, native_think, sc_14, mcts_early_stop) x
  (math500, gsm8k, aime) cells.
- Local re-runs in this repo currently use a reduced wall-clock and MAC
  budget (CTS_EVAL_TAU_CAP=1e13, CTS_EVAL_EPISODE_TIMEOUT=180s) so
  absolute accuracy is expected to be lower than the paper headline; the
  *relative ordering* across methods is the headline reproducibility
  signal.

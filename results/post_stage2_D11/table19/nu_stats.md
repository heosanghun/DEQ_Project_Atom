## Table 19 — ν cross-domain statistics (mean ± std)

Per-step ν aggregated across (problem × seed × MCTS step) inside each (method, domain) cell. Markers denote paper-highlighted directional claims (Welch one-sided, Bonferroni-corrected at n=2):

- `↑ nu_expl` &nbsp;⇒&nbsp; nu_expl(AIME) > nu_expl(GSM8K), p_corr < 0.05 (high-entropy AIME drives wider PUCT)
- `↑ nu_act` &nbsp;⇒&nbsp; nu_act(GSM8K) > nu_act(AIME), p_corr < 0.05 (early termination on easy arithmetic)

| method | domain | nu_expl | nu_tol | nu_temp | nu_act | n_steps |
|---|---|---|---|---|---|---:|
| cts_4nu | math | 1.196 ± 0.000 | 0.489 ± 0.000 | 1.178 ± 0.000 | 1.203 ± 0.000 | 186 |

### Welch one-sided p-values (paper directional claims)

| method | nu_expl(AIME)>nu_expl(GSM8K) raw_p / corr_p | nu_act(GSM8K)>nu_act(AIME) raw_p / corr_p |
|---|---|---|
| cts_4nu | 1.0000 / 1.0000 | 1.0000 / 1.0000 |

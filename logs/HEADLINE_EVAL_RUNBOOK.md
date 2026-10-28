# Headline Eval Runbook — Paper Table 2 Alignment (NeurIPS 2026)

**Purpose:** Single operator playbook to close the **local vs paper** accuracy gap
as far as hardware allows, without disrupting the in-flight Stage 2 PPO job.

**Companion docs:**

| Doc | Role |
|---|---|
| [`STAGE2_RESUME_STATE.md`](STAGE2_RESUME_STATE.md) | In-flight training SSOT |
| [`../REVIEWER_FAQ.md`](../REVIEWER_FAQ.md) Q13 | Gap decomposition (4 scaling factors) |
| [`../results/table2/PAPER_VS_LOCAL.md`](../results/table2/PAPER_VS_LOCAL.md) | Paper cross-reference index |
| [`../OPENREVIEW_RESPONSE_PREP.md`](../OPENREVIEW_RESPONSE_PREP.md) | Rebuttal templates R1/R7/R8 |

**Last refreshed:** 2026-05-19 (Stage 2 ~6910/10000 in flight).

---

## 0. Safety rules (read first)

| Rule | Why |
|---|---|
| **Never kill** the detached Stage 2 PID while step &lt; 10000 | ~77 GPU-h lost; meta-policy incomplete |
| **No GPU eval / S1 / S2** on `cuda:0` until in-flight training exits | VRAM contention, step slowdown, OOM |
| Use **`artifacts/stage2_meta_value.pt` only after** log shows `stage2 step=10000/10000` | Stale final ckpt may exist from an older run |
| Always **`--verify-final`** before Wave 1 | Confirms `training_meta.step=10000` |
| ARC / HumanEval absolute numbers **≠ paper** | ARC-AGI-Text proxy + HumanEval footnote — see `LIMITATIONS.md` §13 |

**Read-only progress (safe during training):**

```powershell
cd D:\AI\cts
python scripts/check_stage2_progress.py
```

**Full autopilot (steps 3–15 after Stage 2):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_post_s2_autopilot.ps1
# status: results/post_s2_autopilot/autopilot_status.json
# tail:   Get-Content results\post_s2_autopilot\logs\autopilot.log -Wait -Tail 5
```

---

## 1. Paper headline envelope (what “match the paper” means)

Paper Table 2 (Gemma 4 E4B, 5 seeds, ≤ 1×10¹⁴ MACs per query):

| Knob | Paper | Local default (gap) | Headline profile |
|---|---|---|---|
| `CTS_EVAL_TAU_CAP` | 1×10¹⁴ | 1×10¹³ | **`1e14`** |
| `CTS_EVAL_EPISODE_TIMEOUT` | unbounded | 180 s | **unset** (or `3600` if needed) |
| Eval `--limit` | full split | often 10 | **omit** (full) |
| Stage 2 steps | 10k (§6.2, `paper_parity.yaml`) | in-flight 10k | **complete current run** |
| Stage 2 reward | Eq.(5) answer | converged proxy (this run) | **Phase 4 fork** (optional) |
| Benchmarks (primary) | MATH-500, GSM8K, AIME **2026** | ✅ data fixed | use `aime` → 30 problems |

**Expected ceiling (1× RTX 4090, after Headline profile):**

| Wave | MATH-500 CTS-4ν (est.) | vs paper 64.1% |
|---|---|:---:|
| Proxy S2 + τ=1e13 + limit | ~10–40% | 0.15–0.62 |
| **Wave 2 (Headline env + full primary)** | **48–56%** | **0.75–0.87** |
| + Phase 4 (answer S2 + S1 retrain) | 52–60% | 0.81–0.94 |
| + Cloud A100 full Table 2 | 58–64% | 0.90–1.00 |

---

## 2. Headline Eval Profile (environment)

Apply **only when starting GPU eval** (not during Stage 2 training).

### PowerShell (Windows)

```powershell
$env:CTS_EVAL_TAU_CAP = "1e14"
Remove-Item Env:CTS_EVAL_EPISODE_TIMEOUT -ErrorAction SilentlyContinue
# Optional long cap instead of unbounded:
# $env:CTS_EVAL_EPISODE_TIMEOUT = "3600"

$env:CTS_GLOBAL_SEED = "42"
# Backbone (if not default):
# $env:CTS_GEMMA_MODEL_DIR = "D:\AI\cts\gemma-4-E4B-it"
```

### Bash (cloud)

```bash
export CTS_EVAL_TAU_CAP=1e14
unset CTS_EVAL_EPISODE_TIMEOUT
export CTS_GLOBAL_SEED=42
```

**Verify eval reads the profile (CPU, no GPU):**

```powershell
python scripts/run_cts_eval_full.py --dry-run `
  --config paper_parity `
  --benchmarks math500 `
  --methods cts_4nu `
  --seeds 1
```

Checkpoints for eval are loaded from fixed paths:

- `artifacts/stage1_last.pt`
- `artifacts/stage2_meta_value.pt`

Back up before overwriting; restore via copy when switching ckpt variants.

---

## 3. Phase 0 — While Stage 2 runs (CPU only, ETA ~77 h from step 6910)

| # | Task | Command |
|---:|---|---|
| 0-1 | Poll training | `python scripts/check_stage2_progress.py` |
| 0-2 | Re-download Stage-2 JSONL **with `solution`** (Phase 4 prep) | `python scripts/download_experiment_data.py --skip-math500 --skip-openmath` |
| 0-3 | Inspect Stage-1 ckpt | See §3.1 below |
| 0-4 | (Optional) Cloud account + setup script on box | `bash scripts/cloud_setup.sh` |

### 3.1 Inspect `artifacts/stage1_last.pt`

```powershell
python -c "
import torch
p='artifacts/stage1_last.pt'
ck=torch.load(p,map_location='cpu',weights_only=False)
print('keys:', list(ck.keys())[:12])
meta=ck.get('training_meta') or ck.get('meta') or {}
if isinstance(meta, dict):
    for k in ('step','stage1_lr','paper_faithful','max_steps'):
        if k in meta: print(k, meta[k])
w=[n for n,(t,_) in ck.named_parameters() if 'w_proj' in n][:3]
print('w_proj params sample:', w)
"
```

**Phase 4 needed if:** no `w_proj` in trainable state, or `stage1_lr` ≠ 1e-4, or step ≪ 5000.

---

## 4. Phase 1 — T+0: Stage 2 complete (~30 min, GPU)

### 4.1 Gate checks

```powershell
cd D:\AI\cts

# Must show step=10000/10000, pid may be dead, fresh final ckpt
python scripts/check_stage2_progress.py --verify-final
```

**PASS criteria:**

- Log last line: `stage2 step=10000/10000`
- `verify-final`: status PASS or WARN with `paper_faithful=True`
- `training_meta.step` = 10000 (in verify details)

### 4.2 Backup checkpoints

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path "artifacts/backups" | Out-Null
Copy-Item artifacts/stage2_meta_value.pt "artifacts/backups/stage2_meta_value_10k_proxy_$ts.pt"
Copy-Item artifacts/stage1_last.pt "artifacts/backups/stage1_last_$ts.pt" -ErrorAction SilentlyContinue
Copy-Item artifacts/stage2_meta_value.intermediate.pt "artifacts/backups/stage2_intermediate_$ts.pt" -ErrorAction SilentlyContinue
```

### 4.3 Apply Headline Eval Profile

Run §2 PowerShell block in the **same shell** used for eval.

---

## 5. Wave 1 — OOM / depth smoke (~6–12 GPU-h)

**Goal:** Confirm τ=1e14 runs on this GPU; mean tree depth &gt; 4; no Cultura-style garbage on AIME.

```powershell
python scripts/run_cts_eval_full.py `
  --config paper_parity `
  --benchmarks math500 `
  --methods cts_4nu greedy native_think sc_14 mcts_early_stop `
  --seeds 0 `
  --device cuda:0 `
  --output-dir results/headline_w1_math500_s0
```

**PASS criteria:**

| Check | PASS |
|---|---|
| Run completes without CUDA OOM | ✅ |
| `results/headline_w1_math500_s0/table2_results.json` exists | ✅ |
| CTS-4ν accuracy ≥ greedy on same 500 problems | ✅ (directionality) |
| Peak VRAM stable across problems | ✅ |

**FAIL → OOM:** Use Phase 5 cloud (A100 80GB) for Waves 1–3, or set
`$env:CTS_EVAL_EPISODE_TIMEOUT = "3600"` and retry.

**Compare to paper (partial OK):**

```powershell
python scripts/compare_to_paper_table2.py results/headline_w1_math500_s0 --out results/headline_w1_math500_s0/PAPER_COMPARE.md
```

---

## 6. Wave 2 — Primary family full (highest ROI, ~40–120 GPU-h)

Paper §7.1 Bonferroni primary: **math500 + gsm8k + aime**, 5 methods, 5 seeds, **no `--limit`**.

```powershell
python scripts/run_cts_eval_full.py `
  --config paper_parity `
  --benchmarks math500 gsm8k aime `
  --methods cts_4nu greedy native_think sc_14 mcts_early_stop `
  --seeds 5 `
  --device cuda:0 `
  --output-dir results/headline_w2_primary_full
```

**PASS criteria:**

| Metric | Target |
|---|---|
| CTS-4ν MATH-500 | **≥ 48%** (stretch ≥ 52%) |
| CTS-4ν vs greedy (all 3 benches) | CTS ≥ greedy (mean across seeds) |
| CTS-4ν vs mcts_early_stop | CTS ≥ MCTS-ES (math500) |
| AIME 2026 preds | numeric / `\boxed{}`; no English garbage |

**Post-run:**

```powershell
python scripts/compare_to_paper_table2.py results/headline_w2_primary_full `
  --out results/headline_w2_primary_full/PAPER_COMPARE.md

python scripts/compare_to_paper_table2.py results/headline_w2_primary_full `
  --out results/table2/PAPER_VS_LOCAL_HEADLINE.md
```

Update [`../EXPERIMENTAL_RESULTS.md`](../EXPERIMENTAL_RESULTS.md) §0 with Wave 2 numbers.

---

## 7. Wave 3 — Full Table 2 (12 methods, optional, ~100+ GPU-h)

Only after Wave 2 PASS. Uses post-Stage-2 master pipeline (Table 2 + 17 + 19 + zip audit).

```powershell
python scripts/run_post_stage2_pipeline.py `
  --device cuda:0 `
  --config paper_parity `
  --seeds 5 `
  --output-root results/headline_w3_table2_full
```

**Notes:**

- **No `--table2-limit`** → full splits (MATH-500=500, GSM8K=1319, …).
- Default timeouts: Table 2 = 24 h subprocess cap; extend if killed early:
  `--table2-timeout-s 172800` (48 h).
- Status JSON: `results/headline_w3_table2_full/pipeline_status.json`.

**Smoke first (if unsure):**

```powershell
python scripts/run_post_stage2_pipeline.py `
  --device cuda:0 --table2-limit 50 `
  --output-root results/headline_w3_smoke
```

---

## 8. Phase 4 — Training fork (if Wave 2 MATH-500 &lt; 52%)

Run **sequentially** on GPU (after Wave 2, or instead of Wave 3 if time-limited).

### 8.1 Stage 1 retrain (paper §6.1, ~4 GPU-h)

```powershell
python scripts/run_stage1_openmath.py --config paper_parity --device cuda:0

Copy-Item artifacts/stage1_last.pt artifacts/stage1_paper_parity.pt -Force
```

### 8.2 Answer-mode Stage 2 (paper Eq.(5), ~112 GPU-h @ 90 s/step)

Ensure JSONL has `solution` (Phase 0-2). Override reward in a scratch config or edit
`configs/default.yaml` **for this run only**:

```yaml
stage2_reward_mode: answer
stage2_rollout_decode_tokens: 32
```

Train (fresh run — do **not** use converged-proxy ckpt as resume unless intentional):

```powershell
python scripts/run_stage2_math_ppo.py `
  --config paper_parity --parallel-map `
  --K 64 --collect-batch 64 --ppo-epochs 4 `
  --steps 10000 --log-every 10 --save-every 500 `
  --device cuda:0 --broyden-max-iter 30 `
  --stage1-ckpt artifacts/stage1_paper_parity.pt
```

Backup:

```powershell
Copy-Item artifacts/stage2_meta_value.pt artifacts/stage2_meta_value_10k_answer.pt -Force
```

### 8.3 Re-run Wave 2

Re-apply Headline Eval Profile (§2), then repeat §6.

**PASS uplift:** MATH-500 CTS-4ν **+3–8 pp** vs Wave 2 proxy ckpt.

---

## 9. Phase 5 — Cloud (paper ±5 pp)

Use when Wave 1 OOMs on 4090 or Wave 2+4 still &lt; 58% MATH-500.

### 9.1 Local: upload ckpts

```bash
# Git Bash / WSL from repo root
bash scripts/upload_checkpoints.sh --full root@YOUR_CLOUD_HOST 22
```

Minimum upload: `artifacts/stage2_meta_value.pt`  
Recommended: `--full` (includes 15 GB `stage1_last.pt`).

### 9.2 Cloud box

```bash
git clone <anonymous-or-private-repo-url> cts && cd cts
bash scripts/cloud_setup.sh
# scp checkpoints into artifacts/ (upload script does this)
export CTS_EVAL_TAU_CAP=1e14
unset CTS_EVAL_EPISODE_TIMEOUT
export CTS_GLOBAL_SEED=42

python scripts/run_cts_eval_full.py \
  --config paper_parity \
  --benchmarks math500 gsm8k aime \
  --methods cts_4nu greedy native_think sc_14 mcts_early_stop \
  --seeds 5 --device cuda:0 \
  --output-dir results/headline_cloud_w2
```

Download `results/headline_cloud_w2/` back to local for `compare_to_paper_table2.py`.

---

## 10. Decision tree

```
[Stage 2 in flight] ──poll──▶ step=10000 ?
                                  │
                    NO ──────────▶ Phase 0 only (CPU)
                                  │
                    YES ─────────▶ Phase 1 backup + Headline env
                                  │
                                  ▼
                              Wave 1 (math500, s0)
                                  │
                         OOM? ────┴──── OK
                          │              │
                          ▼              ▼
                     Phase 5       Wave 2 (primary full)
                     cloud              │
                                  MATH≥52%?
                              NO ────┴──── YES
                               │              │
                               ▼              ▼
                          Phase 4         Wave 3 (optional)
                          S1→answer S2         │
                               │              ▼
                               └──▶ re-Wave 2 ──▶ compare + docs
```

---

## 11. Success checklist (rebuttal-ready)

- [ ] `check_stage2_progress.py --verify-final` PASS at step 10000
- [ ] Headline env documented in result JSON / shell history
- [ ] Wave 2 `PAPER_COMPARE.md` generated
- [ ] CTS-4ν ≥ greedy on math500 + gsm8k (primary family)
- [ ] AIME 2026 predictions human-inspect (no garbage tokens)
- [ ] `EXPERIMENTAL_RESULTS.md` §0 updated
- [ ] `OPENREVIEW_RESPONSE_PREP.md` post-S2 placeholder filled
- [ ] (Optional) `run_post_stage2_pipeline.py` pipeline_status.json PASS

---

## 12. Quick reference — all output dirs

| Dir | Wave | Description |
|---|---|---|
| `results/headline_w1_math500_s0` | 1 | OOM / depth smoke |
| `results/headline_w2_primary_full` | 2 | **Main headline attempt** |
| `results/headline_w3_smoke` | 3 | Table 2 limit 50 |
| `results/headline_w3_table2_full` | 3 | Full 12×4×5 |
| `results/headline_cloud_w2` | 5 | Cloud primary family |
| `artifacts/backups/` | — | Timestamped ckpt copies |

---

## 13. What this runbook does *not* claim

- **Absolute Table 2 match** on 1×4090 without Headline env (τ + timeout + full split).
- **Eq.(5) training** for the in-flight May 2026 run (converged proxy — `LIMITATIONS.md` §15).
- **ARC-AGI-Text** or **HumanEval** parity with paper columns (proxy / footnote benchmarks).
- **COCONUT / Recurrent Depth** rows (camera-ready scope).

For reviewer-facing gap narrative, cite **REVIEWER_FAQ Q13** + Wave 2 `PAPER_COMPARE.md`.

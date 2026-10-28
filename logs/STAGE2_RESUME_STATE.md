# Stage 2 PPO — Resume State Note

**This file is the single source of truth for the in-flight Stage 2 PPO
training run.** It survives a PC reboot, a Cursor restart, and any
chat-session reset. After any interruption, read this file first, then
follow the "How to resume after a reboot" section at the bottom.

---

## Snapshot — last confirmed state

| Field | Value |
|---|---|
| Captured at (KST) | **2026-05-19** (autonomous progress poll) |
| Captured by | Cursor session — read-only poll, training untouched |
| Last visible PPO step | **6350 / 10000** (log tail) |
| Last persisted intermediate ckpt | `artifacts/stage2_meta_value.intermediate.pt` — step **6000** |
| Running Python PID at snapshot | **37264** (detached background) |
| Stage 2 log file | `D:\AI\cts\logs\stage2_paper_full_resume_20260517_204944.log` |
| Stage 2 stderr file | `D:\AI\cts\logs\stage2_paper_full_resume_20260517_204944.err` |
| Stage 1 backbone ckpt | `artifacts/stage1_last.pt` (used as `--stage1-ckpt`) |
| Backbone model dir | `D:\AI\cts\gemma-4-E4B-it` (env `CTS_GEMMA_MODEL_DIR`) |
| Global seed | `42` (env `CTS_GLOBAL_SEED`) |

## Training configuration (paper-faithful)

- `--config paper_parity --parallel-map`
- `--K 64 --collect-batch 64 --ppo-epochs 4`
- `--steps 10000 --log-every 10 --save-every 500`
- `--device cuda:0 --broyden-max-iter 30`
- AdamW actor lr = `3e-5`, critic lr = `1e-4`, λ_halt = `0.05`
- All `paper_faithful_p0_4 = True` (NeurIPS audit PASS)

## Estimated remaining cost (rough)

- Remaining steps: poll with `python scripts/check_stage2_progress.py` (read-only).
- **Autopilot (steps 3–15):** `powershell -ExecutionPolicy Bypass -File scripts\start_post_s2_autopilot.ps1`
- Empirical pace on this RTX 4090: ≈ **90 s / step** with K=64 + parallel_map.
- Example (6350/10000 on 2026-05-19): 3650 steps ≈ **~91 h ≈ ~3.8 days**.
- Next disk ckpt: every `save_every=500` (6500, 7000, …, 10000 final).

---

## How to resume after a reboot

> Run these from an **elevated-or-normal PowerShell window opened *outside*
> Cursor** (so the process is not tied to a Cursor session). Or, equivalently,
> let the agent run them with `Start-Process` (which detaches the child).

### 1. Re-confirm the intermediate checkpoint step

```powershell
python -c "import torch; d=torch.load(r'D:\AI\cts\artifacts\stage2_meta_value.intermediate.pt', map_location='cpu', weights_only=False); print(d.get('training_meta'))"
```

Expected output (example): `{'step': 6000, 'total_steps': 10000, ...}`. Whatever
`step` reports here is the actual resume point on disk.

### 2. Confirm no stale Python is holding the GPU

```powershell
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id, ProcessName -AutoSize
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
```

If a python.exe is hogging VRAM from a prior run that did not die cleanly:

```powershell
Stop-Process -Id <pid> -Force
```

### 3. Launch the resume run (detached, with file logging)

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$env:CTS_GEMMA_MODEL_DIR = "D:\AI\cts\gemma-4-E4B-it"
$env:CTS_GLOBAL_SEED = "42"
Start-Process -FilePath "python" `
  -ArgumentList "-u","scripts/run_stage2_math_ppo.py","--config","paper_parity","--parallel-map",`
                "--stage1-ckpt","artifacts/stage1_last.pt",`
                "--resume-from","artifacts/stage2_meta_value.intermediate.pt",`
                "--K","64","--collect-batch","64","--ppo-epochs","4",`
                "--steps","10000","--log-every","10","--save-every","500",`
                "--device","cuda:0","--broyden-max-iter","30" `
  -WorkingDirectory "D:\AI\cts" `
  -WindowStyle Hidden `
  -RedirectStandardOutput "D:\AI\cts\logs\stage2_paper_full_resume_$ts.log" `
  -RedirectStandardError  "D:\AI\cts\logs\stage2_paper_full_resume_$ts.err" `
  -PassThru | Select-Object Id
```

The first two lines of the new log should read (proof of correct resume):

```
[lora] manual LoRA r=8 alpha=16 applied to 108 nn.Linear modules in Gemma4TextModel
[stage2] resumed from artifacts\stage2_meta_value.intermediate.pt at step <N>/10000 (idx=..., collect_batch=64, ppo_epochs=4). AdamW optimiser state was not restored; expect a brief loss-spike while moments re-warm.
```

### 4. Tail the new log to confirm progress

```powershell
$f = Get-ChildItem "D:\AI\cts\logs\stage2_paper_full_resume_*.log" |
     Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content $f.FullName -Wait -Tail 30
```

`stage2 step=<N+10>/10000 ...` must start appearing within ~3 minutes
(Gemma load ≈ 2 min, first PPO batch ≈ 1 min).

---

## Why a reboot loses progress between intermediate saves

The Stage 2 trainer flushes to `stage2_meta_value.intermediate.pt` every
`save_every = 500` steps (overwriting the same file). If the process dies
at step 5750, the ckpt on disk still says step 5500, so resume will redo
those 250 steps. This is the most we can lose per reboot. To tighten it,
relaunch with `--save-every 100` (more disk I/O, less rollback) when
restarting the run.

## Caveats baked into resume

- **AdamW optimiser state is not restored** (the ckpt does not carry it).
  Expect a brief loss-spike for the first ~50–100 resumed steps while
  AdamW re-warms; this is documented in `cts/train/stage2_ppo_train.py`
  next to the `resume_from` kwarg and is also reported on the first log
  line above.
- **RNG state is not restored**. Bit-exact continuation is impossible;
  the run remains paper-faithful in distribution but not in trajectory.
- **Data-iteration order** is rewound to `idx = step * collect_batch %
  len(prompts)` so the model does not silently re-train on the first
  `step` prompts.

---

*This file is regenerated by hand. Update the "Snapshot" block whenever
you (or the agent) restart Stage 2 so the next reboot has a current
anchor.*

# REPORT-0001_fix — AUDIT-0001 FIX 닫기 결과  (EXE → B)

근거 로그: `shared/logs/server_s0_TASK0001_fix.log` (SHA-256: `3bc2d2d916bb3f078cf2c0fdf5fdb11242458c7d9b5676c626d7103280be24cc`)

---

## F1. 시각 대조 및 파이프라인 마지막 행
- **서버 현재 시각**: `2026-09-28 11:16:26 +0900` (`server_s0_TASK0001_fix.log:2`)
- **로그 mtime**: `2026-09-28 11:15:46.187411321 +0900` (`server_s0_TASK0001_fix.log:3`)
- **두 시각 차이**: 40초 (기준 600초 이내)
- **파이프라인 마지막 행**:
  `[109/680] Q#0780 (alpha=0.25) | rho@1000=7.7222e-05 | rho@2000=7.7222e-05 | min=7.7222e-05 | shape=monotonic_decrease | norm=4900.55 | 176.9s` (`server_s0_TASK0001_fix.log:4`)

---

## F2. 커널 OOM 기록
- `journalctl -k --since '2026-09-27 18:00' ...`: 출력 0행
- `dmesg`: `dmesg: read kernel buffer failed: Operation not permitted` (`server_s0_TASK0001_fix.log:7`)
- **OOM 행 수**: 0

---

## F3. 프로세스 수 및 GPU compute app 목록
- `verify_frozen_parity` 프로세스 수: 0 (`server_s0_TASK0001_fix.log:10`)
  - 참고 포인터: `shared/logs/server_health_20260928_A0.log:18` (count=0)
- `run_s7_zstar` / `s6_damping_run` 프로세스:
  - PID 3527236 (`server_s0_TASK0001_fix.log:12`)
  - PID 3527238 (`server_s0_TASK0001_fix.log:13`)
  - PID 3781314 (`server_s0_TASK0001_fix.log:14`)
- GPU compute app 목록:
  - `gpu_uuid, pid, used_gpu_memory [MiB]`
  - `GPU-71be98b0-6b81-9257-c622-97dc97d46480, 3781314, 31422 MiB` (`server_s0_TASK0001_fix.log:17`)
- GPU 상태:
  - GPU 0: `31452 MiB, 99 %` (`server_s0_TASK0001_fix.log:18`)
  - GPU 1~7: 각 `16 MiB, 0 %` (`server_s0_TASK0001_fix.log:19-25`)

---

## F4. tmux 세션 전체 목록 및 GPU 4 상태
- tmux 세션 수: 12개 (`server_s0_TASK0001_fix.log:27-38`)
  1. `cts_speed_profile_gpu1_20260716: 1 windows` (`:27`)
  2. `cts_stage1_repeat1_fixedz0_300_20260716: 1 windows` (`:28`)
  3. `cts_stage1_rootwarm_dynamic100_20260716: 1 windows` (`:29`)
  4. `cts_stage1_rootwarm_dynamic100_v2_20260716: 1 windows` (`:30`)
  5. `cts_stage1_rootwarm_fixedsupport100_20260716: 1 windows` (`:31`)
  6. `equiphase_gpu4_bench: 1 windows` (`:32`)
  7. `equiphase_gpu4_eval: 1 windows` (`:33`)
  8. `equiphase_gpu4_training: 1 windows` (`:34`)
  9. `full6: 1 windows` (`:35`)
  10. `repro: 1 windows` (`:36`)
  11. `s7_watcher: 1 windows` (`:37`)
  12. `s7_zstar_gpu0: 1 windows` (`:38`)
- equiphase 프로세스: PID 712777, PID 1215407 (`server_s0_TASK0001_fix.log:40,41`)
- GPU 4 memory.used: `16 MiB, 0 %` (`server_s0_TASK0001_fix.log:22`)

---

## F5. 호출 행 번호 및 문항 수 관계
- `scripts/run_s7_zstar_pipeline.sh` 내 `s6_damping_run.py` 호출 행:
  - Step 1 (400 items): 행 72 (`server_s0_TASK0001_fix.log:72`)
  - Step 2 (train 680 items): 행 84 (`server_s0_TASK0001_fix.log:84`)
  - Step 3 (heldout 520 items): 행 96 (`server_s0_TASK0001_fix.log:96`)
- `configs/s7_*.json` 문항 수:
  - `configs/s7_items_400.json`: 400 (`server_s0_TASK0001_fix.log:116`)
  - `configs/s7_new_train_680.json`: 680 (`server_s0_TASK0001_fix.log:119`)
  - `configs/s7_new_heldout_520.json`: 520 (`server_s0_TASK0001_fix.log:118`)
  - `configs/s7_new_1200.json`: 1200 (`server_s0_TASK0001_fix.log:117`)
  - `configs/s7_train_1000.json`: 1000 (`server_s0_TASK0001_fix.log:120`)
  - `configs/s7_heldout_600.json`: 600 (`server_s0_TASK0001_fix.log:115`)
- 문항 수 산술 관계:
  - Step 1 (400) + Step 2 (680) + Step 3 (520) = 1,600
  - Step 2 (680) + Step 3 (520) = 1,200 (`s7_new_1200.json`)
  - Step 1 (400) + Step 2 (680) = 1,080

---

## F6. REPORT-0001 §5 대체문 (수치+포인터 전용)
- nvidia-smi PID 3781314 = ps PID 3781314 (cmd: `scripts/s6_damping_run.py`, `server_s0_TASK0001_fix.log:14,17`). compute apps PID 일치수 = 1/1, 미등록 PID = 0.

---

## 관찰 (FIX 아님): s6_damping_run.py의 rho@2000 · max_nfe 관련 행
- `scripts/s6_damping_run.py` grep 행 (`server_s0_TASK0001_fix.log:122-136`):
```text
10:Logs: rho@1000, rho@2000, min_rho, trajectory every 10 steps, and 4-way classification.
131:    max_nfe: int = 2000,
148:    print(f"Device: {device_str} | Max NFE: {max_nfe} | Tol: {tol} | Sigma: {sigma_noise}")
180:        log_f.write(f"Device: {device_str} | Max NFE: {max_nfe} | Tol: {tol} | Sigma: {sigma_noise}\n")
198:                "rho_at_2000": float(existing_rec["rho_2000"]),
217:            max_nfe=max_nfe,
225:        rho_2000 = float(rho_traj[1999]) if len(rho_traj) >= 2000 else float(rho_traj[-1])
233:        log_f.write(f"rho@1000: {rho_1000:.8e} | rho@2000: {rho_2000:.8e} | min_rho: {min_rho:.8e} | reached_5e-3: {converged} | shape: {shape_type}\n")
243:            f"rho@1000={rho_1000:.4e} | rho@2000={rho_2000:.4e} | min={min_rho:.4e} | "
252:            "max_nfe": int(max_nfe),
269:            "rho_2000": rho_2000,
282:            "rho_at_2000": rho_2000,
310:            "max_nfe": max_nfe,
359:    parser.add_argument("--max_nfe", type=int, default=2000, help="Maximum Picard iterations (default 2000)")
382:        max_nfe=args.max_nfe,
```

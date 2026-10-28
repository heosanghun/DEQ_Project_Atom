# AUDIT-0001 FIX 1~5 — server read-only check (kill/attach/edit/delete 0)
cd /home/shoon/cts_e1_s5 || exit 1
L=raw_s7/s7_zstar_pipeline.log
echo "=== F1 Time (server date / log mtime / last line)"
date '+%F %T %z'
stat -c '%y %n' "$L"
tail -n 1 "$L"
echo "=== F2 Kernel OOM (permission denied verbatim included)"
journalctl -k --since '2026-09-27 18:00' --no-pager 2>&1 | grep -iE 'oom|killed process|out of memory|permission|denied|no journal' | tail -n 20
echo "--- dmesg"
dmesg 2>&1 | grep -iE 'oom|killed process|out of memory|not permitted' | tail -n 20
echo "=== F3 Processes"
echo "verify_frozen_parity:"
ps -eo pid,etime,rss,cmd | grep '[v]erify_frozen_parity' || echo "(0건)"
echo "run_s7_zstar / s6_damping_run:"
ps -eo pid,ppid,etime,cmd | grep -E '[r]un_s7_zstar|[s]6_damping_run' || echo "(0건)"
echo "--- GPU compute apps"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
echo "=== F4 tmux sessions and GPU4 equiphase status"
tmux ls
echo "equiphase processes:"
ps -eo pid,etime,cmd | grep '[e]quiphase' || echo "(0건)"
echo "=== F5 run_s7_zstar_pipeline.sh content and config counts"
sed -n '1,80p' scripts/run_s7_zstar_pipeline.sh
ls -la configs/ | grep -iE 's7_'
for f in configs/s7_*.json; do python3 -c 'import json, sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], len(d) if isinstance(d, list) else {k: len(v) for k, v in d.items() if isinstance(v, list)})' "$f"; done
echo "=== Observation: rho@2000 in s6_damping_run.py"
grep -nE 'rho@2000|rho_2000|max_nfe' scripts/s6_damping_run.py | head -n 30

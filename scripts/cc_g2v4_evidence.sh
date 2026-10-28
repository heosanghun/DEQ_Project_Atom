#!/usr/bin/env bash
# scripts/cc_g2v4_evidence.sh - AUDIT-P014 FIX-1/2 raw evidence (CC, 2026-10-11). Read-only except writing this output.
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
echo "=== evidence $(date '+%F %T') ==="
echo "## [A] dev master log"; cat raw_s7/cc_g2v4_dev_m7.log
sha256sum results_s7/p6r/g2v4dev/dev_score.json results_s7/p6r/g2v4dev/dev_score.md results_s7/g2v4m7rep/decoder_Ah.pt results_s7/g2v2/decoder_M7A_v36.pt results_s7/g2v4m7rep/decoder_Eh.pt results_s7/g2v2/decoder_M7E_v36.pt results_s7/g2v4m7rep/deq_Ah.pt results_s7/g2v4m7rep/deq_Eh.pt
echo "## [B] selection files"; sha256sum scripts/g2v4_select_fresh.py scripts/g2v4_select_fresh_v2.py configs/g2v4_fresh*.json configs/g2v4b_*.json configs/g2v4_exclude.json configs/g2v4_server_json_union.json
ls -la --time-style=full-iso configs/g2v4*.json
echo "## [B2] run1 check (re-run, read-only)"; ls -la --time-style=full-iso configs/archive/ | grep run1
echo "-- grep for run1 file names in .sh/.log/.py/.txt under cts_e1_s5 and cts_e1:"; grep -rl 'run1_48b7f09d\|g2v2_select_run1' --include=*.sh --include=*.log --include=*.py --include=*.txt . /home/shoon/cts_e1 2>/dev/null | grep -v g2v4_select_fresh_v2.py | grep -v cc_g2v4_evidence; echo "-- (end grep)"
echo "-- files created 2026-10-08 15:00..16:10 under results_s7 raw_s7 configs scratch:"; find results_s7 raw_s7 configs scratch -newermt '2026-10-08 15:00' ! -newermt '2026-10-08 16:10' -type f 2>/dev/null; echo "-- (end find)"
echo "## [C] id scan"; nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v4_idscan.py" 2>&1 | grep -v Warning
echo "## [D] training"; for t in Ah Eh Az Ez; do echo "-- $t"; grep -h "\[DATA\]\|PROJ-V2\|PARAMS\|LM-LORA" results_s7/g2v4m7rep/train_$t.log; grep -h "Step 2995\|Step 2999" results_s7/g2v4m7rep/train_$t.log; done
sha256sum results_s7/g2v4m7rep/manifest_*.json results_s7/g2v2/zstar_zsrcinj_train1000.pt results_s7/zsrc_train1000_P4.pt configs/s7_train_993.json
for t in Ah Eh Az Ez; do echo "-- manifest $t"; cat results_s7/g2v4m7rep/manifest_$t.json; echo; done
echo "## [E] runtime-imported files"; sha256sum scripts/s7_train_decoder_v36.py scripts/g2v3_m3_full_train.py scripts/check_p6_decoder_version.sh scripts/s7_eval_gate_p6_v36f.py scripts/g2v4_proj_v2.py scripts/g2v3_m3_eval_latents_v2.py scripts/g2v4_fresh_prep.py scripts/g2v2_zstar_run.py cts/deq/zstar_inject_v2.py configs/zstar_canonical.yaml scripts/s7_eval_score_p6_v36.py scripts/s5_eval_grader_v2.py scripts/g2v4_m4_pilot_train.py
echo "## [F] confirm outputs absent?"; ls -d results_s7/g2v4 results_s7/p6r/g2v4fresh results_s7/p6r/g2v4fresh_merged 2>&1; ls results_s7/g2v4 2>&1 | head
echo "=== evidence end $(date '+%F %T') ==="

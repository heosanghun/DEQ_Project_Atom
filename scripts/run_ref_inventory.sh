#!/usr/bin/env bash
set -e

LOG_FILE=/home/shoon/cts_e1_s5/raw_s7/ref_inventory_EXE260928_40.log
echo "=== EXE-260928-40 Task 1: Inventory & Grep Log ===" > "$LOG_FILE"
date -u >> "$LOG_FILE"

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "1. Curated Inventory Table:" | tee -a "$LOG_FILE"
python3 /home/shoon/cts_e1_s5/scripts/inventory_ref_repos.py | tee -a "$LOG_FILE"

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "2. Systematic Grep across deq, mdeq, torchdeq:" | tee -a "$LOG_FILE"

cd /home/shoon/ref

for kw in broyden anderson line_search f_thres b_thres stop_mode lowest power_method jac_loss_estimate weight_norm VariationalHidDropout "backward(" phantom; do
    echo "==================== KEYWORD: $kw ====================" | tee -a "$LOG_FILE"
    grep -rn --include="*.py" "$kw" deq/ mdeq/ torchdeq/ | head -n 30 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "Task 1 Complete." | tee -a "$LOG_FILE"

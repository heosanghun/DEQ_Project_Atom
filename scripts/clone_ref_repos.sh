#!/usr/bin/env bash
set -e

mkdir -p /home/shoon/ref
cd /home/shoon/ref

LOG_FILE=/home/shoon/cts_e1_s5/raw_s7/ref_clone_EXE260928_40.log
echo "=== EXE-260928-40 Task 0: Isolated Clone ===" > "$LOG_FILE"
date -u >> "$LOG_FILE"

for repo in deq mdeq torchdeq; do
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Cloning $repo..." | tee -a "$LOG_FILE"
    if [ ! -d "$repo" ]; then
        git clone "https://github.com/locuslab/${repo}.git" | tee -a "$LOG_FILE"
    else
        echo "$repo already exists." | tee -a "$LOG_FILE"
    fi
    cd "$repo"
    echo "[REPO] $repo" | tee -a "$LOG_FILE"
    echo "Commit HEAD: $(git rev-parse HEAD)" | tee -a "$LOG_FILE"
    echo "LICENSE head:" | tee -a "$LOG_FILE"
    if [ -f "LICENSE" ]; then
        head -n 5 LICENSE | tee -a "$LOG_FILE"
    elif [ -f "LICENSE.txt" ]; then
        head -n 5 LICENSE.txt | tee -a "$LOG_FILE"
    elif [ -f "LICENSE.md" ]; then
        head -n 5 LICENSE.md | tee -a "$LOG_FILE"
    else
        echo "No LICENSE file found." | tee -a "$LOG_FILE"
    fi
    echo "Directory Tree (depth 2):" | tee -a "$LOG_FILE"
    find . -maxdepth 2 -not -path '*/.*' | sort | tee -a "$LOG_FILE"
    cd /home/shoon/ref
done

echo "Task 0 Complete." | tee -a "$LOG_FILE"

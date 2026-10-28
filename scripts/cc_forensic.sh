#!/usr/bin/env bash
# scripts/cc_forensic.sh - seed-2047 forensic on existing checkpoints (CC 2026-10-13, CTO CTS-433). No training, no parameter update. Dev items only.
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== forensic start $(date '+%F %T') ==="; sha256sum scripts/g2v5_forensic.py
CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v5_forensic.py Ah,R1A1s49,R1A1s50,R1A0s47 > /tmp/cts_forensic_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=$U6 $PY scripts/g2v5_forensic.py R1A0s49,R1A0s50,Eh,R1E0s49 > /tmp/cts_forensic_6.log 2>&1 &
CUDA_VISIBLE_DEVICES=$U7 $PY scripts/g2v5_forensic.py R1E0s50,R1E1s47,R1E1s49,R1E1s50 > /tmp/cts_forensic_7.log 2>&1 &
wait
grep -h "\[FOR\]" /tmp/cts_forensic_*.log > /tmp/cts_forensic_all.jsonl; wc -l /tmp/cts_forensic_all.jsonl; grep -h "Error\|Traceback" /tmp/cts_forensic_*.log | head
echo "=== forensic end $(date '+%F %T') ==="

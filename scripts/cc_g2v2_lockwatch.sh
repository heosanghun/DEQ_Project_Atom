#!/usr/bin/env bash
# scripts/cc_g2v2_lockwatch.sh - run lock checks the moment the compute flag appears (CC, 2026-10-08). Read-only.
cd /home/shoon/cts_e1_s5
until [ -f raw_s7/G2V2_COMPUTE_DONE.flag ]; do sleep 20; done
echo "flag seen $(date '+%F %T')" > raw_s7/g2v2_lockcheck_out.txt
nice -n 5 /home/shoon/system1_5/.venv/bin/python scripts/cc_g2v2_lockcheck.py >> raw_s7/g2v2_lockcheck_out.txt 2>&1
echo "lockcheck done $(date '+%F %T')" >> raw_s7/g2v2_lockcheck_out.txt

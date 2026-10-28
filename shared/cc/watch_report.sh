# CC 감시 도구: 지정한 REPORT가 생기거나 새 ESC가 생길 때까지(최대 115분)
S=/d/AI/cts/shared; R=$1; end=$(( $(date +%s) + 115*60 )); M=$(mktemp); touch $M
while [ $(date +%s) -lt $end ]; do
  [ -f $S/reports/$R.md ] && { sleep 20; echo "REPORT_READY $S/reports/$R.md"; exit 0; }
  n=$(find $S/escalate -newer $M -type f -name 'ESC-*' 2>/dev/null | head -1); [ -n "$n" ] && { echo "NEW_ESC $n"; exit 0; }
  sleep 30; done; echo TIMER

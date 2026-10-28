# CC 감시 도구: 파일에 정규식이 나타나거나 새 ESC가 생길 때까지(최대 115분). 사용: watch_grep.sh <파일> <정규식>
S=/d/AI/cts/shared; F=$1; P=$2; end=$(( $(date +%s) + 115*60 )); M=$(mktemp); touch $M
while [ $(date +%s) -lt $end ]; do
  [ -f "$F" ] && grep -Eq "$P" "$F" && { echo "MATCH $(grep -E "$P" "$F" | tail -1)"; exit 0; }
  n=$(find $S/escalate -newer $M -type f -name 'ESC-*' 2>/dev/null | head -1); [ -n "$n" ] && { echo "NEW_ESC $n"; exit 0; }
  sleep 60; done; echo TIMER

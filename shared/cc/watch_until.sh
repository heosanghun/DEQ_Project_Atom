# CC 감시 도구: 새 보고서·ESC 파일·ESC-0005/0006 답변 또는 목표 시각(최대 115분)까지 대기
S=/d/AI/cts/shared; cap=$(( $(date +%s) + 115*60 )); target=${1:+$(date -d "$1" +%s)}; [ -z "$target" -o "$target" -gt "$cap" ] && target=$cap; M=$(mktemp); touch $M
while [ $(date +%s) -lt $target ]; do
  n=$(find $S/reports $S/escalate -newer $M -type f 2>/dev/null | head -1); [ -n "$n" ] && { sleep 30; echo "NEW $n"; exit 0; }
  for e in 5 6; do [ "$(tail -n 1 $S/escalate/ESC-000$e.md)" != "답:" ] && { echo "ESC${e}_ANSWERED"; exit 0; }; done
  sleep 60; done; echo TIMER

cd "d:/dev_slow/cogangrass_detection"
MASTER="C:/Users/Joshua/AppData/Local/Temp/claude/D--dev-slow-cogangrass-detection/0c753bd7-b1ee-46d8-828b-5e27ccf12c48/tasks/b08yhm9qi.output"
sig() {
  d=$(grep -ac "MATRIX COMPLETE" "$MASTER" 2>/dev/null)
  p=$(grep -ac "##########" "$MASTER" 2>/dev/null)
  e=0; for f in log_256_noclahe log_256_clahe log_512_noclahe log_512_clahe; do [ -f "$f.log" ] && e=$((e+$(tr '\r' '\n' < "$f.log" | grep -acE "^epoch"))); done
  echo "${d}|${p}|${e}"
}
prev=$(sig); stale=0
while true; do
  sleep 1200
  cur=$(sig)
  if [ "${cur%%|*}" = "1" ]; then echo "WATCHDOG: MATRIX COMPLETE (sig=$cur)"; exit 0; fi
  if [ "$cur" = "$prev" ]; then stale=$((stale+1)); else stale=0; fi
  prev="$cur"
  if [ "$stale" -ge 2 ]; then echo "WATCHDOG: NO PROGRESS for ~40min (sig=$cur) -> likely HANG/FAILURE"; exit 1; fi
done

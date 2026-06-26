#!/usr/bin/env bash
# Concurrent full cross-collection sweep over the fit-gated backbones (single-GPU).
# Each model runs as its own isolated process (failure-isolated: an OOM/crash in one
# writes its own failed/oom row and never aborts the others). DINOv3 excluded (gated).
# Real rows land in arch_sweep/results/<job_id>.jsonl the instant each finishes.
cd /home/josh/dev/cogangrass_detection
export HF_HOME=/home/josh/hf_cache
PY=.venv/bin/python
LOG_DIR=arch_sweep/results/logs
STATUS="$LOG_DIR/concurrent.out"
mkdir -p "$LOG_DIR"
: > "$STATUS"

one() {  # name  script-args...
  local name="$1"; shift
  local log="$LOG_DIR/${name}.full.log"
  echo "RUN_START ${name}" >> "$STATUS"
  if timeout 14400 $PY arch_sweep/models/"$@" >"$log" 2>&1; then
    line=$(grep "0422 balanced accuracy" "$log" | tail -1)
    echo "RUN_DONE ${name} | ${line:-no-metric-line}" >> "$STATUS"
  else
    echo "RUN_FAIL ${name} (exit/timeout — see $log)" >> "$STATUS"
  fi
}

one resnet18  train_resnet18.py &
one dinov2    train_dinov2.py &
one plantclef train_plantclef.py &
one siglip2   train_siglip2.py &
one aimv2     train_aimv2.py &
one cradio    train_cradio.py &
wait
echo "RUN_ALL_DONE" >> "$STATUS"

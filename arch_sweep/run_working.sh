#!/usr/bin/env bash
# Stopgap full cross-collection sweep over all 10 R2 backbones (the U11 orchestrator is a
# later phase). Runs each model script on the FULL 0606->0422 protocol, continue-on-failure,
# writing real result rows to arch_sweep/results/. DINOv3 needs the HF token (run
# `hf auth login` once; weights are license-gated). Re-run is safe: cached features make
# repeats fast.
cd /home/josh/dev/cogangrass_detection
export HF_HOME=/home/josh/hf_cache
PY=.venv/bin/python
LOG_DIR=arch_sweep/results/logs
mkdir -p "$LOG_DIR"

run() {  # name  script-args...
  local name="$1"; shift
  local log="$LOG_DIR/${name}.full.log"
  echo "RUN_START ${name}"
  if timeout 7200 $PY arch_sweep/models/"$@" >"$log" 2>&1; then
    line=$(grep "0422 balanced accuracy" "$log" | tail -1)
    echo "RUN_DONE ${name} | ${line:-no-metric-line}"
  else
    echo "RUN_FAIL ${name} (exit/timeout — see $log)"
  fi
}

run resnet18   train_resnet18.py
run dinov2     train_dinov2.py
run plantclef  train_plantclef.py
run siglip2    train_siglip2.py
run aimv2      train_aimv2.py
run cradio     train_cradio.py
run dinov3_s   train_dinov3.py --size s
run dinov3_b   train_dinov3.py --size b
run dinov3_l   train_dinov3.py --size l
run dinov3_sat train_dinov3.py --size sat
echo "RUN_ALL_DONE"

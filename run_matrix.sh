#!/usr/bin/env bash
# 2x2 experiment matrix: {256, 512} tile size x {no-CLAHE, CLAHE}.
# Each trained + tested on the held-out 0422 collection (cross-collection).
# CLAHE precompute runs on CPU in parallel with the no-CLAHE GPU training.
cd "d:/dev_slow/cogangrass_detection"

for SIZE in 256 512; do
  echo "########## PHASE ${SIZE}px : TILING ##########"
  TILE_PX=$SIZE TILE_SAVE_PX=$SIZE python boxes_to_tiles.py

  echo "########## PHASE ${SIZE}px : train no-CLAHE (GPU)  ||  precompute CLAHE (CPU) ##########"
  python -u train_tiles_da.py tiles_dataset > "log_${SIZE}_noclahe.log" 2>&1 &
  TRAIN_PID=$!
  rm -rf tiles_dataset_clahe
  python precompute_clahe.py          # CPU, runs while the GPU trains the no-CLAHE model
  wait $TRAIN_PID                      # let the no-CLAHE training finish

  echo "########## PHASE ${SIZE}px : train CLAHE (GPU) ##########"
  python -u train_tiles_da.py tiles_dataset_clahe > "log_${SIZE}_clahe.log" 2>&1

  echo "########## PHASE ${SIZE}px DONE ##########"
done
echo "########## MATRIX COMPLETE ##########"

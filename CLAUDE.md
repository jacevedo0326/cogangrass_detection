# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Detect **cogongrass** (an invasive grass) in oblique **drone imagery**. Two
independent modeling tracks share the same drone frames:

1. **Tile classifier** (primary) — grid each frame into tiles, classify each tile
   `cogongrass` / `not_cogongrass`, render a coverage heatmap. Pipeline:
   `prep_images.py` → `boxes_to_tiles.py` → `train_tiles.py` → `heatmap_infer.py`.
2. **YOLO patch detector** — single-class box detector on whole frames. Pipeline:
   `prep_dataset.py` → `train_yolo.py` → `test_yolo.py`.

There is no package, test suite, or build step — these are standalone scripts run
directly with `python <script>.py`. Most read/write fixed directories relative to
the repo root rather than taking path arguments. See `TILE_DATASET_USAGE.md` for
the canonical end-to-end tile walkthrough.

## Environment & hardware constraints

- Windows + PowerShell. Every training script guards `main()` with
  `if __name__ == "__main__":` — **required** because the torch/ultralytics
  DataLoaders use multiprocessing on Windows. Keep this guard on anything new.
- Tuned for a single **RTX 2060 (6 GB)** — batch sizes, AMP autocast, and
  `cache=True` choices assume this. Drop batch if you hit OOM.
- Dependencies are imported ad hoc (no requirements file): `torch torchvision
  ultralytics pillow numpy opencv-python scikit-learn matplotlib fpdf2`.

## Critical conventions (violating these silently corrupts results)

- **Split by FRAME, not by tile.** Many tiles share one source frame; letting a
  frame's tiles span train/test leaks and inflates the score. Tile filenames
  encode the frame as `<framestem>_r<row>_c<col>.jpg`; group on the part before
  `_r#_c#` (`frame_of()` / `grouped_split()` in `train_tiles.py`). `leakage_check.py`
  red-teams a split for temporal-adjacency and near-duplicate leakage.
- **The script defaults are the OLD values.** `boxes_to_tiles.py` defaults to
  `TILE_PX=160` / `TILE_SAVE_PX=224` and `prep_images.py` to `PREP_MAX=1280` —
  these downscale. The current full-res pipeline **requires env overrides**:
  ```powershell
  $env:PREP_MAX="4096"; python prep_images.py <raw_folder>
  $env:TILE_PX="512"; $env:TILE_SAVE_PX="512"; python boxes_to_tiles.py
  ```
- **Report balanced accuracy + per-class recall, never raw accuracy** — the tile
  data is ~88% positive, so raw accuracy is misleading. False negatives (missed
  cogongrass) are costlier than false positives; `threshold_sweep.py` lowers the
  decision threshold below 0.5 and optimizes F2 accordingly.
- **Labels are derived from YOLO boxes.** `boxes_to_tiles.py` marks a tile
  `cogongrass` when ≥30% of its area falls inside a box; a frame with no `.txt` is
  treated as **all-negative**. An ExG (green-ness) filter drops sky / bare-ground
  tiles before labeling.
- **New experiments save to their own `tile_classifier_*.pt` and `*.log`** and must
  not touch `tile_classifier.pt` / `train_tiles.py`. This isolation is intentional
  so the baseline stays reproducible — preserve it.

## The cross-collection domain-generalization protocol

The real problem is generalizing to a *new flight it has never seen*. The DA-track
scripts all train on the **2026-06-06** collection and test on the **entirely
held-out 2026-04-22** collection (NOT a random tile split). When editing or adding
to these, keep that exact protocol so numbers stay comparable:

- `train_tiles_collection.py` — establishes the honest cross-collection baseline.
- `train_tiles_da.py` — adds CLAHE normalization + heavy domain-randomization aug +
  dropout head / label smoothing / weight decay. Produces the deployable models.
- `train_tiles_dino*.py` — DINOv2 backbone experiments (`_spatial` uses dense patch
  tokens + a conv head **with BatchNorm**, kept so AdaBN still applies).
- `tta_eval.py` — test-time adaptation (AdaBN, TENT) on the target collection with
  no labels; the realistic deployment setting.
- **AdaBN at inference** (`heatmap_infer.py`) recomputes BatchNorm running stats on
  the target frames' own tiles before predicting (~+2 pts). Any new backbone meant
  for deployment should retain BatchNorm so this works.

`precompute_clahe.py` writes `tiles_dataset/` → `tiles_dataset_clahe/` once (CLAHE
is deterministic, so doing it per-epoch in the dataloader is wasted CPU); DA scripts
take the dataset dir as `argv[1]`. `run_matrix.sh` runs the {256,512}×{CLAHE,no}
sweep; `watchdog.sh` polls a training log and exits on completion or stall.

## YOLO detector specifics

`prep_dataset.py` reads the **original** labels under `data/` (which still carry the
per-site `15`/`18` classes), merges everything to single class `0 cogongrass`, and
writes a stratified **70/15/15 split by original site** so both fields appear in
every split — plus `train/val/test.txt` and `data.yaml` for ultralytics. It reads
from `data/`, not `drone_dataset/labels`, so it is idempotent regardless of that
folder's state. Transfer-learns from COCO-pretrained `yolov8s.pt`.

## Data layout (all git-ignored, regenerable)

- `data/` — raw drone collections (original multi-class labels live here).
- `drone_dataset/{images,labels}/` — normalized frames + single-class YOLO labels
  (built by `prep_images.py` / `prep_dataset.py`).
- `tiles_dataset/{cogongrass,not_cogongrass}/` — torchvision `ImageFolder`; the
  folder name **is** the label. `tiles_dataset_clahe/` mirrors it, CLAHE-applied.
- `tile_labels/<frame>.json` — per-frame tile labels, editable in the
  `label_tiles.py` GUI.
- `runs/` — ultralytics + heatmap outputs. `*.pt` model weights and `*.log` are
  also ignored; everything here regenerates from the scripts.

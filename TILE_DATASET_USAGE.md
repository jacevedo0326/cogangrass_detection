# Cogongrass Tile Dataset — How to Use It

This explains how to turn the **original drone frames** into the **tile dataset**
the classifier trains on, and how to load it. The tile dataset is a folder of
small image crops sorted into two class folders — the folder name *is* the label.

---

## What you need

- **Original drone frames** (`.jpg`, e.g. 4096×3072 DJI frames).
- **Their YOLO label files** (`.txt`, one per frame, normalized `class cx cy w h`
  boxes around cogongrass patches). **The labels come entirely from these boxes** —
  a frame with no `.txt` is treated as all-negative (no cogongrass).
- Python with: `torch torchvision pillow numpy opencv-python scikit-learn`

> If you only have images **without** labels, the pipeline still runs but marks
> every tile `not_cogongrass`. You'd then label frames by hand in `label_tiles.py`.

---

## The pipeline (two steps)

Raw frames are turned into tiles by **two scripts in sequence**:

```
<raw frames + .txt labels>
        │  prep_images.py      ← normalize + collect into one place
        ▼
drone_dataset/images/  +  drone_dataset/labels/
        │  boxes_to_tiles.py   ← cut into tiles, label each by box coverage
        ▼
tiles_dataset/cogongrass/  +  tiles_dataset/not_cogongrass/
```

### Step 1 — `prep_images.py`: prep the frames

```powershell
$env:PREP_MAX="4096"; python prep_images.py <raw_folder> [more_folders ...]
```

- Copies frames into `drone_dataset/images/` and their YOLO labels into
  `drone_dataset/labels/`.
- Frames with no `.txt` get an empty label → they become all-negative frames.
- **`PREP_MAX=4096` keeps full resolution** (our adopted setting). Without it the
  script defaults to downscaling the long side to 1280 — don't do that for the
  current pipeline.
- Appends to the existing dataset, so you can add new fields incrementally.

### Step 2 — `boxes_to_tiles.py`: cut the tiles

```powershell
$env:TILE_PX="512"; $env:TILE_SAVE_PX="512"; python boxes_to_tiles.py
```

- Grids each frame into `TILE_PX` tiles (**512 px** = our adopted setting; ~1.7 m
  of ground at ~30 ft).
- A tile is labeled **`cogongrass`** if **≥30%** of its area falls inside a box,
  else **`not_cogongrass`**.
- Sky / non-vegetation tiles are dropped automatically (green-ness / ExG filter).
- Saves each tile at `TILE_SAVE_PX` px on disk (resize to the model's 224 happens
  later, during training).
- **Wipes and rebuilds `tiles_dataset/` every run**, so the output is always clean.
- Also writes `tile_labels/<frame>.json` so any frame can be reviewed/corrected in
  the labeling GUI.

> ⚠️ The defaults baked into both scripts are the **old** values (1280 / 160 px).
> The `$env:` overrides above are what produce the current full-res / 512 px dataset.

---

## Output: what the tile dataset looks like

```
tiles_dataset/
├── cogongrass/                     ← label = "cogongrass"
│   ├── DJI_..._0042_r3_c5.jpg
│   └── ...
└── not_cogongrass/                 ← label = "not_cogongrass"
    ├── DJI_..._0042_r0_c1.jpg
    └── ...
```

- **The folder name is the label** — no separate label file needed.
- **The filename encodes the source frame:** `<frame>_r<row>_c<col>.jpg`.

---

## Loading the dataset

It's in standard torchvision `ImageFolder` layout:

```python
from torchvision import datasets
ds = datasets.ImageFolder("tiles_dataset")
print(ds.classes)   # ['cogongrass', 'not_cogongrass']  (0 and 1, alphabetical)
```

**Important — split by frame, not by tile.** Many tiles share a source frame; if
some land in train and others in test you get leakage and an inflated score. Group
by the `<frame>` part of the filename (everything before `_r#_c#`) and keep one
frame's tiles entirely in one split. See `frame_of()` / `grouped_split()` in
`train_tiles.py`.

---

## Training (optional)

```powershell
python train_tiles.py
```

Transfer-learns a ResNet18, splits by frame, balances the classes, and reports
**balanced accuracy** + per-class recall on a held-out test set. Saves
`tile_classifier.pt`.

---

## TL;DR

```powershell
# 1. frames + labels  ->  drone_dataset/
$env:PREP_MAX="4096"; python prep_images.py <raw_folder>

# 2. drone_dataset/   ->  tiles_dataset/{cogongrass,not_cogongrass}
$env:TILE_PX="512"; $env:TILE_SAVE_PX="512"; python boxes_to_tiles.py

# 3. (optional) train
python train_tiles.py
```

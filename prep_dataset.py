"""Prepare the drone YOLO dataset: stratified train/val/test split, single class.

- Reads ORIGINAL labels from data/ (which still carry the 15/18 site classes),
  so this is idempotent no matter what state drone_dataset/labels is in.
- Writes merged single-class labels (everything -> class 0 'cogongrass').
- Stratified 70/15/15 split BY ORIGINAL SITE so both fields appear in every split.
- Emits train.txt / val.txt / test.txt / data.yaml for ultralytics.
"""
from pathlib import Path
import random

ROOT = Path(__file__).parent
DS = ROOT / "drone_dataset"
IMG, LBL = DS / "images", DS / "labels"
ORIG = ROOT / "data"                      # untouched originals with classes 15/18
VAL_FRACTION, TEST_FRACTION, SEED = 0.15, 0.15, 42
random.seed(SEED)

orig_lbl = {p.stem: p for p in ORIG.rglob("*.txt")}   # stem -> original label path
LBL.mkdir(exist_ok=True)

site_of = {}
for img in IMG.glob("*.jpg"):
    st = img.stem
    src = orig_lbl.get(st)
    if src is None:
        continue
    rows = [ln.split() for ln in src.read_text().split("\n") if ln.strip()]
    site_of[st] = int(rows[0][0])         # original site/class (all boxes share it)
    # write merged-class label: class id -> 0, keep the box geometry
    (LBL / f"{st}.txt").write_text("\n".join("0 " + " ".join(b[1:5]) for b in rows) + "\n")

# stratified 70/15/15 split, per site
by_site = {}
for st, c in site_of.items():
    by_site.setdefault(c, []).append(st)
train, val, test = [], [], []
for c, stems in by_site.items():
    random.shuffle(stems)
    n = len(stems); nt = int(n * TEST_FRACTION); nv = int(n * VAL_FRACTION)
    test += stems[:nt]; val += stems[nt:nt + nv]; train += stems[nt + nv:]

def write_list(name, stems):
    (DS / name).write_text("\n".join(f"./images/{s}.jpg" for s in stems) + "\n")
write_list("train.txt", train); write_list("val.txt", val); write_list("test.txt", test)

(DS / "data.yaml").write_text(
    f"path: {DS.as_posix()}\n"
    "train: train.txt\n"
    "val: val.txt\n"
    "test: test.txt\n"
    "names:\n"
    "  0: cogongrass\n"
)

print(f"images with labels: {len(site_of)}")
print(f"split -> train {len(train)} | val {len(val)} | test {len(test)}")
for split, name in [(train, "train"), (val, "val"), (test, "test")]:
    print(f"  {name:5s} by original site:", {c: sum(1 for s in split if site_of[s] == c) for c in sorted(by_site)})
print(f"wrote {DS/'data.yaml'}")

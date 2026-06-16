"""Red-team the 96.7%: check for train/test leakage from overlapping drone frames.

Reproduces train_tiles.py's exact frame split, then measures two leakage signals:
  1. Temporal/sequence adjacency: does each TEST frame have a sequence-neighbor
     (adjacent DJI frame index, seconds apart -> overlapping ground) in TRAIN?
  2. Visual near-duplicates: aHash Hamming distance from each TEST frame to its
     closest TRAIN frame (small = near-identical ground in both splits).
"""
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets

import train_tiles as T   # reuse SEED, DATA, grouped_split, frame_of (import is guarded)

random.seed(T.SEED)
base = datasets.ImageFolder(T.DATA)
cog_idx = base.classes.index("cogongrass")
tr, va, te, _ = T.grouped_split(base.samples, cog_idx)

frame = lambda i: T.frame_of(base.samples[i][0])
tr_frames = sorted(set(frame(i) for i in tr))
te_frames = sorted(set(frame(i) for i in te))
print(f"train frames {len(tr_frames)} | test frames {len(te_frames)}")

# ---- 1. sequence adjacency (DJI_YYYYMMDDHHMMSS_NNNN_D) ----
def parse(stem):
    m = re.match(r"DJI_(\d{14})_(\d+)_", stem)
    return (m.group(1), int(m.group(2))) if m else None

tr_idx_by_day = {}
for f in tr_frames:
    p = parse(f)
    if p:
        tr_idx_by_day.setdefault(p[0], set()).add(p[1])

adj = nochk = 0
for f in te_frames:
    p = parse(f)
    if not p:
        nochk += 1; continue
    day, n = p
    neigh = tr_idx_by_day.get(day, set())
    if any((n + d) in neigh for d in (-2, -1, 1, 2)):
        adj += 1
djichk = len(te_frames) - nochk
print(f"\n[1] sequence adjacency:")
print(f"  test frames with an adjacent (±1/±2 index) TRAIN frame: {adj}/{djichk} "
      f"({100*adj/max(djichk,1):.0f}%)  <- overlapping ground in both splits")

# ---- 2. visual near-duplicates (aHash) ----
def ahash(stem):
    p = Path(T.DATA) / "images" / f"{stem}.jpg"
    # tiles_dataset has no full frames; use the prepared drone_dataset frames
    p = Path("drone_dataset/images") / f"{stem}.jpg"
    a = np.asarray(Image.open(p).convert("L").resize((8, 8)), dtype=np.float32)
    return np.packbits((a > a.mean()).flatten().astype(np.uint8)).view(np.uint64)[0]

def popcount(x):
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)

tr_h = np.array([ahash(f) for f in tr_frames], dtype=np.uint64)
te_h = np.array([ahash(f) for f in te_frames], dtype=np.uint64)
mins = np.array([popcount(np.bitwise_xor(tr_h, h)).min() for h in te_h])
print(f"\n[2] visual nearest-TRAIN aHash distance for each TEST frame (0=identical):")
print(f"  median {np.median(mins):.0f} | <=2 (near-identical): {(mins<=2).sum()} "
      f"| <=5: {(mins<=5).sum()} | <=10: {(mins<=10).sum()} of {len(mins)}")
print(f"  (baseline: random unrelated frames usually differ by ~12-20 bits)")

"""Rank the 440 NEGATIVE (empty-label) frames by how much cogongrass the
classifier sees in them -> finds frames likely MISLABELED (unlabeled cogongrass).

Same tiling/sky-filter/norm as fp_audit.py; writes a full ranked CSV instead of
only the top 10, so the whole suspect list is inspectable.

Run:  python suspect_negatives.py
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import tile_common

# Full-res pipeline config (CLAUDE.md): tile at 512 on PREP_MAX=4096 frames; the resnet18
# tile_classifier still takes a 224 crop. Repointed from the legacy 160/1280 (arch_sweep U3).
TILE, MAX, CNN, VEG, COG = 512, 4096, 224, 0.03, 0.5
SUSPECT = 25.0   # frames with >= this % cogongrass tiles are flagged as likely mislabeled
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def main():
    ckpt = torch.load("tile_classifier.pt", map_location=device)
    classes = ckpt["classes"]; cog = classes.index("cogongrass")
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ckpt["state_dict"]); m.eval().to(device)
    print("note: ceil tiling (edge tiles now scored) — outputs are a new regime vs pre-refactor audits")

    neg = [p.stem for p in Path("drone_dataset/labels").glob("*.txt") if not p.read_text().strip()]
    results = []
    for st in neg:
        ip = Path("drone_dataset/images") / f"{st}.jpg"
        if not ip.exists():
            continue
        im = Image.open(ip).convert("RGB"); W, H = im.size
        if max(W, H) > MAX:
            s = MAX / max(W, H); im = im.resize((round(W * s), round(H * s))); W, H = im.size
        exg = tile_common.exg_map(im)
        tiles = []
        for r, c, box in tile_common.tile_boxes(W, H, TILE):   # ceil grid + clamped edges (R5)
            if not tile_common.tile_is_veg(exg, box, VEG):
                continue
            tiles.append(norm(tile_common.cut_tile(im, box, CNN)))
        if not tiles:
            continue
        with torch.no_grad():
            p = m(torch.stack(tiles).to(device)).softmax(1)[:, cog].cpu().numpy()
        fp = int((p >= COG).sum())
        results.append((st, fp, len(p), 100 * fp / len(p)))

    results.sort(key=lambda x: -x[3])
    with open("suspect_negatives.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "cog_tiles", "veg_tiles", "pct_cogongrass"])
        for st, fp, n, pct in results:
            w.writerow([st, fp, n, f"{pct:.1f}"])

    covs = np.array([r[3] for r in results])
    suspects = [r for r in results if r[3] >= SUSPECT]
    print(f"negative frames audited: {len(results)}  (wrote suspect_negatives.csv)")
    print("per-frame distribution:")
    for lo, hi in [(0, 5), (5, 25), (25, 50), (50, 75), (75, 101)]:
        print(f"  {lo:>3}-{hi - 1 if hi <= 100 else 100:>3}% cogongrass : "
              f"{int(((covs >= lo) & (covs < hi)).sum()):>4} frames")
    print(f"\nLIKELY MISLABELED (>= {SUSPECT:.0f}% cogongrass tiles): {len(suspects)} frames")
    for st, fp, n, pct in suspects:
        print(f"  {pct:5.0f}%  ({fp:>3}/{n:>3} tiles)  {st}")


if __name__ == "__main__":
    main()

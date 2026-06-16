"""Audit false positives: run the tile classifier over every NEGATIVE frame
(no cogongrass labeled) and measure how often it wrongly flags cogongrass.

Note: most negatives were seen in training, so this is an OPTIMISTIC view; the
clean held-out false-positive rate is train_tiles.py's test not_cogongrass recall.
This audit is to find the distribution + worst offenders (which often turn out to
contain UNlabeled cogongrass).

Run:  python fp_audit.py
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

TILE, MAX, CNN, VEG, COG = 160, 1280, 224, 0.03, 0.5
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def main():
    ckpt = torch.load("tile_classifier.pt", map_location=device)
    classes = ckpt["classes"]; cog = classes.index("cogongrass")
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ckpt["state_dict"]); m.eval().to(device)

    neg = [p.stem for p in Path("drone_dataset/labels").glob("*.txt") if not p.read_text().strip()]
    results, all_fp, all_tiles = [], 0, 0
    for st in neg:
        ip = Path("drone_dataset/images") / f"{st}.jpg"
        if not ip.exists():
            continue
        im = Image.open(ip).convert("RGB"); W, H = im.size
        if max(W, H) > MAX:
            s = MAX / max(W, H); im = im.resize((round(W * s), round(H * s))); W, H = im.size
        arr = np.asarray(im).astype(np.float32); ss = arr.sum(2) + 1e-6
        exg = 2 * arr[..., 1] / ss - arr[..., 0] / ss - arr[..., 2] / ss
        cols, rows = W // TILE, H // TILE
        tiles = []
        for r in range(rows):
            for c in range(cols):
                y0, x0 = r * TILE, c * TILE
                if exg[y0:y0 + TILE, x0:x0 + TILE].mean() < VEG:
                    continue
                tiles.append(norm(im.crop((x0, y0, x0 + TILE, y0 + TILE)).resize((CNN, CNN))))
        if not tiles:
            continue
        with torch.no_grad():
            p = m(torch.stack(tiles).to(device)).softmax(1)[:, cog].cpu().numpy()
        fp = int((p >= COG).sum())
        results.append((st, fp, len(p), 100 * fp / len(p)))
        all_fp += fp; all_tiles += len(p)

    results.sort(key=lambda x: -x[3])
    covs = np.array([r[3] for r in results])
    print(f"negative frames audited: {len(results)}")
    print(f"overall false-positive tile rate: {100 * all_fp / all_tiles:.1f}%  "
          f"({all_fp}/{all_tiles} vegetation tiles flagged cogongrass)")
    print("per-frame distribution:")
    for lo, hi in [(0, 5), (5, 25), (25, 50), (50, 75), (75, 101)]:
        print(f"  {lo:>3}-{hi - 1 if hi <= 100 else 100:>3}% cogongrass : {int(((covs >= lo) & (covs < hi)).sum()):>4} frames")
    print("worst 10 offenders (likely false positives OR unlabeled cogongrass):")
    for st, fp, n, pct in results[:10]:
        print(f"  {st}: {pct:.0f}%  ({fp}/{n} tiles)")


if __name__ == "__main__":
    main()

"""Visualize the highest-confidence ISOLATED false positives.

Isolated FP = model says cogongrass (high prob), label says not-cogongrass, AND
no labeled cogongrass in any of the 8 neighboring tiles. If these look like
cogongrass -> label gaps (true precision higher). If clearly other grass/ground
-> real false alarms. Uses 512 DA model + AdaBN on held-out 0422.
"""
import re
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

import tile_common

DATA, MODEL, IMG = "tiles_dataset", "tile_classifier_da_noclahe.pt", 224
HELDOUT = tile_common.HELDOUT_DATES
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_tf = transforms.Compose([transforms.Resize(IMG), transforms.CenterCrop(IMG),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
_RC = re.compile(r".*_r(\d+)_c(\d+)$")
def parse(p):
    """(frame, row, col) — frame identity via the shared contract (R1)."""
    m = _RC.match(Path(p).stem)
    return tile_common.frame_of(p), int(m.group(1)), int(m.group(2))


def main():
    base = datasets.ImageFolder(DATA, transform=eval_tf)
    classes = base.classes; cog = classes.index("cogongrass")
    te = [i for i, (p, _) in enumerate(base.samples)
          if tile_common.date_of(parse(p)[0]) in HELDOUT]
    loader = DataLoader(Subset(base, te), 64, shuffle=False, num_workers=4)
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, len(classes)))
    m.load_state_dict(torch.load(MODEL, map_location=device)["state_dict"]); m = m.to(device)
    tile_common.adapt_bn(m, loader, device=device, verbose=False)   # AdaBN (shared impl)
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs += m(x.to(device)).softmax(1)[:, cog].cpu().tolist()

    truecog = defaultdict(set)
    for i in te:
        f, r, c = parse(base.samples[i][0])
        if base.samples[i][1] == cog:
            truecog[f].add((int(r), int(c)))
    nbrs = lambda r, c: [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc)]

    iso = []
    for k, i in enumerate(te):
        f, r, c = parse(base.samples[i][0]); r, c = int(r), int(c)
        if base.samples[i][1] != cog and probs[k] >= 0.5:           # FP
            if not any(n in truecog[f] for n in nbrs(r, c)):        # isolated
                iso.append((probs[k], base.samples[i][0]))
    iso.sort(reverse=True)
    print(f"isolated FPs (prob>=0.5): {len(iso)}; showing top 20 by confidence")

    plt.figure(figsize=(16, 10))
    for j, (pr, path) in enumerate(iso[:20]):
        plt.subplot(4, 5, j + 1)
        plt.imshow(Image.open(path).convert("RGB"))
        plt.title(f"p={pr:.2f}\n{Path(path).stem[-12:]}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    out = Path("runs/isolated_fp.png"); out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

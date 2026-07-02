"""Cogongrass coverage heatmap inference, with AdaBN test-time adaptation.

Matches the trained DA pipeline: full-resolution frames, 512px tiles resized to
224, ExG sky filter. Before predicting, AdaBN recomputes BatchNorm statistics on
the TARGET frames' own tiles (no labels) -> cancels cross-collection covariate
shift (+~2 pts in our held-out tests). Adapts over ALL frames you pass (the new
field), then predicts each.

Run:  python heatmap_infer.py <image_or_folder> [more ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import tile_common


def _plt():
    """Lazy matplotlib (Agg): tiling/AdaBN helpers stay importable without it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

MODEL = "tile_classifier_da_noclahe.pt"   # best deployable 512 DA model (CLAHE optional, ~tied)
TILE, MAX, CNN, VEG, COG_THRESH = 512, 4096, 224, 0.03, 0.5
ADABN = True
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
OUT = Path("runs/heatmap")
device = "cuda" if torch.cuda.is_available() else "cpu"
to_norm = transforms.Compose([transforms.Resize(CNN), transforms.CenterCrop(CNN),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_model():
    ckpt = torch.load(MODEL, map_location=device)
    classes = ckpt["classes"]
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, len(classes)))  # DA head
    m.load_state_dict(ckpt["state_dict"])
    return m.eval().to(device), classes.index("cogongrass")


def tiles_of(path):
    """Return (tensor_batch, coords, rows, cols, pil_image) for one frame's vegetation tiles.

    Tiling comes from tile_common (plan U2, R5): ceil grid with clamped partial
    edge crops resized up — training's rule. The legacy floor grid silently
    dropped partial right/bottom edge tiles.
    """
    im = Image.open(path).convert("RGB")
    W, H = im.size
    if max(W, H) > MAX:
        s = MAX / max(W, H); im = im.resize((round(W * s), round(H * s))); W, H = im.size
    exg = tile_common.exg_map(im)
    cols, rows = tile_common.tile_grid(W, H, TILE)
    ts, coords = [], []
    for r, c, box in tile_common.tile_boxes(W, H, TILE):
        if not tile_common.tile_is_veg(exg, box, VEG):
            continue
        ts.append(to_norm(tile_common.cut_tile(im, box, TILE))); coords.append((r, c))
    batch = torch.stack(ts) if ts else torch.empty(0, 3, CNN, CNN)
    return batch, coords, rows, cols, im


def adapt_bn(model, paths):
    """AdaBN: recompute BatchNorm running stats on the target frames' tiles (shared impl)."""
    n = tile_common.adapt_bn(model, (tiles_of(p)[0] for p in paths),
                             device=device, verbose=False)
    print(f"AdaBN: recomputed BatchNorm stats on {n} target tiles from {len(paths)} frame(s)")


def render(path, model, cog_idx):
    plt = _plt()
    b, coords, rows, cols, im = tiles_of(path)
    if len(b) == 0:
        print(f"{Path(path).name}: no vegetation tiles"); return
    with torch.no_grad():
        probs = torch.cat([model(b[i:i + 256].to(device)).softmax(1)[:, cog_idx].cpu()
                           for i in range(0, len(b), 256)]).numpy()
    grid = np.full((rows, cols), np.nan)
    for (r, c), p in zip(coords, probs):
        grid[r, c] = p
    cov = 100 * (probs >= COG_THRESH).mean()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(im)
    hm = ax.imshow(np.ma.masked_invalid(grid), extent=[0, cols * TILE, rows * TILE, 0],
                   cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.5, interpolation="nearest")
    ax.set_title(f"{Path(path).name}\ncogongrass coverage: {cov:.0f}%  "
                 f"({int((probs >= COG_THRESH).sum())}/{len(probs)} veg tiles)  [AdaBN={ADABN}]", fontsize=11)
    ax.axis("off"); fig.colorbar(hm, ax=ax, fraction=0.035, label="P(cogongrass)")
    out = OUT / f"{Path(path).stem}_heatmap.png"
    fig.savefig(out, bbox_inches="tight", dpi=110); plt.close(fig)
    print(f"{Path(path).name}: {cov:.0f}% cogongrass ({len(probs)} veg tiles) -> {out}")


def main(args):
    paths = []
    for a in args:
        p = Path(a)
        paths += sorted(p.glob("*.jpg")) + sorted(p.glob("*.JPG")) if p.is_dir() else [p]
    model, cog_idx = load_model()
    print(f"loaded {MODEL} on {device}; {len(paths)} frame(s)")
    print("note: ceil tiling (edge tiles now scored) — outputs are a new regime vs pre-refactor heatmaps")
    if ADABN:
        adapt_bn(model, paths)          # adapt to this field's frames before predicting
    for p in paths:
        render(p, model, cog_idx)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python heatmap_infer.py <image_or_folder> [more ...]"); sys.exit(1)
    main(sys.argv[1:])

"""Decision-threshold sweep for the cogongrass tile classifier.

False negatives (missed cogongrass) are far costlier than false positives, so we
lower the threshold below 0.5 to trade precision for recall. Uses the deployable
512 DA model + AdaBN on the held-out 0422 collection, then sweeps the threshold
and reports recall / precision / F1 / F2 (F2 weights recall 2x).

Run:  python threshold_sweep.py [tiles_dataset] [tile_classifier_da_noclahe.pt]
"""
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

import tile_common

DATA = sys.argv[1] if len(sys.argv) > 1 else "tiles_dataset"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "tile_classifier_da_noclahe.pt"
HELDOUT, IMG = tile_common.HELDOUT_DATES, 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_tf = transforms.Compose([transforms.Resize(IMG), transforms.CenterCrop(IMG),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
# identity comes from the shared contract (plan U2, R1)
frame_of = tile_common.frame_of
date_of = tile_common.date_of


def main():
    base = datasets.ImageFolder(DATA, transform=eval_tf)
    classes = base.classes; cog = classes.index("cogongrass")
    te = [i for i, (p, _) in enumerate(base.samples) if date_of(frame_of(p)) in HELDOUT]
    loader = DataLoader(Subset(base, te), 64, shuffle=False, num_workers=4)
    ckpt = torch.load(MODEL, map_location=device)
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, len(classes)))
    m.load_state_dict(ckpt["state_dict"]); m = m.to(device)

    # AdaBN: recompute BN stats on target (shared impl, R1)
    tile_common.adapt_bn(m, loader, device=device, verbose=False)

    # collect P(cogongrass) and true labels
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(m(x.to(device)).softmax(1)[:, cog].cpu()); ys += y.tolist()
    p = torch.cat(probs).numpy()
    true_cog = np.array([1 if y == cog else 0 for y in ys])
    n_cog = int(true_cog.sum())
    print(f"model={MODEL}  test tiles={len(p)}  ({n_cog} cogongrass)")
    print(f"\n thr | recall  prec   F1     F2   | missed cogongrass (FN)")
    print("-----+----------------------------+----------------------")
    for thr in [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]:
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (true_cog == 1)).sum())
        fp = int(((pred == 1) & (true_cog == 0)).sum())
        fn = int(((pred == 0) & (true_cog == 1)).sum())
        rec = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        f2 = 5 * prec * rec / (4 * prec + rec + 1e-9)
        print(f"{thr:4.2f} | {rec:.3f}  {prec:.3f}  {f1:.3f}  {f2:.3f} | {fn}/{n_cog}")


if __name__ == "__main__":
    main()

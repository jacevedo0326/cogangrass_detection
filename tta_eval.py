"""Test-time adaptation on the held-out 0422 collection (AdaBN + TENT).

Takes an already-trained DA model and adapts it to the TARGET collection's frames
at inference (no labels, no retraining) — the realistic deployment setting. Reports
SOURCE (no adaptation), AdaBN (recompute BatchNorm stats on target), and TENT
(entropy-minimize BN affine params on target).

Run:  python tta_eval.py [tiles_dataset_clahe] [tile_classifier_da_clahe.pt]
"""
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import balanced_accuracy_score, classification_report

import tile_common

DATA = sys.argv[1] if len(sys.argv) > 1 else "tiles_dataset_clahe"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "tile_classifier_da_clahe.pt"
HELDOUT, IMG, SEED = tile_common.HELDOUT_DATES, 224, 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_tf = transforms.Compose([transforms.Resize(IMG), transforms.CenterCrop(IMG),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
# identity comes from the shared contract (plan U2, R1) — Path-based, so the old
# Windows "\\" path special-casing is gone
frame_of = tile_common.frame_of
date_of = tile_common.date_of


def test_indices(samples):
    return [i for i, (p, _) in enumerate(samples) if date_of(frame_of(p)) in HELDOUT]


def build(classes):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, len(classes)))
    return m


def main():
    base = datasets.ImageFolder(DATA, transform=eval_tf)
    classes = base.classes
    loader = DataLoader(Subset(base, test_indices(base.samples)), 64, shuffle=False, num_workers=2)
    ckpt = torch.load(MODEL, map_location=device)
    print(f"model={MODEL}  data={DATA}  test tiles={len(loader.dataset)}  classes={classes}")

    def fresh():
        m = build(classes); m.load_state_dict(ckpt["state_dict"]); return m.to(device)

    @torch.no_grad()
    def predict(m):
        yt, yp = [], []
        for x, y in loader:
            yp += m(x.to(device)).argmax(1).cpu().tolist(); yt += y.tolist()
        return balanced_accuracy_score(yt, yp), yt, yp

    bn = lambda m: [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]

    # --- SOURCE (no adaptation) ---
    m = fresh(); m.eval()
    b0, yt, yp = predict(m)

    # --- AdaBN: recompute BN running stats on target inputs (shared impl, R1) ---
    m = fresh(); m.eval()
    tile_common.adapt_bn(m, loader, device=device, verbose=False)
    b1, _, _ = predict(m)

    # --- TENT: entropy-minimize BN affine (gamma/beta) on target ---
    m = fresh(); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    params = []
    for mod in bn(m):
        mod.train(); mod.track_running_stats = False; mod.running_mean = None; mod.running_var = None
        mod.weight.requires_grad_(True); mod.bias.requires_grad_(True)
        params += [mod.weight, mod.bias]
    opt = torch.optim.Adam(params, lr=1e-3)
    for _ in range(2):                      # 2 adaptation passes over target
        for x, _ in loader:
            out = m(x.to(device))
            p = out.softmax(1)
            loss = -(p * p.clamp_min(1e-8).log()).sum(1).mean()   # prediction entropy
            opt.zero_grad(); loss.backward(); opt.step()
    b2, _, _ = predict(m)                   # BN still in batch-stat mode, dropout off, no grad

    print(f"\n===== held-out {','.join(HELDOUT)} balanced accuracy =====")
    print(f"  SOURCE (no adaptation) : {b0:.3f}")
    print(f"  AdaBN  (target BN stats): {b1:.3f}")
    print(f"  TENT   (entropy adapt)  : {b2:.3f}")


if __name__ == "__main__":
    main()

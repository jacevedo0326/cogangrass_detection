"""Are false-positive cogongrass tiles ADJACENT to real cogongrass?

If FP tiles sit next to true cogongrass, they're likely boundary/label effects
(the model is ~right at a stand edge), not random false alarms. We also compute
the BASELINE adjacency (how often ANY not-cogongrass tile is next to cogongrass)
-- if the field is cogongrass-dense, 'adjacent' is trivially common and the
signal is meaningless. Uses the 512 DA model + AdaBN on held-out 0422.
"""
import re
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DATA, MODEL, TEST_DATE, IMG = "tiles_dataset", "tile_classifier_da_noclahe.pt", "20260422", 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_tf = transforms.Compose([transforms.Resize(IMG), transforms.CenterCrop(IMG),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def parse(p):
    m = re.match(r"(.+)_r(\d+)_c(\d+)$", Path(p).stem)
    return m.group(1), int(m.group(2)), int(m.group(3))


def main():
    base = datasets.ImageFolder(DATA, transform=eval_tf)
    classes = base.classes; cog = classes.index("cogongrass")
    te = [i for i, (p, _) in enumerate(base.samples)
          if re.match(r"DJI_(\d{8})", parse(p)[0]) and parse(p)[0][4:12] == TEST_DATE]
    loader = DataLoader(Subset(base, te), 64, shuffle=False, num_workers=4)

    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, len(classes)))
    m.load_state_dict(torch.load(MODEL, map_location=device)["state_dict"]); m = m.to(device)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats(); mod.momentum = None
    m.eval()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d): mod.train()
    with torch.no_grad():
        for x, _ in loader: m(x.to(device))
    m.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs += m(x.to(device)).softmax(1)[:, cog].cpu().tolist()

    # build per-frame grids
    truecog = defaultdict(set); cells = defaultdict(dict)
    for k, i in enumerate(te):
        frame, r, c = parse(base.samples[i][0])
        is_cog = base.samples[i][1] == cog
        if is_cog:
            truecog[frame].add((r, c))
        cells[frame][(r, c)] = (is_cog, probs[k])
    nbrs = lambda r, c: [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc)]

    # density context
    tot = sum(len(v) for v in cells.values()); cogtot = sum(len(v) for v in truecog.values())
    print(f"frames={len(cells)}  veg tiles={tot}  true cogongrass tiles={cogtot} ({100*cogtot/tot:.0f}%)")

    print(f"\n thr | FP total | FP adjacent to cogongrass | FP ISOLATED | baseline(any not-cog adj)")
    print("-----+----------+---------------------------+-------------+--------------------------")
    for thr in (0.50, 0.30, 0.20):
        fp = fp_adj = notcog = notcog_adj = 0
        for frame, cs in cells.items():
            cg = truecog[frame]
            for (r, c), (tc, pr) in cs.items():
                if tc:
                    continue
                adj = any(n in cg for n in nbrs(r, c))
                notcog += 1; notcog_adj += adj
                if pr >= thr:
                    fp += 1; fp_adj += adj
        iso = fp - fp_adj
        print(f"{thr:4.2f} | {fp:8d} | {fp_adj:5d} ({100*fp_adj/max(fp,1):3.0f}%)            "
              f"   | {iso:4d} ({100*iso/max(fp,1):3.0f}%) | {100*notcog_adj/max(notcog,1):3.0f}%")


if __name__ == "__main__":
    main()

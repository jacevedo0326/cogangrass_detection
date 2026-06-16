"""DINOv2 done right: spatial PATCH features + conv head (not CLS-linear).

Per the BASF/Tecnalia paper, DINOv2 generalizes across domains when its dense
patch tokens feed a real decoder/head — our earlier frozen DINOv2 used a CLS
linear probe and regressed. Here: frozen DINOv2 ViT-S/14 -> 16x16x384 patch
feature map -> small conv head WITH BatchNorm (so AdaBN test-time adaptation
still applies). Cross-collection protocol (train 0606, TEST held-out 0422),
full-res 512px tiles -> 224, our augmentation. Reports SOURCE and AdaBN.

Run:  python -u train_tiles_dino_spatial.py
"""
import re
import copy
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import balanced_accuracy_score, classification_report

DATA = "tiles_dataset"          # 512px no-CLAHE tiles on disk
IMG, BATCH, MAX_EPOCHS, PATIENCE, LR, SEED = 224, 48, 60, 10, 1e-3, 42
VAL_FRAC, TEST_DATE = 0.12, "20260422"
OUT_MODEL = "tile_classifier_dino_spatial.pt"
random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30), transforms.RandomPerspective(0.3, p=0.3),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.15), transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([transforms.Resize(IMG), transforms.CenterCrop(IMG),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

frame_of = lambda p: re.match(r"(.+)_r\d+_c\d+$", Path(p).stem).group(1)
def date_of(f):
    m = re.match(r"DJI_(\d{8})", f); return m.group(1) if m else "other"


def split_by_collection(samples, cog):
    fr = {}
    for i, (p, lab) in enumerate(samples):
        f = frame_of(p); d = fr.setdefault(f, {"idx": [], "pos": False, "date": date_of(f)})
        d["idx"].append(i)
        if lab == cog: d["pos"] = True
    test = [f for f, d in fr.items() if d["date"] == TEST_DATE]
    pool = [f for f, d in fr.items() if d["date"] != TEST_DATE]
    rng = random.Random(SEED)
    pos = [f for f in pool if fr[f]["pos"]]; neg = [f for f in pool if not fr[f]["pos"]]
    rng.shuffle(pos); rng.shuffle(neg)
    nvp, nvn = int(len(pos) * VAL_FRAC), int(len(neg) * VAL_FRAC)
    val = pos[:nvp] + neg[:nvn]; tr = pos[nvp:] + neg[nvn:]
    idx = lambda fl: [i for f in fl for i in fr[f]["idx"]]
    return idx(tr), idx(val), idx(test)


def balance(idx, samples, cog, rng):
    pos = [i for i in idx if samples[i][1] == cog]; neg = [i for i in idx if samples[i][1] != cog]
    if pos and neg:
        if len(neg) > len(pos): neg = rng.sample(neg, len(pos))
        elif len(pos) > len(neg): pos = rng.sample(pos, len(neg))
    out = pos + neg; rng.shuffle(out); return out


class DinoSpatial(nn.Module):
    def __init__(self, backbone, dim=384, grid=16, n=2, p=0.4):
        super().__init__()
        self.backbone = backbone
        for q in self.backbone.parameters():
            q.requires_grad_(False)
        self.grid = grid
        self.head = nn.Sequential(
            nn.Conv2d(dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p), nn.Linear(128, n))

    def train(self, mode=True):
        super().train(mode); self.backbone.eval(); return self

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone.forward_features(x)["x_norm_patchtokens"]   # [B, grid*grid, dim]
        B = x.shape[0]
        f = f.reshape(B, self.grid, self.grid, -1).permute(0, 3, 1, 2).contiguous()
        return self.head(f)


def run(model, loader, crit, opt=None, scaler=None):
    train = opt is not None
    model.train() if train else model.eval()
    ls, yp, yt = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(x); loss = crit(out, y)
            if train:
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ls += loss.item() * x.size(0); yp += out.argmax(1).cpu().tolist(); yt += y.cpu().tolist()
    return ls / len(yt), balanced_accuracy_score(yt, yp), yt, yp


@torch.no_grad()
def predict(model, loader):
    model.eval(); yp, yt = [], []
    for x, y in loader:
        yp += model(x.to(device)).argmax(1).cpu().tolist(); yt += y.tolist()
    return balanced_accuracy_score(yt, yp), yt, yp


def main():
    print("loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    base_tr = datasets.ImageFolder(DATA, transform=train_tf)
    base_ev = datasets.ImageFolder(DATA, transform=eval_tf)
    classes = base_tr.classes; cog = classes.index("cogongrass")
    tr, va, te = split_by_collection(base_ev.samples, cog)
    rng = random.Random(SEED)
    tr = balance(tr, base_ev.samples, cog, rng); va = balance(va, base_ev.samples, cog, rng)
    print(f"tiles -> train {len(tr)} | val {len(va)} | TEST(0422) {len(te)}")
    tl = DataLoader(Subset(base_tr, tr), BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    vl = DataLoader(Subset(base_ev, va), BATCH, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    el = DataLoader(Subset(base_ev, te), BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = DinoSpatial(backbone, n=len(classes)).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.3, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best, best_state, since = 0.0, copy.deepcopy(model.head.state_dict()), 0
    for ep in range(1, MAX_EPOCHS + 1):
        _, tb, *_ = run(model, tl, crit, opt, scaler)
        _, vb, *_ = run(model, vl, crit)
        sched.step(vb); flag = ""
        if vb > best + 1e-4:
            best, best_state, since = vb, copy.deepcopy(model.head.state_dict()), 0; flag = "  <- best"
            torch.save({"head_state": best_state, "classes": classes, "backbone": "dinov2_vits14"}, OUT_MODEL)
        else:
            since += 1
        print(f"epoch {ep:2d}  train_bacc {tb:.3f} | val_bacc {vb:.3f}{flag}")
        if since >= PATIENCE:
            print(f"early stop after {ep} epochs"); break
    model.head.load_state_dict(best_state)
    print(f"\nin-collection VAL best bacc: {best:.3f}")

    # held-out 0422: SOURCE
    b0, yt, yp = predict(model, el)
    # held-out 0422: AdaBN (recompute head BN stats on target)
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats(); mod.momentum = None
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d): mod.train()
    with torch.no_grad():
        for x, _ in el: model(x.to(device))
    b1, _, _ = predict(model, el)

    print("\n===== HELD-OUT TEST (0422) =====")
    print(f"  DINOv2-spatial SOURCE : {b0:.3f}")
    print(f"  DINOv2-spatial AdaBN  : {b1:.3f}   (vs ResNet-DA AdaBN 0.84)")
    print(classification_report(yt, yp, target_names=classes, digits=3))


if __name__ == "__main__":
    main()

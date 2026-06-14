"""Train a tile classifier: does this tile contain cogongrass or not?

Reads tiles_dataset/{cogongrass,not_cogongrass}/ (built by boxes_to_tiles.py),
transfer-learns a ResNet18, and reports honest metrics. Tuned for the current
imbalanced data (~88% positive) and for the RTX 2060.

Key choices:
  * split BY FRAME (not by tile) so one image's tiles never span train/test
    -> no leakage inflating the score
  * class-weighted loss so it can't win by always predicting cogongrass
  * reports balanced accuracy + per-class recall, not just raw accuracy

Run:  python train_tiles.py
"""
import re
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

DATA = "tiles_dataset"
IMG_SIZE = 224
BATCH = 64
MAX_EPOCHS = 60
PATIENCE = 8
LR = 1e-4
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15

random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),  # aerial: both valid
    transforms.RandomRotation(20), transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


def frame_of(path):
    """tiles_dataset/<cls>/<imgstem>_r#_c#.jpg -> imgstem (the source frame)."""
    return re.match(r"(.+)_r\d+_c\d+$", Path(path).stem).group(1)


def grouped_split(samples, cog_idx):
    """Split FRAMES into train/val/test, stratified by whether a frame contains any
    cogongrass tile (so cogongrass-bearing and all-negative frames are both spread
    across splits). One frame's tiles never span splits -> no leakage."""
    frames = {}
    for i, (p, lab) in enumerate(samples):
        d = frames.setdefault(frame_of(p), {"idx": [], "pos": False})
        d["idx"].append(i)
        if lab == cog_idx:
            d["pos"] = True
    groups = {True: [], False: []}
    for f, d in frames.items():
        groups[d["pos"]].append(f)
    tr, va, te = [], [], []
    for fl in groups.values():
        random.shuffle(fl)
        n = len(fl); nt = int(n * TEST_FRAC); nv = int(n * VAL_FRAC)
        te += fl[:nt]; va += fl[nt:nt + nv]; tr += fl[nt + nv:]
    idx = lambda fl: [i for f in fl for i in frames[f]["idx"]]
    return idx(tr), idx(va), idx(te), (len(tr), len(va), len(te))


def run(model, loader, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum, yps, yts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(x); loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * x.size(0)
            yps += out.argmax(1).cpu().tolist(); yts += y.cpu().tolist()
    return loss_sum / len(yts), balanced_accuracy_score(yts, yps), yts, yps


def main():
    base_train = datasets.ImageFolder(DATA, transform=train_tf)
    base_eval = datasets.ImageFolder(DATA, transform=eval_tf)
    classes = base_train.classes
    print("classes:", classes, "(label index order)")

    cog_idx = classes.index("cogongrass") if "cogongrass" in classes else 0
    tr_idx, va_idx, te_idx, (nf_tr, nf_va, nf_te) = grouped_split(base_eval.samples, cog_idx)
    train_set = Subset(base_train, tr_idx)
    val_set = Subset(base_eval, va_idx)
    test_set = Subset(base_eval, te_idx)
    print(f"frames -> train {nf_tr} | val {nf_va} | test {nf_te}")
    print(f"tiles  -> train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}")

    # class weights from the TRAIN split (counter the 88% imbalance)
    tr_labels = [base_eval.samples[i][1] for i in tr_idx]
    counts = np.bincount(tr_labels, minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float, device=device)
    print("train class counts:", dict(zip(classes, counts.tolist())), "| loss weights:", weights.tolist())

    train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, BATCH, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_bacc, best_state, since = 0.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        tl, tb, *_ = run(model, train_loader, criterion, optimizer, scaler)
        vl, vb, *_ = run(model, val_loader, criterion)
        sched.step(vb)
        flag = ""
        if vb > best_bacc + 1e-4:
            best_bacc, best_state, since = vb, copy.deepcopy(model.state_dict()), 0; flag = "  <- best"
        else:
            since += 1
        print(f"epoch {epoch:2d}  train_loss {tl:.3f} bacc {tb:.3f} | val_loss {vl:.3f} bacc {vb:.3f}{flag}")
        if since >= PATIENCE:
            print(f"early stop after {epoch} epochs"); break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "classes": classes}, "tile_classifier.pt")

    # final held-out test
    _, test_bacc, yt, yp = run(model, test_loader, criterion)
    print("\n===== TEST =====")
    print(f"balanced accuracy: {test_bacc:.3f}   (raw accuracy is misleading while imbalanced)")
    print(classification_report(yt, yp, target_names=classes, digits=3))
    print("confusion (rows=true, cols=pred):")
    print("            " + "  ".join(f"{c[:10]:>12}" for c in classes))
    for name, row in zip(classes, confusion_matrix(yt, yp)):
        print(f"{name[:10]:>10}  " + "  ".join(f"{v:>12}" for v in row))
    print("\nsaved model -> tile_classifier.pt")


if __name__ == "__main__":   # required on Windows (dataloader multiprocessing)
    main()

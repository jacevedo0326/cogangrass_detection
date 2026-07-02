"""Cross-collection generalization test (kept SEPARATE from the main run).

Trains on the 2026-06-06 flight and evaluates on the ENTIRELY held-out
2026-04-22 flight - a proxy for "a new collection it has never seen."
Reuses tiles_dataset/ (no re-tiling). Saves to tile_classifier_collection.pt and
tile_train_collection.log - does NOT touch tile_classifier.pt or train_tiles.py.

Run:  python -u train_tiles_collection.py
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

import tile_common

DATA = "tiles_dataset"
IMG_SIZE, BATCH, MAX_EPOCHS, PATIENCE, LR, SEED = 224, 64, 60, 8, 1e-4, 42
VAL_FRAC = 0.12
HELDOUT = tile_common.HELDOUT_DATES        # held-out collection(s), default ["20260422"]
OUT_MODEL = "tile_classifier_collection.pt"

random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20), transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

# identity/split/balance now come from the shared contract (plan U2, R1); behavior
# at the defaults is bit-identical to the private copies these replaced
frame_of = tile_common.frame_of
date_of = tile_common.date_of
split_by_collection = tile_common.split_by_collection
balance = tile_common.balance


def run(model, loader, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum, yps, yts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
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
    cog_idx = classes.index("cogongrass")
    print("classes:", classes)

    tr, va, te, (nf_tr, nf_va, nf_te) = split_by_collection(
        base_eval.samples, cog_idx, heldout_dates=HELDOUT, seed=SEED, val_frac=VAL_FRAC)
    print(f"frames -> train {nf_tr} | val {nf_va} | TEST(held-out {','.join(HELDOUT)}) {nf_te}")
    rng = random.Random(SEED)
    tr = balance(tr, base_eval.samples, cog_idx, rng)
    va = balance(va, base_eval.samples, cog_idx, rng)        # test left at natural distribution
    def counts(idx):
        c = np.bincount([base_eval.samples[i][1] for i in idx], minlength=2)
        return {classes[k]: int(c[k]) for k in range(2)}
    print(f"train tiles {len(tr)} {counts(tr)} | val tiles {len(va)} {counts(va)} | "
          f"TEST tiles {len(te)} {counts(te)}")

    train_loader = DataLoader(Subset(base_train, tr), BATCH, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(base_eval, va), BATCH, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(Subset(base_eval, te), BATCH, shuffle=False, num_workers=2, pin_memory=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes)); model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_bacc, best_state, since = 0.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        tl, tb, *_ = run(model, train_loader, criterion, optimizer, scaler)
        vl, vb, *_ = run(model, val_loader, criterion)
        sched.step(vb); flag = ""
        if vb > best_bacc + 1e-4:
            best_bacc, best_state, since = vb, copy.deepcopy(model.state_dict()), 0; flag = "  <- best"
            torch.save({"state_dict": best_state, "classes": classes}, OUT_MODEL)
        else:
            since += 1
        print(f"epoch {epoch:2d}  train_loss {tl:.3f} bacc {tb:.3f} | val_loss {vl:.3f} bacc {vb:.3f}{flag}")
        if since >= PATIENCE:
            print(f"early stop after {epoch} epochs"); break

    model.load_state_dict(best_state)
    print(f"\nin-collection VAL (0606) best balanced accuracy: {best_bacc:.3f}")
    _, te_bacc, yt, yp = run(model, test_loader, criterion)
    print(f"\n===== HELD-OUT TEST (collection {','.join(HELDOUT)}) =====")
    print(f"balanced accuracy: {te_bacc:.3f}")
    print(classification_report(yt, yp, target_names=classes, digits=3))
    print("confusion (rows=true, cols=pred):")
    print("            " + "  ".join(f"{c[:10]:>12}" for c in classes))
    for name, row in zip(classes, confusion_matrix(yt, yp, labels=[0, 1])):
        print(f"{name[:10]:>10}  " + "  ".join(f"{v:>12}" for v in row))
    print(f"\nsaved -> {OUT_MODEL}")


if __name__ == "__main__":
    main()

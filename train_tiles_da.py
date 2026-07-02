"""Domain-generalization experiment (Stage 1): fix preprocessing + regularization.

Same cross-collection protocol as train_tiles_collection.py (train 2026-06-06,
TEST entirely held-out 2026-04-22) so improvements are measured honestly. Adds:
  PREPROCESSING:  CLAHE illumination normalization (train+eval) + heavy
                  domain-randomization augmentation (photometric + distortion).
  ARCHITECTURE:   dropout head, label smoothing, stronger weight decay.
Kept separate: saves tile_classifier_da.pt; touches no other model/script.

Run:  python -u train_tiles_da.py
"""
import sys
import copy
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

import tile_common

DATA = sys.argv[1] if len(sys.argv) > 1 else "tiles_dataset_clahe"   # pass tiles_dataset (no CLAHE) or tiles_dataset_clahe
IMG_SIZE, BATCH, MAX_EPOCHS, PATIENCE, LR, SEED = 224, 64, 60, 10, 1e-4, 42
VAL_FRAC = 0.12
HELDOUT = tile_common.HELDOUT_DATES
_tag = "clahe" if "clahe" in DATA else "noclahe"
OUT_MODEL = f"tile_classifier_da_{_tag}.pt"

random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class CLAHE:
    """Illumination normalization on the L channel of LAB - the key cross-collection fix."""
    def __init__(self, clip=2.0, grid=8):
        self.clip, self.grid = clip, grid

    def __call__(self, img):
        lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        cl = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.grid, self.grid))
        lab[..., 0] = cl.apply(lab[..., 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


# CLAHE applied to BOTH train and eval (it is normalization, not augmentation).
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),   # gentle: avoids upsampling 256px tiles
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),          # optical/grid distortion
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.15),                        # strong photometric
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25),
])
eval_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])

# identity/split/balance come from the shared contract (plan U2, R1)
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
    va = balance(va, base_eval.samples, cog_idx, rng)
    print(f"train tiles {len(tr)} | val tiles {len(va)} | TEST tiles {len(te)} (natural)")

    train_loader = DataLoader(Subset(base_train, tr), BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(Subset(base_eval, va), BATCH, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(Subset(base_eval, te), BATCH, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, len(classes)))  # dropout head
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)                       # label smoothing
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)  # stronger weight decay
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
    print(f"balanced accuracy: {te_bacc:.3f}   (baseline w/o DA fixes was 0.804)")
    print(classification_report(yt, yp, target_names=classes, digits=3))
    print("confusion (rows=true, cols=pred):")
    print("            " + "  ".join(f"{c[:10]:>12}" for c in classes))
    for name, row in zip(classes, confusion_matrix(yt, yp, labels=[0, 1])):
        print(f"{name[:10]:>10}  " + "  ".join(f"{v:>12}" for v in row))
    print(f"\nsaved -> {OUT_MODEL}")


if __name__ == "__main__":
    main()

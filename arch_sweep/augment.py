"""Domain-generalization augmentation for the train path (U7).

Three composable, **train-only** techniques applied to the 0606 path and **never** to the
0422 eval path (the trainer builds a separate un-augmented eval dataset — eval purity):

- **domain_randomization** — bounded photometric + geometric jitter. Bounds are deliberately
  capped (``AUG_BOUNDS``) so the green channel that the ExG cogongrass cue rides on is not
  washed out (saturation/hue stay small).
- **fourier_amplitude_swap** — swap the low-frequency *amplitude* between an image and a
  reference while keeping the image's *phase*, transferring "style" (illumination/sensor)
  without changing semantics. Swapping an image with itself is a no-op.
- **MixStyle** — mix per-sample feature statistics across the batch in train mode; a no-op in
  eval. Operates in feature space so it composes with the frozen/fine-tune paths.

CLAHE on/off is **not** here — it is a materialized data variant (U2), selected by pointing a
cell at the CLAHE variant dir.
"""
from __future__ import annotations

# Bounded jitter so the ExG green cue survives (documented green-cue guard).
AUG_BOUNDS = {"brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.05,
              "blur_p": 0.2, "grayscale_p": 0.05}


def domain_randomization(img_size: int = 224):
    """A bounded torchvision transform (PIL -> normalized tensor) for the train path."""
    from torchvision import transforms
    b = AUG_BOUNDS
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(b["brightness"], b["contrast"], b["saturation"], b["hue"]),
        transforms.RandomGrayscale(p=b["grayscale_p"]),     # low p so green isn't usually removed
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=b["blur_p"]),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])


def fourier_amplitude_swap(img, ref, beta: float = 0.1):
    """Swap the low-frequency amplitude of ``img`` with ``ref`` (keep ``img`` phase).

    ``img``/``ref`` are ``(C, H, W)`` tensors. A central square of side ``beta`` of the
    amplitude spectrum is replaced; phase (structure/semantics) is untouched. Swapping with
    self returns the input (identity), the property the test checks.
    """
    import torch

    fi = torch.fft.fftshift(torch.fft.fft2(img.float(), dim=(-2, -1)), dim=(-2, -1))
    fr = torch.fft.fftshift(torch.fft.fft2(ref.float(), dim=(-2, -1)), dim=(-2, -1))
    amp_i, pha_i = fi.abs(), fi.angle()
    amp_r = fr.abs()
    _, H, W = img.shape
    ch, cw = H // 2, W // 2
    bh, bw = max(1, int(H * beta) // 2), max(1, int(W * beta) // 2)
    amp_i[:, ch - bh:ch + bh, cw - bw:cw + bw] = amp_r[:, ch - bh:ch + bh, cw - bw:cw + bw]
    f = torch.fft.ifftshift(amp_i * torch.exp(1j * pha_i), dim=(-2, -1))
    return torch.fft.ifft2(f, dim=(-2, -1)).real.to(img.dtype)


def make_mixstyle(p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
    """MixStyle over pooled feature vectors: mixes per-sample mean/std across the batch.

    Train mode only (with probability ``p``); eval mode and batches < 2 are exact no-ops.
    """
    import torch
    import torch.nn as nn

    class MixStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.p, self.alpha, self.eps = p, alpha, eps
            self.beta = torch.distributions.Beta(alpha, alpha)

        def forward(self, x):
            if not self.training or x.size(0) < 2 or float(torch.rand(1)) > self.p:
                return x
            mu = x.mean(1, keepdim=True)
            sig = (x.var(1, keepdim=True) + self.eps).sqrt()
            x_norm = (x - mu) / sig
            lam = self.beta.sample((x.size(0), 1)).to(x.device)
            perm = torch.randperm(x.size(0), device=x.device)
            mu_mix = lam * mu + (1 - lam) * mu[perm]
            sig_mix = lam * sig + (1 - lam) * sig[perm]
            return x_norm * sig_mix + mu_mix

    return MixStyle()

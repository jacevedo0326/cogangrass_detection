"""U7 tests: Fourier swap identity/shape, MixStyle train-vs-eval, green-cue bound, eval purity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import augment as A  # noqa: E402


def test_fourier_self_swap_is_identity_and_preserves_shape_range():
    import torch
    img = torch.rand(3, 32, 32)
    out = A.fourier_amplitude_swap(img, img, beta=0.1)   # swap with self -> no change
    assert out.shape == img.shape
    assert torch.allclose(out, img, atol=1e-5)


def test_fourier_swap_preserves_phase_structure_changes_style():
    import torch
    torch.manual_seed(0)
    img = torch.rand(3, 32, 32)
    ref = torch.rand(3, 32, 32) * 5.0           # different amplitude scale ("style")
    out = A.fourier_amplitude_swap(img, ref, beta=0.15)
    assert out.shape == img.shape
    assert not torch.allclose(out, img, atol=1e-3)   # style transferred (low-freq amplitude moved)
    # phase preserved -> structure correlates with the original, not the reference
    assert torch.corrcoef(torch.stack([out.flatten(), img.flatten()]))[0, 1] > \
        torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1]


def test_mixstyle_active_in_train_noop_in_eval():
    import torch
    torch.manual_seed(0)
    ms = A.make_mixstyle(p=1.0)        # always mix when training
    x = torch.randn(8, 16)
    ms.eval()
    assert torch.allclose(ms(x), x)    # eval -> exact no-op
    ms.train()
    assert not torch.allclose(ms(x), x)   # train -> mixes feature stats


def test_mixstyle_noop_on_singleton_batch():
    import torch
    ms = A.make_mixstyle(p=1.0)
    ms.train()
    x = torch.randn(1, 16)
    assert torch.allclose(ms(x), x)    # needs >=2 samples to mix


def test_green_cue_bounds_are_capped():
    # saturation/hue jitter stay small so the ExG green cue isn't washed out (documented guard)
    assert A.AUG_BOUNDS["saturation"] <= 0.25
    assert A.AUG_BOUNDS["hue"] <= 0.06
    assert A.AUG_BOUNDS["grayscale_p"] <= 0.1


def test_green_channel_survives_domain_randomization():
    import numpy as np
    import torch
    from PIL import Image
    tf = A.domain_randomization(img_size=32)
    green = Image.new("RGB", (40, 40), (0, 200, 0))
    # over several draws, the green channel stays the largest on average (cue not washed out)
    greens_win = 0
    for _ in range(10):
        t = tf(green)                  # normalized tensor (C,H,W)
        means = t.mean(dim=(1, 2))     # per-channel mean (normalized space)
        greens_win += int(means[1] >= means[0] and means[1] >= means[2])
    assert greens_win >= 7             # green dominates in the large majority of draws


def test_domain_randomization_outputs_normalized_tensor():
    import torch
    from PIL import Image
    tf = A.domain_randomization(img_size=24)
    out = tf(Image.new("RGB", (30, 30), (10, 150, 10)))
    assert isinstance(out, torch.Tensor) and out.shape == (3, 24, 24)


def test_augment_wires_into_finetune_and_tags_extra(tmp_path, monkeypatch):
    """Integration: a full-finetune cell with --augment trains through the augmented path."""
    from PIL import Image
    import features as FEAT
    import trainer as T

    frames = [("20260606", "cogongrass", ["0001", "0002"]),
              ("20260606", "not_cogongrass", ["0003", "0004"]),
              ("20260422", "cogongrass", ["0010", "0011"]),
              ("20260422", "not_cogongrass", ["0012", "0013"])]
    root = tmp_path / "tiles"
    for date, cls, ids in frames:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        color = (0, 200, 0) if cls == "cogongrass" else (120, 120, 120)
        for fid in ids:
            for t in range(2):
                Image.new("RGB", (16, 16), color).save(d / f"DJI_{date}_{fid}_r0_c{t}.jpg")
    monkeypatch.setattr(FEAT, "_variant_dir", lambda variant, _root=None: str(root))
    cfg = T.TrainConfig(model="resnet18", tuning_mode="full", augment=True, extra="aug",
                        ft_epochs=1, ft_batch=4, hidden=8)
    row = T.train_and_eval(cfg, results_dir=tmp_path)
    assert row.status == "ok", row.error
    assert row.extra == "aug"                 # augmented cell is a distinct job_id
    assert row.balanced_accuracy is not None

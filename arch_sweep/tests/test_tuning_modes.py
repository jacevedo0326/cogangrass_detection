"""U6 tests: LoRA/full param scoping (stub, no download) + a full-finetune smoke on CPU."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backbones as B  # noqa: E402
import features as FEAT  # noqa: E402
import trainer as T  # noqa: E402

COG = 0


def _tiny_vit():
    import torch.nn as nn

    class TinyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(16, 48)      # name matches LoRA hint -> wrapped
            self.proj = nn.Linear(16, 16)     # matches -> wrapped
            self.mlp = nn.Linear(16, 16)      # no hint -> not wrapped
    return TinyViT()


def test_lora_param_count_much_smaller_than_full():
    import torch.nn as nn
    m_full = _tiny_vit()
    for p in m_full.parameters():
        p.requires_grad = True
    full = B.count_trainable(m_full)

    m_lora = _tiny_vit()
    wrapped = B.inject_lora(m_lora, rank=4)
    lora = B.count_trainable(m_lora)
    assert wrapped == 2                         # qkv + proj wrapped, mlp left alone
    assert 0 < lora < full                      # LoRA trains far fewer params than full
    assert lora < full * 0.5


def test_lora_adapters_land_on_base_device():
    # regression: LoRA params must be created on the base layer's device (GPU-safe), not CPU
    import torch
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no cuda")
    base = torch.nn.Linear(8, 8).to("cuda")
    lora = B.make_lora_linear(base, rank=4)
    assert lora.A.device.type == "cuda" and lora.B.device.type == "cuda"
    out = lora(torch.randn(2, 8, device="cuda"))   # forward must not raise a device mismatch
    assert out.shape == (2, 8)


def test_lora_adapter_starts_as_identity():
    import torch
    m = _tiny_vit()
    B.inject_lora(m, rank=4)
    x = torch.randn(2, 16)
    # B is zero-init -> the LoRA update is zero at start (output == base linear)
    lora_qkv = m.qkv
    assert torch.allclose(lora_qkv(x), lora_qkv.base(x))


def _tiny_imagefolder(root: Path):
    from PIL import Image
    frames = [("20260606", "cogongrass", ["0001", "0002"]),
              ("20260606", "not_cogongrass", ["0003", "0004"]),
              ("20260422", "cogongrass", ["0010", "0011"]),
              ("20260422", "not_cogongrass", ["0012", "0013"])]
    for date, cls, ids in frames:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for fid in ids:
            color = (0, 200, 0) if cls == "cogongrass" else (120, 120, 120)
            for t in range(2):
                Image.new("RGB", (16, 16), color).save(d / f"DJI_{date}_{fid}_r0_c{t}.jpg")
    return root


def test_full_finetune_smoke_writes_row(tmp_path, monkeypatch):
    folder = _tiny_imagefolder(tmp_path / "tiles")
    monkeypatch.setattr(FEAT, "_variant_dir", lambda variant, root=None: str(folder))
    cfg = T.TrainConfig(model="resnet18", tuning_mode="full", head="mlp_bn",
                        ft_epochs=1, ft_batch=4, hidden=8)
    row = T.train_and_eval(cfg, results_dir=tmp_path)
    assert row.status == "ok", row.error
    assert row.tuning_mode == "full"
    assert row.trainable_params and row.trainable_params > 100000   # whole resnet18 is trainable
    assert row.balanced_accuracy is not None
    assert len(list(tmp_path.glob("*.jsonl"))) == 1


def test_mode_is_recorded_in_identity():
    # job_id distinguishes frozen vs lora vs full (so the orchestrator schedules all three)
    import common as C
    base = {"model": "siglip2", "variant": "reference"}
    ids = {C.job_id({**base, "tuning_mode": m}) for m in ("frozen", "lora", "full")}
    assert len(ids) == 3

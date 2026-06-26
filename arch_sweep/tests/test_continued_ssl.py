"""U9 tests: freeze discipline, atomic checkpoint round-trip, unlabeled SSL loop."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continued_ssl as CSSL  # noqa: E402


def _stub_vit():
    import torch.nn as nn

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_qkv = nn.Linear(8, 24)   # 'qkv' -> LoRA-wrapped
            self.mlp = nn.Linear(8, 8)          # not wrapped

    class StubViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Blk(), Blk(), Blk()])
            self.norm = nn.Linear(8, 8)
    return StubViT()


def test_freeze_discipline_lora_plus_last_blocks_only():
    m = _stub_vit()
    info = CSSL.freeze_for_ssl(m, n_blocks=1, lora_rank=4)
    assert info["lora_layers"] == 3          # qkv wrapped in all 3 blocks
    assert info["unfrozen_blocks"] == 1
    # last block fully trainable; earlier blocks' mlp frozen; norm frozen
    assert all(p.requires_grad for p in m.blocks[2].mlp.parameters())
    assert not any(p.requires_grad for p in m.blocks[0].mlp.parameters())
    assert not any(p.requires_grad for p in m.norm.parameters())
    # earlier blocks still expose trainable LoRA adapters (base frozen)
    qkv0 = m.blocks[0].attn_qkv
    assert qkv0.A.requires_grad and qkv0.B.requires_grad
    assert not any(p.requires_grad for p in qkv0.base.parameters())


def test_checkpoint_atomic_round_trip(tmp_path):
    import torch
    state = {"w": torch.arange(6).float()}
    path = CSSL.save_checkpoint_atomic(state, tmp_path / "ck.pt", {"steps": 10, "lora_rank": 8})
    back = CSSL.load_checkpoint(path)
    assert torch.equal(back["state"]["w"], state["w"])
    assert back["meta"]["steps"] == 10


def test_checkpoint_write_is_atomic_on_interrupt(tmp_path, monkeypatch):
    import torch
    final = tmp_path / "ck.pt"

    def boom(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(CSSL.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        CSSL.save_checkpoint_atomic({"w": torch.zeros(3)}, final)
    assert not final.exists()                                   # no partial final checkpoint
    assert not list(tmp_path.glob(".tmp-ckpt-*"))               # temp cleaned up


def test_ssl_loop_consumes_unlabeled_batches_and_checkpoints(tmp_path):
    import torch
    import torch.nn as nn

    backbone = nn.Linear(6, 8)               # stand-in backbone: features dim 8

    def fwd(x):
        return backbone(x)

    def two_views(x):
        return x + 0.01 * torch.randn_like(x), x + 0.01 * torch.randn_like(x)

    # loader yields (image, label) but the label is None — proving labels are never read
    loader = [(torch.randn(4, 6), None) for _ in range(3)]

    def state_fn():
        return backbone.state_dict()
    state_fn.module = backbone

    out = tmp_path / "ssl.pt"
    steps = CSSL.run_ssl(fwd, two_views, loader, in_dim=8, steps=3, out_path=out,
                         device="cpu", checkpoint_every=1, state_fn=state_fn)
    assert steps == 3
    assert out.exists()
    ck = CSSL.load_checkpoint(out)
    assert ck["meta"]["steps"] == 3 and "weight" in ck["state"]


def test_find_blocks_handles_common_layouts():
    import torch.nn as nn
    m = _stub_vit()
    assert CSSL.find_blocks(m) is m.blocks

    class HF(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([nn.Linear(4, 4)])
    assert len(CSSL.find_blocks(HF())) == 1

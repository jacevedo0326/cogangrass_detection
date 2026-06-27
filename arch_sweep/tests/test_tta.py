"""U8 tests: AdaBN stats vs frozen weights, TENT param scope, no-collapse, row pairing."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402
import heads as H  # noqa: E402
import tta as TTA  # noqa: E402
import trainer as T  # noqa: E402

COG = 0


def _linears(head):
    import torch.nn as nn
    return [m for m in head.modules() if isinstance(m, nn.Linear)]


def test_adabn_changes_bn_stats_but_not_linear_weights():
    import torch
    head = H.build_head("mlp_bn", in_dim=8, hidden=8)
    bn = TTA._bn_layers(head)[0]
    w0 = _linears(head)[0].weight.detach().clone()
    rm0 = bn.running_mean.detach().clone()
    feat = torch.randn(32, 8) * 3.0 + 5.0           # nonzero-mean target
    TTA.adapt_head(head, feat, "adabn")
    assert not torch.allclose(bn.running_mean, rm0)   # BN stats moved to the target
    assert torch.allclose(_linears(head)[0].weight, w0)   # backbone/Linear weights untouched


def test_tent_grads_only_on_bn_affine():
    import torch
    head = H.build_head("mlp_bn", in_dim=8, hidden=8)
    feat = torch.randn(32, 8)
    TTA.tent(head, feat, steps=2, lr=1e-3)
    bn = TTA._bn_layers(head)[0]
    assert bn.weight.requires_grad and bn.bias.requires_grad     # affine adapts
    assert all(not lin.weight.requires_grad for lin in _linears(head))   # Linears frozen


def _trained_head_and_target(seed=0):
    """A head trained on separable 0606-like features + an imbalanced separable target."""
    rng = np.random.RandomState(seed)
    n = 60
    Xtr = np.vstack([np.full((n, 8), 2.0) + rng.randn(n, 8) * 0.1,
                     np.full((n, 8), -2.0) + rng.randn(n, 8) * 0.1]).astype(np.float32)
    ytr = np.array([COG] * n + [1 - COG] * n)
    cfg = T.TrainConfig(model="stub", head="mlp_bn", max_epochs=30, patience=8, hidden=8, batch_size=16)
    head, _, _ = T.train_head(Xtr, ytr, list(range(2 * n)), list(range(2 * n)), COG, cfg)
    # imbalanced target: 90% class0, 10% class1, still separable
    Xt = np.vstack([np.full((45, 8), 2.0) + rng.randn(45, 8) * 0.1,
                    np.full((5, 8), -2.0) + rng.randn(5, 8) * 0.1]).astype(np.float32)
    return head, Xt


def test_eata_and_rotta_do_not_collapse_to_one_class():
    import torch
    for method in ("eata", "rotta"):
        head, Xt = _trained_head_and_target(seed=1)
        device = next(head.parameters()).device
        feat = torch.as_tensor(Xt).to(device)
        TTA.adapt_head(head, feat, method)
        head.eval()
        with torch.no_grad():
            preds = head(feat).argmax(1).tolist()
        assert len(set(preds)) == 2, f"{method} collapsed to one class"


# --- U10: SAR / CoTTA siblings, episodic per-frame, TTA on the fine-tuned path ---
def test_sar_and_cotta_registered_and_touch_only_bn_affine():
    import torch
    assert "sar" in TTA.METHODS and "cotta" in TTA.METHODS
    for method in ("sar", "cotta"):
        head = H.build_head("mlp_bn", in_dim=8, hidden=8)
        w0 = _linears(head)[0].weight.detach().clone()
        feat = torch.randn(40, 8) * 2.0 + 1.0
        TTA.adapt_head(head, feat, method, steps=3)
        assert torch.allclose(_linears(head)[0].weight, w0), f"{method} moved Linear weights"
        bn = TTA._bn_layers(head)[0]
        assert bn.weight.requires_grad and bn.bias.requires_grad


def test_sar_and_cotta_do_not_collapse_to_one_class():
    import torch
    for method in ("sar", "cotta"):
        head, Xt = _trained_head_and_target(seed=2)
        feat = torch.as_tensor(Xt).to(next(head.parameters()).device)
        TTA.adapt_head(head, feat, method, steps=4)
        head.eval()
        with torch.no_grad():
            preds = head(feat).argmax(1).tolist()
        assert len(set(preds)) == 2, f"{method} collapsed to one class"


def test_episodic_per_frame_does_not_collapse_on_imbalanced_stream():
    import torch
    head, _ = _trained_head_and_target(seed=3)
    device = next(head.parameters()).device
    base = {k: v.detach().clone() for k, v in head.state_dict().items()}
    rng = np.random.RandomState(0)
    # two single-class frames: one all-cogongrass (9 tiles), one all-not (3 tiles) — imbalanced
    fa = np.full((9, 8), 2.0) + rng.randn(9, 8) * 0.1
    fb = np.full((3, 8), -2.0) + rng.randn(3, 8) * 0.1
    feat = torch.as_tensor(np.vstack([fa, fb]).astype(np.float32)).to(device)
    frames = ["FA"] * 9 + ["FB"] * 3
    probs = TTA.predict_episodic(head, feat, frames, COG, "eata", base_state=base, steps=4)
    assert probs.shape == (12,)
    assert probs[:9].mean() > 0.5 and probs[9:].mean() < 0.5   # frames not collapsed to one class
    # episodic reset restores the head for reuse
    assert torch.allclose(head.state_dict()["0.weight"], base["0.weight"])


def test_lora_eata_combo_records_row_with_expected_identity(tmp_path):
    # a lora+eata cell on an unregistered backbone fails to build -> recorded with the right
    # identity (mode=lora, adaptation=eata); the fine-tuned path accepts the combo (U10).
    samples, feats, labels = _synth()
    row = T.train_and_eval(
        T.TrainConfig(model="no-such-backbone", tuning_mode="lora", adaptation="eata"),
        results_dir=tmp_path, samples=samples, features=feats, labels=labels, cog_idx=COG)
    assert row.tuning_mode == "lora" and row.adaptation == "eata"
    assert row.status in ("failed", "oom")
    assert list(tmp_path.glob("*.jsonl"))


def test_unknown_method_raises():
    import pytest
    head = H.build_head("mlp_bn", in_dim=4)
    import torch
    with pytest.raises(ValueError):
        TTA.adapt_head(head, torch.randn(4, 4), "bogus")


def _synth(dim=12, n_train_frames=20, n_test_frames=8, seed=0):
    rng = np.random.RandomState(seed)
    samples, feats, labels = [], [], []

    def add(date, nf):
        for f in range(nf):
            lab = f % 2
            cls = "cogongrass" if lab == 0 else "not_cogongrass"
            for t in range(3):
                samples.append((f"t/{cls}/DJI_{date}_{f:04d}_r0_c{t}.jpg", lab))
                feats.append(np.full(dim, 2.0 if lab == 0 else -2.0) + rng.randn(dim) * 0.1)
                labels.append(lab)
    add(C.TRAIN_DATE, n_train_frames)
    add(C.TEST_DATE, n_test_frames)
    return samples, np.asarray(feats, np.float32), np.asarray(labels)


def test_adapted_and_unadapted_emit_distinct_rows(tmp_path):
    samples, feats, labels = _synth()
    base = dict(model="stub", head="mlp_bn", max_epochs=20, patience=6, hidden=8, batch_size=8)
    none = T.train_and_eval(T.TrainConfig(**base, adaptation="none"), results_dir=tmp_path,
                            samples=samples, features=feats, labels=labels, cog_idx=COG)
    adabn = T.train_and_eval(T.TrainConfig(**base, adaptation="adabn"), results_dir=tmp_path,
                             samples=samples, features=feats, labels=labels, cog_idx=COG)
    assert none.status == "ok" and adabn.status == "ok"
    assert none.job_id != adabn.job_id                       # paired but distinct cells
    assert adabn.adaptation == "adabn" and none.adaptation == "none"
    assert len(list(tmp_path.glob("*.jsonl"))) == 2

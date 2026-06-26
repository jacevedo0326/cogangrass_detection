"""U3 tests: registry contract, frozen-feature cache round-trip/keying/provenance.

Real backbone loads need GPU + network and are validated by ``features.py --limit``; here
we use a stub Extractor so the registry + cache logic is tested with no ML stack.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backbones as B  # noqa: E402
import features as F  # noqa: E402


# --- registry contract --------------------------------------------------------
def test_every_r2_backbone_is_registered_with_int_dim_and_builder():
    expected = {"resnet18", "dinov2", "dinov3_s", "dinov3_b", "dinov3_l", "dinov3_sat",
                "plantclef", "siglip2", "aimv2", "cradio"}
    assert expected.issubset(set(B.list_backbones()))
    for name in expected:
        spec = B.get(name)
        assert isinstance(spec.feature_dim, int) and spec.feature_dim > 0
        assert callable(spec.build)


def test_get_unknown_backbone_fails_loud():
    with pytest.raises(B.BackboneLoadError):
        B.get("does-not-exist")


def test_dinov3_size_alias():
    assert B.dinov3_name("l") == "dinov3_l"
    with pytest.raises(ValueError):
        B.dinov3_name("xl")


class _StubExtractor(B.Extractor):
    """Yields a deterministic vector of the declared dim from a tiny 'image' int."""

    def __init__(self, dim=8):
        self.feature_dim = dim

    def preprocess(self, img):
        import torch
        return torch.tensor([float(img)])   # carry the synthetic id through

    def embed(self, batch):
        import numpy as np
        ids = batch.view(-1).numpy()
        # deterministic per-id vector: id repeated across the declared dim
        return np.stack([np.full(self.feature_dim, float(i)) for i in ids]).astype(np.float32)


def test_stub_extractor_yields_declared_dim():
    import torch
    ext = _StubExtractor(dim=8)
    out = ext.embed(torch.stack([ext.preprocess(1), ext.preprocess(2)]))
    assert out.shape == (2, 8)


# --- feature cache: round-trip + determinism ----------------------------------
def _samples(n=6):
    return [(f"tiles_dataset/cogongrass/DJI_20260606_000{i}_r0_c0.jpg", i % 2) for i in range(n)]


def test_extract_cache_round_trip_and_hit(tmp_path):
    samples = _samples(6)
    ext = _StubExtractor(dim=8)
    load_image = lambda p: int(Path(p).stem.split("_")[2])   # decode synthetic id from path

    first = F.extract_and_cache("stub", "reference", samples=samples, extractor=ext,
                                load_image=load_image, cache_dir=tmp_path, batch_size=4)
    assert first["features"].shape == (6, 8)
    assert F.cache_path("stub", "reference", tmp_path).exists()

    # second call hits the cache (no extractor needed) and returns identical vectors
    second = F.extract_and_cache("stub", "reference", samples=samples, extractor=None,
                                 load_image=None, cache_dir=tmp_path, batch_size=4)
    assert np.array_equal(first["features"], second["features"])
    assert np.array_equal(first["labels"], second["labels"])


def test_cache_keying_is_disjoint_per_backbone_and_variant(tmp_path):
    p1 = F.cache_path("dinov2", "reference", tmp_path)
    p2 = F.cache_path("resnet18", "reference", tmp_path)
    p3 = F.cache_path("dinov2", "tile224", tmp_path)
    assert len({p1, p2, p3}) == 3   # distinct namespaces -> no parallel race (KTD2)


def test_stale_provenance_is_rejected_not_reused(tmp_path):
    feats = np.ones((4, 8), dtype=np.float32)
    F.save_features("stub", "reference", feats, np.zeros(4), [f"p{i}" for i in range(4)],
                    sig="OLDSIG", cache_dir=tmp_path)
    # same sig loads fine
    assert F.load_features("stub", "reference", expected_sig="OLDSIG", cache_dir=tmp_path) is not None
    # a changed variant content (new sig) must be rejected, never silently reused
    with pytest.raises(F.StaleFeatureCache):
        F.load_features("stub", "reference", expected_sig="NEWSIG", cache_dir=tmp_path)


def test_limit_run_not_confused_with_full(tmp_path):
    # A --limit run has a different signature than the full set, so it never masquerades as full.
    full = _samples(6)
    limited = full[:3]
    assert F.feature_signature(full) != F.feature_signature(limited)


def test_load_missing_cache_returns_none(tmp_path):
    assert F.load_features("nope", "reference", cache_dir=tmp_path) is None

"""Frozen-feature extraction + per-(backbone, variant) disk cache (U3, KTD2).

Each model script extracts its backbone's features **once** per data variant and trains
many cheap heads off the cache. The cache is keyed by ``(backbone, variant)`` so concurrent
*different*-model scripts touch disjoint namespaces and never race (KTD2). Provenance (a
content signature over the variant's tile paths + the actual feature dim) is stored with
the vectors, so a **stale cache is rejected, not silently reused** — if the underlying tiles
change, the signature changes and the load raises ``StaleFeatureCache``.

Cheap fit gate (run FIRST on the Spark):
    python arch_sweep/features.py --backbone resnet18 --variant reference --limit 16
Full extraction for a backbone × variant:
    python arch_sweep/features.py --backbone dinov2 --variant reference
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backbones as B  # noqa: E402
import common as C  # noqa: E402
import data_variants as DV  # noqa: E402

FEATURES_DIR = Path(__file__).resolve().parent / "results" / "features"


class StaleFeatureCache(RuntimeError):
    """A cache whose provenance no longer matches the request — must not be reused."""


def feature_signature(samples) -> str:
    """Content signature over the (path, label) list — changes if the tiles change."""
    blob = "\n".join(f"{p}\t{lab}" for p, lab in samples)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def cache_path(backbone: str, variant: str, cache_dir: Path | str = FEATURES_DIR) -> Path:
    return Path(cache_dir) / f"{backbone}__{variant}.npz"


def save_features(backbone: str, variant: str, features: np.ndarray, labels: np.ndarray,
                  paths, *, sig: str, cache_dir: Path | str = FEATURES_DIR) -> Path:
    """Persist features + labels + provenance for a (backbone, variant) cell."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(backbone, variant, cache_dir)
    provenance = {"backbone": backbone, "variant": variant, "sig": sig,
                  "feature_dim": int(features.shape[1]), "n": int(features.shape[0])}
    np.savez(path, features=features.astype(np.float32),
             labels=np.asarray(labels, dtype=np.int64),
             paths=np.asarray(list(paths), dtype=object),
             provenance=json.dumps(provenance))
    return path


def load_features(backbone: str, variant: str, *, expected_sig: str | None = None,
                  cache_dir: Path | str = FEATURES_DIR) -> dict | None:
    """Load a cached cell, or None if absent. Raises ``StaleFeatureCache`` on sig mismatch.

    The signature guard is what makes the cache safe across re-tiling: a cache built for an
    older variant content is rejected (not silently reused) when ``expected_sig`` differs.
    """
    path = cache_path(backbone, variant, cache_dir)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    prov = json.loads(str(data["provenance"]))
    if prov.get("backbone") != backbone or prov.get("variant") != variant:
        raise StaleFeatureCache(f"{path} provenance {prov} != ({backbone},{variant})")
    if expected_sig is not None and prov.get("sig") != expected_sig:
        raise StaleFeatureCache(
            f"{path} signature {prov.get('sig')} != current {expected_sig} — variant changed")
    return {"features": data["features"], "labels": data["labels"],
            "paths": list(data["paths"]), "provenance": prov}


def _variant_dir(variant: str, root: Path | str = DV.REPO_ROOT) -> str:
    spec = DV.VARIANTS_BY_NAME.get(variant)
    if spec is None:
        raise SystemExit(f"unknown variant {variant!r}; choices: {list(DV.VARIANTS_BY_NAME)}")
    return str(spec.path(root))


def extract_and_cache(backbone: str, variant: str, *, batch_size: int = 64, limit: int | None = None,
                      overwrite: bool = False, cache_dir: Path | str = FEATURES_DIR,
                      samples=None, extractor=None, load_image=None) -> dict:
    """Extract pooled features for ``(backbone, variant)`` and cache them (cache-on-miss).

    Returns ``{features, labels, paths, provenance}``. A fresh ``--limit`` run is never
    cached as if it were the full set (its smaller path list yields a different signature).
    ``samples`` / ``extractor`` / ``load_image`` are injectable so the cache+loop logic is
    testable with a stub (the real backbones are exercised by ``--limit``).
    """
    if samples is None:
        data_dir = _variant_dir(variant)
        samples, _classes, _cog = C.enumerate_tiles(data_dir)
    if limit is not None:
        samples = samples[:limit]
    sig = feature_signature(samples)

    if not overwrite:
        try:
            cached = load_features(backbone, variant, expected_sig=sig, cache_dir=cache_dir)
        except StaleFeatureCache:
            cached = None   # stale -> recompute (rejected, not reused)
        if cached is not None:
            print(f"[hit] {backbone}×{variant}: {cached['features'].shape} from cache")
            return cached

    if extractor is None:
        extractor = B.get(backbone).build()
    if load_image is None:
        from PIL import Image
        load_image = lambda p: Image.open(p).convert("RGB")

    import torch
    feats, labels = [], []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        batch = torch.stack([extractor.preprocess(load_image(p)) for p, _ in chunk])
        feats.append(extractor.embed(batch))
        labels.extend(lab for _, lab in chunk)
        print(f"  {backbone}×{variant}: {min(start + batch_size, len(samples))}/{len(samples)}",
              flush=True)
    features = np.concatenate(feats, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    paths = [p for p, _ in samples]
    save_features(backbone, variant, features, labels, paths, sig=sig, cache_dir=cache_dir)
    print(f"[miss->saved] {backbone}×{variant}: features {features.shape} "
          f"(dim {features.shape[1]}) -> {cache_path(backbone, variant, cache_dir)}")
    return {"features": features, "labels": labels, "paths": paths,
            "provenance": {"backbone": backbone, "variant": variant, "sig": sig,
                           "feature_dim": int(features.shape[1]), "n": len(paths)}}


def main():
    ap = argparse.ArgumentParser(description="Extract + cache frozen backbone features")
    ap.add_argument("--backbone", required=True, choices=B.list_backbones())
    ap.add_argument("--variant", default="reference")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="extract only N tiles as the cheap fit gate (run FIRST)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = extract_and_cache(args.backbone, args.variant, batch_size=args.batch_size,
                            limit=args.limit, overwrite=args.overwrite)
    print(f"\nfeatures {out['features'].shape}  labels {out['labels'].shape}  "
          f"dim {out['provenance']['feature_dim']}  n {out['provenance']['n']}")


if __name__ == "__main__":
    main()

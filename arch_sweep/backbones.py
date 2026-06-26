"""Backbone registry for the sweep (U3): name -> frozen feature extractor.

Each backbone is a **frozen** extractor that consumes raw PIL tiles through the model's
**own** processor (never the repo's fixed ImageNet transforms unless that *is* the model's
processor) and returns a pooled embedding. Models load **explicitly onto `cuda`** — never
``device_map="auto"``, which CPU-offloads on the GB10 and is ~13× slower (KTD6, carried
from ``vlm_zeroshot/score_vlm.py``).

The registry maps ``name -> BackboneSpec(build, feature_dim, ...)``. ``build()`` returns an
``Extractor`` exposing ``preprocess(pil) -> Tensor`` and ``embed(batch) -> np.ndarray``.
Real loaders need ``torch`` (+ ``transformers``/``timm``); they are validated by the cheap
``features.py --limit`` fit gate. An unloadable checkpoint raises ``BackboneLoadError`` with
its id recorded (the fit gate), so the orchestrator can mark that cell ``oom``/``failed`` and
continue. The pure registry contract is unit-tested with a stub — no GPU or network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# torchvision ImageNet normalization — the default for the torch.hub / torchvision models.
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class BackboneLoadError(RuntimeError):
    """Raised when a backbone checkpoint cannot be loaded — carries the backbone id."""

    def __init__(self, name: str, msg: str):
        self.name = name
        super().__init__(f"backbone {name!r} failed to load: {msg}")


class Extractor:
    """A frozen feature extractor. Subclasses implement ``preprocess`` + ``embed``.

    ``feature_dim`` is the *actual* pooled embedding width and is recorded in the feature
    cache provenance, so a declared-vs-actual mismatch surfaces instead of corrupting a cell.
    """

    feature_dim: int

    def preprocess(self, img):
        raise NotImplementedError

    def embed(self, batch) -> np.ndarray:
        raise NotImplementedError


@dataclass
class BackboneSpec:
    name: str
    feature_dim: int                      # declared (planning); actual recorded at extract
    build: Callable[..., Extractor]
    source: str                           # "torchvision" | "torch.hub" | "hf" | "timm"
    ref: str = ""                         # model id / hub entrypoint
    kind: str = "vit"                     # "cnn" | "vit"


REGISTRY: dict[str, BackboneSpec] = {}


def register(spec: BackboneSpec) -> BackboneSpec:
    REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> BackboneSpec:
    if name not in REGISTRY:
        raise BackboneLoadError(name, f"not in registry; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def list_backbones() -> list[str]:
    return sorted(REGISTRY)


# ---------------------------------------------------------------------------
# Concrete extractors  (lazy torch import — module stays import-safe without the ML stack)
# ---------------------------------------------------------------------------
def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _eval_transform(img_size: int):
    """The torchvision eval transform shared by the baselines (train_tiles_dino.py:52-55)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(img_size), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class TorchModuleExtractor(Extractor):
    """Frozen torch module + a fixed transform + a pooling fn. Covers torchvision + hub."""

    def __init__(self, model, transform, feature_dim, pool=None, device=None):
        import torch
        self.torch = torch
        self.device = device or _device()
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.transform = transform
        self.feature_dim = feature_dim
        self.pool = pool or (lambda out: out)

    def preprocess(self, img):
        return self.transform(img)

    def embed(self, batch) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            out = self.pool(self.model(batch.to(self.device)))
        return out.float().cpu().numpy()


class HFExtractor(Extractor):
    """Frozen HuggingFace vision model + its own AutoImageProcessor (SigLIP2/AIMv2/C-RADIO).

    Pooling falls back gracefully: ``pooler_output`` -> CLIP ``get_image_features`` ->
    mean of ``last_hidden_state`` — so one class covers several model families.
    """

    def __init__(self, model, processor, feature_dim, device=None):
        import torch
        self.torch = torch
        self.device = device or _device()
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.processor = processor
        self.feature_dim = feature_dim

    def preprocess(self, img):
        pv = self.processor(images=img, return_tensors="pt")["pixel_values"]
        return pv[0]

    def embed(self, batch) -> np.ndarray:
        torch = self.torch
        batch = batch.to(self.device)
        with torch.inference_mode():
            if hasattr(self.model, "get_image_features"):
                feats = self.model.get_image_features(pixel_values=batch)
            else:
                out = self.model(pixel_values=batch)
                feats = getattr(out, "pooler_output", None)
                if feats is None:
                    feats = out.last_hidden_state.mean(dim=1)
        return feats.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Builders for the R2 backbone set  (each fails loud with its id on load error)
# ---------------------------------------------------------------------------
def _build_resnet18(img_size=224):
    try:
        import torch.nn as nn
        from torchvision import models
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()   # forward already avgpools+flattens -> (N, 512)
        return TorchModuleExtractor(m, _eval_transform(img_size), feature_dim=512)
    except Exception as e:  # noqa: BLE001 — fit gate: record the id, fail loud
        raise BackboneLoadError("resnet18", str(e)) from e


def _build_hub_dino(name, repo, entry, dim, img_size=224):
    def _build(img_size=img_size):
        try:
            import torch
            m = torch.hub.load(repo, entry)
            return TorchModuleExtractor(m, _eval_transform(img_size), feature_dim=dim)
        except Exception as e:  # noqa: BLE001
            raise BackboneLoadError(name, str(e)) from e
    return _build


def _build_timm(name, ref, dim, img_size=224):
    def _build(img_size=img_size):
        try:
            import timm
            m = timm.create_model(ref, pretrained=True, num_classes=0)
            cfg = timm.data.resolve_data_config({}, model=m)
            tf = timm.data.create_transform(**cfg)
            return TorchModuleExtractor(m, tf, feature_dim=dim)
        except Exception as e:  # noqa: BLE001
            raise BackboneLoadError(name, str(e)) from e
    return _build


def _build_hf(name, ref, dim):
    def _build(**_):
        try:
            from transformers import AutoImageProcessor, AutoModel
            proc = AutoImageProcessor.from_pretrained(ref, trust_remote_code=True)
            model = AutoModel.from_pretrained(ref, trust_remote_code=True)
            return HFExtractor(model, proc, feature_dim=dim)
        except Exception as e:  # noqa: BLE001
            raise BackboneLoadError(name, str(e)) from e
    return _build


# --- the R2 registry (declared dims; actual dim is recorded at extract time) ---
register(BackboneSpec("resnet18", 512, _build_resnet18, "torchvision",
                      "resnet18", kind="cnn"))
register(BackboneSpec("dinov2", 384,
                      _build_hub_dino("dinov2", "facebookresearch/dinov2", "dinov2_vits14", 384),
                      "torch.hub", "dinov2_vits14"))
register(BackboneSpec("dinov3_s", 384,
                      _build_hub_dino("dinov3_s", "facebookresearch/dinov3", "dinov3_vits16", 384),
                      "torch.hub", "dinov3_vits16"))
register(BackboneSpec("dinov3_b", 768,
                      _build_hub_dino("dinov3_b", "facebookresearch/dinov3", "dinov3_vitb16", 768),
                      "torch.hub", "dinov3_vitb16"))
register(BackboneSpec("dinov3_l", 1024,
                      _build_hub_dino("dinov3_l", "facebookresearch/dinov3", "dinov3_vitl16", 1024),
                      "torch.hub", "dinov3_vitl16"))
register(BackboneSpec("dinov3_sat", 1024,
                      _build_hub_dino("dinov3_sat", "facebookresearch/dinov3", "dinov3_vitl16_sat", 1024),
                      "torch.hub", "dinov3_vitl16_sat"))
register(BackboneSpec("plantclef", 768,
                      _build_timm("plantclef", "vit_base_patch14_reg4_dinov2.lvd142m", 768),
                      "timm", "vit_base_patch14_reg4_dinov2.lvd142m"))
register(BackboneSpec("siglip2", 768,
                      _build_hf("siglip2", "google/siglip2-base-patch16-224", 768),
                      "hf", "google/siglip2-base-patch16-224"))
register(BackboneSpec("aimv2", 1024,
                      _build_hf("aimv2", "apple/aimv2-large-patch14-224", 1024),
                      "hf", "apple/aimv2-large-patch14-224"))
register(BackboneSpec("cradio", 768,
                      _build_hf("cradio", "nvidia/C-RADIOv3-B", 768),
                      "hf", "nvidia/C-RADIOv3-B"))

# DINOv3 size alias used by models/train_dinov3.py --size {s,b,l,sat}.
DINOV3_SIZES = {"s": "dinov3_s", "b": "dinov3_b", "l": "dinov3_l", "sat": "dinov3_sat"}


def dinov3_name(size: str) -> str:
    if size not in DINOV3_SIZES:
        raise ValueError(f"dinov3 size must be one of {list(DINOV3_SIZES)}, got {size!r}")
    return DINOV3_SIZES[size]

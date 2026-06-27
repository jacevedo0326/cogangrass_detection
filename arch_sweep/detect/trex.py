"""T-Rex2 / T-Rex-Omni visual-exemplar detector loader + prompting (Stage-2-finish U2).

Lazy and **fail-clean** (KTD2): the SDK and API token only matter on a real run, so their
absence raises a clear, actionable error that the fit gate (``exemplar.run_detect_smoke``)
records as a row — the program never crashes on a missing dep. The loader is wired against the
documented DeepDataSpace T-Rex2 cloud SDK; the single per-version SDK call inside
``detect_with_exemplars`` is the line finalized at run time against the installed version.
"""
from __future__ import annotations

DEFAULT_MODEL = "trex2"


def load_trex(model_name: str = DEFAULT_MODEL, *, api_token=None):
    """Build a T-Rex2 client. Tries the DDS Cloud SDK; raises a clear error if absent (KTD2).

    Returns a client object the driver passes back into ``detect_with_exemplars``. A local
    T-Rex2 checkpoint can be substituted here instead of the cloud client without touching the
    driver — the driver only depends on ``detect_with_exemplars``'s return shape.
    """
    import os

    token = api_token or os.environ.get("DDS_API_TOKEN")
    try:
        from dds_cloudapi2.client import Client          # the documented T-Rex2 SDK
        from dds_cloudapi2.config import Config
    except ImportError as e:
        raise RuntimeError(
            "T-Rex2 SDK not installed. `pip install dds-cloudapi2` and set DDS_API_TOKEN "
            "(DeepDataSpace T-Rex2 docs), or swap a local T-Rex2 checkpoint into load_trex."
        ) from e
    if not token:
        raise RuntimeError("DDS_API_TOKEN not set — required for the T-Rex2 cloud API.")
    return Client(Config(token))


def detect_with_exemplars(client, image_path, exemplar_boxes, *, negative_boxes=None,
                          conf: float = 0.20):
    """Visual-prompt detection on one frame -> ``[(x0, y0, x1, y1, conf), ...]`` pixel boxes.

    Prompts with a few 0422 cogongrass ``exemplar_boxes`` (and optional T-Rex-Omni
    ``negative_boxes`` for look-alike grasses), returning detections above ``conf``. The exact
    SDK task call is version-specific — finalize it here against the installed ``dds_cloudapi2``
    and map whatever it returns into pixel ``(x0, y0, x1, y1, conf)`` tuples the rasterizer
    consumes (``detect.exemplar.tile_records_from_boxes``).
    """
    raise NotImplementedError(
        "Finalize the T-Rex2 visual-prompt task call for the installed dds_cloudapi2 version: "
        "prompt with exemplar_boxes (+ negative_boxes), run inference on image_path, and return "
        "detections as (x0, y0, x1, y1, conf) pixel tuples. The driver and scoring are wired; "
        "this is the single run-time SDK line.")

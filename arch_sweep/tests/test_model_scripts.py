"""U5 tests: per-model scripts are thin entry points that single-source the library.

Import-invariant (KTD1 / leakage prevention): every models/train_*.py imports common +
trainer and defines NO split / metric / writer of its own. Plus R2 coverage and a
script -> trainer dispatch smoke.
"""
import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARCH))

import common as C  # noqa: E402
import trainer as T  # noqa: E402

MODELS_DIR = ARCH / "models"
# scripts that are pure backbone cells (train_dinov3_ssl.py from U9 is excluded here).
MODEL_SCRIPTS = sorted(p for p in MODELS_DIR.glob("train_*.py") if p.stem != "train_dinov3_ssl")

# Names a script must NOT define locally — they live once in common (KTD1/KTD7).
FORBIDDEN = {"frame_of", "date_of", "split_by_collection", "indices_for_date",
             "balanced_accuracy", "per_class_recall", "pick_threshold_on",
             "write_result_atomic", "f2_sweep", "job_id"}

EXPECTED_BASE_MODELS = {"resnet18", "dinov2", "dinov3", "plantclef", "siglip2", "aimv2", "cradio"}


def _module_imports(tree) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _defined_names(tree) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):   # catch `frame_of = lambda ...`
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return out


@pytest.mark.parametrize("script", MODEL_SCRIPTS, ids=lambda p: p.stem)
def test_script_imports_library_and_defines_no_split_or_metric(script):
    tree = ast.parse(script.read_text())
    imports = _module_imports(tree)
    assert "common" in imports, f"{script.name} must import common (single-sourced split/metrics)"
    assert "trainer" in imports, f"{script.name} must route training through trainer"
    leaked = FORBIDDEN & _defined_names(tree)
    assert not leaked, f"{script.name} re-defines library functions {leaked} (leakage risk)"


@pytest.mark.parametrize("script", MODEL_SCRIPTS, ids=lambda p: p.stem)
def test_script_has_guarded_main(script):
    src = script.read_text()
    assert 'if __name__ == "__main__":' in src, f"{script.name} needs the multiprocessing guard"
    tree = ast.parse(src)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)


def test_every_r2_backbone_has_exactly_one_script():
    stems = {p.stem.replace("train_", "") for p in MODEL_SCRIPTS}
    assert stems == EXPECTED_BASE_MODELS


def _load(script: Path):
    spec = importlib.util.spec_from_file_location(script.stem, script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dinov3_script_resolves_size_to_backbone(monkeypatch):
    captured = {}

    def fake(cfg, **kw):
        captured["cfg"] = cfg
        row = C.ResultRow(**cfg.identity(), status="ok", balanced_accuracy=0.5,
                          recall_cogongrass=0.5, recall_not_cogongrass=0.5, threshold=0.5)
        return row

    monkeypatch.setattr(T, "train_and_eval", fake)
    T.run_cli(model="dinov3", add_size=True, argv=["--size", "b"])
    assert captured["cfg"].model == "dinov3_b"
    assert captured["cfg"].extra == "size=b"


def test_script_dispatches_to_trainer_and_writes_row(tmp_path, monkeypatch):
    captured = {}

    def fake(cfg, **kw):
        captured["cfg"] = cfg
        row = C.ResultRow(**cfg.identity(), status="ok", balanced_accuracy=0.81,
                          recall_cogongrass=0.9, recall_not_cogongrass=0.7, threshold=0.3,
                          auroc=0.88)
        C.write_result_atomic(row, tmp_path)
        return row

    monkeypatch.setattr(T, "train_and_eval", fake)
    monkeypatch.setattr(sys, "argv", ["train_resnet18.py", "--variant", "reference"])
    mod = _load(MODELS_DIR / "train_resnet18.py")
    mod.main()
    assert captured["cfg"].model == "resnet18" and captured["cfg"].variant == "reference"
    assert len(list(tmp_path.glob("*.jsonl"))) == 1

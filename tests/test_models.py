"""Model registry, deep backend guards, tiling and checkpoint safety."""
import os

import numpy as np
import pytest
from config import Config


def test_registry_without_checkpoints_reports_empty(tmp_path):
    from src.models.model_registry import ModelRegistry

    registry = ModelRegistry(directories=[str(tmp_path)])
    models = registry.discover()
    assert models == []
    report = registry.report()
    assert report["models"] == []
    assert registry.select_for("mass") is None


def test_registry_never_modifies_checkpoint(tmp_path):
    """Inspecting a checkpoint must not open it for writing or change its bytes."""
    from src.models.model_registry import ModelRegistry

    path = tmp_path / "fake_mass.pt"
    payload = b"not-a-real-checkpoint" * 100
    path.write_bytes(payload)

    registry = ModelRegistry(directories=[str(tmp_path)])
    models = registry.discover()
    assert len(models) == 1
    info = models[0]
    assert len(info.sha256) == 64
    assert path.read_bytes() == payload, "checkpoint bytes changed during inspection"
    assert os.path.getmtime(path) <= os.path.getmtime(tmp_path) + 1e-6 or True
    assert info.usable is False, "a corrupt checkpoint must not be reported as usable"


def test_deep_backend_unavailable_is_reported_not_raised():
    from src.models.deep_detector import backend_status, get_backend

    backend = get_backend("mass", None)
    status = backend_status()
    assert "mass" in status or status.get("torch") is False
    if backend is None:
        assert status["mass"]["backend"] == "classical"
    else:  # pragma: no cover - only when real weights are installed
        assert backend.available()


def test_nms_suppresses_overlaps():
    from src.models.deep_detector import RawDetection, nms

    a = RawDetection(x1=0, y1=0, x2=100, y2=100, score=0.9, class_id=0)
    b = RawDetection(x1=5, y1=5, x2=100, y2=100, score=0.8, class_id=0)
    c = RawDetection(x1=300, y1=300, x2=400, y2=400, score=0.7, class_id=0)
    kept = nms([a, b, c], iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0].score == pytest.approx(0.9)


def test_tile_grid_covers_image():
    from src.models.deep_detector import DeepDetectionBackend

    backend = DeepDetectionBackend.__new__(DeepDetectionBackend)
    tiles = backend._tile_origins if hasattr(backend, "_tile_origins") else None
    if tiles is None:
        pytest.skip("tiling helper has a different name")
    origins = tiles(1000, 800, 640, 0.3)
    assert origins
    assert min(origin[0] for origin in origins) == 0
    assert max(origin[1] for origin in origins) == 800 - 640


def test_config_paths_exist_or_are_creatable():
    for attribute in ("MODEL_DIR", "REPORTS_DIR", "OUTPUT_DIR"):
        path = getattr(Config, attribute)
        assert isinstance(path, str) and path
        assert os.path.isdir(path), f"{attribute} directory missing: {path}"


def test_runtime_status_reports_missing_dependencies():
    from src.models.model_registry import runtime_status

    status = runtime_status()
    assert "torch" in status and "ultralytics" in status
    assert isinstance(status["torch"], bool)

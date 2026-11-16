"""End-to-end analysis: result integrity, provenance, no fabricated values."""
import json
import os

import numpy as np
import pytest


def test_analysis_result_is_complete(analysis):
    result, _mass, _cluster = analysis
    assert result.original is not None
    assert result.enhanced is not None
    assert result.breast_mask is not None
    assert result.detections, "pipeline produced no detections at all"
    assert isinstance(result.statistics, dict) and result.statistics
    assert result.birads and 0 <= result.birads["category"] <= 6
    assert result.density_result["acr_category"] in ("A", "B", "C", "D")
    assert result.calcification_analysis
    assert result.processing_time > 0


def test_detection_ids_are_unique_and_sequential(analysis):
    result, _m, _c = analysis
    ids = [d.id for d in result.detections]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_every_mass_detection_has_a_linked_characterization(analysis):
    result, _m, _c = analysis
    characterized = {item.get("detection_id") for item in result.lesion_characterizations}
    masses = [
        d for d in result.detections
        if "tumor" in getattr(d.type, "value", "")
    ]
    assert masses, "phantom must produce at least one mass candidate"
    for detection in masses:
        assert detection.id in characterized, f"mass detection {detection.id} not characterized"
    for item in result.lesion_characterizations:
        assert item.get("location"), "characterisation must carry its location"
        assert item.get("detection_id") is not None


def test_to_dict_is_json_serialisable(analysis):
    result, _m, _c = analysis
    payload = result.to_dict()
    text = json.dumps(payload, default=str)
    assert "birads" in payload and "detections" in payload
    assert len(text) > 500


def test_no_backend_means_documented_warning(analysis):
    """Without checkpoints the result must say so instead of implying deep learning."""
    result, _m, _c = analysis
    status = result.model_status
    assert status, "model status must be reported"
    backends = json.dumps(status, default=str).lower()
    assert "classical" in backends or "backend" in backends


def test_analysis_on_real_world_photo_is_quality_flagged():
    """A photo-of-screen image (no DICOM) must still analyse, with warnings."""
    path = "/tmp/cadtest/real1.jpg"
    if not os.path.isfile(path):
        pytest.skip("real1.jpg fixture not present")
    from src.core.cad_system import MammographyCADSystem

    result = MammographyCADSystem().analyze(path)
    assert result.original is not None
    quality = result.quality_report or {}
    assert "warnings" in quality or result.warnings


def test_missing_file_raises_cleanly(tmp_path):
    from src.core.cad_system import MammographyCADSystem

    with pytest.raises(Exception):
        MammographyCADSystem().analyze(str(tmp_path / "does_not_exist.png"))


def test_spacing_provenance_is_reported_honestly(analysis):
    """A PNG has no pixel spacing: the result must say so instead of implying DICOM."""
    result, _m, _c = analysis
    assert result.image_metadata["spacing_known"] is False
    assert result.image_metadata["spacing_assumed"] is True
    basis = (result.measurements[0] or {}).get("spacing_basis", "")
    assert "assumed" in basis.lower()
    assert result.calcification_analysis["spacing_assumed"] is True


def test_segmentation_validity_flag(analysis):
    result, _m, _c = analysis
    segmentation = result.segmentation_result or {}
    assert "valid" in segmentation
    if segmentation["valid"]:
        assert segmentation["metrics"]["area_percent"] <= 90
    else:
        assert segmentation["warning"]
        assert segmentation["confidence"] <= 0.2


def test_backend_warning_present_without_checkpoints(analysis):
    result, _m, _c = analysis
    joined = " ".join(result.warnings or []).lower()
    assert "checkpoint" in joined or "classical" in joined

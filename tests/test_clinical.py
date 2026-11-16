"""BI-RADS, ACR density, calcification lexicon, measurements."""
import numpy as np
import pytest

from config import Config
from src.core.detection_types import Detection, DetectionType


def _detection(**kwargs):
    defaults = dict(
        location=(100, 100),
        bbox=(80, 80, 120, 120),
        confidence=0.9,
        type=DetectionType.TUMOR_SUSPICIOUS,
        features={},
        risk_scores={},
    )
    defaults.update(kwargs)
    return Detection(**defaults)


def test_density_gmm_reports_fraction_and_acr(preprocessed):
    from src.models.breast_density_classifier import BreastDensityClassifier

    data, _m, _c = preprocessed
    result = BreastDensityClassifier().classify(data["enhanced"], data["breast_mask"])
    stats = result["statistics"]
    assert 0.0 <= stats["dense_area_percentage"] <= 100.0
    assert result["acr_category"] in ("A", "B", "C", "D")
    assert stats["dense_fraction_method"] in ("gmm2", "otsu")


@pytest.mark.parametrize(
    "acr,dense_pct",
    [("A", 10), ("B", 35), ("C", 60), ("D", 85)],
)
def test_acr_thresholds_map_to_expected_category(acr, dense_pct):
    from src.models.breast_density_classifier import BreastDensityClassifier

    classifier = BreastDensityClassifier()
    category, _confidence = classifier._predict_density_category(
        float(dense_pct), {"contrast": 0.4, "homogeneity": 0.5, "entropy": 5.0}
    )
    assert classifier.to_acr_category(category) == acr


def test_birads_categories_cover_zero_to_six():
    from src.models.birads_estimator import BIRADSEstimator

    estimator = BIRADSEstimator()
    assert set(estimator.BIRADS_DESCRIPTIONS.keys()) >= set(range(0, 7))


def test_birads_without_findings_is_category_one_or_two():
    from src.models.birads_estimator import BIRADSEstimator

    result = BIRADSEstimator().estimate(
        detections=[],
        density_result={"acr_category": "B", "statistics": {"dense_area_percentage": 35}},
    )
    assert result["category"] in (0, 1, 2)
    assert result["confidence"] is not None


def test_birads_flags_unvalidated_backend():
    from src.models.birads_estimator import BIRADSEstimator

    result = BIRADSEstimator().estimate(
        detections=[_detection()],
        density_result={"acr_category": "B", "statistics": {"dense_area_percentage": 35}},
        detection_backend="classical",
    )
    assert result["validation_status"] == "unvalidated_classical_fallback"
    assert result["validation_warning"]
    assert any("validated" in reason for reason in result["reasons"])


def test_low_confidence_findings_cannot_reach_high_category():
    from src.models.birads_estimator import BIRADSEstimator

    weak = _detection(
        confidence=0.3,
        features={"shape": "irregular", "margin": "spiculated", "density": "high"},
    )
    result = BIRADSEstimator().estimate(
        detections=[weak],
        density_result={"acr_category": "A", "statistics": {"dense_area_percentage": 10}},
        detection_backend="classical",
    )
    assert result["category"] <= 3


def test_calcification_lexicon_descriptors():
    from src.models.calcification_analysis import CalcificationAnalyzer

    detections = [
        _detection(
            type=DetectionType.MICROCALCIFICATION,
            location=(100 + 6 * index, 100 + (index % 3)),
            bbox=(96 + 6 * index, 96, 8, 8),  # (x, y, w, h) convention
            features={
                "num_calcifications": 8,
                "cluster_size_mm": 6.0,
                "morphology": "amorphous",
                "individual_calcifications": [
                    {"area": 3.0, "mean_intensity": 220.0, "circularity": 0.4}
                    for _ in range(8)
                ],
            },
        )
        for index in range(3)
    ]
    analysis = CalcificationAnalyzer().analyze(detections)
    assert analysis["dominant_morphology"]
    assert analysis["dominant_distribution"]
    assert analysis["suspicion_level"] in ("low", "intermediate", "high")
    assert analysis["cluster_count"] >= 1
    assert analysis["spacing_assumed"] is True
    assert "ASSUMED" in analysis["measurement_basis"]


def test_calcification_uses_known_spacing():
    from src.models.calcification_analysis import CalcificationAnalyzer

    detection = _detection(
        type=DetectionType.MICROCALCIFICATION, bbox=(100, 100, 20, 20),
        features={"num_calcifications": 1},
    )
    analysis = CalcificationAnalyzer().analyze([detection], pixel_spacing=(0.1, 0.1))
    assert analysis["spacing_known"] is True
    assert analysis["spacing_assumed"] is False
    assert "DICOM" in analysis["measurement_basis"]


def test_measurements_use_pixel_spacing():
    """A 30x40 px bounding box at 0.1 mm/px must measure 5.0 mm diagonally."""
    from src.measurement.lesion_measurement_engine import LesionMeasurementEngine

    masks = np.zeros((200, 200), np.uint8)
    masks[60:100, 60:100] = 255                      # 40x40 px square lesion
    detection = _detection(bbox=(60, 60, 40, 40), mask=masks)
    engine = LesionMeasurementEngine()
    result = engine.measure_detection(detection, 0.1, 0.1)
    assert result["diameter_px"] > 0
    assert np.isclose(result["diameter_mm"], result["diameter_px"] * 0.1, rtol=1e-3)
    assert np.isclose(result["area_mm2"], result["area_px"] * 0.01, rtol=1e-3)

    # same geometry measured without spacing must keep pixel values intact
    result_px = engine.measure_detection(detection, 1.0, 1.0)
    assert np.isclose(result_px["diameter_px"], result["diameter_px"], rtol=1e-6)


def test_birads_label_is_assistive_not_diagnostic():
    assert "Software" in Config.BIRADS_LABEL or "Assisted" in Config.BIRADS_LABEL
    assert "radiolog" in Config.DISCLAIMER.lower()

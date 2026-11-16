"""Detection quality on a phantom with known ground truth."""
import numpy as np
import pytest

from src.models.calcification_detector import CalcificationDetector
from src.models.tumor_detector import TumorDetector


def test_mass_detected_at_ground_truth(preprocessed):
    data, mass_centre, _cluster = preprocessed
    detections = TumorDetector().detect(data["enhanced"], data["breast_mask"])
    assert detections, "no mass candidate was produced"
    best = min(
        detections,
        key=lambda d: np.hypot(d.location[0] - mass_centre[0], d.location[1] - mass_centre[1]),
    )
    distance = np.hypot(best.location[0] - mass_centre[0], best.location[1] - mass_centre[1])
    assert distance < 20, f"closest detection is {distance:.1f} px from the true mass"
    assert best.confidence > 0.3


def test_calcification_cluster_detected(preprocessed):
    data, _mass, cluster_centre = preprocessed
    detections = CalcificationDetector().detect(data["enhanced"], data["breast_mask"])
    assert detections, "no calcification cluster was produced"
    best = min(
        detections,
        key=lambda d: np.hypot(d.location[0] - cluster_centre[0], d.location[1] - cluster_centre[1]),
    )
    distance = np.hypot(best.location[0] - cluster_centre[0], best.location[1] - cluster_centre[1])
    assert distance < 30, f"closest cluster is {distance:.1f} px from the truth"
    assert best.features.get("num_calcifications", 0) >= 3


def test_detections_stay_inside_the_breast(preprocessed):
    data, _m, _c = preprocessed
    mask = data["breast_mask"]
    for detection in TumorDetector().detect(data["enhanced"], mask):
        x, y = int(detection.location[0]), int(detection.location[1])
        assert 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
        assert mask[y, x] > 0, "detection centre outside the breast mask"


def test_detection_cap_and_runtime(preprocessed):
    import time

    data, _m, _c = preprocessed
    start = time.time()
    detections = TumorDetector().detect(data["enhanced"], data["breast_mask"])
    elapsed = time.time() - start
    assert len(detections) <= 6
    assert elapsed < 30, elapsed


def test_empty_mask_returns_no_detections():
    empty_mask = np.zeros((300, 300), np.uint8)
    image = np.zeros((300, 300), np.uint8)
    assert TumorDetector().detect(image, empty_mask) == []

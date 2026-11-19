"""Technical regression cases, not clinical performance evidence."""
import cv2
import numpy as np
import pytest

from src.core.detection_types import Detection, DetectionType
from src.models.calcification_detector import CalcificationDetector
from src.models.mass_pipeline import detect_masses
from src.models.visualizer import ResultVisualizer


@pytest.mark.parametrize('sign', [1, -1])
def test_irregular_original_grid_mass(sign, monkeypatch):
    from config import Config

    monkeypatch.setattr(Config, 'MASS_LOCALIZATION_MAX_SIDE', 200)
    h, w = 244, 260
    tissue = np.zeros((h, w), np.uint8)
    cv2.ellipse(tissue, (130,120), (119,108), 0, 0, 360, 255, -1)
    truth = np.zeros_like(tissue)
    cv2.fillPoly(truth, [np.array([[104,103],[124,87],[138,92],[144,108],[159,115],
                                 [146,133],[135,137],[126,156],[106,144],[97,122]])], 1)
    rng = np.random.default_rng(11)
    image = (120 + cv2.GaussianBlur(rng.normal(0, 5, truth.shape), (0,0), 2)
             + 28 * sign * cv2.GaussianBlur(truth.astype(float), (0,0), 1.0)).astype(np.float32)
    image[tissue == 0] = 0
    original = image.copy()
    debug = {}
    detections = detect_masses(image, tissue, debug)
    d = min((d for d in detections if d.mask is not None), key=lambda d: np.hypot(d.location[0]-127,d.location[1]-120))
    assert d.mask.shape == truth.shape and not np.any(d.mask[tissue == 0])
    assert d.features['polarity'] == sign
    assert d.features['confidence_kind'] == 'uncalibrated heuristic'
    x,y = d.segmentation_origin
    masks = list(d.segmentation_hypotheses)
    masks.append(d.mask[y:y+masks[0].shape[0], x:x+masks[0].shape[1]])
    y0,x0 = np.nonzero(truth)
    results = []
    for mask in masks:
        patch = np.zeros_like(truth, dtype=bool)
        patch[y:y+mask.shape[0], x:x+mask.shape[1]] = mask > 0
        intersection = np.count_nonzero(patch & (truth > 0))
        results.append((2*intersection/(patch.sum()+len(y0)), intersection/np.count_nonzero(patch | (truth > 0))))
    print(f'phantom sign={sign}, Dice/IoU [FCM, gradient, adaptive, fused]: {np.round(results, 3).tolist()}')
    assert results[-1][0] > .5
    assert len(d.segmentation_hypotheses) == 3
    assert debug['maps'].shape == (5, 188, 200)
    assert np.array_equal(original, image)


def test_visualizer_layers_and_missing_geometry():
    image = np.zeros((100,100), np.uint8)
    mask = image.copy()
    mask[52:56, 51:57] = 255
    d = Detection(DetectionType.TUMOR_SUSPICIOUS, (54,54), (48,48,16,16), mask, .55, id=1)
    no_geometry = Detection(DetectionType.TUMOR_SUSPICIOUS, (80,80), id=2)
    v = ResultVisualizer()
    v.show_arrows = v.show_labels = v.show_confidence = False
    v.show_tumors = False
    rendered = v.visualize_detections(image, [d, no_geometry])
    assert np.array_equal(np.any(rendered != 0, axis=2), mask > 0)
    v.show_masks = False
    assert not v.visualize_detections(image, [d, no_geometry]).any()
    v.show_tumors = True
    v.show_centers = False
    assert v.visualize_detections(image, [d, no_geometry])[48,48].any()
    assert not v.visualize_detections(image, [d, no_geometry])[55,56].any()
    v.show_tumors = False
    v.show_arrows = True
    assert v.visualize_detections(image, [d, no_geometry])[25,54].any()
    v.show_arrows = False
    v.show_labels = True
    labels = v.visualize_detections(image, [d, no_geometry])
    v.show_labels = False
    v.show_confidence = True
    assert not np.array_equal(labels, v.visualize_detections(image, [d, no_geometry]))


def test_calc_hysteresis_preserves_single_pixel_and_anisotropic_clustering():
    image = np.full((101,101), 60, np.uint8)
    image[50,50] = 245
    valid = np.ones_like(image, bool)
    detector = CalcificationDetector()
    seed, score = detector._hysteresis(image, valid, valid)
    assert seed[50,50] and score[50,50]
    assert not seed[0,0]
    spots = [dict(center=(x,50), area=4., max_intensity=240., mean_intensity=220.,
                  local_contrast=80., prominence=70., circularity=.9, bbox=(x,50,1,1)) for x in (30, 50, 70)]
    groups = detector._cluster_microcalcifications(spots, image, spacing=(.05,.1))
    f = next(g['features'] for g in groups if g['features']['num_calcifications'] == 3)
    assert f['distance_unit'] == 'mm' and f['cluster_diameter'] == pytest.approx(4.)
    distant = detector._cluster_microcalcifications(spots, image, spacing=(.05,.3))
    assert all(g['features']['num_calcifications'] == 1 for g in distant)

"""Preprocessing: polarity, mask geometry, bit depths, performance."""
import cv2
import numpy as np
import pytest

from src.processing.preprocessor import MammographyPreprocessor as P


def test_breast_mask_is_plausible(preprocessed):
    data, _mass, _cluster = preprocessed
    mask = data["breast_mask"]
    coverage = np.count_nonzero(mask) / mask.size
    assert 0.2 < coverage < 0.75, coverage
    # Background must be excluded
    assert mask[5, 5] == 0


def test_enhanced_image_preserves_geometry_and_dtype(preprocessed):
    data, _m, _c = preprocessed
    assert data["enhanced"].dtype == np.uint8
    assert data["enhanced"].shape == data["original"].shape
    # Enhancement must not introduce intensity outside the breast
    assert np.count_nonzero(data["enhanced"][data["breast_mask"] == 0]) == 0


def test_pectoral_removal_is_vectorised_and_bounded(preprocessed):
    data, _m, _c = preprocessed
    pectoral = data["pectoral_mask"]
    if pectoral is None or not np.any(pectoral):
        pytest.skip("no pectoral muscle present in this phantom")
    fraction = np.count_nonzero(pectoral) / np.count_nonzero(data["breast_mask"])
    assert fraction <= 0.35


def test_inverted_image_is_flipped():
    height, width = 200, 200
    image = np.full((height, width), 240, np.uint8)          # bright background
    cv2.ellipse(image, (100, 100), (60, 80), 0, 0, 360, 40, -1)  # dark breast
    flipped = P.ensure_display_polarity(image)
    assert flipped[100, 100] > flipped[2, 2], "polarity was not normalised"


def test_normal_image_is_not_flipped():
    image = np.zeros((200, 200), np.uint8)
    cv2.ellipse(image, (100, 100), (60, 80), 0, 0, 360, 180, -1)
    flipped = P.ensure_display_polarity(image)
    np.testing.assert_array_equal(image, flipped)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_to_uint8_handles_all_dtypes(dtype):
    if dtype == np.uint8:
        array = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    elif dtype == np.uint16:
        array = np.random.default_rng(0).integers(0, 65535, (64, 64), dtype=np.uint16)
    else:
        array = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    converted = P.to_uint8(array)
    assert converted.dtype == np.uint8
    assert converted.shape == array.shape
    assert converted.max() > 0


def test_frame_removal_ignores_uniform_page_border():
    """A light page around the film must not become the 'breast'."""
    height, width = 300, 300
    image = np.full((height, width), 250, np.uint8)     # page
    image[40:260, 40:260] = 5                           # film
    cv2.ellipse(image, (150, 160), (70, 90), 0, 0, 360, 170, -1)  # breast
    mask = P.extract_breast_mask(image)
    coverage = np.count_nonzero(mask) / mask.size
    assert coverage < 0.75, f"page was included in the mask ({coverage:.2f})"
    assert mask[45, 45] == 0, "page corner must be excluded"


def test_preprocess_is_fast():
    """Guards against the previous per-pixel Hough implementation."""
    import time

    image = np.zeros((1200, 900), np.uint8)
    cv2.ellipse(image, (400, 600), (330, 480), 0, 0, 360, 120, -1)
    start = time.time()
    P.preprocess_array(image)
    assert time.time() - start < 5.0

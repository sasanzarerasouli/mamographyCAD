"""Shared fixtures: synthetic mammograms with known ground truth."""
import os
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root() -> str:
    return PROJECT_ROOT


def make_phantom(seed: int = 0, noise: float = 4.0):
    """Synthetic MLO-like mammogram with a spiculated mass and a calc cluster.

    Returns ``(image, mass_centre, cluster_centre)``.
    """
    rng = np.random.default_rng(seed)
    height, width = 800, 600
    image = np.zeros((height, width), np.uint8)
    # breast body (left of image = chest wall side)
    cv2.ellipse(image, (260, 400), (230, 330), 0, 0, 360, 90, -1)
    cv2.ellipse(image, (260, 400), (160, 240), 0, 0, 360, 130, -1)

    # correlated (structured) parenchymal texture: realistic for mammograms
    texture = cv2.GaussianBlur(rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), 4.0)
    texture *= noise / (texture.std() + 1e-6)
    image = np.clip(image.astype(np.float32) + texture, 0, 255).astype(np.uint8)

    # mass: bright disc with spicules
    mass_centre = (260, 430)
    cv2.circle(image, mass_centre, 40, 170, -1)
    cv2.circle(image, mass_centre, 32, 190, -1)
    for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        start = (
            int(mass_centre[0] + 40 * np.cos(angle)),
            int(mass_centre[1] + 40 * np.sin(angle)),
        )
        end = (
            int(mass_centre[0] + 56 * np.cos(angle)),
            int(mass_centre[1] + 56 * np.sin(angle)),
        )
        cv2.line(image, start, end, 165, 2)

    # microcalcification cluster: 7 bright dots within ~1 cm of each other
    cluster_centre = (250, 600)
    for _ in range(7):
        x = int(cluster_centre[0] + rng.normal(0, 12))
        y = int(cluster_centre[1] + rng.normal(0, 10))
        cv2.circle(image, (x, y), 2, 245, -1)

    return image, mass_centre, cluster_centre


@pytest.fixture(scope="session")
def phantom():
    return make_phantom()


@pytest.fixture(scope="session")
def phantom_path(tmp_path_factory, phantom):
    image, mass, cluster = phantom
    directory = tmp_path_factory.mktemp("images")
    path = os.path.join(directory, "phantom.png")
    cv2.imwrite(path, image)
    return path, mass, cluster


@pytest.fixture(scope="session")
def preprocessed(phantom):
    from src.processing.preprocessor import MammographyPreprocessor

    image, mass, cluster = phantom
    return MammographyPreprocessor.preprocess_array(image), mass, cluster


@pytest.fixture(scope="session")
def analysis(phantom_path):
    """Full pipeline result for the phantom (computed once)."""
    from src.core.cad_system import MammographyCADSystem

    path, mass, cluster = phantom_path
    return MammographyCADSystem().analyze(path), mass, cluster

"""End-to-end verification checklist.

    python -m tools.verify            # human readable
    python -m tools.verify --json     # machine readable

Each check is executed for real (no mocking) and prints PASS/FAIL with the
measured values.  Checks that depend on optional components (checkpoints, torch,
a Qt platform) report SKIP with the reason instead of pretending to pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import Config  # noqa: E402

RESULTS = []


def check(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            started = time.time()
            try:
                detail = func(*args, **kwargs) or ""
                status = "SKIP" if str(detail).startswith("SKIP:") else "PASS"
            except AssertionError as exc:
                status, detail = "FAIL", str(exc)
            except Exception as exc:  # noqa: BLE001
                status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
            RESULTS.append(
                {
                    "check": name,
                    "status": status,
                    "detail": str(detail),
                    "seconds": round(time.time() - started, 2),
                }
            )
            print(f"[{status}] {name} ({time.time() - started:.1f}s)")
            if detail:
                print(f"       {detail}")
            return status

        return wrapper

    return decorator


def phantom(path: str) -> str:
    """Deterministic synthetic mammogram with a known mass and calc cluster."""
    rng = np.random.default_rng(0)
    height, width = 800, 600
    image = np.zeros((height, width), np.uint8)
    cv2.ellipse(image, (260, 400), (230, 330), 0, 0, 360, 90, -1)
    cv2.ellipse(image, (260, 400), (160, 240), 0, 0, 360, 130, -1)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), 4.0)
    texture *= 4.0 / (texture.std() + 1e-6)
    image = np.clip(image.astype(np.float32) + texture, 0, 255).astype(np.uint8)
    cv2.circle(image, (260, 430), 40, 170, -1)
    cv2.circle(image, (260, 430), 32, 190, -1)
    for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        cv2.line(
            image,
            (int(260 + 40 * np.cos(angle)), int(430 + 40 * np.sin(angle))),
            (int(260 + 56 * np.cos(angle)), int(430 + 56 * np.sin(angle))),
            165, 2,
        )
    for _ in range(7):
        cv2.circle(
            image,
            (int(250 + rng.normal(0, 12)), int(600 + rng.normal(0, 10))),
            2, 245, -1,
        )
    cv2.imwrite(path, image)
    return path


def make_dicom(path: str) -> str:
    """16-bit MONOCHROME1 DICOM with spacing, for the decoding check."""
    pydicom = __import__("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    image = cv2.imread(os.path.join(TMP, "phantom.png"), cv2.IMREAD_GRAYSCALE)
    array = (np.flipud(image).astype(np.uint16) * 16)  # MONOCHROME1 + 16-bit

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID, ds.PatientName = "VERIFY01", "Verify^Phantom"
    ds.Modality, ds.StudyDate, ds.ViewPosition, ds.ImageLaterality = "MG", "20260101", "MLO", "L"
    ds.PixelSpacing = [0.05, 0.05]
    ds.Rows, ds.Columns = array.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation, ds.PhotometricInterpretation, ds.SamplesPerPixel = 0, "MONOCHROME1", 1
    ds.PixelData = array.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


TMP = tempfile.mkdtemp(prefix="cadverify_")


# ---------------------------------------------------------------------- #
@check("1. application imports and configuration loads")
def check_imports():
    import main  # noqa: F401  (imports the whole GUI stack)
    assert os.path.isdir(Config.REPORTS_DIR)
    return f"backend mode={Config.DETECTION_BACKEND}, device={Config.DL_DEVICE}"


@check("2. image loading + pre-processing (PNG/JPG)")
def check_preprocess():
    from src.processing.preprocessor import MammographyPreprocessor

    path = phantom(os.path.join(TMP, "phantom.png"))
    gray, metadata = MammographyPreprocessor.load_image(path)
    data = MammographyPreprocessor.preprocess_array(gray)
    coverage = np.count_nonzero(data["breast_mask"]) / data["breast_mask"].size
    assert 0.2 < coverage < 0.8, f"implausible mask coverage {coverage:.3f}"
    assert data["enhanced"].shape == gray.shape
    return f"{gray.shape[1]}x{gray.shape[0]} px, mask coverage {coverage:.1%}"


@check("3. DICOM decoding (16-bit, MONOCHROME1, spacing, MLO)")
def check_dicom():
    try:
        import pydicom  # noqa: F401
    except ImportError:
        return "SKIP: pydicom not installed"
    from src.io.dicom_handler import DICOMHandler

    path = make_dicom(os.path.join(TMP, "verify.dcm"))
    handler = DICOMHandler()
    gray, metadata = handler.load_grayscale(path)
    spacing = handler.get_pixel_spacing(metadata)
    view = handler.get_view_position(metadata)
    assert gray is not None and gray.dtype == np.uint8
    assert metadata["image"]["monochrome1_inverted"] is True
    assert spacing == (0.05, 0.05), spacing
    assert view == "MLO", view
    return f"MONOCHROME1 inverted, spacing {spacing} mm, view {view}"


@check("4. mass detection on a phantom with known ground truth")
def check_mass():
    from src.core.cad_system import MammographyCADSystem

    result = MammographyCADSystem().analyze(os.path.join(TMP, "phantom.png"))
    masses = [d for d in result.detections if "tumor" in d.type.value]
    assert masses, "no mass candidate"
    distances = [float(np.hypot(d.location[0] - 260, d.location[1] - 430)) for d in masses]
    best = min(distances)
    assert best < 20, f"closest candidate {best:.1f} px from the true mass"
    return f"{len(masses)} candidate(s), closest {best:.1f} px from truth"


@check("5. calcification (cluster) detection on the same phantom")
def check_calcification():
    from src.core.cad_system import MammographyCADSystem

    result = MammographyCADSystem().analyze(os.path.join(TMP, "phantom.png"))
    calcs = [d for d in result.detections if "calc" in d.type.value]
    assert calcs, "no calcification detection"
    best = min(float(np.hypot(d.location[0] - 250, d.location[1] - 600)) for d in calcs)
    assert best < 40, f"closest cluster {best:.1f} px from the truth"
    clusters = (result.calcification_analysis or {}).get("cluster_count")
    return f"{len(calcs)} detection(s), closest {best:.1f} px, clusters={clusters}"


@check("6. BI-RADS 0-6 and ACR density are produced with evidence")
def check_clinical():
    from src.core.cad_system import MammographyCADSystem

    result = MammographyCADSystem().analyze(os.path.join(TMP, "phantom.png"))
    birads = result.birads
    assert 0 <= birads["category"] <= 6
    assert birads["reasons"] and birads["management"]
    assert result.density_result["acr_category"] in "ABCD"
    assert birads.get("validation_status") in (
        "validated_model", "unvalidated_classical_fallback"
    )
    return (
        f"BI-RADS {birads['category']} ({birads.get('subcategory') or 'n/a'}), "
        f"ACR {result.density_result['acr_category']} "
        f"({result.density_result['statistics']['dense_area_percentage']:.1f}% dense), "
        f"validation={birads.get('validation_status')}"
    )


@check("7. reporting (JSON + self-contained HTML with provenance)")
def check_report():
    from src.core.cad_system import MammographyCADSystem
    from src.reporting.report_builder import ReportBuilder

    result = MammographyCADSystem().analyze(os.path.join(TMP, "phantom.png"))
    out = os.path.join(TMP, "reports")
    paths = ReportBuilder(output_dir=out).save(result)
    with open(paths["html"], encoding="utf-8") as fh:
        html = fh.read()
    assert "data:image/png;base64," in html
    assert "BI-RADS" in html and "radiolog" in html.lower()
    assert "http://" not in html and "https://" not in html
    with open(paths["json"], encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["clinical_summary"]["birads_category"] >= 0
    return f"{os.path.basename(paths['json'])} + {os.path.basename(paths['html'])} ({os.path.getsize(paths['html']) // 1024} kB)"


@check("8. model registry / CPU inference availability")
def check_models():
    from src.models.deep_detector import backend_status
    from src.models.model_registry import runtime_status

    status = runtime_status()
    backends = backend_status()
    models = [m for m in (backends.get("mass", {}), backends.get("calcification", {}))]
    assert all(isinstance(m, dict) for m in models)
    reason = (
        "no checkpoint found and torch/ultralytics not installed"
        if not status["torch"] and not status["ultralytics"]
        else "runtime present but no usable checkpoint"
    )
    return (
        f"torch={status['torch']}, ultralytics={status['ultralytics']}, "
        f"mass backend={backends['mass']['backend']}, "
        f"calc backend={backends['calcification']['backend']} ({reason})"
    )


@check("9. forced deep backend without weights fails gracefully")
def check_forced_backend():
    from src.core.cad_system import MammographyCADSystem

    original = Config.DETECTION_BACKEND
    try:
        Config.DETECTION_BACKEND = "on"
        system = MammographyCADSystem()
        result = System_analyze = system.analyze(os.path.join(TMP, "phantom.png"))
    finally:
        Config.DETECTION_BACKEND = original
    assert result is not None, "analysis crashed instead of degrading"
    warnings = " ".join(result.warnings or [])
    assert result.detections or warnings, "no detections and no warning - silent failure"
    return f"analysis completed with {len(result.warnings or [])} warning(s): {warnings[:120]}"


@check("10. GUI constructs offscreen with all tabs")
def check_gui():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        return f"SKIP: Qt unavailable ({exc})"
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        return f"SKIP: Qt platform plugin unavailable ({exc})"
    from src.ui.main_window import MammographyCADApp

    window = MammographyCADApp()
    window.silent_dialogs = True
    image_tabs = window.image_tabs.count()
    side_tabs = window.side_tabs.count()
    window.close()
    app.processEvents()
    assert image_tabs == 5, image_tabs
    assert side_tabs == 5, side_tabs
    return f"{image_tabs} image tabs, {side_tabs} side tabs"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    print(f"Temporary workspace: {TMP}\n")
    for function in (
        check_imports, check_preprocess, check_dicom, check_mass,
        check_calcification, check_clinical, check_report, check_models,
        check_forced_backend, check_gui,
    ):
        function()

    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    skipped = sum(1 for r in RESULTS if r["status"] == "SKIP")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed "
          f"of {len(RESULTS)} checks")

    if args.json:
        print(json.dumps({"results": RESULTS}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

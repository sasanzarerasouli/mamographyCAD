"""GUI tests (head-less / offscreen).

Requires the Qt platform plugin.  In a minimal container run with::

    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui.py

On a normal desktop nothing special is needed.  If Qt cannot load its X
libraries the whole module is skipped instead of failing the suite.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

try:  # pragma: no cover - environment dependent
    from PyQt6.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    _QT_OK = True
except Exception as exc:  # pragma: no cover
    _QT_OK = False
    _QT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(not _QT_OK, reason="Qt platform plugin unavailable")


def run_analysis(window, timeout: float = 180.0) -> bool:
    """Start the (threaded) analysis and pump the Qt event loop until it ends."""
    import time

    window.on_process()
    deadline = time.time() + timeout
    while time.time() < deadline:
        _APP.processEvents()
        if window.current_result is not None:
            # let the queued UI updates finish
            for _ in range(5):
                _APP.processEvents()
            return True
        if window.worker is not None and not window.worker.isRunning() and window.current_result is None:
            _APP.processEvents()
            return window.current_result is not None
        time.sleep(0.05)
    return False


@pytest.fixture(scope="module")
def window():
    from src.ui.main_window import MammographyCADApp

    win = MammographyCADApp()
    win.silent_dialogs = True
    win.show()
    yield win
    win.close()


def test_window_builds_all_tabs(window):
    image_titles = [window.image_tabs.tabText(i) for i in range(window.image_tabs.count())]
    assert len(image_titles) == 5, image_titles
    for expected in ("Original", "Enhanced", "Mask", "Density", "Heatmap"):
        assert any(expected in title for title in image_titles), (expected, image_titles)

    side_titles = [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())]
    for expected in ("Results", "Clinical", "Lesion detail", "Settings", "Models"):
        assert any(expected in title for title in side_titles), (expected, side_titles)


def test_processing_populates_every_panel_consistently(window, phantom_path):
    path, _mass, _cluster = phantom_path
    assert window.load_image(path)
    assert run_analysis(window), "analysis did not complete"

    result = window.current_result
    assert result is not None, "on_process did not produce a result"

    # overlay widget must hold the rendering of this result
    overlay_widget = window.lbl_original
    pixmap = overlay_widget.current_image()
    assert pixmap is not None and not pixmap.isNull()
    assert not overlay_widget.rendered_pixmap().isNull()
    assert window.current_overlay is not None
    assert window.current_overlay.shape[:2] == result.original.shape[:2]

    # results table rows == detections
    assert window.table_detections.rowCount() == len(result.detections)

    # clinical panel mirrors the result object (no divergence)
    assert str(result.birads["category"]) in window.lbl_birads_category.text()
    assert result.density_result["acr_category"] in window.lbl_breast_density_category.text()
    assert window.lbl_backend.text() not in ("", "N/A")
    # calcification panel reflects the analysis object
    calc = result.calcification_analysis or {}
    if calc.get("available"):
        assert window.lbl_calc_morphology.text() not in ("", "N/A")


def test_zoom_does_not_hide_overlays(window, phantom_path):
    """Regression: overlays used to appear only after a zoom/resize event."""
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QWheelEvent

    path, _m, _c = phantom_path
    window.load_image(path)
    assert run_analysis(window)
    widget = window.lbl_original

    before = widget.rendered_pixmap().toImage()
    from PyQt6.QtCore import QPoint, Qt

    event = QWheelEvent(
        QPointF(100, 100),          # position
        QPointF(100, 100),          # global position
        QPoint(0, 0),               # pixel delta
        QPoint(0, 120),             # angle delta (zoom in)
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,                      # inverted
    )
    widget.wheelEvent(event)
    _APP.processEvents()
    after = widget.rendered_pixmap().toImage()

    # the widget must still render content of the same size after zooming
    assert after.width() > 0 and after.height() > 0
    assert before.size() == after.size()


def test_detection_selection_updates_lesion_panel(window, phantom_path):
    path, _m, _c = phantom_path
    window.load_image(path)
    assert run_analysis(window)
    if window.table_detections.rowCount() == 0:
        pytest.skip("no detections to select")
    window.table_detections.selectRow(0)
    _APP.processEvents()
    assert window.lbl_selected_lesion.text() not in ("", "No region selected")
    assert window.lbl_shape.text() not in ("", "N/A")
    assert window.lbl_margin.text() not in ("", "N/A")


def test_ignore_toggle_keeps_overlay_and_table_in_sync(window, phantom_path):
    path, _m, _c = phantom_path
    window.load_image(path)
    assert run_analysis(window)
    total = len(window.current_result.detections)
    assert total >= 2
    window.current_result.detections[0].is_ignored = True
    window.update_results_display()
    window.update_visualization()
    _APP.processEvents()
    assert window.table_detections.rowCount() == total


def test_settings_persist_backend_choice(window):
    from config import Config

    original = Config.DETECTION_BACKEND
    try:
        window.combo_backend.setCurrentText("off")
        window.apply_settings()
        assert Config.DETECTION_BACKEND == "off"
        assert window.cad_system._deep_checked is False
    finally:
        window.combo_backend.setCurrentText(
            original if original in ("auto", "deep", "off") else "auto"
        )
        window.apply_settings()


def test_batch_mode_processes_a_folder(window, tmp_path, phantom_path):
    import shutil

    path, _m, _c = phantom_path
    for index in range(2):
        shutil.copy(path, tmp_path / f"study_{index}.png")
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    assert window.start_batch(str(tmp_path), str(out_dir))
    import time

    for _ in range(1800):
        _APP.processEvents()
        if not window.batch_running:
            break
        time.sleep(0.1)
    assert not window.batch_running, "batch did not finish within the timeout"
    assert window.batch_ok == 2, window.batch_failures
    assert window.batch_failed == 0, window.batch_failures
    reports = [name for name in os.listdir(out_dir) if name.endswith(".html")]
    assert len(reports) == 2, reports


def test_mask_correction_mouse_undo_restore_and_report(window, analysis, tmp_path):
    from copy import deepcopy
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtTest import QTest
    from src.reporting.report_builder import ReportBuilder

    result = deepcopy(analysis[0])
    window.on_analysis_ready(result)
    assert 'Mass candidate stage complete' in window.activity_log.toPlainText() or 'Analysis returned' in window.activity_log.toPlainText()
    mass = next(d for d in result.detections if d.mask is not None and 'segmentation_confidence' in d.features)
    window.on_detection_clicked(mass.id)
    window.check_show_original.setChecked(True)
    assert not window.lbl_original.detections
    window.check_show_original.setChecked(False)
    auto = mass.mask.copy()
    original = result.original.copy()
    y, x = np.argwhere(auto > 0)[len(np.argwhere(auto > 0))//2]
    window.mask_mode.setCurrentIndex(2)
    widget = window.lbl_original
    widget.fit_to_window()
    click = widget._image_to_screen_coords(QPointF(x+.5,y+.5)).toPoint()
    assert widget.rect().contains(click)
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=click)
    assert np.count_nonzero(mass.mask) < np.count_nonzero(auto)
    assert np.array_equal(mass.automatic_mask, auto)
    assert np.array_equal(result.original, original)
    assert mass.features['measurement']['area_px'] == np.count_nonzero(mass.mask)
    builder = ReportBuilder(str(tmp_path))
    assert 'manual correction' in builder.build_html(result, include_images=False)
    target = tmp_path / 'research.pdf'
    builder.save_pdf(result, str(target))
    assert target.read_bytes().startswith(b'%PDF')
    window._undo_mask_edit()
    assert np.array_equal(mass.mask, auto)
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=click)
    window._restore_automatic_mask()
    assert np.array_equal(mass.mask, auto)
    window.mask_mode.setCurrentIndex(0)


def test_docx_research_export(analysis, tmp_path):
    docx = pytest.importorskip('docx')
    from src.reporting.report_builder import ReportBuilder

    path = tmp_path / 'research.docx'
    ReportBuilder(str(tmp_path)).save_docx(analysis[0], str(path))
    content = docx.Document(path)
    assert any('independent diagnostic validity' in paragraph.text for paragraph in content.paragraphs)
    assert any('evidence score' in paragraph.text for paragraph in content.paragraphs)
    assert len(content.inline_shapes) == 1

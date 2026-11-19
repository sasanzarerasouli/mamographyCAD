"""PyQt6 main window of the Mammography CAD application.

The window is organised around the analysis pipeline:

* toolbar - load / analyse / cancel / save / export / batch / reset view
* image tabs - original+overlay, enhanced, breast mask, density map, heatmap
* side panel - results and clinical summary, lesion detail, explainability,
  settings, and model/status information

Every value that the pipeline computes is displayed somewhere in this window;
if a stage was skipped the panel says so explicitly instead of showing a void.
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QSettings, pyqtSlot
from PyQt6.QtGui import QAction, QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import Config
from ..core.analysis_result import AnalysisResult
from ..core.batch_processor import BatchProcessor
from ..core.cad_system import MammographyCADSystem
from ..core.detection_types import Detection, DetectionType
from ..core.worker import CADWorker
from ..models.deep_detector import backend_status
from ..models.model_registry import ModelRegistry, runtime_status
from ..reporting.report_builder import ReportBuilder
from .zoomable_label import ZoomableImageLabel

logger = logging.getLogger("MammographyCAD.ui")

IMAGE_EXTENSIONS = (
    ".dcm", ".dicom", ".ima", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
)

IMAGE_FILTER = (
    "Mammography images (*.dcm *.dicom *.ima *.png *.jpg *.jpeg *.tif *.tiff *.bmp);;"
    "DICOM (*.dcm *.dicom *.ima);;"
    "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;"
    "All files (*)"
)


class MammographyCADApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "Mammography CAD - Mass & Calcification Detection 'mater tesis' "
        )

        self.settings = QSettings("MammographyCAD", "Settings")
        self.cad_system = MammographyCADSystem()
        self.report_builder = ReportBuilder()

        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.92), int(screen.height() * 0.92))
        self.setMinimumSize(1200, 800)

        self.current_image_path: Optional[str] = None
        self.current_result: Optional[AnalysisResult] = None
        self.current_overlay: Optional[np.ndarray] = None
        self.current_detections: List[Detection] = []
        self.current_statistics: Dict = {}
        self.processing_time = 0.0
        self.worker: Optional[CADWorker] = None
        self.batch_worker: Optional[BatchProcessor] = None
        self.selected_detection_id: Optional[int] = None
        self._mask_undo = None
        self.batch_running = False
        self.batch_ok = 0
        self.batch_failed = 0
        self.batch_failures: List[str] = []
        # Modal dialogs block head-less test runners; they are suppressed when
        # the Qt platform is offscreen or CAD_SILENT_DIALOGS=1 is exported.
        self.silent_dialogs = (
            os.environ.get("CAD_SILENT_DIALOGS") == "1"
            or os.environ.get("QT_QPA_PLATFORM", "") in ("offscreen", "minimal")
        )

        self.init_ui()
        self.load_settings()
        self.refresh_model_status()

    # ================================================================== #
    #  UI construction                                                   #
    # ================================================================== #
    def init_ui(self) -> None:
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        main_layout.addLayout(self._build_toolbar())
        main_layout.addLayout(self._build_visualization_controls())

        content_layout = QHBoxLayout()
        self.image_tabs = QTabWidget()
        self.lbl_original = self._add_image_tab("Original / Overlay")
        self.lbl_original.maskBrushClicked.connect(self._apply_mask_brush)
        self.lbl_enhanced = self._add_image_tab("Enhanced")
        self.lbl_mask = self._add_image_tab("Breast Mask")
        self.lbl_density = self._add_image_tab("Density Map")
        self.lbl_heatmap = self._add_image_tab("Suspicion Heatmap")
        self.lbl_original.detectionClicked.connect(self.on_detection_clicked)
        content_layout.addWidget(self.image_tabs, 3)
        self.side_tabs = self._build_side_panel()
        content_layout.addWidget(self.side_tabs, 2)

        main_layout.addLayout(content_layout, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.setCentralWidget(central)
        self.status.showMessage("Ready")

    def _build_toolbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Mammogram")
        self.btn_process = QPushButton("Run Detection")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_save = QPushButton("Save Overlay")
        self.btn_export = QPushButton("Export Report")
        self.btn_batch = QPushButton("Batch Process")
        self.btn_reset_view = QPushButton("Reset View")

        self.btn_load.clicked.connect(self.on_load)
        self.btn_process.clicked.connect(self.on_process)
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_batch.clicked.connect(self.on_batch)
        self.btn_reset_view.clicked.connect(self.on_reset_view)

        self.btn_process.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_export.setEnabled(False)

        for button in (
            self.btn_load, self.btn_process, self.btn_cancel, self.btn_save,
            self.btn_export, self.btn_batch, self.btn_reset_view,
        ):
            layout.addWidget(button)
        layout.addStretch()
        return layout

    def _build_visualization_controls(self) -> QVBoxLayout:
        rows = QVBoxLayout()
        layout = QHBoxLayout()
        rows.addLayout(layout)

        self.check_show_labels = QCheckBox("Labels")
        self.check_show_tumors = QCheckBox("Mass boxes")
        self.check_show_confidence = QCheckBox("Scores")
        self.check_show_calc_masks = QCheckBox("Calc pixels")
        self.check_show_calc_labels = QCheckBox("Calc labels")
        self.check_show_calc_labels.setChecked(True)
        self.check_show_calc_labels.toggled.connect(self.on_visualization_changed)
        self.check_show_clusters = QCheckBox("Clusters")
        self.check_show_original = QCheckBox("Original only")
        self.check_show_original.toggled.connect(self.on_visualization_changed)
        layout.addWidget(self.check_show_original)
        self.check_show_calcs = QCheckBox("Calcifications")
        self.check_show_arrows = QCheckBox("Arrows")
        self.check_show_mask_layer = QCheckBox("Mass masks")
        self.check_show_pectoral = QCheckBox("Pectoral boundary")
        self.check_show_heatmap = QCheckBox("Density heatmap")

        for check in (
            self.check_show_labels, self.check_show_tumors, self.check_show_calcs,
            self.check_show_arrows, self.check_show_mask_layer,
            self.check_show_pectoral, self.check_show_heatmap, self.check_show_confidence,
            self.check_show_calc_masks, self.check_show_clusters,
        ):
            check.setChecked(True)
            check.toggled.connect(self.on_visualization_changed)
            layout.addWidget(check)

        layout.addStretch()
        layout = QHBoxLayout()
        rows.addLayout(layout)
        layout.addWidget(self.check_show_calc_labels)
        layout.addWidget(QLabel("Thickness:"))
        self.slider_line_thickness = QSlider(Qt.Orientation.Horizontal)
        self.slider_line_thickness.setRange(1, 8)
        self.slider_line_thickness.setValue(2)
        self.slider_line_thickness.setFixedWidth(80)
        self.slider_line_thickness.valueChanged.connect(self.on_visualization_changed)
        layout.addWidget(self.slider_line_thickness)

        layout.addWidget(QLabel("Opacity:"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(45)
        self.slider_opacity.setFixedWidth(80)
        self.slider_opacity.valueChanged.connect(self.on_visualization_changed)
        layout.addWidget(self.slider_opacity)

        layout.addStretch()
        self.lbl_processing_info = QLabel("Ready")
        layout.addWidget(self.lbl_processing_info)
        return rows

    def _add_image_tab(self, title: str) -> ZoomableImageLabel:
        label = ZoomableImageLabel()
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        self.image_tabs.addTab(scroll, title)
        return label

    # ------------------------------------------------------------------ #
    #  Side panel                                                        #
    # ------------------------------------------------------------------ #
    def _build_side_panel(self) -> QTabWidget:
        panel = QTabWidget()
        panel.addTab(self._build_results_tab(), "Results")
        panel.addTab(self._build_clinical_tab(), "Clinical")
        panel.addTab(self._build_lesion_tab(), "Lesion detail")
        panel.addTab(self._build_settings_tab(), "Settings")
        panel.addTab(self._build_info_tab(), "Models / Info")
        panel.addTab(self._build_log_tab(), "Log management")
        return panel

    def _build_log_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.document().setMaximumBlockCount(1000)
        layout.addWidget(self.activity_log)
        button = QPushButton("Export activity log")
        button.clicked.connect(self._export_activity_log)
        layout.addWidget(button)
        self._activity("Application started; primary processing device: CPU")
        return tab

    def _activity(self, message, level="INFO"):
        self.activity_log.append(f"{datetime.now().isoformat(timespec='seconds')} [{level}] {message}")

    def _export_activity_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export activity log", "activity.txt", "Text (*.txt)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as stream:
                    stream.write(self.activity_log.toPlainText() + '\n')
            except OSError as exc:
                self._notify("Log export failed", str(exc), "error")

    def _log_processing_stage(self, value):
        stages = {3: 'Loading image', 10: 'Image decoded', 18: 'Breast preprocessing complete',
                  30: 'Segmentation stage complete', 36: 'Pectoral stage complete',
                  55: 'Mass candidate stage complete', 68: 'Calcification candidate stage complete'}
        if value in stages:
            self._activity(stages[value])

    def _build_results_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.text_statistics = QTextEdit()
        self.text_statistics.setReadOnly(True)
        self.text_statistics.setMaximumHeight(120)
        layout.addWidget(QLabel("Detection statistics:"))
        layout.addWidget(self.text_statistics)

        layout.addWidget(QLabel("Detected regions (click a row to highlight):"))
        self.table_detections = QTableWidget(0, 7)
        self.table_detections.setHorizontalHeaderLabels(
            ["#", "Type", "Conf.", "Location", "Size (mm)", "Shape / Margin", "Status"]
        )
        self.table_detections.verticalHeader().setVisible(False)
        self.table_detections.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_detections.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table_detections.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        header = self.table_detections.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table_detections.itemSelectionChanged.connect(self.on_table_selection)
        layout.addWidget(self.table_detections, 1)

        self.list_detections = QListWidget()
        self.list_detections.setMaximumHeight(110)
        self.list_detections.itemClicked.connect(self.on_detection_list_clicked)
        layout.addWidget(QLabel("Quick list (ignored regions excluded):"))
        layout.addWidget(self.list_detections)

        return widget

    def _build_clinical_tab(self) -> QWidget:
        widget = QWidget()
        outer = QVBoxLayout(widget)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        self.group_assessment = QGroupBox("BI-RADS (software-assisted estimate)")
        assessment_form = QFormLayout(self.group_assessment)
        self.lbl_birads_category = QLabel("N/A")
        self.lbl_birads_category.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.lbl_birads_text = QLabel("N/A")
        self.lbl_birads_text.setWordWrap(True)
        self.lbl_birads_confidence = QLabel("N/A")
        self.lbl_birads_management = QLabel("N/A")
        self.lbl_birads_management.setWordWrap(True)
        assessment_form.addRow("Category:", self.lbl_birads_category)
        assessment_form.addRow("Assessment:", self.lbl_birads_text)
        assessment_form.addRow("Confidence:", self.lbl_birads_confidence)
        assessment_form.addRow("Suggested management:", self.lbl_birads_management)
        layout.addWidget(self.group_assessment)

        self.group_density = QGroupBox("Breast density (ACR BI-RADS)")
        density_form = QFormLayout(self.group_density)
        self.lbl_breast_density_category = QLabel("N/A")
        self.lbl_breast_density_label = QLabel("N/A")
        self.lbl_breast_density_label.setWordWrap(True)
        self.lbl_breast_density_percent = QLabel("N/A")
        density_form.addRow("ACR category:", self.lbl_breast_density_category)
        density_form.addRow("Description:", self.lbl_breast_density_label)
        density_form.addRow("Dense area (estimate):", self.lbl_breast_density_percent)
        layout.addWidget(self.group_density)

        self.group_calc = QGroupBox("Calcification analysis")
        calc_form = QFormLayout(self.group_calc)
        self.lbl_calc_counts = QLabel("N/A")
        self.lbl_calc_morphology = QLabel("N/A")
        self.lbl_calc_distribution = QLabel("N/A")
        self.lbl_calc_clusters = QLabel("N/A")
        self.lbl_calc_suspicion = QLabel("N/A")
        calc_form.addRow("Candidates:", self.lbl_calc_counts)
        calc_form.addRow("Dominant morphology:", self.lbl_calc_morphology)
        calc_form.addRow("Dominant distribution:", self.lbl_calc_distribution)
        calc_form.addRow("Groups (cluster criterion):", self.lbl_calc_clusters)
        calc_form.addRow("Suspicion level:", self.lbl_calc_suspicion)
        layout.addWidget(self.group_calc)

        self.group_other = QGroupBox("Examination summary")
        other_form = QFormLayout(self.group_other)
        self.lbl_segmentation_confidence = QLabel("N/A")
        self.lbl_tumor_risk_score = QLabel("N/A")
        self.lbl_quality_score = QLabel("N/A")
        self.lbl_quality_warnings = QLabel("N/A")
        self.lbl_quality_warnings.setWordWrap(True)
        self.lbl_backend = QLabel("N/A")
        self.lbl_backend.setWordWrap(True)
        other_form.addRow("Segmentation confidence:", self.lbl_segmentation_confidence)
        other_form.addRow("Highest lesion risk score:", self.lbl_tumor_risk_score)
        other_form.addRow("Image quality score:", self.lbl_quality_score)
        other_form.addRow("Quality warnings:", self.lbl_quality_warnings)
        other_form.addRow("Detection backend:", self.lbl_backend)
        layout.addWidget(self.group_other)

        self.text_reasons = QTextEdit()
        self.text_reasons.setReadOnly(True)
        self.text_reasons.setPlaceholderText(
            "The basis of the software-assisted assessment will appear here."
        )
        layout.addWidget(QLabel("Basis of the assessment:"))
        layout.addWidget(self.text_reasons, 1)

        disclaimer = QLabel(Config.DISCLAIMER)
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet(
            "color:#7a4b00; background:#fff6e5; border:1px solid #e0b96a;"
            "padding:6px; font-size:11px;"
        )
        layout.addWidget(disclaimer)

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return widget

    def _build_lesion_tab(self) -> QWidget:
        widget = QWidget()
        outer = QVBoxLayout(widget)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        self.lbl_selected_lesion = QLabel("No region selected")
        self.lbl_selected_lesion.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.lbl_selected_lesion)

        self.lbl_mass_evidence = QLabel()
        self.lbl_mass_evidence.setWordWrap(True)
        layout.addWidget(self.lbl_mass_evidence)
        self.mask_mode = QComboBox()
        self.mask_mode.addItems(["Inspect / pan", "Add pixels (click)", "Remove pixels (click)"])
        self.mask_mode.currentIndexChanged.connect(self.on_visualization_changed)
        layout.addWidget(self.mask_mode)
        edit_actions = QHBoxLayout()
        for text, slot in (("Undo brush", self._undo_mask_edit), ("Restore automatic", self._restore_automatic_mask)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            edit_actions.addWidget(button)
        layout.addLayout(edit_actions)

        self.group_characterization = QGroupBox("Mass characterisation (ACR lexicon)")
        char_form = QFormLayout(self.group_characterization)
        self.lbl_shape = QLabel("N/A")
        self.lbl_margin = QLabel("N/A")
        self.lbl_lesion_density = QLabel("N/A")
        self.lbl_orientation = QLabel("N/A")
        self.lbl_spiculation = QLabel("N/A")
        char_form.addRow("Shape:", self.lbl_shape)
        char_form.addRow("Margin:", self.lbl_margin)
        char_form.addRow("Density:", self.lbl_lesion_density)
        char_form.addRow("Orientation:", self.lbl_orientation)
        char_form.addRow("Spiculation index:", self.lbl_spiculation)
        layout.addWidget(self.group_characterization)

        self.group_measurement = QGroupBox("Measurements")
        measure_form = QFormLayout(self.group_measurement)
        self.lbl_max_diameter = QLabel("N/A")
        self.lbl_area = QLabel("N/A")
        self.lbl_perimeter = QLabel("N/A")
        self.lbl_circularity = QLabel("N/A")
        self.lbl_measurement_basis = QLabel("N/A")
        measure_form.addRow("Max diameter:", self.lbl_max_diameter)
        measure_form.addRow("Area:", self.lbl_area)
        measure_form.addRow("Perimeter:", self.lbl_perimeter)
        measure_form.addRow("Circularity:", self.lbl_circularity)
        measure_form.addRow("Basis:", self.lbl_measurement_basis)
        layout.addWidget(self.group_measurement)

        self.group_explain = QGroupBox("Explainability scores")
        explain_form = QFormLayout(self.group_explain)
        self.lbl_contrast_score = QLabel("N/A")
        self.lbl_texture_score = QLabel("N/A")
        self.lbl_shape_score = QLabel("N/A")
        self.lbl_edge_irregularity_score = QLabel("N/A")
        self.lbl_density_difference_score = QLabel("N/A")
        explain_form.addRow("Contrast score:", self.lbl_contrast_score)
        explain_form.addRow("Texture score:", self.lbl_texture_score)
        explain_form.addRow("Shape score:", self.lbl_shape_score)
        explain_form.addRow("Edge irregularity:", self.lbl_edge_irregularity_score)
        explain_form.addRow("Density difference:", self.lbl_density_difference_score)
        layout.addWidget(self.group_explain)

        self.text_explanation = QTextEdit()
        self.text_explanation.setReadOnly(True)
        self.text_explanation.setPlaceholderText(
            "Select a detected region to see why it was flagged."
        )
        layout.addWidget(QLabel("Human-readable explanation:"))
        layout.addWidget(self.text_explanation, 1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return widget

    def _build_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QGridLayout(widget)

        layout.addWidget(QLabel("<b>Mass detection</b>"), 0, 0, 1, 2)

        layout.addWidget(QLabel("Minimum area (px):"), 1, 0)
        self.spin_min_tumor_area = QSpinBox()
        self.spin_min_tumor_area.setRange(30, 2000)
        self.spin_min_tumor_area.setValue(self.cad_system.tumor_detector.min_tumor_area)
        layout.addWidget(self.spin_min_tumor_area, 1, 1)

        layout.addWidget(QLabel("Suspicious threshold:"), 2, 0)
        self.slider_suspicious = QSlider(Qt.Orientation.Horizontal)
        self.slider_suspicious.setRange(10, 95)
        self.slider_suspicious.setValue(
            int(self.cad_system.tumor_detector.suspicious_threshold * 100)
        )
        layout.addWidget(self.slider_suspicious, 2, 1)

        layout.addWidget(QLabel("Confirmed threshold:"), 3, 0)
        self.slider_confirmed = QSlider(Qt.Orientation.Horizontal)
        self.slider_confirmed.setRange(10, 99)
        self.slider_confirmed.setValue(
            int(self.cad_system.tumor_detector.confirmed_threshold * 100)
        )
        layout.addWidget(self.slider_confirmed, 3, 1)

        layout.addWidget(QLabel("<b>Calcification detection</b>"), 4, 0, 1, 2)

        layout.addWidget(QLabel("Minimum confidence:"), 5, 0)
        self.slider_calc_confidence = QSlider(Qt.Orientation.Horizontal)
        self.slider_calc_confidence.setRange(10, 95)
        self.slider_calc_confidence.setValue(
            int(self.cad_system.calcification_detector.min_detection_confidence * 100)
        )
        layout.addWidget(self.slider_calc_confidence, 5, 1)

        layout.addWidget(QLabel("Maximum reports:"), 6, 0)
        self.spin_calc_max = QSpinBox()
        self.spin_calc_max.setRange(5, 300)
        self.spin_calc_max.setValue(self.cad_system.calcification_detector.max_detections)
        layout.addWidget(self.spin_calc_max, 6, 1)

        layout.addWidget(QLabel("<b>Display</b>"), 7, 0, 1, 2)
        layout.addWidget(QLabel("Mask opacity (%):"), 8, 0)
        self.settings_slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.settings_slider_opacity.setRange(0, 100)
        self.settings_slider_opacity.setValue(self.slider_opacity.value())
        self.settings_slider_opacity.valueChanged.connect(self.on_settings_opacity_changed)
        layout.addWidget(self.settings_slider_opacity, 8, 1)

        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["auto", "on", "off"])
        self.combo_backend.setCurrentText(Config.DETECTION_BACKEND)
        layout.addWidget(QLabel("Detection backend:"), 9, 0)
        layout.addWidget(self.combo_backend, 9, 1)

        self.btn_apply_settings = QPushButton("Apply settings")
        self.btn_apply_settings.clicked.connect(self.apply_settings)
        layout.addWidget(self.btn_apply_settings, 10, 0, 1, 2)

        note = QLabel(
            "Settings are applied to the detectors and stored with QSettings.\n"
            "They do not modify any trained model or image file."
        )
        note.setWordWrap(True)
        layout.addWidget(note, 11, 0, 1, 2)
        layout.setRowStretch(12, 1)
        return widget

    def _build_info_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.text_models = QTextEdit()
        self.text_models.setReadOnly(True)
        layout.addWidget(self.text_models)

        self.text_help = QTextEdit()
        self.text_help.setReadOnly(True)
        self.text_help.setMaximumHeight(220)
        self.text_help.setHtml(
            """
            <h3>Mammography CAD</h3>
            <p><b>Navigation:</b> mouse wheel zooms, left drag pans,
            double click fits the image, right click ignores/restores a region.</p>
            <p><b>Colours:</b> red = confirmed mass candidate, orange = suspicious
            mass candidate, yellow = microcalcification, cyan = macrocalcification,
            green dashed = selected region.</p>
            <p><b>Detection backends:</b> a trained YOLO/RT-DETR checkpoint is used
            when one is found in <code>model/</code>; otherwise the deterministic
            classical detectors run and the report states this explicitly.</p>
            """
        )
        layout.addWidget(self.text_help)
        return widget

    # ================================================================== #
    #  Model status                                                      #
    # ================================================================== #
    def refresh_model_status(self) -> None:
        runtime = runtime_status()
        registry = ModelRegistry()
        models = registry.discover()
        try:
            status = backend_status()
        except Exception as exc:  # pragma: no cover - defensive
            status = {"error": str(exc)}

        lines = ["<h3>Detection backends</h3>", "<table cellspacing='4'>"]
        for task in ("mass", "calcification"):
            info = status.get(task, {}) or {}
            lines.append(
                f"<tr><td><b>{task.capitalize()}</b></td><td>"
                f"{info.get('backend', 'classical')}"
                + (f" - {info.get('model')}" if info.get("model") else "")
                + (f"<br><i>{info.get('reason')}</i>" if info.get("reason") else "")
                + "</td></tr>"
            )
        lines.append("</table>")

        lines.append("<h3>Runtime</h3><ul>")
        lines.append(
            f"<li>torch: {'yes ' + runtime['torch_version'] if runtime['torch'] else 'not installed'}</li>"
        )
        lines.append(
            f"<li>ultralytics: {'yes ' + runtime['ultralytics_version'] if runtime['ultralytics'] else 'not installed'}</li>"
        )
        lines.append(f"<li>onnxruntime: {'yes' if runtime['onnxruntime'] else 'not installed'}</li>")
        lines.append(f"<li>device: {Config.DL_DEVICE}</li></ul>")

        lines.append("<h3>Checkpoints found</h3>")
        if not models:
            lines.append(
                "<p>No checkpoint files were found in:<br><code>"
                + "<br>".join(registry.directories)
                + "</code><br>Place trained weights there (they are never "
                "modified by the application).</p>"
            )
        else:
            lines.append(
                "<table border='1' cellpadding='4' cellspacing='0'>"
                "<tr><th>File</th><th>Format</th><th>Task</th><th>Classes</th>"
                "<th>Routed to</th><th>SHA-256</th></tr>"
            )
            for info in models:
                lines.append(
                    "<tr>"
                    f"<td>{info.filename}</td><td>{info.file_format}</td>"
                    f"<td>{info.task or 'n/a'}</td>"
                    f"<td>{info.num_classes}: {', '.join(info.class_names[:8]) or '-'}</td>"
                    f"<td>{info.task_routing}</td>"
                    f"<td style='font-size:10px'>{info.sha256[:16]}...</td>"
                    "</tr>"
                )
            lines.append("</table>")

        self.text_models.setHtml("".join(lines))

    # ================================================================== #
    #  Loading and processing                                            #
    # ================================================================== #
    def on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select mammography image", "", IMAGE_FILTER
        )
        if not path:
            return
        self.load_image(path)

    def load_image(self, path: str) -> bool:
        """Load a file for display (analysis is a separate, explicit step)."""
        try:
            from ..processing.preprocessor import MammographyPreprocessor

            gray, metadata = MammographyPreprocessor.load_image(path)
        except Exception as exc:
            self._activity("Image load failed; see error dialog for details", "ERROR")
            self._notify("Error", f"Failed to load image:\n{exc}", "error")
            return False

        self._activity("Image loaded")
        self._mask_undo = None
        self.lbl_original.mask_brush_active = False
        self.current_image_path = path
        self.current_result = None
        self.current_overlay = None
        self.current_detections = []
        self.current_statistics = {}
        self.processing_time = 0.0
        self.selected_detection_id = None

        self.lbl_original.set_image(gray)
        self.lbl_original.fit_to_window()
        self.lbl_original.set_detections([])
        self.lbl_original.highlight_detection(None)
        self.lbl_enhanced.clear_image()
        self.lbl_mask.clear_image()
        self.lbl_density.clear_image()
        self.lbl_heatmap.clear_image()

        self.update_results_display()
        self.clear_clinical_summary()
        self.clear_lesion_panel()

        self.btn_process.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)

        self.lbl_processing_info.setText("Loaded")
        view = metadata.get("image", {}).get("view_position", "")
        spacing = metadata.get("image", {}).get("pixel_spacing", [1.0, 1.0])
        source = metadata.get("source", "standard")
        self.status.showMessage(
            f"Loaded {os.path.basename(path)} ({source}"
            + (f", view {view}" if view else "")
            + f", spacing {spacing[0]}x{spacing[1]} mm)"
        )
        return True

    def on_process(self) -> None:
        if not self.current_image_path:
            self._notify("No image", "Please load a mammogram first.", "warn")
            return

        self.btn_process.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)

        self.worker = CADWorker(self.current_image_path, self, self.cad_system)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.progress.connect(self._log_processing_stage)
        self.worker.result_ready.connect(self.on_analysis_ready)
        self.worker.error.connect(self.on_process_error)
        self.worker.cancelled.connect(self.on_process_cancelled)
        self.worker.start()
        self.status.showMessage("Processing...")

    def on_cancel(self) -> None:
        if self.batch_running and self.batch_worker is not None:
            self.batch_worker.cancel()
            self.status.showMessage("Cancelling batch after the current file...")
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        self._set_idle("Processing cancelled")
        self.progress.setValue(0)

    def _set_idle(self, message: str = "Ready") -> None:
        self.btn_process.setEnabled(bool(self.current_image_path))
        self.btn_load.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status.showMessage(message)

    @pyqtSlot(object)
    def on_analysis_ready(self, result: AnalysisResult) -> None:
        self._mask_undo = None
        self.mask_mode.setCurrentIndex(0)
        self.current_result = result
        self.current_detections = list(result.detections)
        self.current_statistics = dict(result.statistics or {})
        self.processing_time = float(result.processing_time or 0.0)
        self._activity(f"Analysis returned {len(result.detections)} findings in {self.processing_time:.2f}s")
        if result.warnings:
            self._activity(f"{len(result.warnings)} analysis warning(s); see result report", "WARNING")
        self.selected_detection_id = None

        self.lbl_processing_info.setText(
            f"Time: {self.processing_time:.2f}s | "
            f"{len(result.detections)} detection(s)"
        )

        self.update_visualization()
        self._update_image_tabs(result)
        self.update_results_display()
        self.update_clinical_summary()
        self.clear_lesion_panel()
        self.refresh_model_status()

        # Auto-select the most suspicious finding so the lesion panel is never
        # empty after an analysis: the reviewer starts at the top-ranked lesion
        # instead of having to guess where the interesting region is.
        top = self._top_detection()
        if top is not None:
            self.on_detection_clicked(top.id)

        self.btn_process.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.progress.setValue(100)

        birads = (result.birads or {}).get("category")
        density = (result.breast_density or {}).get("acr_category")
        self.status.showMessage(
            f"Analysis completed in {self.processing_time:.2f}s | "
            f"BI-RADS {birads} (software-assisted) | density {density}"
        )

    def on_process_error(self, message: str) -> None:
        self._activity("Analysis failed; see error dialog for details", "ERROR")
        self._notify("Processing error", f"Analysis failed:\n{message}", "error")
        self._set_idle("Processing failed")
        self.progress.setValue(0)

    def on_process_cancelled(self) -> None:
        self._set_idle("Processing cancelled")
        self.progress.setValue(0)

    def _update_image_tabs(self, result: AnalysisResult) -> None:
        if result.enhanced is not None:
            self.lbl_enhanced.set_image(result.enhanced)
        if result.breast_mask is not None:
            mask_vis = cv2.cvtColor(result.breast_mask, cv2.COLOR_GRAY2BGR)
            if result.pectoral_mask is not None and np.any(result.pectoral_mask):
                mask_vis[result.pectoral_mask > 0] = (255, 0, 255)
            self.lbl_mask.set_image(mask_vis)
        if result.density_heatmap is not None:
            self.lbl_density.set_image(result.density_heatmap)
        else:
            self.lbl_density.clear_image()
        if result.heatmap_image is not None:
            self.lbl_heatmap.set_image(result.heatmap_image)
        else:
            self.lbl_heatmap.clear_image()

    # ================================================================== #
    #  Visualisation                                                     #
    # ================================================================== #
    def on_visualization_changed(self) -> None:
        if self.current_result is not None:
            self.update_visualization()

    def on_settings_opacity_changed(self, value: int) -> None:
        self.slider_opacity.setValue(value)

    def update_visualization(self) -> None:
        """Re-render the overlay from the stored result and current controls."""
        result = self.current_result
        if result is None:
            return

        visualizer = self.cad_system.visualizer
        visualizer.show_labels = self.check_show_labels.isChecked()
        visualizer.show_confidence = self.check_show_confidence.isChecked()
        visualizer.show_calc_labels = self.check_show_calc_labels.isChecked()
        visualizer.show_calc_masks = self.check_show_calc_masks.isChecked()
        visualizer.show_cluster_outlines = self.check_show_clusters.isChecked()
        visualizer.show_tumors = self.check_show_tumors.isChecked()
        visualizer.show_calcifications = self.check_show_calcs.isChecked()
        visualizer.show_arrows = self.check_show_arrows.isChecked()
        visualizer.show_masks = self.check_show_mask_layer.isChecked()
        visualizer.show_pectoral = self.check_show_pectoral.isChecked()
        visualizer.show_density_heatmap = False  # density gets its own tab
        visualizer.line_thickness = self.slider_line_thickness.value()
        visualizer.mask_opacity = (
            self.slider_opacity.value() / 100.0
            if self.check_show_mask_layer.isChecked()
            else 0.0
        )

        try:
            self.current_overlay = visualizer.visualize_detections(
                result.original,
                result.detections,
                highlight_id=self.selected_detection_id,
                pectoral_mask=result.pectoral_mask
                if self.check_show_pectoral.isChecked()
                else None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Overlay rendering failed")
            self.status.showMessage(f"Overlay rendering failed: {exc}")
            self.current_overlay = cv2.cvtColor(result.original, cv2.COLOR_GRAY2BGR)

        original_only = self.check_show_original.isChecked()
        self.lbl_original.mask_brush_active = not original_only and self.mask_mode.currentIndex() != 0
        self.lbl_original.set_image(result.original if original_only else self.current_overlay)
        self.lbl_original.set_detections([] if original_only else result.detections)
        self.lbl_original.highlight_detection(
            None if original_only or not self.check_show_tumors.isChecked() else self.selected_detection_id
        )

    # ================================================================== #
    #  Selection                                                         #
    # ================================================================== #
    def on_detection_clicked(self, detection_id: int) -> None:
        self.selected_detection_id = detection_id
        self._select_table_row(detection_id)
        self.update_visualization()
        self.show_lesion_details(detection_id)

    def on_detection_list_clicked(self, item: QListWidgetItem) -> None:
        detection_id = item.data(Qt.ItemDataRole.UserRole)
        if detection_id is not None:
            self.on_detection_clicked(int(detection_id))

    def on_table_selection(self) -> None:
        rows = self.table_detections.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table_detections.item(rows[0].row(), 0)
        if item is None:
            return
        detection_id = item.data(Qt.ItemDataRole.UserRole)
        if detection_id is None:
            return
        self.selected_detection_id = int(detection_id)
        self.update_visualization()
        self.show_lesion_details(int(detection_id))

    def _select_table_row(self, detection_id: int) -> None:
        for row in range(self.table_detections.rowCount()):
            item = self.table_detections.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == detection_id:
                self.table_detections.blockSignals(True)
                self.table_detections.selectRow(row)
                self.table_detections.blockSignals(False)
                return

    def _top_detection(self) -> Optional[Detection]:
        """Highest-ranked non-ignored detection (risk score, then confidence)."""
        candidates = [
            d for d in (self.current_result.detections if self.current_result else [])
            if not d.is_ignored
        ]
        if not candidates:
            return None

        def rank(detection: Detection):
            # Masses first: the lesion-detail panel is built around the ACR mass
            # lexicon; among equal types the risk score, then confidence, wins.
            is_mass = "tumor" in getattr(detection.type, "value", "")
            risk = float((detection.risk_scores or {}).get("risk_score", 0.0) or 0.0)
            return (is_mass, risk, float(detection.confidence or 0.0))

        return max(candidates, key=rank)

    def get_detection_by_id(self, detection_id: int) -> Optional[Detection]:
        for detection in self.current_detections:
            if getattr(detection, "id", None) == detection_id:
                return detection
        return None

    # ================================================================== #
    #  Results / clinical panels                                         #
    # ================================================================== #
    def update_results_display(self) -> None:
        statistics = self.current_statistics or {}
        if statistics:
            conf = statistics.get("average_confidence", 0.0) or 0.0
            self.text_statistics.setPlainText(
                f"Total regions: {statistics.get('total_detections', 0)}\n"
                f"Mass candidates: {statistics.get('mass_count', statistics.get('tumors_confirmed', 0) + statistics.get('tumors_suspicious', 0))}"
                f"  (confirmed {statistics.get('tumors_confirmed', 0)}, "
                f"suspicious {statistics.get('tumors_suspicious', 0)})\n"
                f"Microcalcifications: {statistics.get('microcalcifications', 0)}  |  "
                f"Macrocalcifications: {statistics.get('macrocalcifications', 0)}\n"
                f"Average confidence: {conf:.1%}   |   "
                f"Ignored (false positive): {statistics.get('ignored_count', 0)}"
            )
        else:
            self.text_statistics.clear()

        self._fill_detection_table()
        self._fill_detection_list()

    def _fill_detection_table(self) -> None:
        detections = sorted(
            self.current_detections, key=lambda d: getattr(d, "id", 0)
        )
        self.table_detections.blockSignals(True)
        self.table_detections.setRowCount(0)
        for index, detection in enumerate(detections, start=1):
            row = self.table_detections.rowCount()
            self.table_detections.insertRow(row)

            features = detection.features or {}
            characterization = features.get("mass_characterization") or {}
            measurement = features.get("measurement") or {}
            shape = characterization.get("shape", "-")
            margin = characterization.get("margin", "-")

            values = [
                str(index),
                detection.type.value.replace("_", " ").title(),
                f"{detection.confidence:.2f}",
                f"{detection.location[0]}, {detection.location[1]}",
                (
                    f"{measurement.get('max_diameter_mm', 0):.1f} mm"
                    if measurement.get('spacing_available') else
                    f"{measurement.get('max_diameter_px', 0):.1f} px" if measurement else "-"
                ),
                f"{shape} / {margin}" if characterization else "-",
                "ignored" if detection.is_ignored else "included",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, detection.id)
                if detection.is_ignored:
                    item.setForeground(QColor(150, 150, 150))
                self.table_detections.setItem(row, column, item)
        self.table_detections.blockSignals(False)

    def _fill_detection_list(self) -> None:
        self.list_detections.clear()
        for detection in self.current_detections:
            if detection.is_ignored:
                continue
            text = (
                f"{detection.lesion_id or detection.id}. "
                f"{detection.type.value.replace('_', ' ').title()} "
                f"(score {detection.confidence:.2f})"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, detection.id)
            self.list_detections.addItem(item)

    def update_clinical_summary(self) -> None:
        result = self.current_result
        if result is None:
            self.clear_clinical_summary()
            return

        birads = result.birads or result.birads_result or {}
        density = result.breast_density or result.density_result or {}
        calc = result.calcification_analysis or {}
        statistics = self.current_statistics or {}
        quality = result.quality_report or {}

        category = birads.get("category")
        subcategory = birads.get("subcategory")
        if category is None:
            self.lbl_birads_category.setText("N/A")
        else:
            label = f"BI-RADS {category}"
            if subcategory:
                label += f" ({subcategory})"
            self.lbl_birads_category.setText(label)
        self.lbl_birads_text.setText(birads.get("category_text") or "N/A")
        self.lbl_birads_confidence.setText(self._format_metric(birads.get("confidence"), percent=True))
        self.lbl_birads_management.setText(birads.get("management") or "N/A")

        self.lbl_breast_density_category.setText(
            str(density.get("acr_category") or "N/A")
        )
        self.lbl_breast_density_label.setText(
            density.get("category_label")
            or density.get("category")
            or "N/A"
        )
        dense_pct = (density.get("statistics") or {}).get("dense_area_percentage")
        self.lbl_breast_density_percent.setText(
            f"{dense_pct:.1f} % (software estimate)" if dense_pct is not None else "N/A"
        )

        if calc.get("available"):
            self.lbl_calc_counts.setText(
                f"{calc.get('microcalcification_count', 0)} micro / "
                f"{calc.get('macrocalcification_count', 0)} macro"
            )
            self.lbl_calc_morphology.setText(str(calc.get("dominant_morphology") or "not assessed"))
            self.lbl_calc_distribution.setText(str(calc.get("dominant_distribution") or "not assessed"))
            self.lbl_calc_clusters.setText(str(calc.get("cluster_count", 0)))
            self.lbl_calc_suspicion.setText(
                f"{calc.get('suspicion_level', 'n/a')} "
                f"(score {calc.get('suspicion_score', 0):.2f})"
            )
        else:
            for label in (
                self.lbl_calc_counts, self.lbl_calc_morphology,
                self.lbl_calc_distribution, self.lbl_calc_clusters,
                self.lbl_calc_suspicion,
            ):
                label.setText("N/A" if label is not self.lbl_calc_counts else "no calcification detected")

        self.lbl_segmentation_confidence.setText(
            self._format_metric(statistics.get("segmentation_confidence"), percent=True)
        )
        self.lbl_tumor_risk_score.setText(
            self._format_metric(statistics.get("tumor_risk_score"), percent=False)
        )
        self.lbl_quality_score.setText(
            self._format_metric(quality.get("quality_score"), percent=True)
        )
        warnings = quality.get("warnings") or []
        self.lbl_quality_warnings.setText("; ".join(warnings) if warnings else "none")

        model_status = result.model_status or {}
        backend_bits = []
        for task in ("mass", "calcification"):
            info = (model_status or {}).get(task, {}) or {}
            backend_bits.append(
                f"{task}: {info.get('backend', 'unknown')}"
                + (f" ({info.get('model')})" if info.get("model") else "")
            )
        self.lbl_backend.setText(
            "; ".join(backend_bits)
            + ("  [UNVALIDATED FALLBACK]" if birads.get("validation_status")
               == "unvalidated_classical_fallback" else "")
        )

        reasons = birads.get("reasons") or []
        extra = []
        if birads.get("validation_warning"):
            extra.append("VALIDATION: " + str(birads["validation_warning"]))
        if result.warnings:
            extra.append("Analysis warnings:\n- " + "\n- ".join(result.warnings))
        if calc.get("note"):
            extra.append(calc["note"])
        self.text_reasons.setPlainText(
            ("Basis of the assessment:\n- " + "\n- ".join(reasons) if reasons else "")
            + ("\n\n" + "\n\n".join(extra) if extra else "")
        )

    def clear_clinical_summary(self) -> None:
        for label in (
            self.lbl_birads_category, self.lbl_birads_text, self.lbl_birads_confidence,
            self.lbl_birads_management, self.lbl_breast_density_category,
            self.lbl_breast_density_label, self.lbl_breast_density_percent,
            self.lbl_calc_counts, self.lbl_calc_morphology, self.lbl_calc_distribution,
            self.lbl_calc_clusters, self.lbl_calc_suspicion,
            self.lbl_segmentation_confidence, self.lbl_tumor_risk_score,
            self.lbl_quality_score, self.lbl_quality_warnings, self.lbl_backend,
        ):
            label.setText("N/A")
        self.text_reasons.clear()

    # ================================================================== #
    #  Lesion detail                                                     #
    # ================================================================== #
    def _editable_mass(self):
        if self.current_result is None or self.check_show_original.isChecked():
            return None
        detection = self.get_detection_by_id(self.selected_detection_id)
        if detection and not detection.is_ignored and detection.mask is not None and detection.type in (
                DetectionType.TUMOR_SUSPICIOUS, DetectionType.TUMOR_CONFIRMED):
            return detection
        return None

    def _apply_mask_brush(self, x, y):
        detection = self._editable_mass()
        mode = self.mask_mode.currentIndex()
        if detection is None or not mode:
            self.status.showMessage("Select a mass with an automatic mask before editing.")
            return
        h, w = detection.mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        radius = Config.MASK_BRUSH_RADIUS_PX
        x0, x1, y0, y1 = max(0,x-radius), min(w,x+radius+1), max(0,y-radius), min(h,y+radius+1)
        patch = detection.mask[y0:y1,x0:x1]
        yy, xx = np.ogrid[y0:y1,x0:x1]
        brush = (xx-x)**2+(yy-y)**2 <= radius**2
        if mode == 1:
            brush &= self.current_result.breast_mask[y0:y1,x0:x1] > 0
            if not np.any((patch > 0) & cv2.dilate(brush.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)):
                self.status.showMessage("Add pixels next to the selected mask boundary.")
                return
        new_value = 255 if mode == 1 else 0
        if not np.any(brush & (patch != new_value)):
            return
        if mode == 2 and np.count_nonzero(patch[brush]) == np.count_nonzero(detection.mask):
            self.status.showMessage("The final remaining mask pixels cannot be erased.")
            return
        if detection.automatic_mask is None:
            detection.automatic_mask = detection.mask.copy()
        self._mask_undo = (detection, (x0,y0,x1,y1), patch.copy())
        patch[brush] = new_value
        self._refresh_mask_measurements(detection)

    def _undo_mask_edit(self):
        detection = self._editable_mass()
        if self._mask_undo is None or self._mask_undo[0] is not detection:
            return
        _, (x0,y0,x1,y1), patch = self._mask_undo
        detection.mask[y0:y1,x0:x1] = patch
        self._mask_undo = None
        self._refresh_mask_measurements(detection)

    def _restore_automatic_mask(self):
        detection = self._editable_mass()
        if detection is not None and detection.automatic_mask is not None:
            detection.mask = detection.automatic_mask.copy()
            self._mask_undo = None
            self._refresh_mask_measurements(detection)

    def _refresh_mask_measurements(self, detection):
        mask = detection.mask
        detection.bbox = cv2.boundingRect(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        detection.contour = max(contours, key=cv2.contourArea)
        edited = not np.array_equal(mask, detection.automatic_mask)
        detection.features['mask_origin'] = 'manual correction' if edited else 'automatic'
        meta = self.current_result.image_metadata or {}
        row, col = meta.get('pixel_spacing', (1., 1.))
        measurement = self.cad_system.measurement_engine.measure_detection(
            detection, col, row, spacing_known=bool(meta.get('spacing_known')))
        for name in ('area_px', 'area_mm2', 'diameter_px', 'diameter_mm'):
            setattr(detection, name, float(measurement.get(name, 0) or 0))
        detection.features['measurement'] = measurement
        self.current_result.measurements = [d.features['measurement'] for d in self.current_result.detections
                                            if d.features.get('measurement')]
        warning = 'Manual mask corrections update geometry only; automated scores and descriptors remain from the original run.'
        if edited and warning not in self.current_result.warnings:
            self.current_result.warnings.append(warning)
        self.update_visualization()
        self.update_results_display()
        self.show_lesion_details(detection.id)

    def show_lesion_details(self, detection_id: int) -> None:
        detection = self.get_detection_by_id(detection_id)
        if detection is None:
            self.clear_lesion_panel()
            return

        features = detection.features or {}
        characterization = features.get("mass_characterization") or {}
        measurement = features.get("measurement") or {}
        explainability = detection.explainability or features.get("explainability") or {}
        scores = explainability.get("scores", {}) if isinstance(explainability, dict) else {}

        self.lbl_selected_lesion.setText(
            f"{detection.lesion_id or detection.id} - "
            f"{detection.type.value.replace('_', ' ').title()} "
            f"(evidence score {detection.confidence:.2f})"
        )

        if 'segmentation_confidence' in features:
            state = features.get('mask_origin', 'automatic' if detection.mask is not None else 'no reliable mask')
            self.lbl_mass_evidence.setText(
                f"Mask: {state}; automatic agreement score: {features['segmentation_confidence']:.2f}. "
                f"Detector support: {features['detector_support']}/5. "
                f"Pectoral overlap: {features.get('pectoral_overlap', 0.):.0%}. "
                + ('Review advised.' if features.get('needs_review') else ''))
        else:
            self.lbl_mass_evidence.clear()

        if characterization.get("available"):
            self.lbl_shape.setText(str(characterization.get("shape")))
            self.lbl_margin.setText(str(characterization.get("margin")))
            self.lbl_lesion_density.setText(str(characterization.get("density")))
            self.lbl_orientation.setText(str(characterization.get("orientation")))
            self.lbl_spiculation.setText(
                self._format_metric(characterization.get("spiculation_index"))
            )
        else:
            reason = characterization.get("reason", "no mass characterisation for this region")
            for label in (
                self.lbl_shape, self.lbl_margin, self.lbl_lesion_density,
                self.lbl_orientation, self.lbl_spiculation,
            ):
                label.setText(reason)

        if measurement:
            self.lbl_max_diameter.setText(
                f"{measurement.get('max_diameter_mm', 0):.2f} mm "
                f"({measurement.get('max_diameter_px', 0):.0f} px)"
            )
            self.lbl_area.setText(
                f"{measurement.get('area_mm2', 0):.2f} mm2 "
                f"({measurement.get('area_px', 0):.0f} px)"
            )
            self.lbl_perimeter.setText(
                f"{measurement.get('perimeter_mm', 0):.2f} mm"
            )
            self.lbl_circularity.setText(self._format_metric(measurement.get("circularity")))
            if not measurement.get("spacing_available"):
                self.lbl_max_diameter.setText(f"{measurement.get('max_diameter_px', 0):.1f} px")
                self.lbl_area.setText(f"{measurement.get('area_px', 0):.0f} px²")
                self.lbl_perimeter.setText(f"{measurement.get('perimeter_px', 0):.1f} px")
            self.lbl_measurement_basis.setText(
                measurement.get("spacing_basis")
                or (
                    "DICOM pixel spacing"
                    if measurement.get("spacing_available")
                    else "pixel units (pixel spacing unavailable)"
                )
            )
        else:
            for label in (
                self.lbl_max_diameter, self.lbl_area, self.lbl_perimeter,
                self.lbl_circularity,
            ):
                label.setText("N/A")
            self.lbl_measurement_basis.setText("no measurement available")

        self.lbl_contrast_score.setText(self._format_metric(scores.get("contrast_score")))
        self.lbl_texture_score.setText(self._format_metric(scores.get("texture_score")))
        self.lbl_shape_score.setText(self._format_metric(scores.get("shape_score")))
        self.lbl_edge_irregularity_score.setText(
            self._format_metric(scores.get("edge_irregularity_score"))
        )
        self.lbl_density_difference_score.setText(
            self._format_metric(scores.get("density_difference_score"))
        )

        explanations = explainability.get("explanations") or []
        summary = explainability.get("summary") or ""
        risk = (detection.risk_scores or {}).get("risk", {})
        lines = []
        if explanations:
            lines.append("Flagged because:\n" + "\n".join(f"- {item}" for item in explanations))
        elif summary:
            lines.append(summary)
        else:
            lines.append("No explainability data for this region.")
        if risk:
            lines.append(
                f"\nRisk score: {risk.get('risk_score', 'n/a')}/100 "
                f"({risk.get('risk_category', 'n/a')})"
            )
        calibration = features.get("calibration") or {}
        if calibration:
            lines.append(
                f"Reliability: {calibration.get('reliability', 'n/a')} | "
                f"Uncertainty: {calibration.get('uncertainty', 'n/a')}"
            )
        self.text_explanation.setPlainText("\n".join(lines))

    def clear_lesion_panel(self) -> None:
        self.lbl_mass_evidence.clear()
        self.lbl_selected_lesion.setText("No region selected")
        for label in (
            self.lbl_shape, self.lbl_margin, self.lbl_lesion_density,
            self.lbl_orientation, self.lbl_spiculation, self.lbl_max_diameter,
            self.lbl_area, self.lbl_perimeter, self.lbl_circularity,
            self.lbl_measurement_basis, self.lbl_contrast_score,
            self.lbl_texture_score, self.lbl_shape_score,
            self.lbl_edge_irregularity_score, self.lbl_density_difference_score,
        ):
            label.setText("N/A")
        self.text_explanation.clear()

    # ================================================================== #
    #  Export / save                                                     #
    # ================================================================== #
    def on_save(self) -> None:
        if self.current_overlay is None:
            self._notify("Nothing to save", "Run the detection first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save overlay image", "overlay.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self.current_overlay)
            self.status.showMessage(f"Overlay saved: {path}")

    def on_export(self) -> None:
        if self.current_result is None:
            self._notify("No results", "Run the detection first.")
            return
        try:
            paths = self.report_builder.save(self.current_result)
        except Exception as exc:
            logger.exception("Report export failed")
            self._notify("Export failed", str(exc), "error")
            return
        extras = []
        for extension, export in (("pdf", self.report_builder.save_pdf), ("docx", self.report_builder.save_docx)):
            try:
                extras.append(export(self.current_result, f"{paths['base']}.{extension}"))
            except ImportError:
                extras.append(f"{extension.upper()} unavailable: install python-docx for Word export")
            except Exception as exc:
                logger.exception("%s report export failed", extension.upper())
                extras.append(f"{extension.upper()} export failed: {type(exc).__name__}")
        self._activity("Report export finished; see export dialog for format status")
        additional = "\n".join(extras)
        self._notify(
            "Report exported",
            "Reports written to:\n\n"
            f"{paths['json']}\n{paths['html']}\n{additional}\n\n"
            "The report is a software-assisted analysis, not a diagnosis.",
        )
        self.status.showMessage(f"Report exported: {paths['html']}")

    def on_batch(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select mammograms to process", "", IMAGE_FILTER
        )
        if not paths:
            return
        self.start_batch(paths)

    def start_batch(self, paths, output_dir: Optional[str] = None) -> bool:
        """Run a batch over ``paths`` (or every image inside a directory).

        Returns ``False`` when a batch is already running.  Exposed separately
        from :meth:`on_batch` so the batch path can be driven programmatically
        (tests, scripting) without a file dialog.
        """
        if isinstance(paths, str):
            if os.path.isdir(paths):
                collected = []
                for name in sorted(os.listdir(paths)):
                    full = os.path.join(paths, name)
                    if os.path.isfile(full) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                        collected.append(full)
                paths = collected
            else:
                paths = [paths]
        paths = list(paths)
        if not paths:
            self.status.showMessage("Batch: no images found")
            return False
        if self.batch_worker is not None and self.batch_worker.isRunning():
            self._notify("Batch running", "A batch is already running.", "info")
            return False

        self.batch_running = True
        self.batch_ok = 0
        self.batch_failed = 0
        self.batch_failures = []
        self.status.showMessage(f"Batch started: {len(paths)} image(s)")

        self.batch_worker = BatchProcessor(
            paths, export_reports=True, output_dir=output_dir, parent=self
        )
        self.batch_worker.progress.connect(
            lambda index, total, name: self.status.showMessage(
                f"Batch {index}/{total}: {name}"
            )
        )
        self.batch_worker.file_done.connect(
            lambda path, status, message: self.status.showMessage(
                f"{os.path.basename(path)}: {status} - {message}"
            )
        )
        self.batch_worker.batch_done.connect(self.on_batch_done)
        self.progress.setRange(0, 0)
        self.btn_batch.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.batch_worker.start()
        return True

    def on_batch_done(self, ok: int, failed: int, failures: list) -> None:
        self.batch_running = False
        self.batch_ok = int(ok)
        self.batch_failed = int(failed)
        self.batch_failures = list(failures or [])
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.btn_batch.setEnabled(True)
        message = f"Batch finished: {ok} succeeded, {failed} failed."
        if failures:
            message += "\n\nFailures:\n" + "\n".join(failures[:10])
        message += f"\n\nReports: {Config.REPORTS_DIR}"
        self.status.showMessage(
            f"Batch finished: {ok} ok, {failed} failed"
        )
        self._notify("Batch complete", message, "info")
        # Refresh the model/status panel: a batch may have produced reports
        self.refresh_model_status()

    def _notify(self, title: str, message: str, level: str = "info") -> None:
        """Show a message box, or just log it in head-less/silent mode."""
        if self.silent_dialogs:
            logger.info("%s: %s", title, message.replace("\n", " "))
            self.status.showMessage(message.replace("\n", " ")[:180])
            return
        if level == "error":
            QMessageBox.critical(self, title, message)
        elif level == "warn":
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.information(self, title, message)

    def on_reset_view(self) -> None:
        for label in (
            self.lbl_original, self.lbl_enhanced, self.lbl_mask,
            self.lbl_density, self.lbl_heatmap,
        ):
            label.reset_view()

    # ================================================================== #
    #  Settings                                                          #
    # ================================================================== #
    def apply_settings(self) -> None:
        tumor = self.cad_system.tumor_detector
        tumor.min_tumor_area = int(self.spin_min_tumor_area.value())
        tumor.suspicious_threshold = self.slider_suspicious.value() / 100.0
        tumor.confirmed_threshold = max(
            tumor.suspicious_threshold + 0.05, self.slider_confirmed.value() / 100.0
        )
        calc = self.cad_system.calcification_detector
        calc.min_detection_confidence = self.slider_calc_confidence.value() / 100.0
        calc.max_detections = int(self.spin_calc_max.value())

        Config.DETECTION_BACKEND = self.combo_backend.currentText()
        # Re-read the backend configuration for the next analysis
        self.cad_system._deep_checked = False
        self.cad_system.deep_mass_backend = None
        self.cad_system.deep_calc_backend = None

        self.settings.setValue("tumor_min_area", tumor.min_tumor_area)
        self.settings.setValue("tumor_suspicious", tumor.suspicious_threshold)
        self.settings.setValue("tumor_confirmed", tumor.confirmed_threshold)
        self.settings.setValue("calc_confidence", calc.min_detection_confidence)
        self.settings.setValue("calc_max", calc.max_detections)
        self.settings.setValue("backend", Config.DETECTION_BACKEND)
        self.settings.sync()

        self.refresh_model_status()
        self.status.showMessage("Settings applied")

    def load_settings(self) -> None:
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        for key, widget in (
            ("show_labels", self.check_show_labels),
            ("show_tumors", self.check_show_tumors),
            ("show_calcs", self.check_show_calcs),
            ("show_arrows", self.check_show_arrows),
            ("show_mask_layer", self.check_show_mask_layer),
        ):
            value = self.settings.value(key)
            if value is not None:
                widget.setChecked(str(value).lower() in ("true", "1"))

        tumor = self.cad_system.tumor_detector
        tumor.min_tumor_area = int(self.settings.value("tumor_min_area", tumor.min_tumor_area))
        tumor.suspicious_threshold = float(
            self.settings.value("tumor_suspicious", tumor.suspicious_threshold)
        )
        tumor.confirmed_threshold = float(
            self.settings.value("tumor_confirmed", tumor.confirmed_threshold)
        )
        calc = self.cad_system.calcification_detector
        calc.min_detection_confidence = float(
            self.settings.value("calc_confidence", calc.min_detection_confidence)
        )
        calc.max_detections = int(self.settings.value("calc_max", calc.max_detections))

        backend = self.settings.value("backend")
        if backend:
            Config.DETECTION_BACKEND = str(backend)
            self.combo_backend.setCurrentText(str(backend))

        self.spin_min_tumor_area.setValue(tumor.min_tumor_area)
        self.slider_suspicious.setValue(int(tumor.suspicious_threshold * 100))
        self.slider_confirmed.setValue(int(tumor.confirmed_threshold * 100))
        self.slider_calc_confidence.setValue(int(calc.min_detection_confidence * 100))
        self.spin_calc_max.setValue(calc.max_detections)

    # ================================================================== #
    #  Utilities                                                         #
    # ================================================================== #
    @staticmethod
    def _format_metric(value, percent: bool = False) -> str:
        if value is None:
            return "N/A"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if percent:
            return f"{number:.1%}" if number <= 1.0 else f"{number:.1f}%"
        return f"{number:.3f}"

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Never destroy a live QThread: it would abort the process.
        for worker in (self.worker, self.batch_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                worker.quit()
                worker.wait(10000)
        self.settings.setValue("geometry", self.saveGeometry())
        for key, widget in (
            ("show_labels", self.check_show_labels),
            ("show_tumors", self.check_show_tumors),
            ("show_calcs", self.check_show_calcs),
            ("show_arrows", self.check_show_arrows),
            ("show_mask_layer", self.check_show_mask_layer),
        ):
            self.settings.setValue(key, widget.isChecked())
        self.settings.sync()

        for worker in (self.worker, self.batch_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                worker.wait(3000)
        super().closeEvent(event)

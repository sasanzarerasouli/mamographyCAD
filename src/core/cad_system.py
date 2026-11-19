"""End-to-end mammography CAD pipeline.

This module is the integration point of the application.  It runs, in order:

1. loading / DICOM decoding and preprocessing (breast mask, pectoral removal,
   contrast enhancement)
2. image quality assessment
3. mass detection (deep model when a checkpoint is available, otherwise the
   classical multi-strategy detector)
4. calcification detection (tiled deep model when available, otherwise the
   classical multi-threshold detector)
5. per-lesion analysis: ACR lexicon characterisation, measurements (px + mm),
   risk scoring, explainability, confidence calibration
6. examination level analysis: breast density (ACR A-D), calcification
   morphology/distribution, suspicion heatmap
7. BI-RADS estimate (0-6) and the clinical summary

Every stage is defensive: a failure in one optional module is recorded in
``AnalysisResult.warnings`` and the pipeline continues, because a partial CAD
result is more useful than a crashed application.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config

from .analysis_result import AnalysisResult
from .detection_types import Detection, DetectionType
from ..models.birads_estimator import BIRADSEstimator
from ..models.breast_density_classifier import BreastDensityClassifier
from ..models.calcification_analysis import CalcificationAnalyzer
from ..models.calcification_detector import CalcificationDetector
from ..models.confidence_analysis import ConfidenceAnalysisModule
from ..models.deep_detector import backend_status, get_backend, last_error
from ..models.explainability_engine import ExplainabilityEngine
from ..models.lesion_characterization import (
    LesionCharacterizer,
    apply_characterization_to_detections,
)
from ..models.model_registry import ModelRegistry
from ..models.suspicion_heatmap_generator import SuspicionHeatmapGenerator
from ..models.tumor_detector import TumorDetector
from ..models.tumor_risk_analyzer import TumorRiskAnalyzer
from ..models.visualizer import ResultVisualizer
from ..measurement.lesion_measurement_engine import LesionMeasurementEngine
from ..measurement.prior_study_comparison_engine import PriorStudyComparisonEngine
from ..processing.pectoral_muscle_analyzer import PectoralMuscleAnalyzer
from ..processing.preprocessor import MammographyPreprocessor
from ..processing.quality_analyzer import MammogramQualityAnalyzer
from ..processing.segmentation_pipeline import SegmentationPipeline

logger = logging.getLogger("MammographyCAD")


class MammographyCADSystem:
    """Coordinates the whole CAD workflow and produces an AnalysisResult."""

    def __init__(self) -> None:
        # detection
        self.tumor_detector = TumorDetector()
        self.calcification_detector = CalcificationDetector()
        self.deep_mass_backend = None
        self.deep_calc_backend = None
        self._deep_checked = False
        #: notes about backend resolution failures (missing checkpoint, disabled
        #: backend) that are appended to the result warnings
        self.backend_notes: List[str] = []

        # processing
        self.preprocessor = MammographyPreprocessor
        self.segmentation_pipeline = SegmentationPipeline()
        self.pectoral_analyzer = PectoralMuscleAnalyzer()
        self.quality_analyzer = MammogramQualityAnalyzer()

        # lesion + examination analysis
        self.characterizer = LesionCharacterizer()
        self.risk_analyzer = TumorRiskAnalyzer()
        self.explainability_engine = ExplainabilityEngine()
        self.confidence_module = ConfidenceAnalysisModule()
        self.measurement_engine = LesionMeasurementEngine()
        self.density_classifier = BreastDensityClassifier()
        self.calcification_analyzer = CalcificationAnalyzer()
        self.birads_estimator = BIRADSEstimator()
        self.heatmap_generator = SuspicionHeatmapGenerator()
        self.prior_comparison_engine = PriorStudyComparisonEngine()

        self.visualizer = ResultVisualizer()
        self.model_registry = ModelRegistry()
        self._next_detection_id = 1

    # ================================================================== #
    #  Backward compatible API                                           #
    # ================================================================== #
    def process(
        self, image_path: str, progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Detection], Dict]:
        """Legacy entry point returning ``(original, enhanced, mask, dets, stats)``."""
        result = self.analyze(image_path, progress_callback)
        return (
            result.original,
            result.enhanced,
            result.breast_mask,
            result.detections,
            result.statistics,
        )

    def process_array(
        self, gray: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """Analyse an in-memory grayscale array (used by tests and batch jobs)."""
        result = self._analyze_gray(
            image_path="<array>",
            gray=MammographyPreprocessor.to_uint8(gray),
            metadata=metadata or {},
            progress_callback=None,
        )
        result.source_pixels = np.array(gray, copy=True)
        result.source_pixels.flags.writeable = False
        return result

    # ================================================================== #
    #  Main entry point                                                  #
    # ================================================================== #
    def analyze(self, image_path: str, progress_callback=None) -> AnalysisResult:
        """Run the full pipeline for one file and return an AnalysisResult."""
        start = time.time()
        if progress_callback:
            progress_callback(3)

        retained_pixels = []
        gray, metadata = MammographyPreprocessor.load_image(image_path, source_pixels=retained_pixels)
        if progress_callback:
            progress_callback(10)

        result = self._analyze_gray(
            image_path=image_path,
            gray=gray,
            metadata=metadata,
            progress_callback=progress_callback,
        )
        if retained_pixels:
            result.source_pixels = retained_pixels[0]
            result.source_pixels.flags.writeable = False
        result.processing_time = time.time() - start
        logger.info("Pipeline duration: %.2fs", result.processing_time)
        return result

    # ================================================================== #
    #  Pipeline                                                          #
    # ================================================================== #
    def _analyze_gray(
        self,
        image_path: str,
        gray: np.ndarray,
        metadata: Dict[str, Any],
        progress_callback=None,
    ) -> AnalysisResult:
        warnings: List[str] = []
        self._next_detection_id = 1

        def progress(value: int) -> None:
            if progress_callback:
                progress_callback(int(value))

        # ---------------- 1. preprocessing ----------------------------- #
        pre = MammographyPreprocessor.preprocess_array(gray, metadata=metadata)
        original = pre["original"]
        enhanced = pre["enhanced"]
        breast_mask = pre["breast_mask"]
        pre_pectoral_mask = pre["pectoral_mask"]
        progress(18)

        detection_metadata = dict(metadata or {})
        image_meta = dict((metadata or {}).get("image", {}) or {})
        spacing = _resolve_spacing(image_meta)
        # Was the spacing measured (DICOM PixelSpacing) or assumed?
        raw_spacing = image_meta.get("pixel_spacing")
        try:
            spacing_known = bool(
                raw_spacing
                and image_meta.get("spacing_known", tuple(raw_spacing) != (1., 1.))
                and np.isfinite(raw_spacing[:2]).all()
                and float(raw_spacing[0]) > 0
                and float(raw_spacing[1]) > 0
            )
        except (TypeError, ValueError, IndexError):
            spacing_known = False
        view_position = str(image_meta.get("view_position", "") or "").upper()
        if not view_position:
            view_position = self._infer_view_position(breast_mask, image_meta)
        laterality = str(image_meta.get("laterality", "") or "").upper()

        # ---------------- 2. quality ----------------------------------- #
        quality_report = self._safe(
            "quality analysis",
            lambda: self.quality_analyzer.analyze(enhanced, breast_mask),
            warnings,
        )
        progress(24)

        # ---------------- 3. segmentation refinement ------------------- #
        segmentation = self._safe(
            "segmentation pipeline",
            lambda: self.segmentation_pipeline.segment(original),
            warnings,
        )
        segmentation_confidence = None
        if segmentation and segmentation.get("warning"):
            warnings.append(str(segmentation["warning"]))
        if segmentation and segmentation.get("mask") is not None:
            segmentation_confidence = self.confidence_module.analyze_module(
                [
                    float(segmentation.get("confidence", 0.0)),
                    float(quality_report.get("quality_score", 0.5))
                    if quality_report else 0.5,
                ],
                label="segmentation",
            )
        progress(30)

        # ---------------- 4. pectoral muscle --------------------------- #
        pectoral_result = self._safe(
            "pectoral muscle analysis",
            lambda: self.pectoral_analyzer.analyze(original, breast_mask, view_position or "MLO"),
            warnings,
        )
        pectoral_mask = pre_pectoral_mask
        pectoral_confidence = None
        if pectoral_result:
            pectoral_confidence = float(pectoral_result.get("confidence", 0.0) or 0.0)
            candidate = pectoral_result.get("mask")
            if isinstance(candidate, np.ndarray) and candidate.size:
                if candidate.shape[:2] == original.shape[:2] and candidate.max() > 0:
                    pectoral_mask = candidate
        progress(36)

        # ---------------- 5. mass detection ---------------------------- #
        mass_input = enhanced
        if np.any(pre_pectoral_mask):
            mass_input = MammographyPreprocessor.enhance_contrast(pre["cleaned"], breast_mask)
        mass_detections = self._detect_masses(mass_input, breast_mask, warnings)
        self._annotate_pectoral_overlap(mass_detections, pectoral_mask, pectoral_confidence)
        progress(55)

        # ---------------- 6. calcification detection ------------------- #
        calc_detections = self._detect_calcifications(enhanced, breast_mask, warnings, spacing=spacing if spacing_known else None)
        progress(68)

        detections = self._finalise_detections(
            original=original,
            enhanced=enhanced,
            breast_mask=breast_mask,
            mass_detections=mass_detections,
            calc_detections=calc_detections,
            spacing=spacing,
            warnings=warnings,
            spacing_known=spacing_known,
        )
        progress(78)

        # ---------------- 7. examination level analysis ---------------- #
        density_result = self._safe(
            "breast density classification",
            lambda: self.density_classifier.classify(enhanced, breast_mask),
            warnings,
        )
        calcification_analysis = self._safe(
            "calcification analysis",
            lambda: self.calcification_analyzer.analyze(
                detections, spacing, spacing_known=spacing_known
            ),
            warnings,
        )
        progress(86)

        # ---------------- 8. heatmap ----------------------------------- #
        heatmap = None
        density_heatmap = None
        if detections:
            heatmap = self._safe(
                "suspicion heatmap",
                lambda: self.heatmap_generator.generate(
                    original,
                    detections,
                    breast_mask,
                    self._risk_payloads(detections),
                ),
                warnings,
            )
        if density_result and density_result.get("heatmap") is not None:
            density_heatmap = density_result.get("heatmap")
        progress(90)

        # ---------------- 9. BI-RADS ----------------------------------- #
        risk_results = self._risk_payloads(detections)
        birads_result = self._safe(
            "BI-RADS estimation",
            lambda: self.birads_estimator.estimate(
                detections=detections,
                density_result=density_result,
                risk_results=risk_results,
                calcification_count=sum(
                    1 for d in detections if self._is_calcification(d)
                ),
                calcification_analysis=calcification_analysis,
                quality_report=quality_report,
                detection_backend=self._active_backend_mode(),
            ),
            warnings,
        )
        progress(94)

        # ---------------- 10. statistics + summary --------------------- #
        statistics = self._calculate_statistics(detections)
        statistics.update(self._clinical_summary_payload(
            birads_result=birads_result,
            density_result=density_result,
            segmentation_confidence=segmentation_confidence,
            quality_report=quality_report,
            detections=detections,
            risk_results=risk_results,
        ))

        tumor_risk = None
        if risk_results:
            tumor_risk = max(
                risk_results,
                key=lambda r: r.get("risk", {}).get("risk_score", 0),
            )

        explainability = self._build_explainability(detections)
        confidence_summary = self._build_confidence_summary(
            detections, quality_report, segmentation_confidence
        )

        model_status = self._safe(
            "model status", backend_status, warnings
        ) or {"mass": {"backend": "classical"}, "calcification": {"backend": "classical"}}
        # Backend resolution notes (missing checkpoint / disabled backend)
        for note in self.backend_notes:
            if note not in warnings:
                warnings.append(note)

        result = AnalysisResult(
            original=original,
            enhanced=enhanced,
            breast_mask=breast_mask,
            detections=detections,
            statistics=statistics,
            pectoral_removed=pre.get("pectoral_removed"),
            pectoral_mask=pectoral_mask,
            quality_report=quality_report,
            segmentation_result=segmentation,
            pectoral_result=pectoral_result,
            density_result=density_result,
            risk_results=risk_results,
            birads_result=birads_result,
            heatmap=heatmap,
            dicom_metadata=detection_metadata,
            measurements=[
                d.features.get("measurement", {}) for d in detections
                if d.features.get("measurement")
            ],
            confidence_summary=confidence_summary,
            backend_info=model_status,
            segmentation_confidence=segmentation_confidence,
            tumor_risk=tumor_risk,
            birads=birads_result,
            breast_density=density_result,
            explainability=explainability,
            pectoral_confidence=pectoral_confidence,
            backend=model_status,
            calcification_analysis=calcification_analysis,
            lesion_characterizations=self._characterization_list(detections),
            heatmap_image=heatmap,
            density_heatmap=density_heatmap,
            model_status=model_status,
            warnings=warnings,
            image_metadata={
                "path": image_path,
                "view_position": view_position,
                "laterality": laterality,
                "pixel_spacing": list(spacing),
                "spacing_known": spacing_known,
                "spacing_assumed": not spacing_known,
                "spacing_used_mm": list(spacing),
                "raw": detection_metadata,
            },
        )

        logger.info("Analysis stages complete: %d detection(s)", len(detections))
        progress(98)
        return result

    @staticmethod
    def _characterization_list(detections: List[Detection]) -> List[Dict[str, Any]]:
        """Characterisations keyed to their detection.

        Previously the list was built from the per-detection dict alone, so the
        GUI had no way of knowing which detection a characterisation belonged
        to (and the list was shorter than the detection list).  Every entry now
        carries ``detection_id``, ``location`` and ``bbox``.
        """
        items: List[Dict[str, Any]] = []
        for detection in detections:
            characterization = (detection.features or {}).get("mass_characterization")
            if not characterization:
                continue
            entry = dict(characterization)
            entry["detection_id"] = detection.id
            entry["location"] = list(detection.location)
            entry["bbox"] = list(detection.bbox) if detection.bbox else None
            entry["detection_type"] = getattr(detection.type, "value", str(detection.type))
            entry["confidence"] = float(detection.confidence)
            items.append(entry)
        return items

    # ================================================================== #
    #  Detection stages                                                  #
    # ================================================================== #
    def _ensure_deep_backends(self) -> None:
        if self._deep_checked:
            return
        self._deep_checked = True
        if not Config.TRUST_LOCAL_CHECKPOINTS or Config.DETECTION_BACKEND == "off":
            self.backend_notes.append(
                "Optional deep inference was disabled or not explicitly trusted: the classical "
                "(unvalidated) detectors were used."
            )
            return
        try:
            self.deep_calc_backend = get_backend("calcification", self.model_registry)
        except Exception as exc:  # pragma: no cover - runtime specific
            logger.warning("Deep backend initialisation failed: %s", exc)
            self.deep_mass_backend = None
            self.deep_calc_backend = None
            self.backend_notes.append(f"Deep backend initialisation failed: {exc}")

        if self.deep_calc_backend is None:
            reason = last_error("calcification") or "no compatible checkpoint"
            self.backend_notes.append(f"Deep calcification backend unavailable ({reason}); classical detector used.")

    @staticmethod
    def _annotate_pectoral_overlap(detections, mask, confidence):
        if mask is None or not np.any(mask):
            return
        strength = float(np.clip(confidence if confidence is not None else 0., 0., 1.))
        for detection in detections:
            x, y = detection.location
            if detection.mask is not None:
                region = detection.mask > 0
                overlap = float(np.count_nonzero(region & (mask > 0)) / max(1, region.sum()))
            else:
                overlap = float(mask[y, x] > 0)
            detection.features['pectoral_overlap'] = overlap
            detection.features['pectoral_mask_score'] = strength
            if overlap >= Config.ANATOMY_PECTORAL_REVIEW_OVERLAP:
                detection.features['needs_review'] = True
                detection.features['pre_anatomy_score'] = detection.confidence
                detection.confidence *= 1 - Config.ANATOMY_PECTORAL_MAX_PENALTY * strength * overlap

    def _detect_masses(
        self, enhanced: np.ndarray, breast_mask: np.ndarray, warnings: List[str]
    ) -> List[Detection]:
        try:
            detections = self.tumor_detector.detect(enhanced, breast_mask)
        except Exception as exc:
            warnings.append(f"Mass detection failed: {exc}")
            logger.exception("Classical mass detection failed")
            return []
        for detection in detections:
            detection.features.setdefault("source", "classical")
        return detections

    def _detect_calcifications(
        self, enhanced: np.ndarray, breast_mask: np.ndarray, warnings: List[str], *, spacing=None
    ) -> List[Detection]:
        self._ensure_deep_backends()

        if self.deep_calc_backend is not None:
            try:
                raw = self.deep_calc_backend.predict(
                    enhanced,
                    tile=Config.CALC_PATCH_ENABLED,
                    patch_size=Config.CALC_PATCH_SIZE,
                    overlap=Config.CALC_PATCH_OVERLAP,
                    scales=(Config.DL_IMAGE_SIZE,),
                    conf=Config.DL_CONF_THRESHOLD,
                )
                detections = self.deep_calc_backend.to_detections(
                    raw, enhanced.shape, breast_mask
                )
                if Config.DETECTION_BACKEND == "on" or detections:
                    logger.info(
                        "Calcification detection: deep backend -> %d", len(detections)
                    )
                    return detections
                warnings.append(
                    "Deep calcification model returned no candidates; classical "
                    "detector was used as a supplementary pass."
                )
            except Exception as exc:
                warnings.append(f"Deep calcification detection failed: {exc}")
                logger.exception("Deep calcification detection failed")

        try:
            detections = self.calcification_detector.detect(enhanced, breast_mask, spacing=spacing)
        except Exception as exc:
            warnings.append(f"Calcification detection failed: {exc}")
            logger.exception("Classical calcification detection failed")
            return []
        for detection in detections:
            detection.features.setdefault("source", "classical")
        return detections

    # ================================================================== #
    #  Per-lesion post processing                                        #
    # ================================================================== #
    def _finalise_detections(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        breast_mask: np.ndarray,
        mass_detections: List[Detection],
        calc_detections: List[Detection],
        spacing: Tuple[float, float],
        warnings: List[str],
        spacing_known: bool = False,
    ) -> List[Detection]:
        # Drop anything outside the breast or degenerate
        detections: List[Detection] = []
        for detection in list(mass_detections) + list(calc_detections):
            if detection is None:
                continue
            if detection.mask is not None:
                mask = detection.mask
                if mask.dtype != np.uint8:
                    detection.mask = (mask > 0).astype(np.uint8) * 255
                if detection.mask.shape[:2] != original.shape[:2]:
                    detection.mask = cv2.resize(
                        detection.mask,
                        (original.shape[1], original.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
            if not self._inside_breast(detection, breast_mask):
                continue
            detection.id = self._next_detection_id
            detection.lesion_id = f"L{self._next_detection_id:02d}"
            self._next_detection_id += 1
            detections.append(detection)

        if not detections:
            return []

        # Measurements (pixels + millimetres) ---------------------------------
        self._safe(
            "lesion measurements",
            lambda: self.measurement_engine.measure_all(
                detections, spacing, spacing_known=spacing_known
            ),
            warnings,
        )
        for detection in detections:
            measurement = self._safe(
                "lesion measurement",
                lambda d=detection: self.measurement_engine.measure_detection(
                    d, spacing[1], spacing[0], spacing_known=spacing_known
                ),
                warnings,
            )
            if measurement:
                detection.diameter_px = float(measurement.get("diameter_px", 0.0))
                detection.diameter_mm = float(measurement.get("diameter_mm", 0.0))
                detection.area_px = float(measurement.get("area_px", 0.0))
                detection.area_mm2 = float(measurement.get("area_mm2", 0.0))
                detection.features["measurement"] = measurement

        # Contours for the mass detections (needed by risk/characterisation) -
        for detection in detections:
            if detection.contour is None and detection.mask is not None:
                contours, _ = cv2.findContours(
                    detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                if contours:
                    detection.contour = max(contours, key=cv2.contourArea)

        # ACR lexicon characterisation (masses) -------------------------------
        characterization = self._safe(
            "mass characterisation",
            lambda: self.characterizer.characterize_all(
                enhanced, detections, breast_mask, spacing
            ),
            warnings,
        ) or {}
        apply_characterization_to_detections(detections, characterization)

        # Risk + explainability + confidence ----------------------------------
        for detection in detections:
            if detection.features.get("source") == "deep" and detection.mask is not None:
                pass  # deep boxes already carry a mask

            risk = self._safe(
                "lesion risk analysis",
                lambda d=detection: self.risk_analyzer.analyze_lesion(
                    enhanced, d, breast_mask
                ),
                warnings,
            )
            if risk:
                detection.risk_scores = risk
                detection.features["risk"] = risk

            explanation = self._safe(
                "lesion explainability",
                lambda d=detection: self._explain_detection(d),
                warnings,
            )
            if explanation:
                detection.explainability = explanation
                detection.features["explainability"] = explanation

            calibrated = self._safe(
                "confidence calibration",
                lambda d=detection: self.confidence_module.analyze_detection(
                    d.features.get("risk", {}).get("features", {}),
                    float(d.confidence),
                ),
                warnings,
            )
            if calibrated:
                detection.reliability = float(calibrated.get("reliability", 0.0))
                detection.uncertainty = float(calibrated.get("uncertainty", 0.0))
                detection.features["calibration"] = calibrated

        return self._merge_overlapping_calcifications(detections)

    def _explain_detection(self, detection: Detection) -> Dict[str, Any]:
        risk_features = (detection.features.get("risk") or {}).get("features", {})
        features = {
            "contrast": float(risk_features.get("contrast", 0.0)),
            "texture_entropy": float(risk_features.get("texture_entropy", 0.0)),
            "shape_irregularity": float(risk_features.get("shape_irregularity", 0.0)),
            "spiculation_index": float(risk_features.get("spiculation", 0.0)),
            "density_difference": float(risk_features.get("density_difference", 0.0)),
        }
        output = self.explainability_engine.analyze(
            [{"id": detection.lesion_id or detection.id, "features": features}]
        )
        lesion = (output.get("lesions") or [{}])[0]
        return {
            "scores": lesion.get("scores", {}),
            "explanations": lesion.get("explanations", []),
            "summary": lesion.get("summary", ""),
            "features": features,
        }

    def _merge_overlapping_calcifications(
        self, detections: List[Detection]
    ) -> List[Detection]:
        """Suppress duplicate calcification markers from overlapping tiles."""
        masses = [d for d in detections if not self._is_calcification(d)]
        calcs = [d for d in detections if self._is_calcification(d)]
        if len(calcs) < 2:
            return detections

        calcs.sort(key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for detection in calcs:
            duplicate = False
            for existing in kept:
                if existing.type != detection.type:
                    continue
                left = {tuple(s['center']) for s in existing.features.get('individual_calcifications', [])}
                right = {tuple(s['center']) for s in detection.features.get('individual_calcifications', [])}
                same_members = bool(left and right and left == right)
                legacy_overlap = not (left and right) and self._overlap_ratio(existing, detection) > .75
                if same_members or legacy_overlap:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)
        return masses + kept

    @staticmethod
    def _overlap_ratio(a: Detection, b: Detection) -> float:
        """Intersection over the *smaller* box (robust for nested boxes)."""
        if not a.bbox or not b.bbox:
            return 0.0
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        smallest = min(max(aw * ah, 1), max(bw * bh, 1))
        return inter / smallest

    @staticmethod
    def _inside_breast(detection: Detection, breast_mask: np.ndarray) -> bool:
        if breast_mask is None or breast_mask.size == 0:
            return True
        height, width = breast_mask.shape[:2]
        if detection.bbox:
            x, y, w, h = [int(v) for v in detection.bbox]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + max(1, w)), min(height, y + max(1, h))
            if x1 <= x0 or y1 <= y0:
                return False
            region = breast_mask[y0:y1, x0:x1]
            if region.size == 0:
                return False
            coverage = float(np.count_nonzero(region)) / float(region.size)
            return coverage >= 0.25
        cx, cy = detection.location
        if 0 <= int(cy) < height and 0 <= int(cx) < width:
            return breast_mask[int(cy), int(cx)] > 0
        return False

    def _active_backend_mode(self) -> str:
        """'deep' when both tasks used a checkpoint, otherwise 'classical'."""
        if self.deep_mass_backend is not None or self.deep_calc_backend is not None:
            return "deep"
        return "classical"

    # ================================================================== #
    #  Aggregation helpers                                               #
    # ================================================================== #
    @staticmethod
    def _is_calcification(detection: Detection) -> bool:
        return detection.type in (
            DetectionType.MICROCALCIFICATION,
            DetectionType.MACROCALCIFICATION,
        )

    @staticmethod
    def _risk_payloads(detections: List[Detection]) -> List[Dict[str, Any]]:
        payloads = []
        for detection in detections:
            risk = detection.risk_scores or detection.features.get("risk")
            if not risk:
                continue
            payload = dict(risk)
            payload.setdefault("risk", {})
            payload["risk"].setdefault(
                "risk_score", round(float(detection.confidence) * 100, 1)
            )
            payload["lesion_id"] = detection.lesion_id or detection.id
            payloads.append(payload)
        return payloads

    def _build_explainability(self, detections: List[Detection]) -> Dict[str, Any]:
        return {
            "engine": "ExplainabilityEngine",
            "lesions": [
                {
                    "lesion_id": d.lesion_id or d.id,
                    "type": d.type.value,
                    "scores": (d.explainability or {}).get("scores", {}),
                    "explanations": (d.explainability or {}).get("explanations", []),
                    "summary": (d.explainability or {}).get("summary", ""),
                }
                for d in detections
            ],
            "summary": (
                f"{sum(1 for d in detections if (d.explainability or {}).get('explanations'))}"
                f" of {len(detections)} lesion(s) show notable morphological features."
                if detections
                else "No lesions detected."
            ),
        }

    def _build_confidence_summary(
        self,
        detections: List[Detection],
        quality_report: Optional[Dict[str, Any]],
        segmentation_confidence: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        modules: Dict[str, Dict[str, float]] = {}
        if quality_report:
            modules["image_quality"] = self.confidence_module.analyze_module(
                [float(quality_report.get("quality_score", 0.5))], "image_quality"
            )
        if segmentation_confidence:
            modules["segmentation"] = segmentation_confidence
        if detections:
            modules["detection"] = self.confidence_module.analyze_module(
                [float(d.confidence) for d in detections], "detection"
            )
        return self.confidence_module.summarize(modules)

    def _clinical_summary_payload(
        self,
        birads_result: Optional[Dict[str, Any]],
        density_result: Optional[Dict[str, Any]],
        segmentation_confidence: Optional[Dict[str, Any]],
        quality_report: Optional[Dict[str, Any]],
        detections: List[Detection],
        risk_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Flat keys consumed by the GUI / reports (backward compatible names)."""
        payload: Dict[str, Any] = {
            "birads_category": (birads_result or {}).get("category"),
            "birads_category_text": (birads_result or {}).get("category_text"),
            "birads_confidence": (birads_result or {}).get("confidence"),
            "birads_label": (birads_result or {}).get("label"),
            "breast_density_category": (density_result or {}).get("acr_category"),
            "breast_density_label": (density_result or {}).get("category_label"),
            "breast_density_percent": (density_result or {})
            .get("statistics", {})
            .get("dense_area_percentage"),
            "segmentation_confidence": (segmentation_confidence or {}).get("confidence"),
            "quality_score": (quality_report or {}).get("quality_score"),
            "quality_warnings": (quality_report or {}).get("warnings", []),
        }
        if risk_results:
            payload["tumor_risk_score"] = max(
                (r.get("risk", {}).get("risk_score", 0.0) for r in risk_results),
                default=0.0,
            )
            payload["tumor_risk_category"] = max(
                risk_results, key=lambda r: r.get("risk", {}).get("risk_score", 0)
            ).get("risk", {}).get("risk_category")
        else:
            payload["tumor_risk_score"] = 0.0
            payload["tumor_risk_category"] = "No lesion detected"

        masses = [d for d in detections if not self._is_calcification(d)]
        payload["mass_count"] = len(masses)
        payload["calcification_count"] = len(detections) - len(masses)
        payload["largest_mass_mm"] = max(
            (d.diameter_mm for d in masses), default=0.0
        )
        return payload

    def _calculate_statistics(self, detections: List[Detection]) -> Dict[str, Any]:
        active_detections = [d for d in detections if not d.is_ignored]
        confidences = [float(d.confidence) for d in active_detections]
        return {
            "total_detections": len(active_detections),
            "tumors_confirmed": sum(
                1 for d in active_detections if d.type == DetectionType.TUMOR_CONFIRMED
            ),
            "tumors_suspicious": sum(
                1 for d in active_detections if d.type == DetectionType.TUMOR_SUSPICIOUS
            ),
            "microcalcifications": sum(
                1 for d in active_detections
                if d.type == DetectionType.MICROCALCIFICATION
            ),
            "macrocalcifications": sum(
                1 for d in active_detections
                if d.type == DetectionType.MACROCALCIFICATION
            ),
            "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "ignored_count": sum(1 for d in detections if d.is_ignored),
        }

    @staticmethod
    def _infer_view_position(
        breast_mask: np.ndarray, image_meta: Dict[str, Any]
    ) -> str:
        """Best-effort view inference from the breast mask geometry.

        MLO views contain the pectoral muscle in the upper chest-wall corner,
        which makes the breast region wider at the top; CC views are more
        symmetric.  The heuristic is only a hint used to gate pectoral
        detection, and it is reported as inferred.
        """
        if breast_mask is None or breast_mask.size == 0:
            return ""
        try:
            ys, xs = np.nonzero(breast_mask)
            if xs.size == 0:
                return ""
            left_half = float(np.mean(xs < breast_mask.shape[1] / 2))
            chest_wall_left = left_half > 0.5
            top = np.count_nonzero(breast_mask[: breast_mask.shape[0] // 3, :])
            bottom = np.count_nonzero(
                breast_mask[2 * breast_mask.shape[0] // 3:, :]
            )
            if top > bottom * 1.15:
                return "MLO" if chest_wall_left else "MLO"
            return "CC"
        except Exception:  # pragma: no cover - defensive
            return ""

    # ================================================================== #
    #  Utilities                                                         #
    # ================================================================== #
    def _safe(self, label: str, func, warnings: List[str]):
        try:
            return func()
        except Exception as exc:
            logger.exception("%s failed", label)
            warnings.append(f"{label} failed: {type(exc).__name__}: {exc}")
            return None


def _resolve_spacing(image_meta: Dict[str, Any]) -> Tuple[float, float]:
    """Return ``(row_spacing_mm, column_spacing_mm)`` from DICOM metadata."""
    try:
        spacing = image_meta.get("pixel_spacing") or [1.0, 1.0]
        row, col = float(spacing[0]), float(spacing[1])
        if not np.isfinite([row, col]).all() or row <= 0 or col <= 0:
            raise ValueError("invalid spacing")
        if not image_meta.get("spacing_known", (row, col) != (1., 1.)):
            # 1.0 mm/px is never a real mammography pitch: it means "unknown".
            assumed = float(getattr(Config, "ASSUMED_PIXEL_SPACING_MM", 0.05))
            return assumed, assumed
        return row, col
    except (TypeError, ValueError, IndexError):
        assumed = float(getattr(Config, "ASSUMED_PIXEL_SPACING_MM", 0.05))
        return assumed, assumed

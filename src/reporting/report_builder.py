"""Structured clinical report generation (JSON + HTML).

Reports contain only what the pipeline actually measured.  Every generated
document carries the software-assistance disclaimer, the model provenance
(checkpoint file name + digest) and the list of analysis warnings, so a reader
can never mistake the output for a radiologist's diagnosis.
"""
from __future__ import annotations

import base64
import datetime
import html
import json
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import Config

logger = logging.getLogger("MammographyCAD.report")


class ReportBuilder:
    """Build JSON and HTML reports from an :class:`AnalysisResult`."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.reports_dir = output_dir or Config.REPORTS_DIR
        self._patient_override: Dict[str, str] = {}
        os.makedirs(self.reports_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  JSON                                                              #
    # ------------------------------------------------------------------ #
    def build_json(self, result, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = result.to_dict()
        payload["report_meta"] = self._meta(result)
        payload["clinical_summary"] = self._clinical_summary(result)
        payload["disclaimer"] = Config.REPORT_DISCLAIMER
        if extra:
            payload["extra"] = extra
        return payload

    def save_json(self, result, path: Optional[str] = None) -> str:
        payload = self.build_json(result)
        path = path or self.default_path(result, "json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("JSON report written: %s", path)
        return path

    # ------------------------------------------------------------------ #
    #  HTML                                                              #
    # ------------------------------------------------------------------ #
    def build_html(self, result, include_images: bool = True) -> str:
        meta = self._meta(result)
        summary = self._clinical_summary(result)
        statistics = result.statistics or {}
        birads = result.birads or result.birads_result or {}
        density = result.breast_density or result.density_result or {}
        calc = result.calcification_analysis or {}
        quality = result.quality_report or {}
        model_status = result.model_status or {}

        rows = []
        rows.append(self._row("Patient ID", meta["patient_id"]))
        rows.append(self._row("Study date", meta["study_date"]))
        rows.append(self._row("View / laterality", meta["view"]))
        rows.append(self._row("Image file", meta["file_name"]))
        rows.append(self._row("Pixel spacing", meta["pixel_spacing"]))
        rows.append(self._row("Analysis time", f"{result.processing_time:.2f} s"))
        header_table = self._table(["Field", "Value"], rows)

        assessment_rows = [
            self._row(
                "BI-RADS (software-assisted estimate)",
                f"{birads.get('category', 'N/A')}"
                + (f" {birads.get('subcategory')}" if birads.get("subcategory") else "")
                + f" &mdash; {html.escape(str(birads.get('category_text', '')))}",
            ),
            self._row("Assessment confidence", self._fmt(birads.get("confidence"))),
            self._row(
                "Assessment validation",
                html.escape(
                    str(
                        birads.get("validation_warning")
                        or ("validated model" if birads.get("validation_status") == "validated_model" else "not validated")
                    )
                ),
            ),
            self._row(
                "Breast density (ACR)",
                html.escape(str(density.get("category_label", density.get("acr_category", "N/A")))),
            ),
            self._row(
                "Dense area estimate",
                self._fmt(
                    (density.get("statistics") or {}).get("dense_area_percentage"), suffix=" %"
                ),
            ),
            self._row("Mass candidates", statistics.get("mass_count", 0)),
            self._row("Calcification candidates", statistics.get("calcification_count", 0)),
            self._row(
                "Largest mass (mm)", self._fmt(statistics.get("largest_mass_mm"), suffix=" mm")
            ),
            self._row(
                "Image quality score", self._fmt(quality.get("quality_score"))
            ),
        ]
        assessment_table = self._table(["Assessment", "Value"], assessment_rows)

        reasons = birads.get("reasons") or []
        reasons_html = (
            "<ul>" + "".join(f"<li>{html.escape(str(r))}</li>" for r in reasons) + "</ul>"
            if reasons
            else "<p>No reasons were recorded.</p>"
        )

        detections_html = self._detections_table(result)
        calc_html = self._calcification_section(calc)
        warnings_html = self._warnings_section(result)
        provenance_html = self._provenance_section(model_status)
        images_html = self._images_section(result) if include_images else ""

        disclaimer = html.escape(Config.REPORT_DISCLAIMER)
        generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Mammography CAD report</title>
<style>
 body {{ font-family: Arial, Helvetica, sans-serif; margin: 28px; color: #222; }}
 h1 {{ font-size: 21px; margin-bottom: 2px; }}
 h2 {{ font-size: 15px; margin-top: 26px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
 table {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 12.5px; }}
 th, td {{ border: 1px solid #bbb; padding: 5px 7px; text-align: left; vertical-align: top; }}
 th {{ background: #eef1f5; }}
 .disclaimer {{ margin-top: 26px; padding: 12px; background: #fff6e5;
                border: 1px solid #e0b96a; font-size: 12px; }}
 .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
           background: #e8eef7; border: 1px solid #b6c7de; font-size: 11.5px; }}
 .warn {{ background: #fdeaea; border: 1px solid #e2a0a0; padding: 8px; font-size: 12px; }}
 .images {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }}
 .images figure {{ margin: 0; text-align: center; font-size: 11px; }}
 img {{ max-width: 330px; border: 1px solid #999; }}
</style></head><body>
<h1>Mammography CAD &mdash; Software-Assisted Analysis Report</h1>
<div><span class="badge">Research / decision support only</span>
<span class="badge">Not a medical diagnosis</span>
<span class="badge">Generated {generated}</span></div>

<p style="color:#a00000"><b>{disclaimer}</b></p>
<h2>Study and image</h2>
{header_table}

<h2>Software-assisted assessment</h2>
{assessment_table}
<p><b>Basis for the assessment</b></p>
{reasons_html}

<h2>Detected regions</h2>
{detections_html}

<h2>Calcification analysis</h2>
{calc_html}

{images_html}

<h2>Model provenance</h2>
{provenance_html}

<h2>Analysis warnings and limitations</h2>
{warnings_html}

<div class="disclaimer"><b>Disclaimer.</b> {disclaimer}<br><br>
{html.escape(Config.DISCLAIMER)}<br><br>
Detection and characterisation results are computed automatically from image
features. They may contain false positives and false negatives. All findings
must be verified by a qualified radiologist.</div>
</body></html>"""

    def save_html(self, result, path: Optional[str] = None, include_images: bool = True) -> str:
        path = path or self.default_path(result, "html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.build_html(result, include_images=include_images))
        logger.info("HTML report written: %s", path)
        return path

    def save(
        self,
        result,
        base_path: Optional[str] = None,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """Write both a JSON and an HTML report; returns the created paths.

        ``patient_id`` / ``patient_name`` override the values read from the
        image metadata (anonymised files, or a name entered in the GUI).
        """
        self._patient_override = {
            "patient_id": patient_id or "",
            "patient_name": patient_name or "",
        }
        if base_path:
            root, _ext = os.path.splitext(base_path)
            json_path = f"{root}.json"
            html_path = f"{root}.html"
        else:
            json_path = self.default_path(result, "json")
            html_path = self.default_path(result, "html")
            root = os.path.splitext(json_path)[0]
        self.save_json(result, json_path)
        self.save_html(result, html_path)
        return {"json": json_path, "html": html_path, "base": root}

    def save_pdf(self, result, path: str) -> str:
        """Export from the GUI thread using the existing Qt runtime."""
        from PyQt6.QtGui import QPageSize, QPdfWriter, QTextDocument
        from PyQt6.QtWidgets import QApplication

        if QApplication.instance() is None:
            raise RuntimeError("PDF export requires a running Qt application")
        document = QTextDocument()
        document.setHtml(self.build_html(result, include_images=False))
        writer = QPdfWriter(path)
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        document.print(writer)
        return path

    def save_docx(self, result, path: str) -> str:
        """Write a research summary; the companion JSON holds full spot records."""
        from io import BytesIO
        from docx import Document
        from docx.shared import Inches, RGBColor

        doc = Document()
        doc.add_heading('MammographyCAD research report', 0)
        warning = doc.add_paragraph().add_run(Config.REPORT_DISCLAIMER)
        warning.bold = True
        warning.font.color.rgb = RGBColor(160, 0, 0)
        doc.add_heading('Image and analysis', level=1)
        meta = self._meta(result)
        doc.add_paragraph(f"Grid: {result.original.shape[1]} × {result.original.shape[0]} pixels")
        doc.add_paragraph(f"Pixel spacing: {meta['pixel_spacing']}")
        from datetime import datetime, timezone
        doc.add_paragraph(f"Exported: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
        density = result.density_result or {}
        doc.add_paragraph(f"Density estimate: {density.get('acr_category', 'not available')}")
        doc.add_heading('Candidate findings', level=1)
        for d in result.detections:
            f = d.features or {}
            m = f.get('measurement', {})
            source = f.get('mask_origin', 'automatic' if d.mask is not None else 'none')
            doc.add_paragraph(f"{d.lesion_id}: {d.type.value}; centre {d.location} px; "
                              f"evidence score {d.confidence:.2f}; mask: {source}; "
                              f"area {m.get('area_px', 'n/a')} px²; "
                              f"equivalent diameter {m.get('equivalent_diameter_px', 'n/a')} px; "
                              f"uncertainty {d.uncertainty:.2f}" + ('; ignored' if d.is_ignored else ''))
            spots = f.get('individual_calcifications', [])
            if spots:
                doc.add_paragraph(f"Member coordinates (px): {[s['center'] for s in spots]}")
            if f.get('needs_review'):
                doc.add_paragraph('Review advised: weak segmentation agreement or anatomical overlap.')
        overlay = self._overlay(result)
        if overlay is not None:
            ok, png = cv2.imencode('.png', overlay)
            if ok:
                doc.add_picture(BytesIO(png.tobytes()), width=Inches(5))
        doc.add_heading('Warnings', level=1)
        for message in result.warnings:
            doc.add_paragraph(str(message), style='List Bullet')
        doc.save(path)
        return path

    def default_path(self, result, extension: str) -> str:
        name = self._base_name(result)
        return os.path.join(self.reports_dir, f"{name}.{extension}")

    # ------------------------------------------------------------------ #
    #  Sections                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _base_name(result) -> str:
        metadata = (result.image_metadata or {}).get("raw", {}) or {}
        file_name = os.path.splitext(
            os.path.basename(metadata.get("file_name", "") or "mammogram")
        )[0]
        patient = (
            metadata.get("patient", {}).get("patient_id", "") or "nopatient"
        )
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(ch for ch in f"{patient}_{file_name}" if ch.isalnum() or ch in "-_")
        return f"cad_report_{safe[:60]}_{stamp}"

    def _meta(self, result) -> Dict[str, Any]:
        metadata = (result.image_metadata or {}).get("raw", {}) or {}
        patient = metadata.get("patient", {}) or {}
        study = metadata.get("study", {}) or {}
        image = metadata.get("image", {}) or {}
        spacing = (result.image_metadata or {}).get("pixel_spacing", [1.0, 1.0])
        return {
            "patient_id": self._patient_override.get("patient_id")
            or patient.get("patient_id")
            or "not available",
            "patient_name": self._patient_override.get("patient_name")
            or patient.get("patient_name")
            or "",
            "study_date": study.get("study_date") or "not available",
            "institution": study.get("institution") or "",
            "view": (
                f"{(result.image_metadata or {}).get('view_position', '') or 'unknown'}"
                f" / {(result.image_metadata or {}).get('laterality', '') or 'unknown'}"
            ),
            "file_name": os.path.basename(metadata.get("file_name", "") or result.image_metadata.get("path", "") or ""),
            "pixel_spacing": (
                f"{spacing[0]:.4g} x {spacing[1]:.4g} mm (DICOM)"
                if (result.image_metadata or {}).get("spacing_known")
                else (
                    f"{spacing[0]:.4g} x {spacing[1]:.4g} mm ASSUMED "
                    "(no DICOM spacing; mm values are estimates)"
                )
            ),
            "photometric": image.get("photometric_interpretation", "") or "n/a",
            "source": metadata.get("source", "unknown"),
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        }

    def _clinical_summary(self, result) -> Dict[str, Any]:
        birads = result.birads or result.birads_result or {}
        density = result.breast_density or result.density_result or {}
        calc = result.calcification_analysis or {}
        return {
            "birads_category": birads.get("category"),
            "birads_subcategory": birads.get("subcategory"),
            "birads_text": birads.get("category_text"),
            "birads_confidence": birads.get("confidence"),
            "birads_management": birads.get("management"),
            "birads_is_software_assessment": True,
            "breast_density_acr": density.get("acr_category"),
            "breast_density_label": density.get("category_label"),
            "dense_area_percentage": (density.get("statistics") or {}).get(
                "dense_area_percentage"
            ),
            "calcification_morphology": calc.get("dominant_morphology"),
            "calcification_distribution": calc.get("dominant_distribution"),
            "calcification_clusters": calc.get("cluster_count"),
            "detection_backend": {
                task: (status or {}).get("backend", "unknown")
                for task, status in (result.model_status or {}).items()
                if isinstance(status, dict)
            },
            "warnings": list(result.warnings or []),
            "disclaimer": Config.REPORT_DISCLAIMER,
        }

    def _detections_table(self, result) -> str:
        detections = [d for d in (result.detections or [])]
        if not detections:
            return "<p>No regions were detected inside the breast.</p>"
        rows = []
        for index, detection in enumerate(
            sorted(detections, key=lambda d: getattr(d, "id", 0)), start=1
        ):
            features = detection.features or {}
            characterization = features.get("mass_characterization") or {}
            measurement = features.get("measurement") or {}
            diameter = (self._fmt(measurement.get('max_diameter_mm'), suffix=' mm')
                        if measurement.get('spacing_available') else
                        self._fmt(measurement.get('max_diameter_px'), suffix=' px'))
            rows.append(
                "<tr>"
                f"<td>{index}</td>"
                f"<td>{html.escape(detection.lesion_id or str(detection.id))}</td>"
                f"<td>{html.escape(detection.type.value.replace('_', ' ').title())}</td>"
                f"<td>{detection.location[0]}, {detection.location[1]}</td>"
                f"<td>{self._fmt(detection.confidence)}</td>"
                f"<td>{html.escape(str(features.get('mask_origin', 'automatic' if detection.mask is not None else 'none')))}</td>"
                f"<td>{diameter}</td>"
                f"<td>{html.escape(str(characterization.get('shape', '-')))}</td>"
                f"<td>{html.escape(str(characterization.get('margin', '-')))}</td>"
                f"<td>{html.escape(str(characterization.get('density', '-')))}</td>"
                f"<td>{'ignored' if detection.is_ignored else 'included'}</td>"
                "</tr>"
            )
        header = (
            "<tr><th>#</th><th>Lesion</th><th>Type</th><th>Centre (px)</th>"
            "<th>Evidence score</th><th>Mask source</th><th>Max diameter</th><th>Shape</th><th>Margin</th>"
            "<th>Density</th><th>Status</th></tr>"
        )
        return f"<table>{header}{''.join(rows)}</table>"

    def _calcification_section(self, calc: Dict[str, Any]) -> str:
        if not calc or not calc.get("available"):
            reason = (calc or {}).get("reason", "not analysed")
            return f"<p>No calcification analysis available ({html.escape(str(reason))}).</p>"

        rows = [
            self._row("Microcalcification candidates", calc.get("microcalcification_count", 0)),
            self._row("Macrocalcification candidates", calc.get("macrocalcification_count", 0)),
            self._row("Groups meeting cluster criterion", calc.get("cluster_count", 0)),
            self._row("Dominant morphology", calc.get("dominant_morphology") or "not assessed"),
            self._row("Dominant distribution", calc.get("dominant_distribution") or "not assessed"),
            self._row("Suspicion level", calc.get("suspicion_level")),
            self._row("Measurement basis", calc.get("measurement_basis")),
        ]
        clusters = calc.get("clusters") or []
        cluster_rows = "".join(
            "<tr>"
            f"<td>{i + 1}</td><td>{c.get('count')}</td>"
            f"<td>{html.escape(str(c.get('morphology')))}</td>"
            f"<td>{html.escape(str(c.get('distribution')))}</td>"
            f"<td>{self._fmt(c.get('extent_mm'), suffix=' mm')}</td>"
            f"<td>{self._fmt(c.get('mean_diameter_mm'), suffix=' mm')}</td>"
            "</tr>"
            for i, c in enumerate(clusters[:20])
        )
        clusters_table = (
            "<table><tr><th>Group</th><th>Calcifications</th><th>Morphology</th>"
            "<th>Distribution</th><th>Extent</th><th>Mean size</th></tr>"
            f"{cluster_rows}</table>"
            if cluster_rows
            else "<p>No group met the cluster criterion.</p>"
        )
        note = html.escape(str(calc.get("note", "")))
        return (
            self._table(["Descriptor", "Value"], rows)
            + clusters_table
            + f"<p style='font-size:11.5px'>{note}</p>"
        )

    def _warnings_section(self, result) -> str:
        items: List[str] = list(result.warnings or [])
        quality = result.quality_report or {}
        items.extend(quality.get("warnings", []) or [])
        if not items:
            return "<p>No analysis warnings were raised.</p>"
        return "<div class='warn'><ul>" + "".join(
            f"<li>{html.escape(str(item))}</li>" for item in items
        ) + "</ul></div>"

    def _provenance_section(self, model_status: Dict[str, Any]) -> str:
        status = (model_status or {}).get("runtime", {}) or {}
        rows = [
            self._row("Configured backend mode", (model_status or {}).get("configured_mode", "auto")),
            self._row(
                "Deep runtime",
                "available" if (model_status or {}).get("deep_inference_available")
                else "not installed (classical fallback)",
            ),
        ]
        for task in ("mass", "calcification"):
            info = (model_status or {}).get(task, {}) or {}
            rows.append(
                self._row(
                    f"{task.capitalize()} detector",
                    f"{info.get('backend', 'classical')}"
                    + (f" - {info.get('model')}" if info.get("model") else "")
                    + (f" ({info.get('reason')})" if info.get("reason") else ""),
                )
            )
        return self._table(["Item", "Value"], rows)

    def _images_section(self, result) -> str:
        figures = []
        for label, array in (
            ("Overlay", self._overlay(result)),
            ("Enhanced", result.enhanced),
            ("Breast mask", result.breast_mask),
        ):
            data = self._encode_image(array)
            if data:
                figures.append(
                    f"<figure><img src='data:image/png;base64,{data}' "
                    f"alt='{label}'><figcaption>{label}</figcaption></figure>"
                )
        if not figures:
            return ""
        return f"<h2>Images</h2><div class='images'>{''.join(figures)}</div>"

    def _overlay(self, result) -> Optional[np.ndarray]:
        try:
            from ..models.visualizer import ResultVisualizer

            visualizer = ResultVisualizer()
            return visualizer.visualize_detections(
                result.original,
                result.detections,
                pectoral_mask=result.pectoral_mask,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Overlay rendering for report failed: %s", exc)
            return None

    @staticmethod
    def _encode_image(array, max_width: int = 520) -> Optional[str]:
        if array is None or not isinstance(array, np.ndarray) or array.size == 0:
            return None
        image = array
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / width
            image = cv2.resize(
                image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA
            )
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            return None
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    # ------------------------------------------------------------------ #
    #  Small formatting helpers                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _table(headers: List[str], rows: List[str]) -> str:
        head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body = "".join(
            f"<tr><td>{html.escape(str(h))}</td><td>{r}</td></tr>"
            for h, r in zip(headers, rows)
        )
        return f"<table><tr>{head}</tr>{body}</table>"

    @staticmethod
    def _row(label: str, value: Any) -> str:
        if value is None or value == "":
            value = "not available"
        return html.escape(str(value)) if not isinstance(value, str) or "<" not in value else value

    @staticmethod
    def _fmt(value: Any, percent: bool = False, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return html.escape(str(value))
        if percent:
            return f"{number * 100:.1f} %" if number <= 1.0 else f"{number:.1f} %"
        return f"{number:.2f}{suffix}"

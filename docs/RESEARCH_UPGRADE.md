# MammographyCAD research upgrade — delivery record

**Source:** `https://github.com/sasanzarerasouli/mamographyCAD` at
`1a146a1218b7302a53de8178366cbe9629dd289a`. Changes are in a local copy. GitHub
was not changed. This is a research implementation, not a clinically validated
CAD system or a complete implementation of every requirement in the brief.

## Original architecture and preserved contracts

`main.py` starts `MammographyCADApp`. The existing original/overlay, enhanced,
breast mask, density and heatmap tabs feed the same PyQt6 window and zoom/pan
widget. Results, Clinical, Lesion Detail, Settings and Models/Info remain; a log
tab was added. `CADWorker` and `BatchProcessor` wrap `MammographyCADSystem` in
Qt threads. Worker signal signatures did not change. Pipeline: `DICOMHandler`
(or raster decode) → `MammographyPreprocessor` → breast mask, enhancement,
quality and pectoral analysis → mass and calcification detectors → `Detection`
IDs, measurements, characterization and existing rule-based summaries →
`AnalysisResult` → Qt worker signals → `ResultVisualizer` and report builder.
The `Detection` type, legacy tuple, statistics and BI-RADS grouping contracts
are preserved. Clinical/density displays are software estimates, not diagnoses.
`docs/ARCHITECTURE.md` has the subsystem map. Existing configuration, batch,
session and training paths were not replaced.

## Implemented paths

- **Mass localization.** Percentile-normalized breast intensities; nearest
  tissue extension at the boundary; configurable coarse localization grid for
  large images. DoG, scale-normalized LoG, Hessian compact-curvature term,
  square center-versus-annulus local contrast, and white top-hat each produce
  a separate response. Positive and negative polarities compete per scale;
  median/MAD normalization has a percentile floor and clipping. Initial
  fusion weights are equal, not fitted. Per-scale peak selection and radius-
  dependent suppression bound ROI work; mask IoU catches cross-scale duplicates.
  `TumorDetector.detect(..., debug={})` fills diagnostics when supplied a dict.
- **Mass segmentation.** Original-grid ROIs generate three hypotheses:
  deterministic FCM (cluster selection uses candidate location, ring exposure
  and contrast), foreground/background-seeded watershed on a Sobel gradient
  surface, and scale-adaptive Gaussian thresholding. A two-of-three vote keeps
  the seed-associated region and fills only small holes. A weak or implausible
  vote can yield a location-only review candidate, not a fabricated shape.
  ROI mask hypotheses and origin are retained separately from JSON metadata.
  Candidate scores are uncalibrated; the independent segmentation score uses
  agreement. Geometry features include perimeter, area, equivalent diameter,
  aspect ratio, circularity, solidity, convexity and radial roughness. Candidate
  location and ROI scale are not substitutes for mask geometry.
- **Anatomy.** Existing breast-boundary and pectoral stages remain. When the
  legacy preprocessing reports a pectoral mask, the mass stage uses a separately
  enhanced image without destructive pectoral suppression. The detected
  pectoral overlap adds review metadata and a confidence-weighted score penalty;
  overlap alone does not delete a mass. Near-skin mass and calcification signals
  are downweighted rather than uniformly suppressed. There is no new validated
  nipple, marker or text-artifact detector.
- **Calcifications.** Adds full-grid compact DoG/LoG, small top-hats and a local
  median-residual noise estimate with seed-and-grow hysteresis alongside the
  legacy adaptive candidate path. Connected components feed the existing size,
  shape, contrast and prominence filters. Grouping uses physical coordinates
  when spacing is known, pixels otherwise; known-spacing grouping defaults to
  a 3 mm DBSCAN neighborhood and a 10 mm extent guard. Accepted group members
  carry individual coordinates and sparse pixel masks; macrocalcifications also
  carry component pixels. Strong isolated microcalcifications can survive as
  singleton records. Legacy thresholds still gate weak individual candidates;
  not all spots are standalone `Detection` objects. Group area is a bounding
  extent measure, not mineral area.
- **GUI and edits.** `Original only` hides all overlays and selection marks.
  Mass fill, box, label, score and arrow controls are independent. Calcification
  pixels, markers, group outlines and labels are separate controls. Pectoral
  boundary and density controls remain. To edit, select a mass with a mask in
  Results, choose Add or Remove in Lesion Detail, then click in the overlay.
  Add must touch the current mask and stay inside the breast. A dab uses a
  six-original-pixel radius; Undo restores the last dab, Restore returns to the
  preserved automatic mask. Return to Inspect/Pan for usual panning. Edited
  geometry is exported; original scores and descriptors are not recomputed and
  this is disclosed. Edits are session-local, not a persistent annotation store.
- **Input and reporting.** Native decoded pixels are retained read-only on
  `AnalysisResult.source_pixels` (not serialized); normal display and current
  primary processing still use the legacy 8-bit presentation pipeline. The
  source file is never written. DICOM spacing detection honors row/column
  spacing, including anisotropy. Where unavailable, measured geometry is shown
  in pixels; legacy result fields with assumed mm remain for compatibility and
  are labeled as assumed. Existing JSON/HTML exports remain. GUI export adds
  text/table PDF and optional DOCX with an annotated overview; neither replaces
  JSON for detailed spot data. The required report warning appears prominently.
- **Logging.** The GUI activity log includes time, levels, load/processing
  milestones, warning counts and export status. File paths and patient fields
  are not copied into that activity log. Exception detail remains in the
  existing diagnostic logger/dialogs; those paths may contain filenames, and
  exported reports may contain the original DICOM metadata. Handle accordingly.

## Reproduce the tests and launch

No new virtual environment, Python change or CUDA install is required. On a
machine with this application's dependencies already working, only
`python-docx` is new, and then only if DOCX export is wanted.

```powershell
# From the extracted MammographyCAD directory
python -m pip install -r requirements.txt
python -m pip install "pydicom>=2.4" "pytest>=7.4"
python -m pip install "python-docx>=1.1"  # Optional, only for DOCX export
python main.py
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest -q -s -rs
```

Sandbox: Linux, Python 3.13.18, NumPy 2.3.5, SciPy 1.17.1, OpenCV 4.11.0.86,
scikit-image 0.25.2, scikit-learn 1.6.1, PyQt6 6.12.1, pydicom 3.0.2,
python-docx 1.1.2 and pytest 9.0.3. The sandbox needed the Linux system library
`libxkbcommon0` for Qt; that is not a Windows install step.

`QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 python -m pytest -q -s -rs`:
**77 passed, 3 skipped, 12 warnings**. Skips: old tiling helper test, missing
real-image fixture, and the pectoral-specific test on a phantom without pectoral
muscle. Warnings were pydicom deprecations in synthetic test setup. `python
main.py` entered the offscreen event loop and remained running until a deliberate
five-second timeout. The offscreen suite exercised threaded load/run and GUI
refresh, zoom, selection, mask edits and PDF generation. DOCX was reopened to
check disclaimer, findings and embedded overview. Windows native GUI and the
user's NVIDIA/CPU setup were not tested.

Synthetic irregular polygon test (Dice / IoU). The two cases use the same
geometry with opposite contrast signs; these are not independent clinical cases:

| Method | Bright phantom | Dark phantom |
|---|---:|---:|
| FCM | .783 / .643 | .788 / .650 |
| Gradient | .458 / .297 | .483 / .318 |
| Adaptive Gaussian | .761 / .614 | .773 / .630 |
| Majority vote | .761 / .614 | .775 / .632 |

Fusion did **not** outperform FCM on these cases. No clinical sensitivity,
specificity, FROC, precision, calcification F1 or boundary accuracy is inferred.
For thesis evaluation, split by patient/study before tuning, use consensus lesion
masks and spot/cluster annotations, and report method-wise Dice/IoU/HD95 plus
localization sensitivity at false positives per image. Use a locked test split
and stratify by view, density, size, spacing and contrast. Record reader and
annotation uncertainty. Initial detector weights and scoring rules remain
uncalibrated heuristics.

## Feature completion

| Feature | Status | Evidence or gap |
|---|---|---|
| Classical five-detector mass localization | IMPLEMENTED AND TESTED | Synthetic peak localization, both polarities, original-grid mapping and CPU regression. |
| Independent ROI hypotheses and pixel-mask fusion | IMPLEMENTED AND TESTED | Synthetic masks scored above; weakest stage is gradient hypothesis. No real lesion-boundary validation. |
| Candidate features, scoring and review flags | PARTIALLY IMPLEMENTED | Heuristic evidence only; no learned candidate classification, calibrated probability or clinical cutoff. |
| Anatomy-aware handling | PARTIALLY IMPLEMENTED | Pectoral/skin handling; no dedicated nipple/marker segmentation or validation of anatomical FP rate. |
| Independent calcification pipeline and clustering | PARTIALLY IMPLEMENTED | New compact responses, synthetic cluster/spacing tests, accepted sparse member masks. Legacy filters, singleton limitations and no real recall/precision testing. |
| GUI integration, refresh and key overlay controls | IMPLEMENTED AND TESTED | Offscreen window, worker, rendering and zoom regressions; not a full Windows/native visual QA. |
| Manual correction and export of corrected measurements | IMPLEMENTED AND TESTED | Mouse removal, undo and restore exercised. Add is implemented; not separately exercised by the automated GUI sequence. |
| DICOM handling and original data integrity | PARTIALLY IMPLEMENTED | Verified 16-bit native retention, polarity and anisotropic spacing. Detection still uses presentation pixels. No DICOM writer added. |
| CPU execution | IMPLEMENTED AND TESTED | Offscreen pipeline and tests; no clinical batch throughput or memory profile on target hardware. |
| Auto/CPU/GPU device manager | NOT IMPLEMENTED | Primary stages are CPU only; no validated GPU path or device picker. |
| Separate YOLO comparison | NOT IMPLEMENTED | Three checkpoints inventoried without loading; no trusted runtime validation, comparison display or fusion. |
| JSON/HTML/PDF/DOCX reporting | IMPLEMENTED AND TESTED | PDF file smoke-tested, no page-by-page inspection; DOCX parsed. Batch uses the preserved JSON/HTML path. |
| GUI log management | PARTIALLY IMPLEMENTED | Activity display verified; log-save dialog and exhaustive per-stage diagnostic integration not verified. |
| Breast-density / BI-RADS-related behavior | PARTIALLY IMPLEMENTED | Existing estimates preserved and exercised; not clinically validated or algorithmically retuned. |
| Application icon, splash and installer | NOT IMPLEMENTED | Existing startup retained. |
| Real-data evaluation, clinical claims and Windows QA | BLOCKED BY ENVIRONMENT LIMITATION | No annotated real dataset available in the workspace; no Windows desktop or compatible GPU tested. |

## Exact file manifest

### Modified
- `config.py`
- `docs/ARCHITECTURE.md`
- `docs/DIGITAL_EYE_CHECKPOINT_ANALYSIS.md`
- `requirements-optional.txt`
- `src/core/analysis_result.py`
- `src/core/cad_system.py`
- `src/core/detection_types.py`
- `src/io/dicom_handler.py`
- `src/measurement/lesion_measurement_engine.py`
- `src/models/calcification_detector.py`
- `src/models/deep_detector.py`
- `src/models/model_registry.py`
- `src/models/tumor_detector.py`
- `src/models/visualizer.py`
- `src/processing/preprocessor.py`
- `src/reporting/report_builder.py`
- `src/ui/main_window.py`
- `src/ui/zoomable_label.py`
- `tests/test_dicom.py`
- `tests/test_gui.py`

### Added

- `docs/RESEARCH_UPGRADE.md`
- `src/models/mass_pipeline.py`
- `tests/test_research_upgrade.py`

### Removed

None. Original bundled weights were retained unchanged. No new mandatory Python
package was added. `requirements-optional.txt` documents `python-docx`.

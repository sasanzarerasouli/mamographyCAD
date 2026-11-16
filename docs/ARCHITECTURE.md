# Architecture

```
main.py
└── src/ui/main_window.py  (MammographyCADApp, QMainWindow)
    ├── 5 image tabs    Original/Overlay · Enhanced · Mask · Density · Heatmap
    ├── side tabs       Results · Clinical · Lesion detail · Settings · Models/Info
    ├── CADWorker (QThread)        → src/core/cad_system.py
    ├── BatchProcessor (QThread)   → src/core/batch_processor.py
    └── ReportBuilder              → src/reporting/report_builder.py
```

## Analysis pipeline (`MammographyCADSystem.analyze`)

`analyze` (path) → `MammographyPreprocessor.load_image` → `process_array`:

1. **decode** – DICOM via `src/io/dicom_handler.py` (MONOCHROME1/2, 8/16-bit,
   signed/unsigned, `BitsStored`, windowing), else PNG/JPG/TIFF.
2. **pre-process** – polarity normalisation, page/film-frame removal, breast
   mask (Otsu + morphological refinement), pectoral detection (Hough 15–75° with
   a masked fallback), CLAHE enhancement.
3. **quality analysis** – exposure, contrast, coverage; insufficient coverage
   raises a warning that flows into BI-RADS 0 handling.
4. **segmentation refinement** – `segmentation_pipeline.py`.
5. **mass detection** – deep backend when a usable checkpoint exists, else
   multi-stage classical detector (LoG/DoG candidates → per-candidate
   evaluation → contour, ROI statistics, spiculation).
6. **calcification detection** – deep backend (overlapping 640 px tiles, 30 %
   overlap, multi-scale, class-aware NMS) or classical ring/intensity spot
   detector with micro/macro separation and vectorised clustering.
7. **characterisation** – ACR shape/margin/density/orientation per mass;
   morphology/distribution/cluster analysis for calcifications.
8. **measurements** – px and mm values (spacing-aware).
9. **risk / explainability / confidence** – per-lesion scores.
10. **density** – GMM dense-area estimate + ACR category.
11. **BI-RADS** – rule-based evidence aggregation (see CLINICAL_MAPPING.md).
12. **visualisation** – overlay, density heatmap, suspicion heatmap.
13. **result assembly** – `AnalysisResult` with provenance, warnings, backend.

Every stage is wrapped by `_safe(...)`: a failing stage appends a warning and the
pipeline continues, so a partial result is always explainable rather than a
crash.

## Threading

* `CADWorker(QThread)` emits `progress(int, str)`, `result_ready(AnalysisResult)`,
  the legacy 6-argument `finished`, `error(str)`, `cancelled`. `cancel()` is
  checked at every progress callback and raises an internal `_Cancelled`.
* `BatchProcessor(QThread)` emits `progress(int,int,str)`, `file_done(path,
  status, msg)`, `batch_done(ok, failed, failures)` and writes one report per
  file.
* Both are cancelled and joined in `closeEvent`, so no `QThread` is destroyed
  while running.

## Rendering

`ZoomableImageLabel` paints the image and every overlay in `paintEvent` on each
update. This removed a real bug: the previous widget cached a rendered pixmap and
only refreshed it on resize/zoom, so overlays appeared "only after zooming or
dragging". Hit-testing (bounding box then point radius), a dash-ring highlight,
right-click *ignore/restore* and keyboard-free panning are implemented on top of
the same widget; `current_image()` / `rendered_pixmap()` expose the display for
code and tests since `QLabel.pixmap()` is meaningless for a paint-based widget.

## Reporting

`ReportBuilder` writes `{root}.json` and a **self-contained** `{root}.html`
(base64 PNG overlay/enhanced/mask/heatmap, no external resources). The report
contains the provenance table (file, SHA-256 of the model, backend, spacing
basis, timestamps), the acquisition parameters, every finding with its
descriptors and measurements, the BI-RADS estimate with evidence and management,
the density estimate, warnings, and the disclaimer. Missing values are printed
as `n/a` / `not available`; nothing is interpolated or invented.

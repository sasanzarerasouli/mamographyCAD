# Final report — MammographyCAD (thesis project)

Date: 2026-11-13 · Repository: `sasanzarerasouli/MammographyCAD` (local clone)
Status: **implementation complete and verified in this environment**; deep-learning
detection is implemented but **not trained** (no weights and no datasets exist in
the repository — see §4).

---

## 1. Work summary

The application was audited end to end and repaired rather than re-architected.
The original structure (`main.py` → `src/ui/main_window.py` → `src/core/cad_system.py`
→ `src/models/*`, `src/processing/*`, `src/measurement/*`, `src/io/*`) is
unchanged; existing imports, class names and public method signatures still work.

Delivered in this pass:

1. **Detection layer** — optional deep-learning backend (ultralytics YOLO /
   RT-DETR, ONNX) with checkpoint discovery, SHA-256 provenance, multi-scale
   full-image inference for masses and **overlapping tiled inference (640 px,
   30 % overlap) plus full-image context** for calcifications, class-aware NMS,
   per-task result caps, lazy imports and CPU-only operation. When no usable
   checkpoint exists the classical detectors run and are explicitly flagged
   *unvalidated*.
2. **Clinical layer** — ACR mass characterisation (shape/margin/density/
   orientation), BI-RADS 0–6 with evidence + management + validation status, ACR
   density A–D, calcification morphology/distribution/cluster analysis,
   localisation and measurements in px **and** mm with explicit spacing
   provenance, exam-level summary, JSON + self-contained HTML reporting.
3. **GUI** — rewritten main window (5 image tabs, 5 side tabs), worker-thread
   inference, cancel, batch processing, settings persistence, model-status panel,
   lesion detail panel with auto-selected top finding, consistent panel ↔ overlay
   ↔ report values, and a fixed zoom/overlay rendering root cause.
4. **Datasets/training** — converters for CBIS-DDSM, INbreast, VinDr-Mammo,
   MIAS, EMBED into one YOLO layout with **patient-level splits**, training CLI
   (YOLO/RT-DETR) that never overwrites checkpoints, evaluation CLI with
   mAP@0.5, mAP@0.5:0.95, P/R/F1 and **false positives per image**.
5. **Tests & tooling** — 73 automated tests (71 pass, 2 environment skips), a
   10-point end-to-end verification checklist (`python -m tools.verify`) and a
   checkpoint inspection CLI (`python -m tools.inspect_models`).

## 2. Files modified or created

### Modified

| File | Change |
|---|---|
| `config.py` | Model/DL/calcification blocks, `REPORT_DISCLAIMER`, `DETECTION_BACKEND`, `DL_DEVICE`, `BIRADS_MIN_FINDING_CONFIDENCE`, `BIRADS_UNVALIDATED_LABEL`, `ASSUMED_PIXEL_SPACING_MM`, model dirs auto-created. **Bug fixed:** `RISK_WEIGHTS` used `dataclasses.field()` outside a dataclass, i.e. it was a `Field` object, so every lesion-risk stage silently failed with `AttributeError`. Now a plain dict. Disclaimer now states the software is not a radiologist report. |
| `src/core/cad_system.py` | Rewritten pipeline: `analyze/process/process_array`, stage isolation via `_safe`, detection-ID assignment, per-detection measurement + characterisation, calcification analysis, density, BI-RADS with backend provenance, explainability/confidence, heatmaps, `_characterization_list` (adds `detection_id`/`location`/`bbox`), explicit `spacing_known/spacing_assumed`, backend notes appended to `warnings`. |
| `src/core/worker.py` | `CADWorker` emits `progress`, `result_ready(AnalysisResult)` **and** the legacy 6-arg `finished`, `error`, `cancelled`; cooperative `cancel()` at every progress callback. |
| `src/core/analysis_result.py` | Extended dataclass, `to_dict()`, JSON-safe conversion for numpy/enum/dataclass values. |
| `src/io/dicom_handler.py` | Rewritten: MONOCHROME1/2 inversion, 8/16-bit, `BitsStored`/`BitsAllocated`, signed pixel data, `WindowCenter/Width`, `RescaleSlope/Intercept`, `PixelSpacing`, view/laterality, patient/study tags, PNG/JPG/TIFF fallback. |
| `src/processing/preprocessor.py` | Rewritten: polarity normalisation, page/film-frame removal, mask via Otsu + morphological refinement + largest component, pectoral Hough 15–75° with masked fallback, CLAHE; vectorised (0.13 s on 1200×900). |
| `src/processing/segmentation_pipeline.py` | `nbins` fix; **new**: hard sanity check — an implausible (>95 %) mask triggers an Otsu foreground fallback; if it still fails the stage returns `valid=False`, confidence ≤ 0.15 and a warning instead of a flattering 0.85. |
| `src/processing/quality_analyzer.py` | Mask-coverage warning, quality score in result. |
| `src/models/tumor_detector.py` | Per-candidate evaluation (merging candidates before evaluation suppressed the true mass), `min_prominence`, deterministic cap (`max_detections=6`), ring-density guards. |
| `src/models/calcification_detector.py` | Performance rewrite (65.8 s → 1.3–3.6 s), micro/macro separation, vectorised cluster building, ring-density guards, `_erode_skin_line`, per-spot measurements (`individual_calcifications`) exposed to the lexicon stage. |
| `src/models/birads_estimator.py` | Full rewrite: evidence model over margin/shape/density/calcification lexicon, categories 0–6 with 4A/4B/4C, management text, reasons, confidence, `BIRADS_DESCRIPTIONS`, **low-confidence findings capped at category 3**, `validation_status`/`validation_warning`. |
| `src/models/breast_density_classifier.py` | ACR A–D labels, integer-threshold reshaping, **two-component GMM dense-area estimate** (replaces Otsu; the Otsu path produced 21 % on an obviously dense breast), method reported (`gmm2`/`otsu`). |
| `src/measurement/lesion_measurement_engine.py` | `spacing_known`/`spacing_basis` provenance through `measure_all`/`measure_detection`. |
| `src/models/visualizer.py` | Honours `show_labels/show_tumors/show_calcifications/show_arrows/show_masks/show_bboxes/show_centers`, `line_thickness`, `mask_opacity`, `highlight_id`; labels limited to the 10 highest-confidence findings so the overlay stays readable (all boxes/markers still drawn). |
| `src/ui/main_window.py` | Full rewrite (`MammographyCADApp`, ~1250 lines) — see §3. |
| `src/ui/zoomable_label.py` | Full rewrite — paint-based rendering (root cause of "overlays appear only after zoom/resize"), auto-fit on resize, dash-ring highlight, bbox→point-radius hit testing, right-click ignore/restore, `current_image()`/`rendered_pixmap()`. |
| `README.md` | Rewritten: pipeline table, run/train commands, backend resolution order, honest status section. |
| `requirements.txt` | Pinned core deps + notes; new `requirements-optional.txt` (pydicom, pytest, torch, ultralytics, onnxruntime, pyyaml). |
| `.gitignore` | Ignores `outputs/`, `sessions/`, `runs/`, `datasets/`, `*.pt/pth/onnx/engine`. |

### Created

| File | Purpose |
|---|---|
| `src/models/model_registry.py` | Checkpoint discovery, read-only inspection, SHA-256, class-vocabulary-based task routing, `runtime_status()`, `report()`. |
| `src/models/deep_detector.py` | `DeepDetectionBackend`: ultralytics/ONNX inference, multi-scale full-image + tiled prediction, class-aware `nms`, `to_detections()`, `backend_status()`, `last_error()`. |
| `src/models/lesion_characterization.py` | ACR mass lexicon (shape/margin/density/orientation/spiculation) with per-descriptor scores. |
| `src/models/calcification_analysis.py` | Morphology/distribution/cluster analysis, cluster criterion (≥3 within 10 mm), suspicion scoring, per-cluster measurements, spacing provenance. |
| `src/core/batch_processor.py` | `BatchProcessor(QThread)`: sequential analysis, per-file signals, cancellation, one report per file. |
| `src/reporting/report_builder.py`, `__init__.py` | JSON + self-contained HTML (base64 PNG overlay/enhanced/mask/heatmaps, provenance table, findings, BI-RADS, density, warnings, disclaimer). |
| `training/converters.py` | MIAS, CBIS-DDSM, INbreast, VinDr-Mammo, EMBED → unified layout, patient-level splits, manifest + dataset.yaml. |
| `training/prepare_dataset.py`, `train_detector.py`, `evaluate.py`, `__init__.py` | CLIs (see §8). |
| `tests/` (10 modules) | 73 tests: preprocessing, DICOM, detection, clinical, pipeline, reporting, models, GUI, datasets. |
| `tools/inspect_models.py`, `tools/verify.py`, `tools/__init__.py` | Checkpoint inspection; 10-point verification checklist. |
| `docs/ARCHITECTURE.md`, `DATASETS.md`, `CLINICAL_MAPPING.md`, `DIGITAL_EYE_CHECKPOINT_ANALYSIS.md`, `FINAL_REPORT.md` | Documentation. |
| `model/README.md` (+ empty `model/mass`, `model/calcification`) | Where checkpoints go; nothing is ever written there. |

No pre-existing file was deleted; no weight file was created, moved or
overwritten; nothing was committed or pushed.

## 3. GUI work (audit fixes)

| Reported problem | Root cause found | Fix |
|---|---|---|
| Overlays appear only after zoom/drag/resize | `ZoomableImageLabel` cached a composed `QPixmap` and only refreshed it in `resizeEvent`/zoom handlers | painting moved into `paintEvent`; every `update()` repaints image + overlays |
| Wrong coordinates / clicks miss | hit-testing compared only point radius in screen space | bbox test then point radius, both through `_screen_to_image_coords` |
| Hidden results / missing tabs | BI-RADS, density, calcification, lesion detail and model status had no widgets | 5 side tabs (Results, Clinical, Lesion detail, Settings, Models/Info) + 5 image tabs |
| Panel ≠ overlay ≠ report values | three independent re-computations | all views read the single `AnalysisResult`; characterisations are linked by `detection_id` |
| Broken workers/cancel | 6-arg `finished` only, no cancellation | new signals + cooperative cancel checked at each progress step |
| Empty lesion panel after analysis | nothing auto-selected | highest-ranked mass is selected automatically |
| Export/Batch buttons dead | stubs | `on_export` → `ReportBuilder.save`; `start_batch`/`BatchProcessor` with progress, per-file status, cancel, report export |
| Unreadable overlay (77 labels) | every detection labelled | labels limited to the 10 highest-confidence findings; all boxes kept |

Also: `silent_dialogs` mode (head-less/CI), `closeEvent` joins running threads
(no `QThread destroyed while running` aborts), settings persisted via
`QSettings`, backend combo writes `Config.DETECTION_BACKEND` and invalidates the
cached deep backends.

## 4. Digital Eye checkpoints — analysis result

`model/digital_eye/` **does not exist in the repository** and no `*.pt/.pth/.onnx/.engine`
is present anywhere (verified by `git status`, `find`, and
`python -m tools.inspect_models` → `Checkpoints: 0`).

* Architecture, class list, input requirements and capabilities of the three
  checkpoints **could not be determined** — they were not available. No value was
  invented for them.
* **No checkpoint was reused**, so no result in this build comes from a pretrained
  YOLO model. The classical detectors run and are reported as
  `backend: classical` / `unvalidated_classical_fallback`.
* Weights are opened read-only; `tests/test_models.py::test_registry_never_modifies_checkpoint`
  asserts the file bytes are unchanged after inspection.
* Mechanism delivered instead: `python -m tools.inspect_models` prints, per file,
  path, size, SHA-256, format, task, framework, **class names read from the
  file**, input size, inspection level, usability and issues; the registry only
  routes a checkpoint to a task when its own class names match, so a COCO
  vocabulary is refused rather than misused. Full details:
  `docs/DIGITAL_EYE_CHECKPOINT_ANALYSIS.md`.

## 5. Chosen architectures

| Task | Choice | Rationale |
|---|---|---|
| Mass detection | YOLOv8s / YOLO11s at 1024 px (RT-DETR-L optional if GPU allows), multi-scale (1024, 1536) inference | one-stage detectors give the best speed/accuracy balance for 1–6 cm masses and run on CPU |
| Calcification detection | YOLOv8s at 1280 px **with overlapping 640 px tiles (30 % overlap) + full-image context**, class-aware NMS, individual calcification **and** cluster detections (cap 300/task) | calcifications are 0.1–1 mm; tiling preserves resolution, the full-image pass preserves context |
| Calibration / FP control | confidence + NMS thresholds in `Config`, ring-density and prominence gates, per-task caps, cluster criterion (≥3 within 10 mm), BI-RADS confidence gate | the classical fallback is FP-heavy; the guards keep FPs from inflating the assessment |
| Runtime | CPU (`CAD_DEVICE=cpu`), lazy `torch`/`ultralytics` import, ONNX path | works on the 2-core / 1 GB target, no new mandatory dependency |

## 6. Clinical status (BI-RADS, density, characterisation)

* **BI-RADS 0–6** implemented including 4A/4B/4C; every category ships with
  `category_text`, `management`, `reasons`, `evidence`, `confidence`,
  `validation_status`. Category 6 is never inferred from pixels. Findings below
  0.60 confidence cannot exceed category 3. Verified on the phantom: **BI-RADS 4
  (4A)** for a spiculated mass, and category 0 when technical quality is
  inadequate.
* **ACR density A–D** from a two-component GMM on breast pixels; thresholds
  25/50/75 %. Real image measurements: dense 70.7 % → C, 49.4 % → C,
  phantom 52.3 % → C; the previous Otsu implementation gave 21 % → A on an
  obviously dense breast (wrong category).
* **Mass characterisation**: shape, margin, density, orientation, spiculation
  index, lobulation/spike counts, size in mm.
* **Calcifications**: morphology (punctate/amorphous/pleomorphic/linear/branching/
  coarse/vascular/…), distribution (diffuse/regional/grouped/linear/segmental/
  isolated), cluster count/extent/spacing, suspicion score.
* **Measurements**: max diameter, area, perimeter, circularity in px and mm;
  `spacing_basis` says DICOM / assumed / pixel units.
* Everything is labelled *software-assisted*; the disclaimer states it is not a
  radiologist report and must be confirmed by a radiologist.

## 7. Tests actually run (with results)

Environment: 2 vCPU, ~1 GB RAM, Python 3.13, no CUDA, no torch/ultralytics, no
datasets. Qt run with `QT_QPA_PLATFORM=offscreen` (system X libs are absent in
this container).

```
$ QT_QPA_PLATFORM=offscreen python3 -m pytest tests/ -q
71 passed, 2 skipped in 41 s        (73 collected)
```
(2 skips: no pectoral muscle in the synthetic phantom; an optional tiling helper
name that does not exist in the final API.)

```
$ python3 -m tools.verify
10 passed, 0 skipped, 0 failed of 10 checks
```

| Check | Result |
|---|---|
| 1 application imports / config | PASS (backend=auto, device=cpu) |
| 2 image loading + pre-processing | PASS (600×800, mask coverage 49.7 %) |
| 3 DICOM 16-bit MONOCHROME1 + spacing + view | PASS (inverted, 0.05 mm/px, MLO) |
| 4 mass detection vs ground truth | PASS (6 candidates, closest **1.0 px**) |
| 5 calcification cluster detection | PASS (closest **2.0 px**, 1 cluster of 7 spots) |
| 6 BI-RADS + ACR density + evidence | PASS (BI-RADS 4 (4A), ACR C 52.3 %, validation flagged) |
| 7 reporting JSON+HTML | PASS (274 kB self-contained HTML, base64 images, no external URLs) |
| 8 model registry / CPU inference availability | PASS (torch/ultralytics absent → classical, reported) |
| 9 forced deep backend without weights | PASS (degrades with warning, no crash) |
| 10 GUI constructs offscreen | PASS (5 image tabs, 5 side tabs) |

Measured timings on this machine: pre-processing 0.13 s; mass detection
1.6–3.9 s; calcification detection 1.3–3.6 s; full analysis 3.6–5.6 s for
600×800 – 960×720 images.

**Not measured, therefore not claimed:** no mAP, no accuracy, no sensitivity/
specificity on real data. Training could not be executed (no torch, no datasets).

## 8. Exact commands

```bash
# install
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt      # pydicom, torch, ultralytics, pytest

# run the GUI
python main.py

# inspect checkpoints (put them in model/digital_eye/ first)
python -m tools.inspect_models
python -m tools.inspect_models --json

# verification checklist + tests
python -m tools.verify
python -m pytest tests/ -q
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui.py -q

# dataset preparation (patient-level splits)
python -m training.prepare_dataset --dataset mias        --root data/MIAS      --out datasets/mias
python -m training.prepare_dataset --dataset cbis-ddsm   --root data/CBIS-DDSM --out datasets/cbis
python -m training.prepare_dataset --dataset inbreast    --root data/INbreast  --out datasets/inbreast
python -m training.prepare_dataset --dataset vindr-mammo --root data/VinDr     --out datasets/vindr
python -m training.prepare_dataset --dataset embed       --root data/EMBED     --out datasets/embed

# train
python -m training.train_detector --data datasets/cbis/dataset.yaml --task mass \
    --model yolov8n.pt --epochs 50 --imgsz 1024 --batch 4 --device cpu --project runs
python -m training.train_detector --data datasets/calc/dataset.yaml --task calcification \
    --model yolov8s.pt --epochs 80 --imgsz 1280 --device cpu

# evaluate (mAP + FP/image)
python -m training.evaluate --weights runs/mass/weights/best.pt \
    --data datasets/cbis/dataset.yaml --split test --conf 0.25 --imgsz 1024

# install a trained checkpoint without overwriting anything
python -m training.train_detector ... --install
# or simply: CAD_MASS_MODEL=/path/mass_best.pt python main.py
```

## 9. Remaining limitations

1. **No weights, no datasets in the repository** → the DL detectors are
   implemented and unit-tested at the inference-plumbing level but have never
   produced a real detection; every result here is classical and flagged
   unvalidated.
2. **The classical detectors are FP-heavy on real mammograms** (real2.jpg: 77
   candidates, 6 mass + 71 calcification). Two tuning rounds (alternative
   area/circularity/fill/prominence filters, ring-density gate 0.55/0.60) did not
   improve the trade-off and were reverted rather than over-fitted; the BI-RADS
   confidence gate prevents FPs from inflating the category, but the FP rate
   remains the dominant quality limit.
3. **BI-RADS thresholds are heuristic** (documented weights, no ROC calibration
   on a labelled cohort). They must be re-fitted after training.
4. **No clinical validation**: no reader study, no ground-truth comparison on
   real cases; the descriptor values are image measurements, not diagnoses.
5. **ACR density** is a 2D histogram/GMM estimate, not volumetric (no MRI/DXA
   correlation).
6. **Photo-of-screen images** (`real1.jpg`, mask 93.8 %) cannot be salvaged by
   the segmentation; they are accepted, quality-warned and reported with
   `segmentation.valid=false`.
7. **Training/evaluation scripts were only exercised on synthetic datasets**
   (MIAS-like, CSV bbox); CBIS-DDSM/INbreast/VinDr/EMBED converters need a real
   download to be validated against the actual CSV/XML variants, and their column
   matchers fail loudly with the observed columns if a layout differs.
8. **No `torch` in this environment** → checkpoint inspection runs in header mode;
   with torch installed the same command reads authoritative metadata.
9. **Sandbox-specific**: PyQt6 needs X libs (`libxkbcommon`, `libxcb*`, `libGL`);
   head-less runs use `QT_QPA_PLATFORM=offscreen` with extracted libraries in
   `LD_LIBRARY_PATH`. On a normal desktop this is not needed.

## 10. Suggested next steps (in order)

1. Drop the three Digital Eye checkpoints into `model/digital_eye/`, run
   `python -m tools.inspect_models`, and record the SHA-256 values in the thesis
   appendix. Only checkpoints whose own class names include a mass/calcification
   class will be routed automatically.
2. Download CBIS-DDSM (mass + calcification) and VinDr-Mammo, convert, train
   YOLOv8s mass (1024 px) and calcification (1280 px + tiling), then evaluate with
   `training/evaluate.py` and paste the real mAP / FP-per-image table into the
   thesis — replacing every "not measured" note here.
3. Re-fit the BI-RADS weights on the resulting score distributions per category
   and report the ROC/AUC per category.

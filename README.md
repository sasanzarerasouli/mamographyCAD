# MammographyCAD

Simulation program for detecting breast abnormalities in mammograms — master's
thesis project. PyQt6 desktop application, classical computer vision plus an
optional deep-learning backend, BI-RADS / ACR-density decision support, and
JSON/HTML reporting.

> **Not a medical device.** Every output is a software-assisted estimate for
> research and decision support. It is not a radiologist report and must be
> reviewed by a qualified radiologist.

![GUI](Screenshot%202025-07-16%20164159.jpg)

---

## 1. What the application does

| Stage | Module | Output |
|---|---|---|
| Image decoding | `src/io/dicom_handler.py` | grayscale array + metadata (DICOM MONOCHROME1/2, 8/16-bit, signed, `BitsStored`, `WindowCenter/Width`, `PixelSpacing`, view/laterality; PNG/JPG/TIFF fallback) |
| Pre-processing | `src/processing/preprocessor.py` | orientation normalisation, page/frame removal, breast mask, pectoral-muscle removal, contrast enhancement |
| Mass detection | `src/models/tumor_detector.py` (+ `deep_detector.py`) | mass candidates with contours, ROI statistics, spiculation |
| Calcification detection | `src/models/calcification_detector.py` (+ `deep_detector.py`) | individual calcifications and clusters (micro/macro) |
| Characterisation | `src/models/lesion_characterization.py` | ACR lexicon shape / margin / density / orientation |
| Measurements | `src/measurement/lesion_measurement_engine.py` | max diameter, area, perimeter, circularity in px and mm |
| Calcification lexicon | `src/models/calcification_analysis.py` | morphology, distribution, cluster count/extent, suspicion |
| Density | `src/models/breast_density_classifier.py` | ACR A–D + dense-area percentage (2-component GMM) |
| Assessment | `src/models/birads_estimator.py` | BI-RADS 0–6 with evidence, management, and validation status |
| Visualisation | `src/models/visualizer.py`, `src/models/suspicion_heatmap_generator.py` | overlay, density heatmap, suspicion heatmap |
| Reporting | `src/reporting/report_builder.py` | JSON + self-contained HTML (base64 images, provenance, disclaimer) |

## 2. Run

```bash
python -m pip install -r requirements.txt            # core
python -m pip install -r requirements-optional.txt   # pydicom, torch, ultralytics, pytest

python main.py
```

If PyQt6 cannot load its X libraries in a head-less container, install the
`libxkbcommon`/`libxcb`/`libGL` packages or set
`QT_QPA_PLATFORM=offscreen` for tests.

### Command line helpers

```bash
python -m tools.inspect_models            # what checkpoints are visible + SHA-256
python -m tools.inspect_models --json
python -m pytest tests/ -q                # full test suite (73 tests)
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui.py -q   # GUI only
```

## 3. Detection backends

Resolution order per task (mass, calcification):

1. `CAD_MASS_MODEL` / `CAD_CALC_MODEL` environment override;
2. `model/mass/*`, `model/calcification/*`;
3. `model/digital_eye/*` — **only** if the checkpoint's own class names contain a
   matching class. The class vocabulary is read from the checkpoint and is never
   guessed; a checkpoint whose classes cannot be read is reported as unusable;
4. otherwise the classical detector. The result is then marked
   `backend: classical` / `unvalidated_classical_fallback` and every exported
   report carries the warning that no validated model was used.

Checkpoints are opened read-only and their SHA-256 is reported in the GUI
(*Models / Info* tab) and in every report. Nothing in `model/` is ever written.

`Config.DETECTION_BACKEND` selects the mode: `auto` (default), `on`, `off`.

## 4. Training and evaluation

Datasets are converted to one unified YOLO layout with **patient-level** splits:

```bash
python -m training.prepare_dataset --dataset mias        --root data/MIAS        --out datasets/mias
python -m training.prepare_dataset --dataset cbis-ddsm   --root data/CBIS-DDSM   --out datasets/cbis
python -m training.prepare_dataset --dataset inbreast    --root data/INbreast    --out datasets/inbreast
python -m training.prepare_dataset --dataset vindr-mammo --root data/VinDr       --out datasets/vindr
python -m training.prepare_dataset --dataset embed       --root data/EMBED       --out datasets/embed

python -m training.train_detector --data datasets/cbis/dataset.yaml --task mass \
    --model yolov8n.pt --epochs 50 --imgsz 1024 --device cpu

python -m training.evaluate --weights runs/mass/weights/best.pt \
    --data datasets/cbis/dataset.yaml --split test --conf 0.25
```

`training/evaluate.py` reports mAP@0.5, mAP@0.5:0.95, precision, recall, F1 and
**false positives per image** (independent IoU-0.5 matching). See
[docs/DATASETS.md](docs/DATASETS.md).

## 5. Status of this repository snapshot

* The source tree contains **no** trained weights and **no** datasets, so
  `model/digital_eye/` could not be analysed — see
  [docs/DIGITAL_EYE_CHECKPOINT_ANALYSIS.md](docs/DIGITAL_EYE_CHECKPOINT_ANALYSIS.md).
  No checkpoint was reused; the classical detectors run and are flagged as
  unvalidated.
* No accuracy, mAP or clinical validation number is claimed anywhere in this
  repository, because none was measured on real data here. The numbers in
  `tests/` are regression checks on synthetic phantoms with known ground truth.

Documentation: [architecture](docs/ARCHITECTURE.md) ·
[datasets](docs/DATASETS.md) · [clinical mapping](docs/CLINICAL_MAPPING.md) ·
[Digital Eye analysis](docs/DIGITAL_EYE_CHECKPOINT_ANALYSIS.md)

## 6. Licence

See [LICENSE](LICENSE).

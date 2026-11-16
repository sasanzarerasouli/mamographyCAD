# Datasets, splits, metrics

## Unified layout produced by `training/prepare_dataset.py`

```
<out>/
  images/{train,val,test}/<name>.png
  labels/{train,val,test}/<name>.txt     # "cls cx cy w h" normalised
  manifest.csv                           # file, split, patient, source, boxes, w, h
  dataset.yaml                           # ultralytics data description
```

Classes: `0 mass`, `1 microcalcification`, `2 macrocalcification`.

## Per-dataset conversion

| Dataset | Access | Annotation used | Converter |
|---|---|---|---|
| CBIS-DDSM | public (TCIA) | ROI masks + case-description CSVs (mass and calcification) | `convert_cbis_ddsm` (needs `pydicom`) |
| INbreast | public (agreement) | XML `<roi>` circles/ellipses, `Type` mass/calcification | `convert_inbreast` |
| VinDr-Mammo | public (PhysioNet) | `training/annotations.csv` bounding boxes | `convert_vindr_mammo` |
| MIAS | public | `Info.txt` circles (x, y, radius) | `convert_mias` |
| EMBED | public (Duke/Emory) | metadata CSV bounding boxes | `convert_embed` |

Notes that matter for honesty:

* CBIS-DDSM ROIs are binary masks; the converter derives the bounding box from
  the **full-mammogram ROI mask**, so boxes are always in full-image
  coordinates. The cropped-image path is not silently mixed in.
* INbreast ROI coordinates are normalised 0..1 in the XML and are converted to
  pixels through the image size read from disk.
* VinDr-Mammo/EMBED use a tolerant column matcher (`image_id`/`filename`,
  `xmin`/`x1`, …). If a CSV layout is not recognised the converter fails with the
  list of columns it found instead of guessing.
* Labels are mapped by keyword: `mass|tumor|lesion → 0`, calcification →
  `1` unless it is described as coarse/large/popcorn/vascular/dystrophic/benign
  (`→ 2`). Unknown labels fall back to `0` and are visible in the manifest.

## Splits

`DatasetWriter._split_by_patient` splits **by patient id**, default 70/15/15,
seeded (`--seed`), with non-empty val/test guaranteed when enough patients
exist. `tests/test_datasets.py::test_patient_level_split_has_no_leakage` asserts
that no patient appears in two splits and that the split is reproducible for a
fixed seed.

## Recommended detector configuration

| Task | Architecture | Why |
|---|---|---|
| Mass | YOLOv8s / YOLO11s, or RT-DETR-L if the GPU budget allows | one-stage detectors handle 1–6 cm masses well at 1024 px |
| Calcification | YOLOv8s at 1280 px with 640 px tiles (30 % overlap) + full-image context | individual calcifications are 0.1–1 mm; tiling preserves resolution |

Training is CPU-feasible only for the smallest variants; the scripts default to
`--device cpu` but accept `0` for CUDA.

## Metrics reported

`training/evaluate.py` returns mAP@0.5, mAP@0.5:0.95, precision, recall, F1 from
ultralytics **plus** an independent IoU-0.5 matching pass that reports
`false_positives_per_image`, point precision/recall and per-split counts. FP per
image is the number a CAD reader actually cares about and is not part of the
standard mAP output.

**No metric in this repository was measured on real data**, because no dataset
was available in this environment. Any thesis table must be regenerated with the
commands above.

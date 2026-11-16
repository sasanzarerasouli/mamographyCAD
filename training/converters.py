"""Unified dataset conversion for mammography detection training.

Supported public datasets and their *verified* annotation content:

====================  =========================  ==============================
Dataset               Mass annotations           Calcification annotations
====================  =========================  ==============================
CBIS-DDSM             bounding boxes + ROI masks bounding boxes + ROI masks
INbreast              XML circles/ellipses       XML circles/ellipts (clusters)
VinDr-Mammo           bounding boxes             bounding boxes
MIAS                  centre + radius circles    centre + radius circles
EMBED                 bounding boxes             bounding boxes
====================  =========================  ==============================

Every converter writes the same layout, which both ultralytics YOLO training and
the evaluation script consume::

    <out>/
      images/{train,val,test}/<name>.png
      labels/{train,val,test}/<name>.txt      # "cls cx cy w h" normalized
      manifest.csv                            # file, split, patient, source, n_boxes
      dataset.yaml                            # ultralytics data description

Class ids are fixed: ``0 = mass``, ``1 = microcalcification``,
``2 = macrocalcification``.  A dataset that only annotates "calcification"
without separating micro/macro is mapped to class 1 (the clinically relevant
one) and the mapping is recorded in the manifest.

Splitting is **patient level**: all images of a patient stay in the same split,
which prevents the leakage that an image-level split would introduce.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import logging
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("MammographyCAD.dataset")

CLASS_NAMES = {0: "mass", 1: "microcalcification", 2: "macrocalcification"}

MASS_TOKENS = ("mass", "tumor", "tumour", "lesion")
CALC_TOKENS = ("calc",)
MICRO_TOKENS = ("micro", "small", "fine", "pleomorphic", "amorphous", "punctate")
MACRO_TOKENS = (
    "macro", "coarse", "large", "popcorn", "vascular", "dystrophic",
    # "benign calcification" (VinDr-Mammo) is the coarse/secretory category
    "benign",
)


@dataclass
class Annotation:
    """One bounding box in full-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""
    class_id: int = 0

    def normalized(self, width: int, height: int) -> Optional[str]:
        x1 = max(0.0, min(self.x1, width - 1))
        x2 = max(0.0, min(self.x2, width - 1))
        y1 = max(0.0, min(self.y1, height - 1))
        y2 = max(0.0, min(self.y2, height - 1))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def label_to_class(label: str) -> int:
    """Map a dataset-specific finding label to the unified class id."""
    text = (label or "").lower()
    if any(token in text for token in MASS_TOKENS):
        return 0
    if any(token in text for token in CALC_TOKENS):
        if any(token in text for token in MACRO_TOKENS):
            return 2
        return 1
    return 0


@dataclass
class DatasetSample:
    """One image and its annotations before conversion."""

    image_path: str
    patient_id: str
    annotations: List[Annotation] = field(default_factory=list)
    source: str = ""
    extra: Dict[str, str] = field(default_factory=dict)


# ====================================================================== #
#  Generic writer                                                        #
# ====================================================================== #
class DatasetWriter:
    """Writes converted samples into the unified YOLO-style layout."""

    def __init__(
        self,
        output_dir: str,
        splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        image_size: Optional[int] = 1024,
        copy_images: bool = True,
    ) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.splits = splits
        self.seed = seed
        self.image_size = image_size
        self.copy_images = copy_images
        self.manifest_rows: List[Dict[str, str]] = []

        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(self.output_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "labels", split), exist_ok=True)

    # ------------------------------------------------------------------ #
    def write(self, samples: Sequence[DatasetSample], dataset_name: str) -> Dict[str, object]:
        if not samples:
            raise ValueError("no samples to write")

        grouped = self._split_by_patient(samples)
        stats: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

        for split, sample_list in grouped.items():
            for sample in sample_list:
                image = self._load_image(sample.image_path)
                if image is None:
                    logger.warning("Skipping unreadable image: %s", sample.image_path)
                    continue
                height, width = image.shape[:2]

                name = self._unique_name(sample)
                lines = []
                for annotation in sample.annotations:
                    line = annotation.normalized(width, height)
                    if line:
                        lines.append(line)
                if not lines:
                    # Background images are useful negatives for a detector
                    logger.debug("Sample has no valid boxes: %s", sample.image_path)

                if self.copy_images:
                    image_rel = os.path.join("images", split, f"{name}.png")
                    target = os.path.join(self.output_dir, image_rel)
                    if self.image_size and max(height, width) > self.image_size:
                        scale = self.image_size / max(height, width)
                        resized = cv2.resize(
                            image, (int(width * scale), int(height * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                        cv2.imwrite(target, resized)
                    else:
                        cv2.imwrite(target, image)

                label_rel = os.path.join("labels", split, f"{name}.txt")
                with open(os.path.join(self.output_dir, label_rel), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + ("\n" if lines else ""))

                self.manifest_rows.append(
                    {
                        "file": name,
                        "split": split,
                        "patient_id": sample.patient_id,
                        "source": sample.source or dataset_name,
                        "source_path": sample.image_path,
                        "boxes": str(len(lines)),
                        "width": str(width),
                        "height": str(height),
                    }
                )
                stats[split] += 1

        self._write_manifest()
        self._write_yaml(dataset_name)
        logger.info(
            "Dataset written to %s (train=%d, val=%d, test=%d)",
            self.output_dir, stats["train"], stats["val"], stats["test"],
        )
        return {
            "output_dir": self.output_dir,
            "stats": stats,
            "patients": len({s.patient_id for s in samples}),
            "files": len(samples),
        }

    # ------------------------------------------------------------------ #
    def _split_by_patient(self, samples: Sequence[DatasetSample]) -> Dict[str, List[DatasetSample]]:
        patients = sorted({sample.patient_id for sample in samples})
        rng = random.Random(self.seed)
        rng.shuffle(patients)

        n = len(patients)
        n_train = max(1, int(round(n * self.splits[0]))) if n > 2 else max(1, n - 1)
        n_val = int(round(n * self.splits[1])) if n > 2 else (1 if n > 1 else 0)
        train_patients = set(patients[:n_train])
        val_patients = set(patients[n_train:n_train + n_val])

        grouped: Dict[str, List[DatasetSample]] = {"train": [], "val": [], "test": []}
        for sample in samples:
            if sample.patient_id in train_patients:
                grouped["train"].append(sample)
            elif sample.patient_id in val_patients:
                grouped["val"].append(sample)
            else:
                grouped["test"].append(sample)

        # Guarantee non-empty validation/test sets when enough patients exist
        if not grouped["val"] and len(grouped["train"]) > 2:
            grouped["val"].append(grouped["train"].pop())
        if not grouped["test"] and len(grouped["train"]) > 2:
            grouped["test"].append(grouped["train"].pop())
        return grouped

    @staticmethod
    def _load_image(path: str) -> Optional[np.ndarray]:
        # ``training`` is a top-level package (not a sub-package of src), so the
        # import is absolute with the repository root on sys.path.
        try:
            from src.processing.preprocessor import MammographyPreprocessor
        except ImportError:  # pragma: no cover - when run from another cwd
            import sys

            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.processing.preprocessor import MammographyPreprocessor

        try:
            image, _ = MammographyPreprocessor.load_image(path)
            return image
        except Exception:
            return None

    @staticmethod
    def _unique_name(sample: DatasetSample) -> str:
        digest = hashlib.sha1(sample.image_path.encode("utf-8")).hexdigest()[:8]
        base = os.path.splitext(os.path.basename(sample.image_path))[0]
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)[:60]
        return f"{base}_{digest}"

    def _write_manifest(self) -> None:
        path = os.path.join(self.output_dir, "manifest.csv")
        fieldnames = ["file", "split", "patient_id", "source", "source_path", "boxes", "width", "height"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.manifest_rows)

    def _write_yaml(self, dataset_name: str) -> None:
        path = os.path.join(self.output_dir, "dataset.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# generated from {dataset_name}\n")
            fh.write(f"path: {self.output_dir}\n")
            fh.write("train: images/train\nval: images/val\ntest: images/test\n")
            fh.write("names:\n")
            for class_id, name in CLASS_NAMES.items():
                fh.write(f"  {class_id}: {name}\n")


# ====================================================================== #
#  MIAS                                                                  #
# ====================================================================== #
def convert_mias(root: str, output_dir: str, seed: int = 42) -> Dict[str, object]:
    """Convert the MIAS mini-mammographic database.

    ``Info.txt`` lists, per image, the reference code, the background tissue
    class, the abnormality class (``CALC``, ``CIRC``, ``SPIC``, ``MISC``,
    ``ASYM``, ``NORM``), the severity, and the annotation circle as
    ``x y radius`` in pixel coordinates.  Both the space-delimited file and
    fixed-width variants without separators are accepted.
    """
    info_path = None
    for candidate in ("Info.txt", "info.txt", "README"):
        path = os.path.join(root, candidate)
        if os.path.isfile(path) and candidate.lower() != "readme":
            info_path = path
            break
    if info_path is None:
        raise FileNotFoundError(
            "MIAS Info.txt not found. Expected <root>/Info.txt next to the PGM images."
        )

    samples: Dict[str, DatasetSample] = {}
    with open(info_path, "r", encoding="latin-1") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if len(line.strip()) < 10 or line.strip().startswith("#"):
                continue
            # --- reference / annotation circle -------------------------- #
            head = line.strip()
            # Known class tokens terminate the reference when columns run
            # together (fixed-width variants with no separators).
            for marker in ("CALC", "CIRC", "SPIC", "MISC", "ASYM", "NORM"):
                index = head.upper().find(marker)
                if index > 0:
                    head = head[:index]
                    break
            reference = head.split()[0] if head.split() else ""
            if not reference:
                continue

            image_path = None
            resolved_reference = reference
            # Try the longest prefix first, then shorter ones: that resolves
            # "mdb100_L" out of a glued "mdb100_LFCALC" too.
            prefixes = [reference] + [
                reference[:length] for length in range(len(reference) - 1, 2, -1)
            ]
            for prefix in prefixes:
                for pattern in (
                    f"{prefix}.pgm", f"{prefix}.PGM",
                    f"**/{prefix}.pgm", f"**/{prefix}.PGM",
                ):
                    matches = glob.glob(os.path.join(root, pattern), recursive=True)
                    if matches:
                        image_path = matches[0]
                        resolved_reference = prefix
                        break
                if image_path:
                    break
            if image_path is None:
                continue

            abnormality = next(
                (token.upper() for token in line.split()
                 if token.upper() in ("CALC", "CIRC", "SPIC", "MISC", "ASYM", "NORM")),
                "",
            )
            if not abnormality:
                match_abnormality = re.search(
                    r"(CALC|CIRC|SPIC|MISC|ASYM|NORM)", line.upper()
                )
                abnormality = match_abnormality.group(1) if match_abnormality else ""
            if not abnormality:
                continue

            tail = line[line.find(resolved_reference) + len(resolved_reference):]
            numbers = re.findall(r"-?\d+(?:\.\d+)?", tail)
            if len(numbers) < 3:
                continue
            try:
                x, y, radius = (float(value) for value in numbers[-3:])
            except ValueError:
                continue
            reference = resolved_reference

            sample = samples.setdefault(
                image_path,
                DatasetSample(
                    image_path=image_path,
                    patient_id=reference.split("_")[0],
                    source="MIAS",
                ),
            )
            if abnormality == "NORM" or radius <= 0:
                continue
            # CIRC / SPIC / MISC are mass findings; ASYM is an asymmetric
            # density, which is treated as a mass-class finding (there is no
            # separate class for it in the unified schema); CALC is a
            # calcification.
            label = (
                "mass"
                if abnormality in ("CIRC", "SPIC", "MISC", "ASYM")
                else "calcification"
            )
            sample.annotations.append(
                Annotation(
                    x1=x - radius,
                    y1=y - radius,
                    x2=x + radius,
                    y2=y + radius,
                    label=label,
                    class_id=label_to_class(label),
                )
            )

    writer = DatasetWriter(output_dir, seed=seed)
    return writer.write(list(samples.values()), "MIAS")


# ====================================================================== #
#  CBIS-DDSM                                                             #
# ====================================================================== #
def convert_cbis_ddsm(root: str, output_dir: str, seed: int = 42) -> Dict[str, object]:
    """Convert CBIS-DDSM (mass and calcification case description CSVs).

    The CSV rows reference three DICOM files per case: the cropped image, the
    ROI mask and the full mammogram.  Annotations are taken from the **ROI mask
    of the full mammogram** when present, otherwise the ROI is derived from the
    cropped image and mapped through its bounding box, so the resulting boxes
    are always in full-image coordinates.
    """
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("CBIS-DDSM conversion requires pydicom") from exc

    csv_paths = glob.glob(os.path.join(root, "**", "*case_description*test_set.csv"), recursive=True)
    csv_paths += glob.glob(os.path.join(root, "**", "*case_description*train_set.csv"), recursive=True)
    if not csv_paths:
        raise FileNotFoundError(
            "No CBIS-DDSM case description CSV found under the given root."
        )

    samples: List[DatasetSample] = []
    for csv_path in sorted(set(csv_paths)):
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                normalised = {self_key.strip().lower(): value for self_key, value in row.items()}
                paths = next(
                    (
                        value for key, value in normalised.items()
                        if "file path" in key or "image file path" in key
                    ),
                    "",
                )
                parts = [part.strip() for part in str(paths).split("/") if part.strip()]
                if len(parts) < 3:
                    continue
                full_rel = parts[-1]
                mask_rel = parts[-2]
                image_path = _resolve_dicom(root, full_rel)
                mask_path = _resolve_dicom(root, mask_rel)
                if image_path is None:
                    continue

                label = "mass" if "mass" in os.path.basename(csv_path).lower() else "calcification"
                annotations: List[Annotation] = []
                if mask_path is not None:
                    bbox = _dicom_mask_bbox(mask_path, pydicom)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        annotations.append(
                            Annotation(x1, y1, x2, y2, label, label_to_class(label))
                        )

                sample = DatasetSample(
                    image_path=image_path,
                    patient_id=normalised.get("patient id", "") or os.path.basename(image_path),
                    source="CBIS-DDSM",
                    annotations=annotations,
                )
                sample.extra["abnormality"] = normalised.get("abnormality type", "")
                samples.append(sample)

    writer = DatasetWriter(output_dir, seed=seed)
    return writer.write(samples, "CBIS-DDSM")


def _resolve_dicom(root: str, relative: str) -> Optional[str]:
    for ext in ("", ".dcm", ".DCM"):
        candidate = os.path.join(root, relative + ext)
        if os.path.isfile(candidate):
            return candidate
    matches = glob.glob(os.path.join(root, "**", os.path.basename(relative) + "*"), recursive=True)
    return matches[0] if matches else None


def _dicom_mask_bbox(path: str, pydicom) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of a binary ROI mask stored as a DICOM image."""
    try:
        dataset = pydicom.dcmread(path)
        array = np.asarray(dataset.pixel_array)
        mask = array > 0
        if not mask.any():
            return None
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
        return x1, y1, x2, y2
    except Exception as exc:  # pragma: no cover - corrupt DICOM
        logger.warning("ROI mask could not be read (%s): %s", path, exc)
        return None


# ====================================================================== #
#  INbreast                                                              #
# ====================================================================== #
def convert_inbreast(root: str, output_dir: str, seed: int = 42) -> Dict[str, object]:
    """Convert INbreast XML annotations.

    INbreast XML uses ``<image>`` elements with ``<roi>`` children that carry
    ``Center X``/``Center Y`` (normalised 0..1) and ``Radius`` (normalised),
    plus a ``Type`` of ``Mass`` or ``Calcification``.
    """
    xml_paths = glob.glob(os.path.join(root, "**", "*.xml"), recursive=True)
    if not xml_paths:
        raise FileNotFoundError("No INbreast XML annotation files found.")

    samples: List[DatasetSample] = []
    for xml_path in xml_paths:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            logger.warning("Invalid XML skipped (%s): %s", xml_path, exc)
            continue
        root_element = tree.getroot()
        for image_element in root_element.iter("image"):
            name = image_element.get("name") or ""
            image_path = _resolve_dicom(root, name) or _resolve_any(root, name)
            if image_path is None:
                continue
            sample = DatasetSample(
                image_path=image_path,
                patient_id=os.path.splitext(name)[0].split("_")[0] or name,
                source="INbreast",
            )
            for roi in image_element.iter("roi"):
                label = (roi.get("Type") or "").strip() or "Mass"
                center_x = _first_float(roi, ("Center X", "center_x", "x"))
                center_y = _first_float(roi, ("Center Y", "center_y", "y"))
                radius = _first_float(roi, ("Radius", "radius"))
                if center_x is None or center_y is None or radius is None:
                    continue
                # INbreast coordinates are normalised to the image size
                x1 = (center_x - radius) * 1000.0
                x2 = (center_x + radius) * 1000.0
                y1 = (center_y - radius) * 1000.0
                y2 = (center_y + radius) * 1000.0
                sample.annotations.append(
                    Annotation(x1, y1, x2, y2, label, label_to_class(label))
                )
            samples.append(sample)

    writer = DatasetWriter(output_dir, seed=seed)
    return writer.write(samples, "INbreast")


def _first_float(element, names: Iterable[str]) -> Optional[float]:
    for name in names:
        value = element.get(name)
        if value is None:
            for child in element:
                if child.tag.lower() == name.lower():
                    value = child.text
                    break
        if value:
            try:
                return float(str(value).strip())
            except ValueError:
                continue
    return None


def _resolve_any(root: str, name: str) -> Optional[str]:
    if not name:
        return None
    base = os.path.splitext(os.path.basename(name))[0]
    for pattern in (f"{name}", f"{base}.*", f"**/{base}.*"):
        matches = glob.glob(os.path.join(root, pattern), recursive=True)
        if matches:
            return matches[0]
    return None


# ====================================================================== #
#  VinDr-Mammo / EMBED (CSV with bounding boxes)                          #
# ====================================================================== #
_COLUMN_ALIASES = {
    "image": ("image_id", "image", "filename", "file_name", "sop_instance_uid", "image_path"),
    "x1": ("xmin", "x1", "x_min", "bbox_x1"),
    "y1": ("ymin", "y1", "y_min", "bbox_y1"),
    "x2": ("xmax", "x2", "x_max", "bbox_x2"),
    "y2": ("ymax", "y2", "y_max", "bbox_y2"),
    "label": ("finding_categories", "finding_category", "category", "label", "finding", "type"),
    "patient": ("patient_id", "patient", "study_id"),
}


def _pick(row: Dict[str, str], key: str) -> Optional[str]:
    for alias in _COLUMN_ALIASES[key]:
        for column, value in row.items():
            if column.strip().lower() == alias:
                return value
    return None


def convert_bbox_csv(
    root: str,
    output_dir: str,
    csv_name: str,
    dataset_name: str,
    image_glob: str = "**/*.dicom",
    seed: int = 42,
) -> Dict[str, object]:
    """Convert a bounding-box CSV dataset (VinDr-Mammo, EMBED, custom)."""
    csv_path = os.path.join(root, csv_name)
    if not os.path.isfile(csv_path):
        candidates = glob.glob(os.path.join(root, "**", csv_name), recursive=True)
        if not candidates:
            raise FileNotFoundError(f"{csv_name} not found under {root}")
        csv_path = candidates[0]

    samples: Dict[str, DatasetSample] = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        available = [column for column in (reader.fieldnames or [])]
        for row in reader:
            image_id = _pick(row, "image")
            if not image_id:
                continue
            image_path = _resolve_any(root, image_id)
            if image_path is None:
                matches = glob.glob(
                    os.path.join(root, "**", f"{image_id}*"), recursive=True
                )
                image_path = matches[0] if matches else None
            if image_path is None:
                continue

            patient = _pick(row, "patient") or image_id
            sample = samples.setdefault(
                image_path,
                DatasetSample(image_path=image_path, patient_id=str(patient), source=dataset_name),
            )
            coords = [_pick(row, key) for key in ("x1", "y1", "x2", "y2")]
            if any(value in (None, "") for value in coords):
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in coords)
            except (TypeError, ValueError):
                continue
            label = _pick(row, "label") or "Mass"
            sample.annotations.append(
                Annotation(x1, y1, x2, y2, label, label_to_class(label))
            )

    if not samples:
        raise ValueError(
            f"No usable rows in {csv_path}. Columns found: {available}. "
            "Required: an image id plus xmin/ymin/xmax/ymax."
        )

    writer = DatasetWriter(output_dir, seed=seed)
    return writer.write(list(samples.values()), dataset_name)


def convert_vindr_mammo(root: str, output_dir: str, seed: int = 42) -> Dict[str, object]:
    """VinDr-Mammo: DICOM images plus ``training/annotations.csv``."""
    csv_name = "annotations.csv"
    for candidate in (
        os.path.join("training", "annotations.csv"),
        "annotations.csv",
    ):
        if os.path.isfile(os.path.join(root, candidate)):
            csv_name = candidate
            break
    return convert_bbox_csv(root, output_dir, csv_name, "VinDr-Mammo", seed=seed)


def convert_embed(root: str, output_dir: str, seed: int = 42) -> Dict[str, object]:
    """EMBED: DICOM images plus a metadata CSV with bounding boxes."""
    for candidate in ("metadata.csv", "annotations.csv", "embed.csv"):
        if glob.glob(os.path.join(root, "**", candidate), recursive=True):
            return convert_bbox_csv(root, output_dir, candidate, "EMBED", seed=seed)
    raise FileNotFoundError(
        "EMBED metadata CSV not found (expected metadata.csv / annotations.csv)."
    )


CONVERTERS: Dict[str, Callable[..., Dict[str, object]]] = {
    "mias": convert_mias,
    "cbis-ddsm": convert_cbis_ddsm,
    "inbreast": convert_inbreast,
    "vindr-mammo": convert_vindr_mammo,
    "embed": convert_embed,
}


def convert(dataset: str, root: str, output_dir: str, seed: int = 42) -> Dict[str, object]:
    key = dataset.strip().lower()
    if key not in CONVERTERS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {', '.join(sorted(CONVERTERS))}"
        )
    return CONVERTERS[key](root, output_dir, seed=seed)

"""Dataset conversion: unified layout, class mapping, patient-level splits."""
import csv
import os

import cv2
import numpy as np
import pytest


@pytest.fixture()
def mias_like(tmp_path):
    """Minimal MIAS structure: PGM images + fixed-width Info.txt."""
    root = tmp_path / "MIAS"
    root.mkdir()
    lines = []
    rng = np.random.default_rng(0)
    for patient in range(6):
        for side in ("L", "R"):
            ref = f"mdb{patient:03d}_{side}"
            image = (rng.random((128, 128)) * 60 + 20).astype(np.uint8)
            cv2.circle(image, (64, 64), 40, 180, -1)
            cv2.imwrite(str(root / f"{ref}.pgm"), image)
            if patient % 2 == 0:
                abnormality, radius = "CIRC", 30
            else:
                abnormality, radius = "CALC", 10
            # MIAS Info.txt layout: reference, tissue class, abnormality class,
            # severity, then the annotation circle (x, y, radius).
            line = (
                f"{ref:<8} "
                f"{'G' if patient % 2 else 'F':<2} "
                f"{abnormality:<6} "
                f"{'B' if patient % 3 else 'M':<2} "
                f"{64:>6} {64:>6} {radius:>6}"
            )
            lines.append(line)
    (root / "Info.txt").write_text("\n".join(lines) + "\n", encoding="latin-1")
    return str(root)


def test_mias_conversion_layout_and_classes(mias_like, tmp_path):
    from training.converters import convert

    out = tmp_path / "out_mias"
    summary = convert("mias", mias_like, str(out))
    assert summary["files"] == 12
    assert os.path.isfile(out / "dataset.yaml")
    assert os.path.isfile(out / "manifest.csv")

    total_boxes = 0
    classes = set()
    for split in ("train", "val", "test"):
        for name in os.listdir(out / "labels" / split):
            with open(out / "labels" / split / name) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) == 5:
                        classes.add(int(parts[0]))
                        total_boxes += 1
                        cx, cy, w, h = (float(v) for v in parts[1:])
                        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0
    assert total_boxes > 0
    assert classes <= {0, 1, 2}
    assert 0 in classes and 1 in classes, "both masses and calcifications expected"


def test_patient_level_split_has_no_leakage(mias_like, tmp_path):
    """Images of one patient must never appear in two different splits."""
    from training.converters import convert

    out = tmp_path / "out_split"
    convert("mias", mias_like, str(out))
    patient_splits = {}
    with open(out / "manifest.csv") as fh:
        for row in csv.DictReader(fh):
            patient_splits.setdefault(row["patient_id"], set()).add(row["split"])
    for patient, splits in patient_splits.items():
        assert len(splits) == 1, f"patient {patient} leaked across {splits}"
    assert {s for splits in patient_splits.values() for s in splits} <= {"train", "val", "test"}


def test_split_is_deterministic(mias_like, tmp_path):
    from training.converters import convert

    first = tmp_path / "a"
    second = tmp_path / "b"
    convert("mias", mias_like, str(first), seed=7)
    convert("mias", mias_like, str(second), seed=7)
    with open(first / "manifest.csv") as fh:
        rows_a = [(r["file"], r["split"]) for r in csv.DictReader(fh)]
    with open(second / "manifest.csv") as fh:
        rows_b = [(r["file"], r["split"]) for r in csv.DictReader(fh)]
    assert rows_a == rows_b


def test_bbox_csv_conversion(tmp_path):
    from training.converters import convert_bbox_csv

    root = tmp_path / "vindr"
    images = root / "images"
    images.mkdir(parents=True)
    for index in range(4):
        image = np.zeros((200, 150), np.uint8)
        cv2.rectangle(image, (40, 50), (90, 110), 200, -1)
        cv2.imwrite(str(images / f"img{index}.png"), image)
    with open(root / "annotations.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "patient_id", "xmin", "ymin", "xmax", "ymax", "finding_categories"])
        writer.writerow(["img0", "P1", 40, 50, 90, 110, "Mass"])
        writer.writerow(["img1", "P2", 40, 50, 90, 110, "Suspicious Calcification"])
        writer.writerow(["img2", "P3", 40, 50, 90, 110, "Benign Calcification"])
        writer.writerow(["img3", "P4", "", "", "", "", "No Finding"])

    out = tmp_path / "out_csv"
    summary = convert_bbox_csv(str(root), str(out), "annotations.csv", "VinDr-Mammo")
    assert summary["files"] == 4

    found = {}
    for split in ("train", "val", "test"):
        for name in os.listdir(out / "labels" / split):
            with open(out / "labels" / split / name) as fh:
                found[name] = [line.split()[0] for line in fh if line.strip()]
    assert any("0" in values for values in found.values()), "mass class missing"
    assert any("1" in values for values in found.values()), "microcalcification class missing"
    assert any("2" in values for values in found.values()), "macrocalcification class missing"
    assert any(values == [] for values in found.values()), "negative image lost"


def test_label_to_class_mapping():
    from training.converters import label_to_class

    assert label_to_class("Mass") == 0
    assert label_to_class("Suspicious Calcification") == 1
    assert label_to_class("Benign Calcification") == 2
    assert label_to_class("Calcification") == 1
    assert label_to_class("Lymph node") == 0  # unknown labels fall back to mass


def test_invalid_dataset_name_rejected(tmp_path):
    from training.converters import convert

    with pytest.raises(ValueError):
        convert("not-a-dataset", str(tmp_path), str(tmp_path / "x"))


def test_missing_dataset_root_is_a_clear_error(tmp_path):
    from training.converters import convert

    with pytest.raises(FileNotFoundError):
        convert("mias", str(tmp_path / "nope"), str(tmp_path / "o"))


def test_normalisation_clamps_out_of_range_boxes():
    from training.converters import Annotation

    box = Annotation(x1=-20, y1=-20, x2=1500, y2=1200, class_id=0)
    line = box.normalized(1000, 900)
    assert line is not None
    parts = [float(v) for v in line.split()]
    assert 0.0 <= parts[1] <= 1.0 and 0.0 <= parts[2] <= 1.0


def test_degenerate_box_is_dropped():
    from training.converters import Annotation

    assert Annotation(x1=10, y1=10, x2=10.5, y2=10.5).normalized(100, 100) is None


def test_fixed_width_info_is_also_supported(tmp_path):
    """Some MIAS mirrors strip the column separators; parsing must still work."""
    from training.converters import convert

    root = tmp_path / "MIAS2"
    root.mkdir()
    image = np.zeros((128, 128), np.uint8)
    cv2.circle(image, (64, 64), 30, 200, -1)
    cv2.imwrite(str(root / "mdb100_L.pgm"), image)
    (root / "Info.txt").write_text("mdb100_LFCALC B    64    64    30\n", encoding="latin-1")
    summary = convert("mias", str(root), str(tmp_path / "out_fixed"))
    assert summary["files"] == 1

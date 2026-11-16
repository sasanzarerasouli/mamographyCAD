"""CLI: evaluate a trained detector and report clinically meaningful metrics.

Reported metrics
----------------
* mAP@0.5, mAP@0.5:0.95, precision, recall, F1 (ultralytics validation)
* false positives *per image* at a fixed operating threshold
* per-class results

The FP/image value is computed independently of ultralytics: predictions are
matched to ground truth at IoU 0.5 and every unmatched prediction counts as a
false positive. That is the metric CAD readers care about, and it is not part of
the standard mAP report.

Usage
-----
    python -m training.evaluate --weights model/mass/mass_yolo_best.pt \
        --data datasets/cbis/dataset.yaml --split val --conf 0.25
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("MammographyCAD.evaluate")


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(label_path: str, width: int, height: int) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    boxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(value) for value in parts[1:5])
            x1 = (cx - w / 2) * width
            x2 = (cx + w / 2) * width
            y1 = (cy - h / 2) * height
            y2 = (cy + h / 2) * height
            boxes.append((class_id, (x1, y1, x2, y2)))
    return boxes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="", help="metrics JSON output path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        import yaml
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error(
            "evaluation requires ultralytics and pyyaml:\n"
            "    pip install ultralytics pyyaml"
        )
        return 2

    with open(args.data, "r", encoding="utf-8") as fh:
        data_cfg = yaml.safe_load(fh)
    root = data_cfg.get("path", os.path.dirname(os.path.abspath(args.data)))
    split_rel = data_cfg.get(args.split, f"images/{args.split}")
    image_dir = split_rel if os.path.isabs(split_rel) else os.path.join(root, split_rel)
    label_dir = os.path.join(root, "labels", args.split)

    model = YOLO(args.weights)

    # --- standard validation metrics ---------------------------------- #
    val_metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )
    summary: Dict[str, float] = {}
    try:
        summary["mAP50"] = float(val_metrics.box.map50)
        summary["mAP50_95"] = float(val_metrics.box.map)
        summary["precision"] = float(val_metrics.box.mp)
        summary["recall"] = float(val_metrics.box.mr)
        if summary["precision"] + summary["recall"] > 0:
            summary["f1"] = (
                2 * summary["precision"] * summary["recall"]
                / (summary["precision"] + summary["recall"])
            )
    except AttributeError:  # pragma: no cover - API drift
        logger.warning("Could not read standard metrics from ultralytics")

    # --- false positives per image ------------------------------------ #
    images = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        images.extend(glob.glob(os.path.join(image_dir, pattern)))
    images.sort()

    total_fp = 0
    total_tp = 0
    total_fn = 0
    total_gt = 0
    for image_path in images:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        height, width = image.shape[:2]
        stem = os.path.splitext(os.path.basename(image_path))[0]
        ground_truth = load_ground_truth(
            os.path.join(label_dir, f"{stem}.txt"), width, height
        )
        total_gt += len(ground_truth)

        prediction = model.predict(
            source=image_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        predicted: List[Tuple[int, Tuple[float, float, float, float]]] = []
        for result in prediction or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for (x1, y1, x2, y2), class_id in zip(
                boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)
            ):
                predicted.append((int(class_id), (float(x1), float(y1), float(x2), float(y2))))

        matched_gt = set()
        for class_id, box in predicted:
            best_index, best_iou = -1, 0.5
            for index, (gt_class, gt_box) in enumerate(ground_truth):
                if index in matched_gt or gt_class != class_id:
                    continue
                iou = _iou(box, gt_box)
                if iou > best_iou:
                    best_index, best_iou = index, iou
            if best_index >= 0:
                matched_gt.add(best_index)
                total_tp += 1
            else:
                total_fp += 1
        total_fn += len(ground_truth) - len(matched_gt)

    n_images = max(1, len(images))
    summary.update(
        {
            "images": float(len(images)),
            "ground_truth_boxes": float(total_gt),
            "true_positives": float(total_tp),
            "false_positives": float(total_fp),
            "false_negatives": float(total_fn),
            "false_positives_per_image": round(total_fp / n_images, 4),
        }
    )
    if total_tp + total_fp > 0:
        summary["point_precision"] = round(total_tp / (total_tp + total_fp), 4)
    if total_tp + total_fn > 0:
        summary["point_recall"] = round(total_tp / (total_tp + total_fn), 4)

    payload = {
        "weights": os.path.abspath(args.weights),
        "data": os.path.abspath(args.data),
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "metrics": summary,
    }

    print(json.dumps(payload, indent=2))
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.weights)), f"eval_{args.split}.json"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nMetrics written to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

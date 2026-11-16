"""CLI: train a YOLO / RT-DETR detector for mammography CAD.

The script never overwrites an existing checkpoint: a new run always writes to a
fresh directory, and copying the finished weights into ``model/<task>/`` is a
separate, explicit step (``--install``).

Examples
--------
    # mass detector, CPU, from a COCO-pretrained YOLOv8n
    python -m training.train_detector --data datasets/cbis/dataset.yaml \
        --task mass --model yolov8n.pt --epochs 50 --imgsz 1024 --device cpu

    # calcification detector with high-resolution tiled training
    python -m training.train_detector --data datasets/combined/dataset.yaml \
        --task calcification --model yolov8s.pt --epochs 80 --imgsz 1280

    # RT-DETR variant
    python -m training.train_detector --data ... --task mass --model rtdetr-l.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

from config import Config

logger = logging.getLogger("MammographyCAD.train")

TASK_CLASSES = {
    # only the clinically relevant class(es) are trained per task
    "mass": [0],
    "calcification": [1, 2],
    "both": [0, 1, 2],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset.yaml produced by prepare_dataset")
    parser.add_argument("--task", default="both", choices=sorted(TASK_CLASSES))
    parser.add_argument("--model", default="yolov8n.pt", help="pretrained checkpoint or model name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=Config.DL_DEVICE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="")
    parser.add_argument("--project", default=os.path.join(Config.BASE_DIR, "runs"))
    parser.add_argument("--install", action="store_true",
                        help="copy best.pt into model/<task>/ after training")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error(
            "ultralytics is required for training. Install with:\n"
            "    pip install ultralytics torch torchvision"
        )
        return 2

    name = args.name or f"{args.task}_{os.path.basename(os.path.dirname(args.data))}"
    model = YOLO(args.model)

    logger.info(
        "Training %s detector: model=%s imgsz=%d epochs=%d device=%s",
        args.task, args.model, args.imgsz, args.epochs, args.device,
    )
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        name=name,
        exist_ok=False,
        pretrained=True,
        val=True,
        plots=True,
        resume=args.resume,
    )

    save_dir = str(getattr(results, "save_dir", os.path.join(args.project, name)))
    best = os.path.join(save_dir, "weights", "best.pt")
    metrics_path = os.path.join(save_dir, "final_metrics.json")

    metrics = {}
    try:
        metrics = dict(getattr(results, "results_dict", {}) or {})
    except Exception:  # pragma: no cover
        metrics = {}
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "task": args.task,
                "model": args.model,
                "data": args.data,
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "device": args.device,
                "best_weights": best,
                "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            },
            fh,
            indent=2,
        )

    print(f"\nTraining finished. Best weights: {best}")
    print(f"Metrics: {metrics_path}")

    if args.install:
        target_dir = {
            "mass": Config.MASS_MODEL_DIR,
            "calcification": Config.CALC_MODEL_DIR,
            "both": Config.MODEL_DIR,
        }[args.task]
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, f"{args.task}_{os.path.basename(save_dir)}_best.pt")
        if os.path.exists(target):
            logger.error("Refusing to overwrite existing checkpoint: %s", target)
            return 3
        shutil.copy2(best, target)
        print(f"Installed checkpoint: {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

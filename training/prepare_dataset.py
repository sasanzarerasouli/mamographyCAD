"""CLI: convert public mammography datasets into the unified YOLO layout.

Examples
--------
    python -m training.prepare_dataset --dataset mias \
        --root data/MIAS --out datasets/mias

    python -m training.prepare_dataset --dataset cbis-ddsm \
        --root data/CBIS-DDSM --out datasets/cbis

Splits are patient level; ``manifest.csv`` records the mapping so a split can
be audited (and reused) later.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .converters import CONVERTERS, convert

logger = logging.getLogger("MammographyCAD.prepare")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(CONVERTERS))
    parser.add_argument("--root", required=True, help="dataset root directory")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="patient split seed")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        summary = convert(args.dataset, args.root, args.out, seed=args.seed)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Conversion failed: %s", exc)
        return 2

    print(json.dumps(summary, indent=2))
    print(
        f"\nDataset written to {summary['output_dir']}\n"
        f"Next: python -m training.train_detector --data "
        f"{summary['output_dir']}/dataset.yaml --task mass --name mass_yolo"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

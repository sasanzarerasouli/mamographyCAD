"""Inspect every checkpoint the application can see and report what it can do.

    python -m tools.inspect_models                 # human readable
    python -m tools.inspect_models --json          # machine readable
    python -m tools.inspect_models --dir model/digital_eye

For each ``*.pt`` / ``*.pth`` / ``*.onnx`` / ``*.engine`` file the tool reports
the path, size, SHA-256, detected architecture, class names (read from the
checkpoint itself, never guessed), input requirements, and whether the file can
actually be used for the mass and/or calcification task.

The checkpoints are opened **read-only**; nothing is written or renamed.  When
torch is not installed only the file header is inspected, which is reported
explicitly so the output is never mistaken for a full inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# allow "python tools/inspect_models.py" as well as "python -m tools.inspect_models"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from src.models.deep_detector import backend_status  # noqa: E402
from src.models.model_registry import ModelRegistry, runtime_status  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        action="append",
        default=None,
        help="directory to scan (repeatable); defaults to every configured model directory",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)

    registry = ModelRegistry(directories=args.dir)
    models = registry.discover()

    payload = {
        "directories_scanned": list(registry.directories),
        "runtime": runtime_status(),
        "backends": backend_status(),
        "checkpoints": [model.as_dict() for model in models],
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print("Directories scanned:")
    for directory in payload["directories_scanned"]:
        state = "present" if os.path.isdir(directory) else "MISSING"
        print(f"  - {directory}  [{state}]")

    runtime = payload["runtime"]
    print("\nRuntime:")
    print(f"  torch        : {'available' if runtime.get('torch') else 'not installed'}")
    print(f"  ultralytics  : {'available' if runtime.get('ultralytics') else 'not installed'}")
    print(f"  onnxruntime  : {'available' if runtime.get('onnxruntime') else 'not installed'}")
    if runtime.get("torch_version"):
        print(f"  torch version: {runtime['torch_version']}")
    if runtime.get("ultralytics_version"):
        print(f"  ultralytics  : {runtime['ultralytics_version']}")

    print("\nBackends in use:")
    for task, status in (payload["backends"] or {}).items():
        if not isinstance(status, dict):
            continue
        print(
            f"  {task:<14}: {status.get('backend')}"
            + (f"  ({status.get('reason')})" if status.get("reason") else "")
        )

    print(f"\nCheckpoints: {len(models)}")
    if not models:
        print(
            "  none found - the application runs its classical (unvalidated) "
            "detectors and flags every result accordingly."
        )
        return 0

    for info in models:
        data = info.as_dict()
        print(f"\n  {data['path']}")
        print(f"    file            : {data['filename']} ({data['size_bytes'] / 1e6:.2f} MB)")
        print(f"    sha256          : {data['sha256']}")
        print(f"    format          : {data.get('file_format')}")
        print(f"    task            : {data.get('task') or 'not declared'}")
        print(f"    framework       : {data.get('framework') or 'not declared'}")
        print(f"    classes ({data.get('num_classes', 0)})       : {data.get('class_names') or 'not readable'}")
        print(f"    input size      : {data.get('input_size') or 'not declared'}")
        print(f"    inspection      : {data.get('inspect_method')}")
        print(f"    usable          : {data.get('usable')}  (routing: {data.get('task_routing')})")
        for issue in data.get("issues") or []:
            print(f"    issue           : {issue}")
        for note in data.get("notes") or []:
            print(f"    note            : {note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

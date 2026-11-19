"""Inspection and registry of the project's pretrained detection checkpoints.

The application ships *no* weights inside the repository (``model/`` is not part
of the Git tree).  This module is the single place that discovers checkpoint
files, reports what they actually contain, and decides which of them can be
used for which clinical task.

Design rules
------------
* **Read only.** Nothing in this module ever writes to, renames or re-saves a
  checkpoint.  A SHA-256 digest is computed so a model can be identified
  unambiguously in reports.
* **No guessing.** If the trained class names cannot be determined, the model
  is reported as ``unknown`` and it is *not* used for a clinical task.  A
  checkpoint is only routed to mass or calcification detection when its own
  class vocabulary says so.
* **Graceful degradation.** ``torch``/``ultralytics`` are optional.  Without
  them the registry still reports file facts (format, size, digest) and clearly
  states that deep inference is unavailable.

Typical layout::

    model/
      digital_eye/
        *.pt            <- the three legacy YOLO checkpoints
      mass/
        mass_yolo.pt
      calcification/
        calc_yolo.pt
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import Config

logger = logging.getLogger("MammographyCAD.models")

_TORCH_CLASSES = {"ultralytics", "torch", "yolo"}

# Strings that appear in ultralytics YOLO checkpoints; used only for the
# torch-free heuristic inspection path.
_TEXT_PATTERNS = {
    "task": re.compile(rb"(detect|segment|classify|pose|obb)\x00"),
    "imgsz": re.compile(rb"imgsz\x94[^\x00]{0,40}"),
}


@dataclass
class ModelInfo:
    """Facts about one checkpoint file."""

    path: str
    filename: str = ""
    size_bytes: int = 0
    sha256: str = ""
    modified: str = ""
    file_format: str = "unknown"          # ultralytics_yolo | torchscript | onnx | torch_pickle | unknown
    inspect_method: str = "none"          # torch | header | none
    task: str = ""                        # detect | segment | classify
    framework: str = ""
    framework_version: str = ""
    class_names: List[str] = field(default_factory=list)
    num_classes: int = 0
    input_size: Optional[int] = None      # native imgsz when discoverable
    task_routing: str = "unknown"         # mass | calcification | mixed | unknown
    usable: bool = False
    issues: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def summary(self) -> str:
        if not self.usable:
            reason = self.issues[0] if self.issues else "unusable"
            return f"{self.filename}: {self.file_format} ({reason})"
        return (
            f"{self.filename}: {self.task or 'detect'} / {self.num_classes} classes "
            f"[{', '.join(self.class_names[:6])}"
            f"{'…' if len(self.class_names) > 6 else ''}] -> {self.task_routing}"
        )


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_status() -> Dict[str, Any]:
    """Report which optional deep-learning runtimes are importable."""
    status: Dict[str, Any] = {
        "torch": False,
        "torch_version": "",
        "ultralytics": False,
        "ultralytics_version": "",
        "onnxruntime": False,
    }
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # noqa: F401

            status["torch"] = True
            status["torch_version"] = str(torch.__version__)
        except Exception:  # pragma: no cover - broken install
            pass
    if importlib.util.find_spec("ultralytics") is not None:
        try:
            import ultralytics  # noqa: F401

            status["ultralytics"] = True
            status["ultralytics_version"] = str(ultralytics.__version__)
        except Exception:  # pragma: no cover - broken install
            pass
    if importlib.util.find_spec("onnxruntime") is not None:
        status["onnxruntime"] = True
    return status


def deep_inference_available() -> bool:
    status = runtime_status()
    return bool(status["ultralytics"] and status["torch"])


class ModelRegistry:
    """Discover, inspect and route checkpoint files."""

    WEIGHT_EXTENSIONS = (".pt", ".pth", ".onnx", ".engine", ".torchscript")

    def __init__(self, directories: Optional[Iterable[str]] = None):
        if directories is None:
            directories = [
                Config.DIGITAL_EYE_DIR,
                Config.MASS_MODEL_DIR,
                Config.CALC_MODEL_DIR,
                Config.MODEL_DIR,
            ]
        self.directories = [d for d in directories if d]
        self._cache: Dict[Tuple[str, float], ModelInfo] = {}

    # ------------------------------------------------------------------ #
    #  Discovery                                                         #
    # ------------------------------------------------------------------ #
    def discover(self) -> List[ModelInfo]:
        """Inspect every checkpoint found under the configured directories."""
        infos: List[ModelInfo] = []
        seen: set = set()
        for directory in self.directories:
            if not os.path.isdir(directory):
                continue
            for root, _dirs, files in os.walk(directory):
                for name in sorted(files):
                    if not name.lower().endswith(self.WEIGHT_EXTENSIONS):
                        continue
                    path = os.path.join(root, name)
                    real = os.path.realpath(path)
                    if real in seen:
                        continue
                    seen.add(real)
                    infos.append(self.inspect(path))
        return infos

    def inspect(self, path: str) -> ModelInfo:
        """Inspect a single checkpoint (cached on path + mtime)."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        key = (path, mtime)
        if key in self._cache:
            return self._cache[key]

        info = ModelInfo(path=path, filename=os.path.basename(path))
        try:
            info.size_bytes = os.path.getsize(path)
            info.modified = _iso(mtime)
            info.sha256 = _sha256(path)
        except OSError as exc:
            info.issues.append(f"cannot read file: {exc}")
            self._cache[key] = info
            return info

        info.file_format = self._detect_format(path)
        with_torch = Config.TRUST_LOCAL_CHECKPOINTS and runtime_status()["torch"]

        if info.file_format in ("ultralytics_yolo", "torchscript", "torch_pickle"):
            if with_torch:
                self._inspect_with_torch(path, info)
            if not info.class_names:
                self._inspect_header(path, info)
        elif info.file_format == "onnx":
            info.inspect_method = "header"
            info.notes.append(
                "ONNX graph is not introspected; metadata (class names, imgsz) "
                "must be supplied by the user to route this model."
            )

        self._route(info)
        self._cache[key] = info
        return info

    # ------------------------------------------------------------------ #
    #  Format detection                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_format(path: str) -> str:
        lower = path.lower()
        if lower.endswith(".onnx"):
            return "onnx"
        if lower.endswith((".engine", ".torchscript")):
            return "torchscript"
        if lower.endswith((".pt", ".pth")):
            if zipfile.is_zipfile(path):
                try:
                    with zipfile.ZipFile(path) as zf:
                        names = zf.namelist()
                    joined = " ".join(names)
                    if "data.pkl" in joined and any(
                        n.startswith("archive") for n in names
                    ):
                        return "torch_pickle"
                except zipfile.BadZipFile:  # pragma: no cover
                    pass
            # legacy (non-zip) torch.save -> still a pickle stream
            try:
                with open(path, "rb") as fh:
                    head = fh.read(4)
                if head[:2] == b"\x80\x02" or head[:1] == b"\x80":
                    return "torch_pickle"
            except OSError:  # pragma: no cover
                pass
            return "torch_pickle"
        return "unknown"

    # ------------------------------------------------------------------ #
    #  Inspection                                                        #
    # ------------------------------------------------------------------ #
    def _inspect_with_torch(self, path: str, info: ModelInfo) -> None:
        """Load the checkpoint metadata with torch (weights are not modified)."""
        try:
            import torch
        except Exception:  # pragma: no cover
            return

        checkpoint = None
        try:
            try:
                checkpoint = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            except TypeError:  # torch < 1.13 has no weights_only
                checkpoint = torch.load(path, map_location="cpu")
            info.inspect_method = "torch"
        except Exception as exc:
            info.issues.append(f"torch.load failed: {type(exc).__name__}: {exc}")
            return

        names: List[str] = []
        try:
            model = None
            if isinstance(checkpoint, dict):
                model = checkpoint.get("model") or checkpoint.get("ema")
                train_args = checkpoint.get("train_args") or checkpoint.get("args") or {}
                if isinstance(train_args, dict):
                    imgsz = train_args.get("imgsz")
                    if isinstance(imgsz, (list, tuple)) and imgsz:
                        imgsz = imgsz[0]
                    if isinstance(imgsz, (int, float)) and imgsz > 0:
                        info.input_size = int(imgsz)
                    info.task = str(train_args.get("task", info.task) or info.task)
                    version = train_args.get("version")
                    if version:
                        info.framework = "ultralytics"
                        info.framework_version = str(version)
                for key in ("task", "version"):
                    if key in checkpoint and not info.task:
                        info.task = str(checkpoint[key])
            elif checkpoint is not None:
                model = checkpoint

            if model is not None and getattr(model, "names", None):
                raw = model.names
                if isinstance(raw, dict):
                    names = [str(raw[k]) for k in sorted(raw, key=lambda x: int(x))]
                else:
                    names = [str(n) for n in raw]
                info.framework = info.framework or "ultralytics"
                model_task = getattr(model, "task", "")
                if model_task:
                    info.task = str(model_task)
                yaml = getattr(model, "yaml", None)
                if isinstance(yaml, dict):
                    imgsz = yaml.get("imgsz")
                    if isinstance(imgsz, int):
                        info.input_size = info.input_size or imgsz
                    if not info.task and yaml.get("task"):
                        info.task = str(yaml["task"])
            if not names and getattr(checkpoint, "names", None):
                raw = checkpoint.names
                names = [str(raw[k]) for k in sorted(raw, key=lambda x: int(x))] \
                    if isinstance(raw, dict) else [str(n) for n in raw]
        except Exception as exc:  # pragma: no cover - exotic checkpoints
            info.issues.append(f"metadata parse error: {exc}")

        if names:
            info.class_names = names
            info.num_classes = len(names)
        else:
            info.issues.append(
                "checkpoint loaded but no class names were found; "
                "cannot route this model safely"
            )

    def _inspect_header(self, path: str, info: ModelInfo) -> None:
        """Torch-free heuristic inspection of the checkpoint header.

        This is best effort: it looks for the class-name list that ultralytics
        stores in the pickle stream.  Results are explicitly marked as
        heuristic so they are never silently trusted for clinical routing.
        """
        info.inspect_method = "header"
        info.notes.append(
            "class names recovered heuristically from the checkpoint header; "
            "install torch for authoritative metadata"
        )
        try:
            with open(path, "rb") as fh:
                blob = fh.read(min(info.size_bytes, 64 << 20))
        except OSError as exc:
            info.issues.append(f"cannot read header: {exc}")
            return

        for key, pattern in _TEXT_PATTERNS.items():
            if key == "task":
                match = pattern.search(blob)
                if match and not info.task:
                    info.task = match.group(1).decode("ascii", "ignore")

        # ultralytics stores names as a dict of int -> str inside the pickle
        candidates = re.findall(rb"\{0: ['\"]([^'\"]{1,60})['\"]", blob)
        if not candidates:
            candidates = re.findall(
                rb"\(\d{1,3},\s*'([A-Za-z][A-Za-z0-9 _\-\./]{1,50})'\)", blob
            )
        if candidates:
            seen: List[str] = []
            for cand in candidates:
                text = cand.decode("utf-8", "ignore").strip()
                if text and text not in seen and len(text) < 60:
                    seen.append(text)
            if 1 <= len(seen) <= 50:
                info.class_names = seen
                info.num_classes = len(seen)

    # ------------------------------------------------------------------ #
    #  Routing                                                           #
    # ------------------------------------------------------------------ #
    def _route(self, info: ModelInfo) -> None:
        """Decide which clinical task (if any) a checkpoint can serve."""
        if not info.class_names:
            info.task_routing = "unknown"
            info.usable = False
            if not info.issues:
                info.issues.append("class names unknown")
            return

        mass_hits = [
            n for n in info.class_names
            if any(tok in n.lower() for tok in Config.MASS_CLASS_TOKENS)
        ]
        calc_hits = [
            n for n in info.class_names
            if any(tok in n.lower() for tok in Config.CALC_CLASS_TOKENS)
        ]

        unknown = [
            n for n in info.class_names
            if n not in mass_hits and n not in calc_hits
        ]
        if unknown:
            info.notes.append(
                "classes not mapped to a task: " + ", ".join(sorted(set(unknown))[:10])
            )

        if mass_hits and calc_hits:
            info.task_routing = "mixed"
        elif mass_hits:
            info.task_routing = "mass"
        elif calc_hits:
            info.task_routing = "calcification"
        else:
            info.task_routing = "unknown"
            info.issues.append(
                "no class name matches a mass or calcification vocabulary"
            )

        info.usable = info.task_routing != "unknown" and not any(
            "cannot route" in issue for issue in info.issues
        )

    # ------------------------------------------------------------------ #
    #  Selection                                                         #
    # ------------------------------------------------------------------ #
    def select_for(self, task: str) -> Optional[ModelInfo]:
        """Return the best checkpoint for ``'mass'`` or ``'calcification'``.

        Explicit paths from the environment / Config win, then task-specific
        directories, then ``model/digital_eye``.
        """
        task = task.lower()
        explicit = {
            "mass": Config.MASS_MODEL_PATH,
            "calcification": Config.CALC_MODEL_PATH,
        }.get(task, "")
        if explicit:
            if os.path.isfile(explicit):
                info = self.inspect(explicit)
                if info.class_names:
                    return info
                logger.warning(
                    "Explicit %s model %s has unknown classes; ignoring",
                    task,
                    explicit,
                )
            else:
                logger.warning("Configured %s model not found: %s", task, explicit)

        infos = [i for i in self.discover() if i.usable]
        ranked = []
        for info in infos:
            if info.task_routing not in (task, "mixed"):
                continue
            priority = 0
            parent = os.path.dirname(os.path.abspath(info.path))
            if task == "mass" and parent == os.path.abspath(Config.MASS_MODEL_DIR):
                priority = 3
            elif task == "calcification" and parent == os.path.abspath(
                Config.CALC_MODEL_DIR
            ):
                priority = 3
            elif parent == os.path.abspath(Config.DIGITAL_EYE_DIR):
                priority = 2
            else:
                priority = 1
            # Mixed models are usable for both tasks but ranked lower than a
            # dedicated checkpoint.
            if info.task_routing == "mixed":
                priority -= 1
            ranked.append((priority, info.size_bytes, info))
        if not ranked:
            return None
        ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return ranked[0][2]

    # ------------------------------------------------------------------ #
    #  Reporting                                                         #
    # ------------------------------------------------------------------ #
    def report(self) -> Dict[str, Any]:
        """Full audit of checkpoint assets for the UI/report."""
        infos = self.discover()
        status = runtime_status()
        selected = {
            "mass": (self.select_for("mass") or None),
            "calcification": (self.select_for("calcification") or None),
        }
        return {
            "runtime": status,
            "deep_inference_available": deep_inference_available(),
            "directories": self.directories,
            "models": [i.as_dict() for i in infos],
            "selected": {
                task: (info.as_dict() if info else None)
                for task, info in selected.items()
            },
            "notes": [
                "Checkpoints are opened read-only; no weight file is modified.",
                "A model is only routed when its own class vocabulary names the task.",
            ],
        }


def _iso(timestamp: float) -> str:
    try:
        import datetime

        return datetime.datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")
    except Exception:  # pragma: no cover
        return ""


_default_registry: Optional[ModelRegistry] = None


def default_registry() -> ModelRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry

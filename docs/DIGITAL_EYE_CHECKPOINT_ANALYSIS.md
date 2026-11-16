# Digital Eye checkpoints — analysis status

## What was requested

Inspect the three pretrained YOLO checkpoints expected in `model/digital_eye/`
and report their architecture, class vocabulary, input requirements,
compatibility with this code base, and their real capabilities; reuse them for
mass detection **only** if justified; never assume calcification support; never
overwrite the weights.

## What was actually found

```
$ ls model/digital_eye
ls: cannot access 'model/digital_eye': No such file or directory
```

The cloned repository contains application source only: `main.py`, `config.py`,
`src/**`, a one-line `README.md`, `LICENSE` and one screenshot. There is **no
`model/` directory, no `*.pt` / `*.pth` / `*.onnx` / `*.engine` file and no
dataset** anywhere in the tree.

Consequences, stated without hedging:

* The three checkpoints **could not be analysed**. No architecture, class list,
  input size or capability is reported for them, because inventing those values
  would be fabrication.
* **No checkpoint was reused.** `python -m tools.inspect_models` reports
  `Checkpoints: 0` and the application runs its classical detectors, flagged
  `backend: classical` in the GUI and `unvalidated_classical_fallback` in every
  exported report.
* No weight file was created, renamed, moved or overwritten: the inspection
  path opens files read-only and only ever reads bytes to compute a SHA-256.

## How to run the inspection on the machine that has the weights

```bash
# copy the three checkpoints into model/digital_eye/ first (read-only for the app)
python -m tools.inspect_models                       # human readable
python -m tools.inspect_models --json > models.json  # for the thesis appendix
```

For every file the tool prints the path, size, SHA-256, detected format, task,
framework, class names read from the checkpoint, declared input size, inspection
method (`torch` = authoritative, `header` = heuristic because torch is not
installed), usability and any issues. With torch installed the same metadata is
read through `torch.load(..., weights_only=False)` on a **copy-free read-only**
path; without torch only the zip/pickle header is parsed and the output says so.

## Decision rules implemented for the digital-eye checkpoints

Encoded in `src/models/model_registry.py` + `src/models/deep_detector.py`:

1. `ModelRegistry.discover()` scans `model/mass`, `model/calcification`,
   `model/digital_eye`, `model` (and any `CAD_MODEL_DIR`), computes SHA-256 and
   inspects each file.
2. `select_for("mass")` / `select_for("calcification")` only accept a
   checkpoint whose class names, **as stored in the file**, contain a matching
   class (e.g. `mass`, `tumor`, `lesion` for mass; `calc*` for calcification).
   A generic `person/car/...` COCO vocabulary matches nothing, so such a
   checkpoint is excluded instead of producing nonsense detections.
3. A checkpoint that declares no class names is reported `usable: false` with
   the issue `class names unknown` — the vocabulary is never guessed.
4. If both tasks resolve to no usable checkpoint, the classical detectors run
   and the backend is reported as classical/unvalidated.

With that rule set, the only scenario in which a digital-eye checkpoint is used
for masses is the one the thesis allows: the checkpoint's own metadata names a
mass/tumour class. Anything else (a COCO YOLO used as a "detector") is refused.

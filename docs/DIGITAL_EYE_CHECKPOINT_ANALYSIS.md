# Digital Eye checkpoint inventory

At the pinned source revision `1a146a1218b7302a53de8178366cbe9629dd289a`, all
three files existed under `model/digital_eye`. This supersedes the earlier
source-only inventory in this document. No weights were modified.

| File | Bytes (decimal MB) | SHA-256 |
|---|---:|---|
| `yolo11_n.pt` | 5.44 MB | `4c6ed69c85775bfe7642a428489fc0a90f53cb3ac7b164eb7da7220ef9549a14` |
| `yolo11_s.pt` | 19.15 MB | `4d4ff958accc0c165a51cd6a5fd39962cefb05dca15c7302e81d790d60cfe758` |
| `yolo11_m.pt` | 40.48 MB | `f67ef4e01f90c159d995653064e756a466fa6302f727772d3b9774f19824960a` |

`python -m tools.inspect_models` identified Torch pickle archives but did not
extract a verified class vocabulary or input size. Filename labels do not prove
architecture, training data, licensing terms for weights, or suitability. Torch,
Ultralytics and ONNX Runtime were not installed in the test sandbox. No model
was deserialized or run, and no comparison display was enabled. No external
reference repository was used as application source.

The application inventories checkpoints without pickle deserialization by
default. `CAD_TRUST_LOCAL_CHECKPOINTS=1` opts into existing trusted-local
inspection paths; only use it for checkpoints whose origin and integrity you
have independently verified. It is not required for CPU classical processing.
The clinical-style mass pipeline never invokes a YOLO model.

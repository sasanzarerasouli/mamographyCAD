# Model directory

Place trained checkpoints here. **Nothing in this directory is ever written by
the application** — checkpoints are opened read-only and their SHA-256 is
reported in the GUI and in every exported report.

```
model/
├── digital_eye/          # the three pre-existing pretrained YOLO checkpoints
├── mass/                 # mass detector(s), e.g. mass_yolo_best.pt
└── calcification/        # calcification detector(s), e.g. calc_yolo_best.pt
```

Resolution order for a task:

1. the explicit environment override `CAD_MASS_MODEL` / `CAD_CALC_MODEL`;
2. `model/mass/*` resp. `model/calcification/*`;
3. `model/digital_eye/*` (only if the checkpoint's own class names contain a
   matching class — the class vocabulary is never guessed);
4. otherwise the classical detector runs and the result is flagged
   `backend: classical` with the warning that no validated model was used.

Check what the application sees at any time:

```bash
python -m tools.inspect_models
```

# Clinical mapping implemented in the software

Everything below is a **software-derived descriptor**, not a diagnosis. Each
value carries the evidence that produced it and the GUI/report labels the
assessment as a software estimate.

## BI-RADS (0–6)

| Category | Meaning in this software | How it is produced |
|---|---|---|
| 0 | Incomplete — additional imaging needed | technical quality inadequate, or a finding cannot be characterised |
| 1 | Negative | no candidate above threshold, dense area well below the C boundary |
| 2 | Benign | circumscribed, round/oval, low density, no suspicious calcifications |
| 3 | Probably benign | score 0.20–0.39: mostly benign morphology with one atypical feature |
| 4A/4B/4C | Suspicious | score 0.40–0.54 / 0.55–0.79 / ≥0.80, driven by margin+shape+density+calcification lexicon |
| 5 | Highly suggestive of malignancy | score ≥0.85 with a spiculated/irregular mass or pleomorphic+linear-branching calcifications |
| 6 | Known biopsy-proven malignancy | **only** when supplied as clinical context; never inferred from pixels |

Evidence model (weights in `src/models/birads_estimator.py`, documented for audit):

```
margin    spiculated .75 | obscured .55 | indistinct .50 | microlobulated .42 | circumscribed .12
shape     irregular .35 | lobulated .20 | oval .12 | round .05
density   high +.10 | equal +.04 | low +.02 | fat-containing 0
calc      amorphous/pleomorphic/linear-branching + distribution weight
score = clip(max(finding score) + 0.05 * (n_suspicious - 1), 0, 1)
```

Guards against inflated categories:

* a finding whose confidence is below `Config.BIRADS_MIN_FINDING_CONFIDENCE`
  (0.60) can only support categories ≤ 3 — this stops one weak classical false
  positive from producing BI-RADS 5;
* dense breasts (C/D) add a sensitivity warning but never silently downgrade;
* `validation_status` is `unvalidated_classical_fallback` unless a trained model
  produced the detections, and that string reaches the GUI and the report.

## ACR breast density (A–D)

Two-component Gaussian mixture on the breast pixels (fibroglandular vs fat);
the dense fraction is the mean posterior of the dense component, cleaned
morphologically. Otsu is kept only as a fallback when scikit-learn is missing.

| Dense area | Category |
|---|---|
| < 25 % | A — almost entirely fatty |
| 25–50 % | B — scattered fibroglandular densities |
| 50–75 % | C — heterogeneously dense |
| ≥ 75 % | D — extremely dense |

Reported as a **software estimate**; visual/volumetric ACR assessment by a
radiologist remains the reference standard. The method actually used
(`gmm2`/`otsu`) is reported next to the value.

## Calcification lexicon

Morphology: `amorphous`, `pleomorphic`, `punctate`, `linear`, `linear-branching`,
`coarse/popcorn`, `large rod-like`, `round`, `lucent-centred`, `dystrophic`,
`vascular`, `indeterminate`.
Distribution: `diffuse`, `regional`, `grouped/clustered`, `linear`,
`segmental`, `isolated`.
Cluster criterion: ≥ 3 calcifications within 10 mm (`CALC_CLUSTER_MIN_COUNT`,
`CALC_CLUSTER_RADIUS_MM`). Each descriptor carries a confidence and the note that
it is measured from image appearance.

## Measurements

mm values are only produced with real `PixelSpacing`. Without it the software
uses `Config.ASSUMED_PIXEL_SPACING_MM = 0.05` and flags the result
`spacing_assumed: true` with the basis string
`"ASSUMED pixel spacing 0.05 mm/px (no DICOM spacing available) - measurements are estimates"`,
so an estimate can never be mistaken for a measurement.

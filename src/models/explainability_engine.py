"""Human-readable explainable AI for lesion classifications.

This module generates per-lesion explainability outputs (feature scores + text)
to be consumed by GUI and PDF reports.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MammographyCAD.explainability")


class ExplainabilityEngine:
    """Generate feature scores and human-readable explanations for lesions."""

    # Any score above this is considered a notable/suspicious contributor
    THRESHOLD = 0.5

    # Default max values for normalization (can be overridden per call if needed)
    DEFAULT_NORMALIZATION = {
        "contrast": 80.0,
        "texture_entropy": 50.0,
        "shape_irregularity": 1.0,
        "spiculation_index": 1.0,
        "density_difference": 60.0,
    }

    def analyze(
        self,
        lesions: List[Dict[str, Any]],
        *,
        normalization: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build explainability output for a list of lesions.

        Args:
            lesions: list of lesions. Each lesion is expected to have:
                - an identifier: `id` (optional but recommended)
                - `features`: dict[str, float] with keys:
                    contrast, texture_entropy, shape_irregularity,
                    spiculation_index, density_difference
              If features are provided at top-level, they are also accepted.
            normalization: optional overrides for normalization max values.
            threshold: optional override for notable-score threshold.

        Returns:
            Structured output suitable for GUI and PDF.
        """
        norm = {**self.DEFAULT_NORMALIZATION, **(normalization or {})}
        thr = self.THRESHOLD if threshold is None else float(threshold)

        lesion_outputs: List[Dict[str, Any]] = []
        for idx, lesion in enumerate(lesions):
            lesion_id = lesion.get("id", idx)
            features: Dict[str, float] = lesion.get("features") or lesion.get("feature") or {}
            if not features:
                # Also accept features at top-level (backward tolerance)
                features = {
                    k: float(lesion.get(k, 0.0))
                    for k in (
                        "contrast",
                        "texture_entropy",
                        "shape_irregularity",
                        "spiculation_index",
                        "density_difference",
                    )
                }

            scores = self._compute_scores(features, norm)
            explanations = self._generate_explanations(scores, thr)
            summary = self._build_summary(explanations)

            lesion_outputs.append(
                {
                    "lesion_id": lesion_id,
                    "scores": scores,
                    "explanations": explanations,
                    "summary": summary,
                    # helpful for report tables/debugging
                    "features": {
                        "contrast": float(features.get("contrast", 0.0)),
                        "texture_entropy": float(features.get("texture_entropy", 0.0)),
                        "shape_irregularity": float(features.get("shape_irregularity", 0.0)),
                        "spiculation_index": float(features.get("spiculation_index", 0.0)),
                        "density_difference": float(features.get("density_difference", 0.0)),
                    },
                }
            )

        # a light-weight global section for GUI/PDF; keep it deterministic
        suspicious_count = sum(1 for o in lesion_outputs if o["explanations"])
        overall_summary = (
            f"{suspicious_count} of {len(lesion_outputs)} lesion(s) have notable suspicious features."
            if lesion_outputs
            else "No lesions provided."
        )

        return {
            "engine": "ExplainabilityEngine",
            "threshold": thr,
            "lesions": lesion_outputs,
            "summary": overall_summary,
            # explicit sections for consumers (GUI/PDF) to avoid guessing structure
            "gui": {
                "lesions": [
                    {
                        "lesion_id": o["lesion_id"],
                        "scores": o["scores"],
                        "summary": o["summary"],
                    }
                    for o in lesion_outputs
                ],
                "summary": overall_summary,
            },
            "pdf": {
                "sections": [
                    {
                        "title": f"Lesion {o['lesion_id']} – Explainability",
                        "summary": o["summary"],
                        "bullets": o["explanations"],
                        "scores": o["scores"],
                    }
                    for o in lesion_outputs
                ],
                "summary": overall_summary,
            },
        }

    # --- Backward-compatible single-lesion analysis (if your code calls analyze(features)) ---
    def analyze_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze a single lesion from a plain feature dict (legacy-friendly)."""
        out = self.analyze([{"id": 0, "features": features}])
        lesion0 = out["lesions"][0]
        return {
            "scores": lesion0["scores"],
            "explanations": lesion0["explanations"],
            "summary": lesion0["summary"],
        }

    def _compute_scores(self, features: Dict[str, float], norm: Dict[str, float]) -> Dict[str, float]:
        """Calculate the 5 required explainability scores for a lesion."""
        return {
            "contrast_score": self._normalize(float(features.get("contrast", 0.0)), norm["contrast"]),
            "texture_score": self._normalize(
                float(features.get("texture_entropy", 0.0)), norm["texture_entropy"]
            ),
            "shape_score": self._normalize(
                float(features.get("shape_irregularity", 0.0)), norm["shape_irregularity"]
            ),
            "edge_irregularity_score": self._normalize(
                float(features.get("spiculation_index", 0.0)), norm["spiculation_index"]
            ),
            "density_difference_score": self._normalize(
                float(features.get("density_difference", 0.0)), norm["density_difference"]
            ),
        }

    def _normalize(self, value: float, max_val: float) -> float:
        max_val = max(float(max_val), 1e-6)
        return round(min(1.0, max(0.0, value / max_val)), 3)

    def _generate_explanations(self, scores: Dict[str, float], threshold: float) -> List[str]:
        """Generate human-readable explanations based on which scores exceed threshold."""
        reasons: List[str] = []

        if scores.get("shape_score", 0.0) > threshold:
            reasons.append("Irregular shape")
        if scores.get("density_difference_score", 0.0) > threshold:
            reasons.append("Elevated local density")
        if scores.get("texture_score", 0.0) > threshold:
            reasons.append("High texture entropy")
        if scores.get("contrast_score", 0.0) > threshold:
            reasons.append("Strong contrast difference")
        if scores.get("edge_irregularity_score", 0.0) > threshold:
            reasons.append("Irregular / spiculated edge pattern")

        return reasons

    def _build_summary(self, explanations: List[str]) -> str:
        if not explanations:
            return "Region appears within normal morphological parameters."
        return "Region classified as suspicious because:\n• " + "\n• ".join(explanations)
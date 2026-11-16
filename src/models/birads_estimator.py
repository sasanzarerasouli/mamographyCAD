"""Rule-based, auditable BI-RADS (0-6) assessment estimation.

The estimator combines the outputs of the detection and characterisation
stages into a BI-RADS category **estimate**.  It is a decision-support tool:
every category is accompanied by the evidence that produced it, and the result
is explicitly labelled as a software-assisted estimate, never as a radiologist
diagnosis.

Category semantics implemented (ACR BI-RADS atlas)
--------------------------------------------------
0  Incomplete - additional imaging evaluation needed
1  Negative
2  Benign
3  Probably benign (<2% likelihood of malignancy)
4  Suspicious (4A >2-10%, 4B >10-50%, 4C >50-<95%)
5  Highly suggestive of malignancy (>=95%)
6  Known biopsy-proven malignancy (only when supplied as clinical context;
   the software never infers category 6 from image features)
"""
import logging
from typing import Any, Dict, Iterable, List, Optional

from config import Config

logger = logging.getLogger("MammographyCAD.birads")

CATEGORY_TEXT = {
    0: "BI-RADS 0: Incomplete - additional imaging evaluation required",
    1: "BI-RADS 1: Negative",
    2: "BI-RADS 2: Benign",
    3: "BI-RADS 3: Probably benign - short-interval follow-up suggested",
    4: "BI-RADS 4: Suspicious abnormality - tissue diagnosis should be considered",
    5: "BI-RADS 5: Highly suggestive of malignancy - appropriate action should be taken",
    6: "BI-RADS 6: Known biopsy-proven malignancy",
}

CATEGORY_MANAGEMENT = {
    0: "Additional imaging (e.g. extra views, ultrasound) is needed before a "
       "final assessment can be made.",
    1: "Routine screening.",
    2: "Routine screening.",
    3: "Short-interval follow-up (typically 6 months).",
    4: "Tissue diagnosis (biopsy) should be considered.",
    5: "Appropriate clinical action (biopsy / surgical consultation).",
    6: "Management according to the known proven malignancy.",
}

# Morphology / margin weights used by the evidence model (documented for audit).
MARGIN_WEIGHTS = {
    "spiculated": 0.75,
    "obscured": 0.55,
    "indistinct": 0.50,
    "microlobulated": 0.42,
    "circumscribed": 0.12,
    "not assessed": 0.20,
}
SHAPE_WEIGHTS = {
    "irregular": 0.35,
    "lobulated": 0.20,
    "oval": 0.12,
    "round": 0.05,
    "not assessed": 0.10,
}
DENSITY_WEIGHTS = {
    "high density": 0.10,
    "equal density": 0.04,
    "low density": 0.02,
    "fat-containing": 0.0,
    "not assessed": 0.03,
}


class BIRADSEstimator:
    """Transparent rule-based BI-RADS estimation based on morphology and risk."""

    #: Public mapping of every supported category to its atlas description.
    BIRADS_DESCRIPTIONS = CATEGORY_TEXT
    BIRADS_MANAGEMENT = CATEGORY_MANAGEMENT

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def estimate(
        self,
        detections: List,
        density_result: Optional[Dict] = None,
        risk_results: Optional[List[Dict]] = None,
        calcification_count: int = 0,
        *,
        calcification_analysis: Optional[Dict[str, Any]] = None,
        quality_report: Optional[Dict[str, Any]] = None,
        clinical_context: Optional[Dict[str, Any]] = None,
        detection_backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the BI-RADS estimate dictionary.

        Parameters
        ----------
        detections:
            All detections of the examination (mass and calcification).
        density_result:
            Output of :class:`BreastDensityClassifier` (optional).
        risk_results:
            Output of :class:`TumorRiskAnalyzer.analyze_lesion` (optional).
        calcification_count:
            Legacy argument: number of calcification candidates.
        calcification_analysis:
            Output of :class:`CalcificationAnalyzer` (morphology/distribution).
        quality_report:
            Output of :class:`MammogramQualityAnalyzer`; a technically
            inadequate image leads to category 0.
        clinical_context:
            Optional dictionary supplied by the user, e.g.
            ``{"known_biopsy_proven_malignancy": True}`` for category 6.
        """
        detections = list(detections or [])
        risk_results = list(risk_results or [])
        active = [d for d in detections if not getattr(d, "is_ignored", False)]
        masses = [d for d in active if self._is_mass(d)]
        calcs = [d for d in active if self._is_calcification(d)]

        evidence: List[Dict[str, Any]] = []

        # ---- 1. Technical adequacy (BI-RADS 0) --------------------------
        quality_score = None
        if quality_report:
            quality_score = quality_report.get("quality_score")
        inadequate = bool(
            quality_report
            and (
                (quality_score is not None and float(quality_score) < 0.35)
                or quality_report.get("analysis_limited")
            )
        )

        # ---- 2. Clinical context supplied by the user -------------------
        if clinical_context and clinical_context.get("known_biopsy_proven_malignancy"):
            result = self._result(
                category=6,
                confidence=0.99,
                reasons=[
                    "Known biopsy-proven malignancy was supplied as clinical context.",
                ],
                evidence=[
                    {
                        "source": "clinical_context",
                        "detail": "known_biopsy_proven_malignancy=True",
                    }
                ],
            )
            result["subcategory"] = None
            return result

        # ---- 3. Per-lesion evidence -------------------------------------
        mass_scores = [self._score_mass(d, risk_results) for d in masses]
        mass_scores = [s for s in mass_scores if s is not None]
        for score in mass_scores:
            evidence.append(score)

        calc_score = self._score_calcifications(
            calcs, calcification_analysis, calcification_count
        )
        if calc_score is not None:
            evidence.append(calc_score)

        # ---- 4. No findings --------------------------------------------
        if not evidence:
            if inadequate:
                return self._finish_inadequate(quality_report)
            return self._result(
                category=1,
                confidence=0.85,
                reasons=[
                    "No mass or calcification candidate was detected inside the "
                    "breast region.",
                    "Image quality was adequate for analysis."
                    if quality_report and not inadequate
                    else "No image quality restriction was reported.",
                ],
                evidence=[],
            )

        # ---- 5. Aggregate ----------------------------------------------
        # Findings below the confidence threshold cannot drive a suspicious
        # category on their own (they may only support categories 0-3).
        min_confidence = float(
            getattr(Config, "BIRADS_MIN_FINDING_CONFIDENCE", 0.60)
        )
        for entry in evidence:
            if entry.get("type") != "calcification" and float(
                entry.get("confidence", 1.0)
            ) < min_confidence:
                entry["low_confidence"] = True
                entry["score"] = min(float(entry["score"]), 0.35)

        best = max(evidence, key=lambda e: e["score"])
        final_score = float(best["score"])

        # Multiple suspicious findings raise the assessment slightly
        suspicious_findings = [e for e in evidence if e["score"] >= 0.45]
        if len(suspicious_findings) >= 2:
            final_score = min(1.0, final_score + 0.05)

        # Dense breasts reduce sensitivity -> never downgrade, only annotate
        density_category = self._density_category(density_result)
        density_note = None
        if density_category in ("C", "D"):
            density_note = (
                f"Breast density category {density_category} can obscure "
                "findings and lowers CAD sensitivity."
            )

        category = self._score_to_category(final_score)
        subcategory = self._subcategory(category, final_score)

        reasons = self._build_reasons(best, evidence, density_category)
        if density_note:
            reasons.append(density_note)
        if inadequate:
            reasons.append(
                "Technical quality limited the analysis, therefore the "
                "assessment is reported as incomplete (BI-RADS 0)."
            )

        confidence = self._confidence(category, best, evidence)

        if inadequate:
            result = self._finish_inadequate(quality_report, provisional={
                "category": category,
                "subcategory": subcategory,
                "score": round(final_score, 3),
            })
            result["provisional_reasons"] = reasons
            result["validation_status"] = (
                "validated_model" if (detection_backend or "").lower() == "deep"
                else "unvalidated_classical_fallback"
            )
            if result["validation_status"] != "validated_model":
                result["validation_warning"] = getattr(
                    Config, "BIRADS_UNVALIDATED_LABEL", ""
                )
            return result

        result = self._result(
            category=category,
            confidence=confidence,
            reasons=reasons,
            evidence=evidence,
        )
        result["subcategory"] = subcategory
        result["risk_score"] = round(final_score, 3)
        result["validation_status"] = (
            "validated_model" if (detection_backend or "").lower() == "deep"
            else "unvalidated_classical_fallback"
        )
        if result["validation_status"] != "validated_model":
            result["validation_warning"] = getattr(
                Config, "BIRADS_UNVALIDATED_LABEL", ""
            )
            result["reasons"].append(
                "No validated detection model was used; the category is a "
                "software estimate for research and decision support only."
            )
        result["density_category"] = density_category
        result["finding_counts"] = {
            "mass_candidates": len(masses),
            "calcification_candidates": len(calcs),
            "suspicious_findings": len(suspicious_findings),
        }
        return result

    # ------------------------------------------------------------------ #
    #  Scoring                                                           #
    # ------------------------------------------------------------------ #
    def _score_mass(
        self, detection, risk_results: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Evidence entry for one mass candidate."""
        features = getattr(detection, "features", {}) or {}
        characterization = features.get("mass_characterization") or {}
        margin = str(characterization.get("margin", "") or "").lower()
        shape = str(characterization.get("shape", "") or "").lower()
        density = str(characterization.get("density", "") or "").lower()

        # Fall back to the legacy attributes used by earlier versions
        if not margin:
            margin = str(features.get("margin", "") or "").lower()
        if not shape:
            shape = str(features.get("shape", "") or "").lower()

        margin_w = MARGIN_WEIGHTS.get(margin, MARGIN_WEIGHTS["not assessed"])
        shape_w = SHAPE_WEIGHTS.get(shape, SHAPE_WEIGHTS["not assessed"])
        density_w = DENSITY_WEIGHTS.get(density, DENSITY_WEIGHTS["not assessed"])

        score = margin_w + shape_w + density_w

        detection_type = getattr(getattr(detection, "type", None), "value", "") or ""
        if "confirmed" in detection_type:
            score += 0.08
        confidence = float(getattr(detection, "confidence", 0.0) or 0.0)
        score *= 0.75 + 0.25 * min(1.0, max(0.0, confidence))

        # Risk module agreement (0-100 -> 0-1)
        risk_score = self._risk_for(detection, risk_results)
        if risk_score is not None:
            score = 0.7 * score + 0.3 * risk_score
        score = float(min(1.0, max(0.0, score)))

        return {
            "type": "mass",
            "score": score,
            "margin": margin or "not assessed",
            "shape": shape or "not assessed",
            "density": density or "not assessed",
            "confidence": confidence,
            "risk_score": risk_score,
            "detection_type": detection_type or None,
            "components": {
                "margin": round(margin_w, 3),
                "shape": round(shape_w, 3),
                "density": round(density_w, 3),
            },
        }

    def _score_calcifications(
        self,
        calcs: List,
        analysis: Optional[Dict[str, Any]],
        calcification_count: int,
    ) -> Optional[Dict[str, Any]]:
        if not calcs and not analysis and not calcification_count:
            return None

        if analysis and analysis.get("available"):
            score = float(analysis.get("suspicion_score", 0.0) or 0.0)
            return {
                "type": "calcification",
                "score": min(1.0, score),
                "morphology": analysis.get("dominant_morphology"),
                "distribution": analysis.get("dominant_distribution"),
                "cluster_count": analysis.get("cluster_count", 0),
                "count": analysis.get("microcalcification_count", 0)
                + analysis.get("macrocalcification_count", 0),
                "components": {
                    "morphology": analysis.get("dominant_morphology"),
                    "distribution": analysis.get("dominant_distribution"),
                },
            }

        # Legacy path: only the number of candidates is known.
        count = max(len(calcs), int(calcification_count or 0))
        score = 0.15 if count < 3 else (0.30 if count < 6 else 0.42)
        return {
            "type": "calcification",
            "score": score,
            "morphology": None,
            "distribution": None,
            "count": count,
            "components": {"count": count},
        }

    @staticmethod
    def _risk_for(detection, risk_results: List[Dict]) -> Optional[float]:
        """Match a risk result to a detection by lesion id or position."""
        if not risk_results:
            return None
        lesion_id = getattr(detection, "lesion_id", "") or ""
        for index, result in enumerate(risk_results):
            if not isinstance(result, dict):
                continue
            rid = result.get("lesion_id") or result.get("id")
            if lesion_id and rid == lesion_id:
                return float(result.get("risk", {}).get("risk_score", 0)) / 100.0
            if rid is None and index == getattr(detection, "id", None):
                return float(result.get("risk", {}).get("risk_score", 0)) / 100.0
            risk_score = result.get("risk", {}).get("risk_score")
            if risk_score is not None and not lesion_id and len(risk_results) == 1:
                return float(risk_score) / 100.0
        return None

    # ------------------------------------------------------------------ #
    #  Category mapping                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _score_to_category(score: float) -> int:
        thresholds = getattr(Config, "BIRADS_THRESHOLDS", {})
        birads_2_max = float(thresholds.get("birads_2_max", 0.20))
        birads_3_max = float(thresholds.get("birads_3_max", 0.40))
        birads_4_max = float(thresholds.get("birads_4_max", 0.70))

        if score < 0.10:
            return 1
        if score < birads_2_max:
            return 2
        if score < birads_3_max:
            return 3
        if score < birads_4_max:
            return 4
        return 5

    @staticmethod
    def _subcategory(category: int, score: float) -> Optional[str]:
        if category != 4:
            return None
        if score < 0.55:
            return "4A"
        if score < 0.65:
            return "4B"
        return "4C"

    def _confidence(self, category: int, best: Dict[str, Any], evidence: List[Dict]) -> float:
        # Confidence grows when category is decisive and evidence is strong.
        base = 0.6
        if category in (1, 5):
            base = 0.75
        elif category in (2, 4):
            base = 0.65
        strength = min(1.0, float(best.get("score", 0.0)) + 0.2)
        agreement = 0.05 * max(0, len(evidence) - 1)
        return round(min(0.92, base * strength + agreement), 2)

    # ------------------------------------------------------------------ #
    #  Output helpers                                                    #
    # ------------------------------------------------------------------ #
    def _build_reasons(
        self,
        best: Dict[str, Any],
        evidence: List[Dict],
        density_category: Optional[str],
    ) -> List[str]:
        reasons: List[str] = []
        if best["type"] == "mass":
            reasons.append(
                "Dominant finding is a mass candidate with "
                f"shape={best.get('shape')}, margin={best.get('margin')}, "
                f"density={best.get('density')}."
            )
            if best.get("risk_score") is not None:
                reasons.append(
                    "Lesion risk module score: "
                    f"{float(best['risk_score']) * 100:.0f}/100."
                )
        else:
            reasons.append(
                "Dominant finding is calcification pattern with "
                f"morphology={best.get('morphology') or 'not assessed'}, "
                f"distribution={best.get('distribution') or 'not assessed'}."
            )
            if best.get("count"):
                reasons.append(
                    f"{best['count']} calcification candidate(s) contributed."
                )

        other = [e for e in evidence if e is not best]
        if other:
            reasons.append(
                f"{len(other)} additional finding(s) contributed to the assessment."
            )
        if density_category in ("C", "D"):
            reasons.append(
                f"Dense breast tissue (category {density_category}) was taken "
                "into account."
            )
        if not reasons:  # pragma: no cover - defensive
            reasons.append("No specific high-risk morphological features detected.")
        return reasons

    def _finish_inadequate(
        self,
        quality_report: Optional[Dict[str, Any]],
        provisional: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        warnings = list((quality_report or {}).get("warnings", []) or [])
        reasons = ["Technically inadequate image for a final assessment (BI-RADS 0)."]
        reasons.extend(warnings)
        result = self._result(
            category=0,
            confidence=0.5,
            reasons=reasons,
            evidence=[{"source": "quality", "detail": w} for w in warnings],
        )
        result["provisional_assessment"] = provisional
        return result

    def _result(
        self,
        category: int,
        confidence: float,
        reasons: List[str],
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "category": int(category),
            "category_text": CATEGORY_TEXT.get(category, "Unknown"),
            "management": CATEGORY_MANAGEMENT.get(category, ""),
            "confidence": round(float(confidence), 2),
            "reasons": reasons,
            "evidence": evidence,
            "label": Config.BIRADS_LABEL,
            "is_software_assessment": True,
            "categories_supported": [0, 1, 2, 3, 4, 5, 6],
            "disclaimer": Config.DISCLAIMER,
        }

    # ------------------------------------------------------------------ #
    #  Small helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_mass(detection) -> bool:
        name = getattr(getattr(detection, "type", None), "value", "") or ""
        return "tumor" in name or "mass" in name

    @staticmethod
    def _is_calcification(detection) -> bool:
        name = getattr(getattr(detection, "type", None), "value", "") or ""
        return "calcification" in name

    @staticmethod
    def _density_category(density_result: Optional[Dict]) -> Optional[str]:
        if not density_result:
            return None
        acr = density_result.get("acr_category")
        if acr:
            return str(acr)[:1].upper()
        category = density_result.get("category") or ""
        text = str(category).upper().replace("DENSITY", "").strip()
        return text[:1] if text[:1] in ("A", "B", "C", "D") else None

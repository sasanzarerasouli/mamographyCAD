"""Temporal comparison between current and prior mammography studies."""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("MammographyCAD.temporal")


class PriorStudyComparisonEngine:
    """Compare current study detections with prior study results."""

    GROWTH_THRESHOLD = 0.20  # 20% size increase
    SHRINK_THRESHOLD = 0.20

    def compare(
        self,
        current_detections: List,
        prior_detections: Optional[List] = None,
        current_density: Optional[Dict] = None,
        prior_density: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if not prior_detections:
            return {
                "has_prior": False,
                "new_lesions": [],
                "disappeared_lesions": [],
                "growth": [],
                "shrinkage": [],
                "density_changes": [],
                "summary": "No prior study available for comparison",
            }

        current_active = [d for d in current_detections if not d.is_ignored]
        prior_active = [d for d in prior_detections if not d.is_ignored]

        matched, new_lesions, disappeared = self._match_lesions(current_active, prior_active)
        growth, shrinkage = self._detect_size_changes(matched)
        density_changes = self._compare_density(current_density, prior_density)

        return {
            "has_prior": True,
            "new_lesions": [d.lesion_id for d in new_lesions],
            "disappeared_lesions": [d.lesion_id for d in disappeared],
            "growth": growth,
            "shrinkage": shrinkage,
            "density_changes": density_changes,
            "summary": self._build_summary(new_lesions, disappeared, growth, shrinkage),
        }

    def _match_lesions(self, current: List, prior: List, distance_threshold: float = 50):
        matched = []
        new_lesions = list(current)
        disappeared = list(prior)

        for c in current:
            best_dist = float("inf")
            best_p = None
            for p in prior:
                dist = np.sqrt((c.location[0] - p.location[0]) ** 2 + (c.location[1] - p.location[1]) ** 2)
                if dist < best_dist and dist < distance_threshold:
                    best_dist = dist
                    best_p = p
            if best_p:
                matched.append((c, best_p))
                if c in new_lesions:
                    new_lesions.remove(c)
                if best_p in disappeared:
                    disappeared.remove(best_p)

        return matched, new_lesions, disappeared

    def _detect_size_changes(self, matched: List) -> tuple:
        growth, shrinkage = [], []
        for current, prior in matched:
            curr_area = current.area_mm2 or current.area_px
            prior_area = prior.area_mm2 or prior.area_px
            if prior_area <= 0:
                continue
            change = (curr_area - prior_area) / prior_area
            entry = {
                "current_id": current.lesion_id,
                "prior_id": prior.lesion_id,
                "change_percent": round(change * 100, 1),
            }
            if change > self.GROWTH_THRESHOLD:
                growth.append(entry)
            elif change < -self.SHRINK_THRESHOLD:
                shrinkage.append(entry)
        return growth, shrinkage

    def _compare_density(self, current: Optional[Dict], prior: Optional[Dict]) -> List[str]:
        changes = []
        if current and prior:
            curr = current.get("dense_area_percent", 0)
            prev = prior.get("dense_area_percent", 0)
            diff = curr - prev
            if abs(diff) > 5:
                direction = "increased" if diff > 0 else "decreased"
                changes.append(f"Breast density {direction} by {abs(diff):.1f}%")
        return changes

    def _build_summary(self, new, disappeared, growth, shrinkage) -> str:
        parts = []
        if new:
            parts.append(f"{len(new)} new lesion(s)")
        if disappeared:
            parts.append(f"{len(disappeared)} disappeared lesion(s)")
        if growth:
            parts.append(f"{len(growth)} growing lesion(s)")
        if shrinkage:
            parts.append(f"{len(shrinkage)} shrinking lesion(s)")
        return "; ".join(parts) if parts else "No significant changes detected"

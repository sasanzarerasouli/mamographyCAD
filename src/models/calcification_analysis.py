"""BI-RADS calcification morphology and distribution analysis.

The detector reports *spots* (individual calcifications) and *clusters*.  This
module turns those detections into the two descriptors the BI-RADS lexicon
uses for calcifications:

* **Morphology** – typically benign (round, punctate, coarse/popcorn, large
  rod-like, dystrophic, lucent-centred, eggshell, vascular) versus suspicious
  (amorphous, coarse heterogeneous, fine pleomorphic, fine linear/branching).
* **Distribution** – diffuse, regional, grouped, linear, segmental.

Only geometric/appearance measurements of the detected regions are used; each
descriptor exposes the measurement it came from.  This is decision support,
not a diagnosis.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import Config

logger = logging.getLogger("MammographyCAD.calcification")

BENIGN_MORPHOLOGIES = (
    "round", "punctate", "coarse (popcorn)", "large rod-like",
    "dystrophic", "lucent-centred", "eggshell", "vascular",
)
SUSPICIOUS_MORPHOLOGIES = (
    "amorphous", "coarse heterogeneous", "fine pleomorphic",
    "fine linear / branching",
)
SUSPICIOUS_DISTRIBUTIONS = ("linear", "segmental", "grouped")


class CalcificationAnalyzer:
    """Classical (rule-based) BI-RADS calcification descriptor analysis."""

    def __init__(self) -> None:
        self.cluster_min_count = int(getattr(Config, "CALC_CLUSTER_MIN_COUNT", 3))
        self.cluster_radius_mm = float(getattr(Config, "CALC_CLUSTER_RADIUS_MM", 10.0))
        self.segmental_length_mm = float(
            getattr(Config, "CALC_SEGMENTAL_LENGTH_MM", 20.0)
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        detections: List,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        spacing_known: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Analyse all calcification detections of one image.

        ``spacing_known`` states whether ``pixel_spacing`` was measured (DICOM)
        or assumed; ``None`` falls back to inferring it from the values.
        """
        if spacing_known is None:
            spacing_known = not (
                abs(float(pixel_spacing[0]) - 1.0) < 1e-9
                and abs(float(pixel_spacing[1]) - 1.0) < 1e-9
            )
        else:
            spacing_known = bool(spacing_known)
        row_spacing, col_spacing = self._sanitised_spacing(pixel_spacing)
        if not spacing_known:
            # Use a documented assumption so the cluster criteria (defined in
            # mm) still behave sensibly; results are flagged as assumed.
            assumed = float(getattr(Config, "ASSUMED_PIXEL_SPACING_MM", 0.05))
            row_spacing = col_spacing = assumed
        mean_spacing = (row_spacing + col_spacing) / 2.0

        micro = [d for d in detections if self._kind(d) == "micro"]
        macro = [d for d in detections if self._kind(d) == "macro"]

        if not micro and not macro:
            return {
                "available": False,
                "reason": "no calcification detections",
                "microcalcification_count": 0,
                "macrocalcification_count": 0,
                "clusters": [],
                "dominant_morphology": None,
                "dominant_distribution": None,
                "suspicion_score": 0.0,
                "birads_terms": [],
                "spacing_known": spacing_known,
            }

        spots = [self._spot_from_detection(d, mean_spacing) for d in micro]
        clusters = self._cluster_spots(spots, mean_spacing)
        cluster_infos = [
            self._describe_cluster(group, mean_spacing) for group in clusters
        ]

        # The detector itself may already emit cluster-level detections (one
        # detection carrying N individual calcifications).  Those are clusters
        # in their own right and used to be invisible in ``cluster_count``,
        # which made the GUI say "0 clusters" while a cluster was highlighted.
        intrinsic = []
        for detection in micro:
            features = getattr(detection, "features", {}) or {}
            count = int(features.get("num_calcifications", 0) or 0)
            if count < self.cluster_min_count:
                continue
            grouped = any(
                any(
                    math.hypot(
                        spot["center"][0] - float(getattr(detection, "location", (0, 0))[0]),
                        spot["center"][1] - float(getattr(detection, "location", (0, 0))[1]),
                    ) <= self.cluster_radius_mm / max(mean_spacing, 1e-6)
                    for spot in group
                )
                for group in clusters
            )
            if grouped:
                continue
            individual = features.get("individual_calcifications") or []
            spot_models = [
                {
                    "center": tuple(spot.get("center") or (0.0, 0.0)),
                    "diameter_mm": float(spot.get("equivalent_diameter") or 0.0) * mean_spacing,
                    "width_mm": float(spot.get("equivalent_diameter") or 0.0) * mean_spacing,
                    "height_mm": float(spot.get("equivalent_diameter") or 0.0) * mean_spacing,
                    "circularity": float(spot.get("circularity") or 0.0),
                    "max_intensity": float(spot.get("max_intensity") or 0.0),
                    "local_contrast": float(spot.get("local_contrast") or 0.0),
                    "confidence": float(getattr(detection, "confidence", 0.0)),
                }
                for spot in individual
            ]
            if spot_models:
                morphology = self._classify_micro_morphology(spot_models)
                centers = np.array(
                    [spot["center"] for spot in spot_models], dtype=np.float64
                )
                if len(centers) > 1:
                    diff = centers[:, None, :] - centers[None, :, :]
                    pairwise = np.sqrt((diff ** 2).sum(axis=2))
                    extent_px = float(pairwise.max())
                else:
                    extent_px = 0.0
                distribution = self._classify_distribution(
                    centers, extent_px * mean_spacing, mean_spacing
                )
                diameters = np.array(
                    [spot["diameter_mm"] for spot in spot_models], dtype=np.float64
                )
                mean_diameter = float(diameters.mean()) if diameters.size else 0.0
                variability = (
                    float(diameters.std() / (mean_diameter + 1e-6)) if diameters.size else 0.0
                )
                max_diameter = float(diameters.max()) if diameters.size else 0.0
                extent_mm = extent_px * mean_spacing
            else:
                morphology, distribution, variability = None, None, 0.0
                extent_mm = float(features.get("cluster_size_mm") or 0.0)
                mean_diameter = float(features.get("mean_diameter_mm") or 0.0)
                max_diameter = mean_diameter
            intrinsic.append(
                {
                    "count": count,
                    "center": list(getattr(detection, "location", (0, 0))),
                    "extent_mm": round(extent_mm, 2),
                    "mean_spacing_mm": round(
                        float(features.get("mean_inter_distance") or 0.0) * mean_spacing, 2
                    ),
                    "mean_diameter_mm": round(mean_diameter, 2),
                    "max_diameter_mm": round(max_diameter, 2),
                    "size_variability": round(variability, 3),
                    "morphology": morphology,
                    "morphology_confidence": 0.55 if morphology else 0.0,
                    "distribution": distribution,
                    "spots": [spot["center"] for spot in spot_models],
                    "source": "detector",
                    "descriptor_basis": (
                        "derived from the per-calcification measurements of this "
                        "cluster detection"
                        if spot_models
                        else "no per-calcification measurements available"
                    ),
                }
            )
        if intrinsic:
            cluster_infos = list(cluster_infos) + intrinsic
        macro_infos = [self._describe_macro(d, mean_spacing) for d in macro]

        # Spots that belong to no real cluster are still described, so a single
        # suspicious calcificación is never silently dropped.
        stray = [s for s in spots if not s["in_cluster"]]
        stray_infos = [
            {
                "count": 1,
                "morphology": self._classify_micro_morphology([s]),
                "morphology_confidence": 0.4,
                "distribution": "isolated",
                "extent_mm": s["diameter_mm"],
                "center": s["center"],
                "spots": [s["center"]],
            }
            for s in stray
        ]
        all_groups = cluster_infos + stray_infos

        dominant_morphology, morph_conf = self._dominant_morphology(
            cluster_infos, macro_infos, stray_infos
        )
        dominant_distribution, dist_conf = self._dominant_distribution(cluster_infos)

        suspicion = self._suspicion_score(
            dominant_morphology, dominant_distribution, cluster_infos, len(micro), len(macro)
        )
        terms = self._birads_terms(
            dominant_morphology, dominant_distribution, len(micro), len(cluster_infos)
        )

        result = {
            "available": True,
            "microcalcification_count": len(micro),
            "macrocalcification_count": len(macro),
            "cluster_count": len(cluster_infos),
            "largest_cluster_count": max(
                (c["count"] for c in cluster_infos), default=0
            ),
            "clusters": cluster_infos,
            "macrocalcifications": macro_infos,
            "isolated_calcifications": stray_infos,
            "dominant_morphology": dominant_morphology,
            "morphology_confidence": round(float(morph_conf), 3),
            "dominant_distribution": dominant_distribution,
            "distribution_confidence": round(float(dist_conf), 3),
            "suspicion_score": round(float(suspicion), 3),
            "suspicious_morphology": dominant_morphology in SUSPICIOUS_MORPHOLOGIES,
            "suspicious_distribution": dominant_distribution in SUSPICIOUS_DISTRIBUTIONS,
            "birads_terms": terms,
            "suspicion_level": self._suspicion_level(suspicion),
            "spacing_known": spacing_known,
            "spacing_assumed": not spacing_known,
            "assumed_spacing_mm": row_spacing if not spacing_known else None,
            "measurement_basis": (
                f"pixel spacing from DICOM ({row_spacing:.4f} mm/px)" if spacing_known
                else (
                    f"ASSUMED pixel spacing {row_spacing:.4f} mm/px (no DICOM "
                    "spacing available) - measurements are estimates"
                )
            ),
            "note": (
                "Descriptor values are software-derived measurements of image "
                "appearance (BI-RADS lexicon terms), not a diagnosis."
            ),
        }
        logger.info(
            "Calcification analysis: %d micro / %d macro, %d cluster(s), "
            "morphology=%s, distribution=%s",
            len(micro), len(macro), len(cluster_infos),
            dominant_morphology, dominant_distribution,
        )
        return result

    # ------------------------------------------------------------------ #
    #  Spot extraction                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitised_spacing(pixel_spacing) -> Tuple[float, float]:
        try:
            row, col = float(pixel_spacing[0]), float(pixel_spacing[1])
            if row <= 0 or col <= 0:
                raise ValueError
            return row, col
        except (TypeError, ValueError, IndexError):
            return 1.0, 1.0

    @staticmethod
    def _kind(detection) -> str:
        name = getattr(getattr(detection, "type", None), "value", "") or ""
        if "micro" in name:
            return "micro"
        if "macro" in name:
            return "macro"
        return ""

    def _spot_from_detection(self, detection, mean_spacing: float) -> Dict[str, Any]:
        features = getattr(detection, "features", {}) or {}
        bbox = getattr(detection, "bbox", None) or (0, 0, 0, 0)
        x, y, w, h = [float(v) for v in bbox]
        width_mm = w * mean_spacing
        height_mm = h * mean_spacing
        diameter_mm = float(np.hypot(width_mm, height_mm)) if w and h else 0.0
        # Equivalent diameter from the segmented area when available
        equivalent = features.get("equivalent_diameter")
        if equivalent:
            diameter_mm = float(equivalent) * mean_spacing
        elif "total_area" in features and features.get("num_calcifications"):
            area_px = float(features["total_area"]) / float(features["num_calcifications"])
            diameter_mm = 2.0 * math.sqrt(max(area_px, 0.0) / math.pi) * mean_spacing

        center_xy = features.get("center") or getattr(detection, "location", (0, 0))
        return {
            "center": (float(center_xy[0]), float(center_xy[1])),
            "bbox": (x, y, w, h),
            "width_mm": width_mm,
            "height_mm": height_mm,
            "diameter_mm": diameter_mm,
            "circularity": float(features.get("mean_circularity", features.get("circularity", 0.0)) or 0.0),
            "max_intensity": float(
                features.get("max_intensity", features.get("max_intensity_enhanced", 0.0)) or 0.0
            ),
            "local_contrast": float(
                features.get("mean_contrast", features.get("local_contrast", 0.0)) or 0.0
            ),
            "confidence": float(getattr(detection, "confidence", 0.0)),
            "in_cluster": False,
            "detection": detection,
        }

    # ------------------------------------------------------------------ #
    #  Clustering                                                        #
    # ------------------------------------------------------------------ #
    def _cluster_spots(
        self, spots: List[Dict[str, Any]], mean_spacing: float
    ) -> List[List[Dict[str, Any]]]:
        """Group neighbouring calcifications (DBSCAN with a 1 cm radius)."""
        if not spots:
            return []
        points = np.array([s["center"] for s in spots], dtype=np.float64)
        eps_mm = max(self.cluster_radius_mm, 5.0)
        eps_px = eps_mm / max(mean_spacing, 1e-6)
        # Cap eps so a coarse pixel spacing cannot merge the whole breast
        eps_px = float(np.clip(eps_px, 8.0, 200.0))

        labels = self._dbscan(points, eps=eps_px, min_samples=2)
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for label, spot in zip(labels, spots):
            if label < 0:
                continue
            groups.setdefault(int(label), []).append(spot)

        clusters: List[List[Dict[str, Any]]] = []
        for group in groups.values():
            if len(group) >= self.cluster_min_count:
                for spot in group:
                    spot["in_cluster"] = True
                clusters.append(group)
        return clusters

    @staticmethod
    def _dbscan(points: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
        """Small self-contained DBSCAN (avoids a sklearn dependency at runtime)."""
        n = len(points)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        neighbours = [np.flatnonzero(distances[i] <= eps) for i in range(n)]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            if len(neighbours[i]) < min_samples:
                continue
            labels[i] = cluster_id
            seeds = list(neighbours[i])
            while seeds:
                j = seeds.pop()
                if not visited[j]:
                    visited[j] = True
                    if len(neighbours[j]) >= min_samples:
                        seeds.extend(neighbours[j].tolist())
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
        return labels

    # ------------------------------------------------------------------ #
    #  Descriptors                                                       #
    # ------------------------------------------------------------------ #
    def _describe_cluster(
        self, group: List[Dict[str, Any]], mean_spacing: float
    ) -> Dict[str, Any]:
        centers = np.array([s["center"] for s in group], dtype=np.float64)
        diameter = [s["diameter_mm"] for s in group]

        if len(centers) > 1:
            pairwise = np.sqrt(
                ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            )
            extent_mm = float(pairwise.max()) * mean_spacing
            mean_distance_mm = float(pairwise[np.triu_indices(len(centers), 1)].mean()) * mean_spacing
        else:
            extent_mm = float(diameter[0]) if diameter else 0.0
            mean_distance_mm = 0.0

        morphology = self._classify_micro_morphology(group)
        distribution = self._classify_distribution(centers, extent_mm, mean_spacing)

        return {
            "count": len(group),
            "center": (
                float(centers[:, 0].mean()),
                float(centers[:, 1].mean()),
            ),
            "extent_mm": round(extent_mm, 2),
            "mean_spacing_mm": round(mean_distance_mm, 2),
            "mean_diameter_mm": round(float(np.mean(diameter)) if diameter else 0.0, 3),
            "max_diameter_mm": round(float(np.max(diameter)) if diameter else 0.0, 3),
            "size_variability": round(
                float(np.std(diameter) / (np.mean(diameter) + 1e-6)) if diameter else 0.0, 3
            ),
            "morphology": morphology,
            "morphology_confidence": round(
                float(np.clip(len(group) / 8.0, 0.35, 0.95)), 3
            ),
            "distribution": distribution,
            "spots": [s["center"] for s in group],
        }

    def _classify_micro_morphology(self, group: List[Dict[str, Any]]) -> str:
        """Morphology from size, shape consistency and density variability."""
        diameters = np.array([s["diameter_mm"] for s in group], dtype=np.float64)
        widths = np.array([s["width_mm"] for s in group], dtype=np.float64)
        heights = np.array([s["height_mm"] for s in group], dtype=np.float64)
        circularity = np.array([s["circularity"] for s in group], dtype=np.float64)

        mean_diameter = float(np.mean(diameters)) if diameters.size else 0.0
        size_cv = float(np.std(diameters) / (mean_diameter + 1e-6)) if diameters.size else 0.0
        mean_circ = float(np.mean(circularity)) if circularity.size else 0.0

        # Elongated, thin, very small, varying -> fine linear/branching
        elongated_ratio = 0.0
        if widths.size and heights.size:
            ratios = np.maximum(widths, heights) / (np.minimum(widths, heights) + 1e-6)
            elongated_ratio = float(np.mean(ratios >= 3.0))

        if elongated_ratio >= 0.5:
            return "fine linear / branching"
        if mean_diameter < 0.5 and mean_circ >= 0.7 and size_cv < 0.35:
            return "punctate"
        if size_cv >= 0.5 or (mean_circ < 0.45 and mean_diameter < 1.5):
            return "fine pleomorphic"
        if mean_diameter < 1.0 and mean_circ < 0.6:
            return "amorphous"
        if mean_diameter >= 0.5 and size_cv >= 0.25:
            return "coarse heterogeneous"
        if mean_circ >= 0.8 and mean_diameter < 1.0:
            return "round"
        return "amorphous" if mean_circ < 0.6 else "punctate"

    def _classify_distribution(
        self,
        centers: np.ndarray,
        extent_mm: float,
        mean_spacing: float,
    ) -> str:
        if len(centers) < self.cluster_min_count:
            return "isolated"
        if extent_mm <= self.cluster_radius_mm:
            return "grouped"

        # Linear / segmental: a straight line fits the cluster tightly and it
        # extends far enough to follow a duct.
        linearity = self._linearity(centers)
        if linearity >= 0.92 and extent_mm >= self.segmental_length_mm:
            return "linear"
        if linearity >= 0.80 and extent_mm >= self.segmental_length_mm:
            return "segmental"
        if extent_mm <= 3.0 * self.cluster_radius_mm:
            return "regional"
        return "diffuse"

    @staticmethod
    def _linearity(centers: np.ndarray) -> float:
        """1 - (residual / extent) of a principal-axis fit."""
        if len(centers) < 3:
            return 0.0
        centered = centers - centers.mean(axis=0)
        try:
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        residuals = np.abs(centered @ normal)
        extent = float(np.ptp(centered @ direction)) or 1e-6
        return float(np.clip(1.0 - (2.0 * float(np.mean(residuals)) / extent), 0.0, 1.0))

    def _describe_macro(self, detection, mean_spacing: float) -> Dict[str, Any]:
        features = getattr(detection, "features", {}) or {}
        bbox = getattr(detection, "bbox", None) or (0, 0, 0, 0)
        x, y, w, h = [float(v) for v in bbox]
        width_mm, height_mm = w * mean_spacing, h * mean_spacing
        aspect = max(width_mm, height_mm) / (min(width_mm, height_mm) + 1e-6)
        circularity = float(features.get("circularity", 0.0) or 0.0)

        if aspect >= 4.0:
            morphology = "vascular"
        elif width_mm >= 5.0 and circularity and circularity < 0.4:
            morphology = "dystrophic"
        elif width_mm >= 3.0:
            morphology = "coarse (popcorn)"
        elif circularity >= 0.75:
            morphology = "round"
        else:
            morphology = "large rod-like"

        return {
            "center": getattr(detection, "location", (0, 0)),
            "width_mm": round(width_mm, 2),
            "height_mm": round(height_mm, 2),
            "circularity": round(circularity, 3),
            "morphology": morphology,
            "benign_appearing": morphology in BENIGN_MORPHOLOGIES,
            "confidence": float(getattr(detection, "confidence", 0.0)),
        }

    # ------------------------------------------------------------------ #
    #  Aggregation                                                       #
    # ------------------------------------------------------------------ #
    def _dominant_morphology(
        self,
        clusters: List[Dict[str, Any]],
        macros: List[Dict[str, Any]],
        strays: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], float]:
        # Suspicious morphology always dominates the summary when present.
        suspicious = [c for c in clusters if c["morphology"] in SUSPICIOUS_MORPHOLOGIES]
        if suspicious:
            suspicious.sort(key=lambda c: (-c["count"], -c["size_variability"]))
            return suspicious[0]["morphology"], suspicious[0]["morphology_confidence"]

        if clusters:
            clusters = sorted(clusters, key=lambda c: (-c["count"], -c["extent_mm"]))
            return clusters[0]["morphology"], clusters[0]["morphology_confidence"]

        if macros:
            return macros[0]["morphology"], 0.6

        if strays:
            return strays[0]["morphology"], 0.4
        return None, 0.0

    def _dominant_distribution(
        self, clusters: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], float]:
        if not clusters:
            return None, 0.0
        priority = {"segmental": 5, "linear": 4, "grouped": 3, "regional": 2, "diffuse": 1}
        ranked = sorted(
            clusters,
            key=lambda c: (priority.get(c["distribution"], 0), c["count"]),
            reverse=True,
        )
        best = ranked[0]
        return best["distribution"], float(
            np.clip(0.5 + best["count"] / 20.0, 0.4, 0.95)
        )

    def _suspicion_score(
        self,
        morphology: Optional[str],
        distribution: Optional[str],
        clusters: List[Dict[str, Any]],
        micro_count: int,
        macro_count: int,
    ) -> float:
        score = 0.0
        morph_weights = {
            "fine linear / branching": 0.55,
            "fine pleomorphic": 0.45,
            "amorphous": 0.38,
            "coarse heterogeneous": 0.32,
            "punctate": 0.08,
            "round": 0.05,
            "vascular": 0.05,
            "dystrophic": 0.03,
            "coarse (popcorn)": 0.03,
            "large rod-like": 0.05,
        }
        if morphology:
            score += morph_weights.get(morphology, 0.15)

        dist_weights = {"segmental": 0.30, "linear": 0.28, "grouped": 0.20, "regional": 0.08, "diffuse": 0.02}
        if distribution:
            score += dist_weights.get(distribution, 0.0)

        largest = max((c["count"] for c in clusters), default=0)
        if largest >= 10:
            score += 0.18
        elif largest >= 5:
            score += 0.12
        elif largest >= self.cluster_min_count:
            score += 0.06

        if micro_count >= 20:
            score += 0.05
        # Pure macrocalcification without any micro pattern is usually benign
        if micro_count == 0 and macro_count > 0:
            score *= 0.5
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _suspicion_level(score: float) -> str:
        if score >= 0.6:
            return "high"
        if score >= 0.35:
            return "intermediate"
        if score > 0.1:
            return "low"
        return "very low"

    def _birads_terms(
        self,
        morphology: Optional[str],
        distribution: Optional[str],
        micro_count: int,
        cluster_count: int,
    ) -> List[str]:
        terms: List[str] = []
        if morphology:
            terms.append(f"Morphology: {morphology}")
        if distribution:
            terms.append(f"Distribution: {distribution}")
        if micro_count:
            terms.append(f"{micro_count} microcalcification candidate(s)")
        if cluster_count:
            terms.append(f"{cluster_count} group(s) meeting the cluster criterion")
        if not terms:
            terms.append("No calcification descriptors available")
        return terms

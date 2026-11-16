"""Suspicion heatmap generation for mammography analysis."""
import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger("MammographyCAD.heatmap")


class SuspicionHeatmapGenerator:
    """Generate risk heatmaps overlaying low/medium/high suspicion regions."""

    def generate(
        self,
        gray: np.ndarray,
        detections: List,
        breast_mask: np.ndarray,
        risk_results: Optional[List[dict]] = None,
    ) -> np.ndarray:
        h, w = gray.shape
        heatmap = np.zeros((h, w), dtype=np.float32)

        risk_map = {}
        if risk_results:
            for i, r in enumerate(risk_results):
                if i < len(detections):
                    risk_map[id(detections[i])] = r.get("risk_score", 50) / 100.0

        for detection in detections:
            if detection.is_ignored:
                continue
            risk = risk_map.get(id(detection), detection.confidence)
            cx, cy = detection.location
            if detection.bbox:
                x, y, bw, bh = detection.bbox
                cv2.ellipse(
                    heatmap,
                    (cx, cy),
                    (max(bw // 2, 10), max(bh // 2, 10)),
                    0, 0, 360,
                    float(risk),
                    -1,
                )
            else:
                cv2.circle(heatmap, (cx, cy), 30, float(risk), -1)

        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap[breast_mask == 0] = 0
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(base, 0.6, colored, 0.4, 0)
        logger.debug("Heatmap generated with %d regions", len(detections))
        return overlay

    def apply_opacity(self, heatmap: np.ndarray, base: np.ndarray, opacity: float) -> np.ndarray:
        if heatmap is None:
            return base
        return cv2.addWeighted(base, 1.0 - opacity, heatmap, opacity, 0)

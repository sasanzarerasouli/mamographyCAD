import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
from scipy import ndimage
import logging

from ..core.detection_types import Detection, DetectionType

logger = logging.getLogger("MammographyCAD")


class CalcificationDetector:
    
    def __init__(self):
        # Size ranges (pixels) - extended ranges for better detection
        self.micro_size_range = (0.5, 10)      # pixels (extended lower bound)
        self.macro_size_range = (8, 60)        # pixels (extended upper bound)
        
        # Multi-scale detection parameters
        self.tophat_kernels = [(9, 9), (15, 15), (21, 21)]
        
        # Detection thresholds (lowered for higher sensitivity)
        self.adaptive_block_size = 11
        self.adaptive_c_values = [-8, -5, -2]  # Multiple thresholds for fusion
        
        # Hard caps that keep CPU inference bounded on large mammograms
        self.max_candidates = 4000
        self.max_filtered_spots = 1200
        self.cluster_max_spots = 60           # a 1 cm group rarely exceeds this
        self.max_cluster_extent_px = 120      # compact-group extent guard
        self.min_detection_confidence = 0.45
        self.min_prominence = 14.0            # grey levels above local background
        self.min_cluster_prominence = 12.0
        self.max_ring_density = 0.55          # surrounded-by-similar-tissue guard
        self.max_detections = 200

        # Clustering parameters
        self.dbscan_eps = 40  # Tighter clustering
        self.dbscan_min_samples = 2  # Lower threshold to catch small clusters
        self.dbscan_eps_macro = 80  # For macro grouping
    
    def detect(self, enhanced: np.ndarray, breast_mask: np.ndarray) -> List[Detection]:
        detections = []

        # Exclude the skin line: the mask boundary is an artificial high-contrast
        # ramp that is otherwise detected as calcification-like specks.
        breast_mask = self._erode_skin_line(breast_mask)

        # Multi-scale preprocessing
        preprocessed = self._multiscale_preprocess(enhanced, breast_mask)
        
        # Multi-threshold detection for higher sensitivity
        bright_spots = self._detect_bright_spots_multithreshold(preprocessed, enhanced)
        
        # Filter by size and shape
        valid_spots = self._filter_spots(bright_spots)
        
        # Classify and cluster
        micro_clusters, macro_spots = self._classify_and_cluster(valid_spots, enhanced, breast_mask)
        
        # Build detections
        for cluster in micro_clusters:
            detection = Detection(
                type=DetectionType.MICROCALCIFICATION,
                location=cluster['center'],
                bbox=cluster['bbox'],
                confidence=cluster['confidence'],
                features=cluster['features']
            )
            detections.append(detection)
        
        for spot in macro_spots:
            detection = Detection(
                type=DetectionType.MACROCALCIFICATION,
                location=spot['center'],
                bbox=spot['bbox'],
                confidence=spot['confidence'],
                features=spot['features']
            )
            detections.append(detection)
        
        # Final relevance gate: discard weak candidates so the result list stays
        # clinically meaningful (and the GUI usable).
        detections = [
            d for d in detections
            if d.confidence >= self.min_detection_confidence
        ]
        if len(detections) > self.max_detections:
            detections.sort(key=lambda d: d.confidence, reverse=True)
            detections = detections[: self.max_detections]

        logger.info(
            "Detected %d micro clusters and %d macro spots (%d reported)",
            len(micro_clusters), len(macro_spots), len(detections),
        )
        return detections
    
    @staticmethod
    def _erode_skin_line(breast_mask: np.ndarray, margin_ratio: float = 0.012) -> np.ndarray:
        """Shrink the breast mask by ~1% of the image size (skin-line guard)."""
        if breast_mask is None or breast_mask.size == 0:
            return breast_mask
        height, width = breast_mask.shape[:2]
        margin = max(3, int(round(min(height, width) * margin_ratio)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
        eroded = cv2.erode((breast_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)
        if np.count_nonzero(eroded) < 0.2 * np.count_nonzero(breast_mask):
            return (breast_mask > 0).astype(np.uint8) * 255
        return eroded

    # -------------------- Preprocessing --------------------
    
    def _multiscale_preprocess(self, enhanced: np.ndarray, breast_mask: np.ndarray) -> np.ndarray:
        """Multi-scale tophat + background subtraction for robust calcification enhancement"""
        tophat_sum = np.zeros_like(enhanced, dtype=np.float32)
        
        for ksize in self.tophat_kernels:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
            th = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            tophat_sum += th.astype(np.float32)
        
        # Average multi-scale response
        tophat_avg = (tophat_sum / len(self.tophat_kernels))
        tophat_avg = cv2.normalize(tophat_avg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply breast mask
        result = cv2.bitwise_and(tophat_avg, tophat_avg, mask=breast_mask)
        return result
    
    # -------------------- Detection --------------------
    
    def _detect_bright_spots_multithreshold(self, preprocessed: np.ndarray,
                                           enhanced: np.ndarray) -> List[Dict]:
        """Detect bright spots using multiple adaptive thresholds and merge results"""
        all_spots = []
        
        for c_val in self.adaptive_c_values:
            thresh = cv2.adaptiveThreshold(
                preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, self.adaptive_block_size, c_val
            )
            spots = self._extract_spots_from_threshold(thresh, preprocessed, enhanced)
            all_spots.extend(spots)
        
        # Also add Otsu-based detection for global sensitivity
        _, otsu_thresh = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_spots = self._extract_spots_from_threshold(otsu_thresh, preprocessed, enhanced)
        all_spots.extend(otsu_spots)
        
        # Deduplicate overlapping detections
        unique_spots = self._deduplicate_spots(all_spots)
        return unique_spots
    
    def _extract_spots_from_threshold(self, thresh: np.ndarray, preprocessed: np.ndarray,
                                      enhanced: np.ndarray) -> List[Dict]:
        """Extract bright-spot candidates from a binary mask.

        Implementation notes
        --------------------
        The original version looped over every contour and allocated a
        full-resolution mask for each one.  Adaptive thresholding can produce
        >10 000 speckles, which made this method take minutes and exhaust
        memory.  Connected-component labelling plus a per-component *bounding
        box* mask keeps the cost linear in the number of pixels, and only
        candidates that pass the geometric pre-filter are analysed further.
        """
        # Morphological opening to remove single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        height, width = thresh.shape[:2]
        max_area = float(self.macro_size_range[1] ** 2 * 1.5)
        min_area = 0.5

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )
        if num_labels <= 1:
            return []

        # Vectorised geometric pre-filter over all components at once
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        keep = np.flatnonzero((areas >= min_area) & (areas <= max_area)) + 1
        if keep.size == 0:
            return []
        # Bound the number of candidates that reach the (more expensive) feature
        # extraction: strongest responses first.
        if keep.size > self.max_candidates:
            # Per-component maximum computed in a single pass (the previous
            # per-label full-image comparison was O(n * H * W)).
            flat_labels = labels.ravel()
            flat_values = preprocessed.ravel().astype(np.float32)
            component_max = np.zeros(int(num_labels), dtype=np.float32)
            np.maximum.at(component_max, flat_labels, flat_values)
            order = np.argsort(component_max[keep])[::-1][: self.max_candidates]
            keep = keep[order]

        spots: List[Dict] = []
        for label_id in keep:
            x, y, w, h = (
                int(stats[label_id, cv2.CC_STAT_LEFT]),
                int(stats[label_id, cv2.CC_STAT_TOP]),
                int(stats[label_id, cv2.CC_STAT_WIDTH]),
                int(stats[label_id, cv2.CC_STAT_HEIGHT]),
            )
            if w <= 0 or h <= 0:
                continue
            component = labels[y:y + h, x:x + w] == label_id
            area = float(component.sum())
            if area <= 0:
                continue

            cx = int(round(centroids[label_id][0]))
            cy = int(round(centroids[label_id][1]))
            if not (0 <= cx < width and 0 <= cy < height):
                continue

            roi_pre = preprocessed[y:y + h, x:x + w]
            roi_enh = enhanced[y:y + h, x:x + w]
            pre_values = roi_pre[component]
            enh_values = roi_enh[component]
            if pre_values.size == 0:
                continue

            perimeter = self._component_perimeter(component)
            circularity = (
                4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
            )
            equivalent_diameter = float(np.sqrt(4 * area / np.pi))
            local_contrast = self._local_contrast(
                enhanced, cx, cy, radius=max(w, h) * 2 + 5
            )
            # Scale-correct prominence: how much brighter is this spot than the
            # immediately surrounding parenchyma?  This is the feature that
            # actually separates a calcification from textured tissue.
            prominence, ring_density = self._prominence(
                enhanced, component, x, y, equivalent_diameter
            )

            spots.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area,
                'perimeter': perimeter,
                'circularity': float(circularity),
                'equivalent_diameter': equivalent_diameter,
                'mean_intensity': float(np.mean(pre_values)),
                'max_intensity': float(np.max(pre_values)),
                'mean_intensity_enhanced': float(np.mean(enh_values)),
                'max_intensity_enhanced': float(np.max(enh_values)),
                'local_contrast': local_contrast,
                'prominence': prominence,
                'ring_density': ring_density,
                'mask': None,          # built on demand for the final lesion only
                'component': (x, y, component),
            })

        return spots

    @staticmethod
    def _prominence(gray: np.ndarray, component: np.ndarray, x: int, y: int,
                    equivalent_diameter: float) -> Tuple[float, float]:
        """Mean component intensity minus the median of its surrounding annulus.

        The annulus starts at 1.6x and ends at 4x the spot radius, which keeps
        the measurement independent of the spot size (the previous fixed-radius
        contrast diluted small spots and made every speckle look suspicious).
        """
        height, width = component.shape
        radius = max(1.0, equivalent_diameter / 2.0)
        inner = max(2, int(round(radius * 1.6)))
        outer = max(inner + 2, int(round(radius * 4.0)))

        y0 = max(0, y - outer)
        y1 = min(gray.shape[0], y + height + outer)
        x0 = max(0, x - outer)
        x1 = min(gray.shape[1], x + width + outer)
        if y1 <= y0 or x1 <= x0:
            return 0.0, 1.0

        roi = gray[y0:y1, x0:x1].astype(np.float32)
        mask = np.zeros(roi.shape, dtype=np.uint8)
        mask[y - y0:y - y0 + height, x - x0:x - x0 + width] = component.astype(np.uint8)
        if not mask.any():
            return 0.0, 1.0
        # Binarise to 0/255: distanceTransform needs a clean two-level mask
        mask = (mask > 0).astype(np.uint8) * 255

        # Distance *from the background* to the spot, so the annulus can be
        # selected by a size-relative radius.
        dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        ring = (dist > inner) & (dist <= outer)
        ring_values = roi[ring]
        spot_values = roi[mask > 0]
        if ring_values.size < 8 or spot_values.size == 0:
            return 0.0, 1.0
        spot_mean = float(np.mean(spot_values))
        prominence = spot_mean - float(np.median(ring_values))
        # Fraction of the annulus that is nearly as bright as the spot: a
        # calcification sits on darker parenchyma, dense tissue does not.
        ring_density = float(np.mean(ring_values >= spot_mean - 15.0))
        return prominence, ring_density

    @staticmethod
    def _component_perimeter(component: np.ndarray) -> float:
        """Perimeter of a binary component via contour arc length."""
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return 0.0
        return float(sum(cv2.arcLength(c, True) for c in contours))

    def _local_contrast(self, image: np.ndarray, cx: int, cy: int, radius: int) -> float:
        """Compute local contrast around a point (center - surroundings)"""
        h, w = image.shape
        r = max(3, min(radius, 30))
        
        # Inner region
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        center_region = image[y1:y2, x1:x2]
        
        # Outer ring
        r_outer = int(r * 2.5)
        y1o, y2o = max(0, cy - r_outer), min(h, cy + r_outer + 1)
        x1o, x2o = max(0, cx - r_outer), min(w, cx + r_outer + 1)
        outer_region = image[y1o:y2o, x1o:x2o]
        
        if center_region.size == 0 or outer_region.size == 0:
            return 0.0
        
        return float(np.mean(center_region) - np.mean(outer_region))
    
    def _deduplicate_spots(self, spots: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Remove duplicate detections from the multiple thresholds.

        Uses a spatial hash keyed on the bounding-box centre so the cost is
        O(n) instead of the previous O(n^2) pairwise comparison (which became
        prohibitive once thousands of candidates were generated).
        """
        if not spots:
            return []

        spots_sorted = sorted(spots, key=lambda s: s['max_intensity'], reverse=True)
        if len(spots_sorted) > self.max_candidates:
            spots_sorted = spots_sorted[: self.max_candidates]

        kept: List[Dict] = []
        grid: Dict[Tuple[int, int], List[int]] = {}
        cell = 4  # px: duplicates of the same calcification are < 3 px apart

        def cells_for(cx: int, cy: int):
            gx, gy = cx // cell, cy // cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    yield gx + dx, gy + dy

        for spot in spots_sorted:
            x, y, w, h = spot['bbox']
            cx, cy = spot['center']
            duplicate = False
            for key in cells_for(cx, cy):
                for index in grid.get(key, ()):
                    other = kept[index]
                    kx, ky, kw, kh = other['bbox']
                    ix1, iy1 = max(x, kx), max(y, ky)
                    ix2, iy2 = min(x + w, kx + kw), min(y + h, ky + kh)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = w * h + kw * kh - inter
                        if union > 0 and inter / union > iou_threshold:
                            duplicate = True
                            break
                    if abs(cx - other['center'][0]) < 3 and abs(cy - other['center'][1]) < 3:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                continue
            kept.append(spot)
            grid.setdefault((cx // cell, cy // cell), []).append(len(kept) - 1)

        return kept

    # -------------------- Filtering --------------------
    
    def _filter_spots(self, spots: List[Dict]) -> List[Dict]:
        """Remove obvious non-calcification candidates and bound the workload.

        A relative-intensity gate is applied on top of the absolute criteria:
        on a real (or noisy synthetic) mammogram the multi-threshold stage can
        emit thousands of low-contrast speckles that are indistinguishable from
        parenchymal texture.  Keeping only candidates that are either bright in
        absolute terms or clearly brighter than their surroundings removes them
        without touching genuine microcalcifications.
        """
        if not spots:
            return []

        geometric = []
        for spot in spots:
            # Reject extremely non-circular large blobs (likely tissue)
            if spot['area'] > 100 and spot['circularity'] < 0.3:
                continue
            # Reject very low intensity
            if spot['max_intensity'] < 100:
                continue
            # Reject candidates that are not distinctly brighter than their
            # surroundings (parenchymal texture, noise, film grain)
            if spot.get('prominence', 0.0) < self.min_prominence:
                continue
            # Microcalcifications are compact: reject elongated specks
            if spot['area'] <= 25 and spot['circularity'] < 0.25:
                continue
            # A spot embedded in tissue of similar brightness is parenchyma
            if spot.get('ring_density', 0.0) > self.max_ring_density:
                continue
            # Reject huge blobs (likely vessels/tissue)
            if spot['area'] > self.macro_size_range[1] ** 2:
                continue
            geometric.append(spot)

        if not geometric:
            return []

        intensities = np.array([s['max_intensity'] for s in geometric], dtype=np.float64)
        contrasts = np.array([s['local_contrast'] for s in geometric], dtype=np.float64)
        intensity_gate = float(np.percentile(intensities, 60))

        filtered = [
            spot for spot in geometric
            if spot['max_intensity'] >= intensity_gate or spot['local_contrast'] >= 30.0
        ]
        if not filtered:
            filtered = geometric

        # Hard cap: rank by a combined relevance score
        if len(filtered) > self.max_filtered_spots:
            filtered.sort(
                key=lambda s: (s['max_intensity'] / 255.0) + max(0.0, s['local_contrast']) / 60.0,
                reverse=True,
            )
            filtered = filtered[: self.max_filtered_spots]
        return filtered

    # -------------------- Classification & Clustering --------------------
    
    def _classify_and_cluster(self, spots: List[Dict], enhanced: np.ndarray,
                              breast_mask: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        micro_candidates, macro_candidates = [], []
        
        for spot in spots:
            d = spot['equivalent_diameter']
            if self.micro_size_range[0] <= d <= self.micro_size_range[1]:
                micro_candidates.append(spot)
            elif self.macro_size_range[0] <= d <= self.macro_size_range[1]:
                macro_candidates.append(spot)
            else:
                # Borderline: classify by intensity & contrast
                if spot['local_contrast'] > 25 and spot['max_intensity'] > 200:
                    if d < self.micro_size_range[1]:
                        micro_candidates.append(spot)
                    else:
                        macro_candidates.append(spot)
        
        micro_clusters = self._cluster_microcalcifications(micro_candidates, enhanced)
        macro_spots = self._process_macros(macro_candidates, enhanced)
        
        return micro_clusters, macro_spots
    
    def _cluster_microcalcifications(self, micro_candidates: List[Dict],
                                     enhanced: np.ndarray) -> List[Dict]:
        if not micro_candidates:
            return []
        
        points = np.array([spot['center'] for spot in micro_candidates], dtype=np.float64)
        
        # Adaptive eps based on local density
        eps = self._adaptive_eps(points, base_eps=self.dbscan_eps)
        
        clustering = DBSCAN(eps=eps, min_samples=self.dbscan_min_samples).fit(points)
        labels = clustering.labels_
        
        clusters = []
        n_noise_as_cluster = 0
        
        for label in set(labels):
            cluster_indices = np.where(labels == label)[0]
            cluster_spots = [micro_candidates[i] for i in cluster_indices]
            
            # If noise (label == -1), check if any spot is suspicious enough to keep solo
            if label == -1:
                for spot in cluster_spots:
                    if self._is_strong_isolated_micro(spot):
                        single_cluster = self._build_micro_cluster([spot], enhanced)
                        clusters.append(single_cluster)
                        n_noise_as_cluster += 1
                continue
            
            # If cluster has only 1-2 spots, only keep if intense
            if len(cluster_spots) < 2:
                if not self._is_strong_isolated_micro(cluster_spots[0]):
                    continue

            for group in self._split_by_extent(cluster_spots, eps):
                if len(group) < 2 and not self._is_strong_isolated_micro(group[0]):
                    continue
                if self._mean_prominence(group) < self.min_cluster_prominence:
                    continue
                clusters.append(self._build_micro_cluster(group, enhanced))
        
        # Merge clusters that are very close
        clusters = self._merge_nearby_clusters(clusters, distance_threshold=eps * 1.5)
        return clusters

    def _split_by_extent(self, spots: List[Dict], eps: float) -> List[List[Dict]]:
        """Split a chained DBSCAN group into spatially compact clusters.

        DBSCAN links points transitively, so on a speckled background a single
        label can span the whole breast.  A calcification *cluster* is a compact
        1 cm group, therefore groups are re-grown greedily while both the link
        distance and the maximum extent stay bounded.
        """
        if not spots:
            return []
        if len(spots) == 1:
            return [spots]

        points = np.array([s['center'] for s in spots], dtype=np.float64)
        if len(spots) <= self.cluster_max_spots:
            extent = float(
                np.sqrt(((points.max(axis=0) - points.min(axis=0)) ** 2).sum())
            )
            if extent <= self.max_cluster_extent_px:
                return [spots]

        remaining = sorted(spots, key=lambda s: s['max_intensity'], reverse=True)
        groups: List[List[Dict]] = []
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            group_points = [np.array(seed['center'], dtype=np.float64)]
            grew = True
            while grew:
                grew = False
                centroid = np.mean(group_points, axis=0)
                for candidate in list(remaining):
                    if len(group) >= self.cluster_max_spots:
                        break
                    point = np.array(candidate['center'], dtype=np.float64)
                    if np.linalg.norm(point - centroid) > eps:
                        continue
                    all_points = np.vstack([group_points, point[None, :]])
                    span = all_points.max(axis=0) - all_points.min(axis=0)
                    if float(np.hypot(span[0], span[1])) > self.max_cluster_extent_px:
                        continue
                    group.append(candidate)
                    group_points.append(point)
                    remaining.remove(candidate)
                    grew = True
            groups.append(group)
        return groups
    
    @staticmethod
    def _mean_prominence(spots: List[Dict]) -> float:
        values = [float(s.get('prominence', 0.0)) for s in spots]
        return float(np.mean(values)) if values else 0.0

    def _adaptive_eps(self, points: np.ndarray, base_eps: float) -> float:
        """Slightly adjust eps based on median nearest-neighbor distance"""
        if len(points) < 3:
            return base_eps
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = nn.kneighbors(points)
        median_dist = np.median(distances[:, 1])
        # Clamp eps in a reasonable range
        return float(np.clip(median_dist * 1.5, 20, 80))
    
    def _is_strong_isolated_micro(self, spot: Dict) -> bool:
        """Decide whether a single bright microcalcification should be reported"""
        return (spot['max_intensity'] > 220 and
                spot['local_contrast'] > 30 and
                spot['circularity'] > 0.5)
    
    def _build_micro_cluster(self, cluster_spots: List[Dict],
                             enhanced: np.ndarray) -> Dict:
        cluster_points = np.array([s['center'] for s in cluster_spots])
        
        min_x = int(np.min(cluster_points[:, 0]))
        min_y = int(np.min(cluster_points[:, 1]))
        max_x = int(np.max(cluster_points[:, 0]))
        max_y = int(np.max(cluster_points[:, 1]))
        
        center_x = int(np.mean(cluster_points[:, 0]))
        center_y = int(np.mean(cluster_points[:, 1]))
        
        total_area = sum(s['area'] for s in cluster_spots)
        mean_intensity = float(np.mean([s['mean_intensity'] for s in cluster_spots]))
        max_intensity = float(np.max([s['max_intensity'] for s in cluster_spots]))
        mean_contrast = float(np.mean([s['local_contrast'] for s in cluster_spots]))
        mean_circ = float(np.mean([s['circularity'] for s in cluster_spots]))
        
        cluster_w = max_x - min_x + 1
        cluster_h = max_y - min_y + 1
        cluster_area = cluster_w * cluster_h
        
        # Inter-spot distances (vectorised; a Python double loop over multiple
        # thousand spots previously dominated the runtime)
        mean_inter_dist = 0.0
        std_inter_dist = 0.0
        if len(cluster_points) > 1:
            sample = cluster_points
            if len(sample) > 400:
                # Deterministic subsample keeps the statistics representative
                step = int(np.ceil(len(sample) / 400.0))
                sample = sample[::step]
            diff = sample[:, None, :] - sample[None, :, :]
            pairwise = np.sqrt((diff ** 2).sum(axis=2))
            upper = pairwise[np.triu_indices(len(sample), 1)]
            mean_inter_dist = float(np.mean(upper)) if upper.size else 0.0
            std_inter_dist = float(np.std(upper)) if upper.size else 0.0

        
        features = {
            'num_calcifications': len(cluster_spots),
            'total_area': total_area,
            'cluster_area': cluster_area,
            'density': len(cluster_spots) / (cluster_area + 1),
            'mean_intensity': mean_intensity,
            'max_intensity': max_intensity,
            'mean_contrast': mean_contrast,
            'mean_circularity': mean_circ,
            'mean_inter_distance': mean_inter_dist,
            'std_inter_distance': std_inter_dist,
            'mean_prominence': float(np.mean([s.get('prominence', 0.0) for s in cluster_spots])),
            # Per-spot measurements are kept so the BI-RADS lexicon stage can
            # describe morphology / size variability / distribution from real
            # individual calcification measurements instead of re-deriving them.
            'individual_calcifications': [
                {
                    'center': [float(s['center'][0]), float(s['center'][1])],
                    'area': float(s.get('area', 0.0)),
                    'equivalent_diameter': float(s.get('equivalent_diameter', 0.0)),
                    'circularity': float(s.get('circularity', 0.0)),
                    'mean_intensity': float(s.get('mean_intensity', 0.0)),
                    'max_intensity': float(s.get('max_intensity', 0.0)),
                    'local_contrast': float(s.get('local_contrast', 0.0)),
                }
                for s in cluster_spots
            ],
        }
        
        confidence = self._calculate_micro_confidence(features)
        
        # Add padding to bbox
        pad = 10
        bbox = (max(0, min_x - pad), max(0, min_y - pad),
                (max_x - min_x) + 2 * pad, (max_y - min_y) + 2 * pad)
        
        return {
            'center': (center_x, center_y),
            'bbox': bbox,
            'confidence': confidence,
            'features': features,
            'spots': cluster_spots
        }
    
    def _merge_nearby_clusters(self, clusters: List[Dict],
                               distance_threshold: float) -> List[Dict]:
        """Merge clusters whose centers are within distance_threshold"""
        if len(clusters) < 2:
            return clusters
        
        centers = np.array([c['center'] for c in clusters], dtype=np.float64)
        merged_flag = [False] * len(clusters)
        result = []
        
        for i in range(len(clusters)):
            if merged_flag[i]:
                continue
            group = [clusters[i]]
            merged_flag[i] = True
            for j in range(i + 1, len(clusters)):
                if merged_flag[j]:
                    continue
                if np.linalg.norm(centers[i] - centers[j]) < distance_threshold:
                    group.append(clusters[j])
                    merged_flag[j] = True
            if len(group) == 1:
                result.append(group[0])
            else:
                # Merge group
                all_spots = []
                for c in group:
                    all_spots.extend(c.get('spots', []))
                result.append(self._build_micro_cluster(all_spots, np.zeros((1, 1), dtype=np.uint8)))
        return result
    
    def _process_macros(self, macro_candidates: List[Dict],
                        enhanced: np.ndarray) -> List[Dict]:
        macro_spots = []
        for spot in macro_candidates:
            if spot.get('prominence', 0.0) < self.min_prominence:
                continue
            x, y, w, h = spot['bbox']
            roi = enhanced[y:y + h, x:x + w] if (w > 0 and h > 0) else np.zeros((1, 1))
            
            features = {
                'area': spot['area'],
                'circularity': spot['circularity'],
                'equivalent_diameter': spot['equivalent_diameter'],
                'mean_intensity': spot['mean_intensity'],
                'max_intensity': spot['max_intensity'],
                'mean_intensity_enhanced': spot['mean_intensity_enhanced'],
                'max_intensity_enhanced': spot['max_intensity_enhanced'],
                'local_contrast': spot['local_contrast'],
                'prominence': spot.get('prominence', 0.0),
                'contrast': float(np.std(roi)) if roi.size > 0 else 0.0,
            }
            
            confidence = self._calculate_macro_confidence(features)
            
            macro_spots.append({
                'center': spot['center'],
                'bbox': spot['bbox'],
                'confidence': confidence,
                'features': features
            })
        return macro_spots
    
    # -------------------- Confidence Estimation --------------------
    
    def _calculate_micro_confidence(self, features: Dict) -> float:
        """Improved confidence using multiple features with smoother weighting"""
        score = 0.0
        
        # Number of calcifications (0-0.25)
        n = features['num_calcifications']
        if n >= 5:
            score += 0.25
        elif n >= 3:
            score += 0.18
        elif n >= 2:
            score += 0.10
        
        # Density (0-0.15)
        d = features['density']
        if d > 0.005:
            score += 0.15
        elif d > 0.002:
            score += 0.10
        elif d > 0.0005:
            score += 0.05
        
        # Intensity (0-0.20)
        if features['max_intensity'] > 230:
            score += 0.20
        elif features['max_intensity'] > 200:
            score += 0.15
        elif features['max_intensity'] > 170:
            score += 0.08
        
        # Local contrast (0-0.15)
        if features['mean_contrast'] > 40:
            score += 0.15
        elif features['mean_contrast'] > 25:
            score += 0.10
        elif features['mean_contrast'] > 15:
            score += 0.05
        
        # Spatial compactness (0-0.10): tight clusters are more suspicious
        if features['num_calcifications'] > 1:
            mid = features['mean_inter_distance']
            if 5 < mid < 30:
                score += 0.10
            elif mid < 50:
                score += 0.05
        
        # Circularity bonus (0-0.10)
        if features['mean_circularity'] > 0.7:
            score += 0.10
        elif features['mean_circularity'] > 0.5:
            score += 0.05
        
        # Slight mean intensity contribution
        if features['mean_intensity'] > 180:
            score += 0.05

        # Prominence scaling: a cluster that barely stands out from the
        # parenchyma must not receive a high confidence.
        prominence = float(features.get('mean_prominence', 0.0))
        score *= float(np.clip(prominence / 35.0, 0.35, 1.0))

        return float(min(score, 1.0))
    
    def _calculate_macro_confidence(self, features: Dict) -> float:
        """Improved macro confidence with additional features"""
        score = 0.0
        
        # Circularity (0-0.20)
        if features['circularity'] > 0.8:
            score += 0.20
        elif features['circularity'] > 0.7:
            score += 0.15
        elif features['circularity'] > 0.5:
            score += 0.08
        
        # Size in valid range (0-0.15)
        a = features['area']
        if self.macro_size_range[0] ** 2 < a < self.macro_size_range[1] ** 2:
            score += 0.15
        elif a <= self.macro_size_range[1] ** 2 * 1.2:
            score += 0.08
        
        # Max intensity on enhanced (0-0.25)
        if features['max_intensity_enhanced'] > 245:
            score += 0.25
        elif features['max_intensity_enhanced'] > 230:
            score += 0.18
        elif features['max_intensity_enhanced'] > 210:
            score += 0.10
        
        # Local contrast (0-0.20) - critical for reducing false positives
        if features['local_contrast'] > 50:
            score += 0.20
        elif features['local_contrast'] > 30:
            score += 0.15
        elif features['local_contrast'] > 15:
            score += 0.08
        
        # ROI contrast (0-0.10)
        if features['contrast'] > 50:
            score += 0.10
        elif features['contrast'] > 30:
            score += 0.05
        
        # Mean intensity (0-0.10)
        if features['mean_intensity_enhanced'] > 200:
            score += 0.10
        elif features['mean_intensity_enhanced'] > 170:
            score += 0.05
        
        return float(min(score, 1.0))
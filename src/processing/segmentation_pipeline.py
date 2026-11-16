"""Advanced breast segmentation pipeline."""
import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from scipy import ndimage

from config import Config

logger = logging.getLogger("MammographyCAD.segmentation")


class SegmentationPipeline:
    """
    Enhanced multi-stage breast segmentation pipeline:
    1. Adaptive Thresholding
    2. Multi-Level Otsu
    3. Morphological Closing
    4. Morphological Opening
    5. Largest Connected Component
    6. Convex Hull
    7. Boundary Smoothing
    8. Edge Refinement
    
    Optimized for:
    - Dense breast tissue
    - Low contrast images
    - Label and artifact removal
    """

    def __init__(self, settings: Dict[str, Any] = None):
        s = settings or {}
        self.block_size = s.get("block_size", Config.SEG_ADAPTIVE_BLOCK_SIZE)
        self.adaptive_c = s.get("adaptive_c", Config.SEG_ADAPTIVE_C)
        self.close_kernel = s.get("close_kernel", Config.SEG_MORPH_CLOSE_KERNEL)
        self.open_kernel = s.get("open_kernel", Config.SEG_MORPH_OPEN_KERNEL)
        self.smooth_kernel = s.get("smooth_kernel", Config.SEG_SMOOTH_KERNEL)
        
        # Enhanced parameters for dense breasts and low contrast
        self.dense_breast_support = s.get("dense_breast_support", True)
        self.artifact_removal = s.get("artifact_removal", True)
        self.min_area_ratio = s.get("min_area_ratio", 0.05)  # Minimum 5% of image
        self.max_area_ratio = s.get("max_area_ratio", 0.90)  # Maximum 90% of image

    def segment(self, gray: np.ndarray) -> Dict[str, Any]:
        """Run full segmentation pipeline and return mask, contour, metrics."""
        logger.debug("Starting enhanced segmentation pipeline")

        # Preprocessing: enhance contrast for low contrast images
        preprocessed = self._preprocess_image(gray)

        # Stage 1: Adaptive Thresholding
        adaptive_mask = self._adaptive_thresholding(preprocessed)

        # Stage 2: Multi-Level Otsu
        otsu_mask = self._multi_level_otsu(preprocessed)

        # Combine masks (use OR to preserve details)
        combined = cv2.bitwise_or(adaptive_mask, otsu_mask)

        # Remove labels and artifacts before morphological operations
        if self.artifact_removal:
            combined = self._remove_labels_and_artifacts(combined, gray.shape)

        # Stage 3: Morphological Closing
        closed = self._morphological_closing(combined)

        # Stage 4: Morphological Opening
        opened = self._morphological_opening(closed)

        # Stage 5: Largest Connected Component
        mask = self._largest_component(opened)

        # Stage 6: Convex Hull
        mask = self._apply_convex_hull(mask)

        # Stage 7: Boundary Smoothing
        mask = self._boundary_smoothing(mask)

        # Stage 8: Edge Refinement
        mask = self._edge_refinement(preprocessed, mask)

        # Final validation and cleanup
        mask = self._validate_mask(mask, gray.shape)

        # Hard sanity check: a mask that covers (almost) the whole frame is not a
        # breast segmentation.  Fall back to a plain Otsu foreground, and if that
        # is still implausible report the stage as invalid instead of returning a
        # rectangle with a flattering confidence.
        total_pixels = float(gray.shape[0] * gray.shape[1])
        area_ratio = float(np.sum(mask > 0)) / total_pixels
        if area_ratio > self.max_area_ratio:
            fallback = self._otsu_foreground_fallback(preprocessed)
            fallback_ratio = float(np.sum(fallback > 0)) / total_pixels
            if fallback_ratio <= self.max_area_ratio and fallback_ratio >= self.min_area_ratio:
                logger.info(
                    "Segmentation fallback applied (area %.1f%% -> %.1f%%)",
                    area_ratio * 100, fallback_ratio * 100,
                )
                mask = fallback
                area_ratio = fallback_ratio

        valid = self.min_area_ratio <= area_ratio <= self.max_area_ratio

        # Extract contour
        contour = self._extract_contour(mask)

        # Compute metrics
        metrics = self._compute_metrics(mask, gray)

        # Compute confidence
        confidence = self._compute_confidence(metrics, gray)

        logger.info(
            "Segmentation complete: confidence=%.2f, area=%.1f%%, compactness=%.3f",
            confidence,
            metrics["area_percent"],
            metrics["compactness"]
        )

        warning = None
        if not valid:
            # Never report a high confidence for a mask that failed validation.
            confidence = min(float(confidence), 0.15)
            warning = (
                "Breast segmentation did not produce a plausible mask "
                f"(area {area_ratio * 100:.1f}% of the image); segmentation-based "
                "metrics should not be trusted for this image."
            )
            logger.warning(warning)

        return {
            "mask": mask,
            "contour": contour,
            "confidence": confidence,
            "metrics": metrics,
            "overlay_preview": self._create_overlay(gray, mask, contour),
            "valid": bool(valid),
            "area_ratio": round(area_ratio, 4),
            "warning": warning,
        }

    def _otsu_foreground_fallback(self, image: np.ndarray) -> np.ndarray:
        """Plain Otsu foreground mask used when the pipeline overshoots."""
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        )
        mask = self._largest_component(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
            mask = filled
        return mask

    def _preprocess_image(self, gray: np.ndarray) -> np.ndarray:
        """Enhance contrast for low contrast images and dense breasts."""
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
        
        logger.debug("Image preprocessing completed")
        return denoised

    def _adaptive_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Stage 1: Adaptive thresholding with automatic block size."""
        block = self.block_size if self.block_size % 2 == 1 else self.block_size + 1
        
        # Ensure block size is appropriate for image size
        min_dim = min(image.shape)
        if block > min_dim // 4:
            block = (min_dim // 4) | 1  # Make it odd
        
        adaptive = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            self.adaptive_c,
        )
        
        logger.debug("Adaptive thresholding completed with block_size=%d", block)
        return adaptive

    def _multi_level_otsu(self, image: np.ndarray) -> np.ndarray:
        """Stage 2: Multi-level Otsu thresholding for dense breast support."""
        try:
            # Use 3 classes for better separation of dense tissue
            try:
                thresholds = threshold_multiotsu(image, classes=3, nbins=256)
            except TypeError:  # scikit-image < 0.19 used `bins`
                thresholds = threshold_multiotsu(image, classes=3, bins=256)
            
            # For dense breasts, use lower threshold
            otsu_mask = np.zeros_like(image, dtype=np.uint8)
            if self.dense_breast_support:
                otsu_mask[image > thresholds[0]] = 255
            else:
                otsu_mask[image > thresholds[1]] = 255
                
            logger.debug("Multi-Otsu thresholds: %s", thresholds)
            
        except (ValueError, RuntimeError) as e:
            logger.warning("Multi-Otsu failed, falling back to standard Otsu: %s", e)
            _, otsu_mask = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        
        return otsu_mask

    def _remove_labels_and_artifacts(
        self, binary: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Remove text labels and small artifacts before main processing."""
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        if num_labels < 2:
            return binary
        
        cleaned = np.zeros_like(binary)
        total_pixels = image_shape[0] * image_shape[1]
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT + 1]
            
            # Remove components that are likely labels or artifacts
            area_ratio = area / total_pixels
            aspect_ratio = w / (h + 1e-6)
            
            # Skip if:
            # 1. Too small (< 0.5% of image)
            # 2. Near borders (likely labels)
            # 3. Extreme aspect ratio (likely text)
            if area_ratio < 0.005:
                continue
            
            border_margin = min(image_shape) * 0.02
            if (x < border_margin or y < border_margin or 
                x + w > image_shape[1] - border_margin or 
                y + h > image_shape[0] - border_margin):
                if area_ratio < 0.05:  # Small border components are likely labels
                    continue
            
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                if area_ratio < 0.1:  # Elongated small components are likely text
                    continue
            
            # Keep this component
            cleaned[labels == i] = 255
        
        logger.debug("Removed %d artifacts/labels", num_labels - 1 - np.sum(cleaned > 0))
        return cleaned

    def _morphological_closing(self, binary: np.ndarray) -> np.ndarray:
        """Stage 3: Morphological closing to fill gaps."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.close_kernel, self.close_kernel)
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        logger.debug("Morphological closing completed")
        return closed

    def _morphological_opening(self, binary: np.ndarray) -> np.ndarray:
        """Stage 4: Morphological opening to remove noise."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.open_kernel, self.open_kernel)
        )
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        logger.debug("Morphological opening completed")
        return opened

    def _largest_component(self, binary: np.ndarray) -> np.ndarray:
        """Stage 5: Extract largest connected component (breast region)."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        if num_labels < 2:
            logger.warning("No connected components found")
            return binary
        
        # Find largest component (excluding background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        
        mask = (labels == largest_idx).astype(np.uint8) * 255
        
        logger.debug(
            "Largest component: %d pixels (%.1f%%)",
            areas[largest_idx - 1],
            (areas[largest_idx - 1] / binary.size) * 100
        )
        return mask

    def _apply_convex_hull(self, mask: np.ndarray) -> np.ndarray:
        """Stage 6: Apply convex hull for smooth breast boundary."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("No contours found for convex hull")
            return mask
        
        # Get largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Compute convex hull
        hull = cv2.convexHull(main_contour)
        
        # Create hull mask
        hull_mask = np.zeros_like(mask)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)
        
        # Combine with original to preserve internal details
        # Use hull as outer boundary, keep original internal structure
        result = cv2.bitwise_and(mask, hull_mask)
        
        logger.debug("Convex hull applied")
        return result

    def _boundary_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """Stage 7: Smooth boundary using Gaussian blur."""
        smooth_k = self.smooth_kernel if self.smooth_kernel % 2 == 1 else self.smooth_kernel + 1
        
        # Apply Gaussian smoothing
        smoothed = cv2.GaussianBlur(mask, (smooth_k, smooth_k), 0)
        
        # Re-threshold to binary
        _, smoothed_binary = cv2.threshold(smoothed, 128, 255, cv2.THRESH_BINARY)
        
        logger.debug("Boundary smoothing completed")
        return smoothed_binary

    def _edge_refinement(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Stage 8: Refine edges using gradient information."""
        # Compute edges using Canny
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate edges slightly to create a band
        edge_band = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        
        # Create distance transform of mask
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Find mask boundary region
        _, boundary = cv2.threshold(dist_transform, 10, 255, cv2.THRESH_BINARY_INV)
        boundary = cv2.bitwise_and(boundary.astype(np.uint8), mask)
        
        # Refine boundary using edge information
        refined_boundary = cv2.bitwise_and(boundary, edge_band)
        
        # Subtract refined edges from mask to create cleaner boundary
        # Only apply where edges are strong
        result = mask.copy()
        edge_strength = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        adjustment = cv2.bitwise_and(refined_boundary, edge_strength)
        
        # Apply morphological gradient for final refinement
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
        result = cv2.bitwise_or(mask, result)
        
        # Fill any holes created
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(result, contours, -1, 255, -1)
        
        logger.debug("Edge refinement completed")
        return result

    def _validate_mask(self, mask: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Validate and correct mask based on expected breast characteristics."""
        total_pixels = image_shape[0] * image_shape[1]
        mask_area = np.sum(mask > 0)
        area_ratio = mask_area / total_pixels
        
        # Check if area is within acceptable range
        if area_ratio < self.min_area_ratio:
            logger.warning(
                "Mask area too small (%.1f%%), may indicate segmentation failure",
                area_ratio * 100
            )
        elif area_ratio > self.max_area_ratio:
            logger.warning(
                "Mask area too large (%.1f%%), may include background",
                area_ratio * 100
            )
            # Apply more aggressive opening to remove background
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = self._largest_component(mask)
        
        # Fill any internal holes
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1:  # Has parent (is a hole)
                    cv2.drawContours(mask, contours, i, 255, -1)
        
        logger.debug("Mask validation completed")
        return mask

    def _extract_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract breast contour from mask."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("No contour found in mask")
            return None
        
        # Return largest contour
        main_contour = max(contours, key=cv2.contourArea)
        logger.debug("Contour extracted with %d points", len(main_contour))
        return main_contour

    def _compute_metrics(self, mask: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive segmentation metrics."""
        # Basic metrics
        area = float(np.sum(mask > 0))
        total = float(mask.size)
        area_percent = (area / total) * 100
        
        # Contour metrics
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        perimeter = 0.0
        solidity = 0.0
        circularity = 0.0
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(main_contour, True))
            
            # Solidity: ratio of contour area to convex hull area
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
            
            # Circularity (compactness)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # Intensity metrics
        roi = gray[mask > 0]
        mean_intensity = float(np.mean(roi)) if roi.size > 0 else 0.0
        std_intensity = float(np.std(roi)) if roi.size > 0 else 0.0
        
        # Contrast metric
        contrast = std_intensity / (mean_intensity + 1e-6)
        
        metrics = {
            "area_pixels": area,
            "area_percent": area_percent,
            "perimeter_pixels": perimeter,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "contrast": contrast,
            "compactness": circularity,
            "solidity": solidity,
            "aspect_ratio": self._compute_aspect_ratio(mask),
        }
        
        logger.debug("Metrics computed: %s", metrics)
        return metrics

    def _compute_aspect_ratio(self, mask: np.ndarray) -> float:
        """Compute aspect ratio of bounding box."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return float(w) / (h + 1e-6)

    def _compute_confidence(self, metrics: Dict[str, float], gray: np.ndarray) -> float:
        """Compute confidence score based on multiple factors."""
        score = 0.0
        
        # Area check (15-85% of image is reasonable for breast)
        area_pct = metrics["area_percent"]
        if 15 < area_pct < 85:
            score += 0.25
        elif 10 < area_pct < 90:
            score += 0.15
        
        # Compactness check (breast should be reasonably compact)
        compactness = metrics["compactness"]
        if compactness > 0.3:
            score += 0.20
        elif compactness > 0.15:
            score += 0.10
        
        # Solidity check (breast should be fairly solid)
        solidity = metrics.get("solidity", 0)
        if solidity > 0.85:
            score += 0.20
        elif solidity > 0.75:
            score += 0.10
        
        # Perimeter check
        if metrics["perimeter_pixels"] > 100:
            score += 0.15
        
        # Contrast check (good separation from background)
        contrast = metrics.get("contrast", 0)
        if contrast > 0.3:
            score += 0.20
        elif contrast > 0.15:
            score += 0.10
        
        # Aspect ratio check (breast shouldn't be too elongated)
        aspect = metrics.get("aspect_ratio", 1.0)
        if 0.5 < aspect < 2.0:
            score += 0.10
        
        confidence = min(score, 1.0)
        logger.debug("Confidence score: %.3f", confidence)
        return confidence

    def _create_overlay(
        self, gray: np.ndarray, mask: np.ndarray, contour: Optional[np.ndarray]
    ) -> np.ndarray:
        """Create visualization overlay with mask and contour."""
        # Convert grayscale to BGR
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Create colored mask (green)
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 0] = (0, 200, 0)
        
        # Blend with original image
        cv2.addWeighted(colored_mask, 0.3, overlay, 0.7, 0, overlay)
        
        # Draw contour if available
        if contour is not None:
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
        
        # Add confidence and metrics text
        logger.debug("Overlay preview created")
        return overlay
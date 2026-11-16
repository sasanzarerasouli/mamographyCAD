import cv2
import numpy as np
from typing import Dict, Tuple, Any
import logging
from skimage.feature import graycomatrix, graycoprops

logger = logging.getLogger("MammographyCAD")

# ACR BI-RADS 5th edition breast density categories
ACR_LABELS = {
    "A": "A - Almost entirely fatty",
    "B": "B - Scattered areas of fibroglandular density",
    "C": "C - Heterogeneously dense",
    "D": "D - Extremely dense",
}


class BreastDensityClassifier:
    """
    Classifies breast density according to BI-RADS standards:
    - Density A: Almost entirely fatty
    - Density B: Scattered areas of fibroglandular density
    - Density C: Heterogeneously dense
    - Density D: Extremely dense
    """

    def __init__(self):
        self.last_method = "otsu"
        self.last_dense_posterior = None
        # BI-RADS Density Thresholds (Percentage range of dense tissue)
        self.density_thresholds = {
            'A': (0.0, 25.0),
            'B': (25.0, 50.0),
            'C': (50.0, 75.0),
            'D': (75.0, 100.0)
        }

    def classify(self, image: np.ndarray, breast_mask: np.ndarray) -> Dict[str, Any]:
        """
        Main method to classify breast density.

        Args:
            image (np.ndarray): Enhanced/Preprocessed mammogram image (grayscale).
            breast_mask (np.ndarray): Binary mask representing the breast region.

        Returns:
            Dict: Containing 'category', 'statistics', and 'heatmap'.
        """
        logger.info("Starting Breast Density Classification...")

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            image_normalized = image.copy()

        # 1. Fibroglandular Estimation & Dense Area Percentage
        dense_mask, density_percentage = self._estimate_fibroglandular_tissue(image_normalized, breast_mask)

        # 2. Extract Texture Features (GLCM & Statistical)
        texture_features = self._extract_texture_features(image_normalized, breast_mask)

        # 3. Predict Density Category (Density A, B, C, D)
        category, confidence = self._predict_density_category(density_percentage, texture_features)

        # 4. Generate Density Heatmap
        heatmap = self._generate_heatmap(image_normalized, dense_mask, breast_mask)

        # Compile Statistics
        breast_area_pixels = int(np.count_nonzero(breast_mask))
        dense_area_pixels = int(np.count_nonzero(dense_mask))

        acr_category = self.to_acr_category(category)

        statistics = {
            'dense_area_percentage': round(float(density_percentage), 2),
            'breast_area_pixels': breast_area_pixels,
            'dense_area_pixels': dense_area_pixels,
            'confidence': round(float(confidence), 2),
            'texture_features': texture_features,
            'acr_category': acr_category,
            'dense_fraction_method': getattr(self, 'last_method', 'otsu'),
        }

        logger.info(
            "Classification completed: %s / ACR %s (dense area %.2f%%)",
            category, acr_category, density_percentage,
        )

        return {
            'category': category,
            'acr_category': acr_category,
            'category_label': ACR_LABELS.get(acr_category, category),
            'statistics': statistics,
            'heatmap': heatmap,
            'method': (
                'two-component Gaussian mixture (fibroglandular vs fat) with '
                'texture refinement'
                if getattr(self, 'last_method', '') == 'gmm2'
                else 'histogram (Otsu) dense-area estimate with texture refinement'
            ),
            'disclaimer': (
                'Software-estimated breast density category; visual or '
                'volumetric ACR assessment by a radiologist remains the '
                'reference standard.'
            ),
        }

    @staticmethod
    def to_acr_category(category: str) -> str:
        """Map 'Density A'..'Density D' (or 'A'..'D') to the ACR letter."""
        if not category:
            return 'B'
        text = str(category).strip().upper()
        if text.startswith('DENSITY'):
            text = text.replace('DENSITY', '').strip()
        return text[:1] if text[:1] in ACR_LABELS else 'B'


    def _estimate_fibroglandular_tissue(self, image: np.ndarray, breast_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Estimate the fibroglandular (dense) area of the breast.

        A two-component Gaussian mixture is fitted to the intensities inside the
        breast: the component with the higher mean is the fibroglandular tissue,
        the lower one is fat.  The dense fraction is the mean posterior of the
        dense component, which behaves far better than a single Otsu threshold
        (Otsu systematically under-estimates because the dense peak is the
        smaller of the two modes).  Otsu is kept as the fallback when
        scikit-learn is unavailable.
        """
        breast_pixels = image[breast_mask > 0]
        if breast_pixels.size == 0:
            return np.zeros_like(breast_mask), 0.0

        dense_mask = np.zeros_like(breast_mask)
        self.last_method = "otsu"

        posterior_dense = None
        try:
            from sklearn.mixture import GaussianMixture

            values = breast_pixels.reshape(-1, 1).astype(np.float64)
            if values.shape[0] > 2000:
                sample = values[:: max(1, values.shape[0] // 20000)]
            else:  # pragma: no cover - tiny images
                sample = values
            model = GaussianMixture(
                n_components=2, covariance_type="full", random_state=0, max_iter=200
            )
            model.fit(sample)
            means = model.means_.ravel()
            dense_component = int(np.argmax(means))
            posterior = model.predict_proba(values)[:, dense_component]
            # The posterior has one entry per breast pixel: scatter it back into
            # full-image coordinates (reshaping to the image shape raises
            # ValueError because the background is excluded).
            dense_area = np.zeros(image.shape[:2], dtype=np.float64)
            dense_area[breast_mask > 0] = posterior
            posterior_dense = dense_area
            dense_area_count = float(posterior.sum())
            total = float(posterior.size)
            percentage = 100.0 * dense_area_count / total if total > 0 else 0.0
            dense_mask[(breast_mask > 0) & (dense_area >= 0.5)] = 255
            self.last_method = "gmm2"
        except Exception:  # pragma: no cover - sklearn missing / degenerate
            otsu_value, _ = cv2.threshold(
                breast_pixels.reshape(-1, 1).astype(np.uint8),
                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            threshold = float(otsu_value)
            dense_mask[(breast_mask > 0) & (image >= threshold)] = 255
            total_breast_pixels = np.count_nonzero(breast_mask)
            percentage = (
                100.0 * np.count_nonzero(dense_mask) / total_breast_pixels
                if total_breast_pixels > 0
                else 0.0
            )

        # Clean isolated pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dense_mask = cv2.morphologyEx(dense_mask, cv2.MORPH_OPEN, kernel)
        dense_mask = cv2.bitwise_and(dense_mask, (breast_mask > 0).astype(np.uint8) * 255)

        self.last_dense_posterior = posterior_dense
        return dense_mask, float(percentage)

    def _extract_texture_features(self, image: np.ndarray, breast_mask: np.ndarray) -> Dict[str, float]:
        """
        Extracts GLCM (Gray-Level Co-occurrence Matrix) and statistical texture features.
        """
        roi = image[breast_mask > 0]
        if len(roi) == 0:
            return {'contrast': 0.0, 'homogeneity': 0.0, 'energy': 0.0, 'correlation': 0.0, 'std_dev': 0.0}

        std_dev = float(np.std(roi))

        # Crop ROI bounding box for GLCM calculation
        x, y, w, h = cv2.boundingRect(breast_mask)
        cropped_img = image[y:y+h, x:x+w]
        cropped_mask = breast_mask[y:y+h, x:x+w]

        # Mask background
        cropped_img_masked = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)

        # Quantize image to 16 gray levels to optimize GLCM
        quantized = (cropped_img_masked // 16).astype(np.uint8)

        # Compute GLCM
        glcm = graycomatrix(quantized, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=16, symmetric=True, normed=True)

        contrast = float(np.mean(graycoprops(glcm, 'contrast')))
        homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
        energy = float(np.mean(graycoprops(glcm, 'energy')))
        correlation = float(np.mean(graycoprops(glcm, 'correlation')))

        return {
            'contrast': round(contrast, 4),
            'homogeneity': round(homogeneity, 4),
            'energy': round(energy, 4),
            'correlation': round(correlation, 4),
            'std_dev': round(std_dev, 2)
        }

    def _predict_density_category(self, density_percentage: float, texture_features: Dict[str, float]) -> Tuple[str, float]:
        """
        Predicts BI-RADS Density category based on tissue percentage and refines using texture features.
        """
        # Primary BI-RADS classification based on area percentage
        if density_percentage < 25.0:
            category = "Density A"
            confidence = 0.85 + (25.0 - density_percentage) / 25.0 * 0.15
        elif 25.0 <= density_percentage < 50.0:
            category = "Density B"
            confidence = 0.80
        elif 50.0 <= density_percentage < 75.0:
            category = "Density C"
            confidence = 0.80
        else:
            category = "Density D"
            confidence = 0.85 + (density_percentage - 75.0) / 25.0 * 0.15

        # Refinement using texture heterogeneity (Std Dev / Contrast)
        std_dev = texture_features.get('std_dev', 0)
        if category == "Density B" and density_percentage > 42.0 and std_dev > 38.0:
            category = "Density C"
        elif category == "Density C" and density_percentage < 58.0 and std_dev < 22.0:
            category = "Density B"

        return category, min(confidence, 1.0)

    def _generate_heatmap(self, image: np.ndarray, dense_mask: np.ndarray, breast_mask: np.ndarray) -> np.ndarray:
        """
        Generates a continuous RGB density heatmap visualization overlaid on the mammogram.
        """
        # Apply Gaussian blur to create a smooth density visual transition
        blurred_mask = cv2.GaussianBlur(dense_mask, (31, 31), 0)

        # Apply JET Colormap for heatmap effect
        heatmap_color = cv2.applyColorMap(blurred_mask, cv2.COLORMAP_JET)

        # Convert original grayscale image to 3 channels
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Blend original image and color heatmap
        alpha = 0.45
        blended = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)

        # Mask background to keep only breast region visible
        result_heatmap = cv2.bitwise_and(blended, blended, mask=breast_mask)

        return result_heatmap
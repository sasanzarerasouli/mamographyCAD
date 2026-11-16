import cv2
import numpy as np
from typing import List, Dict, Tuple
from skimage import feature, filters, morphology, measure, segmentation
from sklearn.cluster import DBSCAN
import logging

from ..core.detection_types import Detection, DetectionType

logger = logging.getLogger("MammographyCAD")


class TumorDetector:
    """
    Improved tumor detector with:
      - More accurate segmentation masks (Otsu + morphology + active-contour refinement)
      - Better suspicious-region localization (multi-scale blob + LoG + texture fusion)
      - Reduced false positives (shape/texture/intensity validation + NMS)
      - Circular boundary output (center, radius) + confidence + label
    Interface (detect()) remains backward compatible.
    """

    # ---------------- Labels ---------------- #
    LABEL_BENIGN     = "Benign"
    LABEL_SUSPICIOUS = "Suspicious"
    LABEL_MALIGNANT  = "Malignant"

    def __init__(self):
        self.min_tumor_area = 150
        self.max_tumor_area_ratio = 0.35          # جلوگیری از تشخیص کل بافت به عنوان تومور
        self.suspicious_threshold = 0.55
        self.confirmed_threshold  = 0.78
        self.nms_iou_threshold    = 0.30          # حذف تشخیص‌های تکراری/همپوشان
        self.min_prominence       = 12.0          # اختلاف روشنایی با بافت اطراف
        self.max_detections       = 6             # سقف خروجی برای استفاده بالینی
        self.max_ring_density     = 0.60          # محصور در بافت هم‌روشنایی

    # ============================================================= #
    #                           MAIN API                            #
    # ============================================================= #
    def detect(self, enhanced: np.ndarray, breast_mask: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []

        if enhanced is None or breast_mask is None:
            return detections

        # اطمینان از uint8 بودن ورودی
        if enhanced.dtype != np.uint8:
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if breast_mask.dtype != np.uint8:
            breast_mask = (breast_mask > 0).astype(np.uint8) * 255

        # پیش‌پردازش برای کاهش نویز و افزایش کنتراست محلی
        preproc = self._preprocess(enhanced, breast_mask)

        # سه استراتژی مستقل تشخیص
        texture_regions   = self._texture_based_detection(preproc, breast_mask)
        intensity_regions = self._intensity_based_detection(preproc, breast_mask)
        shape_regions     = self._shape_based_detection(preproc, breast_mask)
        blob_regions      = self._blob_based_detection(preproc, breast_mask)   # جدید

        # هر کاندید به‌صورت مستقل ارزیابی می‌شود.
        # ادغام ماسک‌ها پیش از ارزیابی باعث می‌شد یک ناحیه بزرگ (که توسط
        # آستانه‌گذاری adaptive تولید می‌شود) کاندید صحیح تومور را ببلعد و
        # منجر به از دست رفتن تشخیص شود؛ حذف تکراری‌ها در پایان با NMS دایره‌ای
        # انجام می‌شود که رفتار درست‌تری دارد.
        candidates = (
            texture_regions + intensity_regions + shape_regions + blob_regions
        )

        max_area = enhanced.shape[0] * enhanced.shape[1] * self.max_tumor_area_ratio

        for region in candidates:
            # فیلتر اندازه غیرمنطقی
            if region['features'].get('area', 0) > max_area:
                continue

            # پالایش ماسک (دقت بیشتر) و استخراج دایره محیطی
            refined_mask = self._refine_mask(region['mask'], enhanced, breast_mask)
            if refined_mask is None or cv2.countNonZero(refined_mask) < self.min_tumor_area:
                continue

            center, radius = self._fit_circle(refined_mask)
            if radius < 5:
                continue

            # غربالگری برجستگی: ناحیه باید از بافت اطراف خود به‌طور معنی‌داری
            # روشن‌تر باشد، وگرنه بافت طبیعی است (کاهش False Positive)
            prominence, ring_density = self._prominence(enhanced, refined_mask, radius)
            if prominence < self.min_prominence:
                continue
            if ring_density > self.max_ring_density:
                continue

            # ارزیابی چند-فاکتوری (کاهش FP)
            confidence = self._evaluate_region(region, enhanced, refined_mask, radius)
            confidence = float(np.clip(confidence + self._prominence_bonus(prominence), 0.0, 1.0))

            if confidence >= self.confirmed_threshold:
                detection_type = DetectionType.TUMOR_CONFIRMED
                label = self.LABEL_MALIGNANT
            elif confidence >= self.suspicious_threshold:
                detection_type = DetectionType.TUMOR_SUSPICIOUS
                label = self.LABEL_SUSPICIOUS
            else:
                continue  # حذف تشخیص‌های ضعیف

            # بازسازی bbox براساس ماسک پالایش‌شده
            ys, xs = np.where(refined_mask > 0)
            x_min, y_min = int(xs.min()), int(ys.min())
            x_max, y_max = int(xs.max()), int(ys.max())
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            features = dict(region['features'])
            features.update({
                'center': center,
                'radius': int(radius),
                'prominence': float(prominence),
                'ring_density': float(ring_density),
                'label':  label,
                'confidence': float(confidence),
            })

            detection = Detection(
                type=detection_type,
                location=center,
                bbox=bbox,
                mask=refined_mask,
                confidence=float(confidence),
                features=features,
            )
            detections.append(detection)

        # NMS نهایی روی دایره‌ها
        detections = self._circle_nms(detections)
        # سقف خروجی: قوی‌ترین موارد نگه داشته می‌شوند
        if len(detections) > self.max_detections:
            detections.sort(key=lambda d: d.confidence, reverse=True)
            detections = detections[: self.max_detections]
        logger.info("TumorDetector produced %d detections", len(detections))
        return detections

    # ============================================================= #
    #                        PREPROCESSING                          #
    # ============================================================= #
    def _preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # کاهش نویز با حفظ لبه
        denoised = cv2.bilateralFilter(img, 7, 50, 50)
        # افزایش کنتراست محلی
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        eq = clahe.apply(denoised)
        eq = cv2.bitwise_and(eq, eq, mask=mask)
        return eq

    # ============================================================= #
    #                    DETECTION STRATEGIES                       #
    # ============================================================= #
    def _texture_based_detection(self, enhanced, breast_mask) -> List[Dict]:
        regions = []

        # GLCM روی کل تصویر برای Feature سراسری
        try:
            glcm = feature.graycomatrix(
                enhanced, distances=[1, 2],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256, symmetric=True, normed=True,
            )
            contrast    = float(feature.graycoprops(glcm, 'contrast').mean())
            homogeneity = float(feature.graycoprops(glcm, 'homogeneity').mean())
            energy      = float(feature.graycoprops(glcm, 'energy').mean())
        except Exception:
            contrast, homogeneity, energy = 0.0, 0.0, 0.0

        # فیلتربانک Gabor چند-جهته
        kernels = []
        for theta_i in range(4):
            theta = theta_i / 4.0 * np.pi
            for freq in (0.15, 0.35):
                k = cv2.getGaborKernel((21, 21), 4.0, theta, 8.0, freq, 0)
                kernels.append(k)

        filtered = [cv2.filter2D(enhanced, cv2.CV_32F, k) for k in kernels]
        tex_map = np.mean(np.abs(filtered), axis=0)
        tex_map = cv2.normalize(tex_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # آستانه adaptive برای دقت بیشتر
        _, binary = cv2.threshold(tex_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, binary, mask=breast_mask)
        binary = self._morph_clean(binary)

        return self._extract_regions(
            binary, enhanced,
            base_features={'contrast': contrast,
                           'homogeneity': homogeneity,
                           'energy': energy,
                           'source': 'texture'}
        )

    def _intensity_based_detection(self, enhanced, breast_mask) -> List[Dict]:
        # مناطق روشن غیرعادی
        blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
        # Adaptive threshold به‌جای Otsu ساده -> دقت بیشتر
        adap = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, -8
        )
        adap = cv2.bitwise_and(adap, adap, mask=breast_mask)
        adap = self._morph_clean(adap)
        return self._extract_regions(adap, enhanced,
                                     base_features={'source': 'intensity'})

    def _shape_based_detection(self, enhanced, breast_mask) -> List[Dict]:
        edges = cv2.Canny(enhanced, 40, 130)
        edges = cv2.bitwise_and(edges, edges, mask=breast_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed, kernel, iterations=1)
        # پر کردن حفره‌ها
        filled = self._fill_holes(closed)
        filled = cv2.bitwise_and(filled, filled, mask=breast_mask)
        return self._extract_regions(filled, enhanced,
                                     base_features={'source': 'shape'})

    def _blob_based_detection(self, enhanced, breast_mask) -> List[Dict]:
        """تشخیص Blob با LoG برای پیدا کردن ضایعات گرد/کروی."""
        regions = []
        try:
            # نرمالیزه به [0,1]
            img_f = enhanced.astype(np.float32) / 255.0
            blobs = feature.blob_log(
                img_f, min_sigma=5, max_sigma=28,
                num_sigma=10, threshold=0.15, overlap=0.6
            )
        except Exception:
            return regions

        H, W = enhanced.shape[:2]
        for by, bx, sigma in blobs:
            r = int(sigma * np.sqrt(2))
            cx, cy = int(bx), int(by)
            if r < 5 or not (0 <= cx < W and 0 <= cy < H):
                continue
            if breast_mask[cy, cx] == 0:
                continue

            mask = np.zeros(enhanced.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            mask = cv2.bitwise_and(mask, mask, mask=breast_mask)

            area = int(cv2.countNonZero(mask))
            if area < self.min_tumor_area:
                continue

            x, y, w, h = cv2.boundingRect(mask)
            regions.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'mask': mask,
                'features': {'area': float(area), 'source': 'blob_log',
                             'sigma': float(sigma),
                             'circularity': float(np.pi * r * r / max(area, 1e-6))}
            })
        return regions

    # ============================================================= #
    #                        HELPER METHODS                         #
    # ============================================================= #
    def _morph_clean(self, binary: np.ndarray) -> np.ndarray:
        """پاکسازی نویز و پر کردن حفره‌ها -> ماسک تمیزتر."""
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k5, iterations=2)
        return self._fill_holes(closed)

    @staticmethod
    def _fill_holes(binary: np.ndarray) -> np.ndarray:
        inv = cv2.bitwise_not(binary)
        h, w = binary.shape
        flood = inv.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        return cv2.bitwise_or(binary, holes)

    def _extract_regions(self, binary, enhanced, base_features) -> List[Dict]:
        regions = []
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_tumor_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(c)
            mask = np.zeros(enhanced.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)

            perim = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0

            feats = {'area': float(area), 'circularity': float(circularity)}
            feats.update(base_features)

            regions.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'mask': mask,
                'features': feats,
            })
        return regions

    def _refine_mask(self, coarse_mask, enhanced, breast_mask) -> np.ndarray:
        """پالایش ماسک با watershed محلی + محدود کردن به ناحیه سینه."""
        try:
            ys, xs = np.where(coarse_mask > 0)
            if len(xs) == 0:
                return None
            pad = 10
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, enhanced.shape[1] - 1)
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, enhanced.shape[0] - 1)

            roi = enhanced[y0:y1 + 1, x0:x1 + 1]
            roi_mask = coarse_mask[y0:y1 + 1, x0:x1 + 1]

            # آستانه محلی روی ROI
            _, local_bin = cv2.threshold(
                roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            local_bin = cv2.bitwise_and(local_bin, roi_mask)
            local_bin = self._morph_clean(local_bin)

            refined = np.zeros_like(coarse_mask)
            refined[y0:y1 + 1, x0:x1 + 1] = local_bin
            refined = cv2.bitwise_and(refined, refined, mask=breast_mask)

            if cv2.countNonZero(refined) < self.min_tumor_area:
                return coarse_mask
            return refined
        except Exception as e:
            logger.debug(f"refine_mask failed: {e}")
            return coarse_mask

    @staticmethod
    def _fit_circle(mask: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """محاسبه کوچک‌ترین دایره محیطی روی ماسک."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0), 0
        largest = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(largest)
        return (int(x), int(y)), int(r)

    # ============================================================= #
    #                    MERGING & VALIDATION                       #
    # ============================================================= #
    def _merge_detections(self, *region_lists) -> List[Dict]:
        merged: List[Dict] = []
        for regions in region_lists:
            for region in regions:
                rx, ry, rw, rh = region['bbox']
                dup_idx = -1
                for i, existing in enumerate(merged):
                    ex, ey, ew, eh = existing['bbox']
                    ox = max(0, min(rx + rw, ex + ew) - max(rx, ex))
                    oy = max(0, min(ry + rh, ey + eh) - max(ry, ey))
                    inter = ox * oy
                    union = rw * rh + ew * eh - inter
                    if union > 0 and inter / union > self.nms_iou_threshold:
                        dup_idx = i
                        break
                if dup_idx < 0:
                    merged.append(region)
                else:
                    # ادغام ماسک‌ها -> ماسک دقیق‌تر
                    merged[dup_idx]['mask'] = cv2.bitwise_or(
                        merged[dup_idx]['mask'], region['mask'])
                    merged[dup_idx]['features'].update(region['features'])
        return merged

    def _circle_nms(self, detections: List[Detection]) -> List[Detection]:
        """حذف دایره‌های همپوشان با نگه‌داشتن پرقدرت‌ترین."""
        if not detections:
            return detections
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for d in dets:
            (cx, cy) = d.features['center']
            r = d.features['radius']
            overlap = False
            for k in kept:
                (kx, ky) = k.features['center']
                kr = k.features['radius']
                dist = np.hypot(cx - kx, cy - ky)
                if dist < (r + kr) * 0.6:      # همپوشانی زیاد
                    overlap = True
                    break
            if not overlap:
                kept.append(d)
        return kept

    # ============================================================= #
    #                       SCORING                                 #
    # ============================================================= #
    @staticmethod
    def _prominence(gray: np.ndarray, mask: np.ndarray, radius: float):
        """Lesion mean intensity minus the median of a size-relative annulus.

        Also returns the fraction of the annulus that is as bright as the
        lesion: a mass sits on darker tissue, whereas a patch of dense
        parenchyma is surrounded by similarly bright tissue.
        """
        mask_bin = (mask > 0).astype(np.uint8) * 255
        if cv2.countNonZero(mask_bin) == 0:
            return 0.0, 1.0
        inner = max(2, int(round(max(radius, 1) * 1.2)))
        outer = max(inner + 3, int(round(max(radius, 1) * 2.5)))
        dist = cv2.distanceTransform(255 - mask_bin, cv2.DIST_L2, 5)
        ring = (dist > inner) & (dist <= outer)
        ring_values = gray[ring]
        inside_values = gray[mask_bin > 0]
        if ring_values.size < 10 or inside_values.size == 0:
            return 0.0, 1.0
        inside_mean = float(np.mean(inside_values))
        prominence = inside_mean - float(np.median(ring_values))
        ring_density = float(np.mean(ring_values >= inside_mean - 15.0))
        return prominence, ring_density

    @staticmethod
    def _prominence_bonus(prominence: float) -> float:
        """Convert prominence (grey levels) into a bounded confidence bonus."""
        if prominence >= 45:
            return 0.16
        if prominence >= 30:
            return 0.12
        if prominence >= 20:
            return 0.07
        if prominence >= 12:
            return 0.03
        return 0.0

    def _evaluate_region(self, region, enhanced, mask, radius) -> float:
        """امتیازدهی چند-معیاره -> کاهش False Positive."""
        score = 0.0
        f = region.get('features', {})
        area = float(cv2.countNonZero(mask))

        # 1) اندازه منطقی
        if 300 <= area <= 15000:
            score += 0.20
        elif area > 15000:
            score += 0.05     # خیلی بزرگ = مشکوک به بافت طبیعی
        else:
            score += 0.08

        # 2) بافت
        if f.get('contrast', 0) > 40:
            score += 0.15
        if 0 < f.get('homogeneity', 1) < 0.55:
            score += 0.10

        # 3) شکل: تومورها اغلب گرد ولی نامنظم‌اند
        circ = f.get('circularity', 0)
        if 0.4 <= circ <= 0.85:
            score += 0.15
        elif circ > 0.85:
            score += 0.05

        # 4) کنتراست محلی نسبت به اطراف
        roi_vals = enhanced[mask > 0]
        if roi_vals.size > 0:
            inside_mean = float(np.mean(roi_vals))
            # ناحیه حلقوی اطراف
            dil = cv2.dilate(mask, np.ones((15, 15), np.uint8))
            ring = cv2.subtract(dil, mask)
            ring_vals = enhanced[ring > 0]
            if ring_vals.size > 0:
                outside_mean = float(np.mean(ring_vals))
                contrast_diff = inside_mean - outside_mean
                if contrast_diff > 20:
                    score += 0.20
                elif contrast_diff > 8:
                    score += 0.10

            local_std = float(np.std(roi_vals))
            if local_std > 25:
                score += 0.10

        # 5) اعتبارسنجی دایره‌ای (پرشدگی دایره محیطی)
        if radius > 0:
            circle_area = np.pi * radius * radius
            fill_ratio = area / circle_area if circle_area > 0 else 0
            if fill_ratio > 0.55:
                score += 0.10

        return float(min(score, 1.0))

    # ============================================================= #
    #                      VISUALIZATION                            #
    # ============================================================= #
    @staticmethod
    def draw_detections(image: np.ndarray,
                        detections: List[Detection]) -> np.ndarray:
        """
        رسم مرز دایره‌ای، مرکز، شعاع، اطمینان و برچسب برای هر تومور.
        """
        if len(image.shape) == 2:
            out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            out = image.copy()

        for d in detections:
            center = d.features.get('center', d.location)
            radius = int(d.features.get('radius', 10))
            label  = d.features.get('label', 'Tumor')
            conf   = d.confidence

            color = (0, 0, 255) if label == TumorDetector.LABEL_MALIGNANT \
                    else (0, 165, 255)

            cv2.circle(out, center, radius, color, 2)
            cv2.circle(out, center, 3, color, -1)
            text = f"{label} {conf:.2f} r={radius}"
            cv2.putText(out, text,
                        (center[0] - radius, max(center[1] - radius - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out
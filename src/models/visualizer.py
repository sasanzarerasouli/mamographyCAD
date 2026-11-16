from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..core.detection_types import Detection, DetectionType


class ResultVisualizer:
    """
    Visualizer for mammography detection results.

    Draws:
    - Tumor circles and bounding boxes
    - Tumor labels
    - Calcification markers
    - Segmentation overlays
    - Pectoral muscle boundary
    - Density heatmap
    """

    COLORS = {
        DetectionType.TUMOR_CONFIRMED: (0, 0, 255),        # قرمز
        DetectionType.TUMOR_SUSPICIOUS: (0, 165, 255),     # نارنجی
        DetectionType.MICROCALCIFICATION: (255, 255, 0),  # زرد
        DetectionType.MACROCALCIFICATION: (0, 255, 255),  # فیروزه‌ای
    }

    def __init__(self) -> None:
        # نمایش‌ها
        self.show_labels: bool = True
        self.show_tumors: bool = True
        self.show_calcifications: bool = True
        self.show_arrows: bool = True
        self.show_masks: bool = True
        self.show_bboxes: bool = True
        self.show_centers: bool = True
        #: labels are drawn for at most this many findings (highest confidence
        #: first); boxes/markers are always drawn for every finding
        self.max_labels: int = 10
        self.show_pectoral: bool = True
        self.show_density_heatmap: bool = True

        # تنظیمات ظاهری
        self.line_thickness: int = 2
        self.mask_opacity: float = 0.5
        self.pectoral_thickness: int = 3
        self.pectoral_color: tuple[int, int, int] = (255, 0, 255)  # بنفش

        # تنظیمات هیت‌مپ
        self.heatmap_opacity: float = 0.4
        self.heatmap_colormap: int = cv2.COLORMAP_JET

        # تنظیمات مارکر کلسیفیکاسیون
        self.calc_marker_size: int = 6
        self.calc_thickness: int = 2

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        highlight_id: Optional[int] = None,
        pectoral_mask: Optional[np.ndarray] = None,
        density_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Visualize all detection results on the image.

        Args:
            image: Input image (grayscale or BGR)
            detections: List of Detection objects
            highlight_id: ID of detection to highlight
            pectoral_mask: Binary mask of pectoral muscle
            density_map: Density probability/map (float, 0-1)

        Returns:
            Visualized BGR image
        """
        # آماده‌سازی تصویر نتیجه
        result = image.copy()
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # 1. کشیدن هیت‌مپ دانسیته (زیر همه لایه‌ها)
        if self.show_density_heatmap and density_map is not None:
            result = self._draw_density_heatmap(result, density_map)

        # 2. کشیدن اورلی سگمنتیشن (ماسک‌ها)
        if self.show_masks:
            result = self._draw_segmentation_overlays(result, detections, highlight_id)

        # 3. کشیدن مرز پکتورال
        if self.show_pectoral and pectoral_mask is not None:
            result = self._draw_pectoral_boundary(result, pectoral_mask)

        # 4. کشیدن دیتکشن‌ها (باکس، دایره، فلش، لیبل، مارکرهای کلسیفیکاسیون)
        result = self._draw_detections(result, detections, highlight_id)

        return result

    def _draw_density_heatmap(
        self, image: np.ndarray, density_map: np.ndarray
    ) -> np.ndarray:
        """
        Draw density heatmap overlay.

        Args:
            image: Input BGR image
            density_map: Density map (float, values in [0, 1])

        Returns:
            Image with heatmap overlay
        """
        if density_map is None or density_map.size == 0:
            return image

        # اطمینان از سایز یکسان
        if density_map.shape[:2] != image.shape[:2]:
            density_map_resized = cv2.resize(
                density_map.astype(np.float32),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            density_map_resized = density_map.astype(np.float32)

        # نرمال‌سازی به [0, 1]
        if density_map_resized.max() > density_map_resized.min():
            density_norm = (density_map_resized - density_map_resized.min()) / (
                density_map_resized.max() - density_map_resized.min() + 1e-8
            )
        else:
            density_norm = np.zeros_like(density_map_resized, dtype=np.float32)

        # محدود کردن به [0,1]
        density_norm = np.clip(density_norm, 0.0, 1.0)

        # تبدیل به uint8
        heatmap_uint8 = (density_norm * 255).astype(np.uint8)

        # اعمال Colormap
        if heatmap_uint8.ndim == 2:
            colored_heatmap = cv2.applyColorMap(heatmap_uint8, self.heatmap_colormap)
        else:
            colored_heatmap = cv2.applyColorMap(
                cv2.cvtColor(heatmap_uint8, cv2.COLOR_BGR2GRAY), self.heatmap_colormap
            )

        # ترکیب با تصویر اصلی
        if self.heatmap_opacity <= 0:
            return image
        if self.heatmap_opacity >= 1:
            return colored_heatmap

        blended = cv2.addWeighted(
            colored_heatmap,
            float(self.heatmap_opacity),
            image,
            1.0 - float(self.heatmap_opacity),
            0,
        )
        return blended

    def _draw_segmentation_overlays(
        self,
        image: np.ndarray,
        detections: List[Detection],
        highlight_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw segmentation masks (overlays).

        Args:
            image: Input BGR image
            detections: List of detections
            highlight_id: Highlighted detection ID

        Returns:
            Image with segmentation overlays
        """
        result = image
        # Labels are only drawn for the most suspicious findings: with a few
        # dozen candidates the overlay becomes unreadable.  Every detection keeps
        # its box/marker and remains listed in the results table, so nothing is
        # hidden - only the text is prioritised.
        visible = [d for d in detections if not d.is_ignored]
        labelled_ids = {
            detection.id
            for detection in sorted(
                visible, key=lambda d: float(getattr(d, "confidence", 0.0)), reverse=True
            )[: max(0, int(getattr(self, "max_labels", 10)))]
        }

        for detection in detections:
            if detection.is_ignored:
                continue

            if detection.mask is None or detection.mask.size == 0:
                continue

            # بررسی نمایش تومور
            if not self.show_tumors and detection.type in (
                DetectionType.TUMOR_CONFIRMED,
                DetectionType.TUMOR_SUSPICIOUS,
            ):
                continue

            # بررسی نمایش کلسیفیکاسیون
            if not self.show_calcifications and detection.type in (
                DetectionType.MICROCALCIFICATION,
                DetectionType.MACROCALCIFICATION,
            ):
                continue

            color = self.COLORS.get(detection.type, (255, 255, 255))

            # اطمینان از سایز ماسک
            mask = detection.mask
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # ایجاد اورلی
            overlay = result.copy()
            overlay[mask > 0] = color

            # تعیین opacity
            opacity = self.mask_opacity
            if detection.id == highlight_id:
                opacity = min(1.0, self.mask_opacity + 0.2)

            # ترکیب
            if opacity <= 0:
                continue
            if opacity >= 1:
                result = overlay
                continue

            result = cv2.addWeighted(overlay, float(opacity), result, 1.0 - float(opacity), 0)

        return result

    def _draw_pectoral_boundary(
        self, image: np.ndarray, pectoral_mask: np.ndarray
    ) -> np.ndarray:
        """
        Draw pectoral muscle boundary.

        Args:
            image: Input BGR image
            pectoral_mask: Binary mask of pectoral muscle

        Returns:
            Image with pectoral boundary
        """
        if pectoral_mask is None or pectoral_mask.size == 0:
            return image

        result = image
        # آماده‌سازی ماسک
        mask = pectoral_mask.astype(np.uint8)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # باینری کردن
        mask = (mask > 0).astype(np.uint8) * 255

        # پیدا کردن کانتورهای مرز
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return result

        # کشیدن کانتورهای اصلی (بزرگ‌ترین یا همه)
        # برای عملکرد بهتر، بزرگ‌ترین کانتور رو می‌کشیم
        if contours:
            # پیدا کردن بزرگ‌ترین کانتور
            best_contour = max(contours, key=cv2.contourArea, default=None)
            if best_contour is not None and cv2.contourArea(best_contour) > 50:
                cv2.drawContours(
                    result,
                    [best_contour],
                    -1,
                    self.pectoral_color,
                    self.pectoral_thickness,
                    cv2.LINE_AA,
                )
            # اگر خواستی همه کانتورها کشیده بشن، می‌تونی کامنت بالا رو برداری
            # for cnt in contours:
            #     if cv2.contourArea(cnt) > 50:
            #         cv2.drawContours(result, [cnt], -1, self.pectoral_color,
            #                          self.pectoral_thickness, cv2.LINE_AA)

        return result

    def _draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        highlight_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw detection annotations (boxes, circles, arrows, labels, calc markers).

        Args:
            image: Input BGR image
            detections: List of detections
            highlight_id: Highlighted detection ID

        Returns:
            Image with detection drawings
        """
        result = image
        # Labels are only drawn for the most suspicious findings: with a few
        # dozen candidates the overlay becomes unreadable.  Every detection keeps
        # its box/marker and remains listed in the results table, so nothing is
        # hidden - only the text is prioritised.
        visible = [d for d in detections if not d.is_ignored]
        labelled_ids = {
            detection.id
            for detection in sorted(
                visible, key=lambda d: float(getattr(d, "confidence", 0.0)), reverse=True
            )[: max(0, int(getattr(self, "max_labels", 10)))]
        }

        for detection in detections:
            if detection.is_ignored:
                continue

            # فیلتر تومورها
            if not self.show_tumors and detection.type in (
                DetectionType.TUMOR_CONFIRMED,
                DetectionType.TUMOR_SUSPICIOUS,
            ):
                continue

            # فیلتر کلسیفیکاسیون‌ها
            if not self.show_calcifications and detection.type in (
                DetectionType.MICROCALCIFICATION,
                DetectionType.MACROCALCIFICATION,
            ):
                continue

            color = self.COLORS.get(detection.type, (255, 255, 255))

            # ضخامت
            thickness = self.line_thickness
            if detection.id == highlight_id:
                thickness = self.line_thickness + 2

            # 1. کشیدن باکس
            if self.show_bboxes and detection.bbox:
                x, y, w, h = detection.bbox
                x0, y0 = int(x), int(y)
                x1, y1 = int(x + w), int(y + h)
                cv2.rectangle(result, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

            # 2. کشیدن مرکز و دایره‌های تومور
            if detection.location:
                cx, cy = detection.location
                cx_i, cy_i = int(cx), int(cy)

                # کشیدن دایره مرکز
                if self.show_centers:
                    cv2.circle(result, (cx_i, cy_i), 5, color, -1, cv2.LINE_AA)

                # کشیدن فلش برای اشاره
                if self.show_arrows:
                    arrow_len = 40
                    # فلش رو به بالا
                    start_point = (cx_i, cy_i - arrow_len)
                    end_point = (cx_i, cy_i - 5)
                    cv2.arrowedLine(
                        result, start_point, end_point, color, thickness, cv2.LINE_AA
                    )

                # کشیدن دایره‌های تومور (برای تومورها)
                if self.show_tumors and detection.type in (
                    DetectionType.TUMOR_CONFIRMED,
                    DetectionType.TUMOR_SUSPICIOUS,
                ):
                    self._draw_tumor_circles(result, detection, color, thickness)

            # 3. کشیدن مارکرهای کلسیفیکاسیون
            if detection.type in (
                DetectionType.MICROCALCIFICATION,
                DetectionType.MACROCALCIFICATION,
            ):
                self._draw_calcification_markers(result, detection, color, thickness)

            # 4. کشیدن لیبل‌ها
            if self.show_labels and detection.location and detection.id in labelled_ids:
                cx, cy = detection.location
                cx_i, cy_i = int(cx), int(cy)
                self._draw_label(result, detection, cx_i, cy_i, color)

        return result

    def _draw_tumor_circles(
        self,
        image: np.ndarray,
        detection: Detection,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """
        Draw tumor circles (additional circles around tumor).

        Args:
            image: Image to draw on (inplace)
            detection: Detection object
            color: Drawing color
            thickness: Line thickness
        """
        if not detection.location:
            return

        cx, cy = detection.location
        cx_i, cy_i = int(cx), int(cy)

        # اگر باکس موجوده، از باکس برای تخمین شعاع استفاده کن
        radius = 15
        if detection.bbox:
            x, y, w, h = detection.bbox
            # شعاع تقریبی بر اساس نصف میانگین ابعاد
            radius = int(max(w, h) / 2.0)
            # حداقل و حداکثر شعاع
            radius = max(10, min(radius, 80))

        # کشیدن دایره بیرونی
        cv2.circle(image, (cx_i, cy_i), radius, color, thickness, cv2.LINE_AA)

        # کشیدن دایره داخلی کوچیک‌تر (برای تاکید بیشتر)
        if radius > 20:
            inner_radius = radius // 2
            cv2.circle(image, (cx_i, cy_i), inner_radius, color, thickness, cv2.LINE_AA)

    def _draw_calcification_markers(
        self,
        image: np.ndarray,
        detection: Detection,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """
        Draw calcification markers.

        Args:
            image: Image to draw on (inplace)
            detection: Detection object
            color: Drawing color
            thickness: Line thickness
        """
        if not detection.location:
            return

        cx, cy = detection.location
        cx_i, cy_i = int(cx), int(cy)
        size = self.calc_marker_size

        if detection.type == DetectionType.MICROCALCIFICATION:
            # برای میکروکلسیفیکاسیون: علامت + کوچک
            # خط افقی
            cv2.line(
                image,
                (cx_i - size, cy_i),
                (cx_i + size, cy_i),
                color,
                self.calc_thickness,
                cv2.LINE_AA,
            )
            # خط عمودی
            cv2.line(
                image,
                (cx_i, cy_i - size),
                (cx_i, cy_i + size),
                color,
                self.calc_thickness,
                cv2.LINE_AA,
            )
        elif detection.type == DetectionType.MACROCALCIFICATION:
            # برای ماکروکلسیفیکاسیون: علامت × یا مربع
            # علامت ×
            half = size // 2
            # خط اول
            cv2.line(
                image,
                (cx_i - half, cy_i - half),
                (cx_i + half, cy_i + half),
                color,
                self.calc_thickness,
                cv2.LINE_AA,
            )
            # خط دوم
            cv2.line(
                image,
                (cx_i + half, cy_i - half),
                (cx_i - half, cy_i + half),
                color,
                self.calc_thickness,
                cv2.LINE_AA,
            )

        # همچنین اگر ماسک خیلی کوچک بود، می‌تونی نقطه هم اضافه کنی
        cv2.circle(image, (cx_i, cy_i), 2, color, -1, cv2.LINE_AA)

    def _draw_label(
        self,
        image: np.ndarray,
        detection: Detection,
        cx: int,
        cy: int,
        color: tuple[int, int, int],
    ) -> None:
        """
        Draw detection label.

        Args:
            image: Image to draw on (inplace)
            detection: Detection object
            cx: Center x
            cy: Center y
            color: Text color
        """
        # ساخت متن
        type_name = detection.type.value.replace("_", " ").title()
        conf_text = f"{detection.confidence:.0%}"
        label = f"{type_name} ({conf_text})"

        # موقعیت متن (بالای فلش یا کنار مرکز)
        text_pos = (cx + 10, cy - 15)

        # اندازه فونت
        font_scale = 0.5
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        # محاسبه سایز متن برای بک‌گراند (اختیاری ولی خواناتر)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # بک‌گراند نیمه‌شفاف برای خوانایی بهتر
        padding = 4
        x0_bg = text_pos[0] - padding
        y0_bg = text_pos[1] - text_h - padding
        x1_bg = text_pos[0] + text_w + padding
        y1_bg = text_pos[1] + baseline + padding

        # کشیدن بک‌گراند (سیاه نیمه‌شفاف)
        bg_color = (0, 0, 0)
        overlay = image.copy()
        cv2.rectangle(overlay, (x0_bg, y0_bg), (x1_bg, y1_bg), bg_color, -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # کشیدن متن
        cv2.putText(
            image,
            label,
            text_pos,
            font,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

    # متدهای کمکی برای تنظیمات
    def set_show_options(
        self,
        *,
        show_labels: Optional[bool] = None,
        show_tumors: Optional[bool] = None,
        show_calcifications: Optional[bool] = None,
        show_arrows: Optional[bool] = None,
        show_masks: Optional[bool] = None,
        show_bboxes: Optional[bool] = None,
        show_centers: Optional[bool] = None,
        show_pectoral: Optional[bool] = None,
        show_density_heatmap: Optional[bool] = None,
    ) -> None:
        """
        Set visibility options.

        Args:
            show_labels: Show labels
            show_tumors: Show tumors
            show_calcifications: Show calcifications
            show_arrows: Show arrows
            show_masks: Show segmentation masks
            show_bboxes: Show bounding boxes
            show_centers: Show center points
            show_pectoral: Show pectoral boundary
            show_density_heatmap: Show density heatmap
        """
        if show_labels is not None:
            self.show_labels = show_labels
        if show_tumors is not None:
            self.show_tumors = show_tumors
        if show_calcifications is not None:
            self.show_calcifications = show_calcifications
        if show_arrows is not None:
            self.show_arrows = show_arrows
        if show_masks is not None:
            self.show_masks = show_masks
        if show_bboxes is not None:
            self.show_bboxes = show_bboxes
        if show_centers is not None:
            self.show_centers = show_centers
        if show_pectoral is not None:
            self.show_pectoral = show_pectoral
        if show_density_heatmap is not None:
            self.show_density_heatmap = show_density_heatmap

    def set_style_options(
        self,
        *,
        line_thickness: Optional[int] = None,
        mask_opacity: Optional[float] = None,
        pectoral_thickness: Optional[int] = None,
        pectoral_color: Optional[tuple[int, int, int]] = None,
        heatmap_opacity: Optional[float] = None,
        heatmap_colormap: Optional[int] = None,
        calc_marker_size: Optional[int] = None,
        calc_thickness: Optional[int] = None,
    ) -> None:
        """
        Set style options.

        Args:
            line_thickness: General line thickness
            mask_opacity: Segmentation mask opacity
            pectoral_thickness: Pectoral boundary thickness
            pectoral_color: Pectoral boundary color (BGR)
            heatmap_opacity: Density heatmap opacity
            heatmap_colormap: OpenCV colormap (e.g., cv2.COLORMAP_JET)
            calc_marker_size: Calcification marker size
            calc_thickness: Calcification marker thickness
        """
        if line_thickness is not None:
            self.line_thickness = max(1, int(line_thickness))
        if mask_opacity is not None:
            self.mask_opacity = float(np.clip(mask_opacity, 0.0, 1.0))
        if pectoral_thickness is not None:
            self.pectoral_thickness = max(1, int(pectoral_thickness))
        if pectoral_color is not None:
            self.pectoral_color = tuple(int(c) for c in pectoral_color)
        if heatmap_opacity is not None:
            self.heatmap_opacity = float(np.clip(heatmap_opacity, 0.0, 1.0))
        if heatmap_colormap is not None:
            self.heatmap_colormap = int(heatmap_colormap)
        if calc_marker_size is not None:
            self.calc_marker_size = max(2, int(calc_marker_size))
        if calc_thickness is not None:
            self.calc_thickness = max(1, int(calc_thickness))
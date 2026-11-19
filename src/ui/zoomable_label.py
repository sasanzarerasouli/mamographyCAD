"""Zoomable / pannable image widget with detection hit-testing.

Rendering model
---------------
The previous implementation cached a rendered ``QPixmap`` and only refreshed it
when an explicit redraw was requested.  That cache was not invalidated on
resize, which is why overlays and highlight rings only appeared after the user
zoomed or dragged.  The widget now paints in :meth:`paintEvent` on every
update, so overlays, highlights and window resizes are always consistent.

The public API is unchanged (``set_image``, ``fit_to_window``,
``set_detections``, ``highlight_detection``, ``reset_view``,
``detectionClicked``, ``detections``, ``highlighted_detection``).
"""
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QLabel, QMenu

from ..core.detection_types import Detection


class ZoomableImageLabel(QLabel):
    detectionClicked = pyqtSignal(int)
    maskBrushClicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)

        self._pixmap: Optional[QPixmap] = None
        self._scale = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._pan_start: Optional[QPointF] = None

        # interaction state
        self._press_pos: Optional[QPointF] = None
        self._has_dragged = False
        self._drag_threshold = 4
        self._auto_fit = True
        self.mask_brush_active = False

        self.detections: List[Detection] = []
        self.highlighted_detection: Optional[int] = None

        self._min_scale = 0.05
        self._max_scale = 8.0

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ #
    #  Image handling                                                    #
    # ------------------------------------------------------------------ #
    def set_image(self, image: np.ndarray, keep_view: bool = True):
        """Display a numpy image (2-D grayscale or 3-channel BGR)."""
        if image is None:
            return
        q_image = self._to_qimage(image)
        if q_image is None:
            return

        was_empty = self._pixmap is None
        self._pixmap = QPixmap.fromImage(q_image)
        if was_empty or not keep_view:
            self.fit_to_window()
        else:
            self.update()

    @staticmethod
    def _to_qimage(image: np.ndarray) -> Optional[QImage]:
        if image.dtype != np.uint8:
            import cv2

            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = np.ascontiguousarray(image)

        if image.ndim == 2:
            height, width = image.shape
            return QImage(
                image.data, width, height, width, QImage.Format.Format_Grayscale8
            ).copy()
        if image.ndim == 3:
            height, width, channels = image.shape
            if channels == 4:
                return QImage(
                    image.data, width, height, 4 * width,
                    QImage.Format.Format_ARGB32
                ).copy()
            return QImage(
                image.data, width, height, 3 * width, QImage.Format.Format_RGB888
            ).rgbSwapped().copy()
        return None

    def clear_image(self):
        self._pixmap = None
        self.detections = []
        self.highlighted_detection = None
        self.update()

    def current_image(self) -> Optional[QPixmap]:
        """The image currently displayed (without the zoom/overlay rendering).

        ``QLabel.pixmap()`` is always null for this widget because rendering
        happens in :meth:`paintEvent`; use this accessor (or
        :meth:`rendered_pixmap`) from code and tests instead.
        """
        return self._pixmap

    def rendered_pixmap(self) -> QPixmap:
        """Grab what the widget actually shows, overlays included."""
        return self.grab()

    def has_image(self) -> bool:
        return self._pixmap is not None

    def fit_to_window(self):
        if self._pixmap is None or self.width() <= 0 or self.height() <= 0:
            return
        w_ratio = self.width() / max(1, self._pixmap.width())
        h_ratio = self.height() / max(1, self._pixmap.height())
        new_scale = min(w_ratio, h_ratio)
        new_scale = max(self._min_scale, min(self._max_scale, new_scale))

        scaled_w = self._pixmap.width() * new_scale
        scaled_h = self._pixmap.height() * new_scale
        self._scale = new_scale
        self._pan_offset = QPointF(
            (self.width() - scaled_w) / 2.0, (self.height() - scaled_h) / 2.0
        )
        self._auto_fit = True
        self.update()

    def reset_view(self):
        self.fit_to_window()

    def zoom_factor(self) -> float:
        return float(self._scale)

    def set_zoom(self, scale: float):
        if self._pixmap is None:
            return
        self._scale = max(self._min_scale, min(self._max_scale, float(scale)))
        self._auto_fit = False
        self.update()

    # ------------------------------------------------------------------ #
    #  Detections                                                        #
    # ------------------------------------------------------------------ #
    def set_detections(self, detections: List[Detection]):
        self.detections = list(detections or [])

    def highlight_detection(self, detection_id: Optional[int]):
        self.highlighted_detection = detection_id
        self.update()

    # ------------------------------------------------------------------ #
    #  Painting                                                          #
    # ------------------------------------------------------------------ #
    def paintEvent(self, event):  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self._pixmap is None:
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded"
            )
            painter.end()
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._scale < 1.0)
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        painter.save()
        painter.translate(self._pan_offset)
        painter.translate(center)
        painter.scale(self._scale, self._scale)
        painter.translate(-center)
        painter.drawPixmap(QPointF(0, 0), self._pixmap)
        painter.restore()

        self._paint_highlight(painter)
        painter.end()

    def _paint_highlight(self, painter: QPainter) -> None:
        """Draw a selection ring around the currently highlighted detection."""
        if self.highlighted_detection is None:
            return
        detection = None
        for candidate in self.detections:
            if getattr(candidate, "id", None) == self.highlighted_detection:
                detection = candidate
                break
        if detection is None:
            return

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 255, 120), 2, Qt.PenStyle.DashLine))

        # Use the bbox when available, otherwise a circle around the centre.
        if detection.bbox:
            x, y, w, h = detection.bbox
            top_left = self._image_to_screen_coords(QPointF(x, y))
            bottom_right = self._image_to_screen_coords(QPointF(x + w, y + h))
            rect = (
                top_left,
                bottom_right,
            )
            painter.drawRect(
                int(min(rect[0].x(), rect[1].x())) - 3,
                int(min(rect[0].y(), rect[1].y())) - 3,
                int(abs(rect[1].x() - rect[0].x())) + 6,
                int(abs(rect[1].y() - rect[0].y())) + 6,
            )
        else:
            center = self._image_to_screen_coords(QPointF(*detection.location))
            radius = max(6.0, 12.0 * self._scale)
            painter.drawEllipse(center, radius, radius)

    # ------------------------------------------------------------------ #
    #  Events                                                            #
    # ------------------------------------------------------------------ #
    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self._auto_fit:
            # Keep the whole image visible while the window is resized; without
            # this the overlay appeared to "jump" or vanish after a resize.
            self.fit_to_window()
        else:
            self.update()

    def wheelEvent(self, event: QWheelEvent):  # noqa: N802 - Qt naming
        if self._pixmap is None:
            return

        old_scale = self._scale
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        new_scale = max(
            self._min_scale, min(self._max_scale, old_scale * zoom_factor)
        )
        if abs(new_scale - old_scale) < 1e-6:
            return

        mouse_pos = event.position()
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        image_pos = (mouse_pos - self._pan_offset - center) / old_scale

        self._scale = new_scale
        self._pan_offset = mouse_pos - (image_pos * new_scale + center)
        self._auto_fit = False
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):  # noqa: N802 - Qt naming
        if self.mask_brush_active and event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            point = self._screen_to_image_coords(event.position())
            self.maskBrushClicked.emit(int(np.floor(point.x())), int(np.floor(point.y())))
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._pan_start = event.position()
            self._press_pos = event.position()
            self._has_dragged = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):  # noqa: N802 - Qt naming
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            if (
                self._press_pos is not None
                and (event.position() - self._press_pos).manhattanLength()
                > self._drag_threshold
            ):
                self._has_dragged = True
            self._pan_offset += delta
            self._pan_start = event.position()
            self._auto_fit = False
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):  # noqa: N802 - Qt naming
        if event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if not self._has_dragged and self._press_pos is not None:
                click_pos = self._screen_to_image_coords(event.position())
                detection = self._find_detection_at(click_pos, include_ignored=False)
                if detection is not None:
                    self.detectionClicked.emit(detection.id)
            self._pan_start = None
            self._press_pos = None
            self._has_dragged = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):  # noqa: N802 - Qt naming
        self.fit_to_window()
        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------------ #
    #  Context menu                                                      #
    # ------------------------------------------------------------------ #
    def show_context_menu(self, pos):
        if not self.detections:
            return
        image_pos = self._screen_to_image_coords(QPointF(pos))
        clicked = self._find_detection_at(image_pos, include_ignored=True)
        if clicked is None:
            return

        menu = QMenu(self)
        if clicked.is_ignored:
            action = QAction("Restore Detection", self)
            action.triggered.connect(
                lambda _checked=False, d=clicked: self._toggle_ignore_detection(d)
            )
        else:
            action = QAction("Ignore Detection (False Positive)", self)
            action.triggered.connect(
                lambda _checked=False, d=clicked: self._toggle_ignore_detection(d)
            )
        menu.addAction(action)
        menu.exec(self.mapToGlobal(pos))

    def _toggle_ignore_detection(self, detection: Detection):
        detection.is_ignored = not detection.is_ignored
        window = self.window()
        if hasattr(window, "update_visualization"):
            window.update_visualization()
        if hasattr(window, "update_results_display"):
            window.update_results_display()
        self.update()

    # ------------------------------------------------------------------ #
    #  Coordinate transforms                                             #
    # ------------------------------------------------------------------ #
    def _screen_to_image_coords(self, screen_pos: QPointF) -> QPointF:
        if self._pixmap is None:
            return QPointF(0, 0)
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        return (screen_pos - self._pan_offset - center) / self._scale + center

    def _image_to_screen_coords(self, image_pos: QPointF) -> QPointF:
        if self._pixmap is None:
            return QPointF(0, 0)
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        return (image_pos - center) * self._scale + self._pan_offset + center

    def _point_hit_radius(self) -> float:
        screen_tolerance = 12.0
        return max(8.0, screen_tolerance / max(self._scale, 1e-6))

    def _hit_test(self, point: QPointF, detection: Detection) -> Tuple[bool, float]:
        if detection.bbox:
            x, y, w, h = detection.bbox
            if x <= point.x() <= x + w and y <= point.y() <= y + h:
                return True, float(w) * float(h)
            return False, float("inf")

        cx, cy = detection.location
        radius = self._point_hit_radius()
        distance = ((point.x() - cx) ** 2 + (point.y() - cy) ** 2) ** 0.5
        if distance <= radius:
            return True, distance
        return False, float("inf")

    def _find_detection_at(
        self, image_pos: QPointF, include_ignored: bool = True
    ) -> Optional[Detection]:
        best: Optional[Detection] = None
        best_score = float("inf")
        for detection in self.detections:
            if not include_ignored and detection.is_ignored:
                continue
            hit, score = self._hit_test(image_pos, detection)
            if hit and score < best_score:
                best = detection
                best_score = score
        return best

    def _is_point_in_detection(self, point: QPointF, detection: Detection) -> bool:
        hit, _ = self._hit_test(point, detection)
        return hit

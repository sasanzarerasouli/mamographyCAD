"""Persistent application settings via QSettings with validation and type safety."""
import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union

from PyQt6.QtCore import QSettings

from config import Config

T = TypeVar("T")
logger = logging.getLogger(__name__)

class SettingsManager:
    """Centralized settings persistence with validation and Config syncing."""

    _instance: Optional["SettingsManager"] = None

    def __init__(self):
        self._settings = QSettings("MammographyCAD", "Settings")
        # Define schema with defaults and their types
        self._defaults = self._build_defaults()

    @classmethod
    def instance(cls) -> "SettingsManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_defaults(self) -> Dict[str, Any]:
        """Centralized source of truth for all settings and types."""
        return {
            "geometry": None,
            "show_mask_layer": True,
            "show_labels": True,
            "show_tumors": True,
            "show_arrows": True,
            "line_thickness": Config.LINE_THICKNESS,
            "mask_opacity": int(Config.MASK_OPACITY * 100),
            "heatmap_opacity": int(Config.HEATMAP_OPACITY * 100),
            "processing_backend": Config.DEFAULT_BACKEND,
            "detection_threshold": Config.DETECTION_THRESHOLD,
            "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
            "calcification_sensitivity": Config.CALCIFICATION_SENSITIVITY,
            "cluster_threshold": Config.CLUSTER_THRESHOLD,
            "seg_adaptive_block_size": Config.SEG_ADAPTIVE_BLOCK_SIZE,
            "seg_adaptive_c": Config.SEG_ADAPTIVE_C,
            "seg_morph_close_kernel": Config.SEG_MORPH_CLOSE_KERNEL,
            "seg_morph_open_kernel": Config.SEG_MORPH_OPEN_KERNEL,
            "seg_smooth_kernel": Config.SEG_SMOOTH_KERNEL,
            "density_threshold_a": Config.DENSITY_THRESHOLDS["A"],
            "density_threshold_b": Config.DENSITY_THRESHOLDS["B"],
            "density_threshold_c": Config.DENSITY_THRESHOLDS["C"],
            "birads_2_max": Config.BIRADS_THRESHOLDS["birads_2_max"],
            "birads_3_max": Config.BIRADS_THRESHOLDS["birads_3_max"],
            "birads_4_max": Config.BIRADS_THRESHOLDS["birads_4_max"],
            "risk_weight_shape": Config.RISK_WEIGHTS["shape_irregularity"],
            "risk_weight_spiculation": Config.RISK_WEIGHTS["spiculation"],
            "risk_weight_contrast": Config.RISK_WEIGHTS["contrast"],
            "risk_weight_entropy": Config.RISK_WEIGHTS["texture_entropy"],
            "risk_weight_density": Config.RISK_WEIGHTS["density_difference"],
            "debug_mode": Config.DEBUG_MODE,
        }

    def _get_validated(self, key: str, expected_type: Type[T]) -> T:
        """Retrieves setting, validates type, and falls back to default if corrupted."""
        value = self._settings.value(key)
        
        # 1. If key is missing, return default
        if value is None:
            return self._defaults.get(key)

        # 2. Try to cast/validate
        try:
            if isinstance(value, expected_type):
                return value
            
            # Special handling for booleans from QSettings (which can be strings)
            if expected_type is bool:
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
            
            return expected_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid setting value for '{key}': {value}. Falling back to default. Error: {e}")
            return self._defaults.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Generic getter. For specific types use specialized get methods."""
        val = self._settings.value(key)
        return val if val is not None else (default if default is not None else self._defaults.get(key))

    def get_float(self, key: str) -> float:
        return float(self._get_validated(key, float))

    def get_int(self, key: str) -> int:
        return int(self._get_validated(key, int))

    def get_bool(self, key: str) -> bool:
        return bool(self._get_validated(key, bool))

    def set(self, key: str, value: Any) -> None:
        """Persist value and sync immediately."""
        self._settings.setValue(key, value)
        self._settings.sync()  # Ensure it writes to disk

    def apply_to_config(self) -> None:
        """Sync all loaded settings into Config class attributes."""
        try:
            Config.DETECTION_THRESHOLD = self.get_float("detection_threshold")
            Config.CONFIDENCE_THRESHOLD = self.get_float("confidence_threshold")
            Config.CALCIFICATION_SENSITIVITY = self.get_float("calcification_sensitivity")
            Config.CLUSTER_THRESHOLD = self.get_int("cluster_threshold")
            Config.SEG_ADAPTIVE_BLOCK_SIZE = self.get_int("seg_adaptive_block_size")
            Config.SEG_ADAPTIVE_C = self.get_int("seg_adaptive_c")
            Config.SEG_MORPH_CLOSE_KERNEL = self.get_int("seg_morph_close_kernel")
            Config.SEG_MORPH_OPEN_KERNEL = self.get_int("seg_morph_open_kernel")
            Config.SEG_SMOOTH_KERNEL = self.get_int("seg_smooth_kernel")
            
            Config.DENSITY_THRESHOLDS = {
                "A": self.get_float("density_threshold_a"),
                "B": self.get_float("density_threshold_b"),
                "C": self.get_float("density_threshold_c"),
            }
            
            Config.BIRADS_THRESHOLDS = {
                "birads_2_max": self.get_float("birads_2_max"),
                "birads_3_max": self.get_float("birads_3_max"),
                "birads_4_max": self.get_float("birads_4_max"),
            }
            
            Config.RISK_WEIGHTS = {
                "shape_irregularity": self.get_float("risk_weight_shape"),
                "circularity": 0.10, # Keep hardcoded/fixed if not in settings
                "spiculation": self.get_float("risk_weight_spiculation"),
                "edge_sharpness": 0.10,
                "contrast": self.get_float("risk_weight_contrast"),
                "texture_entropy": self.get_float("risk_weight_entropy"),
                "density_difference": self.get_float("risk_weight_density"),
            }
            
            Config.DEBUG_MODE = self.get_bool("debug_mode")
            
        except Exception as e:
            logger.error(f"Failed to apply settings to Config: {e}")

    def save_geometry(self, geometry: Any) -> None:
        self.set("geometry", geometry)

    def load_geometry(self) -> Any:
        return self.get("geometry")
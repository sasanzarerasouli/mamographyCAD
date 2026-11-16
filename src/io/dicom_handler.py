"""DICOM metadata extraction and image loading.

The loader is the single place in the application that understands DICOM
photometric interpretation, bit depth, signedness and VOI windowing.  It always
returns an 8-bit grayscale array in *display orientation* (bright = dense) plus
a metadata dictionary, so every downstream module can assume a consistent
uint8 image regardless of which modality/file type produced it.

Supported inputs
----------------
* DICOM (``.dcm``/``.dicom`` or any file with a DICOM preamble):
  - ``MONOCHROME1`` (inverted) and ``MONOCHROME2``
  - ``BitsAllocated`` 8/16/32, ``BitsStored``/``HighBit``
  - signed and unsigned pixel representations
  - ``RescaleSlope``/``RescaleIntercept``, ``WindowCenter``/``WindowWidth``,
    ``VOILUTFunction``, ``PhotometricInterpretation=RGB`` (rare for MG)
* Standard raster images (PNG/JPEG/TIFF/BMP) including 16-bit TIFFs.
"""
import logging
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("MammographyCAD.dicom")


class DICOMHandler:
    """Extract patient, study, and image metadata from DICOM files."""

    DICOM_EXTENSIONS = (".dcm", ".dicom", ".ima")

    # ------------------------------------------------------------------ #
    #  Detection / metadata                                              #
    # ------------------------------------------------------------------ #
    def is_dicom(self, path: str) -> bool:
        """Return True when *path* should be treated as a DICOM file.

        Extension based detection is complemented by a magic-number check so
        extension-less DICOM exports are still recognised.
        """
        if not path or not os.path.isfile(path):
            return False
        if path.lower().endswith(self.DICOM_EXTENSIONS):
            return True
        try:
            with open(path, "rb") as fh:
                fh.seek(128)
                return fh.read(4) == b"DICM"
        except OSError:
            return False

    def extract_metadata(self, path: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "file_path": path,
            "file_name": os.path.basename(path),
            "patient": {},
            "study": {},
            "image": {},
            "source": "unknown",
        }
        if not self.is_dicom(path):
            return meta
        try:
            import pydicom
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            meta["source"] = "dicom"

            def _s(value, default=""):
                if value is None:
                    return default
                text = str(value).strip()
                return text if text else default

            meta["patient"] = {
                "patient_id": _s(getattr(ds, "PatientID", "")),
                "patient_name": _s(getattr(ds, "PatientName", "")),
                "patient_sex": _s(getattr(ds, "PatientSex", "")),
                "patient_age": _s(getattr(ds, "PatientAge", "")),
            }
            meta["study"] = {
                "study_date": _s(getattr(ds, "StudyDate", "")),
                "study_time": _s(getattr(ds, "StudyTime", "")),
                "study_description": _s(getattr(ds, "StudyDescription", "")),
                "series_description": _s(getattr(ds, "SeriesDescription", "")),
                "modality": _s(getattr(ds, "Modality", "")),
                "institution": _s(getattr(ds, "InstitutionName", "")),
                "manufacturer": _s(getattr(ds, "Manufacturer", "")),
                "accession_number": _s(getattr(ds, "AccessionNumber", "")),
            }

            pixel_spacing = self._read_pixel_spacing(ds)
            meta["image"] = {
                "pixel_spacing": list(pixel_spacing),
                "view_position": _s(getattr(ds, "ViewPosition", "")).upper(),
                "laterality": _s(getattr(ds, "ImageLaterality", "")).upper(),
                "rows": int(getattr(ds, "Rows", 0) or 0),
                "columns": int(getattr(ds, "Columns", 0) or 0),
                "photometric_interpretation": _s(
                    getattr(ds, "PhotometricInterpretation", "")
                ).upper(),
                "bits_allocated": int(getattr(ds, "BitsAllocated", 0) or 0),
                "bits_stored": int(getattr(ds, "BitsStored", 0) or 0),
                "pixel_representation": int(
                    getattr(ds, "PixelRepresentation", 0) or 0
                ),
                "rescale_slope": float(getattr(ds, "RescaleSlope", 1.0) or 1.0),
                "rescale_intercept": float(
                    getattr(ds, "RescaleIntercept", 0.0) or 0.0
                ),
                "window_center": self._first_number(
                    getattr(ds, "WindowCenter", None)
                ),
                "window_width": self._first_number(
                    getattr(ds, "WindowWidth", None)
                ),
                "voi_lut_function": _s(
                    getattr(ds, "VOILUTFunction", "")
                ).upper(),
                "body_part": _s(getattr(ds, "BodyPartExamined", "")),
                "target": _s(getattr(ds, "FilterType", "")),
                "implant_present": _s(getattr(ds, "ImplantPresent", "")),
                "compression": _s(
                    getattr(ds, "LossyImageCompression", "")
                ),
            }
            logger.info(
                "DICOM metadata extracted: patient=%s view=%s laterality=%s",
                meta["patient"].get("patient_id") or "<none>",
                meta["image"]["view_position"] or "<none>",
                meta["image"]["laterality"] or "<none>",
            )
        except ImportError:
            logger.warning("pydicom not installed; limited DICOM support")
        except Exception as exc:  # pragma: no cover - corrupt files
            logger.error("DICOM metadata extraction failed: %s", exc)
        return meta

    # ------------------------------------------------------------------ #
    #  Loading                                                           #
    # ------------------------------------------------------------------ #
    def load_image(self, path: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Load an image; returns ``(BGR image, metadata)``.

        Kept for backward compatibility with the original API. New code should
        prefer :meth:`load_grayscale` which avoids the needless BGR conversion
        and exposes the raw 8-bit display array.
        """
        gray, metadata = self.load_grayscale(path)
        if gray is None:
            return None, metadata
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), metadata

    def load_grayscale(
        self, path: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Load *path* as an 8-bit, display-oriented grayscale array."""
        metadata = self.extract_metadata(path)

        if metadata.get("source") == "dicom" or self.is_dicom(path):
            gray = self._load_dicom_grayscale(path, metadata)
            if gray is not None:
                metadata["source"] = "dicom"
                return gray, metadata
            logger.warning(
                "DICOM decoding failed, falling back to raster reader: %s", path
            )

        gray = self._load_raster_grayscale(path)
        if gray is not None:
            metadata.setdefault("source", "standard")
            metadata["source"] = "standard"
            return gray, metadata

        logger.error("Unable to read image: %s", path)
        return None, metadata

    # ------------------------------------------------------------------ #
    #  Internals                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_number(value) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, (list, tuple)):
                return float(value[0])
            # pydicom MultiValue
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                seq = list(value)
                if seq:
                    return float(seq[0])
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _read_pixel_spacing(ds) -> Tuple[float, float]:
        spacing = getattr(ds, "PixelSpacing", None)
        if spacing is None:
            # Some whole-slide style objects use ImagerPixelSpacing
            spacing = getattr(ds, "ImagerPixelSpacing", None)
        try:
            if spacing is not None and len(spacing) >= 2:
                row_spacing, col_spacing = float(spacing[0]), float(spacing[1])
                # A spacing of exactly 0 would poison every mm measurement.
                if row_spacing <= 0 or col_spacing <= 0:
                    raise ValueError("non-positive pixel spacing")
                return row_spacing, col_spacing
        except (TypeError, ValueError):
            pass
        return 1.0, 1.0

    def _load_dicom_grayscale(
        self, path: str, metadata: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        try:
            import pydicom
        except ImportError:
            logger.warning("pydicom is not installed; cannot decode %s", path)
            return None

        try:
            ds = pydicom.dcmread(path)
            array = np.asarray(ds.pixel_array)
        except Exception as exc:
            logger.error("DICOM pixel data decode failed (%s): %s", path, exc)
            return None

        image_meta = metadata.setdefault("image", {})
        photometric = str(
            getattr(ds, "PhotometricInterpretation", "")
        ).upper() or image_meta.get("photometric_interpretation", "")
        metadata["image"]["photometric_interpretation"] = photometric

        if array.ndim == 3:
            # Colour (very unusual for mammography) or multi-frame: keep first
            # frame for multi-frame objects, convert colour to luminance.
            if array.shape[-1] in (3, 4):
                array = cv2.cvtColor(
                    array[..., :3].astype(np.float32).clip(0, 255).astype(np.uint8),
                    cv2.COLOR_RGB2GRAY,
                )
            else:
                array = array[0]
        if array.ndim != 2:
            logger.error("Unsupported DICOM pixel array shape: %s", array.shape)
            return None

        gray = self._to_display_uint8(array, ds, image_meta)

        if photometric == "MONOCHROME1":
            # MONOCHROME1 stores dense tissue as *low* values; invert so that
            # the rest of the pipeline (and the viewer) sees bright = dense.
            gray = cv2.bitwise_not(gray)
            image_meta["monochrome1_inverted"] = True
        else:
            image_meta["monochrome1_inverted"] = False

        return np.ascontiguousarray(gray, dtype=np.uint8)

    def _to_display_uint8(
        self, array: np.ndarray, ds, image_meta: Dict[str, Any]
    ) -> np.ndarray:
        """Convert raw pixel data to 8-bit using stored bit depth and VOI LUT."""
        bits_stored = int(getattr(ds, "BitsStored", 0) or 0)
        bits_allocated = int(getattr(ds, "BitsAllocated", 0) or 0)
        signed = int(getattr(ds, "PixelRepresentation", 0) or 0) == 1

        data = array.astype(np.float64, copy=True)

        # 1. Apply the modality rescale (may legitimately be negative slope).
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if slope != 1.0 or intercept != 0.0:
            data = data * slope + intercept

        # 2. Normalise away the stored bit depth so that a 12-bit-in-16-bit
        #    image is not crushed into the top 1/16 of the 8-bit range.
        if bits_stored and 0 < bits_stored < 16 and not signed:
            max_stored = float((1 << bits_stored) - 1)
            if data.max() > max_stored:
                data = np.clip(data, 0, max_stored)
        image_meta["bits_allocated_used"] = bits_allocated or bits_stored

        # 3. VOI LUT / windowing when present.
        center = self._first_number(getattr(ds, "WindowCenter", None))
        width = self._first_number(getattr(ds, "WindowWidth", None))
        if center is not None and width is not None and width > 0:
            low = center - width / 2.0
            high = center + width / 2.0
            data = np.clip(data, low, high)
            image_meta["windowed"] = True
        else:
            image_meta["windowed"] = False

        # 4. Robust percentile stretch: mammograms are dominated by nearly
        #    uniform background, so min/max would waste most of the 8-bit
        #    range. Percentiles keep the parenchyma contrast intact.
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return np.zeros(array.shape, dtype=np.uint8)
        lo, hi = np.percentile(finite, (0.5, 99.5))
        if hi - lo < 1e-6:
            lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1e-6:
            return np.zeros(array.shape, dtype=np.uint8)
        scaled = (data - lo) / (hi - lo)
        scaled = np.clip(scaled, 0.0, 1.0)
        image_meta["intensity_range"] = [float(lo), float(hi)]
        image_meta["original_dtype"] = str(array.dtype)
        return (scaled * 255.0).round().astype(np.uint8)

    def _load_raster_grayscale(self, path: str) -> Optional[np.ndarray]:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype == np.uint8:
            return img
        # 16-bit / float raster: robust percentile stretch to 8-bit
        data = img.astype(np.float64)
        lo, hi = np.percentile(data, (0.5, 99.5))
        if hi - lo < 1e-6:
            lo, hi = float(data.min()), float(data.max())
        if hi - lo < 1e-6:
            return np.zeros(img.shape, dtype=np.uint8)
        scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        return (scaled * 255.0).round().astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  Legacy helpers (kept for compatibility)                           #
    # ------------------------------------------------------------------ #
    def _load_dicom(self, path: str) -> Optional[np.ndarray]:
        gray, _ = self.load_grayscale(path)
        if gray is None:
            return None
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def get_pixel_spacing(self, metadata: Dict[str, Any]) -> Tuple[float, float]:
        """Return ``(row_spacing_mm, column_spacing_mm)``."""
        spacing = metadata.get("image", {}).get("pixel_spacing", [1.0, 1.0])
        try:
            if spacing is not None and len(spacing) >= 2:
                row = float(spacing[0])
                col = float(spacing[1])
                if row > 0 and col > 0:
                    return row, col
        except (TypeError, ValueError):
            pass
        return 1.0, 1.0

    def is_spacing_known(self, metadata: Dict[str, Any]) -> bool:
        row, col = self.get_pixel_spacing(metadata)
        return not (abs(row - 1.0) < 1e-9 and abs(col - 1.0) < 1e-9)

    def get_view_position(self, metadata: Dict[str, Any]) -> str:
        """Return ``'MLO'``, ``'CC'`` or ``''`` when unknown."""
        view = str(metadata.get("image", {}).get("view_position", "") or "").upper()
        if view.startswith("MLO"):
            return "MLO"
        if view.startswith("CC"):
            return "CC"
        return view

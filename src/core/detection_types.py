from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List


class DetectionType(Enum):
    TUMOR_CONFIRMED = "tumor_confirmed"
    TUMOR_SUSPICIOUS = "tumor_suspicious"
    MACROCALCIFICATION = "macrocalcification"
    MICROCALCIFICATION = "microcalcification"


@dataclass
class Detection:
    type: DetectionType
    location: Tuple[int, int]
    bbox: Optional[Tuple[int, int, int, int]] = None
    mask: Optional["np.ndarray"] = None
    confidence: float = 0.0
    features: Dict = field(default_factory=dict)
    id: int = 0
    lesion_id: str = ""
    is_ignored: bool = False
    diameter_px: float = 0.0
    diameter_mm: float = 0.0
    area_px: float = 0.0
    area_mm2: float = 0.0
    contour: Optional[List] = None
    explainability: Dict = field(default_factory=dict)
    risk_scores: Dict = field(default_factory=dict)
    uncertainty: float = 0.0
    reliability: float = 0.0

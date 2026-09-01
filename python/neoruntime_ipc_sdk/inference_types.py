"""Data types describing inference results and model metadata."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "BatchInferItem",
    "BoundingBox",
    "Classification",
    "DepthMap",
    "DetectedObject",
    "Embedding",
    "InferenceResult",
    "LandmarkPoint",
    "LandmarkSet",
    "ModelInfo",
    "OcrLine",
    "SegmentationMask",
]


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def to_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class DetectedObject:
    label: str
    score: float
    bbox: BoundingBox
    class_id: int = 0
    track_id: Optional[int] = None


@dataclass
class LandmarkPoint:
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class LandmarkSet:
    type: str
    points: List[LandmarkPoint] = field(default_factory=list)


@dataclass
class Classification:
    type: str
    class_id: int
    label: str
    confidence: float


@dataclass
class SegmentationMask:
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox
    mask_rle: bytes
    mask_width: int
    mask_height: int

    def to_numpy_mask(self) -> np.ndarray:
        """Decode RLE to HxW bool numpy array."""
        mask = np.zeros(self.mask_width * self.mask_height, dtype=bool)
        i = 0
        data = self.mask_rle
        while i < len(data):
            # Decode varint
            def read_varint(pos):
                val = 0
                shift = 0
                while pos < len(data):
                    b = data[pos]
                    val |= (b & 0x7F) << shift
                    pos += 1
                    if not (b & 0x80):
                        break
                    shift += 7
                return val, pos
            start, i = read_varint(i)
            length, i = read_varint(i)
            mask[start:start + length] = True
        return mask.reshape(self.mask_height, self.mask_width)


@dataclass
class OcrLine:
    text: str
    confidence: float
    bbox: BoundingBox


@dataclass
class Embedding:
    dim: int
    data: List[float]


@dataclass
class DepthMap:
    width: int
    height: int
    data: np.ndarray  # float32 (H, W)


@dataclass
class InferenceResult:
    frame_sequence: int
    timestamp_ns: int
    objects: List[DetectedObject] = field(default_factory=list)
    classifications: List[Classification] = field(default_factory=list)
    landmarks: List[LandmarkSet] = field(default_factory=list)
    masks: List[SegmentationMask] = field(default_factory=list)
    ocr_lines: List[OcrLine] = field(default_factory=list)
    embeddings: List[Embedding] = field(default_factory=list)
    depth_maps: List[DepthMap] = field(default_factory=list)
    raw_outputs: Optional[List[np.ndarray]] = None
    infer_time_us: int = 0
    queue_time_us: int = 0
    hw_infer_time_us: int = 0  # Pure NPU hardware latency (microseconds), 0 if unavailable
    status_message: str = ""  # Diagnostic: "simulation" if no frame source
    
    def has_person(self) -> bool:
        return any(obj.label == "person" for obj in self.objects)
    
    def count_by_label(self, label: str) -> int:
        return sum(1 for obj in self.objects if obj.label == label)
    
    def get_objects_by_label(self, label: str) -> List[DetectedObject]:
        return [obj for obj in self.objects if obj.label == label]


@dataclass
class ModelInfo:
    model_id: str
    model_path: str
    version: str = ""
    inputs: List[Dict] = field(default_factory=list)
    outputs: List[Dict] = field(default_factory=list)
    estimated_tops: float = 0.0
    estimated_memory: int = 0
    load_timestamp: int = 0


@dataclass
class BatchInferItem:
    """A single inference request within a batch."""
    image: np.ndarray
    model_id: str
    timeout_ms: int = 5000
    priority: int = 4

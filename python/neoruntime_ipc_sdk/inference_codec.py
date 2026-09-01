"""Inference tensor codec: pb2 <-> numpy/dataclass translation.

Pure functions over ``inference_pb2`` messages — no client state, so
they run identically in tests and in the client's hot paths.
"""

from __future__ import annotations

import numpy as np

from .inference_types import (
    BoundingBox,
    Classification,
    DepthMap,
    DetectedObject,
    Embedding,
    InferenceResult,
    LandmarkPoint,
    LandmarkSet,
    OcrLine,
    SegmentationMask,
)
from .proto import inference_pb2


def _numpy_to_tensor(arr: np.ndarray, name: str = "") -> inference_pb2.Tensor:
    dtype_map = {
        np.uint8: inference_pb2.UINT8,
        np.int8: inference_pb2.INT8,
        np.uint16: inference_pb2.UINT16,
        np.int16: inference_pb2.INT16,
        np.float16: inference_pb2.FLOAT16,
        np.float32: inference_pb2.FLOAT32,
        np.int32: inference_pb2.INT32,
        np.uint32: inference_pb2.UINT32,
    }

    dtype = dtype_map.get(arr.dtype.type, inference_pb2.FLOAT32)

    return inference_pb2.Tensor(shape=list(arr.shape), dtype=dtype, data=arr.tobytes())


def _tensor_to_numpy(self, tensor: inference_pb2.Tensor) -> np.ndarray:
    dtype_map = {
        inference_pb2.UINT8: np.uint8,
        inference_pb2.INT8: np.int8,
        inference_pb2.UINT16: np.uint16,
        inference_pb2.INT16: np.int16,
        inference_pb2.FLOAT16: np.float16,
        inference_pb2.FLOAT32: np.float32,
        inference_pb2.INT32: np.int32,
        inference_pb2.UINT32: np.uint32,
    }

    dtype = dtype_map.get(tensor.dtype, np.float32)
    arr = np.frombuffer(tensor.data, dtype=dtype)

    # Handle empty or invalid shape - keep as 1D array
    if tensor.shape and len(tensor.shape) > 0:
        try:
            return arr.reshape(tensor.shape)
        except ValueError:
            # Shape doesn't match data size, return flat array
            return arr
    return arr


def _parse_post_result(
    self, post_result: inference_pb2.PostResult
) -> tuple[
    list[DetectedObject],
    list[Classification],
    list[LandmarkSet],
    list[SegmentationMask],
    list[OcrLine],
    list[Embedding],
    list[DepthMap],
]:
    objects = []
    for det in post_result.detections:
        obj = DetectedObject(
            label=det.label,
            score=det.confidence,
            bbox=BoundingBox(x=det.bbox.x, y=det.bbox.y, width=det.bbox.w, height=det.bbox.h),
            class_id=det.class_id,
        )
        objects.append(obj)

    classifications = []
    for cls in post_result.classifications:
        classifications.append(
            Classification(
                type=cls.type, class_id=cls.class_id, label=cls.label, confidence=cls.confidence
            )
        )

    landmarks = []
    for lm_set in post_result.landmarks:
        points = [LandmarkPoint(x=p.x, y=p.y, confidence=p.confidence) for p in lm_set.points]
        landmarks.append(LandmarkSet(type=lm_set.type, points=points))

    masks = []
    for m in post_result.masks:
        masks.append(
            SegmentationMask(
                class_id=m.class_id,
                label=m.label,
                confidence=m.confidence,
                bbox=BoundingBox(x=m.bbox.x, y=m.bbox.y, width=m.bbox.w, height=m.bbox.h),
                mask_rle=m.mask_rle,
                mask_width=m.mask_width,
                mask_height=m.mask_height,
            )
        )

    ocr_lines = []
    for line in post_result.ocr_lines:
        ocr_lines.append(
            OcrLine(
                text=line.text,
                confidence=line.confidence,
                bbox=BoundingBox(
                    x=line.bbox.x, y=line.bbox.y, width=line.bbox.w, height=line.bbox.h
                ),
            )
        )

    embeddings = []
    for emb in post_result.embeddings:
        embeddings.append(Embedding(dim=emb.dim, data=list(emb.data)))

    depth_maps = []
    for dm in post_result.depth_maps:
        arr = np.frombuffer(dm.depth_data, dtype=np.float32).reshape(dm.height, dm.width)
        depth_maps.append(DepthMap(width=dm.width, height=dm.height, data=arr.copy()))

    return objects, classifications, landmarks, masks, ocr_lines, embeddings, depth_maps


def _parse_infer_response(self, response: inference_pb2.InferResponse) -> InferenceResult:
    """Parse an InferResponse proto into an InferenceResult dataclass.

    Shared by infer() and infer_batch() to avoid duplication.
    """
    objects: list[DetectedObject] = []
    classifications: list[Classification] = []
    landmarks: list[LandmarkSet] = []
    masks: list[SegmentationMask] = []
    ocr_lines: list[OcrLine] = []
    embeddings: list[Embedding] = []
    depth_maps: list[DepthMap] = []

    try:
        if response.HasField("post_result"):
            objects, classifications, landmarks, masks, ocr_lines, embeddings, depth_maps = (
                _parse_post_result(response.post_result)
            )
    except (ValueError, AttributeError):
        pass

    raw_outputs = None
    if response.outputs:
        raw_outputs = [_tensor_to_numpy(t) for t in response.outputs]

    return InferenceResult(
        frame_sequence=0,
        timestamp_ns=0,
        objects=objects,
        classifications=classifications,
        landmarks=landmarks,
        masks=masks,
        ocr_lines=ocr_lines,
        embeddings=embeddings,
        depth_maps=depth_maps,
        raw_outputs=raw_outputs,
        infer_time_us=response.infer_time_us,
        queue_time_us=response.queue_time_us,
        hw_infer_time_us=getattr(response, "hw_infer_time_us", 0),
        status_message=response.status.message if not response.status.success else "",
    )

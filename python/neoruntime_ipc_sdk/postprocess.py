"""
Post-processing helpers for raw model outputs — greedy NMS in numpy,
plus decode adapters for the common YOLO export shapes.

Streaming inference on the platform applies NMS server-side when the
model is registered with NMS parameters, but single-shot ``infer()``
results, custom decode heads and app-side re-filtering still need a local
implementation. Apps currently hand-roll one whenever scores and boxes
arrive separately.

Pure numpy, no accelerator — the vectorised IoU matrix is already
memory-bound at the sizes detector heads produce. When ai-runtime exposes
its create-time NMS registration over the service layer (see
docs/proposals/sdk-hardware-routing.md), prefer that and keep this for
client-side filtering.

The :class:`YoloV8Postprocessor` / :class:`YoloV5Postprocessor` adapters
decode ``InferenceResult.raw_outputs`` from models whose postprocess is
*not* handled server-side (generic plugin exports — the ones whose
thresholds are baked in at export time and cannot be tuned via
``update_postprocess_config``). Thresholds here are constructor
parameters, i.e. tunable at runtime.

.. code-block:: python

    from neoruntime_ipc_sdk.postprocess import YoloV8Postprocessor, nms

    keep = nms(boxes_xyxy, scores, iou_threshold=0.45, class_ids=cls)
    boxes, scores = boxes[keep], scores[keep]

    post = YoloV8Postprocessor(labels=["person", "car"], score_threshold=0.3)
    objects = post(result.raw_outputs, meta)      # list[DetectedObject]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .inference_types import BoundingBox, DetectedObject

__all__ = ["nms", "Postprocessor", "YoloV5Postprocessor", "YoloV8Postprocessor"]


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU of an ``(N, 4)`` xyxy array → ``(N, N)`` float."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    union = area[:, None] + area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    class_ids: np.ndarray | None = None,
) -> list[int]:
    """Greedy non-maximum suppression.

    Args:
        boxes: ``(N, 4)`` xyxy boxes ``[x1, y1, x2, y2]``.
        scores: ``(N,)`` confidence scores.
        iou_threshold: boxes overlapping a kept box by more than this are
            suppressed.
        class_ids: optional ``(N,)`` class labels — boxes of different
            classes never suppress each other.

    Returns:
        Indices of the kept boxes, in descending-score order.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(
            f"boxes ({boxes.shape[0]}) and scores ({scores.shape[0]}) length mismatch"
        )
    if boxes.shape[0] == 0:
        return []
    if not 0 <= iou_threshold <= 1:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    iou = _iou_matrix(boxes)
    if class_ids is not None:
        class_ids = np.asarray(class_ids).reshape(-1)
        if class_ids.shape[0] != boxes.shape[0]:
            raise ValueError(
                f"class_ids ({class_ids.shape[0]}) and boxes ({boxes.shape[0]}) "
                "length mismatch"
            )
        # boxes of different classes never suppress each other
        iou = iou * (class_ids[:, None] == class_ids[None, :])

    order = np.argsort(-scores)
    suppressed = np.zeros(len(order), dtype=bool)
    keep: list[int] = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        suppressed |= iou[idx] > iou_threshold
        suppressed[idx] = True  # self, needed when iou_threshold == 1.0
    return keep


def _squeeze_batch(arr: Any) -> np.ndarray:
    """Drop leading size-1 (batch) dimensions down to a 2D head matrix."""
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


class Postprocessor(ABC):
    """Base class for client-side ``raw_outputs`` decoders.

    Subclasses take the list of raw output tensors plus the optional
    :class:`~neoruntime_ipc_sdk.preprocess.PreprocessMeta` from the
    matching :class:`~neoruntime_ipc_sdk.preprocess.Preprocessor` call
    and return domain objects. Boxes come back in *source-frame* pixel
    coordinates whenever a meta is provided; without one they stay in
    model-input coordinates.
    """

    @abstractmethod
    def __call__(
        self,
        raw_outputs: list[np.ndarray] | None,
        meta: Any | None = None,
    ) -> Any:
        raise NotImplementedError


class _YoloPostprocessorBase(Postprocessor):
    """Shared plumbing: thresholds, labels, NMS hand-off, object build.

    When a ``meta`` is passed, boxes come back in source-frame coordinates
    and (unless ``clip=False``) are clipped to the source frame — boxes
    detected in the letterbox pad would otherwise carry negative or
    beyond-frame coordinates. Boxes with nothing left after clipping are
    dropped.
    """

    def __init__(
        self,
        labels: list[str] | None = None,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        clip: bool = True,
    ):
        self.labels = list(labels) if labels else []
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.clip = bool(clip)

    def _label(self, class_id: int) -> str:
        return self.labels[class_id] if class_id < len(self.labels) else str(class_id)

    def _finish(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        meta: Any | None,
    ) -> list[DetectedObject]:
        keep = scores >= self.score_threshold
        if not keep.any():
            return []
        boxes = boxes_xyxy[keep]
        kept_scores = scores[keep]
        kept_classes = class_ids[keep]
        order = nms(boxes, kept_scores, self.iou_threshold, class_ids=kept_classes)

        objects: list[DetectedObject] = []
        for i in order:
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            if meta is not None:
                x1, y1, x2, y2 = meta.to_source_box((x1, y1, x2, y2))
                if self.clip:
                    src_w, src_h = meta.original_size
                    x1 = max(0.0, x1)
                    y1 = max(0.0, y1)
                    x2 = min(float(src_w), x2)
                    y2 = min(float(src_h), y2)
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue  # entirely in the letterbox pad — nothing real
            objects.append(
                DetectedObject(
                    label=self._label(int(kept_classes[i])),
                    score=float(kept_scores[i]),
                    bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                    class_id=int(kept_classes[i]),
                )
            )
        return objects


class YoloV8Postprocessor(_YoloPostprocessorBase):
    """Decode YOLOv8-family heads (anchor-free, ONNX-standard export).

    Expects ``raw_outputs[0]`` shaped ``(1, 4+nc, N)`` / ``(4+nc, N)``
    (or the transposed variant): boxes as ``cx, cy, w, h`` in input-pixel
    coordinates plus one score column per class, no objectness. Raw
    anchor-grid heads (pre-decode exports) are out of scope, as are
    non-dequantised int8 outputs — standard exports dequantise to float.
    """

    def __call__(
        self,
        raw_outputs: list[np.ndarray] | None,
        meta: Any | None = None,
    ) -> list[DetectedObject]:
        if not raw_outputs:
            return []
        arr = _squeeze_batch(raw_outputs[0])
        if arr.ndim != 2:
            raise ValueError(f"expected a 2D head matrix, got shape {arr.shape}")
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T  # the long axis is the anchor count N
        channels, n = arr.shape
        if channels == 0 or n == 0:
            return []
        nc = channels - 4
        if nc <= 0:
            raise ValueError(f"head has {channels} channels, need at least 5 (4 box + classes)")

        cxcywh = arr[:4, :].T
        boxes_xyxy = np.empty_like(cxcywh)
        boxes_xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

        class_ids = np.argmax(arr[4:, :], axis=0)
        scores = arr[4 + class_ids, np.arange(n)]
        return self._finish(boxes_xyxy, scores.astype(np.float32), class_ids, meta)


class YoloV5Postprocessor(_YoloPostprocessorBase):
    """Decode YOLOv5-family heads (anchor-based, ONNX-standard export).

    Expects ``raw_outputs[0]`` shaped ``(1, N, 5+nc)`` / ``(N, 5+nc)``
    (or the transposed variant): ``cx, cy, w, h, objectness`` plus one
    class score column. The final score is ``objectness * class_score``.
    Same scope limits as :class:`YoloV8Postprocessor`: decoded-grid
    exports only, float outputs assumed.
    """

    def __call__(
        self,
        raw_outputs: list[np.ndarray] | None,
        meta: Any | None = None,
    ) -> list[DetectedObject]:
        if not raw_outputs:
            return []
        arr = _squeeze_batch(raw_outputs[0])
        if arr.ndim != 2:
            raise ValueError(f"expected a 2D head matrix, got shape {arr.shape}")
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T  # the long axis is the anchor count N
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            return []
        if arr.shape[1] < 6:
            raise ValueError(
                f"head has {arr.shape[1]} columns, need at least 6 "
                "(4 box + objectness + classes)"
            )

        cxcywh = arr[:, :4]
        boxes_xyxy = np.empty_like(cxcywh)
        boxes_xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

        objectness = arr[:, 4]
        class_ids = np.argmax(arr[:, 5:], axis=1)
        scores = objectness * arr[np.arange(arr.shape[0]), 5 + class_ids]
        return self._finish(boxes_xyxy, scores.astype(np.float32), class_ids, meta)

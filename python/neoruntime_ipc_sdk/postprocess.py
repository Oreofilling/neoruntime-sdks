"""
Post-processing helpers for raw model outputs — greedy NMS in numpy.

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

.. code-block:: python

    from neoruntime_ipc_sdk.postprocess import nms

    keep = nms(boxes_xyxy, scores, iou_threshold=0.45, class_ids=cls)
    boxes, scores = boxes[keep], scores[keep]
"""

from __future__ import annotations

import numpy as np

__all__ = ["nms"]


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

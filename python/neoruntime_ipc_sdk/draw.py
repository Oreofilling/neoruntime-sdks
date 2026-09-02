"""
Drawing utilities - annotate RGB numpy arrays with detection boxes and text.

All functions take an RGB uint8 array (H, W, 3) and return a NEW array;
the input is never modified. cv2 accelerates rendering when installed,
otherwise Pillow (a hard SDK dependency) is used.

Example:
    frame = client.get_frame("main")            # Frame
    rgb = draw_detections(frame.to_rgb(), result)
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Distinct palette used by draw_detections when color is None (indexed by class_id)
PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),  # green
    (255, 80, 80),  # red
    (80, 160, 255),  # blue
    (255, 220, 0),  # yellow
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
    (180, 120, 255),  # purple
    (255, 140, 0),  # orange
)


def _to_xyxy(box) -> tuple[float, float, float, float]:
    """Coerce a box into (x1, y1, x2, y2).

    Accepts 4-sequences, BoundingBox-like objects (with to_xyxy()),
    or objects carrying .bbox.
    """
    if hasattr(box, "bbox") and not hasattr(box, "to_xyxy"):
        box = box.bbox
    if hasattr(box, "to_xyxy"):
        return box.to_xyxy()
    x1, y1, x2, y2 = box
    return x1, y1, x2, y2


def _format_label(label: str | None, score) -> str | None:
    if label is None:
        return f"{score:.2f}" if score is not None else None
    if score is not None:
        return f"{label} {score:.2f}"
    return label


def draw_boxes(
    image: np.ndarray,
    boxes: Iterable,
    labels: Sequence[str | None] | None = None,
    scores: Sequence[float] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes (pixel coordinates) on an RGB array copy.

    Args:
        image: RGB uint8 array (H, W, 3).
        boxes: iterable of (x1, y1, x2, y2) or BoundingBox-like objects.
        labels: optional per-box text (combined with scores when given).
        scores: optional per-box confidence.
        color: RGB box color.
        thickness: line thickness in pixels.

    Returns: new RGB array; the input array is not modified.
    """
    out = image.copy()
    thickness = max(1, int(thickness))
    items = [_to_xyxy(b) for b in boxes]
    try:
        import cv2

        for i, (x1, y1, x2, y2) in enumerate(items):
            xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(out, (xi1, yi1), (xi2, yi2), color, thickness)
            text = _format_label(
                labels[i] if labels and i < len(labels) else None,
                scores[i] if scores and i < len(scores) else None,
            )
            if text:
                cv2.putText(
                    out,
                    text,
                    (xi1, max(12, yi1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return out
    except ImportError:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        for i, (x1, y1, x2, y2) in enumerate(items):
            draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=color, width=thickness)
            text = _format_label(
                labels[i] if labels and i < len(labels) else None,
                scores[i] if scores and i < len(scores) else None,
            )
            if text:
                draw.text((int(x1) + 2, max(0, int(y1) - 12)), text, fill=color)
        return np.array(pil)


def draw_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.5,
    thickness: int = 1,
) -> np.ndarray:
    """Draw a text string at pixel position xy on an RGB array copy."""
    out = image.copy()
    try:
        import cv2

        cv2.putText(
            out,
            str(text),
            (int(xy[0]), int(xy[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, int(thickness)),
            cv2.LINE_AA,
        )
        return out
    except ImportError:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(out)
        ImageDraw.Draw(pil).text((int(xy[0]), int(xy[1])), str(text), fill=color)
        return np.array(pil)


def draw_detections(
    image: np.ndarray,
    result_or_objects,
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Draw an InferenceResult (or a list of DetectedObject) on an RGB copy.

    Each object gets a box plus a "label score" caption. When color is
    None, a per-class color from PALETTE is chosen via class_id.
    """
    if hasattr(result_or_objects, "objects"):
        objects = list(result_or_objects.objects)
    else:
        objects = list(result_or_objects)
    if not objects:
        return image.copy()

    boxes: list[tuple[float, float, float, float]] = []
    labels: list[str | None] = []
    scores: list[float | None] = []
    colors: list[tuple[int, int, int]] = []
    for obj in objects:
        boxes.append(_to_xyxy(obj.bbox if hasattr(obj, "bbox") else obj))
        labels.append(getattr(obj, "label", None))
        scores.append(getattr(obj, "score", None))
        if color is not None:
            colors.append(color)
        else:
            class_id = getattr(obj, "class_id", 0) or 0
            colors.append(PALETTE[int(class_id) % len(PALETTE)])

    out = image.copy()
    for box, label, score, col in zip(boxes, labels, scores, colors):
        out = draw_boxes(out, [box], labels=[label], scores=[score], color=col, thickness=2)
    return out

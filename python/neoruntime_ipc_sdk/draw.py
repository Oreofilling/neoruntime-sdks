"""
Drawing utilities - annotate RGB numpy arrays with detection boxes and text.

All functions take an RGB uint8 array (H, W, 3) and return a NEW array;
the input is never modified. cv2 accelerates rendering when installed,
otherwise Pillow (a hard SDK dependency) is used.

render_overlay_rgba is the hardware companion: it renders the same
annotation as a minimal straight-alpha RGBA canvas (plus its frame
offset) for DspClient.blend_hw instead of rasterizing onto the pixels.

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


def _as_points(pts) -> np.ndarray:
    """Coerce polygon/track points into an (N, 2) int32 array."""
    arr = np.asarray(pts, np.int32)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        raise ValueError(
            f"polygon/track points must be (N, 2) with N >= 2, got shape {arr.shape}"
        )
    return arr


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

    Routing: NV12 2D arrays ride the accel router (DSP blend when the
    daemon is reachable, the CPU mirror otherwise); RGB arrays go
    straight to the software raster (no doomed hardware attempt, no
    fake degradation row); keep-fd frames raise — the zero-copy blend
    chain that would serve them is gated (state-dependent field wedge).
    """
    if getattr(image, "ndim", 0) == 2:
        from .accel import get_default_router  # noqa: PLC0415 — accel imports draw

        return get_default_router().run(
            "draw_detections", image, result_or_objects, color
        )
    if hasattr(image, "handle") or hasattr(image, "fds"):
        from .accel import HardwareUnavailable  # noqa: PLC0415

        raise HardwareUnavailable(
            "draw_detections takes NV12 arrays, not keep-fd frames: the "
            "zero-copy frame blend chain has wedged the DSP device-wide "
            "in the field (state-dependent; DspClient.blend_hw refuses "
            "it). Call frame.to_array() and route the array."
        )
    return _draw_detections_impl(image, result_or_objects, color)


def _draw_detections_impl(
    image: np.ndarray,
    result_or_objects,
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """CPU raster leg of ``draw_detections`` (boxes + captions on an RGB copy)."""
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


# px reserved above a box for its caption (cv2 scale-0.5 ascent + baseline
# offset 6 + margin) — same strip the software draw_boxes text occupies
_LABEL_STRIP = 20
# daemon DSP floor (dsp_wire._MIN_DIM): blend_hw pads transparently, but a
# canvas born legal avoids a copy at paste time
_MIN_OVERLAY_DIM = 16


def render_overlay_rgba(
    frame_w: int,
    frame_h: int,
    boxes: Iterable = (),
    labels: Sequence[str | None] | None = None,
    scores: Sequence[float | None] | None = None,
    colors: Sequence[tuple[int, int, int]] | None = None,
    thickness: int = 2,
    polygons: Sequence = (),
    tracks: Sequence = (),
) -> tuple[np.ndarray, int, int]:
    """Render boxes + captions + polygons + tracks as a minimal
    straight-alpha RGBA overlay.

    Returns ``(rgba, x0, y0)``: an ``(h, w, 4)`` uint8 canvas holding
    every shape and its top-left position on the frame. Hand it to
    :meth:`DspClient.blend_hw` for the hardware composite — the canvas
    is the union bbox of all shapes (clamped to the frame, floored at
    16x16), so untouched pixels never enter the blend and the quota
    footprint stays small.

    ``polygons`` is a sequence of ``(points, color)`` pairs drawn as
    closed outlines (zone shapes); ``tracks`` the same shape drawn open
    (trajectories). ``points`` is any (N, 2) sequence of frame pixel
    coordinates; ``color=None`` falls back to green like boxes.

    Straight alpha by construction: each shape is first drawn
    white-on-black as a coverage mask (cv2 LINE_AA captions give edge
    coverage t; rectangles stay hard-edged like :func:`draw_boxes`),
    then shapes are colored in draw order and alpha is the mask union —
    so compositing with ``t*C + (1-t)*base`` reproduces the software
    output. Colors default to green; pick from :data:`PALETTE` by
    class_id to match :func:`draw_detections`.
    """
    t = max(1, int(thickness))
    items = [_to_xyxy(b) for b in boxes]
    lines = [(p, c, True) for p, c in polygons] + [(p, c, False) for p, c in tracks]
    if not items and not lines:
        raise ValueError("render_overlay_rgba needs at least one shape")
    texts = [
        _format_label(
            labels[i] if labels and i < len(labels) else None,
            scores[i] if scores and i < len(scores) else None,
        )
        for i in range(len(items))
    ]
    # colors in draw order: boxes first, then polygons, then tracks — the
    # same order masks get appended below, so zip() stays aligned
    cols = [
        tuple(colors[i]) if colors and i < len(colors) else (0, 255, 0)
        for i in range(len(items))
    ]
    cols += [
        tuple(c) if c is not None else (0, 255, 0) for _p, c, _closed in lines
    ]
    line_shapes = [
        (_as_points(p), c, closed) for p, c, closed in lines
    ]

    # canvas = union bbox of strokes + caption strips, clamped to the frame
    x_min, y_min, x_max, y_max = int(frame_w), int(frame_h), 0, 0
    rects = []
    for (x1, y1, x2, y2), text in zip(items, texts):
        xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
        rects.append((xi1, yi1, xi2, yi2, text))
        x_min = min(x_min, xi1 - t - 1)
        y_min = min(y_min, yi1 - t - 1 - (_LABEL_STRIP if text else 0))
        x_max = max(x_max, xi2 + t + 1)
        y_max = max(y_max, yi2 + t + 1)
    for pts, _col, _closed in line_shapes:
        x_min = min(x_min, int(pts[:, 0].min()) - t - 1)
        y_min = min(y_min, int(pts[:, 1].min()) - t - 1)
        x_max = max(x_max, int(pts[:, 0].max()) + t + 1)
        y_max = max(y_max, int(pts[:, 1].max()) + t + 1)
    x0 = max(0, x_min)
    y0 = max(0, y_min)
    x_end = min(int(frame_w), x_max)
    y_end = min(int(frame_h), y_max)
    if x_end <= x0 or y_end <= y0:
        raise ValueError("all shapes lie outside the frame")
    cw = max(_MIN_OVERLAY_DIM, x_end - x0)
    ch = max(_MIN_OVERLAY_DIM, y_end - y0)

    masks: list[np.ndarray] = []
    try:
        import cv2

        for xi1, yi1, xi2, yi2, text in rects:
            m = np.zeros((ch, cw), np.uint8)
            # rectangle default LINE_8, matching draw_boxes' hard edges
            cv2.rectangle(m, (xi1 - x0, yi1 - y0), (xi2 - x0, yi2 - y0), 255, t)
            if text:
                ty = min(max(yi1 - 6 - y0, 12), ch - 2)
                cv2.putText(
                    m, text, (xi1 - x0, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA,
                )
            masks.append(m)
        for pts, _col, closed in line_shapes:
            m = np.zeros((ch, cw), np.uint8)
            shifted = (pts - np.array([x0, y0], np.int32)).reshape(-1, 1, 2)
            cv2.polylines(m, [shifted], closed, 255, t)
            masks.append(m)
    except ImportError:
        from PIL import Image, ImageDraw

        for xi1, yi1, xi2, yi2, text in rects:
            img = Image.new("L", (cw, ch), 0)
            d = ImageDraw.Draw(img)
            d.rectangle(
                [xi1 - x0, yi1 - y0, xi2 - x0, yi2 - y0], outline=255, width=t
            )
            if text:
                ty = min(max(yi1 - 12 - y0, 0), ch - 2)
                d.text((xi1 - x0 + 2, ty), text, fill=255)
            masks.append(np.array(img))
        for pts, _col, closed in line_shapes:
            img = Image.new("L", (cw, ch), 0)
            d = ImageDraw.Draw(img)
            xy = [(int(px) - x0, int(py) - y0) for px, py in pts]
            if closed:
                xy.append(xy[0])
            d.line(xy, fill=255, width=t, joint="curve")
            masks.append(np.array(img))

    rgb = np.zeros((ch, cw, 3), np.uint8)
    alpha = np.zeros((ch, cw), np.uint8)
    for m, col in zip(masks, cols):
        hit = m > 0
        rgb[hit] = col  # draw order: later shapes overwrite inside overlaps
        np.maximum(alpha, m, out=alpha)
    return np.dstack([rgb, alpha]), x0, y0


def draw_polygons(
    image: np.ndarray,
    shapes: Iterable,
    thickness: int = 2,
    closed: bool = True,
) -> np.ndarray:
    """Draw polygon outlines (or open tracks) on an RGB array copy.

    The software counterpart of :func:`render_overlay_rgba`'s
    polygons/tracks legs, mirroring :func:`draw_boxes`' copy-in
    copy-out conventions — reach for it when the frame stays client-side
    and no blend is needed.

    Args:
        image: RGB uint8 array (H, W, 3).
        shapes: iterable of ``(points, color)`` pairs, ``points`` being
            any (N, 2) sequence of pixel coordinates (same schema as
            :func:`render_overlay_rgba`); ``color=None`` means green.
        thickness: line thickness in pixels.
        closed: close each outline (True for zones, False for tracks).

    Returns: new RGB array; the input array is not modified.
    """
    t = max(1, int(thickness))
    parsed = [(_as_points(p), tuple(c) if c is not None else (0, 255, 0))
              for p, c in shapes]
    out = image.copy()
    if not parsed:
        return out
    try:
        import cv2

        for pts, col in parsed:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], closed, col, t)
        return out
    except ImportError:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        for pts, col in parsed:
            xy = [(int(px), int(py)) for px, py in pts]
            if closed:
                xy.append(xy[0])
            draw.line(xy, fill=col, width=t, joint="curve")
        return np.array(pil)

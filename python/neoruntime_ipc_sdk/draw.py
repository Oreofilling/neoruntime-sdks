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

from .dsp_wire import _MAX_BATCH

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


def _text_metrics() -> tuple[int, int, int]:
    """Caption metrics for the active rasterizer:
    ``(anchor_offset, canvas_row_floor, rows_below_anchor)`` — cv2
    anchors text at the baseline (6 px above the box top, floored to
    canvas row 12, descender ~4 px below), Pillow at the glyph top
    (12 px above the box top, no floor, ~11 px tall)."""
    try:
        import cv2  # noqa: F401

        return 6, 12, 6
    except ImportError:
        return 12, 0, 12


def _text_width(text: str) -> int:
    """Painted caption width measured from the box's left edge.

    Backend-matched to :func:`_draw_masks` (cv2 draws at ``xi1`` with
    LINE_AA spill, Pillow at ``xi1 + 2``) plus a 2 px pad, so a caption
    strip tightened to this width never clips mid-glyph — the rasterizer
    itself never clips a caption at the box extent.
    """
    try:
        import cv2

        return int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]) + 2
    except ImportError:
        from PIL import Image, ImageDraw

        canvas = ImageDraw.Draw(Image.new("L", (1, 1)))
        return int(canvas.textlength(text)) + 4


def _text_row(yi1: int, y0: int, ch: int, offset: int, floor: int,
              fragment: bool) -> int | None:
    """Canvas row for a box caption; ``None`` = not this canvas's pixels.

    The single-canvas path (``fragment=False``) keeps the legacy clamps.
    A fragment canvas anchors the caption to the same frame row instead
    of clamping it into view — the clamp is what used to spray captions
    onto bottom/side strips that the caption was never meant for. Only
    the canvas covering the caption strip gets the text; the ``y0 == 0``
    floor keeps the near-top-edge degenerate case at parity with the
    union canvas.
    """
    row = yi1 - offset - y0
    if not fragment:
        return min(max(row, floor), ch - 2)
    if y0 == 0:
        row = max(row, floor)
    return row if 0 <= row < ch else None


def _draw_masks(
    ch: int, cw: int, x0: int, y0: int, t: int, rects, line_shapes,
    fragment: bool = False,
) -> list[np.ndarray]:
    """Coverage masks for ``rects``/``line_shapes`` on a ``(ch, cw)``
    canvas whose frame-space origin is ``(x0, y0)`` — the shared
    rasterizer behind :func:`render_overlay_rgba` and
    :func:`render_overlay_fragments`."""
    masks: list[np.ndarray] = []
    try:
        import cv2

        for xi1, yi1, xi2, yi2, text in rects:
            m = np.zeros((ch, cw), np.uint8)
            # rectangle default LINE_8, matching draw_boxes' hard edges
            cv2.rectangle(m, (xi1 - x0, yi1 - y0), (xi2 - x0, yi2 - y0), 255, t)
            if text:
                ty = _text_row(yi1, y0, ch, 6, 12, fragment)
                if ty is not None:
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
                ty = _text_row(yi1, y0, ch, 12, 0, fragment)
                if ty is not None:
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
    return masks


def _colorize(masks, cols, ch: int, cw: int) -> np.ndarray:
    """Straight-alpha RGBA from coverage masks: color per draw order,
    alpha the mask union."""
    rgb = np.zeros((ch, cw, 3), np.uint8)
    alpha = np.zeros((ch, cw), np.uint8)
    for m, col in zip(masks, cols):
        hit = m > 0
        rgb[hit] = col  # draw order: later shapes overwrite inside overlaps
        np.maximum(alpha, m, out=alpha)
    return np.dstack([rgb, alpha])


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

    masks = _draw_masks(ch, cw, x0, y0, t, rects, line_shapes)
    return _colorize(masks, cols, ch, cw), x0, y0


def _overlaps(a, b) -> bool:
    """Whether two ``(x0, y0, x1, y1)`` half-open rects intersect."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def render_overlay_fragments(
    frame_w: int,
    frame_h: int,
    boxes: Iterable = (),
    labels: Sequence[str | None] | None = None,
    scores: Sequence[float | None] | None = None,
    colors: Sequence[tuple[int, int, int]] | None = None,
    thickness: int = 2,
    polygons: Sequence = (),
    tracks: Sequence = (),
) -> list[tuple[np.ndarray, int, int]]:
    """Render boxes as tight stroke fragments instead of one bbox canvas.

    Same inputs and straight-alpha contract as :func:`render_overlay_rgba`,
    but returns a *list* of ``(rgba, x, y)`` fragments to spread over one
    :meth:`DspClient.blend_hw` call (the daemon composites the whole
    overlay list in a single job, so extra fragments cost only their
    area). Each box becomes up to four thin canvases — caption+top edge,
    bottom edge, left edge, right edge — so a box pays for its
    ``2 * thickness`` stroke perimeter plus its caption strip, not its
    area: a full-frame 1280x720 box drops from a 0.9 MP canvas (render,
    RGBA->ARGB repack, wire bytes and blend are all area-bound; ~3.7 MB
    per frame over the socket) to ~0.06 MP of 16 px strips. Polygons and
    tracks stay one canvas per shape (that shape's own bbox, not the
    union of everything).

    Pixels are identical to :func:`render_overlay_rgba` for
    non-overlapping shapes: every fragment is rasterized by the same
    backend at the same frame coordinates on a smaller canvas, and the
    fragments tile the shape extents without gaps or overlap. Two
    caveats: overlapping shapes composite marginally differently (an AA
    edge landing on another shape's stroke blends against it instead of
    taking the union mask), and in the per-shape fallback below a
    caption wider than its shape clips at that shape's extent (the
    fragment path itself widens a caption's strip to its measured text
    extent instead). Call once per detection for exact single-canvas
    parity.

    Fragment count is bounded for the caller: ``blend_hw`` caps one job
    at 64 overlays and each box costs up to four fragments, so past
    ~15 boxes this falls back to one canvas per shape, and past 64
    shapes to the one shared union canvas — the return always fits one
    blend job.
    """
    t = max(1, int(thickness))
    items = [_to_xyxy(b) for b in boxes]
    lines = [(p, c, True) for p, c in polygons] + [(p, c, False) for p, c in tracks]
    if not items and not lines:
        raise ValueError("render_overlay_fragments needs at least one shape")
    texts = [
        _format_label(
            labels[i] if labels and i < len(labels) else None,
            scores[i] if scores and i < len(scores) else None,
        )
        for i in range(len(items))
    ]
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

    # per-shape stroke extents, identical formulas to the union canvas —
    # fragment pixels then equal canvas pixels region for region
    x_min, y_min, x_max, y_max = int(frame_w), int(frame_h), 0, 0
    rects = []
    rect_bboxes = []
    for (x1, y1, x2, y2), text in zip(items, texts):
        xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
        rects.append((xi1, yi1, xi2, yi2, text))
        strip = _LABEL_STRIP if text else 0
        rect_bboxes.append((xi1 - t - 1, yi1 - t - 1 - strip,
                            xi2 + t + 1, yi2 + t + 1))
        x_min = min(x_min, xi1 - t - 1)
        y_min = min(y_min, yi1 - t - 1 - strip)
        x_max = max(x_max, xi2 + t + 1)
        y_max = max(y_max, yi2 + t + 1)
    line_bboxes = []
    for pts, _col, _closed in line_shapes:
        bx0 = int(pts[:, 0].min()) - t - 1
        by0 = int(pts[:, 1].min()) - t - 1
        bx1 = int(pts[:, 0].max()) + t + 1
        by1 = int(pts[:, 1].max()) + t + 1
        line_bboxes.append((bx0, by0, bx1, by1))
        x_min = min(x_min, bx0)
        y_min = min(y_min, by0)
        x_max = max(x_max, bx1)
        y_max = max(y_max, by1)
    ux0, uy0 = max(0, x_min), max(0, y_min)
    ux1, uy1 = min(int(frame_w), x_max), min(int(frame_h), y_max)
    if ux1 <= ux0 or uy1 <= uy0:
        raise ValueError("all shapes lie outside the frame")

    # fragment rects, tight to what each must carry: horizontal strips
    # to the box stroke extent (a captioned box's top strip widens to
    # the measured text extent — the rasterizer never clips a caption
    # at the box, so its canvas must cover where the text paints),
    # vertical strips to the stroke columns only. Coverage is unchanged:
    # every shape pixel lands in its own fragments, and a shape whose
    # bbox overlaps another fragment's window still rasterizes there
    # too (opaque double-paints are idempotent; AA edges keep the
    # union-canvas caveat). Union-wide strips were tried first and cost
    # 64% of the frame for two boxes on opposite edges.
    frags: list[tuple[int, int, int, int]] = []
    for (xi1, yi1, xi2, yi2, text), _bb in zip(rects, rect_bboxes):
        strip = _LABEL_STRIP if text else 0
        fx0 = max(ux0, xi1 - t - 1)
        fx1 = min(ux1, xi2 + t + 1)
        if text:
            fx1 = min(ux1, max(fx1, xi1 + _text_width(text)))
        top0 = max(uy0, yi1 - t - 1 - strip)
        top1 = min(uy1, yi1 + t + 1)
        bot0 = max(uy0, yi2 - t)
        bot1 = min(uy1, yi2 + t + 1)
        lx1 = min(ux1, xi1 + t + 1)
        rx0 = max(ux0, xi2 - t)
        if top1 > top0:
            frags.append((fx0, top0, fx1, top1))
        if bot1 > bot0:
            frags.append((fx0, bot0, fx1, bot1))
        if bot0 > top1:  # side edges between the horizontal strips
            if lx1 > fx0:
                frags.append((fx0, top1, lx1, bot0))
            if fx1 > rx0:
                frags.append((rx0, top1, fx1, bot0))
    for bb in line_bboxes:
        fx0, fy0 = max(ux0, bb[0]), max(uy0, bb[1])
        fx1, fy1 = min(ux1, bb[2]), min(uy1, bb[3])
        if fx1 > fx0 and fy1 > fy0:
            frags.append((fx0, fy0, fx1, fy1))

    # blend_hw caps one job at _MAX_BATCH overlays and a box costs up to
    # four fragments: past that, fall back to one canvas per shape (a
    # caption wider than its shape then clips at that shape's extent);
    # past _MAX_BATCH shapes, one shared union canvas. Either way the
    # return always fits a single blend job.
    if len(frags) > _MAX_BATCH:
        frags = [
            (max(ux0, bb[0]), max(uy0, bb[1]), min(ux1, bb[2]), min(uy1, bb[3]))
            for bb in rect_bboxes + line_bboxes
            if min(ux1, bb[2]) > max(ux0, bb[0])
            and min(uy1, bb[3]) > max(uy0, bb[1])
        ]
        if len(frags) > _MAX_BATCH:
            frags = [(ux0, uy0, ux1, uy1)]

    m = _text_metrics()
    out: list[tuple[np.ndarray, int, int]] = []
    for fx, fy, fx1, fy1 in frags:
        cw = max(_MIN_OVERLAY_DIM, fx1 - fx)
        ch = max(_MIN_OVERLAY_DIM, fy1 - fy)
        # a caption landing in this fragment dictates its height: the
        # 16 px floor must not clip the descender (the union canvas is
        # always tall enough; row is stable under the later slide)
        for xi1c, yi1c, _xi2c, _yi2c, text in rects:
            if not text:
                continue
            row = _text_row(yi1c, fy, ch, m[0], m[1], True)
            if row is not None:
                ch = max(ch, row + m[2])
        # even height before the slide: blend_hw pads an odd overlay
        # height with one transparent bottom row and enforces bounds on
        # the padded rect, so an odd ch flush to the frame bottom would
        # overshoot the base by that pad row. Growing here instead adds
        # the same transparent row while the slide can still keep the
        # canvas in-bounds.
        ch += ch & 1
        # the 16 px floor can push the canvas past the frame's right or
        # bottom edge — slide the origin back (content offsets follow the
        # origin, so pixels keep their frame position)
        fx = max(0, min(fx, int(frame_w) - cw))
        fy = max(0, min(fy, int(frame_h) - ch))
        window = (fx, fy, fx + cw, fy + ch)
        sel_rects = [r for r, bb in zip(rects, rect_bboxes) if _overlaps(bb, window)]
        sel_lines = [ls for ls, bb in zip(line_shapes, line_bboxes)
                     if _overlaps(bb, window)]
        if not sel_rects and not sel_lines:
            continue  # no visible pixels in this fragment
        sel_cols = [col for col, bb in zip(cols[:len(rects)], rect_bboxes)
                    if _overlaps(bb, window)]
        sel_cols += [col for col, bb in zip(cols[len(rects):], line_bboxes)
                     if _overlaps(bb, window)]
        masks = _draw_masks(ch, cw, fx, fy, t, sel_rects, sel_lines,
                            fragment=True)
        out.append((_colorize(masks, sel_cols, ch, cw), fx, fy))
    return out


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

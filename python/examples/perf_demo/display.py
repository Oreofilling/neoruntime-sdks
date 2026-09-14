"""Burn-in formatting: metric lines plus their per-chain render carriers.

Chain A has no free-text API. Its metric lines ride detection ``label``
fields over tiny corner boxes (``metric_detections``) — the carrier the
on-device spike settled on: polygon labels render at a fixed 5x7 px font
(invisible at 4K) and ``thickness=-1`` bars clamp to a 1 px outline on
the CPU bake path, while detection labels auto-scale with frame height
(~21 px at 4K). The metric layer publishes under its own session_id so
it stacks with the bound real-detection layer instead of replacing it.
Chain B owns its pixels, so its lines become an RGBA text chip
(``text_chip_rgba``) stacked into the same blend_hw overlay list
as the detection graphics.
"""

from __future__ import annotations

from stats import fmt_duration, fmt_ms

# detection-box carrier geometry (normalized frame units), SPIKE-verified
# on the 4K main stream: label ~21 px tall, ~45 px above its box
DET_X = 0.02
DET_Y0 = 0.08
DET_DY = 0.04
DET_W = 0.012
DET_H = 0.006
MAX_BURN_LINES = 4  # annotate() accepts far more; 4 lines is plenty
MAX_LABEL_CHARS = 160  # daemon text buffer is 256; 160 keeps width < 3840 px

# text chip geometry (pixels at 720p-class sub stream)
CHIP_PAD = 6
CHIP_FONT_SCALE = 0.52
CHIP_FONT_THICKNESS = 1
CHIP_BG_ALPHA = 150  # translucent black out of 255
CHIP_MIN = 16       # daemon floor for overlay layers


def format_a_line(a: dict, stream_delta: dict | None) -> str:
    """Chain A OSD line from a hub snapshot's ``a`` section.

    ``stream_delta`` is the ``sys.stream_delta`` entry for the A display
    stream (main): counter deltas over the last status interval, source
    of bake coverage and late-command accounting. A degraded chain (the
    -2814 probe path) reports its reason instead of numbers.
    """
    if a.get("degraded_reason"):
        return f"A sub | DEGRADED: {a['degraded_reason']} | {fmt_duration(a.get('last_result_age_s') or 0.0)} since last"
    fps = a.get("fps")
    lat = a.get("latency") or {}
    skew = a.get("skew") or {}
    parts = [
        "A sub",
        f"fps {fps:.1f}" if fps is not None else "fps --",
        f"lat {fmt_ms(lat.get('p50'))}/{fmt_ms(lat.get('p99'))}ms",
        f"skew {fmt_ms(skew.get('mean'), digits=1)}ms",
        f"qdrop {a.get('dropped') or 0}",
    ]
    if stream_delta:
        bake_pct = stream_delta.get("bake_pct")
        parts.append(f"bake {fmt_ms(bake_pct)}%" if bake_pct is not None else "bake --%")
        late = stream_delta.get("overlay_late_commands")
        if late is not None:
            parts.append(f"late {late}")
    if a.get("epoch") is not None:
        parts.append(f"ep {a['epoch']}")
    return " | ".join(parts)


def format_b_line(b: dict) -> str:
    """Chain B OSD line from a hub snapshot's ``b`` section."""
    if b.get("degraded_reason"):
        return f"B fd | DEGRADED: {b['degraded_reason']} | ok {b.get('frames_ok') or 0}"
    fps = b.get("fps")
    pull = b.get("pull") or {}
    infer = b.get("infer") or {}
    hw = b.get("hw_infer") or {}
    draw = b.get("draw") or {}
    pub = b.get("pub") or {}
    e2e = b.get("e2e") or {}
    parts = [
        "B fd",
        f"fps {fps:.1f}" if fps is not None else "fps --",
        f"pull {fmt_ms(pull.get('p50'))}",
        f"infer {fmt_ms(infer.get('p50'))}(hw {fmt_ms(hw.get('p50'))})",
        f"draw {fmt_ms(draw.get('p50'))}",
        f"pub {fmt_ms(pub.get('p50'))}",
        f"e2e {fmt_ms(e2e.get('p99'))}ms",
        f"idrop {b.get('inject_dropped') or 0}",
    ]
    if b.get("in_flight") is not None and b.get("pool_depth"):
        parts.append(f"lease {b.get('in_flight')}/{b.get('pool_depth')}")
    if b.get("objects_last"):
        parts.append(f"obj {b['objects_last']}")
    return " | ".join(parts)


def format_status_line(sys_snap: dict, uptime_s: float) -> str:
    """Operator line burned under both chains' own lines."""
    parts = []
    npu = sys_snap.get("npu_util")
    dsp = sys_snap.get("dsp_util")
    if npu is not None or dsp is not None:
        parts.append(f"NPU {fmt_ms(npu)}% DSP {fmt_ms(dsp)}%")
    if sys_snap.get("temp_c") is not None:
        parts.append(f"{fmt_ms(sys_snap['temp_c'])}C")
    parts.append(f"up {fmt_duration(uptime_s)}")
    return " | ".join(parts)


def metric_detections(lines: list[str], *, x: float = DET_X, y: float = DET_Y0,
                      dy: float = DET_DY, w: float = DET_W,
                      h: float = DET_H) -> list[dict]:
    """Chain A carrier: one tiny corner box per line, text as its label.

    Returns annotate()-schema detection dicts (overlay.py:275-277 accepts
    plain dicts with ``label``/``score``/``bbox{x,y,width,height}``),
    points normalized to [0,1]. At most MAX_BURN_LINES entries; labels
    clamp at MAX_LABEL_CHARS. Publish under a session_id of its own so
    this unbound ttl layer stacks with the bound detection layer — a
    same-session detections payload would replace it.
    """
    detections = []
    for i, text in enumerate(lines[:MAX_BURN_LINES]):
        detections.append({
            "label": text[:MAX_LABEL_CHARS],
            "score": 1.0,
            "bbox": {"x": x, "y": y + i * dy, "width": w, "height": h},
        })
    return detections


def text_chip_rgba(text: str, *, font_scale: float = CHIP_FONT_SCALE,
                   thickness: int = CHIP_FONT_THICKNESS,
                   pad: int = CHIP_PAD) -> tuple:
    """Chain B carrier: RGBA chip (h, w, 4 uint8), straight alpha.

    Translucent black background, white text — mirrors the daemon's
    label-chip look. Uses cv2 (present on device and in the SDK's host
    venv alongside draw.py); falls back to PIL if cv2 is missing. The
    chip is never smaller than CHIP_MIN in either dimension, matching
    the daemon's 16x16 overlay floor. Returns (rgba, x0, y0) in the
    render_overlay_rgba overlay-layer convention (caller pins position).
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - host without cv2
        return _text_chip_pil(text, font_scale=font_scale, pad=pad)

    import numpy as np

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    h = max(th + baseline + 2 * pad, CHIP_MIN)
    w = max(tw + 2 * pad, CHIP_MIN)
    chip = np.zeros((h, w, 4), dtype=np.uint8)
    chip[:, :, 3] = CHIP_BG_ALPHA  # translucent black bg, straight alpha
    # cv2's putText writes BGR only (alpha untouched), so render the text
    # into a coverage mask and stamp opaque white through all 4 channels.
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.putText(mask, text, (pad, pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                255, thickness, cv2.LINE_AA)
    chip[mask > 0] = (255, 255, 255, 255)
    return chip, 0, 0


def _text_chip_pil(text: str, *, font_scale: float, pad: int) -> tuple:
    """PIL fallback with the same geometry contract as the cv2 path."""
    from PIL import Image, ImageDraw, ImageFont

    import numpy as np

    size = max(int(round(font_scale * 22)), 12)
    font = ImageFont.truetype("DejaVuSansMono.ttf", size) if _has_truetype() \
        else ImageFont.load_default()
    tmp = Image.new("RGBA", (1, 1))
    probe = ImageDraw.Draw(tmp)
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top
    h = max(th + 2 * pad, CHIP_MIN)
    w = max(tw + 2 * pad, CHIP_MIN)
    chip = Image.new("RGBA", (w, h), (0, 0, 0, CHIP_BG_ALPHA))
    draw = ImageDraw.Draw(chip)
    draw.text((pad, pad), text, font=font, fill=(255, 255, 255, 255))
    return np.asarray(chip).copy(), 0, 0


def _has_truetype() -> bool:
    try:
        from PIL import ImageFont
        ImageFont.truetype("DejaVuSansMono.ttf", 12)
        return True
    except Exception:
        return False

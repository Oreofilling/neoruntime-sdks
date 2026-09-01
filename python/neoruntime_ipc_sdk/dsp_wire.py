"""DSP service wire protocol: UDS messages, codec, error codes.

Everything here mirrors the daemon side (camera-daemon dsp_service.cpp
and include/fd_protocol.h) so client modules can encode/decode without
knowing socket plumbing. Pure data + pure functions; no I/O.
"""

import struct
from typing import List, Optional, Sequence, Tuple

from .proto import camera_pb2

# ---- UDS wire constants (platform camera-daemon include/fd_protocol.h) ----
_FD_PUB_MSG_OK = 5                                 # control ack — no payload
_FD_PUB_MSG_ERROR = 6                              # control error ack
_FD_PUB_MSG_DSP_ALLOC = 7
_FD_PUB_MSG_DSP_ALLOC_RESP = 8
_FD_PUB_MSG_DSP_BUF_RELEASE = 9
_FD_PUB_MSG_DSP_IMPORT = 10
_FD_PUB_MSG_DSP_IMPORT_RESP = 11

_ALLOC_REQ_FMT = "<IIIIII"                         # hdr + w, h, fmt, count
_ALLOC_REQ_SIZE = struct.calcsize(_ALLOC_REQ_FMT)  # 24
_ALLOC_RESP_FMT = "<II i I I 3I 3I 4x 64Q"         # C layout incl. u64 align
_ALLOC_RESP_SIZE = struct.calcsize(_ALLOC_RESP_FMT)
_RELEASE_FMT = "<IIQ"
_IMPORT_REQ_FMT = "<12I"                           # hdr + w,h,fmt,planes,strides[3],sizes[3]
_IMPORT_REQ_SIZE = struct.calcsize(_IMPORT_REQ_FMT)   # 48
_IMPORT_RESP_FMT = "<IIi4xq"                       # hdr + code, pad, import_id
_IMPORT_RESP_SIZE = struct.calcsize(_IMPORT_RESP_FMT)  # 24
_DSP_MAX_FDS = 64                                  # FD_PUB_DSP_MAX_FDS

# HalPixelFormat wire values (hal_v2 hal_buffer.h) — deliberately separate
# from the SDK PixelFormat enum, whose numbering does not match the wire.
_HAL_PIXEL_FORMAT = {"nv12": 0, "rgb24": 4, "gray8": 8}
_DSP_FORMATS = ("nv12", "rgb24", "gray8")

# ---- daemon caps (dsp_service.cpp), mirrored for fail-fast validation ----
_MIN_DIM = 16
_MAX_DIM = 8192
_MAX_BATCH = 64

# ---- DspService error codes ----
DSP_SERVICE_UNAVAILABLE = -5

_ERROR_TEXT = {
    -1: "invalid request",
    -2: "no such buffer id",
    -3: "quota exceeded",
    -4: "job timeout",
    -5: "dsp service unavailable",
    -6: "out of memory",
    -7: "client limit exceeded",
}

_OP_RESIZE = 0
_OP_CROP_AND_RESIZE = 1
_OP_MULTI_CROP = 2

_INTERP_WIRE = {
    "nearest": camera_pb2.DSP_INTERP_NEAREST,
    "bilinear": camera_pb2.DSP_INTERP_BILINEAR,
    "area": camera_pb2.DSP_INTERP_AREA,
    "bicubic": camera_pb2.DSP_INTERP_BICUBIC,
}
_SCALING_WIRE = {
    "stretch": camera_pb2.DSP_SCALING_STRETCH,
    "letterbox": camera_pb2.DSP_SCALING_LETTERBOX_MIDDLE,
    "letterbox_middle": camera_pb2.DSP_SCALING_LETTERBOX_MIDDLE,
    "letterbox_up_left": camera_pb2.DSP_SCALING_LETTERBOX_UP_LEFT,
    "scale_crop": camera_pb2.DSP_SCALING_SCALE_AND_CROP,
}
_PRIORITY_WIRE = {
    "background": camera_pb2.DSP_PRIORITY_BACKGROUND,
    "normal": camera_pb2.DSP_PRIORITY_NORMAL,
}


class DspError(Exception):
    """DSP job or buffer error. ``code`` mirrors the daemon error codes."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class _DspUnavailable(DspError):
    """Internal: daemon lacks the DSP surface — take the CPU fallback."""


def alloc_request_bytes(width: int, height: int, fmt_wire: int,
                        count: int) -> bytes:
    """Encode FD_PUB_MSG_DSP_ALLOC (24 bytes)."""
    return struct.pack(_ALLOC_REQ_FMT, _FD_PUB_MSG_DSP_ALLOC,
                       _ALLOC_REQ_SIZE, width, height, fmt_wire, count)


def parse_alloc_resp(payload: bytes) -> Tuple[int, int, int, List[int],
                                              List[int], List[int]]:
    """Decode FD_PUB_MSG_DSP_ALLOC_RESP (560 bytes; fds arrive separately).

    Returns ``(code, count, num_planes, strides[3], sizes[3], ids[count])``.
    """
    if len(payload) < _ALLOC_RESP_SIZE:
        raise DspError(f"short DSP alloc response: {len(payload)} bytes")
    mtype, _size, code, count, num_planes, s0, s1, s2, z0, z1, z2 = \
        struct.unpack_from("<II i I I 3I 3I", payload, 0)
    if mtype != _FD_PUB_MSG_DSP_ALLOC_RESP:
        raise DspError(f"unexpected DSP alloc response type {mtype}")
    ids = list(struct.unpack_from("<64Q", payload, 48))[:count]
    return code, count, num_planes, [s0, s1, s2], [z0, z1, z2], ids


def import_request_bytes(width: int, height: int, fmt_wire: int,
                         num_planes: int, strides: Sequence[int],
                         sizes: Sequence[int]) -> bytes:
    """Encode FD_PUB_MSG_DSP_IMPORT (48 bytes; fds travel via SCM_RIGHTS)."""
    return struct.pack(_IMPORT_REQ_FMT, _FD_PUB_MSG_DSP_IMPORT,
                       _IMPORT_REQ_SIZE, width, height, fmt_wire, num_planes,
                       strides[0], strides[1], strides[2],
                       sizes[0], sizes[1], sizes[2])


def parse_import_resp(payload: bytes) -> Tuple[int, int]:
    """Decode FD_PUB_MSG_DSP_IMPORT_RESP (24 bytes).

    Returns ``(code, import_id)``; ``code`` mirrors the daemon error codes
    (0 on success, -1 validation failure, -7 client import cap).
    """
    if len(payload) < _IMPORT_RESP_SIZE:
        raise DspError(f"short DSP import response: {len(payload)} bytes")
    mtype, _size, code, import_id = struct.unpack(_IMPORT_RESP_FMT, payload)
    if mtype != _FD_PUB_MSG_DSP_IMPORT_RESP:
        raise DspError(f"unexpected DSP import response type {mtype}")
    return code, import_id


def _plane_rows(fmt: str, width: int, height: int) -> List[Tuple[int, int]]:
    """(row_bytes, rows) per plane for a tightly-packed geometry."""
    if fmt == "nv12":
        return [(width, height), (width, height // 2)]
    if fmt == "rgb24":
        return [(width * 3, height)]
    return [(width, height)]


def _plane_count(fmt: str) -> int:
    return 2 if fmt == "nv12" else 1


def _validate_geometry(width: int, height: int, fmt: str, what: str) -> None:
    if fmt not in _DSP_FORMATS:
        raise DspError(f"unsupported format {fmt!r} (nv12/rgb24/gray8)")
    if fmt == "nv12" and (width % 2 or height % 2):
        raise DspError(f"{what}: nv12 needs even width/height "
                       f"(got {width}x{height})")
    if not (_MIN_DIM <= width <= _MAX_DIM and _MIN_DIM <= height <= _MAX_DIM):
        raise DspError(f"{what}: dims {width}x{height} outside daemon range "
                       f"[{_MIN_DIM}, {_MAX_DIM}]")

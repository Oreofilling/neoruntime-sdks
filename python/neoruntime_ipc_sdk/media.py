"""Media clients (compatibility facade).

The implementation lives in :mod:`.frame` (frame types and pixel-format
helpers), :mod:`.encoded` (encoded-stream UDS client) and
:mod:`.fd_client` (zero-copy DMA-BUF fd client). This module re-exports
the full historical surface — including the private wire-protocol names
tests and internal callers patch — so ``from .media import X`` keeps
working unchanged.
"""

from __future__ import annotations

from ._transport import recvmsg_with_fds as _recvmsg_with_fds  # noqa: F401
from ._transport import sendmsg_plain as _sendmsg_plain  # noqa: F401
from .encoded import (  # noqa: F401  # noqa: F401
    _ENC_HEADER_FMT,
    _ENC_HEADER_SIZE,
    EncodedFrame,
    EncodedStreamClient,
)
from .fd_client import (  # noqa: F401
    _FD_PUB_MAX_FDS,
    _FD_PUB_MAX_STREAM_NAME,
    _FD_PUB_MSG_ERROR,
    _FD_PUB_MSG_FRAME,
    _FD_PUB_MSG_OK,
    _FD_PUB_MSG_RELEASE,
    _FD_PUB_MSG_SUBSCRIBE,
    _FD_PUB_MSG_UNSUBSCRIBE,
    _FD_PUB_PROTOCOL_VERSION,
    _FRAME_FMT,
    _FRAME_SIZE,
    _HDR_FMT,
    _HDR_SIZE,
    _REL_FMT,
    _REL_SIZE,
    _RESP_FMT,
    _RESP_SIZE,
    _SUB_FMT,
    _SUB_SIZE,
    FdMediaClient,  # noqa: F401
)
from .frame import (  # noqa: F401
    _DMA_BUF_SYNC_END,
    _DMA_BUF_SYNC_READ,
    _DMA_BUF_SYNC_START,
    _DMA_BUF_SYNC_WRITE,
    _DSP_RESIZE_FORMATS,
    _PACKED_FORMATS,
    _YUV_FORMATS,
    PIXEL_FORMAT_NAMES,
    Frame,
    FrameHandle,
    PixelFormat,
    StreamInfo,
    _decode_raw,
    _dma_buf_sync,
    _encode_jpeg,
    _even,
    _materialize_handle,
    _resize_array,
)

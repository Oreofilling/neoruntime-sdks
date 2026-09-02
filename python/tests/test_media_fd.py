"""SDK-1: Frame fd retention (keep-fd mode) for FdMediaClient.

Harness: a socketpair stands in for the daemon's UDS. A FRAME message is
crafted per the fd-pub wire format and pushed with SCM_RIGHTS-passed fds
(memfd-backed, so ioctl fences no-op like on any non-dma-buf fd).
"""

import gc
import mmap
import os
import socket
import struct

import numpy as np
import pytest

from neoruntime_ipc_sdk import media
from neoruntime_ipc_sdk.media import FdMediaClient, Frame, FrameHandle

W, H = 4, 4
NV12_LEN = W * H * 3 // 2  # 24 bytes


def make_plane_fd(data: bytes) -> int:
    fd = os.memfd_create("sdk1-test-plane", 0)
    os.write(fd, data)
    os.lseek(fd, 0, os.SEEK_SET)
    return fd


def send_frame(peer, fds, frame_id=77, seq=1, sizes=(NV12_LEN, 0, 0),
               strides=(W, 0, 0), fmt_code=0):
    body = struct.pack(
        media._FRAME_FMT,
        media._FD_PUB_MSG_FRAME, media._FRAME_SIZE,
        frame_id, 1_700_000_000_000, seq, W, H, fmt_code, len(fds),
        strides[0], strides[1] if len(strides) > 1 else 0,
        strides[2] if len(strides) > 2 else 0,
        sizes[0], sizes[1] if len(sizes) > 1 else 0,
        sizes[2] if len(sizes) > 2 else 0,
        len(fds),
    )
    anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
            struct.pack(f"{len(fds)}i", *fds))]
    peer.sendmsg([body], anc)


def drain_release(peer, timeout=1.0):
    """Read one RELEASE message from the fake daemon side."""
    peer.settimeout(timeout)
    data = peer.recv(media._REL_SIZE)
    msg_type, _size, frame_id = struct.unpack(media._REL_FMT, data)
    assert msg_type == media._FD_PUB_MSG_RELEASE
    return frame_id


@pytest.fixture
def chan():
    """(client_side, daemon_side) socketpair."""
    ours, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    ours.settimeout(2.0)
    yield ours, peer
    ours.close()
    peer.close()


@pytest.fixture
def client():
    c = FdMediaClient(socket_path="/nonexistent/sdk1-test")
    yield c
    c.close()


def test_default_path_copies_image_and_releases(chan, client):
    ours, peer = chan
    gradient = bytes(range(NV12_LEN))
    fd = make_plane_fd(gradient)
    try:
        send_frame(peer, [fd])
        frame = client._recv_frame(ours)
    finally:
        os.close(fd)

    assert frame is not None
    assert frame.handle is None            # default: no retention
    assert frame.image is not None         # copy was materialized
    assert frame.image.shape == (H * 3 // 2, W)
    np.testing.assert_array_equal(np.frombuffer(frame.image, dtype=np.uint8),
                                  np.frombuffer(gradient, dtype=np.uint8))
    assert drain_release(peer) == 77       # RELEASE sent immediately


def test_keep_fd_retains_handle_and_defers_release(chan, client):
    ours, peer = chan
    gradient = bytes(range(NV12_LEN))
    fd = make_plane_fd(gradient)
    try:
        send_frame(peer, [fd], frame_id=88)
        frame = client._recv_frame(ours, keep_fd=True)
    finally:
        os.close(fd)

    assert frame is not None
    assert isinstance(frame.handle, FrameHandle)
    assert frame.handle.frame_id == 88
    assert len(frame.handle.fds) == 1
    assert frame.handle.strides[0] == W
    assert frame.handle.plane_sizes[0] == NV12_LEN
    assert frame.image is None            # no copy happened
    assert os.fstat(frame.handle.fd).st_size >= NV12_LEN  # fd still open

    # no RELEASE yet: daemon still considers the buffer ours
    peer.settimeout(0.2)
    with pytest.raises(socket.timeout):
        peer.recv(media._REL_SIZE)

    # mapped content matches what the daemon sent
    with mmap.mmap(frame.handle.fd, NV12_LEN, access=mmap.ACCESS_READ) as buf:
        assert bytes(buf[:NV12_LEN]) == gradient

    frame.release()
    assert drain_release(peer) == 88


def test_keep_fd_multi_plane_concatenates(chan, client):
    ours, peer = chan
    y = bytes([0x40] * (W * H))
    uv = bytes([0x80] * (W * H // 2))
    fds = [make_plane_fd(y), make_plane_fd(uv)]
    try:
        send_frame(peer, fds, frame_id=99,
                   sizes=(W * H, W * H // 2, 0), strides=(W, W, 0))
    finally:
        for fd in fds:
            os.close(fd)

    frame = client._recv_frame(ours, keep_fd=True)
    assert len(frame.handle.fds) == 2
    arr = frame.to_array()
    assert arr.shape == (H * 3 // 2, W)
    np.testing.assert_array_equal(arr[:H], np.full((H, W), 0x40, np.uint8))
    frame.release()
    assert drain_release(peer) == 99


def test_to_array_materializes_with_sync_fences(chan, client, monkeypatch):
    ours, peer = chan
    calls = []
    from neoruntime_ipc_sdk import frame as frame_mod
    monkeypatch.setattr(frame_mod, "_dma_buf_sync",
                        lambda fd, flags: calls.append((fd, flags)))
    fd = make_plane_fd(bytes(NV12_LEN))
    try:
        send_frame(peer, [fd])
    finally:
        os.close(fd)

    frame = client._recv_frame(ours, keep_fd=True)
    arr = frame.to_array()

    # fence ordering: READ|START before the read, READ|END after
    starts = [f for _, f in calls
              if f == media._DMA_BUF_SYNC_READ | media._DMA_BUF_SYNC_START]
    ends = [f for _, f in calls
            if f == media._DMA_BUF_SYNC_READ | media._DMA_BUF_SYNC_END]
    assert starts and len(starts) == len(ends)
    assert calls[0][1] == media._DMA_BUF_SYNC_READ | media._DMA_BUF_SYNC_START

    assert frame.image is arr               # cached
    assert frame.to_array() is arr          # second call: no re-map, no new fence
    fences_before = len(calls)
    frame.to_array()
    assert len(calls) == fences_before
    frame.release()


def test_to_rgb_materializes_retained_frame(chan, client):
    ours, peer = chan
    # mid-gray NV12: Y=0x80, UV=0x80 → near-neutral RGB
    payload = bytes([0x80] * (W * H)) + bytes([0x80] * (W * H // 2))
    fd = make_plane_fd(payload)
    try:
        send_frame(peer, [fd])
    finally:
        os.close(fd)

    frame = client._recv_frame(ours, keep_fd=True)
    rgb = frame.to_rgb()
    assert rgb.shape == (H, W, 3)
    assert frame.image is not None
    frame.release()


def test_release_is_idempotent(chan, client):
    ours, peer = chan
    fd = make_plane_fd(bytes(NV12_LEN))
    try:
        send_frame(peer, [fd], frame_id=101)
    finally:
        os.close(fd)

    frame = client._recv_frame(ours, keep_fd=True)
    retained_fd = frame.handle.fd
    frame.release()
    frame.release()                         # second call: no-op, no error
    assert drain_release(peer) == 101
    peer.settimeout(0.2)
    with pytest.raises(socket.timeout):
        peer.recv(64)                       # exactly one RELEASE
    with pytest.raises(OSError):
        os.fstat(retained_fd)               # fd closed


def test_frame_gc_releases_handle(chan, client):
    ours, peer = chan
    fd = make_plane_fd(bytes(NV12_LEN))
    try:
        send_frame(peer, [fd], frame_id=102)
    finally:
        os.close(fd)

    frame = client._recv_frame(ours, keep_fd=True)
    del frame
    gc.collect()
    assert drain_release(peer) == 102


def test_client_close_releases_outstanding_handles(chan, client):
    ours, peer = chan
    fd = make_plane_fd(bytes(NV12_LEN))
    try:
        send_frame(peer, [fd], frame_id=103)
    finally:
        os.close(fd)

    frame = client._recv_frame(ours, keep_fd=True)
    assert client._retained
    client.close()
    assert not client._retained
    assert drain_release(peer) == 103
    with pytest.raises(OSError):
        os.fstat(frame.handle.fd)


def test_plain_frame_without_handle_unchanged():
    frame = Frame(sequence=1, timestamp_ns=1, width=W, height=H,
                  format="NV12", image=np.zeros((H * 3 // 2, W), np.uint8))
    assert frame.handle is None
    frame.release()                          # no-op, must not raise
    assert frame.to_rgb().shape == (H, W, 3)

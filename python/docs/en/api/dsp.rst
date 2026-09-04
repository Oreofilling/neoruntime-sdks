DSP Hardware Acceleration API
=============================

.. automodule:: neoruntime_ipc_sdk.dsp
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

DspClient
---------

.. autoclass:: neoruntime_ipc_sdk.DspClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

DspBufferPool
~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.DspBufferPool
   :members:
   :undoc-members:

DspError
~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.DspError
   :members:
   :undoc-members:

Usage Examples
--------------

Full-frame resize (model input preprocessing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import FdMediaClient, DspClient, DspError

   media = FdMediaClient()
   dsp = DspClient()

   # keep_fd=True retains the dma-buf fd; DspClient references the
   # same buffer zero-copy.
   frame = media.get_frame("main", timeout_ms=3000, keep_fd=True)

   try:
       # NV12 in -> NV12 out (h + h/2 rows); scaling="stretch" matches
       # cv2.resize semantics.
       small = dsp.resize_hw(frame, 640, 384)
   except DspError:
       # When DSP is unavailable and the source is a keep-fd frame the
       # SDK raises instead of silently falling back to CPU — take your
       # own CPU path here.
       small = frame.to_array()

   frame.release()  # return the dma-buf early (idempotent, optional)

Batched crops (letterbox scaling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # rects: (x, y, w, h, dst_w, dst_h) in source pixel coordinates
   # (even-aligned); results match the rects order. Ideal for
   # "many objects per frame" (e.g. plate tiles) — all crops are
   # merged into a single hardware job.
   rects = [
       (320, 500, 160, 48, 320, 48),
       (900, 520, 150, 44, 320, 48),
   ]
   tiles = dsp.multi_crop_hw(frame, rects, scaling="letterbox")

Single-region crop
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # crop_hw crops and optionally rescales to the target size in one go
   tile = dsp.crop_hw(frame, 320, 500, 160, 48, dst_width=320, dst_height=48)

Format conversion (RGB <-> NV12 / grayscale)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # convert_hw swaps the format at identical dimensions (the daemon's
   # CONVERT P0 contract): no rects, a single destination buffer;
   # dst_fmt is one of "nv12"/"rgb24"/"gray8".
   # Byte order: rgb24 on the wire is RGB order — swap BGR pixels
   # beforehand (or stay on the CPU path via color.bgr_to_nv12).
   nv12 = dsp.convert_hw(rgb, "nv12", fmt="rgb24")
   gray = dsp.convert_hw(rgb, "gray8", fmt="rgb24")

   # When you also need scaling, CONVERT first, RESIZE second: NV12 is
   # about half the rgb24 bytes, so the resize moves half the data.
   small = dsp.resize_hw(nv12, 640, 384)

   # When the DSP is unavailable the default is a CPU fallback (with a
   # UserWarning); pass cpu_fallback=False to raise DspError instead —
   # the router uses that for honest degradation accounting.

   # The firmware pair matrix is device-dependent: measured on hailo15
   # (93.72) only rgb24 <-> nv12 runs on the DSP — every gray8 pair is
   # refused by the firmware (HAL rc=-2801). With the default
   # cpu_fallback=True such job rejections also fall back to CPU with a
   # warning; last_used_hw=False records the backend actually used.

One-shot JPEG encode (snapshot / thumbnail)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # encode_jpeg_hw is the daemon's one-shot EncodeImage RPC: the source
   # buffer is pinned in the DSP registry zero-copy (keep-fd frames import
   # their dma-bufs, arrays are copied into a pool buffer) and the
   # complete JPEG bytes ride the response — no destination buffer, no
   # read-back.
   jpeg = dsp.encode_jpeg_hw(frame, quality=85, fmt="nv12")
   # array sources default to rgb24: jpeg = dsp.encode_jpeg_hw(rgb, quality=85)

   # No "hardware block" despite the name: the encoder is N-threaded
   # libjpeg on the DSP core behind a GStreamer dispatch — hailo15 has no
   # dedicated JPEG encode block. The win is central encode + zero-copy
   # input (app images can drop cv2/PIL), not raw speed; tight per-frame
   # loops are still better served by a CPU encode. The daemon reuses one
   # encoder keyed by (width, height, format, quality) and recreates it
   # when that key changes (the first frame after a change pays the
   # pipeline start-up).

Pre-allocated buffer pool (repeated jobs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Repeated jobs with identical geometry can use pooled buffers to
   # avoid per-call dma-buf allocation.
   pool = dsp.alloc_buffers(640, 384, fmt="nv12", count=4)
   small = dsp.resize_hw(frame, 640, 384, dst_pool=pool)

   # Read pooled buffer contents
   arr = pool.read(0)

   # Return the whole pool when done
   pool.release()

Context manager
~~~~~~~~~~~~~~~

.. code-block:: python

   with DspClient() as dsp:
       out = dsp.resize_hw(frame, 416, 416)

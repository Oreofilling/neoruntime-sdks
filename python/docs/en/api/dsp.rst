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

PendingDspJob
~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.PendingDspJob
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
   # After the first refusal the SDK remembers the firmware gap: later
   # gray8 array pairs take the CPU leg directly — no warning, no doomed
   # submit (keep-fd sources cannot take that leg and keep raising
   # honestly).

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

   # P2 pass-through leg: src_buffer_id chains the encode onto a pool
   # buffer / async job — the result never leaves the device, zero
   # read-back (pairs with PendingDspJob.buffer_id under wait=False,
   # see "Async jobs" below).
   jpeg = dsp.encode_jpeg_hw(None, quality=85, src_buffer_id=job.buffer_id)
   # src and src_buffer_id are mutually exclusive; this leg has no
   # client pixels to fall back on, so DSP unavailability raises
   # DspError directly.

   # No "hardware block" despite the name: the encoder is N-threaded
   # libjpeg on the DSP core behind a GStreamer dispatch — hailo15 has no
   # dedicated JPEG encode block. The win is central encode + zero-copy
   # input (app images can drop cv2/PIL), not raw speed; tight per-frame
   # loops are still better served by a CPU encode. The daemon reuses one
   # encoder keyed by (width, height, format, quality) and recreates it
   # when that key changes (the first frame after a change pays the
   # pipeline start-up).

Annotation blending (detection boxes onto NV12, dsp-offload P1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import render_overlay_rgba

   # blend_hw composites ARGB32 overlays onto an NV12 base, pasted 1:1
   # in order (no scaling; later overlays cover earlier ones). The blend
   # runs in place on the pool copy — it returns the annotated NV12
   # array and never touches the input array.
   overlay = np.zeros((64, 96, 4), np.uint8)   # (h, w, 4) RGBA
   overlay[..., :3] = (255, 0, 0)
   overlay[..., 3] = 255                       # straight alpha
   annotated = dsp.blend_hw(nv12, [(overlay, 40, 30)])

   # Pair it with render_overlay_rgba for "detection boxes on hardware":
   rgba, x0, y0 = render_overlay_rgba(w, h, boxes, labels, scores, colors)
   annotated = dsp.blend_hw(nv12, [(rgba, x0, y0)])
   # The accel router is the one-call entry:
   # router.run("draw_detections", nv12, result)

   # Contract notes: the base must be NV12 (the vendor op writes NV12
   # only). Arrays blend in place on the pool copy — the annotated NV12
   # array comes back and the input array is never touched; keep-fd
   # frames are **refused by default** (see the firmware-defect note in
   # the zero-copy section below); overlays smaller than 16x16 (the
   # daemon floor) are padded with fully transparent pixels to 16; the
   # hardware ARGB32 memory byte order is [A, R, G, B] and the SDK packs
   # it internally; quota is charged on (base + overlays) pixel area —
   # keep the canvas minimal (exactly what render_overlay_rgba
   # produces). When the DSP is unreachable or the job is rejected,
   # behavior matches the other *_hw calls: default warns and falls
   # back to CPU (_cpu_blend with identical straight-alpha math);
   # cpu_fallback=False raises instead (a keep-fd base has no client
   # pixels to fall back on — unavailability always raises there).

Async jobs (submit now, wait later — dsp-offload P2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # All five job methods (resize/crop/multi_crop/convert/blend) accept
   # wait=False: SubmitDspJobAsync returns a job_id immediately and you
   # get a PendingDspJob handle. Old daemons without the rpc fall back
   # to the synchronous submit transparently — that handle is "born
   # done" and wait() costs no extra RPC.
   job1 = dsp.resize_hw(frame, 640, 384, wait=False)
   job2 = dsp.multi_crop_hw(frame, rects, wait=False)
   ...  # submission overlaps execution: the single worker thread runs
        # jobs in submission order

   small = job1.wait()          # WaitDspJob + pool read-back + release
   tiles = job2.wait()          # multi_crop returns a list
   if job1.done():              # non-blocking poll (timeout 0); the
       ...                      # completion — including a failure — is
                                # cached after the first report
   job1.wait_result()           # wait without reading — result stays
                                # device-side
   bid = job1.buffer_id         # chains into encode_jpeg_hw(src_buffer_id=)
   job1.release()               # drop the result, return buffers
                                # (idempotent)

   # Semantics worth knowing: a wait timeout (rc=-4) keeps the registry
   # entry, so you can wait again; a failed job raises DspError from
   # the consuming wait (done() polling only reports state); waiting
   # after release raises; wait=False does not change the CPU fallback
   # story — when a fallback runs you get pixels, not a handle (there
   # is no hardware job to wait for).

Zero-copy blend chain (firmware-defect record: refused by default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # blend_hw with a keep-fd frame raises by default — not a missing
   # capability but a firmware defect: on current hailo15 silicon this
   # chain **wedged the DSP device-wide** (2/2 on 93.72: 720p, media
   # pipeline live, 700 MB general CMA free — the blend command simply
   # never returned; only a reboot recovers; the xrp driver latches
   # "fatal error, reboot required"). Array bases via frame.to_array()
   # are the proven, safe path.
   try:
       annotated = dsp.blend_hw(frame, [(rgba, x0, y0)])
   except DspError:
       annotated = dsp.blend_hw(frame.to_array(), [(rgba, x0, y0)])

   # zero_copy=True forces the chain explicitly (import dma-buf -> 1:1
   # RESIZE onto a pool -> in-place BLEND; with wait=False the result
   # never crosses the socket and encode_jpeg_hw(src_buffer_id=...)
   # skips the read-back) — for experiments once firmware is fixed;
   # nothing about it is guaranteed today:
   job = dsp.blend_hw(frame, [(rgba, x0, y0)],
                      wait=False, zero_copy=True)
   job.wait_result()                              # result stays pooled
   jpeg = dsp.encode_jpeg_hw(None, src_buffer_id=job.buffer_id)
   job.release()                                  # return after encoding

   # Ordering note: the RESIZE copy leg always runs synchronously (an
   # async job nobody waits would leak its daemon-side registry entry
   # and occupy one of the 32 pending slots per connection) — only the
   # BLEND compositing leg goes async. Full record:
   # docs/proposals/dsp-offload.md, P2 section.

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

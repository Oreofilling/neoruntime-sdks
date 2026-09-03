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

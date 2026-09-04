Color Conversion API
====================

.. automodule:: neoruntime_ipc_sdk.color
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Functions
---------

nv12_to_rgb / nv12_to_bgr
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: neoruntime_ipc_sdk.color.nv12_to_rgb
   :no-index:

.. autofunction:: neoruntime_ipc_sdk.color.nv12_to_bgr
   :no-index:

rgb_to_nv12 / bgr_to_nv12
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: neoruntime_ipc_sdk.color.rgb_to_nv12
   :no-index:

.. autofunction:: neoruntime_ipc_sdk.color.bgr_to_nv12
   :no-index:

nv12_resize
~~~~~~~~~~~

.. autofunction:: neoruntime_ipc_sdk.color.nv12_resize
   :no-index:

Examples
--------

Feed an NV12 model with an RGB frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import rgb_to_nv12, nv12_resize

   # Encode the annotated RGB frame and resize it as NV12 — no
   # NV12->RGB->resize->RGB->NV12 round-trip anywhere
   nv12 = rgb_to_nv12(rgb_frame)
   small = nv12_resize(nv12, (1920, 1080), (640, 384))

Route through DSP hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prefers the DSP leg (see the Accel Routing API); falls back to the
   # numpy/cv2 software implementation when the DSP is unreachable and
   # records the degradation in health()
   from neoruntime_ipc_sdk import get_default_router

   router = get_default_router()
   small = router.run("resize_nv12", nv12, (1920, 1080), (640, 384))

Accel Routing API
=================

.. automodule:: neoruntime_ipc_sdk.accel
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Classes and Functions
---------------------

AccelRouter
~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.AccelRouter
   :members:
   :undoc-members:
   :no-index:

RoutePolicy
~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.RoutePolicy
   :members:
   :undoc-members:
   :no-index:

RouteDecision / DegradationRecord
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.RouteDecision
   :members:
   :undoc-members:

.. autoclass:: neoruntime_ipc_sdk.DegradationRecord
   :members:
   :undoc-members:

get_default_router
~~~~~~~~~~~~~~~~~~

.. autofunction:: neoruntime_ipc_sdk.accel.get_default_router
   :no-index:

Examples
--------

Default router: hardware first, automatic fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import get_default_router

   router = get_default_router()
   small = router.run("resize_nv12", nv12, (1920, 1080), (640, 384))

   # Falls back to numpy/cv2 when the DSP is unreachable and records
   # the degradation
   health = router.health()
   print(health["ops"]["resize_nv12"]["backend"])
   print(health["recent_degradations"])

Color conversion runs on the hardware leg too
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import get_default_router

   router = get_default_router()
   nv12 = router.run("rgb_to_nv12", rgb)               # DSP convert_hw
   rgb2 = router.run("nv12_to_rgb", nv12, 1920, 1080)  # same leg, reversed
   # The router calls the DSP with cpu_fallback=False — a real
   # degradation is recorded once and the software leg runs exactly
   # once (no hidden in-client CPU pass first).

   # Keep-fd sources (Frame/FrameHandle) pass straight through to the
   # *_hw methods: their dma-bufs import zero-copy inside DspClient and
   # the router legs skip the ascontiguousarray copy (dsp-offload P2).
   # Array sources behave exactly as before.

JPEG encode: daemon one-shot RPC, CPU leg as fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   router = get_default_router()
   jpeg = router.run("encode_jpeg", rgb, quality=85)
   # Hardware leg = camera-daemon EncodeImage (DspClient.encode_jpeg_hw,
   # N-threaded libjpeg on the DSP core — hailo15 has no dedicated JPEG
   # encode block); software leg = cv2/Pillow. When the daemon does not
   # expose the RPC it degrades automatically and honestly:
   # health()["ops"]["encode_jpeg"]["backend"] shows the live backend.

Detection annotation: NV12 via DSP blend, RGB on the software raster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   router = get_default_router()
   annotated = router.run("draw_detections", nv12, result)
   # The hardware leg renders the annotation as a minimal RGBA canvas
   # (draw.render_overlay_rgba) and composites it back onto NV12 in one
   # DspClient.blend_hw job — NV12 in, NV12 out. RGB arrays stay on the
   # software leg (the draw_detections raster): round-tripping RGB
   # through two color converts would cost more than the raster it
   # offloads. An empty detection list returns a copy without touching
   # the DSP.

Forward degradations to the event bus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import EventClient, get_default_router

   bus = EventClient()
   router = get_default_router()

   def report(record):
       bus.publish("app/health/degradation", {
           "op": record.op,
           "reason": record.reason,
       })

   router.on_degradation = report

Hardware-only: fail instead of degrading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import AccelRouter, RoutePolicy

   # A HARDWARE_ONLY router never falls back — a hardware failure
   # raises HardwareUnavailable. For call sites where the zero-CPU
   # guarantee matters more than uptime.
   strict = AccelRouter(policy=RoutePolicy.HARDWARE_ONLY)
   strict.register("resize_nv12", hardware=my_dsp_resize)

Capability probes
~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(get_default_router().probe())
   # {'cv2': True, 'dsp': False}  <- dsp=False: service unreachable now

AI Overlay API
==============

.. automodule:: neoruntime_ipc_sdk.overlay
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

OverlayClient
-------------

.. autoclass:: neoruntime_ipc_sdk.OverlayClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

OverlayConfig
-------------

.. autoclass:: neoruntime_ipc_sdk.OverlayConfig
   :members:
   :undoc-members:

Usage Examples
--------------

Enable the overlay
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import OverlayClient

   overlay = OverlayClient()

   # Enable the hardware overlay: draw detection boxes + labels +
   # confidence on the video.
   overlay.enable(show_label=True, show_confidence=True, line_thickness=2)

   # Disable
   overlay.disable()

Custom styling
~~~~~~~~~~~~~~

.. code-block:: python

   # configure sets every style parameter in one call
   overlay.configure(
       enabled=True,
       show_label=True,
       show_confidence=False,
       line_thickness=3,
       box_color=0x00FF00,     # green boxes
       label_color=0xFFFFFF,   # white labels
       font_size=16,
   )

Structured configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import OverlayClient, OverlayConfig

   config = OverlayConfig(
       enabled=True,
       show_label=True,
       show_confidence=True,
       line_thickness=2,
   )
   OverlayClient().apply(config)

Combining with inference results (annotate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # annotate() pushes detections through the event bus to
   # camera-daemon's overlay renderer: the app never touches video
   # frames — the daemon draws the boxes onto the stream before
   # encoding. Results expire after 500 ms, so call at inference
   # cadence; publish an empty list to clear the screen.
   from neoruntime_ipc_sdk import OverlayClient

   overlay = OverlayClient()
   overlay.enable()

   for result in inference_results:      # e.g. InferenceClient.subscribe(...)
       overlay.annotate("main", result.objects)

   overlay.annotate("main", [])          # clear

Other result kinds
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # annotate_result picks whichever section of an InferenceResult is
   # populated, with precedence objects > classifications > landmarks
   # > ocr_lines
   overlay.annotate_result("main", result)

Context manager
~~~~~~~~~~~~~~~

.. code-block:: python

   with OverlayClient() as overlay:
       overlay.enable()

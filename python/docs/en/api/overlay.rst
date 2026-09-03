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

Combining with inference results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # The overlay is rendered by the daemon before encoding: the app
   # only publishes inference events and the daemon draws the boxes
   # onto the stream — the app never needs the video frames.
   from neoruntime_ipc_sdk import EventClient, OverlayClient

   events = EventClient()
   overlay = OverlayClient()
   overlay.enable()

   # Detection events published afterwards trigger overlay rendering
   events.publish("detection/vehicles", {"objects": [...]})

Context manager
~~~~~~~~~~~~~~~

.. code-block:: python

   with OverlayClient() as overlay:
       overlay.enable()

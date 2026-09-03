Detection Drawing API
=====================

.. automodule:: neoruntime_ipc_sdk.draw
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

draw_boxes
----------

.. autofunction:: neoruntime_ipc_sdk.draw_boxes

draw_text
---------

.. autofunction:: neoruntime_ipc_sdk.draw_text

draw_detections
---------------

.. autofunction:: neoruntime_ipc_sdk.draw_detections

Usage Examples
--------------

Draw boxes and labels
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import draw_boxes

   # boxes are pixel coordinates (x1, y1, x2, y2); labels/scores align
   # with boxes one-to-one
   annotated = draw_boxes(
       image,
       boxes=[(120, 80, 360, 300), (400, 200, 620, 460)],
       labels=["car", "person"],
       scores=[0.92, 0.87],
       color=(0, 255, 0),
       thickness=2,
   )

Draw arbitrary text
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import draw_text

   annotated = draw_text(
       image, "FPS: 15.2", (12, 24),
       color=(255, 255, 255), font_scale=0.5, thickness=1,
   )

Render inference results directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import InferenceClient, draw_detections

   infer = InferenceClient()
   result = infer.infer("yolov5m_vehicles", frame)

   # Accepts an InferenceResult or list[DetectedObject]; draws boxes +
   # labels + confidence automatically
   annotated = draw_detections(frame, result)

.. note::

   ``draw_*`` functions return a **copy** and never modify the input
   image; inputs/outputs are RGB pixel coordinates (matching
   :class:`~neoruntime_ipc_sdk.Frame.to_array` output).

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

draw_polygons
-------------

.. autofunction:: neoruntime_ipc_sdk.draw_polygons

render_overlay_rgba
-------------------

.. autofunction:: neoruntime_ipc_sdk.render_overlay_rgba

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
   rgb = frame.to_rgb()  # materialize the Frame (keep-fd frames cannot draw)

   # Accepts an InferenceResult or list[DetectedObject]; draws boxes +
   # labels + confidence automatically. Since SDK 0.7.4 an NV12 2D array
   # itself rides the DSP blend route; RGB arrays take the software raster
   annotated = draw_detections(rgb, result)

Polygons and tracks (zones / trajectory overlays, dsp-offload P2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import draw_polygons, render_overlay_rgba

   # shapes = [(points, color)]; points is an (N, 2) pixel coordinate
   # sequence, color=None means the default green. closed=False draws
   # open polylines (trajectories).
   zone = [(80, 60), (560, 60), (560, 380), (80, 380)]
   track = [(100, 240), (200, 200), (320, 210), (430, 260)]
   annotated = draw_polygons(image, [(zone, (255, 192, 0)),
                                      (track, (0, 200, 255))], closed=True)
   # tracks alone: draw_polygons(image, [(track, (0, 200, 255))], closed=False)

   # Hardware leg: the same shapes feed render_overlay_rgba's
   # polygons/tracks, join the boxes in the minimal canvas union, and
   # composite in one blend_hw call
   rgba, x0, y0 = render_overlay_rgba(
       w, h, boxes, labels, scores, colors,
       polygons=[(zone, (255, 192, 0))],
       tracks=[(track, (0, 200, 255))],
   )
   annotated = dsp.blend_hw(nv12, [(rgba, x0, y0)])

.. note::

   ``draw_*`` functions return a **copy** and never modify the input
   image; inputs/outputs are RGB pixel coordinates (matching
   :class:`~neoruntime_ipc_sdk.Frame.to_array` output).

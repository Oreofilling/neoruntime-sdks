Post-processing API
===================

.. automodule:: neoruntime_ipc_sdk.postprocess
   :members:
   :undoc-members:
   :no-index:

Functions
---------

nms
~~~

.. autofunction:: neoruntime_ipc_sdk.postprocess.nms
   :no-index:

Examples
--------

Filter overlapping boxes after decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from neoruntime_ipc_sdk import nms

   # Candidate boxes (xywh), scores and classes come from the model's
   # output-tensor decoding
   keep = nms(boxes, scores, iou_threshold=0.45, class_ids=class_ids)

   for i in keep:
       x, y, w, h = boxes[i]
       print(f"{labels[class_ids[i]]} {scores[i]:.2f} @ ({x:.0f},{y:.0f},{w:.0f},{h:.0f})")

Through the accel router (switches to hardware NMS when exposed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Software implementation today; once ai-runtime exposes NMS
   # registration params the hardware leg takes over automatically —
   # see docs/proposals/sdk-hardware-routing.md
   from neoruntime_ipc_sdk import get_default_router

   keep = get_default_router().run("nms", boxes, scores, iou_threshold=0.45)

检测框绘制 API
==============

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

使用示例
--------

绘制检测框与标签
~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import draw_boxes

   # boxes 为像素坐标 (x1, y1, x2, y2)；labels/scores 与 boxes 一一对应
   annotated = draw_boxes(
       image,
       boxes=[(120, 80, 360, 300), (400, 200, 620, 460)],
       labels=["car", "person"],
       scores=[0.92, 0.87],
       color=(0, 255, 0),
       thickness=2,
   )

绘制任意文本
~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import draw_text

   annotated = draw_text(
       image, "FPS: 15.2", (12, 24),
       color=(255, 255, 255), font_scale=0.5, thickness=1,
   )

直接渲染推理结果
~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import InferenceClient, draw_detections

   infer = InferenceClient()
   result = infer.infer("yolov5m_vehicles", frame)

   # 接受 InferenceResult 或 list[DetectedObject]，自动画框 + 标签 + 置信度
   annotated = draw_detections(frame, result)

.. note::

   ``draw_*`` 函数返回**副本**，不修改传入图像；输入/输出均为 RGB
   像素坐标（与 :class:`~neoruntime_ipc_sdk.Frame.to_array` 输出一致）。

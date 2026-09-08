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

draw_polygons
-------------

.. autofunction:: neoruntime_ipc_sdk.draw_polygons

render_overlay_rgba
-------------------

.. autofunction:: neoruntime_ipc_sdk.render_overlay_rgba

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
   rgb = frame.to_rgb()  # Frame 先物化（keep-fd 帧不能直接画）

   # 接受 InferenceResult 或 list[DetectedObject]，自动画框 + 标签 + 置信度
   # SDK 0.7.4 起：NV12 2D 数组本身即走 DSP blend 路由，RGB 数组走软件光栅
   annotated = draw_detections(rgb, result)

多边形与轨迹（区域/轨迹可视化，dsp-offload P2）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import draw_polygons, render_overlay_rgba

   # shapes = [(points, color)]；points 是 (N, 2) 像素坐标序列，
   # color=None 用默认绿色。closed=False 画开放折线（轨迹）。
   zone = [(80, 60), (560, 60), (560, 380), (80, 380)]
   track = [(100, 240), (200, 200), (320, 210), (430, 260)]
   annotated = draw_polygons(image, [(zone, (255, 192, 0)),
                                      (track, (0, 200, 255))], closed=True)
   # 轨迹单独画：draw_polygons(image, [(track, (0, 200, 255))], closed=False)

   # 硬件腿：同样的 shapes 进 render_overlay_rgba 的 polygons/tracks，
   # 与 boxes 一起并入最小画布并集，一次 blend_hw 合成
   rgba, x0, y0 = render_overlay_rgba(
       w, h, boxes, labels, scores, colors,
       polygons=[(zone, (255, 192, 0))],
       tracks=[(track, (0, 200, 255))],
   )
   annotated = dsp.blend_hw(nv12, [(rgba, x0, y0)])

.. note::

   ``draw_*`` 函数返回**副本**，不修改传入图像；输入/输出均为 RGB
   像素坐标（与 :class:`~neoruntime_ipc_sdk.Frame.to_array` 输出一致）。

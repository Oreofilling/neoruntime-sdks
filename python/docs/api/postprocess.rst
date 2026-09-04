后处理 API
==========

.. automodule:: neoruntime_ipc_sdk.postprocess
   :members:
   :undoc-members:
   :no-index:

函数
----

nms
~~~

.. autofunction:: neoruntime_ipc_sdk.postprocess.nms
   :no-index:

使用示例
--------

模型输出解码后过滤重叠框
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from neoruntime_ipc_sdk import nms

   # 解码后的候选框（xywh）、分数与类别来自模型输出层
   keep = nms(boxes, scores, iou_threshold=0.45, class_ids=class_ids)

   for i in keep:
       x, y, w, h = boxes[i]
       print(f"{labels[class_ids[i]]} {scores[i]:.2f} @ ({x:.0f},{y:.0f},{w:.0f},{h:.0f})")

配合加速路由（硬件 NMS 就绪后自动切换）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 当前为软件实现；ai-runtime 暴露 NMS 注册参数后，硬件腿会
   # 自动接管（见 docs/proposals/sdk-hardware-routing.md）
   from neoruntime_ipc_sdk import get_default_router

   keep = get_default_router().run("nms", boxes, scores, iou_threshold=0.45)

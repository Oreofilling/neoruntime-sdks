色彩转换 API
============

.. automodule:: neoruntime_ipc_sdk.color
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

函数
----

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

使用示例
--------

编码结果直推 NV12 模型
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import rgb_to_nv12, nv12_resize

   # 把画好 overlay 的 RGB 帧送进 NV12 模型：先编码、再整帧缩放，
   # 全程不经过 NV12->RGB->缩放->RGB->NV12 的往返转换
   nv12 = rgb_to_nv12(rgb_frame)
   small = nv12_resize(nv12, (1920, 1080), (640, 384))

配合 DSP 硬件缩放
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 优先走 DSP 硬件（见加速路由 API），DSP 不可用时自动回落到
   # 上面的 numpy/cv2 软件实现，并在 health() 中记录一次降级
   from neoruntime_ipc_sdk import get_default_router

   router = get_default_router()
   small = router.run("resize_nv12", nv12, (1920, 1080), (640, 384))

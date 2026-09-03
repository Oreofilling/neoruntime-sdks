DSP 硬件加速 API
================

.. automodule:: neoruntime_ipc_sdk.dsp
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

DspClient
---------

.. autoclass:: neoruntime_ipc_sdk.DspClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

DspBufferPool
~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.DspBufferPool
   :members:
   :undoc-members:

DspError
~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.DspError
   :members:
   :undoc-members:

使用示例
--------

整帧缩放（模型输入预处理）
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import FdMediaClient, DspClient, DspError

   media = FdMediaClient()
   dsp = DspClient()

   # keep_fd=True 保留 dma-buf fd，DspClient 零拷贝引用同一缓冲区
   frame = media.get_frame("main", timeout_ms=3000, keep_fd=True)

   try:
       # NV12 in -> NV12 out（h + h/2 行）；scaling="stretch" 等价 cv2.resize
       small = dsp.resize_hw(frame, 640, 384)
   except DspError:
       # DSP 不可用且源是 keep-fd 帧时 SDK 会抛错（拒绝静默 CPU 回退），
       # 业务侧在此自行走 CPU 路径
       small = frame.to_array()

   frame.release()  # 尽早归还 dma-buf（幂等，可省略）

批量裁剪（letterbox 缩放）
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # rects: (x, y, w, h, dst_w, dst_h)，源坐标为像素值且需偶数对齐，
   # 结果顺序与 rects 一致。适合"一帧多目标"（如车牌 tile）场景，
   # 多个裁剪合并为一次硬件任务。
   rects = [
       (320, 500, 160, 48, 320, 48),
       (900, 520, 150, 44, 320, 48),
   ]
   tiles = dsp.multi_crop_hw(frame, rects, scaling="letterbox")

单区域裁剪
~~~~~~~~~~

.. code-block:: python

   # crop_hw 支持裁剪后同步缩放到目标尺寸
   tile = dsp.crop_hw(frame, 320, 500, 160, 48, dst_width=320, dst_height=48)

预分配缓冲池（高帧率重复任务）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 同几何参数的重复任务可用池化缓冲区，避免每次分配 dma-buf
   pool = dsp.alloc_buffers(640, 384, fmt="nv12", count=4)
   small = dsp.resize_hw(frame, 640, 384, dst_pool=pool)

   # 读取池内缓冲区内容
   arr = pool.read(0)

   # 不再使用时归还整池
   pool.release()

上下文管理器
~~~~~~~~~~~~

.. code-block:: python

   with DspClient() as dsp:
       out = dsp.resize_hw(frame, 416, 416)

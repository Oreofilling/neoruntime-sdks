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

格式转换（RGB ↔ NV12 / 灰度）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # convert_hw 同尺寸换格式（daemon CONVERT 的 P0 契约）：无 rect、
   # 单目的缓冲；dst_fmt 为 "nv12"/"rgb24"/"gray8"。
   # 注意字节序：wire 上的 rgb24 是 RGB 序——BGR 像素需先交换 R/B
   #（或 CPU 路径直接用 color.bgr_to_nv12）。
   nv12 = dsp.convert_hw(rgb, "nv12", fmt="rgb24")
   gray = dsp.convert_hw(rgb, "gray8", fmt="rgb24")

   # 需要缩放时先 CONVERT 再 RESIZE：NV12 字节数约为 rgb24 的一半，
   # 先转再缩搬运的数据量减半。
   small = dsp.resize_hw(nv12, 640, 384)

   # DSP 不可用时默认回落 CPU（UserWarning 提示）；传 cpu_fallback=False
   # 则直接抛 DspError——路由器用它在降级计数里如实记账。

   # 固件支持矩阵与设备相关：hailo15 实测（93.72）仅 rgb24 <-> nv12
   # 走 DSP，gray8 各组合被固件拒绝（HAL rc=-2801）。默认
   # cpu_fallback=True 下这类"作业被拒"同样回落 CPU 并告警，
   # last_used_hw=False 如实记录实际后端。

单帧 JPEG 编码（快照/缩略图）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # encode_jpeg_hw 走 daemon 的 EncodeImage 一发 RPC：源缓冲零拷贝
   # 注册进 DSP registry（keep-fd 帧直接导入 dma-buf，numpy 数组拷入
   # 池缓冲），完整 JPEG 字节随响应返回——无目的缓冲、无回读。
   jpeg = dsp.encode_jpeg_hw(frame, quality=85, fmt="nv12")
   # 数组源则 rgb24 默认：jpeg = dsp.encode_jpeg_hw(rgb, quality=85)

   # 名字里虽无"硬件块"：编码器是 DSP 核上 N 线程 libjpeg（GStreamer
   # 分发）——hailo15 没有专用 JPEG 编码块。收益是集中编码 + 零拷贝
   # 输入（app 镜像可省掉 cv2/PIL），不是原始速度；逐帧高频编码仍以
   # CPU 路径为宜。encoder 按 (宽,高,格式,quality) 复用，参数一变
   # 就重建（变更后首帧承担流水线启动开销）。

标注合成（检测框画到 NV12 上，dsp-offload P1）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import render_overlay_rgba

   # blend_hw 把 ARGB32 overlay 逐个 1:1 粘贴合成到 NV12 base 上
   #（不缩放，后贴的盖先贴的）。合成发生在池拷贝上、原地写回——
   # 返回已标注的 NV12 数组，输入数组绝不改动。
   overlay = np.zeros((64, 96, 4), np.uint8)   # (h, w, 4) RGBA
   overlay[..., :3] = (255, 0, 0)
   overlay[..., 3] = 255                       # 直通 alpha
   annotated = dsp.blend_hw(nv12, [(overlay, 40, 30)])

   # 与 render_overlay_rgba 组合即为"检测画框走硬件"：
   rgba, x0, y0 = render_overlay_rgba(w, h, boxes, labels, scores, colors)
   annotated = dsp.blend_hw(nv12, [(rgba, x0, y0)])
   # 更省事的入口是 accel 路由器：router.run("draw_detections", nv12, result)

   # 契约要点：base 必须是 NV12 数组（vendor op 只写 NV12；keep-fd 帧
   # 属于相机，原地合成会改写它——用 frame.to_array() 接受拷贝）；
   # overlay 小于 16x16（daemon 下限）自动补全透明像素到 16；
   # 硬件 ARGB32 内存字节序为 [A, R, G, B]，SDK 内部打包；
   # quota 按 (base + 各 overlay) 像素量计费——最小画布（上面
   # render_overlay_rgba 正是）才省。DSP 不可用/作业被拒时与其它
   # *_hw 相同：默认告警回落 CPU（_cpu_blend 直通 alpha 数学一致），
   # cpu_fallback=False 直接抛错。

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

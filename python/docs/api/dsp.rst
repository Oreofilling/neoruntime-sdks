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

PendingDspJob
~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.PendingDspJob
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
   # last_used_hw=False 如实记录实际后端。首次被拒后 SDK 会记住该
   # 固件缺口：后续 gray8 数组对直接走 CPU 腿——不再告警、不再提交
   # 必败作业（keep-fd 源无法走 CPU 腿，仍会如实抛错）。

单帧 JPEG 编码（快照/缩略图）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # encode_jpeg_hw 走 daemon 的 EncodeImage 一发 RPC：源缓冲零拷贝
   # 注册进 DSP registry（keep-fd 帧直接导入 dma-buf，numpy 数组拷入
   # 池缓冲），完整 JPEG 字节随响应返回——无目的缓冲、无回读。
   jpeg = dsp.encode_jpeg_hw(frame, quality=85, fmt="nv12")
   # 数组源则 rgb24 默认：jpeg = dsp.encode_jpeg_hw(rgb, quality=85)

   # P2 直通腿：src_buffer_id 把编码接到池缓冲/异步作业上——结果
   # 留在设备侧，零回读（与 wait=False 的 PendingDspJob.buffer_id
   # 组成零拷贝链尾，见下方"异步作业"）。
   jpeg = dsp.encode_jpeg_hw(None, quality=85, src_buffer_id=job.buffer_id)
   # src 与 src_buffer_id 互斥；此腿没有客户端像素可回退，
   # DSP 不可用时直接抛 DspError。

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

   # 契约要点：base 必须是 NV12（vendor op 只写 NV12）。数组走池拷贝
   # 原地合成、返回已标注的 NV12 数组；keep-fd 帧默认**直接拒绝**
   # （见下方"零拷贝合成链"的固件缺陷记录）；
   # overlay 小于 16x16（daemon 下限）自动补全透明像素到 16；
   # 硬件 ARGB32 内存字节序为 [A, R, G, B]，SDK 内部打包；
   # quota 按 (base + 各 overlay) 像素量计费——最小画布（上面
   # render_overlay_rgba 正是）才省。DSP 不可用/作业被拒时与其它
   # *_hw 相同：默认告警回落 CPU（_cpu_blend 直通 alpha 数学一致），
   # cpu_fallback=False 直接抛错（keep-fd base 无客户端像素可回退，
   # 不可用时必抛）。

异步作业（submit 后择机 wait，dsp-offload P2）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 五个作业方法（resize/crop/multi_crop/convert/blend）都接受
   # wait=False：SubmitDspJobAsync 立即返回 job_id，返回
   # PendingDspJob 句柄。老 daemon（无该 rpc）自动回落同步提交——
   # 得到的句柄"生而完成"，wait() 无额外 RPC。
   job1 = dsp.resize_hw(frame, 640, 384, wait=False)
   job2 = dsp.multi_crop_hw(frame, rects, wait=False)
   ...  # 提交与执行重叠：单个 worker 线程按提交顺序执行

   small = job1.wait()          # WaitDspJob + 池回读 + 释放（阻塞）
   tiles = job2.wait()          # multi_crop 返回列表
   if job1.done():              # 非阻塞轮询（timeout 0）；完成即缓存
       ...
   job1.wait_result()           # 只等完成不回读——结果留在设备侧
   bid = job1.buffer_id         # 链喂 encode_jpeg_hw(src_buffer_id=)
   job1.release()               # 丢弃结果、归还缓冲（幂等）

   # 语义要点：wait 超时（rc=-4）表项保留可再等；job 失败在 wait
   # 时抛 DspError（done() 轮询只报状态）；release 后再 wait 抛错；
   # wait=False 不改变 CPU 回退语义——回退发生时拿到的就是像素
   # 数组（没有可等的硬件作业）。

零拷贝合成链（固件缺陷记录：默认拒绝）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # keep-fd 帧直入 blend_hw 默认抛错——不是能力缺失，而是当前
   # hailo15 固件上这条链**会挂死整个 DSP**（93.72 两度复现：720p
   # 主管道在跑、常规 CMA 尚余 700MB，blend 命令提交后固件永不
   # 返回，设备级卡死，仅重启可恢复；xrp 驱动自判 fatal error）。
   # 数组 base（frame.to_array()）是已验证的安全路径。
   try:
       dsp.blend_hw(frame, [(rgba, x0, y0)])
   except DspError:
       annotated = dsp.blend_hw(frame.to_array(), [(rgba, x0, y0)])

   # zero_copy=True 显式强制该链（导入 dma-buf → 1:1 RESIZE 落池 →
   # 原地 BLEND，配 wait=False 结果不过 socket，
   # encode_jpeg_hw(src_buffer_id=job.buffer_id) 免回读）——供未来
   # 固件修复后实验用，今天不保证可用：
   job = dsp.blend_hw(frame, [(rgba, x0, y0)],
                      wait=False, zero_copy=True)
   job.wait_result()                              # 结果留在池里
   jpeg = dsp.encode_jpeg_hw(None, src_buffer_id=job.buffer_id)
   job.release()                                  # 编码完再归还

   # 注意链式顺序：RESIZE 拷贝腿始终同步执行（无人 wait 的异步
   # 作业会泄漏 daemon 侧登记项，也占用每连接 32 个未决槽位）——
   # 异步的只有 BLEND 合成腿。详情见
   # docs/proposals/dsp-offload.md 的 P2 记录。

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

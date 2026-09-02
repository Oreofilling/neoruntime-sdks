视频流 API
==========

.. automodule:: neoruntime_ipc_sdk.media
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

FdMediaClient
-------------

.. autoclass:: neoruntime_ipc_sdk.FdMediaClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

数据类型
--------

Frame
~~~~~

.. autoclass:: neoruntime_ipc_sdk.Frame
   :members:
   :undoc-members:
   :no-index:

StreamInfo
~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.StreamInfo
   :members:
   :undoc-members:

PixelFormat
~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.PixelFormat
   :members:
   :undoc-members:

EncodedStreamClient
~~~~~~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.EncodedStreamClient
   :members:
   :undoc-members:

EncodedFrame
~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.EncodedFrame
   :members:
   :undoc-members:

使用示例
--------

获取原始视频流
~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import FdMediaClient
   import cv2

   media = FdMediaClient()

   # 获取主码流（可用流 ID 见 list_streams()，通常为 main / sub）
   for frame in media.subscribe("main"):
       print(f"帧 {frame.sequence}: {frame.width}x{frame.height}")
       print(f"格式: {frame.format}, 时间戳: {frame.timestamp_ns}")

       # frame.image 是 numpy array (H, W, C) 或 (H*3//2, W) 对于 NV12
       # 可以直接用于 OpenCV 或其他图像处理库
       cv2.imshow("Camera", frame.image)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cv2.destroyAllWindows()

不跳过帧
~~~~~~~~

.. code-block:: python

   # 获取每一帧（不跳过）
   for frame in media.subscribe("main", skip_frames=False):
       process_frame(frame.image)

获取单帧
~~~~~~~~

.. code-block:: python

   # 获取单帧
   frame = media.get_frame("main", timeout_ms=1000)

   if frame:
       print(f"帧: {frame.width}x{frame.height}")
       print(f"格式: {frame.format}")

获取流信息
~~~~~~~~~~

.. code-block:: python

   # FdMediaClient 没有独立的流信息接口，从首帧读取尺寸与格式
   frame = media.get_frame("main", timeout_ms=5000)

   if frame:
       print(f"分辨率: {frame.width}x{frame.height}")
       print(f"格式: {frame.format}")
       print(f"可用流: {media.list_streams()}")

列出可用流
~~~~~~~~~~~

.. code-block:: python

   # 列出所有可用的流
   streams = media.list_streams()
   for stream_id in streams:
       print(f"流: {stream_id}")

帧处理回调
~~~~~~~~~~

.. code-block:: python

   def handle_frame(frame):
       print(f"收到帧: {frame.sequence}")

   # 使用回调处理帧
   thread = media.on_frame("main", handle_frame)

   # 保持运行
   import time
   while True:
       time.sleep(1)

图像处理
~~~~~~~~

.. code-block:: python

   import cv2
   import numpy as np

   media = FdMediaClient()

   for frame in media.subscribe("main"):
       # 转换为 RGB
       rgb = frame.to_rgb()

       # 灰度图
       gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

       # 边缘检测
       edges = cv2.Canny(gray, 100, 200)

       # 显示结果
       cv2.imshow("Original", rgb)
       cv2.imshow("Edges", edges)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

保存图像
~~~~~~~~

.. code-block:: python

   media = FdMediaClient()

   for frame in media.subscribe("main"):
       # 保存为图片
       frame.save("frame.jpg")
       break

获取编码流
~~~~~~~~~~

``get_encoded_stream()`` 返回 :class:`EncodedStreamClient`（不是迭代器），产出 H.264/H.265 Annex-B 码流包，配合 ``recording`` 模块可直接落盘：

.. code-block:: python

   from neoruntime_ipc_sdk import FdMediaClient

   media = FdMediaClient()

   # get_encoded_stream() 返回 EncodedStreamClient，不是迭代器
   client = media.get_encoded_stream("main")

   for packet in client.subscribe():
       print(f"{packet.codec_name()} 包: {len(packet.data)} 字节")
       if packet.is_keyframe():
           print("  关键帧")

多流处理
~~~~~~~~

.. code-block:: python

   import threading

   def process_main_stream():
       media = FdMediaClient()
       for frame in media.subscribe("main"):
           # 处理主码流（高分辨率）
           process_high_res(frame.image)

   def process_sub_stream():
       media = FdMediaClient()
       for frame in media.subscribe("sub"):
           # 处理子码流（低分辨率）
           process_low_res(frame.image)

   # 并行处理多个流
   t1 = threading.Thread(target=process_main_stream)
   t2 = threading.Thread(target=process_sub_stream)

   t1.start()
   t2.start()

   t1.join()
   t2.join()

帧率控制
~~~~~~~~

.. code-block:: python

   import time

   media = FdMediaClient()
   target_fps = 10
   frame_interval = 1.0 / target_fps

   last_time = time.time()

   for frame in media.subscribe("main"):
       current_time = time.time()
       elapsed = current_time - last_time

       if elapsed >= frame_interval:
           # 处理帧
           process_frame(frame.image)
           last_time = current_time

保存视频
~~~~~~~~

如需保存 H.264/H.265 编码流，更推荐直接复用 ``recording`` 模块的 ``TsWriter`` / ``HlsWriter``，无需解码重编码。下面的示例演示解码后用 OpenCV 保存原始帧：

.. code-block:: python

   import cv2

   media = FdMediaClient()

   # 从首帧获取分辨率（FdMediaClient 没有独立的流信息接口）
   first = media.get_frame("main", timeout_ms=5000)
   if first is None:
       raise RuntimeError("未收到视频帧")

   # 创建视频写入器（fps 需按实际码流配置填写，这里以 30 为例）
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter(
       'output.mp4',
       fourcc,
       30.0,
       (first.width, first.height)
   )

   frame_count = 0
   max_frames = 300  # 录制 10 秒 (30fps)

   for frame in media.subscribe("main"):
       rgb = frame.to_rgb()
       bgr = rgb[:, :, ::-1]  # RGB to BGR
       out.write(bgr)
       frame_count += 1

       if frame_count >= max_frames:
           break

   out.release()
   print(f"已保存 {frame_count} 帧到 output.mp4")

上下文管理器
~~~~~~~~~~~~

.. code-block:: python

   # 使用上下文管理器自动管理资源
   with FdMediaClient() as media:
       for frame in media.subscribe("main"):
           process_frame(frame.image)

零拷贝访问
~~~~~~~~~~

.. code-block:: python

   # 默认接收路径在收到帧时拷贝像素数据（frame.image 立即可用）
   # keep_fd=True 保留 dma-buf fd，实现真正的零拷贝：
   # 缓冲区延迟到 frame.release() / GC / 客户端关闭时才归还 daemon

   media = FdMediaClient()

   for frame in media.subscribe("main", keep_fd=True):
       # frame.handle 持有 dma-buf fd
       # frame.to_array() 首次调用时映射一次并缓存像素数据
       result = inference_engine.process(frame.to_array())

       # 处理完尽早归还缓冲区（幂等，可省略）
       frame.release()

错误处理
~~~~~~~~

.. code-block:: python

   media = FdMediaClient()

   try:
       for frame in media.subscribe("invalid_stream"):
           process_frame(frame.image)
   except Exception as e:
       print(f"流访问失败: {e}")
   except KeyboardInterrupt:
       print("用户中断")
   finally:
       media.close()

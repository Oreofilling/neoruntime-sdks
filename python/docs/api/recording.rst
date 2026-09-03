录像 API
========

.. automodule:: neoruntime_ipc_sdk.recording
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

TsWriter
--------

.. autoclass:: neoruntime_ipc_sdk.TsWriter
   :members:
   :undoc-members:

HlsWriter
---------

.. autoclass:: neoruntime_ipc_sdk.HlsWriter
   :members:
   :undoc-members:

PrerollBuffer
-------------

.. autoclass:: neoruntime_ipc_sdk.PrerollBuffer
   :members:
   :undoc-members:

使用示例
--------

录制 MP4（TsWriter）
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import EncodedMediaClient, TsWriter

   media = EncodedMediaClient()
   writer = TsWriter("/data/aipc/recordings/clip.mp4", codec="h264")

   try:
       # 订阅编码流，按 PTS 写入
       for frame in media.subscribe("main"):
           writer.write(frame)
           if enough_recorded(frame):
               break
   finally:
       writer.close()  # 必须 close 以落盘 moov/索引

HLS 直播切片
~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import EncodedMediaClient, HlsWriter

   media = EncodedMediaClient()

   # 每 6s 一个分片，保留最近 5 个分片的播放列表
   hls = HlsWriter("/data/aipc/recordings/live", segment_seconds=6.0, window=5)

   try:
       for frame in media.subscribe("main"):
           hls.write(frame)
   finally:
       hls.close()

预滚录像（事件前缓存）
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import EncodedMediaClient, PrerollBuffer

   media = EncodedMediaClient()

   # 环形缓存最近 10s 编码帧
   preroll = PrerollBuffer(seconds=10.0)

   for frame in media.subscribe("main"):
       preroll.push(frame)
       if event_detected():
           # dump：预滚内容 + 后续帧写成 ts 文件
           writer = preroll.dump("/data/aipc/recordings/alert.ts")
           # ... 继续向 writer.write(frame) 追加事件后内容
           writer.close()
           break

自定义帧回调
~~~~~~~~~~~~

.. code-block:: python

   # dump 的 on_frame 在写出每一帧前被调用，可用来过滤/修改
   def mark(frame):
       print(f"writing pts={frame.pts_ns}")
       return frame

   writer = preroll.dump("/data/aipc/recordings/alert.ts", on_frame=mark)

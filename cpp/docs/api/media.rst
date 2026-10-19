取流与媒体：FdMediaClient / EncodedStreamClient / AudioStreamClient
====================================================================

零拷贝取帧走 ``fd_client.hpp`` （``media.hpp`` 门面），编码流走
``encoded.hpp``，音频采集流走 ``audio_stream.hpp``。

零拷贝视频帧
------------

``FdMediaClient`` 说 FdPublisher UDS 协议：每帧是 80 字节
``FdPubFrameMsg`` + 最多 3 个 DMA-BUF fd 经 ``SCM_RIGHTS`` 传来。默认
路径把 plane ``mmap`` 后拷进 ``cv::Mat`` 立即归还；``keep_fd=true``
保留 dma-buf fd，daemon 侧 RELEASE 推迟到最后一帧副本消失。

拉流（有界自动重连，断流重试耗尽后 ``next()`` 返回 ``std::nullopt``）：

.. code-block:: cpp

   neoruntime_ipc_sdk::FdMediaClient media;

   auto streams = media.list_streams();          // 活动管线流 id
   auto stream  = media.subscribe(streams[0]);   // skip_frames 可抽帧
   while (auto frame = stream.next()) {
       std::printf("seq=%llu %dx%d\n",
                   (unsigned long long)frame->sequence,
                   frame->width, frame->height);
   }

单帧快照带超时；``Frame`` 自带 ``to_rgb()`` / ``crop`` / ``resize`` /
JPEG 编码：

.. code-block:: cpp

   if (auto frame = media.get_frame("main", /*timeout_ms=*/2000)) {
       cv::Mat bgr;
       cv::cvtColor(frame->to_rgb(), bgr, cv::COLOR_RGB2BGR);
       cv::imwrite("snapshot.jpg", bgr);
   }

.. tip::

   只需要每 N 帧处理一次时，用 ``subscribe(stream_id, /*skip_frames=*/N)``
   让 SDK 在源头抽帧，而不是收下全部再自己丢——被抽掉的 keep-fd 帧
   会立即归还 daemon，不占缓冲池。

keep-fd 与缓冲池纪律
--------------------

.. code-block:: cpp

   // 后台线程版：回调每帧一次（cb 异常被吞；自行 detach/join）
   std::thread t = media.on_frame("main", [&](const Frame& f) {
       // 零拷贝持有 dma-buf 的处理路径
   });

.. warning::

   ``retained_frames()`` 报告当前持有的 keep-fd 帧数。camera daemon
   的缓冲池是个位数——持有量逼近池容量会 **饿死直播流**。用完的帧尽快
   ``release()``，退出前调用 ``media.close()`` 归还全部。

编码流与音频流
--------------

编码流客户端从 ``FdMediaClient`` 拿句柄，V3 序号自带缺口 / 重启计数：

.. code-block:: cpp

   auto enc = media.get_encoded_stream("main");   // H.264 / H.265
   while (auto packet = enc.get_frame(2000)) {
       std::printf("codec=%s bytes=%zu\n",
                   packet->codec_name().c_str(), packet->data.size());
   }

音频采集流同一拉取形状，帧携带 ``sample_rate`` / ``channels`` /
``bits_per_sample`` / ``dts_ns``：

.. code-block:: cpp

   neoruntime_ipc_sdk::AudioStreamClient audio;
   if (auto chunk = audio.get_frame(2000)) {
       std::printf("%d Hz %d ch %d-bit\n", chunk->sample_rate,
                   chunk->channels, chunk->bits_per_sample);
   }

.. note::

   RTSP 拉流需先在设备上开启 RTSP （``CameraClient`` 或 REST API），
   然后 ``media.get_rtsp_url("main")`` 取地址。流能力查询用
   ``CameraClient::get_capabilities`` / ``get_stream_status`` （ISP、
   编码器、profile、隐私遮挡——见 :doc:`../modules` 总览）。

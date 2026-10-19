硬件加速：OverlayClient 与 DspClient
=====================================

把像素操作推出你的进程：``overlay.hpp`` 让检测框 / 多边形画在设备
编码侧（应用 CPU 零开销），``dsp.hpp`` 把缩放 / 转换 / 合成 / JPEG
编码交给 DSP。

硬件 overlay
------------

两个半面：**配置** 走 camera-control RPC，**内容** 走事件总线——
daemon 的 overlay 订阅者把检测框画到 NV12 帧上再编码。

.. code-block:: cpp

   neoruntime_ipc_sdk::OverlayClient overlay;
   overlay.connect();
   overlay.enable(/*show_label=*/true, /*show_confidence=*/true, 2);

   // 检测框 + 区域多边形一次推齐（均归一化 [0,1] 坐标）
   std::vector<neoruntime_ipc_sdk::OverlayPolygon> zones;
   neoruntime_ipc_sdk::OverlayPolygon zone;
   zone.points = {{0.10f, 0.10f}, {0.90f, 0.10f},
                  {0.90f, 0.90f}, {0.10f, 0.90f}};
   zone.label  = "watch zone";
   zones.push_back(zone);

   overlay.annotate("main", result.objects, zones, /*ttl_ms=*/2000);

   // 或整包推送 InferenceResult（自动挑 objects/classifications/
   // landmarks/ocr_lines 中已填的那一节）
   overlay.annotate_result("main", result);

.. note::

   内容事件在最后一次推送 ``ttl_ms`` 后过期（缺省按流 fps 推导），所以
   检测新鲜期间要以几 Hz 持续调 ``annotate``；overlay 本身必须先
   ``enable()``。``detections`` 传 ``{}`` 是 **清空** 框，传
   ``std::nullopt`` 是保持不动（只想推多边形时用后者）。

绑帧与会话：``session_id`` 作生命周期标签；``frame_sequence`` /
``stream_epoch`` （取自 ``InferenceResult`` / ``get_stream_status()``）
把事件绑到具体帧 / 流代际。严格帧锁（每帧在烘焙点等 **自己的** 推理
结果，上限 ``strict_wait_cap_ms``，范围 [0, 5000]）用 ``configure()``
开启。

DSP 作业
--------

``DspClient`` 同时保持一条 gRPC 与一条 UDS 连接。作业面：
``resize_hw`` / ``crop_hw`` / ``multi_crop_hw`` / ``convert_hw`` /
``blend_hw`` / ``encode_jpeg_hw``；``last_used_hw()`` 报告上一作业
走的是硬件还是 CPU 回退。

.. code-block:: cpp

   neoruntime_ipc_sdk::DspClient dsp;

   // 硬件缩放：输入 cv::Mat，输出 variant（Mat 或错误）
   auto out = dsp.resize_hw(input_nv12, 320, 240, "nv12");
   if (auto* mat = std::get_if<cv::Mat>(&out)) {
       std::printf("resize done, hw=%d\n", (int)dsp.last_used_hw());
   }

**零读回链** 是 DSP 面的杀手锏——产物以 daemon 侧缓冲 id 存在，下一
个作业直接引用，应用进程不过像素：

.. code-block:: cpp

   neoruntime_ipc_sdk::DspJobParams ref_out;
   ref_out.out_ref = true;

   auto ref_variant = dsp.resize_hw(big_nv12, 1280, 720, "nv12",
                                    nullptr, ref_out);
   if (auto* ref = std::get_if<neoruntime_ipc_sdk::DspBufferRef>(
           &ref_variant)) {
       // JPEG 编码直接吃 buffer_id，无读回
       auto jpeg = dsp.encode_jpeg_hw(nullptr, 85, "nv12", 5.0,
                                      nullptr, false, ref->buffer_id());
       if (auto* bytes = std::get_if<std::vector<std::uint8_t>>(&jpeg)) {
           /* *bytes 即完整 JPEG */
       }
   }

.. tip::

   缓冲池（``BufferPool::alloc_buffers``）适合反复提交同尺寸中间帧的
   场景；一次性作业直接喂 ``cv::Mat`` 即可，daemon 会导入。gray8 输入
   走 DSP 会被固件拒绝——路由层自动落 CPU 腿，无需分支处理。

流水线里想要"订阅 → 推理 → overlay → 结果"一站成型的话，看
:doc:`pipeline` 的 ``StreamPipeline`` （多边形随构造参数直送）。

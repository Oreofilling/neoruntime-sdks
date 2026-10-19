推理：InferenceClient
=====================

头文件 ``inference.hpp`` （值类型在 ``inference_types.hpp``）。AI 推理的
全部入口：单次 / 异步 / 批量 / 多张量、流式订阅、模型注册与统计、
GenAI 流式。

单次与异步推理
--------------

``infer()`` 接受 ``cv::Mat`` 像素或 ``InferInput`` （零拷贝输入：keep-fd
NV12 帧 / 守护进程侧 ``BufferRefInput``）：

.. code-block:: cpp

   neoruntime_ipc_sdk::InferenceClient inference;
   inference.connect();

   cv::Mat probe = cv::imread("probe.jpg");
   auto result = inference.infer(probe, "person_v1");

   // 异步版（std::async / std::future，调用方负责 wait）
   auto fut = inference.infer_async(probe, "person_v1");

.. tip::

   超时（默认 5000 ms）、优先级（默认 4）与 ``session_id`` 是 ``infer()``
   的可选尾参；批量用 ``infer_batch`` （超时默认 10000 ms）。

结果对象 ``InferenceResult`` 按模型类型携带 ``objects`` /
``classifications`` / ``landmarks`` / ``ocr_lines``，并提供标签过滤：

.. code-block:: cpp

   auto persons = result.get_objects_by_label("person");
   for (const auto& p : persons) {
       double cx = p.bbox.x + p.bbox.width / 2.0;
       double cy = p.bbox.y + p.bbox.height / 2.0;
       // p.score、p.track_id（>= 0 时随 overlay 渲染为 "#id"）
   }

流式订阅
--------

对相机流订阅推理，后台 pump 线程把服务端流读进 **有界丢最旧** 队列；
``next()`` 返回 ``(frame_sequence, InferenceResult)``，跳过失败帧，
流干净结束返回 ``std::nullopt``：

.. code-block:: cpp

   auto stream = inference.subscribe(
       "cam0_main", "person_v1",
       /*fps=*/15,
       /*session_id=*/"",
       /*raw_output_only=*/false,
       /*max_consecutive_failures=*/10,
       /*queue_size=*/100);

   while (auto item = stream.next()) {
       const auto frame_seq = item->first;
       const auto& result   = item->second;
       if (!result.objects.empty()) { /* ... */ }
   }

健康遥测直接挂在流对象上——消费太慢时最先看这里：

.. code-block:: cpp

   std::printf("dropped=%llu avg_latency=%.1fms avg_skew=%.0fus\n",
               (unsigned long long)stream.dropped(),
               stream.avg_latency_ms(), stream.avg_skew_us());

.. note::

   ``dropped()`` 计的是 **客户端侧** 因消费慢被丢弃的最旧结果；时延与
   偏移是 EMA（alpha 0.1）。连续失败达 ``max_consecutive_failures``
   （默认 10；``std::nullopt`` 关闭）后 ``next()`` 抛
   ``std::runtime_error`` 而非无限跳帧。``cancel()`` 可从任意线程停止
   订阅（阻塞在 ``next()`` 的消费者会醒来拿到 ``nullopt``）。

模型注册
--------

应用自带模型在启动时注册一次，全设备共享：

.. code-block:: cpp

   const std::string model_id =
       inference.register_model("/app/models/person_v1.hef",
                                "person_v1", "", "detection");

.. warning::

   手动注册 **必须留空 owner_id**：带 owner_id 的注册不进全局模型表，
   其他客户端 ``subscribe()`` / ``list_models()`` 会报 "Model not
   found"。批量模型（``batch_size > 1``，需要 batch 能力的 HEF）只能用
   ``infer_batch()``，单图 ``infer()`` 会被拒绝。

系统统计与 GenAI
----------------

.. code-block:: cpp

   auto stats = inference.get_stats();
   std::printf("NPU util %.0f%%, temp %.1f°C\n",
               stats.device_utilization, stats.device_temperature);

   // GenAI：建会话 → 逐 token 流
   auto sid = inference.genai_create_session("/app/models/llm.hef");
   auto gen = inference.genai_generate(sid, {"用一句话介绍边缘计算"});
   while (auto token = gen.next()) std::fputs(token->c_str(), stdout);
   inference.genai_destroy_session(sid);

完整 API 面见 `Doxygen 参考 <../../en/index.html>`_；客户端面与
Python 侧逐位对齐，语义细节可参照
`/python/zh/ <../../../python/zh/index.html>`_ 对应章节。

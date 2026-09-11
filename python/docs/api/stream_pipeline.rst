整管线 API（平台调度管线）
==========================

.. automodule:: neoruntime_ipc_sdk.stream_pipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

StreamPipeline
--------------

.. autoclass:: neoruntime_ipc_sdk.StreamPipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

StreamPipelineStatus
--------------------

.. autoclass:: neoruntime_ipc_sdk.StreamPipelineStatus
   :members:
   :undoc-members:

使用示例
--------

一次启动：订阅推理 + 硬件叠加 + 应用侧结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import StreamPipeline

   zones = [{"points": [[0, 0], [1, 0], [1, 1]], "label": "yard"}]
   pipe = StreamPipeline(
       "main", "yolov5n", fps=10,
       min_score=0.5, labels=["person"], polygons=zones,
       on_result=lambda r: r if r.objects else None,  # 空结果不画不进队列
   ).start()

   # 可选：应用侧同时消费结果（告警、二次处理、转发）。
   # 叠加画面不经过这里 —— 框已由硬件叠加层烘进编码流。
   for seq, result in pipe.results():
       print(seq, [(o.label, round(o.score, 2)) for o in result.objects])
       if seq >= 100:
           break

   st = pipe.status()          # results_seen / annotated / drops / skew ...
   pipe.stop()                 # 清除检测框与静态多边形，编码画面恢复干净

语义要点
~~~~~~~~

- **与 InferencePipeline（pipeline.py）的定位区分**：StreamPipeline 是
  平台调度管线 —— 帧由 camera daemon 喂给模型（``subscribe``），像素
  不进应用进程，结果经 ``OverlayClient`` 推到硬件叠加层并烘进编码流；
  InferencePipeline 是客户端便捷层 —— 帧拉进应用、pre→infer→post 在
  应用线程同步执行，适合需要像素的场景（自定义预处理、同帧多模型链、
  自绘）。按"像素应该住哪"选择。
- **单发生命周期**：``start()`` 一次、``stop()`` 一次；再次运行须新建
  实例。``stop()`` 幂等、start 前调用为 no-op；先 join 工作线程再清
  框/清静态多边形，只关闭自己创建的客户端（注入的 ``inference=`` /
  ``overlay=`` 不关闭）。
- **过滤只影响绘制**：``min_score`` / ``labels`` 仅过滤画到叠加层的
  内容，``results()`` 仍返回完整结果；全被滤掉的结果发布空检测列表，
  以清除滞留旧框。
- **on_result 钩子**：在工作线程逐结果调用，返回（可替换的）结果继续
  流动，返回 ``None`` 丢弃；抛异常只丢该结果并记入
  ``status().last_error``，非致命。
- **两段有界队列、丢最旧**：subscribe 侧 ``queue_size``
  （``subscribe_dropped`` 透出）与应用侧 ``result_queue_size``
  （``result_queue_drops`` 计数）；叠加失败计入 ``annotate_errors``
  且不中断管线。
- **draw=False 完全不触碰叠加层**：静态 polygons 也不推，编码流保持
  干净；结果仍进 ``results()``。
- **会话标记（P2-13）**：``session_id=`` 贯穿推理与绘制 —— 订阅请求
  与结果绘制带同一标记；进程意外退出（含 SIGKILL）后，推理会话结束
  事件触发 daemon 把带该标记的多边形一并清掉。启动时的静态 zone 与
  ``stop()`` 的清屏写入不带标记，属于运维写入、无条件生效。
- **status() 透传订阅观测面**：last/avg latency_ms 与 last/avg skew_us
  （结果时间戳相对帧时间戳的偏移 —— 实时预览与严格对帧两种模式共用
  的验收指标）。

AI 叠加层 API
=============

.. automodule:: neoruntime_ipc_sdk.overlay
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

OverlayClient
-------------

.. autoclass:: neoruntime_ipc_sdk.OverlayClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

OverlayConfig
-------------

.. autoclass:: neoruntime_ipc_sdk.OverlayConfig
   :members:
   :undoc-members:

使用示例
--------

启用叠加层
~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import OverlayClient

   overlay = OverlayClient()

   # 启用硬件叠加层：在视频上绘制检测框 + 标签 + 置信度
   overlay.enable(show_label=True, show_confidence=True, line_thickness=2)

   # 关闭
   overlay.disable()

自定义样式
~~~~~~~~~~

.. code-block:: python

   # configure 一次设置全部样式参数
   overlay.configure(
       enabled=True,
       show_label=True,
       show_confidence=False,
       line_thickness=3,
       box_color=0x00FF00,     # 绿色框
       label_color=0xFFFFFF,   # 白色标签
       font_size=16,
   )

结构化配置
~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import OverlayClient, OverlayConfig

   config = OverlayConfig(
       enabled=True,
       show_label=True,
       show_confidence=True,
       line_thickness=2,
   )
   OverlayClient().apply(config)

与推理结果联动（annotate）
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # annotate() 把检测结果经事件总线推给 camera-daemon 的叠加渲染器：
   # app 不接触视频帧，检测框在编码前由 daemon 画到码流上。结果
   # 500ms 过期，按推理节奏持续调用；发空列表清屏
   from neoruntime_ipc_sdk import OverlayClient

   overlay = OverlayClient()
   overlay.enable()

   for result in inference_results:      # 如 InferenceClient.subscribe(...)
       overlay.annotate("main", result.objects)

   overlay.annotate("main", [])          # 清屏

其他结果类型
~~~~~~~~~~~~

.. code-block:: python

   # annotate_result 按 objects > classifications > landmarks >
   # ocr_lines 的优先级取 InferenceResult 中已填充的那一段
   overlay.annotate_result("main", result)

上下文管理器
~~~~~~~~~~~~

.. code-block:: python

   with OverlayClient() as overlay:
       overlay.enable()

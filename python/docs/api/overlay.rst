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

与推理结果联动
~~~~~~~~~~~~~~

.. code-block:: python

   # 叠加层由 daemon 在编码前渲染：app 只需发布推理事件，
   # daemon 自动把检测框画到码流上，无需 app 拿到视频帧
   from neoruntime_ipc_sdk import EventClient, OverlayClient

   events = EventClient()
   overlay = OverlayClient()
   overlay.enable()

   # 之后发布的检测事件会自动触发叠加渲染
   events.publish("detection/vehicles", {"objects": [...]})

上下文管理器
~~~~~~~~~~~~

.. code-block:: python

   with OverlayClient() as overlay:
       overlay.enable()

插件 API
========

.. automodule:: neoruntime_ipc_sdk.plugin
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

PluginDiscovery
---------------

.. autoclass:: neoruntime_ipc_sdk.PluginDiscovery
   :members:
   :undoc-members:

PluginEndpoint
--------------

.. autoclass:: neoruntime_ipc_sdk.PluginEndpoint
   :members:
   :undoc-members:

PluginServer
------------

.. autoclass:: neoruntime_ipc_sdk.PluginServer
   :members:
   :undoc-members:

使用示例
--------

App 侧：发现并调用插件
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import PluginDiscovery

   discovery = PluginDiscovery()  # 默认 /run/aipc/plugins

   # 列出可用能力
   for cap in discovery.list_capabilities():
       print(f"{cap}")

   # 取提供某能力的插件端点（可选）
   endpoint = discovery.get("lpr.postprocess")

   # 或阻塞等待插件上线（依赖方先于提供方启动的场景）
   endpoint = discovery.require("lpr.postprocess", timeout=30.0)

   # 连接到插件的 gRPC 服务
   channel = endpoint.connect()

   discovery.close()

监听插件上下线
~~~~~~~~~~~~~~

.. code-block:: python

   def on_change(event):
       print(f"plugin event: {event}")

   discovery.watch(on_change)

插件侧：发布能力
~~~~~~~~~~~~~~~~

.. code-block:: python

   import grpc
   from neoruntime_ipc_sdk import PluginServer

   server = PluginServer("lpr.postprocess")   # 在 /run/aipc/plugins 落盘发现文件

   grpc_server = server.create_server(max_workers=4)
   my_service.add_to_grpc_server(grpc_server)  # 注册你的 servicer
   grpc_server.start()

   server.start()        # 向外宣告能力可用
   grpc_server.wait_for_termination()
   server.stop()

热加载
~~~~~~

.. code-block:: python

   # 插件更新后重新扫描发现目录
   discovery.reload()

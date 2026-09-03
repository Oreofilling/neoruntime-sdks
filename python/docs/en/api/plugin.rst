Plugin API
==========

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

Usage Examples
--------------

App side: discover and call a plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import PluginDiscovery

   discovery = PluginDiscovery()  # defaults to /run/aipc/plugins

   # List available capabilities
   for cap in discovery.list_capabilities():
       print(f"{cap}")

   # Get an endpoint providing a capability (optional)
   endpoint = discovery.get("lpr.postprocess")

   # Or block until the plugin appears (consumer starts before provider)
   endpoint = discovery.require("lpr.postprocess", timeout=30.0)

   # Connect to the plugin's gRPC service
   channel = endpoint.connect()

   discovery.close()

Watch for plugin changes
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def on_change(event):
       print(f"plugin event: {event}")

   discovery.watch(on_change)

Plugin side: advertise a capability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import grpc
   from neoruntime_ipc_sdk import PluginServer

   server = PluginServer("lpr.postprocess")  # writes a discovery file
                                             # under /run/aipc/plugins

   grpc_server = server.create_server(max_workers=4)
   my_service.add_to_grpc_server(grpc_server)  # register your servicer
   grpc_server.start()

   server.start()        # announce the capability
   grpc_server.wait_for_termination()
   server.stop()

Hot reload
~~~~~~~~~~

.. code-block:: python

   # Rescan the discovery directory after plugin updates
   discovery.reload()

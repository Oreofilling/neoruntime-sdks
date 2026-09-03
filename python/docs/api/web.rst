Web 预览 API
============

.. automodule:: neoruntime_ipc_sdk.web
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

MjpegServer
-----------

.. autoclass:: neoruntime_ipc_sdk.MjpegServer
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

MjpegStream
-----------

.. autoclass:: neoruntime_ipc_sdk.MjpegStream
   :members:
   :undoc-members:

mjpeg_wsgi_app
--------------

.. autofunction:: neoruntime_ipc_sdk.mjpeg_wsgi_app

使用示例
--------

独立 MJPEG 预览服务器
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import FdMediaClient, MjpegServer, MjpegStream

   media = FdMediaClient()
   stream = MjpegStream()

   # 独立线程 HTTP 服务器，浏览器打开 http://<ip>:8080/ 即可预览
   server = MjpegServer(port=8080, source=stream, fps=15)
   server.start()

   for frame in media.subscribe("main"):
       stream.push_frame(frame, quality=85)

   server.stop()

推入已有 JPEG
~~~~~~~~~~~~~

.. code-block:: python

   # 应用侧已编码好的 JPEG 直接推送
   import cv2

   ok, jpeg = cv2.imencode(".jpg", annotated)
   if ok:
       stream.push_jpeg(jpeg.tobytes())

配合 WSGI 应用（Flask / 内置服务器）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from wsgiref.simple_server import make_server
   from neoruntime_ipc_sdk import MjpegStream, mjpeg_wsgi_app

   stream = MjpegStream()
   app = mjpeg_wsgi_app(stream, fps=15)

   make_server("0.0.0.0", 8080, app).serve_forever()

低延迟读取最新帧
~~~~~~~~~~~~~~~~

.. code-block:: python

   # latest/wait_new 适合自实现推送逻辑
   seq = stream.latest_seq()
   data = stream.latest()          # bytes | None
   newer = stream.wait_new(seq, timeout=1.0)
   if newer is not None:
       send_to_client(newer)

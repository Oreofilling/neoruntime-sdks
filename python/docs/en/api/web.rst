Web Preview API
===============

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

Usage Examples
--------------

Standalone MJPEG preview server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import FdMediaClient, MjpegServer, MjpegStream

   media = FdMediaClient()
   stream = MjpegStream()

   # Threaded HTTP server; open http://<ip>:8080/ in a browser
   server = MjpegServer(port=8080, source=stream, fps=15)
   server.start()

   for frame in media.subscribe("main"):
       stream.push_frame(frame, quality=85)

   server.stop()

Push an existing JPEG
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Push JPEGs the app has already encoded
   import cv2

   ok, jpeg = cv2.imencode(".jpg", annotated)
   if ok:
       stream.push_jpeg(jpeg.tobytes())

Inside a WSGI app (Flask / stdlib)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from wsgiref.simple_server import make_server
   from neoruntime_ipc_sdk import MjpegStream, mjpeg_wsgi_app

   stream = MjpegStream()
   app = mjpeg_wsgi_app(stream, fps=15)

   make_server("0.0.0.0", 8080, app).serve_forever()

Low-latency latest-frame reads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # latest / wait_new suit custom push logic
   seq = stream.latest_seq()
   data = stream.latest()          # bytes | None
   newer = stream.wait_new(seq, timeout=1.0)
   if newer is not None:
       send_to_client(newer)

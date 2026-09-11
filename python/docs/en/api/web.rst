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

platform_stream_url
-------------------

.. autofunction:: neoruntime_ipc_sdk.platform_stream_url

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

Zero-server preview (reuse the platform stream URL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # After the app injects content into a platform stream via
   # FramePublisher (REPLACE or OVERLAY), it needs no server of its
   # own: the platform gateway already reverse-proxies the encoded
   # stream at /api/v1/h264/{stream_id}, and the web console's player
   # consumes that very URL.
   import os
   from neoruntime_ipc_sdk import FramePublisher, platform_stream_url

   with FramePublisher(camera, dsp, stream_id="sub") as pub:
       pub.publish(frame)
       # host defaults to the AIPC_WEB_HOST env var, then localhost
       url = platform_stream_url("sub", host="192.168.1.10", token=jwt)
       # 'wss://192.168.1.10/api/v1/h264/sub?token=...'
       pub.publish_eos()   # back to the pure ISP path (next IDR)

   # token is required: /api/v1 over the network is authed (the same
   # JWT the console uses). Full origins work as host - an http(s)://
   # prefix maps to ws(s).

Video Stream API
================

.. automodule:: neoruntime_ipc_sdk.media
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

FdMediaClient
-------------

.. autoclass:: neoruntime_ipc_sdk.FdMediaClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Data Types
----------

Frame
~~~~~

.. autoclass:: neoruntime_ipc_sdk.Frame
   :members:
   :undoc-members:
   :no-index:

StreamInfo
~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.StreamInfo
   :members:
   :undoc-members:

PixelFormat
~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.PixelFormat
   :members:
   :undoc-members:

EncodedStreamClient
~~~~~~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.EncodedStreamClient
   :members:
   :undoc-members:

EncodedFrame
~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.EncodedFrame
   :members:
   :undoc-members:

Usage Examples
--------------

Getting Raw Video Stream
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import FdMediaClient
   import cv2

   media = FdMediaClient()

   # Get main stream (available IDs come from list_streams(), usually main / sub)
   for frame in media.subscribe("main"):
       print(f"Frame {frame.sequence}: {frame.width}x{frame.height}")
       print(f"Format: {frame.format}, Timestamp: {frame.timestamp_ns}")

       # frame.image is a numpy array (H, W, C) or (H*3//2, W) for NV12
       # Can be used directly with OpenCV or other image processing libraries
       cv2.imshow("Camera", frame.image)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cv2.destroyAllWindows()

No Frame Skipping
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get every frame (no skipping)
   for frame in media.subscribe("main", skip_frames=False):
       process_frame(frame.image)

Getting a Single Frame
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get a single frame
   frame = media.get_frame("main", timeout_ms=1000)

   if frame:
       print(f"Frame: {frame.width}x{frame.height}")
       print(f"Format: {frame.format}")

Getting Stream Info
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # FdMediaClient has no separate stream-info API;
   # read dimensions and format from the first frame
   frame = media.get_frame("main", timeout_ms=5000)

   if frame:
       print(f"Resolution: {frame.width}x{frame.height}")
       print(f"Format: {frame.format}")
       print(f"Available streams: {media.list_streams()}")

Listing Available Streams
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # List all available streams
   streams = media.list_streams()
   for stream_id in streams:
       print(f"Stream: {stream_id}")

Frame Callback
~~~~~~~~~~~~~~

.. code-block:: python

   def handle_frame(frame):
       print(f"Received frame: {frame.sequence}")

   # Use callback for frame processing
   thread = media.on_frame("main", handle_frame)

   # Keep running
   import time
   while True:
       time.sleep(1)

Image Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   import numpy as np

   media = FdMediaClient()

   for frame in media.subscribe("main"):
       # Convert to RGB
       rgb = frame.to_rgb()

       # Grayscale
       gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

       # Edge detection
       edges = cv2.Canny(gray, 100, 200)

       # Display results
       cv2.imshow("Original", rgb)
       cv2.imshow("Edges", edges)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

Saving Images
~~~~~~~~~~~~~

.. code-block:: python

   media = FdMediaClient()

   for frame in media.subscribe("main"):
       # Save as image
       frame.save("frame.jpg")
       break

Getting Encoded Stream
~~~~~~~~~~~~~~~~~~~~~~

``get_encoded_stream()`` returns an :class:`EncodedStreamClient` (not an iterator) yielding H.264/H.265 Annex-B packets, which the ``recording`` module can write to disk directly:

.. code-block:: python

   from neoruntime_ipc_sdk import FdMediaClient

   media = FdMediaClient()

   # get_encoded_stream() returns an EncodedStreamClient, not an iterator
   client = media.get_encoded_stream("main")

   for packet in client.subscribe():
       print(f"{packet.codec_name()} packet: {len(packet.data)} bytes")
       if packet.is_keyframe():
           print("  keyframe")

Multi-Stream Processing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import threading

   def process_main_stream():
       media = FdMediaClient()
       for frame in media.subscribe("main"):
           # Process main stream (high resolution)
           process_high_res(frame.image)

   def process_sub_stream():
       media = FdMediaClient()
       for frame in media.subscribe("sub"):
           # Process sub stream (low resolution)
           process_low_res(frame.image)

   # Process multiple streams in parallel
   t1 = threading.Thread(target=process_main_stream)
   t2 = threading.Thread(target=process_sub_stream)

   t1.start()
   t2.start()

   t1.join()
   t2.join()

Frame Rate Control
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time

   media = FdMediaClient()
   target_fps = 10
   frame_interval = 1.0 / target_fps

   last_time = time.time()

   for frame in media.subscribe("main"):
       current_time = time.time()
       elapsed = current_time - last_time

       if elapsed >= frame_interval:
           # Process frame
           process_frame(frame.image)
           last_time = current_time

Saving Video
~~~~~~~~~~~~

To save the H.264/H.265 encoded stream, prefer reusing it directly with the ``recording`` module (``TsWriter`` / ``HlsWriter``) — no decode/re-encode needed. The example below decodes frames and saves them with OpenCV:

.. code-block:: python

   import cv2

   media = FdMediaClient()

   # Read the resolution from the first frame
   # (FdMediaClient has no separate stream-info API)
   first = media.get_frame("main", timeout_ms=5000)
   if first is None:
       raise RuntimeError("no frame received")

   # Create video writer (fill in the fps of your actual stream, 30 here)
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter(
       'output.mp4',
       fourcc,
       30.0,
       (first.width, first.height)
   )

   frame_count = 0
   max_frames = 300  # Record 10 seconds (30fps)

   for frame in media.subscribe("main"):
       rgb = frame.to_rgb()
       bgr = rgb[:, :, ::-1]  # RGB to BGR
       out.write(bgr)
       frame_count += 1

       if frame_count >= max_frames:
           break

   out.release()
   print(f"Saved {frame_count} frames to output.mp4")

Context Manager
~~~~~~~~~~~~~~~

.. code-block:: python

   # Use context manager for automatic resource management
   with FdMediaClient() as media:
       for frame in media.subscribe("main"):
           process_frame(frame.image)

Zero-Copy Access
~~~~~~~~~~~~~~~~

.. code-block:: python

   # The default receive path copies pixel data on arrival
   # (frame.image is immediately usable). keep_fd=True retains the
   # dma-buf fds for true zero-copy: the buffer is only returned to
   # the daemon on frame.release() / GC / client close

   media = FdMediaClient()

   for frame in media.subscribe("main", keep_fd=True):
       # frame.handle holds the dma-buf fds
       # frame.to_array() maps them once on first call and caches
       result = inference_engine.process(frame.to_array())

       # Return the buffer as early as possible (idempotent, optional)
       frame.release()

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   media = FdMediaClient()

   try:
       for frame in media.subscribe("invalid_stream"):
           process_frame(frame.image)
   except Exception as e:
       print(f"Stream access failed: {e}")
   except KeyboardInterrupt:
       print("User interrupted")
   finally:
       media.close()

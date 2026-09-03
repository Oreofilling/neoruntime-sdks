Recording API
=============

.. automodule:: neoruntime_ipc_sdk.recording
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

TsWriter
--------

.. autoclass:: neoruntime_ipc_sdk.TsWriter
   :members:
   :undoc-members:

HlsWriter
---------

.. autoclass:: neoruntime_ipc_sdk.HlsWriter
   :members:
   :undoc-members:

PrerollBuffer
-------------

.. autoclass:: neoruntime_ipc_sdk.PrerollBuffer
   :members:
   :undoc-members:

Usage Examples
--------------

Record MP4 (TsWriter)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import EncodedMediaClient, TsWriter

   media = EncodedMediaClient()
   writer = TsWriter("/data/aipc/recordings/clip.mp4", codec="h264")

   try:
       # Subscribe to the encoded stream and write frames by PTS
       for frame in media.subscribe("main"):
           writer.write(frame)
           if enough_recorded(frame):
               break
   finally:
       writer.close()  # must close to finalize the moov/index

HLS live segmentation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import EncodedMediaClient, HlsWriter

   media = EncodedMediaClient()

   # One 6s segment, playlist keeps the latest 5 segments
   hls = HlsWriter("/data/aipc/recordings/live", segment_seconds=6.0, window=5)

   try:
       for frame in media.subscribe("main"):
           hls.write(frame)
   finally:
       hls.close()

Preroll recording (pre-event cache)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import EncodedMediaClient, PrerollBuffer

   media = EncodedMediaClient()

   # Ring buffer holding the last 10s of encoded frames
   preroll = PrerollBuffer(seconds=10.0)

   for frame in media.subscribe("main"):
       preroll.push(frame)
       if event_detected():
           # dump: preroll contents + subsequent frames -> ts file
           writer = preroll.dump("/data/aipc/recordings/alert.ts")
           # ... keep appending post-event frames via writer.write(frame)
           writer.close()
           break

Custom frame callback
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # dump's on_frame runs before each frame is written — use it to
   # filter or modify frames
   def mark(frame):
       print(f"writing pts={frame.pts_ns}")
       return frame

   writer = preroll.dump("/data/aipc/recordings/alert.ts", on_frame=mark)

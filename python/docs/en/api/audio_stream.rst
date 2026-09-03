Audio Streaming API
===================

.. automodule:: neoruntime_ipc_sdk.audio_stream
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

AudioStreamClient
-----------------

.. autoclass:: neoruntime_ipc_sdk.AudioStreamClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

AudioFrame
----------

.. autoclass:: neoruntime_ipc_sdk.AudioFrame
   :members:
   :undoc-members:

Usage Examples
--------------

Iterate audio frames
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import AudioClient, AudioStreamClient

   # Start capture first, then subscribe
   audio = AudioClient()
   audio.start_capture(sample_rate=16000, channels=1, codec="aac")

   stream = AudioStreamClient()
   for frame in stream.subscribe():
       print(f"{frame.codec_name} frame: {len(frame.data)} bytes, "
             f"{frame.duration_ms:.1f}ms")

Fetch a single frame
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   frame = stream.get_frame(timeout_ms=2000)
   if frame is not None:
       print(f"codec={frame.codec_name}, pts={frame.pts_ns}")

Callback subscription
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def on_audio(frame):
       # Process the audio frame (forward, persist, feed to ASR, ...)
       save(frame.data)

   thread = stream.on_frame(on_audio)

Raw PCM vs encoded frames
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   for frame in stream.subscribe():
       if frame.codec == 0:  # PCM
           # frame.data can be parsed directly using
           # sample_rate / channels / bits_per_sample
           process_pcm(frame.data)
       else:  # aac / g711a / g711u
           process_encoded(frame.data, frame.is_keyframe)

Error handling and cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   stream = AudioStreamClient()
   try:
       for frame in stream.subscribe(reconnect=True):
           process(frame)
   finally:
       stream.close()

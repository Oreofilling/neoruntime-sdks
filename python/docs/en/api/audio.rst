Audio Control API
=================

.. automodule:: neoruntime_ipc_sdk.audio
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

AudioClient
-----------

.. autoclass:: neoruntime_ipc_sdk.AudioClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Data Types
----------

AudioDevice
~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.AudioDevice
   :members:
   :undoc-members:

AudioStatus
~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.AudioStatus
   :members:
   :undoc-members:

Usage Examples
--------------

Query devices and status
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import AudioClient

   audio = AudioClient()

   # List capture / playback devices
   for dev in audio.list_capture_devices():
       print(f"capture: {dev.name} ({dev.description})")
   for dev in audio.list_playback_devices():
       print(f"playback: {dev.name} ({dev.description})")

   # Current runtime status
   status = audio.get_status()
   print(f"capturing: {status.capturing}, playing: {status.playing}, "
         f"{status.sample_rate}Hz {status.channels}ch {status.codec}")

Start / stop capture
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Start capture with explicit parameters (0/empty keeps defaults)
   audio.start_capture(sample_rate=16000, channels=1, codec="aac")

   # Subscribe to audio frames via AudioStreamClient
   # (see the audio streaming API)...

   audio.stop_capture()

Play a file (intercom playback)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Blocking playback of a PCM file
   audio.start_playback(sample_rate=48000, channels=1)
   audio.stream_pcm_file(
       "/app/audio/notice.pcm",
       sample_rate=48000, channels=1, fmt="S16LE",
   )
   audio.stop_playback()

Stream PCM (live intercom)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # pcm_iter yields PCM chunks; playback proceeds while chunks arrive
   def pcm_chunks():
       for chunk in network_receive():
           yield chunk

   audio.stream_pcm(pcm_chunks(), sample_rate=16000, channels=1)

Volume and mute
~~~~~~~~~~~~~~~

.. code-block:: python

   audio.set_config(volume=0.8, mute=False)

Context manager
~~~~~~~~~~~~~~~

.. code-block:: python

   with AudioClient() as audio:
       audio.get_status()

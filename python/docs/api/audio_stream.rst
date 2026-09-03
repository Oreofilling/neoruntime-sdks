音频流 API
==========

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

使用示例
--------

迭代订阅音频帧
~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import AudioClient, AudioStreamClient

   # 先启动采集，再订阅
   audio = AudioClient()
   audio.start_capture(sample_rate=16000, channels=1, codec="aac")

   stream = AudioStreamClient()
   for frame in stream.subscribe():
       print(f"{frame.codec_name} 帧: {len(frame.data)} 字节, "
             f"{frame.duration_ms:.1f}ms")

获取单帧
~~~~~~~~

.. code-block:: python

   frame = stream.get_frame(timeout_ms=2000)
   if frame is not None:
       print(f"codec={frame.codec_name}, pts={frame.pts_ns}")

回调订阅
~~~~~~~~

.. code-block:: python

   def on_audio(frame):
       # 处理音频帧（如转发、写文件、送 ASR）
       save(frame.data)

   thread = stream.on_frame(on_audio)

原始 PCM 与编码帧
~~~~~~~~~~~~~~~~~

.. code-block:: python

   for frame in stream.subscribe():
       if frame.codec == 0:  # PCM
           # frame.data 可直接按 sample_rate/channels/bits_per_sample 解析
           process_pcm(frame.data)
       else:  # aac / g711a / g711u
           process_encoded(frame.data, frame.is_keyframe)

错误处理与资源释放
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   stream = AudioStreamClient()
   try:
       for frame in stream.subscribe(reconnect=True):
           process(frame)
   finally:
       stream.close()

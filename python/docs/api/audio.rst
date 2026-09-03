音频控制 API
============

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

数据类型
--------

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

使用示例
--------

查询设备与状态
~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import AudioClient

   audio = AudioClient()

   # 列出采集/播放设备
   for dev in audio.list_capture_devices():
       print(f"采集: {dev.name} ({dev.description})")
   for dev in audio.list_playback_devices():
       print(f"播放: {dev.name} ({dev.description})")

   # 当前运行状态
   status = audio.get_status()
   print(f"采集: {status.capturing}, 播放: {status.playing}, "
         f"{status.sample_rate}Hz {status.channels}ch {status.codec}")

开始/停止采集
~~~~~~~~~~~~~

.. code-block:: python

   # 指定参数启动采集（0/空串表示沿用默认）
   audio.start_capture(sample_rate=16000, channels=1, codec="aac")

   # 配合 AudioStreamClient 订阅音频帧（见音频流 API）...

   audio.stop_capture()

播放文件（对讲放音）
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 阻塞播放一个 PCM 文件
   audio.start_playback(sample_rate=48000, channels=1)
   audio.stream_pcm_file(
       "/app/audio/notice.pcm",
       sample_rate=48000, channels=1, fmt="S16LE",
   )
   audio.stop_playback()

流式播放 PCM（实时对讲）
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # pcm_iter 逐块产出 PCM 字节，边收边播
   def pcm_chunks():
       for chunk in network_receive():
           yield chunk

   audio.stream_pcm(pcm_chunks(), sample_rate=16000, channels=1)

音量与静音
~~~~~~~~~~

.. code-block:: python

   audio.set_config(volume=0.8, mute=False)

上下文管理器
~~~~~~~~~~~~

.. code-block:: python

   with AudioClient() as audio:
       audio.get_status()

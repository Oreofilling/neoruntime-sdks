帧注入 API（输出写路径）
========================

.. automodule:: neoruntime_ipc_sdk.injection
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

FramePublisher
--------------

.. autoclass:: neoruntime_ipc_sdk.FramePublisher
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

InjectionResult
---------------

.. autoclass:: neoruntime_ipc_sdk.InjectionResult
   :members:
   :undoc-members:

InjectionStatus
---------------

.. autoclass:: neoruntime_ipc_sdk.InjectionStatus
   :members:
   :undoc-members:

CameraClient 注入 RPC
---------------------

.. autoclass:: neoruntime_ipc_sdk.CameraClient
   :members: push_frame, injection_status, stop_injection

使用示例
--------

把处理后的 NV12 帧发布到子码流编码器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient, DspClient, FramePublisher

   cam = CameraClient()
   dsp = DspClient()

   # 构造时从 GetStreamStatus 解析子码流编码分辨率并锁定几何；
   # 之后 publish 只接受该分辨率的紧凑 NV12（ndarray 或 bytes）
   with FramePublisher(cam, dsp, stream_id="sub") as pub:
       for frame in processed_frames:      # uint8 (h*3//2, w)
           pub.publish(frame)

       pub.publish_eos()   # 结束注入会话；下个 IDR 起恢复纯 ISP 画面

语义要点
~~~~~~~~

- **两种模式**：``mode="replace"``（P0）注入帧整体替换编码输入；
  ``mode="overlay"``（P2）把小画面合入实况画面（``fmt="nv12"`` 为不
  透明贴块，``fmt="argb"`` 为 CPU alpha 混合，几何由 ``inset``/``dest``
  决定）。
- **帧跨进程以 DSP 注册 id 传递**，不是裸 fd —— 与取帧（FrameHandle）、
  零拷贝推理（``infer(frame)``）共用同一套缓冲命名空间。
- **daemon 侧队列 cap 3、满则丢最旧**：publish 永不反压编码器，
  溢出体现在 ``injection_status().frames_dropped``。
- **结束即恢复**：``publish_eos()``（或 ``stop_injection()``）后，
  纯 ISP 画面从下个 IDR 起恢复，无需重启码流。
- **生命周期标记（P2-13）**：``session_id=`` 给注入会话打标记——
  首个带标记的帧打开会话，标记可在 ``injection_status().session_id``
  观测。回收以连接为键、不以标记为键：注入方断连（含进程被 SIGKILL）
  时 daemon 立即关闭会话并释放该客户端的全部缓冲，无论标记为何。
- ``close()`` 只释放缓冲池、不发 EOS —— 结束会话必须显式调
  ``publish_eos()``。

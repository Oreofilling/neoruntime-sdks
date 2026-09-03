相机控制 API
============

.. automodule:: neoruntime_ipc_sdk.camera
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

CameraClient
------------

.. autoclass:: neoruntime_ipc_sdk.CameraClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

数据类型
--------

ISPConfig
~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.ISPConfig
   :members:
   :undoc-members:

TransformConfig
~~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.TransformConfig
   :members:
   :undoc-members:

Capabilities
~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.Capabilities
   :members:
   :undoc-members:

SensorInfo
~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.SensorInfo
   :members:
   :undoc-members:

StreamStatus
~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.StreamStatus
   :members:
   :undoc-members:

HardwareStatus
~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.HardwareStatus
   :members:
   :undoc-members:

EnvStatus
~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.EnvStatus
   :members:
   :undoc-members:

PipelineStreamConfig
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.PipelineStreamConfig
   :members:
   :undoc-members:

EncoderReconfigResult
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.EncoderReconfigResult
   :members:
   :undoc-members:

InfraredStatus
~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.InfraredStatus
   :members:
   :undoc-members:

IrPreset
~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.IrPreset
   :members:
   :undoc-members:

PrivacyMaskSettings
~~~~~~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.PrivacyMaskSettings
   :members:
   :undoc-members:

使用示例
--------

ISP 图像调节
~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient, ISPConfig

   cam = CameraClient()

   # 读取当前 ISP 配置
   cfg = cam.get_isp()
   print(f"亮度: {cfg.brightness}, 对比度: {cfg.contrast}")

   # 仅修改关心的字段（-1 表示保持不变）
   cam.set_isp(ISPConfig(brightness=60, saturation=50))

   # 或使用关键字参数
   cam.set_isp(brightness=60, sharpness=128)

图像变换（翻转 / 旋转 / 灰度）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient, TransformConfig

   cam = CameraClient()

   # 水平翻转 + 180° 旋转 + 灰度输出
   cam.set_transform(TransformConfig(rotation=180, flip=1, grayscale=True))

红外与夜视
~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient

   cam = CameraClient()

   # 查看当前日夜成像状态
   status = cam.get_infrared_status()
   print(status)

   # 切换成像模式（"auto" / "day" / "night" 等，见 daemon 支持）
   cam.set_imaging_mode("night")

   # 手动微调红外灯亮度（PWM）
   cam.set_infrared_settings(near_pwm=80, far_pwm=120)

   # 按变焦倍率保存/调用 IR 预设
   cam.save_ir_preset("zoom_2x", zoom_ratio=2.0, near_pwm=60, far_pwm=90)
   presets = cam.list_ir_presets()
   cam.delete_ir_preset("zoom_2x")

编码器与码流
~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient, PipelineStreamConfig

   cam = CameraClient()

   # 查看码流状态
   for s in cam.get_stream_status():
       print(f"{s.stream_id}: {s.width}x{s.height}@{s.fps} {s.codec}")

   # 运行时调整单条码流码率（不重启管线）
   cam.set_encoder("main", bitrate_bps=6_000_000, gop=60)

   # 结构化重配（宽高/编码器变化，返回中断时长）
   result = cam.reconfigure_encoder(
       "main", width=2560, height=1440, bitrate_bps=8_000_000,
   )
   print(f"success={result.success}, 中断 {result.interrupt_ms}ms")

   # 增删码流
   cam.add_stream("sub2", 704, 576, fps=15, codec="h265", bitrate=1_500_000)
   cam.remove_stream("sub2")

隐私遮挡
~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient

   cam = CameraClient()

   # 启用静态遮挡区域（归一化坐标 0-1）
   cam.set_privacy_mask(
       enabled=True,
       color=0,
       blur_radius=21,
       regions=[
           {"id": 1, "x": 0.05, "y": 0.05, "w": 0.2, "h": 0.15},
           {"id": 2, "x": 0.7, "y": 0.7, "w": 0.25, "h": 0.2},
       ],
   )

   # 查询当前配置
   print(cam.get_privacy_mask())

环境与外设
~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient

   cam = CameraClient()

   # 光感/温度等硬件状态
   hw = cam.get_hardware_status()
   print(f"光照 {hw.light_sensor_lux} lux, MCU {hw.mcu_temp_millic/1000:.1f}°C")

   # 风扇 / 加热 / 雷达
   cam.set_fan(True)
   cam.set_heat(False)
   print(cam.get_radar())

   # 报警输出
   cam.set_alarm_out(0, True)
   print(cam.get_alarm_outputs())

   # RS485 透传
   cam.rs485_init(baudrate=9600, config="8N1")
   cam.rs485_tx(b"\x01\x02\x03")

能力查询
~~~~~~~~

.. code-block:: python

   caps = cam.get_capabilities()
   print(f"LED: {caps.has_led}, RS485: {caps.has_rs485}, 音频: {caps.has_audio}")

上下文管理器
~~~~~~~~~~~~

.. code-block:: python

   with CameraClient() as cam:
       print(cam.get_sensor_info())

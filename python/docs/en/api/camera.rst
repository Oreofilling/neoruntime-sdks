Camera Control API
==================

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

Data Types
----------

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

Usage Examples
--------------

ISP image adjustment
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient, ISPConfig

   cam = CameraClient()

   # Read the current ISP configuration
   cfg = cam.get_isp()
   print(f"brightness: {cfg.brightness}, contrast: {cfg.contrast}")

   # Only set the fields you care about (-1 means "keep current")
   cam.set_isp(ISPConfig(brightness=60, saturation=50))

   # Or use keyword arguments
   cam.set_isp(brightness=60, sharpness=128)

Image transform (flip / rotate / grayscale)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient, TransformConfig

   cam = CameraClient()

   # Horizontal flip + 180° rotation + grayscale output
   cam.set_transform(TransformConfig(rotation=180, flip=1, grayscale=True))

Infrared and night vision
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient

   cam = CameraClient()

   # Current day/night imaging status
   status = cam.get_infrared_status()
   print(status)

   # Switch imaging mode ("auto" / "day" / "night", see daemon support)
   cam.set_imaging_mode("night")

   # Fine-tune IR LED brightness (PWM) manually
   cam.set_infrared_settings(near_pwm=80, far_pwm=120)

   # Save / list / delete IR presets by zoom ratio
   cam.save_ir_preset("zoom_2x", zoom_ratio=2.0, near_pwm=60, far_pwm=90)
   presets = cam.list_ir_presets()
   cam.delete_ir_preset("zoom_2x")

Encoders and streams
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient, PipelineStreamConfig

   cam = CameraClient()

   # Stream status
   for s in cam.get_stream_status():
       print(f"{s.stream_id}: {s.width}x{s.height}@{s.fps} {s.codec}")

   # Adjust a stream's bitrate at runtime (no pipeline restart)
   cam.set_encoder("main", bitrate_bps=6_000_000, gop=60)

   # Structured reconfiguration (resolution/codec change; returns the
   # interruption duration)
   result = cam.reconfigure_encoder(
       "main", width=2560, height=1440, bitrate_bps=8_000_000,
   )
   print(f"success={result.success}, interrupted {result.interrupt_ms}ms")

   # Add / remove streams
   cam.add_stream("sub2", 704, 576, fps=15, codec="h265", bitrate=1_500_000)
   cam.remove_stream("sub2")

Privacy mask
~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient

   cam = CameraClient()

   # Enable static mask regions (normalized 0-1 coordinates)
   cam.set_privacy_mask(
       enabled=True,
       color=0,
       blur_radius=21,
       regions=[
           {"id": 1, "x": 0.05, "y": 0.05, "w": 0.2, "h": 0.15},
           {"id": 2, "x": 0.7, "y": 0.7, "w": 0.25, "h": 0.2},
       ],
   )

   # Query the current settings
   print(cam.get_privacy_mask())

Environment and peripherals
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient

   cam = CameraClient()

   # Light sensor / temperature and other hardware status
   hw = cam.get_hardware_status()
   print(f"{hw.light_sensor_lux} lux, MCU {hw.mcu_temp_millic/1000:.1f}°C")

   # Fan / heater / radar
   cam.set_fan(True)
   cam.set_heat(False)
   print(cam.get_radar())

   # Alarm outputs
   cam.set_alarm_out(0, True)
   print(cam.get_alarm_outputs())

   # RS485 passthrough
   cam.rs485_init(baudrate=9600, config="8N1")
   cam.rs485_tx(b"\x01\x02\x03")

Capability query
~~~~~~~~~~~~~~~~

.. code-block:: python

   caps = cam.get_capabilities()
   print(f"LED: {caps.has_led}, RS485: {caps.has_rs485}, audio: {caps.has_audio}")

Context manager
~~~~~~~~~~~~~~~

.. code-block:: python

   with CameraClient() as cam:
       print(cam.get_sensor_info())

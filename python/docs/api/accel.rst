加速路由 API
============

.. automodule:: neoruntime_ipc_sdk.accel
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

类与函数
--------

AccelRouter
~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.AccelRouter
   :members:
   :undoc-members:
   :no-index:

RoutePolicy
~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.RoutePolicy
   :members:
   :undoc-members:
   :no-index:

RouteDecision / DegradationRecord
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.RouteDecision
   :members:
   :undoc-members:

.. autoclass:: neoruntime_ipc_sdk.DegradationRecord
   :members:
   :undoc-members:

get_default_router
~~~~~~~~~~~~~~~~~~

.. autofunction:: neoruntime_ipc_sdk.accel.get_default_router
   :no-index:

使用示例
--------

默认路由器：硬件优先，自动降级
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import get_default_router

   router = get_default_router()
   small = router.run("resize_nv12", nv12, (1920, 1080), (640, 384))

   # DSP 不可用时自动回落 numpy/cv2，并记录一次降级
   health = router.health()
   print(health["ops"]["resize_nv12"]["backend"])
   print(health["recent_degradations"])

颜色转换同样走硬件腿
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import get_default_router

   router = get_default_router()
   nv12 = router.run("rgb_to_nv12", rgb)               # DSP convert_hw
   rgb2 = router.run("nv12_to_rgb", nv12, 1920, 1080)  # 反方向同腿
   # 路由器以 cpu_fallback=False 调用 DSP——真降级时只记一次，
   # 软件腿恰好执行一遍（不会先在客户端内部悄悄跑一遍 CPU）。

降级事件转发到事件总线
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import EventClient, get_default_router

   bus = EventClient()
   router = get_default_router()

   def report(record):
       bus.publish("app/health/degradation", {
           "op": record.op,
           "reason": record.reason,
       })

   router.on_degradation = report

只信硬件：失败即抛错
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import AccelRouter, RoutePolicy

   # HARDWARE_ONLY 路由器不降级——硬件失败直接抛 HardwareUnavailable，
   # 适合零 CPU 占用比可用性更重要的调用点
   strict = AccelRouter(policy=RoutePolicy.HARDWARE_ONLY)
   strict.register("resize_nv12", hardware=my_dsp_resize)

能力探测
~~~~~~~~

.. code-block:: python

   print(get_default_router().probe())
   # {'cv2': True, 'dsp': False}  ← dsp=False 表示服务当前不可达

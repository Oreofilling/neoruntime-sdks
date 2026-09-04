Accel Routing API
=================

.. automodule:: neoruntime_ipc_sdk.accel
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Classes and Functions
---------------------

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

Examples
--------

Default router: hardware first, automatic fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import get_default_router

   router = get_default_router()
   small = router.run("resize_nv12", nv12, (1920, 1080), (640, 384))

   # Falls back to numpy/cv2 when the DSP is unreachable and records
   # the degradation
   health = router.health()
   print(health["ops"]["resize_nv12"]["backend"])
   print(health["recent_degradations"])

Forward degradations to the event bus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Hardware-only: fail instead of degrading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import AccelRouter, RoutePolicy

   # A HARDWARE_ONLY router never falls back — a hardware failure
   # raises HardwareUnavailable. For call sites where the zero-CPU
   # guarantee matters more than uptime.
   strict = AccelRouter(policy=RoutePolicy.HARDWARE_ONLY)
   strict.register("resize_nv12", hardware=my_dsp_resize)

Capability probes
~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(get_default_router().probe())
   # {'cv2': True, 'dsp': False}  <- dsp=False: service unreachable now

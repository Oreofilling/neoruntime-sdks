Frame Injection API (Output Write Path)
=======================================

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

CameraClient injection RPCs
---------------------------

.. autoclass:: neoruntime_ipc_sdk.CameraClient
   :members: push_frame, injection_status, stop_injection

Examples
--------

Publishing processed NV12 frames to the sub-stream encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import CameraClient, DspClient, FramePublisher

   cam = CameraClient()
   dsp = DspClient()

   # The constructor resolves the sub-stream encode resolution from
   # GetStreamStatus and pins the geometry; afterwards publish() only
   # accepts tight NV12 at that resolution (ndarray or bytes)
   with FramePublisher(cam, dsp, stream_id="sub") as pub:
       for frame in processed_frames:      # uint8 (h*3//2, w)
           pub.publish(frame)

       pub.publish_eos()   # end the session; pure ISP restores at next IDR

Semantics
~~~~~~~~~

- **Two modes**: ``mode="replace"`` (P0) — an injected frame wholly
  replaces the encoder input; ``mode="overlay"`` (P2) — composites a
  small picture into the live image (``fmt="nv12"`` pastes opaquely,
  ``fmt="argb"`` alpha-blends on the CPU; geometry comes from
  ``inset``/``dest``).
- **Frames cross processes as DSP-registry ids**, never raw fds — the
  same buffer namespace shared by frame capture (FrameHandle) and
  zero-copy inference (``infer(frame)``).
- **Daemon-side queue caps at 3, drop-oldest on overflow**: publish
  never backpressures the encoder; overflow surfaces as
  ``injection_status().frames_dropped``.
- **End = restore**: after ``publish_eos()`` (or ``stop_injection()``)
  the pure ISP path restores at the next IDR — no stream restart.
- **Lifecycle tag (P2-13)**: ``session_id=`` tags the injection
  session — the first tagged frame opens it, and the tag is observable
  at ``injection_status().session_id``. Reclaim is keyed on the
  connection, not the tag: when the injecting client disconnects
  (including a SIGKILLed process), camera-daemon closes the session
  and frees every buffer that client owned, whatever the tag says.
- ``close()`` only releases the buffer pool; it does not send EOS —
  ending the session requires an explicit ``publish_eos()``.

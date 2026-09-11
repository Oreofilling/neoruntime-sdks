AI Overlay API
==============

.. automodule:: neoruntime_ipc_sdk.overlay
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

OverlayClient
-------------

.. autoclass:: neoruntime_ipc_sdk.OverlayClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

OverlayConfig
-------------

.. autoclass:: neoruntime_ipc_sdk.OverlayConfig
   :members:
   :undoc-members:

Usage Examples
--------------

Enable the overlay
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import OverlayClient

   overlay = OverlayClient()

   # Enable the hardware overlay: draw detection boxes + labels +
   # confidence on the video.
   overlay.enable(show_label=True, show_confidence=True, line_thickness=2)

   # Disable
   overlay.disable()

Custom styling
~~~~~~~~~~~~~~

.. code-block:: python

   # configure sets every style parameter in one call
   overlay.configure(
       enabled=True,
       show_label=True,
       show_confidence=False,
       line_thickness=3,
       box_color=0x00FF00,     # green boxes
       label_color=0xFFFFFF,   # white labels
       font_size=16,
   )

Structured configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import OverlayClient, OverlayConfig

   config = OverlayConfig(
       enabled=True,
       show_label=True,
       show_confidence=True,
       line_thickness=2,
   )
   OverlayClient().apply(config)

Strict frame-lock mode (P1-6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # The default is "live preview" semantics: the bake site draws the
   # freshest unexpired result, so boxes and pixels are not guaranteed
   # to be from the same frame. Strict frame-lock is opt-in: once
   # enabled, camera-daemon waits — bounded — at the bake site (before
   # encoding) for each frame's OWN inference result; on timeout it
   # degrades to the freshest result (logged + counted), so encoded
   # output never stalls on the wait.
   overlay.configure(strict_frame_lock=True)              # enable (cap derived from fps)
   overlay.configure(strict_frame_lock=True,
                     strict_wait_cap_ms=0)                # 0 = derive from stream fps (2 frame periods)
   overlay.configure(strict_frame_lock=True,
                     strict_wait_cap_ms=66)               # explicit 66 ms cap
   overlay.configure(strict_frame_lock=False)             # explicitly leave strict mode

   # Both arguments are optional-field semantics: omitted = keep the
   # daemon's current setting; enable()/disable() and a bare configure()
   # never touch strict mode.

   # Strict mode only gates identity-fed display streams — stream_map
   # entries whose display stream is fed its own inference results
   # (e.g. main:main). Cross-fed streams (e.g. third:main) keep live
   # preview semantics.

Combining with inference results (annotate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # annotate() pushes detections through the event bus to
   # camera-daemon's overlay renderer: the app never touches video
   # frames — the daemon draws the boxes onto the stream before
   # encoding. Results expire after 500 ms, so call at inference
   # cadence; publish an empty list to clear the screen.
   from neoruntime_ipc_sdk import OverlayClient

   overlay = OverlayClient()
   overlay.enable()

   for result in inference_results:      # e.g. InferenceClient.subscribe(...)
       overlay.annotate("main", result.objects)

   overlay.annotate("main", [])          # clear

Other result kinds
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # annotate_result picks whichever section of an InferenceResult is
   # populated, with precedence objects > classifications > landmarks
   # > ocr_lines
   overlay.annotate_result("main", result)

Lifecycle session tag (P2-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # annotate / annotate_result / polygon writes all accept session_id:
   # drawings belonging to one session carry the same tag. When the
   # inference session ends (including the connection drop after the
   # app is SIGKILLed), camera-daemon sweeps every polygon carrying
   # that tag in one shot on the session/end event; detection boxes
   # themselves expire via the result TTL.
   overlay.annotate("main", result.objects, session_id="app-1")
   overlay.annotate("main", polygons=zones, session_id="app-1")

   # Omitting session_id = operator write: takes effect unconditionally
   # and never participates in session cleanup. StreamPipeline draws
   # exactly this distinction — result drawings are tagged, while the
   # static zones at start and the clear writes at stop() are not.

Two modes and the video side-effect contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SDK offers two non-interfering ways to consume video data, with an
explicit contract for each:

**Results only (zero video side effects)**: ``InferenceClient.subscribe(...)``
returns inference results and touches no video path. The platform's own
models (results published by ai-runtime / auto-infer) also do **not**
alter the stream by default — unless the operator binds them explicitly
in the daemon config (next section). ``annotate()`` is not a side-effect
violation: it is a drawing command the app issues on purpose.

**Fetch frames, draw yourself (keep_fd / copy)**: ``FdMediaClient``'s
keep_fd / copy modes hand out **clean frames** (never baked by the
overlay). The app composites on its own buffers: copy frames are yours
to write directly; keep_fd frames go through DSP_IMPORT into the app's
own DSP pool for RESIZE→BLEND (per-call cost is acceptable with the
P1-9 resident pools). **A keep_fd frame is never a DSP write target or
blend base** — the platform enforces the read-only contract (writes are
rejected); `dsp-import-write-target-test` guards it.

Platform result bindings (daemon config)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Platform model detections are not drawn by default (behavior
decoupling): a bare ``subscribe()`` must not change any video output.
To enable platform boxes, bind the streams explicitly in the daemon
config yaml under ``[ai_overlay]``:

.. code-block:: yaml

   [ai_overlay]
   # infer stream:display stream, same direction as stream_map;
   # comma-separated for multiple bindings
   bindings: "main:main,third:main"
   # Migration switch: restores the old behavior (every platform result
   # auto-drawn + the default stream_map backfill)
   legacy_auto_bind: 0

**Migration note**: ``legacy_auto_bind`` defaults to off. If platform
boxes disappear after an upgrade, either (1) add the streams you need
to ``bindings`` as above (recommended: enable only what you use), or
(2) set ``legacy_auto_bind: 1`` for the full legacy behavior (every
platform result auto-bound, including the default stream_map
backfill). Drawings the app issues via ``annotate()`` are unaffected
by the switch — they always take effect.

Frame-sync binding (frame_sequence / stream_epoch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # annotate / annotate_result accept two optional frame-sync params
   # (both ride event metadata, never the payload; omitted = unbound,
   # keeping the legacy semantics)
   #
   # frame_sequence: bind this drawing to that frame — it draws only
   #   while the bake site is within the frame-bind slack (+2 frames)
   #   of it, then expires; commands whose frame the pipeline already
   #   passed are rejected (overlay_late_commands counter).
   #   result.frame_sequence from a subscribe() iteration is the
   #   natural source.
   # stream_epoch: the stream generation counter (readable via
   #   get_stream_status()). Reconfiguration (ReconfigureEncoder /
   #   resolution and full transform reinit) bumps the epoch and purges
   #   held layers; events carrying a stale epoch are rejected
   #   (overlay_epoch_rejects counter) instead of turning into ghost
   #   boxes.
   for result in stream.subscribe(model):
       overlay.annotate_result("main", result,
                               frame_sequence=result.frame_sequence,
                               stream_epoch=epoch)

   # Self-healing after a reconfigure: notice the epoch changed, then
   # keep publishing with the new one.
   status = {s.stream_id: s for s in camera.get_stream_status()}
   epoch = status["main"].stream_epoch

The keep_fd read-only recipe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # copy mode: the frame is the app's own memory — write freely
   frame = media.get_frame(stream, copy=True)
   draw_boxes(frame, results)

   # keep_fd mode: the dma-buf is read-only. Import it via DSP_IMPORT
   # into the app's own DSP pool, RESIZE to the target size and use it
   # as the BLEND overlay layer over the app's own base; the frame
   # itself is never a write target — a RESIZE dst / BLEND base
   # pointing at a keep_fd frame is rejected by the platform
   # (enforced, not a convention).
   dsp.import_buffer(frame.fds, frame_id=frame.frame_id)
   dsp.schedule(op=DSP_OP_RESIZE, src=imported, dst=scratch)      # own scratch
   dsp.schedule(op=DSP_OP_BLEND, base=composed, overlay=scratch)  # own base
   # In short: a keep_fd frame = a BLEND overlay input or RESIZE src;
   #           the app's own buffers = every write target.

Context manager
~~~~~~~~~~~~~~~

.. code-block:: python

   with OverlayClient() as overlay:
       overlay.enable()

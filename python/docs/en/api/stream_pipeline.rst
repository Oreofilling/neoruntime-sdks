Whole-Pipeline API (Platform-Scheduled)
=======================================

.. automodule:: neoruntime_ipc_sdk.stream_pipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

StreamPipeline
--------------

.. autoclass:: neoruntime_ipc_sdk.StreamPipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

StreamPipelineStatus
--------------------

.. autoclass:: neoruntime_ipc_sdk.StreamPipelineStatus
   :members:
   :undoc-members:

Examples
--------

One call: subscribe + hardware overlay + app-side results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import StreamPipeline

   zones = [{"points": [[0, 0], [1, 0], [1, 1]], "label": "yard"}]
   pipe = StreamPipeline(
       "main", "yolov5n", fps=10,
       min_score=0.5, labels=["person"], polygons=zones,
       on_result=lambda r: r if r.objects else None,  # skip empty results
   ).start()

   # Optional: consume results app-side too (alerts, forwarding, extra
   # processing). The annotated video never passes through here — boxes
   # are baked into the encoded stream by the hardware overlay.
   for seq, result in pipe.results():
       print(seq, [(o.label, round(o.score, 2)) for o in result.objects])
       if seq >= 100:
           break

   st = pipe.status()          # results_seen / annotated / drops / skew ...
   pipe.stop()                 # clears boxes + static zones; stream is clean

Semantics
~~~~~~~~~

- **Positioning vs InferencePipeline (pipeline.py)**: StreamPipeline is
  the platform-scheduled pipeline — the camera daemon feeds frames to
  the model (``subscribe``), no pixels enter the app process, results go
  to the hardware overlay via ``OverlayClient`` and are baked into the
  encoded stream. InferencePipeline is the client-side convenience —
  frames are pulled into the app and pre→infer→post runs synchronously
  on the app thread; reach for it when the app needs the pixels (custom
  preprocessing, multi-model chains on one frame, its own rasterizer).
  Choose by where the pixels should live.
- **Single-shot lifecycle**: ``start()`` once, ``stop()`` once; build a
  new instance for another run. ``stop()`` is idempotent (no-op before
  ``start()``), joins the worker before clearing boxes/static polygons,
  and closes only the clients it created (injected ``inference=`` /
  ``overlay=`` are left open).
- **Filtering affects drawing only**: ``min_score`` / ``labels`` filter
  what reaches the overlay; ``results()`` still yields the full result.
  A fully-filtered result publishes an empty detection list so stale
  boxes clear.
- **on_result hook** runs per result on the worker thread: return a
  (possibly replaced) result to keep it flowing, or ``None`` to drop it;
  a raised exception drops only that result and is recorded in
  ``status().last_error`` — never fatal.
- **Two bounded queues, drop-oldest**: the subscribe side
  (``queue_size``, surfaced as ``subscribe_dropped``) and the app side
  (``result_queue_size``, counted in ``result_queue_drops``); annotate
  failures increment ``annotate_errors`` without stopping the pipeline.
- **draw=False never touches the overlay** (static polygons included) —
  the encoded stream stays clean; results still flow to ``results()``.
- **Session tag (P2-13)**: ``session_id=`` spans inference and drawing —
  the subscription and every result drawing carry the same tag; when the
  process dies without cleanup (SIGKILL included), the inference
  session-end event makes camera-daemon sweep every polygon carrying
  that tag. The static zones at start and the clear writes at ``stop()``
  carry no tag — operator writes that take effect unconditionally.
- **status() passes the subscribe observability through**: last/avg
  latency_ms and last/avg skew_us (result-timestamp minus
  frame-timestamp — the acceptance metric shared by the preview and
  strict frame-lock modes).

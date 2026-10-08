Learning Path
=============

From zero to your first installed app. Time estimates assume a
NeoRuntime device on hand; every stage links to the artifact that
carries it.

.. note::

   Everything live — video streams, inference, overlay — talks to
   platform daemons over local sockets, so a device is required to
   *run* code. Without one, this page and the API reference still show
   you the shape of the SDK.

The path
--------

.. list-table::
   :header-rows: 1
   :widths: 16 10 40 34

   * - Stage
     - Time
     - Do
     - Where
   * - 1. First run
     - 5 min
     - Install the SDK, run one inference
     - :doc:`quickstart`
   * - 2. See it live
     - 15 min
     - Install the interactive teaching demo, drive its five stations
     - `sdk-teaching-demo`_ (below)
   * - 3. Adopt
     - ~1 h
     - Read short runnable examples, start your app from the skeleton
     - apps repo ``examples/`` + ``templates/basic``
   * - 4. Ship
     - day+
     - Package a ``.neoapp``, restart policy, your own web page
     - :doc:`app_image_guide`
   * - 5. Reference
     - anytime
     - API reference, diagnostics, error lessons
     - API Reference (site menu)

Stage 1 — First run (5 minutes)
-------------------------------

.. code-block:: bash

   python -m pip install neoruntime-ipc-sdk

Then run the first-inference snippet from :doc:`quickstart` on the
device. You are done when one ``infer()`` call returns detections.

Stage 2 — See every core pattern live (15 minutes)
--------------------------------------------------

**sdk-teaching-demo** is a single installable app with five live
stations against real camera streams: subscribe + infer + draw (B form),
platform overlay vs your own pixels, hardware routing (including a
deliberate refusal), A-form ``StreamPipeline``, and cooldown-gated
event publishing. Every station shows the real code behind it (copy
button included) and the error lesson for that station — DMA ``-2811``,
``None`` vs ``[]`` overlay cleanup, exit codes under
``restart_policy: on-failure``.

.. code-block:: bash

   # after downloading the bundle, on the device:
   aipc-cli app install sdk-teaching-demo <path-to-app.yaml> <path-to-image.tar>
   aipc-cli app start sdk-teaching-demo

Then open ``http://<device>:8090``. You are done when you have driven
all five stations and read at least one error lesson.

Bundle: `sdk-teaching-demo-latest-arm64.neoapp
<https://github.com/camthink-ai/neoruntime-apps/releases/download/showcase-bundles-latest/sdk-teaching-demo-latest-arm64.neoapp>`_

.. _sdk-teaching-demo: https://github.com/camthink-ai/neoruntime-apps/tree/main/showcases/sdk-teaching-demo

Stage 3 — Adopt the patterns (~1 hour)
--------------------------------------

Read short, complete apps in the apps repository's `examples directory
<https://github.com/camthink-ai/neoruntime-apps/tree/main/examples>`_
(``hello-world``, ``object-detection``, ``people-counting``,
``person-detection``) — each is an ``app.yaml`` plus one main file.
Then start your own app from the `basic template
<https://github.com/camthink-ai/neoruntime-apps/tree/main/templates/basic>`_.

You are done when your copy of the skeleton runs on the device and you
have changed one thing — a draw color, a topic name.

.. note::

   A numbered, progressive examples ladder (01 hello-app through
   10 dsp-offload) is on the roadmap for ``examples/``; the teaching
   demo's five stations map onto its middle rungs.

Stage 4 — Ship it
-----------------

:doc:`app_image_guide` covers the app image contract: ``app.yaml``
fields, ``restart_policy`` and exit-code semantics, packaging a
``.neoapp``, and registering your web page so the console can proxy it.

Stage 5 — Reference layer (anytime)
-----------------------------------

- The API reference — this site's menu, second section.
- ``diagnostics()`` — the first stop when an app feels slow: it reports
  which operations run on hardware vs software fallbacks.
  ``diagnostics.check(...)`` doubles as a deployment gate.
- The teaching demo's in-page error lessons stay available while the
  demo is installed.

Stuck?
------

- App feels slow → call ``diagnostics()`` and read the ``accel_router``
  section: an operation silently on the CPU fallback is the usual
  cause.
- Overlay shows nothing → the overlay must be ``enable()``\ d first,
  and results expire after about two frame periods — see
  :doc:`api/overlay`.
- ``register_model`` rejects your model → the model's input geometry
  must match the stream geometry you subscribe; streaming subscriptions
  do not rescale frames client-side.
- Otherwise: `GitHub issues
  <https://github.com/camthink-ai/neoruntime-sdks/issues>`_.

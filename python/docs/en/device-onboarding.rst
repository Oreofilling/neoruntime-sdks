Device Onboarding
=================

From zero to your first inference result, all you need is a device.
This page covers getting one and logging into it; installing and using
the SDK is covered in :doc:`installation` and :doc:`quickstart`.

Getting a device
----------------

The device this SDK targets is the **NeoEyes NE503** (4K PoE edge AI
camera), available from the `CamThink store <https://www.camthink.ai/store/>`_.

.. note::

   The store also carries NE101 / NE301 / NE302 product lines — those
   are STM32/ESP32 based and do not run NeoRuntime. Make sure it is an
   **NE503**.

What's in the box: the camera, and one PoE-powered Ethernet cable
(power and data on a single cable; a direct laptop connection needs a
PoE injector or PoE switch). More resources: the
`Developer Center <https://www.camthink.ai/>`_ and
`GitHub <https://github.com/CamThink-AI>`_.

First boot and login
--------------------

Finding the device
~~~~~~~~~~~~~~~~~~

Out of the box (before commissioning) eth0 comes up as a static
address ``10.0.0.1/24``:

1. Set your computer's Ethernet interface to the same subnet, e.g.
   ``10.0.0.2/24``;
2. Connect the device (PoE powered);
3. Browse to ``http://10.0.0.1:8080``.

For devices already provisioned onto a LAN: every NE503 continuously
announces itself (model, version, web port) via the
``device-discovery`` service on multicast ``239.255.255.250:19850``.
On the same switch (layer-2 reachable) you can listen for it::

   python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('', 19850)); s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton('239.255.255.250')+socket.inet_aton('0.0.0.0')); print(s.recvfrom(4096))"

Logging into the web console
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Browse to ``http://<device-ip>:8080``. Factory credentials:

- username ``admin``
- password ``password``

.. warning::

   Factory credentials are for first-time onboarding only. Once the
   device is attached to an untrusted network, change the password
   immediately (deployments can also override both via the
   ``AIPC_AUTH_USERNAME`` / ``AIPC_AUTH_PASSWORD`` environment
   variables).

SSH and the web terminal
~~~~~~~~~~~~~~~~~~~~~~~~

The console ships with a built-in web terminal: once you are logged
in, you get a root shell with no extra credentials. SSH is also
enabled by default (port 22, root password login allowed) and can be
tuned from the console's SSH settings page.

.. note::

   The root SSH password belongs to the OS image layer; see the
   material shipped with your device. After first login, changing it
   via ``passwd`` in the web terminal is recommended.

aipc-cli: preinstalled, nothing to install
------------------------------------------

``aipc-cli`` ships with the firmware (``/usr/bin/aipc-cli``); its
version follows the firmware and it is not installed separately. The
only thing your workstation needs is the SDK itself::

   pip install neoruntime-ipc-sdk

Over SSH or the web terminal, verify on the device::

   aipc-cli --version
   aipc-cli monitor        # live resource panel: daemons and NPU alive = ready

Useful subcommands: ``app`` (install/start/stop/logs), ``model``
(model management), ``media`` (stream config), ``event`` (event bus),
``files``, ``logs``.

Next steps
----------

- :doc:`learning-path` — the full learning path (this page is its
  "stage 0")
- :doc:`quickstart` — your first inference in ten minutes

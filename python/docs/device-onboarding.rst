设备开通
========

从零到第一行推理结果，只差一台设备。本页讲设备怎么到手、怎么开机登录；
SDK 的安装与使用见 :doc:`installation` 与 :doc:`quickstart`。

获得设备
--------

NeoRuntime SDK 对应的整机是 **NeoEyes NE503** (4K PoE 边缘 AI 相机)，
在 `CamThink 商店 <https://www.camthink.ai/store/>`_ 购买。

.. note::

   商店里还有 NE101 / NE301 / NE302 等产品线——它们基于 STM32/ESP32，
   不运行 NeoRuntime，也用不了本 SDK。认准 **NE503** 这个型号。

到手清单：主机、PoE 供电的网线（一根线同时供电和传数据；直连笔记本时
需要 PoE 供电器或 PoE 交换机）。更多资源见
`Developer Center <https://www.camthink.ai/>`_ 与
`GitHub <https://github.com/CamThink-AI>`_。

首次开机与登录
--------------

找到设备
~~~~~~~~

出厂状态（commissioning 之前）eth0 为静态地址 ``10.0.0.1/24``：

1. 把电脑的以太网口设为同网段，例如 ``10.0.0.2/24``；
2. 网线接通设备（PoE 供电）；
3. 浏览器打开 ``http://10.0.0.1:8080``。

已配置到局域网的设备：每台 NE503 都会通过 ``device-discovery`` 服务在
组播 ``239.255.255.250:19850`` 上持续播报自己（型号、版本、Web 端口），
同一交换机（二层可达）内可监听到::

   python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('', 19850)); s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton('239.255.255.250')+socket.inet_aton('0.0.0.0')); print(s.recvfrom(4096))"

登录 Web 控制台
~~~~~~~~~~~~~~~

浏览器打开 ``http://<设备IP>:8080``，出厂账号：

- 用户名 ``admin``
- 密码 ``password``

.. warning::

   出厂凭据仅供首次开通使用。设备一旦接入不可信网络，请立即修改
   密码（部署侧也可用 ``AIPC_AUTH_USERNAME`` / ``AIPC_AUTH_PASSWORD``
   环境变量整机制覆盖）。

SSH 与网页终端
~~~~~~~~~~~~~~

控制台内置网页终端：登录控制台后即可打开 root shell，不需要任何
额外凭据。SSH 同样默认开启（端口 22，允许 root 密码登录），可在
控制台的 SSH 配置页里调整。

.. note::

   root 的 SSH 密码属于系统镜像层，以设备随附资料为准。首次开通后
   建议在网页终端里执行 ``passwd`` 修改。

aipc-cli：设备自带，无需安装
----------------------------

``aipc-cli`` 随固件内置在设备上（``/usr/bin/aipc-cli``），版本跟随
固件，不需要也不支持单独安装。你的工作站上只需要装 SDK 本体::

   pip install neoruntime-ipc-sdk

SSH 或网页终端进入设备后自检::

   aipc-cli --version
   aipc-cli monitor        # 实时资源面板：守护进程与 NPU 都活着就绪了

常用子命令：``app``（应用安装/启停/日志）、``model``（模型管理）、
``media``（流配置）、``event``（事件总线）、``files``、``logs``。

下一步
------

- :doc:`learning-path` —— 完整学习路径（本页即其"阶段 0"）
- :doc:`quickstart` —— 十分钟跑出第一帧推理

学习路径
========

从零到第一个可安装的 app。时间估算以手头有一台 NeoRuntime 设备为前提；每个阶段都指向承载它的资产。

.. note::

   所有"活"的部分——视频流、推理、overlay——都通过本地 socket 与
   平台守护进程通信，因此*运行*代码需要设备。没有设备时，本页与
   API 参考仍能让你看清 SDK 的全貌。

还没有设备？从 :doc:`device-onboarding` 开始：那里讲怎么买到
NE503、首次开机怎么登录。

路径总览
--------

.. list-table::
   :header-rows: 1
   :widths: 16 10 40 34

   * - 阶段
     - 时间
     - 做什么
     - 载体
   * - 1. 首跑
     - 5 分钟
     - 安装 SDK，跑通一次推理
     - :doc:`quickstart`
   * - 2. 看活的
     - 15 分钟
     - 安装交互式教学 demo，开五个站点
     - `sdk-teaching-demo`_ （见下）
   * - 3. 拿来用
     - ~1 小时
     - 读短小可跑的示例，从骨架起自己的 app
     - examples 阶梯（01–05）+ ``templates/basic``
   * - 4. 交付
     - 一天起
     - 打 ``.neoapp`` 包、重启策略、自己的网页
     - :doc:`app_image_guide`
   * - 5. 查阅
     - 随时
     - API 参考、诊断、错误课
     - 本站 API 参考（站点菜单）

阶段 1 —— 首跑（5 分钟）
-------------------------

.. code-block:: bash

   python -m pip install neoruntime-ipc-sdk

然后在设备上跑 :doc:`quickstart` 里的首个推理片段。当一次
``infer()`` 调用返回检测结果，本阶段完成。

阶段 2 —— 看每个核心模式活的样子（15 分钟）
---------------------------------------------

**sdk-teaching-demo** 是一个可安装的单体 app，五个站点对着真实
摄像头流运行：订阅 + 推理 + 绘制（B 形态）、平台 overlay 对比自绘
像素、硬件路由（含一次刻意的拒绝演示）、A 形态 ``StreamPipeline``、
冷却门控的事件发布。每个站点都展示它背后的真实代码（带复制按钮）
和该站的错误课——DMA ``-2811``、``None`` 与 ``[]`` 的 overlay 清理
语义、``restart_policy: on-failure`` 下的退出码。

.. code-block:: bash

   # 下载产物后，在设备上：
   aipc-cli app install sdk-teaching-demo <path-to-app.yaml> <path-to-image.tar>
   aipc-cli app start sdk-teaching-demo

然后打开 ``http://<设备>:8090``。当你开过全部五个站点并读过至少
一条错误课，本阶段完成。

产物：`sdk-teaching-demo-latest-arm64.neoapp
<https://github.com/camthink-ai/neoruntime-apps/releases/download/showcase-bundles-latest/sdk-teaching-demo-latest-arm64.neoapp>`_

.. _sdk-teaching-demo: https://github.com/camthink-ai/neoruntime-apps/tree/main/showcases/sdk-teaching-demo

阶段 3 —— 拿来用（约 1 小时）
------------------------------

先走 `examples 阶梯
<https://github.com/camthink-ai/neoruntime-apps/blob/main/examples/README.md>`_：
五个编号递进的最小应用——01 hello-app（免设备可跑）到 05
live-detection（活循环 + 自建 MJPEG 网页）——每级只引入一个新
概念，每级自带离线测试，05 正是教学 demo 站 1 的独立 app 版。
更大的完整示例（``object-detection``、``people-counting``、
``person-detection``）可作进阶参考。然后从 `basic 模板
<https://github.com/camthink-ai/neoruntime-apps/tree/main/templates/basic>`_
起你自己的 app。

当阶梯读完（或骨架副本在设备上跑起来）、并且你改了一样东西
（一个绘制颜色、一个 topic 名），本阶段完成。

阶段 4 —— 交付
---------------

:doc:`app_image_guide` 讲清应用镜像契约：``app.yaml`` 字段、
``restart_policy`` 与退出码语义、打 ``.neoapp`` 包、注册你的网页让
控制台反代它。

阶段 5 —— 查阅层（随时）
-------------------------

- API 参考——本站菜单第二区。
- ``diagnostics()``——app 感觉变慢时的第一站：它报告哪些操作跑在
  硬件、哪些落在软件回退。``diagnostics.check(...)`` 还可以当部署
  门禁用。
- 教学 demo 的页内错误课在 demo 装着期间随时可查。

卡住了？
--------

- app 变慢 → 调 ``diagnostics()`` 看 ``accel_router`` 段：某操作
  静默落在 CPU 回退是最常见原因。
- overlay 什么都不显示 → overlay 必须先 ``enable()``，且结果在约
  两个帧周期后过期——见 :doc:`api/overlay`。
- ``register_model`` 拒绝你的模型 → 模型输入几何必须与你订阅的流
  几何一致；流式订阅不会在客户端缩放帧。
- 其他：`GitHub issues
  <https://github.com/camthink-ai/neoruntime-sdks/issues>`_。

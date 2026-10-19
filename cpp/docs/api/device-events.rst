设备控制与事件总线：DeviceClient / EventClient
===============================================

``device.hpp`` 管执行器与传感器，``events.hpp`` 是跨应用 / 跨系统的
发布订阅总线。

事件总线
--------

发布：JSON 载荷用 ``nlohmann::json``；``persistent=true`` 让事件在
总线内持久（配合 ``ttl_ms``），离线消费者补得回：

.. code-block:: cpp

   neoruntime_ipc_sdk::EventClient events;

   std::string event_id = events.publish(
       "app/" + app_id + "/perimeter_alert",
       {{"type", "boundary_crossing"},
        {"frame_sequence", frame_seq},
        {"person_count", crossed.size()}},
       /*persistent=*/true);

   // 批量：{topic, payload} 对的向量，一次流式 RPC
   events.publish_batch({{"app/tick", {{"n", 1}}},
                         {"app/tick", {{"n", 2}}}});

订阅两种形状。回调版是后台分离线程（镜像 Python 的 on_event daemon
线程）；拉取版 ``subscribe()`` 返回 ``EventStream``，``next()`` 出
``std::optional<Event>``，同样支持通配符与 filters：

.. code-block:: cpp

   // 通配符主题：+ 单层，# 多层
   events.on_event("model/+/detections", on_detection);
   events.on_event("app/+/alert",        on_alert);
   events.on_event("system/device/#",    on_system);

   auto sub = events.subscribe("model/+/detections");
   while (auto ev = sub.next()) {
       std::printf("%s <- %s\n", ev->topic.c_str(), ev->source.c_str());
   }

.. warning::

   ``on_event`` 的回调在 **分离线程** 里跑且可能活过客户端析构（事件
   still in flight）——回调必须自包含，别捕获指向栈对象或客户端本身
   的引用。``source`` 与 ``timestamp_ns`` 留空时自动填（APP_ID /
   当前时刻）。

观测面：``list_topics()`` / ``get_topic_info()`` / ``get_stats()``
（``SystemStats`` 含 per-topic 的 published / delivered / dropped 与
平均时延）。

设备控制
--------

.. code-block:: cpp

   neoruntime_ipc_sdk::DeviceClient device;

   device.set_white_light(100);   // 0 关；告警联动灯的常见写法
   // ... 无人在画面后延时关灯
   device.set_white_light(0);

   auto status = device.get_device_status();   // 整机状态
   auto lens   = device.get_lens_status();     // soc_temp_c、
   // zoom_pos / focus_pos 与镜头上下限对照

``DeviceClient`` 还覆盖：红外 / IRCUT、云台、变焦 / 对焦、GPIO、
RS485、Wiegand，以及自动对焦套件（``AfJob`` / ``AfStatus`` /
``AfMeasurement``）。

.. tip::

   事件 + 设备联动是平台的惯用组合——参考
   ``examples/event_subscriber.cpp`` （收到 person 检测事件即点灯）
   与 ``examples/perimeter_guard.cpp`` （越线告警 + 5 秒冷却 + 10 秒
   后自动关灯，完整可编译）。

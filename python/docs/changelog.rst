更新日志
========

v0.8.0 (未发布)
-------------------

弃用
~~~~

- **插件系统 API** (``PluginDiscovery`` / ``PluginServer`` / ``PluginEndpoint``) 已弃用,计划在 v0.8.0 移除。当前平台未部署 ``/run/aipc/plugins`` 插件发现机制,SDK 暂保留导入兼容并发出 ``DeprecationWarning``,文档页已下线。如有使用需求请提前反馈。

文档
~~~~

- 修正文档与 README 中的视频流示例:移除对从未存在过的 ``MediaClient`` 的引用,统一改用 ``FdMediaClient``;流 ID 由 ``cam0_main`` / ``cam0_sub`` 修正为设备实际暴露的 ``main`` / ``sub``;移除不存在的 ``get_stream_info()`` / ``get_raw_stream()`` 用法;修正 ``get_encoded_stream()`` 返回值语义与 ``frame.data`` 展平数组误用;补全 ``EncodedStreamClient`` / ``EncodedFrame`` API 文档

v0.7.0 (2026-09-02)
-------------------

新增功能
~~~~~~~~

- **应用开发工具箱** (原 0.6.0 开发版内容,未单独发布,随 0.7.0 一并发布):

  - ``Frame.crop()`` / ``Frame.resize()`` / ``Frame.to_jpeg_bytes()`` — 帧裁剪、三模式缩放(stretch/letterbox/crop)、JPEG 编码;cv2 缺失时自动降级到 numpy/PIL
  - ``draw`` 模块 — 检测结果可视化(``draw_boxes`` / ``draw_text`` / ``draw_detections``)
  - ``recording`` 模块 — 纯 Python TS 复用打包、HLS 切片、事件预录缓冲(``TsWriter`` / ``HlsWriter`` / ``PrerollBuffer``),无需 ffmpeg
  - ``web`` 模块 — MJPEG 推流(``MjpegServer`` / ``MjpegStream`` / ``mjpeg_wsgi_app``)
  - **DeviceClient** 原生对焦组 6 个方法(``start_oneshot_af`` / ``start_zoom_follow`` / ``get_autofocus_status`` / ``cancel_autofocus`` / ``set_af_windows`` / ``get_af_measurement``)
  - **CameraClient** 成像/红外/隐私遮挡/OSD/配置组 12 个方法(含红外预设管理、``get_osd`` 与 ``set_osd`` 读写对称)
- 修复 ``InferenceClient.subscribe()`` 静默丢弃失败帧的问题

重构
~~~~

- 内部架构分层:抽取 ``_transport`` 共享传输原语; ``media`` 拆分为 frame/encoded/fd_client; ``dsp`` 拆分为 dsp_wire/dsp_format; ``inference`` 拆分为 types/codec/genai。门面保持全部历史导入路径, **零公共 API 移除**
- DSP CPU 回退从静默日志改为 ``UserWarning``
- 2D 数组按灰度推理输入触发 ``DeprecationWarning`` (请显式传 ``fmt=``)
- ``EncodedStreamClient()`` 默认套接字路径改为 ``/run/aipc/encoded/{stream}.sock`` (``ENCODED_SOCK_DIR`` 环境变量可覆盖,显式路径仍优先)

修复
~~~~

- 修复 Python 3.8/3.9 下的导入崩溃(全量补齐 ``from __future__ import annotations``)

其他
~~~~

- 打包增加 ``py.typed`` 类型标记;测试扩充至 270 项全绿

v0.5.0 (2026-08-20)
-------------------

重构
~~~~

- 品牌重构:``hailo_ipc_sdk`` → ``neoruntime_ipc_sdk`` (破坏性变更,无兼容层;hailort/ne503/aipc 等功能性名称保留)

修复
~~~~

- 修复 ``audio_capture`` 流视频布局头的双布局解码

其他
~~~~

- CI 建立 TestPyPI/PyPI Trusted Publishing 双发布流程

v0.4.0 (2026-07-14)
-------------------

新增功能
~~~~~~~~

- **DeviceClient** 新增 7 个方法：
  - ``set_lens_limits(zoom_limit, focus_limit)`` — 设置镜头轴限位
  - ``oneshot_autofocus(timeout)`` — 单次自动对焦（复合操作：开启→等待收敛→关闭）
  - ``set_wiegand_out(channel, enable)`` — 韦根输出控制
  - ``get_wiegand_out(channel)`` — 韦根输出状态查询
  - ``rs485_init(baudrate, config)`` — RS-485 串口初始化
  - ``rs485_deinit()`` — RS-485 串口反初始化
  - ``rs485_tx(data)`` — RS-485 数据发送
- **AppClient** 新增 1 个方法：
  - ``restart_app(app_id, timeout_seconds)`` — 应用重启（停止+启动）

改进
~~~~

- 更新 API 文档，补充 DeviceClient 镜头限位、韦根、RS-485 示例
- 更新 API 文档，补充 AppClient 完整使用示例
- 同步中英文文档

v0.2.0 (2026-03-02)
-------------------

新增功能
~~~~~~~~

- 添加插件系统支持 (PluginDiscovery, PluginServer)
- 支持插件能力发现和 gRPC 服务调用
- 新增视频流访问能力（现由 ``FdMediaClient`` / ``EncodedStreamClient`` 提供）
- 支持原始视频流和编码视频流获取

改进
~~~~

- 优化 InferenceClient 性能
- 改进事件总线通配符匹配
- 增强错误处理和日志记录
- 更新 protobuf 到 4.21.0

修复
~~~~

- 修复 EventClient 订阅时的内存泄漏
- 修复 DeviceClient GPIO 控制问题
- 修复多线程环境下的连接池问题

v0.1.0 (2025-12-15)
-------------------

初始版本
~~~~~~~~

- InferenceClient: AI 推理客户端
- EventClient: 事件总线客户端
- DeviceClient: 设备控制客户端
- Config: 配置管理
- 支持 Python 3.8+
- 基于 gRPC 通信

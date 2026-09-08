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
- 修复应用示例中的缺失导入与误导性导入:多模型融合示例补 ``import time``;英文推理页 GenAI 示例补 ``import json``;错误处理示例移除 ``from grpc import RpcError`` (SDK 实际抛出 ``RuntimeError``);清理未使用的 ``numpy`` / ``sys`` 导入
- 补译中文推理页缺失章节:自 v0.3.0 起仅存在于英文文档的分割 / OCR / CLIP 图像嵌入 / CLIP 文本编码 / 深度估计 / 运行时更新后处理配置 / GenAI 七个使用小节,以及 ``SegmentationMask`` / ``OcrLine`` / ``Embedding`` / ``DepthMap`` 四个数据类型条目,中英文档现已同步

v0.7.4 (2026-09-08)
-------------------

新增功能
~~~~~~~~

- **便捷层硬件路由** (sdk-hardware-routing P5): ``color.rgb_to_nv12`` / ``color.nv12_to_rgb`` 与 ``draw.draw_detections``(NV12 数组)默认经 accel 路由器走 DSP 腿,不可用时自动回落软件实现并记账;``Frame.to_jpeg_bytes`` 对 keep-fd 帧硬件优先(camera-daemon ``EncodeImage`` 零拷贝导入),内存帧保持 CPU 编码直达,推流热路径不付额外拷贝与 RPC;软件腿只绑定私有实现,路由公函数不可重入
- ``Frame.resize`` 的 DSP 快路径尊重路由策略: ``SOFTWARE_ONLY`` 跳过尝试,``HARDWARE_ONLY`` 失败即抛错(拒绝静默 CPU 回落),``PREFER_HARDWARE`` 经新增的 ``AccelRouter.note_degradation()`` 记账(同时新增 ``policy`` 属性;外部上报的未注册 op 在 ``health()`` 可见,``on_degradation`` 照常触发)

修复
~~~~

- ``_encode_jpeg_hw`` 硬编码 ``fmt="rgb24"``,NV12 keep-fd 帧的提交必被 ``_resolve_source`` 以格式不匹配拒绝;帧类源现在随句柄自带格式
- ``dsp_format`` 软件腿经路由公函数重入路由器(跨 op 耦合);改为直接绑定 color 私有实现
- 色彩转换两腿的 ``ValueError`` 契约统一(ndim / 声明尺寸 / 偶数校验先于任何 daemon 提交,调用方错误不再计入降级计数)

v0.7.3 (2026-09-07)
-------------------

新增功能
~~~~~~~~

- **硬件优先加速路由器** (``accel`` 模块): ``AccelRouter`` / ``RoutePolicy`` / ``health()`` 降级记账 + ``on_degradation`` 钩子 / ``get_default_router()``;默认路由注册 ``resize_nv12`` / ``rgb_to_nv12`` / ``nv12_to_rgb`` / ``encode_jpeg`` / ``draw_detections`` 的 DSP 腿与 ``nms`` 软件腿
- **DSP 标注合成** (dsp-offload P1): ``DspClient.blend_hw`` 把 ARGB32 覆盖层在 DSP 上原位合成回 NV12;``draw.render_overlay_rgba`` 将框+文字渲染为最小 straight-alpha 画布,复现软件光栅输出
- **异步 DSP 作业** (dsp-offload P2): 全部 ``*_hw`` 操作支持 ``wait=False``,``PendingDspJob`` 提供 ``wait()`` / ``done()`` / ``release()``;``blend_hw`` 零拷贝 keep-fd 链(默认拒用——当前固件上为状态依赖的致命路径);``encode_jpeg_hw`` 支持 ``src_buffer_id`` 免像素读回;keep-fd 帧源在 resize/convert 腿免 ``ascontiguousarray`` 预拷贝;多边形与轨迹形状
- **overlay 标注**: ``annotate()`` / ``annotate_result()`` 经事件总线直推 camera-daemon 渲染器
- **色彩转换与 NMS**: BT.601 limited-range RGB↔NV12(cv2 缺失时的 numpy 路径);跨类感知 ``postprocess.nms``

修复
~~~~

- 推理 codec 重构收尾: ``_tensor_to_numpy`` / ``_parse_post_result`` 遗留 ``self`` 致相应解析必失败;删除被薄委托遮蔽的四份完整方法死实现;新增 AST 结构元测试防止同类漂移

v0.7.2 (2026-09-03)
-------------------

修复
~~~~

- ``_parse_infer_response`` codec 签名遗留 ``self``: 0.7.0 重构后所有 ``infer()`` / ``infer_batch()`` 调用抛 ``TypeError``;补充驱动真实 ``InferResponse`` 的回归测试

v0.7.1 (2026-09-03)
-------------------

- 版本号与 ``setup.py`` 对齐,无 API 变化

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

# Python SDK 性能指南

面向用本 SDK 构建「取帧 → 预处理 → 推理 → 后处理 → 显示」应用的开发者。
核心口诀：**像素尽量不出 daemon、出 daemon 就先变小图。**

## 三种应用形态

同一条需求在这个平台上有三种实现路径，CPU 消耗差一个数量级：

| 形态 | 做法 | 像素进应用 | 适用条件 |
|---|---|---|---|
| **A. 零像素流式** | `InferenceClient.subscribe()` 服务端流式推理；展示用 RTSP + `OverlayClient` 编码前硬件叠加 | 完全不进 | 模型走 family 后处理，展示只要 RTSP/录像带框 |
| **B. 低像素管线** | 订 **sub 流** + `keep_fd=True` → `Preprocessor`（DSP 缩放）→ 小图推理 → 客户端后处理 | 只进小图 | 自定义模型 / 需要自己调阈值 / 需要中间图像（`InferencePipeline` 就是这条路径的官方封装） |
| **C. 朴素管线** | 订 main 流拷出全图 → CPU 转 RGB → resize → 全图上送 → CPU 画框 + 编 JPEG | 全进 | 不了解平台时的默认写法——CPU 最贵，避免 |

## 每阶段 CPU 开销账本（1080p@25fps、模型输入 640×640，数量级）

| 排名 | 消耗点 | 量级 | 削减手段 |
|---|---|---|---|
| 1 | NV12→RGB 色彩转换 | 纯 numpy 数十~上百 ms/帧；cv2 约 2~5 ms | **先缩后转**（640 转换比 1080p 便宜 ~9×）；keep-fd 帧让 DSP 缩放顺带避开 |
| 2 | Web 显示的 JPEG 编码 | cv2 编 1080p 约 10~20 ms/帧 | 预览先缩到 640×360 再编码（2~5 ms）；keep-fd 帧走 daemon EncodeImage 硬件腿 |
| 3 | 帧拷贝 + 推理上送序列化 | 拷贝路径 1080p NV12 ≈ 3.1 MB/帧；全图 RGB 上送 ≈ 6.2 MB/帧 | 订 sub 流；上送前缩到模型输入（省 ~5×）；`subscribe` 服务端推理一个像素都不出门 |
| 4 | resize | cv2 数 ms/帧 | keep-fd 帧的 `Frame.resize` 自动走 DSP，CPU 为零 |
| 5 | 后处理 decode/NMS | 向量化数 ms | `postprocess.nms` 已向量化；YOLO 解码用 `YoloV8/V5Postprocessor`，别写 for 循环 |
| 6 | NPU 推理 | 应用 CPU ≈ 0 | 在 NPU 上。注意 `infer` 同步阻塞；深度流水用 `infer_async` / `InferencePipeline.run_async` |

## 决策树

```
模型是 family 后处理（hailo_yolov8n/s/m 等）且只需要结果？
 ├─ 是 → 形态 A：inf.subscribe(stream, model, fps)
 │        展示带框？→ RTSP + OverlayClient（编码前硬件叠加，零 CPU）
 └─ 否（通用导出模型 / 要调阈值 / 要中间图像）
          → 形态 B：PipelineRunner + InferencePipeline.from_model(model_id, postprocessor=...)
             runner = PipelineRunner(source=media.subscribe("sub", keep_fd=True),
                                     pipeline=pipe, sink=on_result, queue_size=1)
             runner.start()   # latest-wins 背压、丢帧/延迟统计、keep-fd 自动释放都是内建的
```

## 优化 checklist

- [ ] **AI 用 sub 流**：`FdMediaClient().subscribe("sub", keep_fd=True)`；main 流留给录像/RTSP。
- [ ] **先缩后转**：`Preprocessor` 内部已按此顺序（`Frame.resize` → 小图 `to_rgb`）。
- [ ] **keep_fd + DSP**：keep-fd 帧的缩放零拷贝走 DSP；用完 `frame.release()`，别扣住 daemon 缓冲池（扣多了 SDK 会告警，`FdMediaClient.retained_frames` 可查）。
- [ ] **装 opencv**：cv2 是可选依赖，缺它所有转换/缩放/JPEG 走纯 numpy 回退，慢 10~50×——SDK 会在首次回退时告警一次。
- [ ] **各环节限 fps**：`subscribe(fps=...)` 服务端限流；MJPEG 端 `MjpegStream` 是 latest-frame 语义，慢客户端丢帧不积压。
- [ ] **周期性自检**：`diagnostics()` 一次看清 cv2 有无、各 daemon 连通、accel 路由实际 backend；`get_default_router().health()` 看硬件降级计数。
- [ ] **推理订阅注意队列**：`subscribe` 结果队列默认有界（100），慢消费会丢最旧结果并在 `gen.dropped` 计数——它是你的背压信号。
- [ ] **抽帧**：`FdMediaClient.subscribe(stream, skip_frames=N)` 每 N 帧取 1，直接降低 Python 层每帧开销。

## 显示路径选择

| 目标 | 手段 | 应用侧 CPU |
|---|---|---|
| RTSP / 录像流带检测框 | `OverlayClient.annotate()`（编码前硬件叠加，事件总线投递，需几 Hz 刷新） | ≈ 0 |
| 网页预览 | `MjpegStream` + `MjpegServer`，先缩到预览尺寸再 `push_frame` | 小图编码成本 |
| 抓拍/快照 | keep-fd 帧 `to_jpeg_bytes()`（daemon 硬件编码腿） | ≈ 0 |

## 与硬件路线图的关系

SDK 的底层能力（DSP 路由、零拷贝、降级统计）遵循
`docs/proposals/hardware-first-roadmap.md` 与 `sdk-hardware-routing.md`；
本指南描述的是这些能力在应用层的正确组合方式。`AccelRouter` 的
`RoutePolicy.SOFTWARE_ONLY / HARDWARE_ONLY` 可用于基准测试与"零 CPU"验收。

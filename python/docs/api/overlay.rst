AI 叠加层 API
=============

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

使用示例
--------

启用叠加层
~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import OverlayClient

   overlay = OverlayClient()

   # 启用硬件叠加层：在视频上绘制检测框 + 标签 + 置信度
   overlay.enable(show_label=True, show_confidence=True, line_thickness=2)

   # 关闭
   overlay.disable()

自定义样式
~~~~~~~~~~

.. code-block:: python

   # configure 一次设置全部样式参数
   overlay.configure(
       enabled=True,
       show_label=True,
       show_confidence=False,
       line_thickness=3,
       box_color=0x00FF00,     # 绿色框
       label_color=0xFFFFFF,   # 白色标签
       font_size=16,
   )

结构化配置
~~~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import OverlayClient, OverlayConfig

   config = OverlayConfig(
       enabled=True,
       show_label=True,
       show_confidence=True,
       line_thickness=2,
   )
   OverlayClient().apply(config)

严格对帧模式（P1-6）
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 默认是"实时预览"语义：烘焙用的是有效期内的最新结果，框与画面不
   # 保证同帧。严格对帧是 opt-in：configure 打开后，daemon 在烘焙点
   # （编码前）为本帧等待它自己的推理结果——等待有上限，超限降级为
   # 最新结果并打日志/计数，编码输出永不因等待而停住
   overlay.configure(strict_frame_lock=True)              # 打开（cap 按 fps 推导）
   overlay.configure(strict_frame_lock=True,
                     strict_wait_cap_ms=0)                # 0 = 按流 fps 推导（2 帧周期）
   overlay.configure(strict_frame_lock=True,
                     strict_wait_cap_ms=66)               # 显式 66ms 上限
   overlay.configure(strict_frame_lock=False)             # 显式退出严格模式

   # 两个参数都是"可选字段"语义：不传 = 保持 daemon 当前设置；
   # enable()/disable() 与不带参数的 configure() 不会改动严格模式

   # 严格模式只作用于"自喂"显示流——stream_map 里 display 流的推理结果
   # 来自它自己（如 main:main）。交叉喂流的流（如 third:main）保持实时
   # 预览语义

与推理结果联动（annotate）
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # annotate() 把检测结果经事件总线推给 camera-daemon 的叠加渲染器：
   # app 不接触视频帧，检测框在编码前由 daemon 画到码流上。结果
   # 500ms 过期，按推理节奏持续调用；发空列表清屏
   from neoruntime_ipc_sdk import OverlayClient

   overlay = OverlayClient()
   overlay.enable()

   for result in inference_results:      # 如 InferenceClient.subscribe(...)
       overlay.annotate("main", result.objects)

   overlay.annotate("main", [])          # 清屏

其他结果类型
~~~~~~~~~~~~

.. code-block:: python

   # annotate_result 按 objects > classifications > landmarks >
   # ocr_lines 的优先级取 InferenceResult 中已填充的那一段
   overlay.annotate_result("main", result)

生命周期会话标记（P2-13）
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # annotate / annotate_result / 多边形写入都接受 session_id：同一会话
   # 的绘制带同一标记。推理会话结束（含进程被 SIGKILL 后连接断开）时，
   # daemon 按 session/end 事件把带该标记的所有 polygon 一次性清掉；
   # 检测框本身随结果 TTL 自然过期
   overlay.annotate("main", result.objects, session_id="app-1")
   overlay.annotate("main", polygons=zones, session_id="app-1")

   # 不传 session_id = 运维写入：无条件生效、不参与会话清理。
   # StreamPipeline 内部即按此区分——结果绘制带会话标记，启动时的
   # 静态 zone 与 stop() 的清屏写入不带标记

两种模式与视频副作用契约
~~~~~~~~~~~~~~~~~~~~~~~~

SDK 有两条互不干扰的取数路径，契约上明确区分：

**只取结果（零视频副作用）**：``InferenceClient.subscribe(...)`` 只
返回推理结果，不触碰视频通路。平台上自己的模型（ai-runtime /
auto-infer 发布的结果）默认也**不会**改变码流——除非运维在 daemon
配置里显式绑定（见下节）。``annotate()`` 主动画框不算副作用违规：
它是应用显式下发的绘制指令。

**自取帧自画（keep_fd / copy）**：``FdMediaClient`` 的 keep_fd / copy
模式取到的是**干净帧**（未经叠加层烘焙）。应用在自己的缓冲上组
合：copy 帧直接改写即可；keep_fd 帧经 DSP_IMPORT 进应用自有的 DSP
池做 RESIZE→BLEND（P1-9 池驻留后每调用开销可接受）。
**keep_fd 帧永不作为 DSP 写目标或 blend base**——平台侧按只读
契约执法（写入即拒绝），测试 `dsp-import-write-target-test` 守卫。

平台结果绑定（daemon 配置）
~~~~~~~~~~~~~~~~~~~~~~~~~~~

平台模型的检测结果默认不画（行为解耦）：一个裸 ``subscribe()`` 不
应该改变任何视频输出。需要平台框时，在 daemon 配置 yaml 的
``[ai_overlay]`` 段显式绑定：

.. code-block:: yaml

   [ai_overlay]
   # 推理流:显示流，方向与 stream_map 相同；逗号分隔多个绑定
   bindings: "main:main,third:main"
   # 迁移开关：恢复旧行为（所有平台结果自动画 + 默认 stream_map 回填）
   legacy_auto_bind: 0

**迁移说明**：``legacy_auto_bind`` 默认关闭。升级后若发现平台框
消失，两种恢复方式任选——(1) 按上例把需要的流加进 ``bindings``
（推荐：只开需要的流）；(2) 设 ``legacy_auto_bind: 1`` 完整恢复
旧行为（所有平台结果自动绑定，含默认 stream_map 回填）。应用经
``annotate()`` 下发的绘制不受此开关影响，永远生效。

帧同步绑定（frame_sequence / stream_epoch）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # annotate / annotate_result 接受两个可选的帧同步参数（都走事件
   # metadata，不进 payload；不传 = 不绑定，保持既有语义）
   #
   # frame_sequence：把这次绘制绑定到该帧——只在烘焙点走到该帧附近
   #   （+2 帧容差）内绘制，过后过期；帧已被流水线越过的指令会被
   #   拒收（overlay_late_commands 计数）。subscribe() 结果里的
   #   result.frame_sequence 就是天然来源
   # stream_epoch：流代计数（get_stream_status() 可读）。重配置
   #   （ReconfigureEncoder / 分辨率、变换全量重建）会 bump 代数并
   #   清空持有的图层；携带旧代数的事件被拒收（overlay_epoch_rejects
   #   计数），不再化作幽灵框
   for result in stream.subscribe(model):
       overlay.annotate_result("main", result,
                               frame_sequence=result.frame_sequence,
                               stream_epoch=epoch)

   # 重配置后的自愈：读到 epoch 变了就换用新值继续发布
   status = {s.stream_id: s for s in camera.get_stream_status()}
   epoch = status["main"].stream_epoch

keep_fd 帧的只读配方
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # copy 模式：帧已是应用自己的内存，直接改写
   frame = media.get_frame(stream, copy=True)
   draw_boxes(frame, results)               # 随便画

   # keep_fd 模式：dma-buf 只读。经 DSP_IMPORT 导入应用自有的 DSP
   # 池，RESIZE 到目标尺寸后作为 BLEND 的 overlay 层叠加到应用自己
   # 的底图上；帧本体永不作为写目标——RESIZE 的 dst / BLEND 的
   # base 若指向 keep_fd 帧都会被平台拒绝（平台侧执法，非约定）
   dsp.import_buffer(frame.fds, frame_id=frame.frame_id)
   dsp.schedule(op=DSP_OP_RESIZE, src=imported, dst=scratch)    # 自有 scratch
   dsp.schedule(op=DSP_OP_BLEND, base=composed, overlay=scratch)  # 自有底图
   # 简言之：keep_fd 帧 = BLEND 的 overlay 输入或 RESIZE 的 src，
   #           自有缓冲 = 一切写目标

上下文管理器
~~~~~~~~~~~~

.. code-block:: python

   with OverlayClient() as overlay:
       overlay.enable()

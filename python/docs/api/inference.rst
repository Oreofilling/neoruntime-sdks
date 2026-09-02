AI 推理 API
===========

.. automodule:: neoruntime_ipc_sdk.inference
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

InferenceClient
---------------

.. autoclass:: neoruntime_ipc_sdk.InferenceClient
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

数据类型
--------

InferenceResult
~~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.InferenceResult
   :members:
   :undoc-members:
   :no-index:

DetectedObject
~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.DetectedObject
   :members:
   :undoc-members:

BoundingBox
~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.BoundingBox
   :members:
   :undoc-members:
   :no-index:

LandmarkPoint
~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.LandmarkPoint
   :members:
   :undoc-members:

LandmarkSet
~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.LandmarkSet
   :members:
   :undoc-members:

SegmentationMask
~~~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.SegmentationMask
   :members:
   :undoc-members:

OcrLine
~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.OcrLine
   :members:
   :undoc-members:

Embedding
~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.Embedding
   :members:
   :undoc-members:

DepthMap
~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.DepthMap
   :members:
   :undoc-members:

Classification
~~~~~~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.Classification
   :members:
   :undoc-members:

ModelInfo
~~~~~~~~~

.. autoclass:: neoruntime_ipc_sdk.ModelInfo
   :members:
   :undoc-members:

使用示例
--------

单次推理
~~~~~~~~

.. code-block:: python

   from neoruntime_ipc_sdk import InferenceClient
   import numpy as np

   inf = InferenceClient()

   # 准备图像 (numpy array)
   image = np.zeros((1080, 1920, 3), dtype=np.uint8)

   # 执行推理
   result = inf.infer(image, model_id="person_v1")

   # 处理结果
   for obj in result.objects:
       print(f"{obj.label}: {obj.score:.2f}")
       print(f"  位置: ({obj.bbox.x}, {obj.bbox.y})")
       print(f"  大小: {obj.bbox.width}x{obj.bbox.height}")

   # 便捷方法
   if result.has_person():
       print("检测到人员")

   person_count = result.count_by_label("person")
   print(f"人员数量: {person_count}")

   persons = result.get_objects_by_label("person")

流式推理
~~~~~~~~

.. code-block:: python

   # 订阅视频流推理结果
   for frame_seq, result in inf.subscribe(
       stream="cam0_main",
       model="person_v1",
       fps=15
   ):
       print(f"帧 {frame_seq}: 检测到 {len(result.objects)} 个对象")

       for obj in result.objects:
           if obj.score > 0.8:
               print(f"  高置信度: {obj.label} ({obj.score:.2f})")

张量推理
~~~~~~~~

.. code-block:: python

   import numpy as np

   inf = InferenceClient()

   # 准备输入张量
   input1 = np.random.randn(1, 3, 224, 224).astype(np.float32)
   input2 = np.random.randn(1, 3, 112, 112).astype(np.float32)

   # 执行推理
   outputs = inf.infer_with_tensors(
       model_id="custom_model",
       inputs=[input1, input2],
       input_names=["input_main", "input_sub"]
   )

   # 处理输出张量
   for i, output in enumerate(outputs):
       print(f"输出 {i}: shape={output.shape}")

模型管理
~~~~~~~~

.. code-block:: python

   # 列出所有模型
   models = inf.list_models()
   for model in models:
       print(f"模型ID: {model.model_id}")
       print(f"路径: {model.model_path}")
       print(f"版本: {model.version}")
       print(f"输入: {model.inputs}")
       print(f"输出: {model.outputs}")
       print(f"估算 TOPS: {model.estimated_tops}")
       print(f"估算内存: {model.estimated_memory} bytes")

   # 获取模型详情
   info = inf.get_model_info("person_v1")
   if info:
       print(f"模型ID: {info.model_id}")
       print(f"路径: {info.model_path}")
       print(f"版本: {info.version}")

   # 注册新模型
   model_id = inf.register_model(
       model_path="/opt/models/custom.hef",
       model_id="custom_v1"
   )
   print(f"注册模型 ID: {model_id}")

   # 注销模型
   inf.unregister_model("custom_v1")

获取统计信息
~~~~~~~~~~~~

.. code-block:: python

   stats = inf.get_stats()

   print(f"设备利用率: {stats['device_utilization']}%")
   print(f"设备温度: {stats['device_temperature']}°C")
   print(f"总内存: {stats['total_memory_bytes']} bytes")
   print(f"已用内存: {stats['used_memory_bytes']} bytes")

   for model_stat in stats['model_stats']:
       print(f"模型: {model_stat['model_id']}")
       print(f"  总推理次数: {model_stat['total_inferences']}")
       print(f"  总错误数: {model_stat['total_errors']}")
       print(f"  平均延迟: {model_stat['avg_latency_us']}us")
       print(f"  当前 QPS: {model_stat['current_qps']}")
       print(f"  队列深度: {model_stat['queue_depth']}")

会话管理
~~~~~~~~

.. code-block:: python

   # 创建会话
   session_id = inf.create_session(
       session_id="my_session",
       app_id="my_app",
       allowed_models=["person_v1", "car_v1"],
       max_qps=10,
       max_concurrent=2,
       priority=4
   )
   print(f"会话ID: {session_id}")

   # 使用会话进行推理
   result = inf.infer(image, model_id="person_v1", session_id=session_id)

   # 销毁会话
   inf.destroy_session(session_id)

处理不同类型的结果
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = inf.infer(image, model_id="face_detection")

   # 检测结果
   for obj in result.objects:
       # 边界框
       x1, y1, x2, y2 = obj.bbox.to_xyxy()
       print(f"边界框: ({x1}, {y1}) - ({x2}, {y2})")

   # 分类结果
   for cls in result.classifications:
       print(f"分类: {cls.type} - {cls.label}: {cls.confidence:.2f}")

   # 关键点
   for lm_set in result.landmarks:
       print(f"关键点集类型: {lm_set.type}")
       for point in lm_set.points:
           print(f"  点: ({point.x}, {point.y}), 置信度: {point.confidence}")

   # 原始输出
   if result.raw_outputs:
       print(f"原始输出数量: {len(result.raw_outputs)}")

   # 性能信息
   print(f"推理时间: {result.infer_time_us}us")
   print(f"排队时间: {result.queue_time_us}us")

上下文管理器
~~~~~~~~~~~~

.. code-block:: python

   # 使用上下文管理器自动管理连接
   with InferenceClient() as inf:
       result = inf.infer(image, model_id="person_v1")
       print(f"检测到 {len(result.objects)} 个对象")

错误处理
~~~~~~~~

.. code-block:: python

   from grpc import RpcError

   try:
       result = inf.infer(image, model_id="nonexistent_model")
   except RpcError as e:
       print(f"推理失败: {e.details()}")
   except RuntimeError as e:
       print(f"运行时错误: {e}")

分割结果
~~~~~~~~

.. code-block:: python

   result = inf.infer(image, model_id="segmentation_v1")

   for mask in result.masks:
       print(f"掩码: {mask.label} (置信度: {mask.confidence:.2f})")
       print(f"  边界框: ({mask.bbox.x}, {mask.bbox.y}, {mask.bbox.width}, {mask.bbox.height})")

       # 解码 RLE 掩码为 numpy bool 数组 (H x W)
       np_mask = mask.to_numpy_mask()
       print(f"  掩码尺寸: {np_mask.shape}, 像素数: {np_mask.sum()}")

OCR 结果
~~~~~~~~~~

.. code-block:: python

   result = inf.infer(image, model_id="ocr_v1")

   for line in result.ocr_lines:
       print(f"文本: '{line.text}' (置信度: {line.confidence:.2f})")
       print(f"  位置: ({line.bbox.x}, {line.bbox.y})")

嵌入 (CLIP 图像)
~~~~~~~~~~~~~~~~

.. code-block:: python

   result = inf.infer(image, model_id="clip_vit_b32")

   for emb in result.embeddings:
       print(f"嵌入维度: {emb.dim}")
       # 可用于相似度检索等场景
       import numpy as np
       vec = np.array(emb.data)
       print(f"  L2 范数: {np.linalg.norm(vec):.4f}")

CLIP 文本编码
~~~~~~~~~~~~~

.. code-block:: python

   # 通过 NPU 将文本编码为 CLIP 嵌入
   embedding = inf.encode_text("a person walking in the park")
   print(f"嵌入长度: {len(embedding)}")

   # 编码多段文本并计算相似度
   import numpy as np
   emb1 = np.array(inf.encode_text("a cat"))
   emb2 = np.array(inf.encode_text("a dog"))
   similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
   print(f"相似度: {similarity:.4f}")

深度估计
~~~~~~~~~~

.. code-block:: python

   result = inf.infer(image, model_id="depth_v1")

   for dm in result.depth_maps:
       print(f"深度图: {dm.width}x{dm.height}")
       print(f"  最小深度: {dm.data.min():.2f}, 最大深度: {dm.data.max():.2f}")

运行时更新后处理配置
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import json

   # 运行时更新 CLIP 文本提示词
   config = json.dumps({
       "prompts": ["a person", "a car", "a bicycle"],
       "score_threshold": 0.3
   })
   inf.update_postprocess_config("clip_vit_b32", config)

GenAI (LLM/VLM)
~~~~~~~~~~~~~~~

.. code-block:: python

   import json

   # 创建 GenAI 会话
   session_id = inf.genai_create_session(
       hef_path="/opt/models/llm.hef",
       kind="llm"
   )

   # 流式生成 token
   messages = [json.dumps({"role": "user", "content": "Hello, who are you?"})]
   full_response = ""
   for token in inf.genai_generate(
       session_id=session_id,
       messages=messages,
       max_tokens=256,
       temperature=0.7,
       do_sample=True
   ):
       print(token, end="", flush=True)
       full_response += token

   # VLM: 携带图像输入生成
   with open("image.jpg", "rb") as f:
       img_data = f.read()

   vlm_session = inf.genai_create_session("/opt/models/vlm.hef", kind="vlm")
   messages = [json.dumps({"role": "user", "content": "Describe this image."})]
   for token in inf.genai_generate(
       session_id=vlm_session,
       messages=messages,
       images=[img_data]
   ):
       print(token, end="", flush=True)

   # 中止进行中的生成
   inf.genai_abort(session_id)

   # 清理会话
   inf.genai_destroy_session(session_id)
   inf.genai_destroy_session(vlm_session)
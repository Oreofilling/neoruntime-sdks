# Python SDK 接口性能报告(设备实测)

- 日期: 2026-09-09
- 设备: Linux hailo15 5.15.32-yocto-standard-g7919a266e480 #1 SMP PREEMPT Thu Jun 18 14:27:53 UTC 2026 aarch64 aarch64
- SDK: 0.7.4(模块路径 `/data/venv-sdk/lib/python3.10/site-packages/neoruntime_ipc_sdk`)
- Python: 3.10.15
- 用例结论分布: {"PASS": 21, "SKIP-NA": 1, "KNOWN-ISSUE": 1}

## 方法

- 每项操作:预热丢弃(10%)后采样,默认 300 样本 × 3 轮,取 p50 中位轮为首数,附轮间离散度;
- 流式接口:固定时长消费,按 frame_seq/payload seq 连续性计丢帧,到达间隔分布反映抖动;
- 错误只计数不计时;错误率 >50% 的路径提前终止,避免烧预算;
- 异常标注阈值:p99/p50 > 8×(长尾)、err > 0.5%(错误率)、soak 速率衰减 > 10%;标注是提示,不是 PASS/FAIL;
- 环境注:perf 运行窗口内设备不应叠加其他负载(测量窗口由操作者确认)。

## P1 推理

| 接口/操作 | n | ok/err | p50 | p90 | p95 | p99 | max | 标注 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `list_models`  | 300 | 300/0 | 3.9ms | 5.4ms | 6.5ms | 8.4ms | 11.2ms | — |
| `get_model_info`  | 300 | 300/0 | 3.6ms | 5.0ms | 5.6ms | 7.7ms | 8.9ms | — |
| `get_stats`  | 300 | 300/0 | 536.4ms | 547.5ms | 549.2ms | 556.1ms | 10.55s | — |
| `infer_e2e` (轮间1.38%) | 300 | 300/0 | 13.4ms | 15.0ms | 16.0ms | 17.6ms | 18.2ms | — |
| `infer_batch_4`  | 100 | 100/0 | 34.0ms | 38.6ms | 40.8ms | 46.2ms | 48.8ms | — |
| `session_lifecycle`  | 50 | 50/0 | 7.0ms | 10.6ms | 11.5ms | 12.4ms | 12.9ms | — |


## P2 媒体与加速层

| 接口/操作 | n | ok/err | p50 | p90 | p95 | p99 | max | 标注 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `get_frame_main`  | 200 | 200/0 | 33.3ms | 34.9ms | 35.3ms | 36.8ms | 40.9ms | — |
| `frame_to_rgb`  | 30 | 30/0 | 7.5ms | 11.1ms | 12.7ms | 18.2ms | 20.0ms | — |
| `frame_resize_half`  | 30 | 30/0 | 8.9ms | 12.0ms | 12.7ms | 14.0ms | 14.6ms | — |
| `frame_to_jpeg85`  | 30 | 30/0 | 162.1ms | 163.9ms | 165.6ms | 167.1ms | 167.3ms | — |
| `ab_resize_nv12_default`  | 30 | 30/0 | 8.0ms | 43.4ms | 44.9ms | 50.6ms | 53.0ms | — |
| `ab_resize_nv12_swonly`  | 30 | 30/0 | 4.0ms | 5.4ms | 7.0ms | 9.6ms | 10.5ms | — |
| `ab_nv12_to_rgb_default`  | 30 | 30/0 | 12.1ms | 15.7ms | 18.5ms | 19.3ms | 19.4ms | — |
| `ab_nv12_to_rgb_swonly`  | 30 | 30/0 | 8.3ms | 11.9ms | 13.6ms | 16.0ms | 16.5ms | — |
| `ab_rgb_to_nv12_default`  | 30 | 30/0 | 55.7ms | 65.0ms | 67.0ms | 88.9ms | 97.6ms | — |
| `ab_rgb_to_nv12_swonly`  | 30 | 30/0 | 11.5ms | 13.9ms | 14.1ms | 14.3ms | 14.3ms | — |
| `ab_encode_jpeg_default`  | 30 | 30/0 | 205.8ms | 225.1ms | 255.7ms | 281.5ms | 291.4ms | — |
| `ab_encode_jpeg_swonly`  | 30 | 30/0 | 158.3ms | 159.0ms | 159.6ms | 160.0ms | 160.2ms | — |


### accel 路由 A/B(默认策略 vs SOFTWARE_ONLY)

| 操作 | 默认 p50 | 仅软件 p50 | 默认/软件 |
|---|---:|---:|---:|
| `encode_jpeg` | 205.8ms | 158.3ms | 1.3 |
| `nv12_to_rgb` | 12.1ms | 8.3ms | 1.46 |
| `resize_nv12` | 8.0ms | 4.0ms | 2.0 |
| `rgb_to_nv12` | 55.7ms | 11.5ms | 4.86 |

路由决策证据(同一次运行):
- probes: `{"cv2": true, "dsp": true}`
- health: `{"policy": "prefer_hardware", "ops": {"draw_detections": {"hardware_calls": 0, "software_calls": 0, "fallbacks": 0, "backend": "hardware"}, "encode_jpeg": {"hardware_calls": 0, "software_calls": 0, "fallbacks": 0, "backend": "hardware"}, "nms": {"hardware_calls": 0, "software_calls": 0, "fallbacks": 0, "backend": "software"}, "nv12_to_rgb": {"hardware_calls": 0, "software_calls": 0, "fallbacks": 0, "backend": "hardware"}, "resize_nv12": {"hardware_calls": 0, "software_calls": 0, "fallbacks": 0, "backend": "hardware"}, "rgb_to_nv12": {"hardware_calls": 0, "software_calls": 0, "fallbacks": 0, "backend": "hardware"}}, "recent_degradations": []}`
- `draw_detections` → {"op": "draw_detections", "backend": "hardware", "provider": "_draw_detections_hw", "reason": "hardware leg registered"}
- `encode_jpeg` → {"op": "encode_jpeg", "backend": "hardware", "provider": "_encode_jpeg_hw", "reason": "hardware leg registered"}
- `nms` → {"op": "nms", "backend": "software", "provider": "nms", "reason": "HEF-integrated hardware NMS already suppressed pre-app; runtime detection_threshold via InferenceClient.update_postprocess_config (family functions only)"}
- `nv12_to_rgb` → {"op": "nv12_to_rgb", "backend": "hardware", "provider": "_nv12_to_rgb_hw", "reason": "hardware leg registered"}
- `resize_nv12` → {"op": "resize_nv12", "backend": "hardware", "provider": "_resize_nv12_hw", "reason": "hardware leg registered"}
- `rgb_to_nv12` → {"op": "rgb_to_nv12", "backend": "hardware", "provider": "_rgb_to_nv12_hw", "reason": "hardware leg registered"}
注:比值≈1.0 表示两侧执行同一软件腿(记录的是路由决策而非加速比);比值>1 表示硬件路由腿真实参与且比软件腿更慢。硬件腿是 DSP daemon 的 UDS 调用,零拷贝 dma-buf 导入仅对 keep-fd 帧源生效——本 A/B 输入为普通数组,比值主要计价每次调用的像素 socket 传输,属路由质量发现(数组输入场景),非 DSP 算力结论。

## P3 事件

| 接口/操作 | n | ok/err | p50 | p90 | p95 | p99 | max | 标注 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `publish`  | 200 | 200/0 | 3.2ms | 4.3ms | 4.6ms | 7.2ms | 9.4ms | — |
| `publish_batch_100`  | 10 | 10/0 | 109.5ms | 127.4ms | 129.6ms | 131.2ms | 131.7ms | — |
| `delivery_e2e`  | 100 | 100/0 | 4.0ms | 4.9ms | 5.5ms | 6.1ms | 6.2ms | — |
| `get_topic_stats`  | 50 | 50/0 | 2.0ms | 2.8ms | 3.2ms | 3.7ms | 3.8ms | — |

| 流 | 帧 | 时长 | fps | 丢帧 | 丢帧率 | 间隔p50 | 间隔p95 | 间隔max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arrival_10hz` | 300 | 30.0s | 10.0 | 0 | 0.0% | 100.0ms | 100.9ms | 101.8ms |


## P4 设备面

| 接口/操作 | n | ok/err | p50 | p90 | p95 | p99 | max | 标注 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `get_lens_status`  | 200 | 200/0 | 8.2ms | 11.8ms | 13.5ms | 203.4ms | 2.20s | 长尾(p99/p50=25×) |
| `get_autofocus_status`  | 100 | 100/0 | 3.4ms | 4.3ms | 4.5ms | 5.3ms | 7.1ms | — |
| `get_capabilities`  | 100 | 100/0 | 1.9ms | 2.8ms | 3.8ms | 5.9ms | 7.6ms | — |
| `get_sensor_info`  | 100 | 100/0 | 6.5ms | 7.5ms | 7.9ms | 8.9ms | 9.6ms | — |
| `get_stream_status`  | 200 | 200/0 | 2.9ms | 4.0ms | 4.5ms | 6.0ms | 8.1ms | — |
| `get_hardware_status`  | 100 | 100/0 | 16.4ms | 17.7ms | 17.8ms | 18.6ms | 19.6ms | — |
| `get_infrared_status`  | 100 | 100/0 | 2.2ms | 3.2ms | 3.7ms | 4.7ms | 4.7ms | — |

| 流 | 帧 | 时长 | fps | 丢帧 | 丢帧率 | 间隔p50 | 间隔p95 | 间隔max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `device_event_arrivals` | 0 | 30s | 0.0 | — | — | — | — | — |


## P5 长稳


### 长稳结果

- 模式: **media-only (infer degraded)**(infer 冒烟失败:RuntimeError: Inference failed: Inference failed: -2799)
- 时长 1800.0s / 迭代 3600 / 错误 0
- 速率: 首 2.02/s → 末 2.0/s(衰减 0.85%,阈值 10.0%)
- 客户端 RSS: 75.5MiB → 89.0MiB(Δ 13.5MiB); fd 15 → 16

| daemon | RSS Δ |
|---|---:|
| ai-runtime | 1.9MiB |
| app-manager | 0KiB |
| camera-daemon | 2.0MiB |
| event-bus | 52KiB |

分桶序列(客户端 RSS KiB / 迭代数):
```
t=    0.0s  rss=    77264  iters=0
t=   60.0s  rss=    90656  iters=121
t=  120.5s  rss=    90656  iters=242
t=  181.0s  rss=    90656  iters=363
t=  241.5s  rss=    90716  iters=484
t=  302.0s  rss=    91056  iters=605
t=  362.5s  rss=    91056  iters=726
t=  423.0s  rss=    91056  iters=847
t=  483.5s  rss=    91056  iters=968
t=  544.0s  rss=    91064  iters=1089
t=  604.5s  rss=    91064  iters=1210
t=  665.0s  rss=    91064  iters=1331
t=  725.5s  rss=    91068  iters=1452
t=  786.0s  rss=    91076  iters=1573
t=  846.5s  rss=    91076  iters=1694
t=  907.0s  rss=    91076  iters=1815
t=  967.5s  rss=    91080  iters=1936
t= 1028.0s  rss=    91080  iters=2057
t= 1088.5s  rss=    91080  iters=2178
t= 1149.0s  rss=    91084  iters=2299
t= 1209.5s  rss=    91084  iters=2420
t= 1270.0s  rss=    91088  iters=2541
t= 1330.5s  rss=    91092  iters=2662
t= 1391.0s  rss=    91092  iters=2783
t= 1451.5s  rss=    91092  iters=2904
t= 1512.0s  rss=    91092  iters=3025
t= 1572.5s  rss=    91096  iters=3146
t= 1633.0s  rss=    91096  iters=3267
t= 1693.5s  rss=    91096  iters=3388
t= 1754.0s  rss=    91104  iters=3509
t= 1800.0s  rss=    91108  iters=3600
```

### NA(前置缺失/路径不可用)

- `test_60_perf_inference.T03InferDataPlane.test_03_subscribe_stream`: subscribe path dead on this deployment: RuntimeError: Stream inference failed 10 consecutive times (stream='third', model='sdk-perf-yolo', last frame=401649): 'Inference failed: -2814'

### KNOWN-ISSUE(已知缺陷,非本次新失败)

- `test_63_perf_device.T01DeviceStatus.test_01_get_device_status`: SDK 0.7.4 defect (same as test_40): device.py reads response.ir_led_on unconditionally but no bundled proto message carries the field — every call raises AttributeError regardless of daemon health. The probe below raises through; once the SDK guards the field, sampling starts on its own. — `AttributeError: ir_led_on`

---
回滚:测试用 SDK 经 wheel 强装到设备 venv,恢复用
`/data/venv-sdk/bin/pip install --no-deps neoruntime-ipc-sdk==0.6.0`(见 scripts/run-device-tests.sh 头注)。

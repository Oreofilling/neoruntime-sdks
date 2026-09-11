#!/usr/bin/env python3
"""
本地推理管线示例（B 形态：客户端预处理 + 推理 + 客户端后处理 + Web 预览）

适用场景：自定义模型（服务端不做后处理的通用导出）、需要自己调阈值、
或需要拿到中间图像做展示的应用。如果模型走 family 后处理且只需要结果，
请用 InferenceClient.subscribe —— 一个像素都不进应用（见 python/PERFORMANCE.md）。

管线：
- 订阅 sub 流（低分辨率流做 AI，省 CPU 与上送带宽），keep_fd=True
- Preprocessor.from_model 自动推导模型输入规格，缩放走 DSP（零拷贝）
- InferencePipeline.run: 预处理 → 推理 → YOLOv8 解码（阈值自己定）
- 检测框画在小图上，MJPEG 推流到 http://<device>:8080/
"""

import argparse
import signal
import sys
import time

from neoruntime_ipc_sdk import (
    Config,
    FdMediaClient,
    InferenceClient,
    InferencePipeline,
    MjpegServer,
    MjpegStream,
    Preprocessor,
    YoloV8Postprocessor,
    draw_detections,
    get_default_router,
)


class LocalInferenceApp:
    def __init__(self, model_id, stream_id, port, labels):
        self.running = True
        self.app_id = Config.get_app_id()
        self.model_id = model_id
        self.stream_id = stream_id
        self.labels = labels

        self.media = FdMediaClient()
        self.inference = InferenceClient()
        self.source = MjpegStream()
        self.server = MjpegServer(port=port, source=self.source)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\n[{self.app_id}] Received signal {signum}, shutting down...")
        self.running = False

    def _build_pipeline(self):
        """from_model 从 get_model_info 推导输入尺寸/布局，无需手配 640x640。"""
        pre = Preprocessor.from_model(self.inference, self.model_id)
        print(
            f"[{self.app_id}] model {self.model_id}: input "
            f"{pre.size} {pre.layout} normalize={pre.normalize}"
        )
        post = YoloV8Postprocessor(
            labels=self.labels,
            score_threshold=0.3,   # 自己的阈值 —— 通用导出模型在服务端改不动
            iou_threshold=0.45,
        )
        return InferencePipeline(
            client=self.inference, model_id=self.model_id,
            preprocessor=pre, postprocessor=post,
        )

    def run(self):
        pipeline = self._build_pipeline()
        self.server.start()
        print(f"[{self.app_id}] MJPEG preview on http://0.0.0.0:{self.server.port}/")
        print(f"[{self.app_id}] subscribing to stream {self.stream_id!r} (keep_fd)")

        frames = 0
        window_start = time.time()
        try:
            # keep_fd=True: 帧的 dma-buf 直接交给 DSP 缩放（零拷贝），
            # 只有缩放后的小图会被拷回 CPU。
            for frame in self.media.subscribe(self.stream_id, keep_fd=True):
                if not self.running:
                    break

                out = pipeline.run(frame)

                # 在模型输入尺寸的小图上画框：out.objects 已还原到源帧坐标，
                # 用 meta 正向映射回输入坐标即可（dst = src * scale + origin）
                annotated = draw_detections(
                    out.tensor, [_to_input_box(o, out.meta) for o in out.objects]
                )
                self.source.push_frame(_as_frame(annotated))

                frames += 1
                elapsed = time.time() - window_start
                if elapsed >= 5.0:
                    degradations = sum(
                        c.get("fallbacks", 0)
                        for c in get_default_router().health()["ops"].values()
                    )
                    print(
                        f"[{self.app_id}] {frames / elapsed:.1f} fps | "
                        f"{len(out.objects)} objects | "
                        f"infer {out.result.infer_time_us / 1000:.1f} ms | "
                        f"pipeline {out.latency_ms:.1f} ms | "
                        f"hw fallbacks {degradations}"
                    )
                    frames = 0
                    window_start = time.time()
        finally:
            self.cleanup()

    def cleanup(self):
        self.server.stop()
        self.media.close()
        self.inference.close()
        print(f"[{self.app_id}] cleaned up")


def _to_input_box(obj, meta):
    """源帧坐标 → 模型输入坐标（正向映射 dst = src * scale + origin）。"""
    from neoruntime_ipc_sdk import BoundingBox, DetectedObject

    x1 = obj.bbox.x * meta.scale[0] + meta.origin[0]
    y1 = obj.bbox.y * meta.scale[1] + meta.origin[1]
    x2 = (obj.bbox.x + obj.bbox.width) * meta.scale[0] + meta.origin[0]
    y2 = (obj.bbox.y + obj.bbox.height) * meta.scale[1] + meta.origin[1]
    return DetectedObject(
        label=obj.label, score=obj.score, class_id=obj.class_id,
        bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
    )


def _as_frame(array):
    """把预处理输出的小图包回 Frame，供 MjpegStream 编码 JPEG。"""
    from neoruntime_ipc_sdk import Frame

    h, w = array.shape[:2]
    return Frame(sequence=0, timestamp_ns=0, width=w, height=h,
                 format="RGB", image=array)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="person_v1", help="registered model id")
    parser.add_argument("--stream", default="sub", help="stream id (prefer 'sub')")
    parser.add_argument("--port", type=int, default=8080, help="MJPEG port")
    parser.add_argument("--labels", default="person,car", help="comma-separated labels")
    args = parser.parse_args()

    app = LocalInferenceApp(
        model_id=args.model,
        stream_id=args.stream,
        port=args.port,
        labels=[x.strip() for x in args.labels.split(",") if x.strip()],
    )
    app.run()


if __name__ == "__main__":
    sys.exit(main())

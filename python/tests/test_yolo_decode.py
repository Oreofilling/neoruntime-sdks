"""YOLO decode adapters (ONNX-standard export shapes, synthetic tensors)."""

import numpy as np
import pytest

from neoruntime_ipc_sdk import PreprocessMeta, YoloV5Postprocessor, YoloV8Postprocessor

NC = 2  # person, car


def make_v8_head(entries, n=200):
    """entries: list of (col, (cx, cy, w, h), score, class_id)."""
    arr = np.zeros((4 + NC, n), np.float32)
    for col, (cx, cy, w, h), score, cls in entries:
        arr[0, col], arr[1, col], arr[2, col], arr[3, col] = cx, cy, w, h
        arr[4 + cls, col] = score
    return arr


def make_v5_head(entries, n=200):
    """entries: list of (row, (cx, cy, w, h), objectness, class_score, class_id)."""
    arr = np.zeros((n, 5 + NC), np.float32)
    for row, (cx, cy, w, h), obj, cls_score, cls in entries:
        arr[row, :4] = cx, cy, w, h
        arr[row, 4] = obj
        arr[row, 5 + cls] = cls_score
    return arr


META = PreprocessMeta(original_size=(1920, 1080), input_size=(640, 640),
                      scale=(640 / 1920, 360 / 1080), origin=(0, 140))


class TestYoloV8:
    def test_decode_threshold_and_nms(self):
        head = make_v8_head([
            (0, (320, 300, 100, 80), 0.90, 0),   # keep
            (1, (322, 301, 100, 80), 0.80, 0),   # near-duplicate → suppressed
            (2, (100, 100, 50, 50), 0.05, 1),    # below threshold
        ])
        post = YoloV8Postprocessor(labels=["person", "car"],
                                   score_threshold=0.3, iou_threshold=0.45)
        objects = post([head], meta=None)
        assert len(objects) == 1
        obj = objects[0]
        assert obj.label == "person" and obj.score == pytest.approx(0.9)
        assert obj.class_id == 0
        # input-coordinate xyxy: (270, 260) - (370, 340)
        assert obj.bbox.x == pytest.approx(270) and obj.bbox.y == pytest.approx(260)
        assert obj.bbox.width == pytest.approx(100) and obj.bbox.height == pytest.approx(80)

    def test_batched_and_transposed_shapes(self):
        head = make_v8_head([(0, (320, 300, 100, 80), 0.9, 0)])
        post = YoloV8Postprocessor(score_threshold=0.3)
        assert len(post([head[None, :, :]])) == 1     # (1, C, N)
        assert len(post([head.T])) == 1               # (N, C) → transposed

    def test_meta_maps_back_to_source(self):
        head = make_v8_head([(0, (320, 300, 100, 80), 0.9, 0)])
        post = YoloV8Postprocessor(score_threshold=0.3)
        obj = post([head], meta=META)[0]
        # x_src = (x_in - origin) / scale = (270 - 0) / (1/3) = 810
        assert obj.bbox.x == pytest.approx(810, abs=1)
        # y_src = (260 - 140) / (1/3) = 360
        assert obj.bbox.y == pytest.approx(360, abs=1)

    def test_different_classes_do_not_suppress(self):
        head = make_v8_head([
            (0, (320, 300, 100, 80), 0.9, 0),
            (1, (321, 301, 100, 80), 0.8, 1),   # same place, other class → kept
        ])
        post = YoloV8Postprocessor(labels=["person", "car"], score_threshold=0.3)
        objects = post([head])
        assert sorted(o.label for o in objects) == ["car", "person"]

    def test_empty_and_malformed(self):
        post = YoloV8Postprocessor()
        assert post([]) == []
        assert post([np.zeros((0, 0), np.float32)]) == []
        with pytest.raises(ValueError, match="channels"):
            post([np.zeros((3, 10), np.float32)])  # 3 channels < 4+1


class TestYoloV5:
    def test_objectness_multiplies(self):
        head = make_v5_head([(0, (320, 300, 100, 80), 0.9, 0.95, 0)])
        post = YoloV5Postprocessor(labels=["person", "car"], score_threshold=0.5)
        objects = post([head])
        assert len(objects) == 1
        assert objects[0].score == pytest.approx(0.9 * 0.95)
        assert objects[0].bbox.width == pytest.approx(100)

    def test_transposed_shape(self):
        head = make_v5_head([(0, (320, 300, 100, 80), 0.9, 0.95, 0)])
        post = YoloV5Postprocessor(score_threshold=0.5)
        assert len(post([head.T])) == 1  # (C, N) → transposed to (N, C)

    def test_meta_mapping(self):
        head = make_v5_head([(0, (320, 300, 100, 80), 0.9, 0.95, 0)])
        post = YoloV5Postprocessor(score_threshold=0.5)
        obj = post([head], meta=META)[0]
        assert obj.bbox.x == pytest.approx(810, abs=1)
        assert obj.bbox.y == pytest.approx(360, abs=1)


class TestSharedBehaviour:
    def test_scores_sorted_descending(self):
        head = make_v8_head([
            (0, (320, 300, 100, 80), 0.6, 0),
            (1, (100, 100, 50, 50), 0.9, 0),   # far apart, no suppression
        ])
        objects = YoloV8Postprocessor(score_threshold=0.3)([head])
        assert [o.score for o in objects] == sorted((o.score for o in objects), reverse=True)

    def test_fallback_label_without_list(self):
        head = make_v8_head([(0, (320, 300, 100, 80), 0.9, 1)])
        obj = YoloV8Postprocessor(score_threshold=0.3)([head])[0]
        assert obj.label == "1"


class TestSourceClipping:
    """Review P2: boxes landing in the letterbox pad clip to the source frame."""

    def test_pad_boxes_clipped_or_dropped(self):
        # META: origin (0, 140), scale (1/3, 1/3), source 1920x1080.
        head = make_v8_head([
            (0, (320, 40, 100, 40), 0.90, 0),    # entirely in the top pad
            (1, (320, 200, 100, 200), 0.80, 0),  # straddles the content edge
        ])
        post = YoloV8Postprocessor(labels=["person"], score_threshold=0.3)
        objects = post([head], meta=META)

        assert len(objects) == 1                 # pad-only box dropped
        obj = objects[0]
        assert obj.bbox.y == 0.0                 # clipped at the top edge
        assert obj.bbox.x >= 0.0 and obj.bbox.y >= 0.0
        assert obj.bbox.x + obj.bbox.width <= 1920 + 1e-6

    def test_clip_opt_out_keeps_negative_coordinates(self):
        head = make_v8_head([(0, (320, 40, 100, 40), 0.9, 0)])
        post = YoloV8Postprocessor(labels=["person"], score_threshold=0.3, clip=False)
        obj = post([head], meta=META)[0]
        assert obj.bbox.y < 0                     # unclipped, as requested

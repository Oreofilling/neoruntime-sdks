"""
Tests for draw utilities: draw_boxes / draw_text / draw_detections
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from neoruntime_ipc_sdk.draw import draw_boxes, draw_detections, draw_text
from neoruntime_ipc_sdk.inference import BoundingBox, DetectedObject


def black_canvas(h=100, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDrawBoxes:
    def test_box_pixels_drawn(self):
        img = black_canvas()
        out = draw_boxes(img, [(10, 10, 60, 50)], color=(0, 255, 0), thickness=2)
        # top edge of the box is green
        assert (out[10, 12:58, 1] == 255).all()
        assert out.shape == img.shape

    def test_input_not_mutated(self):
        img = black_canvas()
        before = img.copy()
        draw_boxes(img, [(5, 5, 40, 40)])
        np.testing.assert_array_equal(img, before)

    def test_empty_boxes_returns_copy(self):
        img = black_canvas()
        out = draw_boxes(img, [])
        np.testing.assert_array_equal(out, img)
        assert out is not img

    def test_multiple_boxes(self):
        img = black_canvas()
        out = draw_boxes(img, [(0, 0, 20, 20), (40, 40, 80, 80)], color=(255, 0, 0))
        assert (out[0, 2:18, 0] == 255).all()
        assert (out[40, 42:78, 0] == 255).all()

    def test_bbox_object_accepted(self):
        img = black_canvas()
        bb = BoundingBox(x=10, y=10, width=50, height=40)
        out = draw_boxes(img, [bb], color=(0, 0, 255))
        assert (out[10, 12:58, 2] == 255).all()

    def test_label_drawn_near_box(self):
        img = black_canvas()
        out = draw_boxes(img, [(10, 30, 60, 80)], labels=["person"],
                         color=(0, 255, 0))
        # label region above the box top edge has non-zero pixels
        assert out[16:28, 10:60].any()

    def test_scores_accepted(self):
        img = black_canvas()
        out = draw_boxes(img, [(10, 30, 60, 80)], labels=["person"],
                         scores=[0.87], color=(0, 255, 0))
        assert out[10:28, 10:60].any()

    def test_without_cv2(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        img = black_canvas()
        out = draw_boxes(img, [(10, 10, 60, 50)], color=(0, 255, 0), thickness=2)
        assert (out[10, 12:58, 1] == 255).all()
        assert out.shape == img.shape


class TestDrawText:
    def test_text_pixels_drawn(self):
        img = black_canvas()
        out = draw_text(img, "hello", (5, 5), color=(255, 255, 255))
        assert out[5:20, 5:60].any()

    def test_input_not_mutated(self):
        img = black_canvas()
        before = img.copy()
        draw_text(img, "x", (5, 5))
        np.testing.assert_array_equal(img, before)

    def test_without_cv2(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        img = black_canvas()
        out = draw_text(img, "hello", (5, 5), color=(255, 255, 255))
        assert out[5:20, 5:60].any()


class TestDrawDetections:
    def _objects(self):
        return [
            DetectedObject(label="person", score=0.91,
                           bbox=BoundingBox(x=10, y=10, width=50, height=40),
                           class_id=0),
            DetectedObject(label="vehicle", score=0.75,
                           bbox=BoundingBox(x=80, y=50, width=40, height=30),
                           class_id=1),
        ]

    def test_detections_drawn_with_labels(self):
        img = black_canvas()
        out = draw_detections(img, self._objects())
        assert (out[10, 12:58] > 0).any()      # first box edge
        assert (out[50, 82:118] > 0).any()     # second box edge

    def test_accepts_result_like_object(self):
        img = black_canvas()
        result = SimpleNamespace(objects=self._objects())
        out = draw_detections(img, result)
        assert (out[10, 12:58] > 0).any()

    def test_input_not_mutated(self):
        img = black_canvas()
        before = img.copy()
        draw_detections(img, self._objects())
        np.testing.assert_array_equal(img, before)

    def test_distinct_class_colors(self):
        img = black_canvas()
        out = draw_detections(img, self._objects())
        # two class ids -> two different edge colors; both edges must be drawn
        edge1 = out[10, 30]
        edge2 = out[50, 100]
        assert edge1.tolist() != [0, 0, 0]
        assert edge2.tolist() != [0, 0, 0]

    def test_empty_detections(self):
        img = black_canvas()
        out = draw_detections(img, [])
        np.testing.assert_array_equal(out, img)

    def test_without_cv2(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        img = black_canvas()
        out = draw_detections(img, self._objects())
        assert (out[10, 12:58] > 0).any()

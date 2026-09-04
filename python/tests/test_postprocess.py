"""Tests for neoruntime_ipc_sdk.postprocess — greedy NMS."""

from __future__ import annotations

import numpy as np
import pytest

from neoruntime_ipc_sdk.postprocess import nms


def test_overlapping_boxes_keep_higher_score():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],   # ~68% overlap with box 0
            [50, 50, 60, 60],  # disjoint
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]


def test_disjoint_boxes_all_kept():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 0, 50, 10]], dtype=np.float32)
    scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert sorted(keep) == [0, 1, 2]
    # descending-score order
    assert keep[0] == 1


def test_different_classes_do_not_suppress_each_other():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1])
    keep = nms(boxes, scores, iou_threshold=0.5, class_ids=classes)
    assert sorted(keep) == [0, 1]


def test_same_class_still_suppresses():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([3, 3])
    keep = nms(boxes, scores, iou_threshold=0.5, class_ids=classes)
    assert keep == [0]


def test_empty_input():
    assert nms(np.zeros((0, 4)), np.zeros((0,))) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        nms(np.zeros((2, 4)), np.zeros((3,)))
    with pytest.raises(ValueError, match="mismatch"):
        nms(np.zeros((2, 4)), np.zeros((2,)), class_ids=np.zeros((5,)))


def test_bad_threshold_raises():
    boxes = np.zeros((1, 4))
    scores = np.zeros(1)
    with pytest.raises(ValueError, match="iou_threshold"):
        nms(boxes, scores, iou_threshold=1.5)


def test_touching_boxes_have_zero_iou():
    # side-by-side, sharing an edge: inter == 0 -> both survive
    boxes = np.array([[0, 0, 10, 10], [10, 0, 20, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.0)
    assert sorted(keep) == [0, 1]

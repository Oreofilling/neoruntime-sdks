// postprocess.cpp — greedy NMS + YOLO head decoders (port of
// postprocess.py, 0.8.1). Error messages reproduce the python f-strings
// byte for byte (py_float renders floats python-style: 1.5 -> "1.5",
// 1.0 -> "1.0").
#include "neoruntime_ipc_sdk/postprocess.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace neoruntime_ipc_sdk {

namespace {

// Python float repr (shortest round-trip is close enough at test
// tolerances; ensures 1.0 prints "1.0" not "1").
std::string py_float(double v) {
    std::ostringstream out;
    out << v;
    std::string s = out.str();
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find("inf") == std::string::npos && s.find("nan") == std::string::npos)
        s += ".0";
    return s;
}

// numpy-style shape tuple of a Mat: "(rows, cols)" / "(rows, cols, ch)" /
// "(d0, d1, ...)" for >2 dims.
std::string head_shape_text(const cv::Mat& m) {
    std::ostringstream out;
    out << "(";
    if (m.dims <= 2) {
        out << m.rows << ", " << m.cols;
        if (m.channels() > 1) out << ", " << m.channels();
    } else {
        for (int i = 0; i < m.dims; ++i) {
            if (i > 0) out << ", ";
            out << m.size.p[i];
        }
    }
    out << ")";
    return out.str();
}

// python _squeeze_batch: drop leading size-1 (batch) dimensions down to a
// 2D head matrix. Decoded tensors are contiguous single-channel Mats.
cv::Mat squeeze_batch(const cv::Mat& in) {
    cv::Mat m = in;
    while (m.dims > 2 && m.size.p[0] == 1) {
        int rows = m.size.p[1];
        long long cols = 1;
        for (int d = 2; d < m.dims; ++d) cols *= m.size.p[d];
        m = cv::Mat(rows, static_cast<int>(cols), m.type(), m.data);
    }
    return m;
}

// Float element access (float32 heads standard; float64 tolerated).
float at_f(const cv::Mat& m, int r, int c) {
    if (m.depth() == CV_32F) return m.at<float>(r, c);
    if (m.depth() == CV_64F) return static_cast<float>(m.at<double>(r, c));
    throw std::invalid_argument("expected a float head matrix, got depth " +
                                std::to_string(m.depth()));
}

}  // namespace

std::vector<int> nms(const std::vector<std::array<float, 4>>& boxes,
                     const std::vector<float>& scores, double iou_threshold,
                     const std::vector<int>* class_ids) {
    const std::size_t n = boxes.size();
    if (n != scores.size())
        throw std::invalid_argument("boxes (" + std::to_string(n) + ") and scores (" +
                                    std::to_string(scores.size()) + ") length mismatch");
    if (n == 0) return {};
    if (!(iou_threshold >= 0.0 && iou_threshold <= 1.0))
        throw std::invalid_argument("iou_threshold must be in [0, 1], got " +
                                    py_float(iou_threshold));

    // Pairwise IoU (python _iou_matrix, computed row-major here).
    std::vector<float> area(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& b = boxes[i];
        area[i] = std::max(b[2] - b[0], 0.0f) * std::max(b[3] - b[1], 0.0f);
    }
    std::vector<float> iou(n * n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& a = boxes[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            const auto& b = boxes[j];
            const float iw = std::min(a[2], b[2]) - std::max(a[0], b[0]);
            const float ih = std::min(a[3], b[3]) - std::max(a[1], b[1]);
            const float inter = std::max(iw, 0.0f) * std::max(ih, 0.0f);
            const float union_ = area[i] + area[j] - inter;
            const float v = union_ > 0.0f ? inter / union_ : 0.0f;
            iou[i * n + j] = v;
            iou[j * n + i] = v;
        }
    }
    if (class_ids != nullptr) {
        if (class_ids->size() != n)
            throw std::invalid_argument("class_ids (" + std::to_string(class_ids->size()) +
                                        ") and boxes (" + std::to_string(n) +
                                        ") length mismatch");
        // boxes of different classes never suppress each other
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if ((*class_ids)[i] != (*class_ids)[j]) iou[i * n + j] = 0.0f;
    }

    // Descending score; ties keep ascending index (deterministic order —
    // np.argsort(-scores) on the sizes these tests use).
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<char> suppressed(n, 0);
    std::vector<int> keep;
    for (const int idx : order) {
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        const float* row = &iou[static_cast<std::size_t>(idx) * n];
        for (std::size_t j = 0; j < n; ++j)
            if (row[j] > iou_threshold) suppressed[j] = 1;
        suppressed[idx] = 1;  // self, needed when iou_threshold == 1.0
    }
    return keep;
}

// ---- shared YOLO plumbing -------------------------------------------------------

YoloPostprocessorBase::YoloPostprocessorBase(std::vector<std::string> labels_in,
                                             double score_threshold_in,
                                             double iou_threshold_in, bool clip_in)
    : labels(std::move(labels_in)),
      score_threshold(score_threshold_in),
      iou_threshold(iou_threshold_in),
      clip(clip_in) {}

std::string YoloPostprocessorBase::label_for(int class_id) const {
    if (class_id < static_cast<int>(labels.size())) return labels[class_id];
    return std::to_string(class_id);
}

std::vector<DetectedObject> YoloPostprocessorBase::finish(
    const std::vector<std::array<float, 4>>& boxes_xyxy, const std::vector<float>& scores,
    const std::vector<int>& class_ids, const PreprocessMeta* meta) const {
    std::vector<std::array<float, 4>> boxes;
    std::vector<float> kept_scores;
    std::vector<int> kept_classes;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] >= score_threshold) {
            boxes.push_back(boxes_xyxy[i]);
            kept_scores.push_back(scores[i]);
            kept_classes.push_back(class_ids[i]);
        }
    }
    const std::vector<int> order = nms(boxes, kept_scores, iou_threshold, &kept_classes);

    std::vector<DetectedObject> objects;
    for (const int i : order) {
        double x1 = boxes[i][0], y1 = boxes[i][1], x2 = boxes[i][2], y2 = boxes[i][3];
        if (meta != nullptr) {
            const std::array<double, 4> mapped = meta->to_source_box({x1, y1, x2, y2});
            x1 = mapped[0];
            y1 = mapped[1];
            x2 = mapped[2];
            y2 = mapped[3];
            if (clip) {
                const double src_w = static_cast<double>(meta->original_size.first);
                const double src_h = static_cast<double>(meta->original_size.second);
                x1 = std::max(0.0, x1);
                y1 = std::max(0.0, y1);
                x2 = std::min(src_w, x2);
                y2 = std::min(src_h, y2);
                if (x2 - x1 <= 0 || y2 - y1 <= 0)
                    continue;  // entirely in the letterbox pad — nothing real
            }
        }
        DetectedObject obj;
        obj.label = label_for(kept_classes[i]);
        obj.score = kept_scores[i];
        obj.bbox.x = static_cast<float>(x1);
        obj.bbox.y = static_cast<float>(y1);
        obj.bbox.width = static_cast<float>(x2 - x1);
        obj.bbox.height = static_cast<float>(y2 - y1);
        obj.class_id = kept_classes[i];
        objects.push_back(std::move(obj));
    }
    return objects;
}

// ---- YOLOv8 ----------------------------------------------------------------------

std::vector<DetectedObject> YoloV8Postprocessor::operator()(
    const std::optional<std::vector<cv::Mat>>& raw_outputs,
    const PreprocessMeta* meta) const {
    if (!raw_outputs || raw_outputs->empty()) return {};
    cv::Mat arr = squeeze_batch((*raw_outputs)[0]);
    if (arr.dims != 2 || arr.channels() > 1)
        throw std::invalid_argument("expected a 2D head matrix, got shape " +
                                    head_shape_text(arr));
    if (arr.rows > arr.cols) arr = arr.t();  // the long axis is the anchor count N
    const int channels = arr.rows;
    const int n = arr.cols;
    if (channels == 0 || n == 0) return {};
    const int nc = channels - 4;
    if (nc <= 0)
        throw std::invalid_argument("head has " + std::to_string(channels) +
                                    " channels, need at least 5 (4 box + classes)");

    std::vector<std::array<float, 4>> boxes(static_cast<std::size_t>(n));
    std::vector<int> class_ids(static_cast<std::size_t>(n));
    std::vector<float> scores(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float cx = at_f(arr, 0, i);
        const float cy = at_f(arr, 1, i);
        const float w = at_f(arr, 2, i);
        const float h = at_f(arr, 3, i);
        boxes[static_cast<std::size_t>(i)] = {cx - w / 2.0f, cy - h / 2.0f, cx + w / 2.0f,
                                              cy + h / 2.0f};
        int best = 0;
        float best_score = at_f(arr, 4, i);
        for (int c = 1; c < nc; ++c) {
            const float s = at_f(arr, 4 + c, i);
            if (s > best_score) {
                best_score = s;
                best = c;
            }
        }
        class_ids[static_cast<std::size_t>(i)] = best;
        scores[static_cast<std::size_t>(i)] = best_score;
    }
    return finish(boxes, scores, class_ids, meta);
}

// ---- YOLOv5 ----------------------------------------------------------------------

std::vector<DetectedObject> YoloV5Postprocessor::operator()(
    const std::optional<std::vector<cv::Mat>>& raw_outputs,
    const PreprocessMeta* meta) const {
    if (!raw_outputs || raw_outputs->empty()) return {};
    cv::Mat arr = squeeze_batch((*raw_outputs)[0]);
    if (arr.dims != 2 || arr.channels() > 1)
        throw std::invalid_argument("expected a 2D head matrix, got shape " +
                                    head_shape_text(arr));
    if (arr.rows < arr.cols) arr = arr.t();  // the long axis is the anchor count N
    const int n = arr.rows;
    const int cols = arr.cols;
    if (n == 0 || cols == 0) return {};
    if (cols < 6)
        throw std::invalid_argument("head has " + std::to_string(cols) +
                                    " columns, need at least 6 (4 box + objectness + classes)");

    std::vector<std::array<float, 4>> boxes(static_cast<std::size_t>(n));
    std::vector<int> class_ids(static_cast<std::size_t>(n));
    std::vector<float> scores(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float cx = at_f(arr, i, 0);
        const float cy = at_f(arr, i, 1);
        const float w = at_f(arr, i, 2);
        const float h = at_f(arr, i, 3);
        boxes[static_cast<std::size_t>(i)] = {cx - w / 2.0f, cy - h / 2.0f, cx + w / 2.0f,
                                              cy + h / 2.0f};
        const float objectness = at_f(arr, i, 4);
        int best = 0;
        float best_score = at_f(arr, i, 5);
        for (int c = 1; c < cols - 5; ++c) {
            const float s = at_f(arr, i, 5 + c);
            if (s > best_score) {
                best_score = s;
                best = c;
            }
        }
        class_ids[static_cast<std::size_t>(i)] = best;
        scores[static_cast<std::size_t>(i)] = objectness * best_score;
    }
    return finish(boxes, scores, class_ids, meta);
}

}  // namespace neoruntime_ipc_sdk

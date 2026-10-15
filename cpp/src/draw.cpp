// draw.cpp — port of draw.py (0.8.1), cv2 legs only (OpenCV is a hard
// dependency of the C++ SDK; the Pillow fallbacks do not apply). Messages
// and geometry reproduce the python implementation exactly.
#include "neoruntime_ipc_sdk/draw.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/accel.hpp"

namespace neoruntime_ipc_sdk {

namespace {

constexpr Rgb kGreen = {0, 255, 0};

// _text_metrics() cv2 leg: (anchor_offset, canvas_row_floor,
// rows_below_anchor) — cv2 anchors text at the baseline (6 px above the
// box top, floored to canvas row 12, descender ~4 px below; 6 kept for
// margin).
constexpr int kTextOffset = 6;
constexpr int kTextFloor = 12;
constexpr int kTextBelow = 6;

std::string format_score(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return buf;
}

Box box_of(const BoundingBox& b) {
    return {b.x, b.y, b.x + b.width, b.y + b.height};
}

// python _text_row: canvas row for a box caption; nullopt = not this
// canvas's pixels. The single-canvas path keeps the legacy clamps; a
// fragment canvas anchors the caption to its frame row instead.
std::optional<int> text_row(int yi1, int y0, int ch, int offset, int floor_row, bool fragment) {
    int row = yi1 - offset - y0;
    if (!fragment) return std::min(std::max(row, floor_row), ch - 2);
    if (y0 == 0) row = std::max(row, floor_row);
    if (row >= 0 && row < ch) return row;
    return std::nullopt;
}

// python _text_width: painted caption width from the box's left edge
// (cv2 draws at xi1 with LINE_AA spill) plus a 2 px pad.
int text_width(const std::string& text) {
    int baseline = 0;
    const cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    return sz.width + 2;
}

// python _as_points: the C++ input is already an (N, 2) point list, so
// only the N >= 2 rule remains to enforce.
std::vector<cv::Point> as_points(const std::vector<cv::Point>& pts) {
    if (pts.size() < 2)
        throw std::invalid_argument("polygon/track points must be (N, 2) with N >= 2, got shape (" +
                                    std::to_string(pts.size()) + ", 2)");
    return pts;
}

struct RectItem {
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    std::string text;
};

struct LineItem {
    std::vector<cv::Point> pts;
    Rgb color = kGreen;
    bool closed = true;
};

// python _draw_masks: one coverage mask per shape, white-on-black.
// Rectangles stay hard-edged (LINE_8, like draw_boxes); captions keep
// their LINE_AA edge coverage.
std::vector<cv::Mat> draw_masks(int ch, int cw, int x0, int y0, int t,
                                const std::vector<RectItem>& rects,
                                const std::vector<LineItem>& lines, bool fragment) {
    std::vector<cv::Mat> masks;
    masks.reserve(rects.size() + lines.size());
    for (const RectItem& r : rects) {
        cv::Mat m = cv::Mat::zeros(ch, cw, CV_8UC1);
        cv::rectangle(m, cv::Point(r.x1 - x0, r.y1 - y0), cv::Point(r.x2 - x0, r.y2 - y0), 255, t);
        if (!r.text.empty()) {
            const std::optional<int> ty = text_row(r.y1, y0, ch, kTextOffset, kTextFloor, fragment);
            if (ty.has_value())
                cv::putText(m, r.text, cv::Point(r.x1 - x0, *ty), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            255, 1, cv::LINE_AA);
        }
        masks.push_back(std::move(m));
    }
    for (const LineItem& l : lines) {
        cv::Mat m = cv::Mat::zeros(ch, cw, CV_8UC1);
        std::vector<cv::Point> shifted;
        shifted.reserve(l.pts.size());
        for (const cv::Point& p : l.pts) shifted.emplace_back(p.x - x0, p.y - y0);
        cv::polylines(m, std::vector<std::vector<cv::Point>>{shifted}, l.closed, 255, t);
        masks.push_back(std::move(m));
    }
    return masks;
}

// python _colorize: straight-alpha RGBA from coverage masks — color per
// draw order (later shapes overwrite inside overlaps), alpha is the mask
// union (max coverage, preserving AA edges).
cv::Mat colorize(const std::vector<cv::Mat>& masks, const std::vector<Rgb>& cols, int ch, int cw) {
    cv::Mat out(ch, cw, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    for (std::size_t k = 0; k < masks.size(); ++k) {
        const cv::Mat& m = masks[k];
        const Rgb& col = cols[k];
        for (int y = 0; y < ch; ++y) {
            const uchar* mrow = m.ptr<uchar>(y);
            cv::Vec4b* orow = out.ptr<cv::Vec4b>(y);
            for (int x = 0; x < cw; ++x) {
                const uchar cov = mrow[x];
                if (cov > 0) {
                    orow[x] = cv::Vec4b(static_cast<uchar>(col[0]), static_cast<uchar>(col[1]),
                                        static_cast<uchar>(col[2]), std::max(orow[x][3], cov));
                }
            }
        }
    }
    return out;
}

// python _overlaps: half-open (x0, y0, x1, y1) rect intersection.
bool overlaps(const std::array<int, 4>& a, const std::array<int, 4>& b) {
    return a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
}

}  // namespace

std::string format_label(const std::optional<std::string>& label,
                         const std::optional<double>& score) {
    if (!label.has_value()) return score.has_value() ? format_score(*score) : std::string();
    if (score.has_value()) return *label + " " + format_score(*score);
    return *label;
}

Rgb palette_color(int class_id) {
    // python % (always non-negative for a positive modulus)
    const int n = static_cast<int>(PALETTE.size());
    const int idx = ((class_id % n) + n) % n;
    return PALETTE[static_cast<std::size_t>(idx)];
}

Annotations extract_annotations(const Detections& result_or_objects,
                                const std::optional<Rgb>& color) {
    Annotations out;
    if (const auto* result = std::get_if<InferenceResult>(&result_or_objects)) {
        out.objects = result->objects;
    } else {
        out.objects = std::get<std::vector<DetectedObject>>(result_or_objects);
    }
    out.boxes.reserve(out.objects.size());
    out.labels.reserve(out.objects.size());
    out.scores.reserve(out.objects.size());
    out.colors.reserve(out.objects.size());
    for (const DetectedObject& obj : out.objects) {
        out.boxes.push_back(box_of(obj.bbox));
        out.labels.emplace_back(obj.label);
        out.scores.emplace_back(obj.score);
        out.colors.push_back(color.has_value() ? *color : palette_color(obj.class_id));
    }
    return out;
}

cv::Mat draw_boxes(const cv::Mat& image, const std::vector<Box>& boxes,
                   const std::vector<std::optional<std::string>>& labels,
                   const std::vector<std::optional<double>>& scores, const Rgb& color,
                   int thickness) {
    cv::Mat out = image.clone();
    const int t = std::max(1, static_cast<int>(thickness));
    const cv::Scalar cv_color(color[0], color[1], color[2]);
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const Box& b = boxes[i];
        const int xi1 = static_cast<int>(b[0]);
        const int yi1 = static_cast<int>(b[1]);
        const int xi2 = static_cast<int>(b[2]);
        const int yi2 = static_cast<int>(b[3]);
        cv::rectangle(out, cv::Point(xi1, yi1), cv::Point(xi2, yi2), cv_color, t);
        const std::optional<std::string> label = i < labels.size() ? labels[i] : std::nullopt;
        const std::optional<double> score = i < scores.size() ? scores[i] : std::nullopt;
        const std::string text = format_label(label, score);
        if (!text.empty()) {
            cv::putText(out, text, cv::Point(xi1, std::max(12, yi1 - 6)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv_color, 1, cv::LINE_AA);
        }
    }
    return out;
}

cv::Mat draw_text(const cv::Mat& image, const std::string& text, std::pair<int, int> xy,
                  const Rgb& color, double font_scale, int thickness) {
    cv::Mat out = image.clone();
    cv::putText(out, text, cv::Point(xy.first, xy.second), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                cv::Scalar(color[0], color[1], color[2]),
                std::max(1, static_cast<int>(thickness)), cv::LINE_AA);
    return out;
}

cv::Mat draw_detections(const cv::Mat& image, const Detections& result_or_objects,
                        const std::optional<Rgb>& color) {
    // python ndim == 2: a single-channel (H, W) array rides the router as
    // NV12. Multi-channel (RGB) Mats raster directly.
    if (image.dims == 2 && image.channels() == 1) {
        return std::any_cast<cv::Mat>(
            get_default_router().run("draw_detections",
                                     {std::any(image), std::any(result_or_objects),
                                      std::any(color)}));
    }
    return draw_detections_impl(image, result_or_objects, color);
}

cv::Mat draw_detections_impl(const cv::Mat& image, const Detections& result_or_objects,
                             const std::optional<Rgb>& color) {
    const Annotations ann = extract_annotations(result_or_objects, color);
    if (ann.objects.empty()) return image.clone();
    cv::Mat out = image.clone();
    for (std::size_t i = 0; i < ann.objects.size(); ++i) {
        out = draw_boxes(out, {ann.boxes[i]}, {ann.labels[i]}, {ann.scores[i]}, ann.colors[i], 2);
    }
    return out;
}

RgbaCanvas render_overlay_rgba(int frame_w, int frame_h, const std::vector<Box>& boxes,
                               const std::vector<std::optional<std::string>>& labels,
                               const std::vector<std::optional<double>>& scores,
                               const std::vector<std::optional<Rgb>>& colors, int thickness,
                               const std::vector<LineShape>& polygons,
                               const std::vector<LineShape>& tracks) {
    const int t = std::max(1, static_cast<int>(thickness));
    // raw line list: closed-ness comes from the source list (polygons vs
    // tracks), matching python's (points, color) pairs.
    std::vector<std::pair<const std::vector<cv::Point>*, std::optional<Rgb>>> raw;
    raw.reserve(polygons.size() + tracks.size());
    for (const LineShape& p : polygons) raw.emplace_back(&p.points, p.color);
    for (const LineShape& p : tracks) raw.emplace_back(&p.points, p.color);
    if (boxes.empty() && raw.empty())
        throw std::invalid_argument("render_overlay_rgba needs at least one shape");

    std::vector<RectItem> rects;
    rects.reserve(boxes.size());
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const std::optional<std::string> label = i < labels.size() ? labels[i] : std::nullopt;
        const std::optional<double> score = i < scores.size() ? scores[i] : std::nullopt;
        rects.push_back(RectItem{static_cast<int>(boxes[i][0]), static_cast<int>(boxes[i][1]),
                                 static_cast<int>(boxes[i][2]), static_cast<int>(boxes[i][3]),
                                 format_label(label, score)});
    }
    // colors in draw order: boxes first, then polygons, then tracks — the
    // same order masks get appended in draw_masks, so zip stays aligned
    std::vector<Rgb> cols;
    cols.reserve(boxes.size() + raw.size());
    for (std::size_t i = 0; i < boxes.size(); ++i)
        cols.push_back(i < colors.size() && colors[i].has_value() ? *colors[i] : kGreen);
    for (const auto& line : raw) cols.push_back(line.second.value_or(kGreen));
    std::vector<LineItem> lines;
    lines.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i)
        lines.push_back(LineItem{as_points(*raw[i].first), cols[boxes.size() + i],
                                 i < polygons.size()});

    // canvas = union bbox of strokes + caption strips, clamped to the frame
    int x_min = frame_w, y_min = frame_h, x_max = 0, y_max = 0;
    for (const RectItem& r : rects) {
        x_min = std::min(x_min, r.x1 - t - 1);
        y_min = std::min(y_min, r.y1 - t - 1 - (r.text.empty() ? 0 : kLabelStrip));
        x_max = std::max(x_max, r.x2 + t + 1);
        y_max = std::max(y_max, r.y2 + t + 1);
    }
    for (const LineItem& l : lines) {
        int lx_min = x_min, ly_min = y_min, lx_max = x_max, ly_max = y_max;
        for (const cv::Point& p : l.pts) {
            lx_min = std::min(lx_min, p.x);
            ly_min = std::min(ly_min, p.y);
            lx_max = std::max(lx_max, p.x);
            ly_max = std::max(ly_max, p.y);
        }
        x_min = std::min(x_min, lx_min - t - 1);
        y_min = std::min(y_min, ly_min - t - 1);
        x_max = std::max(x_max, lx_max + t + 1);
        y_max = std::max(y_max, ly_max + t + 1);
    }
    const int x0 = std::max(0, x_min);
    const int y0 = std::max(0, y_min);
    const int x_end = std::min(frame_w, x_max);
    const int y_end = std::min(frame_h, y_max);
    if (x_end <= x0 || y_end <= y0)
        throw std::invalid_argument("all shapes lie outside the frame");
    const int cw = std::max(kMinOverlayDim, x_end - x0);
    const int ch = std::max(kMinOverlayDim, y_end - y0);

    const std::vector<cv::Mat> masks = draw_masks(ch, cw, x0, y0, t, rects, lines, false);
    return RgbaCanvas{colorize(masks, cols, ch, cw), x0, y0};
}

std::vector<RgbaCanvas> render_overlay_fragments(
    int frame_w, int frame_h, const std::vector<Box>& boxes,
    const std::vector<std::optional<std::string>>& labels,
    const std::vector<std::optional<double>>& scores,
    const std::vector<std::optional<Rgb>>& colors, int thickness,
    const std::vector<LineShape>& polygons, const std::vector<LineShape>& tracks) {
    const int t = std::max(1, static_cast<int>(thickness));
    std::vector<std::pair<const std::vector<cv::Point>*, std::optional<Rgb>>> raw;
    raw.reserve(polygons.size() + tracks.size());
    for (const LineShape& p : polygons) raw.emplace_back(&p.points, p.color);
    for (const LineShape& p : tracks) raw.emplace_back(&p.points, p.color);
    if (boxes.empty() && raw.empty())
        throw std::invalid_argument("render_overlay_fragments needs at least one shape");

    std::vector<RectItem> rects;
    rects.reserve(boxes.size());
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const std::optional<std::string> label = i < labels.size() ? labels[i] : std::nullopt;
        const std::optional<double> score = i < scores.size() ? scores[i] : std::nullopt;
        rects.push_back(RectItem{static_cast<int>(boxes[i][0]), static_cast<int>(boxes[i][1]),
                                 static_cast<int>(boxes[i][2]), static_cast<int>(boxes[i][3]),
                                 format_label(label, score)});
    }
    std::vector<Rgb> cols;
    cols.reserve(boxes.size() + raw.size());
    for (std::size_t i = 0; i < boxes.size(); ++i)
        cols.push_back(i < colors.size() && colors[i].has_value() ? *colors[i] : kGreen);
    for (const auto& line : raw) cols.push_back(line.second.value_or(kGreen));
    std::vector<LineItem> lines;
    lines.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i)
        lines.push_back(LineItem{as_points(*raw[i].first), cols[boxes.size() + i],
                                 i < polygons.size()});

    // per-shape stroke extents, identical formulas to the union canvas —
    // fragment pixels then equal canvas pixels region for region
    int x_min = frame_w, y_min = frame_h, x_max = 0, y_max = 0;
    std::vector<std::array<int, 4>> rect_bboxes;
    rect_bboxes.reserve(rects.size());
    for (const RectItem& r : rects) {
        const int strip = r.text.empty() ? 0 : kLabelStrip;
        rect_bboxes.push_back({r.x1 - t - 1, r.y1 - t - 1 - strip, r.x2 + t + 1, r.y2 + t + 1});
        x_min = std::min(x_min, r.x1 - t - 1);
        y_min = std::min(y_min, r.y1 - t - 1 - strip);
        x_max = std::max(x_max, r.x2 + t + 1);
        y_max = std::max(y_max, r.y2 + t + 1);
    }
    std::vector<std::array<int, 4>> line_bboxes;
    line_bboxes.reserve(lines.size());
    for (const LineItem& l : lines) {
        int lx_min = frame_w, ly_min = frame_h, lx_max = 0, ly_max = 0;
        for (const cv::Point& p : l.pts) {
            lx_min = std::min(lx_min, p.x);
            ly_min = std::min(ly_min, p.y);
            lx_max = std::max(lx_max, p.x);
            ly_max = std::max(ly_max, p.y);
        }
        const std::array<int, 4> bb = {lx_min - t - 1, ly_min - t - 1, lx_max + t + 1,
                                       ly_max + t + 1};
        line_bboxes.push_back(bb);
        x_min = std::min(x_min, bb[0]);
        y_min = std::min(y_min, bb[1]);
        x_max = std::max(x_max, bb[2]);
        y_max = std::max(y_max, bb[3]);
    }
    const int ux0 = std::max(0, x_min);
    const int uy0 = std::max(0, y_min);
    const int ux1 = std::min(frame_w, x_max);
    const int uy1 = std::min(frame_h, y_max);
    if (ux1 <= ux0 || uy1 <= uy0)
        throw std::invalid_argument("all shapes lie outside the frame");

    // fragment rects, tight to what each must carry: horizontal strips to
    // the box stroke extent (a captioned box's top strip widens to the
    // measured text extent — the rasterizer never clips a caption at the
    // box, so its canvas must cover where the text paints), vertical
    // strips to the stroke columns only. Coverage is unchanged: every
    // shape pixel lands in its own fragments, and a shape whose bbox
    // overlaps another fragment's window still rasterizes there too.
    std::vector<std::array<int, 4>> frags;
    for (const RectItem& r : rects) {
        const int strip = r.text.empty() ? 0 : kLabelStrip;
        const int fx0 = std::max(ux0, r.x1 - t - 1);
        int fx1 = std::min(ux1, r.x2 + t + 1);
        if (!r.text.empty()) fx1 = std::min(ux1, std::max(fx1, r.x1 + text_width(r.text)));
        const int top0 = std::max(uy0, r.y1 - t - 1 - strip);
        const int top1 = std::min(uy1, r.y1 + t + 1);
        const int bot0 = std::max(uy0, r.y2 - t);
        const int bot1 = std::min(uy1, r.y2 + t + 1);
        const int lx1 = std::min(ux1, r.x1 + t + 1);
        const int rx0 = std::max(ux0, r.x2 - t);
        if (top1 > top0) frags.push_back({fx0, top0, fx1, top1});
        if (bot1 > bot0) frags.push_back({fx0, bot0, fx1, bot1});
        if (bot0 > top1) {  // side edges between the horizontal strips
            if (lx1 > fx0) frags.push_back({fx0, top1, lx1, bot0});
            if (fx1 > rx0) frags.push_back({rx0, top1, fx1, bot0});
        }
    }
    for (const std::array<int, 4>& bb : line_bboxes) {
        const int fx0 = std::max(ux0, bb[0]);
        const int fy0 = std::max(uy0, bb[1]);
        const int fx1 = std::min(ux1, bb[2]);
        const int fy1 = std::min(uy1, bb[3]);
        if (fx1 > fx0 && fy1 > fy0) frags.push_back({fx0, fy0, fx1, fy1});
    }

    // blend_hw caps one job at 64 overlays and a box costs up to four
    // fragments: past that, fall back to one canvas per shape (a caption
    // wider than its shape then clips at that shape's extent); past 64
    // shapes, one shared union canvas. Either way the return always fits
    // a single blend job.
    if (frags.size() > static_cast<std::size_t>(kMaxOverlayBatch)) {
        frags.clear();
        for (const std::array<int, 4>& bb : rect_bboxes) {
            const int a = std::max(ux0, bb[0]);
            const int b = std::max(uy0, bb[1]);
            const int c = std::min(ux1, bb[2]);
            const int d = std::min(uy1, bb[3]);
            if (c > a && d > b) frags.push_back({a, b, c, d});
        }
        for (const std::array<int, 4>& bb : line_bboxes) {
            const int a = std::max(ux0, bb[0]);
            const int b = std::max(uy0, bb[1]);
            const int c = std::min(ux1, bb[2]);
            const int d = std::min(uy1, bb[3]);
            if (c > a && d > b) frags.push_back({a, b, c, d});
        }
        if (frags.size() > static_cast<std::size_t>(kMaxOverlayBatch))
            frags.assign(1, {ux0, uy0, ux1, uy1});
    }

    std::vector<RgbaCanvas> out;
    out.reserve(frags.size());
    for (const std::array<int, 4>& frag : frags) {
        const int cw = std::max(kMinOverlayDim, frag[2] - frag[0]);
        int ch = std::max(kMinOverlayDim, frag[3] - frag[1]);
        // a caption landing in this fragment dictates its height: the 16 px
        // floor must not clip the descender (the union canvas is always
        // tall enough; row is stable under the later slide)
        for (const RectItem& r : rects) {
            if (r.text.empty()) continue;
            const std::optional<int> row =
                text_row(r.y1, frag[1], ch, kTextOffset, kTextFloor, true);
            if (row.has_value()) ch = std::max(ch, *row + kTextBelow);
        }
        // even height before the slide: blend_hw pads an odd overlay
        // height with one transparent bottom row and enforces bounds on
        // the padded rect, so an odd ch flush to the frame bottom would
        // overshoot the base by that pad row
        ch += ch & 1;
        // the 16 px floor can push the canvas past the frame's right or
        // bottom edge — slide the origin back (content offsets follow the
        // origin, so pixels keep their frame position)
        const int fx = std::max(0, std::min(frag[0], frame_w - cw));
        const int fy = std::max(0, std::min(frag[1], frame_h - ch));
        const std::array<int, 4> window = {fx, fy, fx + cw, fy + ch};
        std::vector<RectItem> sel_rects;
        std::vector<LineItem> sel_lines;
        for (std::size_t i = 0; i < rects.size(); ++i)
            if (overlaps(rect_bboxes[i], window)) sel_rects.push_back(rects[i]);
        for (std::size_t i = 0; i < lines.size(); ++i)
            if (overlaps(line_bboxes[i], window)) sel_lines.push_back(lines[i]);
        if (sel_rects.empty() && sel_lines.empty())
            continue;  // no visible pixels in this fragment
        std::vector<Rgb> sel_cols;
        for (std::size_t i = 0; i < rects.size(); ++i)
            if (overlaps(rect_bboxes[i], window)) sel_cols.push_back(cols[i]);
        for (std::size_t i = 0; i < lines.size(); ++i)
            if (overlaps(line_bboxes[i], window)) sel_cols.push_back(cols[boxes.size() + i]);
        const std::vector<cv::Mat> masks =
            draw_masks(ch, cw, fx, fy, t, sel_rects, sel_lines, true);
        out.push_back(RgbaCanvas{colorize(masks, sel_cols, ch, cw), fx, fy});
    }
    return out;
}

cv::Mat draw_polygons(const cv::Mat& image, const std::vector<LineShape>& shapes, int thickness,
                      bool closed) {
    const int t = std::max(1, static_cast<int>(thickness));
    std::vector<std::pair<std::vector<cv::Point>, Rgb>> parsed;
    parsed.reserve(shapes.size());
    for (const LineShape& s : shapes)
        parsed.emplace_back(as_points(s.points), s.color.value_or(kGreen));
    cv::Mat out = image.clone();
    if (parsed.empty()) return out;
    for (const auto& [pts, col] : parsed)
        cv::polylines(out, std::vector<std::vector<cv::Point>>{pts}, closed,
                      cv::Scalar(col[0], col[1], col[2]), t);
    return out;
}

}  // namespace neoruntime_ipc_sdk

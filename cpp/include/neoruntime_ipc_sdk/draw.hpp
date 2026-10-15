// draw.hpp — annotate RGB arrays with detection boxes and text (port of
// draw.py, 0.8.1).
//
// All functions take an RGB uint8 (H, W, 3) cv::Mat and return a NEW Mat;
// the input is never modified. OpenCV is a hard dependency here, so the
// Pillow legs do not port — every raster is the cv2 leg.
//
// render_overlay_rgba is the hardware companion: it renders the same
// annotation as a minimal straight-alpha RGBA canvas (plus its frame
// offset) for DspClient::blend_hw instead of rasterizing onto the pixels.
//
//   Frame frame = media.get_frame("main");
//   cv::Mat rgb = draw_detections(frame.to_rgb(), result);
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "neoruntime_ipc_sdk/inference_types.hpp"

namespace neoruntime_ipc_sdk {

// Distinct palette used by draw_detections when color is unset (indexed
// by class_id). Component order mirrors python's tuples; cv drawing
// functions receive them as Scalar(c[0], c[1], c[2]) so channel
// assignment matches python cv2 exactly.
inline constexpr std::array<std::array<int, 3>, 8> PALETTE = {{
    {0, 255, 0},     // green
    {255, 80, 80},   // red
    {80, 160, 255},  // blue
    {255, 220, 0},   // yellow
    {255, 0, 255},   // magenta
    {0, 255, 255},   // cyan
    {180, 120, 255}, // purple
    {255, 140, 0},   // orange
}};

using Rgb = std::array<int, 3>;           // python (r, g, b) tuple
using Box = std::array<double, 4>;        // (x1, y1, x2, y2)

// One polygon/track: (N, 2) pixel points plus its color (unset = green).
struct LineShape {
    std::vector<cv::Point> points;
    std::optional<Rgb> color;
    bool closed = true;
};

// What the draw functions accept as detections: a whole InferenceResult
// or a bare object list (python's result_or_objects duck typing).
using Detections = std::variant<InferenceResult, std::vector<DetectedObject>>;

// A rendered straight-alpha overlay canvas and its top-left position on
// the frame (feed to DspClient::blend_hw).
struct RgbaCanvas {
    cv::Mat rgba;  // (h, w, 4) uint8 RGBA
    int x = 0;
    int y = 0;
};

// blend_hw caps one job at 64 overlays — keep in sync with
// src/detail/dsp_wire.hpp kMaxBatch (python dsp_wire._MAX_BATCH).
inline constexpr int kMaxOverlayBatch = 64;
// px reserved above a box for its caption (cv2 scale-0.5 ascent + baseline
// offset 6 + margin) — same strip the software draw_boxes text occupies.
inline constexpr int kLabelStrip = 20;
// daemon DSP floor (dsp_wire._MIN_DIM): blend_hw pads transparently, but a
// canvas born legal avoids a copy at paste time.
inline constexpr int kMinOverlayDim = 16;

// python _format_label: None label + score -> "%.2f"; label + score ->
// "{label} {score:.2f}"; either alone -> itself.
std::string format_label(const std::optional<std::string>& label,
                         const std::optional<double>& score);

// PALETTE[class_id % 8] — draw_detections' per-class default.
Rgb palette_color(int class_id);

// Split detections into draw conventions: objects, boxes, labels, scores
// and per-object colors (python accel._extract_annotations — the given
// color wins when set, else PALETTE by class_id).
struct Annotations {
    std::vector<DetectedObject> objects;
    std::vector<Box> boxes;
    std::vector<std::optional<std::string>> labels;
    std::vector<std::optional<double>> scores;
    std::vector<Rgb> colors;
};
Annotations extract_annotations(const Detections& result_or_objects,
                                const std::optional<Rgb>& color);

// Draw bounding boxes (pixel coordinates) on an RGB copy. Boxes are
// (x1, y1, x2, y2); labels/scores may be shorter than boxes (missing
// entries draw no caption).
cv::Mat draw_boxes(const cv::Mat& image, const std::vector<Box>& boxes,
                   const std::vector<std::optional<std::string>>& labels = {},
                   const std::vector<std::optional<double>>& scores = {},
                   const Rgb& color = {0, 255, 0}, int thickness = 2);

// Draw a text string at pixel position xy on an RGB copy.
cv::Mat draw_text(const cv::Mat& image, const std::string& text,
                  std::pair<int, int> xy, const Rgb& color = {255, 255, 255},
                  double font_scale = 0.5, int thickness = 1);

// Draw an InferenceResult (or a list of DetectedObject) on an RGB copy.
// Each object gets a box plus a "label score" caption. When color is
// unset, a per-class color from PALETTE is chosen via class_id.
//
// Routing: single-channel (NV12) Mats ride the accel router (DSP blend
// when the daemon is reachable, the CPU mirror otherwise); RGB Mats go
// straight to the software raster (no doomed hardware attempt, no fake
// degradation row). Keep-fd frames are not accepted here — call
// frame.to_array() first and route the array (python raises for the same
// reason: the zero-copy frame blend chain is gated in the field).
cv::Mat draw_detections(const cv::Mat& image, const Detections& result_or_objects,
                        const std::optional<Rgb>& color = std::nullopt);

// CPU raster leg of draw_detections (boxes + captions on an RGB copy) —
// what the router serves for RGB inputs.
cv::Mat draw_detections_impl(const cv::Mat& image, const Detections& result_or_objects,
                             const std::optional<Rgb>& color = std::nullopt);

// Render boxes + captions + polygons + tracks as a minimal straight-alpha
// RGBA overlay. The canvas is the union bbox of all shapes (clamped to
// the frame, floored at 16x16), so untouched pixels never enter the blend
// and the quota footprint stays small.
//
// polygons are drawn as closed outlines (zone shapes), tracks open.
// Throws std::invalid_argument with python's messages for zero shapes
// ("render_overlay_rgba needs at least one shape"), fully off-frame
// shapes ("all shapes lie outside the frame") or bad point lists
// ("polygon/track points must be (N, 2) with N >= 2, got shape (N, 2)").
RgbaCanvas render_overlay_rgba(
    int frame_w, int frame_h, const std::vector<Box>& boxes = {},
    const std::vector<std::optional<std::string>>& labels = {},
    const std::vector<std::optional<double>>& scores = {},
    const std::vector<std::optional<Rgb>>& colors = {}, int thickness = 2,
    const std::vector<LineShape>& polygons = {},
    const std::vector<LineShape>& tracks = {});

// Render boxes as tight stroke fragments instead of one bbox canvas.
// Same inputs and straight-alpha contract as render_overlay_rgba, but
// returns a list of (rgba, x, y) fragments to spread over one blend_hw
// call: each box becomes up to four thin canvases — caption+top edge,
// bottom edge, left edge, right edge — so a box pays for its
// 2*thickness stroke perimeter plus its caption strip, not its area.
// Past kMaxOverlayBatch fragments this falls back to one canvas per
// shape, and past kMaxOverlayBatch shapes to the one shared union canvas
// — the return always fits one blend job.
std::vector<RgbaCanvas> render_overlay_fragments(
    int frame_w, int frame_h, const std::vector<Box>& boxes = {},
    const std::vector<std::optional<std::string>>& labels = {},
    const std::vector<std::optional<double>>& scores = {},
    const std::vector<std::optional<Rgb>>& colors = {}, int thickness = 2,
    const std::vector<LineShape>& polygons = {},
    const std::vector<LineShape>& tracks = {});

// Draw polygon outlines (or open tracks) on an RGB copy — the software
// counterpart of render_overlay_rgba's polygons/tracks legs.
cv::Mat draw_polygons(const cv::Mat& image, const std::vector<LineShape>& shapes,
                      int thickness = 2, bool closed = true);

}  // namespace neoruntime_ipc_sdk

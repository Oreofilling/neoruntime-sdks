// inference_codec.hpp — pb <-> public-type pure functions for the inference
// domain (the C++ equivalent of Python's mockable _parse_post_result /
// _tensor_to_numpy seam). Lifted verbatim from inference.cpp's anonymous
// namespace so tests can assert codec behavior directly (test_inference_codec).
#pragma once

#include <algorithm>
#include <cstring>
#include <utility>

#include "neoruntime_ipc_sdk/inference_types.hpp"

#include "detail/tensor_io.hpp"
#include "ai-runtime/inference.pb.h"

namespace neoruntime_ipc_sdk::detail::inference_codec {

namespace pb = aipc::inference;

inline BoundingBox bbox_from_proto(const pb::BBox& b) {
    BoundingBox bb;
    bb.x = b.x();
    bb.y = b.y();
    bb.width = b.w();
    bb.height = b.h();
    return bb;
}

inline TensorSpec tensor_spec_from_proto(const pb::TensorSpec& s) {
    TensorSpec out;
    out.shape.reserve(s.shape_size());
    for (int d : s.shape()) out.shape.push_back(d);
    out.dtype = detail::pb_to_data_type(s.dtype());
    out.name = s.name();
    out.layout = s.layout();
    return out;
}

// register_model / spec-building side of the same mapping (mirrors Python's
// plain-dict spec -> inference_pb2.TensorSpec, which passes layout through).
inline void tensor_spec_to_proto(const TensorSpec& in, pb::TensorSpec* s) {
    for (int d : in.shape) s->add_shape(d);
    s->set_dtype(detail::data_type_to_pb(in.dtype));
    s->set_name(in.name);
    if (!in.layout.empty()) s->set_layout(in.layout);
}

inline ModelInfo model_info_from_proto(const pb::ModelInfo& m) {
    ModelInfo out;
    out.model_id = m.model_id();
    out.model_path = m.model_path();
    out.version = m.version();
    out.inputs.reserve(m.inputs_size());
    for (const auto& i : m.inputs()) out.inputs.push_back(tensor_spec_from_proto(i));
    out.outputs.reserve(m.outputs_size());
    for (const auto& o : m.outputs()) out.outputs.push_back(tensor_spec_from_proto(o));
    out.estimated_tops = m.estimated_tops();
    out.estimated_memory = m.estimated_memory();
    out.load_timestamp = m.load_timestamp();
    out.owner_id = m.owner_id();
    out.transient = m.transient();
    out.batch_size = m.batch_size();
    return out;
}

// Populate the structured-result fields of `out` from a PostResult. Mirrors
// inference.py::_parse_post_result. Best-effort: a malformed sub-message must
// not lose the whole result (Python wraps this in try/except).
inline void fill_post_result(const pb::PostResult& pr, InferenceResult& out) {
    out.objects.reserve(pr.detections_size());
    for (const auto& det : pr.detections()) {
        DetectedObject o;
        o.label = det.label();
        o.score = det.confidence();
        o.class_id = det.class_id();
        o.bbox = bbox_from_proto(det.bbox());
        out.objects.push_back(std::move(o));
    }
    out.classifications.reserve(pr.classifications_size());
    for (const auto& cls : pr.classifications()) {
        Classification c;
        c.type = cls.type();
        c.class_id = cls.class_id();
        c.label = cls.label();
        c.confidence = cls.confidence();
        out.classifications.push_back(std::move(c));
    }
    for (const auto& set : pr.landmarks()) {
        LandmarkSet ls;
        ls.type = set.type();
        ls.points.reserve(set.points_size());
        for (const auto& p : set.points())
            ls.points.push_back(LandmarkPoint{p.x(), p.y(), p.confidence()});
        out.landmarks.push_back(std::move(ls));
    }
    for (const auto& m : pr.masks()) {
        SegmentationMask sm;
        sm.class_id = m.class_id();
        sm.label = m.label();
        sm.confidence = m.confidence();
        sm.bbox = bbox_from_proto(m.bbox());
        sm.mask_rle = m.mask_rle();
        sm.mask_width = m.mask_width();
        sm.mask_height = m.mask_height();
        out.masks.push_back(std::move(sm));
    }
    for (const auto& line : pr.ocr_lines()) {
        OcrLine ol;
        ol.text = line.text();
        ol.confidence = line.confidence();
        ol.bbox = bbox_from_proto(line.bbox());
        out.ocr_lines.push_back(std::move(ol));
    }
    for (const auto& emb : pr.embeddings()) {
        Embedding e;
        e.dim = emb.dim();
        e.data.assign(emb.data().begin(), emb.data().end());
        out.embeddings.push_back(std::move(e));
    }
    for (const auto& dm : pr.depth_maps()) {
        DepthMap d;
        d.width = dm.width();
        d.height = dm.height();
        cv::Mat mat(static_cast<int>(dm.height()), static_cast<int>(dm.width()), CV_32F);
        const auto& bytes = dm.depth_data();
        const std::size_t want = mat.total() * mat.elemSize();
        std::memcpy(mat.data, bytes.data(), std::min(bytes.size(), want));
        d.data = mat;
        out.depth_maps.push_back(std::move(d));
    }
}

// Decode an InferResponse into an InferenceResult. Shared by infer() /
// infer_batch() / infer_async(). frame_sequence/timestamp_ns stay 0 here
// (only StreamInferResponse carries them).
inline InferenceResult parse_infer_response(const pb::InferResponse& resp) {
    InferenceResult out;
    if (resp.has_post_result()) {
        try {
            fill_post_result(resp.post_result(), out);
        } catch (...) {
            // Mirrors inference.py's (ValueError, AttributeError) swallow.
        }
    }
    if (resp.outputs_size() > 0) {
        std::vector<cv::Mat> raw;
        raw.reserve(resp.outputs_size());
        for (const auto& t : resp.outputs()) raw.push_back(detail::tensor_to_mat(t));
        out.raw_outputs = std::move(raw);
    }
    out.infer_time_us = resp.infer_time_us();
    out.queue_time_us = resp.queue_time_us();
    out.hw_infer_time_us = resp.hw_infer_time_us();
    if (!resp.status().success()) out.status_message = resp.status().message();
    return out;
}

// Decode one StreamInferResponse (subscribe path): carries the identity and
// skew fields the single-shot response lacks. skew_us rides raw (0/negative =
// not measured; the >0 gate for the EMA lives in the health layer).
inline InferenceResult parse_stream_response(const pb::StreamInferResponse& resp) {
    InferenceResult out;
    if (resp.has_post_result()) {
        try {
            fill_post_result(resp.post_result(), out);
        } catch (...) {
        }
    }
    if (resp.outputs_size() > 0) {
        std::vector<cv::Mat> raw;
        raw.reserve(resp.outputs_size());
        for (const auto& t : resp.outputs()) raw.push_back(detail::tensor_to_mat(t));
        out.raw_outputs = std::move(raw);
    }
    out.frame_sequence = resp.frame_sequence();
    out.timestamp_ns = resp.timestamp_ns();
    out.skew_us = resp.skew_us();
    out.status_message = resp.status().message();
    return out;
}

// subscribe() request builder (pure — same fields inference.py sets).
inline pb::StreamInferRequest stream_infer_request(const std::string& stream,
                                                   const std::string& model,
                                                   std::uint32_t fps,
                                                   const std::string& session_id,
                                                   bool raw_output_only) {
    pb::StreamInferRequest req;
    req.set_model_id(model);
    req.set_stream_id(stream);
    req.set_fps_limit(fps);
    req.set_session_id(session_id);
    req.set_raw_output_only(raw_output_only);
    return req;
}

// Zero-copy input: a daemon-registry buffer id referenced by id only — no
// data, no dma_fd. NV12 geometry (shape H*3/2 x W, UINT8) matches
// ai-runtime's registry-side tight-NV12 repack.
inline pb::Tensor buffer_ref_tensor(std::uint64_t buffer_id,
                                    std::uint32_t width,
                                    std::uint32_t height) {
    pb::Tensor t;
    t.set_buffer_id(buffer_id);
    t.add_shape(static_cast<std::int32_t>(height) * 3 / 2);
    t.add_shape(static_cast<std::int32_t>(width));
    t.set_dtype(pb::UINT8);
    return t;
}

// Per-model stats row (skew aggregates included; 0 when never streamed).
inline ModelStats inference_stats_from_proto(const pb::InferenceStats& s) {
    ModelStats ms;
    ms.model_id = s.model_id();
    ms.total_inferences = s.total_inferences();
    ms.total_errors = s.total_errors();
    ms.avg_latency_us = s.avg_latency_us();
    ms.current_qps = s.current_qps();
    ms.queue_depth = s.queue_depth();
    ms.hw_fps = s.hw_fps();
    ms.avg_skew_us = s.avg_skew_us();
    ms.max_skew_us = s.max_skew_us();
    ms.skew_samples = s.skew_samples();
    return ms;
}

}  // namespace neoruntime_ipc_sdk::detail::inference_codec

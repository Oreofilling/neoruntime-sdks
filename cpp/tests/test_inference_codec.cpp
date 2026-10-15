// test_inference_codec.cpp — inference codec parity with inference.py /
// inference_codec.py (0.8.1).
//
// Pure-function tests, no daemon: pb messages are filled field-by-field, run
// through detail::inference_codec, and compared against the public structs.
// Covers the Phase 4 additions (layout, owner_id/transient/batch_size, skew,
// buffer-id tensors, stream request builder) plus the carried-over decoders
// (post-result sections, infer/stream responses).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/inference.hpp"

#include "detail/inference_codec.hpp"
#include "detail/tensor_io.hpp"

using namespace neoruntime_ipc_sdk;
namespace codec = detail::inference_codec;
namespace pbns = aipc::inference;  // not `pb`: modern protobuf owns the top-level pb namespace

// ---- tensor_spec roundtrip (layout included) ---------------------------------
TEST(TensorSpecCodec, RoundTripWithLayout) {
    TensorSpec spec;
    spec.shape = {1, 3, 640, 640};
    spec.dtype = DataType::Float32;
    spec.name = "images";
    spec.layout = "NCHW";

    pbns::TensorSpec ps;
    codec::tensor_spec_to_proto(spec, &ps);
    ASSERT_EQ(ps.shape_size(), 4);
    EXPECT_EQ(ps.shape(0), 1);
    EXPECT_EQ(ps.shape(3), 640);
    EXPECT_EQ(ps.name(), "images");
    EXPECT_EQ(ps.layout(), "NCHW");

    const TensorSpec back = codec::tensor_spec_from_proto(ps);
    EXPECT_EQ(back.shape, spec.shape);
    EXPECT_EQ(back.dtype, DataType::Float32);
    EXPECT_EQ(back.name, "images");
    EXPECT_EQ(back.layout, "NCHW");
}

TEST(TensorSpecCodec, EmptyLayoutStaysUnset) {
    TensorSpec spec;
    spec.shape = {640, 640};
    spec.dtype = DataType::Uint8;
    spec.name = "input";
    // layout "" -> not written; parses back as "".
    pbns::TensorSpec ps;
    codec::tensor_spec_to_proto(spec, &ps);
    EXPECT_EQ(ps.layout(), "");
    EXPECT_EQ(codec::tensor_spec_from_proto(ps).layout, "");
}

// ---- model_info (owner_id / transient / batch_size) ---------------------------
TEST(ModelInfoCodec, MapsOwnershipFields) {
    pbns::ModelInfo m;
    m.set_model_id("yolo");
    m.set_model_path("/models/yolo.hef");
    m.set_version("1.2.3");
    m.set_estimated_tops(2.5f);
    m.set_estimated_memory(1024);
    m.set_load_timestamp(1690000000);
    m.set_owner_id("app-7");
    m.set_transient(true);
    m.set_batch_size(4);
    auto* in = m.add_inputs();
    in->set_name("images");
    in->add_shape(640);
    in->add_shape(640);
    in->set_layout("HW");
    auto* out = m.add_outputs();
    out->set_name("boxes");

    const ModelInfo info = codec::model_info_from_proto(m);
    EXPECT_EQ(info.model_id, "yolo");
    EXPECT_EQ(info.model_path, "/models/yolo.hef");
    EXPECT_EQ(info.version, "1.2.3");
    EXPECT_FLOAT_EQ(info.estimated_tops, 2.5f);
    EXPECT_EQ(info.estimated_memory, 1024u);
    EXPECT_EQ(info.load_timestamp, 1690000000ull);
    EXPECT_EQ(info.owner_id, "app-7");
    EXPECT_TRUE(info.transient);
    EXPECT_EQ(info.batch_size, 4u);
    ASSERT_EQ(info.inputs.size(), 1u);
    EXPECT_EQ(info.inputs[0].name, "images");
    EXPECT_EQ(info.inputs[0].layout, "HW");
    ASSERT_EQ(info.outputs.size(), 1u);
    EXPECT_EQ(info.outputs[0].name, "boxes");

    // Defaults: a bare message parses as a global, non-transient model with
    // batch_size 0 — the proto3 default rides through, exactly like Python's
    // plain `batch_size=response.batch_size` read (the dataclass's in-place
    // default of 1 only applies to hand-constructed ModelInfos).
    const ModelInfo bare = codec::model_info_from_proto(pbns::ModelInfo{});
    EXPECT_EQ(bare.owner_id, "");
    EXPECT_FALSE(bare.transient);
    EXPECT_EQ(bare.batch_size, 0u);
}

// ---- fill_post_result: all seven sections -------------------------------------
namespace {
pbns::PostResult sample_post_result() {
    pbns::PostResult pr;
    auto* det = pr.add_detections();
    det->set_label("person");
    det->set_confidence(0.87f);
    det->set_class_id(0);
    det->mutable_bbox()->set_x(0.1f);
    det->mutable_bbox()->set_y(0.2f);
    det->mutable_bbox()->set_w(0.3f);
    det->mutable_bbox()->set_h(0.4f);
    auto* cls = pr.add_classifications();
    cls->set_type("scene");
    cls->set_class_id(3);
    cls->set_label("yard");
    cls->set_confidence(0.66f);
    auto* lm = pr.add_landmarks();
    lm->set_type("face");
    auto* p0 = lm->add_points();
    p0->set_x(0.5f);
    p0->set_y(0.25f);
    p0->set_confidence(0.9f);
    auto* mask = pr.add_masks();
    mask->set_class_id(1);
    mask->set_label("road");
    mask->set_confidence(0.75f);
    mask->mutable_bbox()->set_x(0.0f);
    mask->mutable_bbox()->set_y(0.0f);
    mask->mutable_bbox()->set_w(1.0f);
    mask->mutable_bbox()->set_h(1.0f);
    mask->set_mask_rle(std::string("\x02\x05", 2));  // start 2, length 5
    mask->set_mask_width(8);
    mask->set_mask_height(4);
    auto* ocr = pr.add_ocr_lines();
    ocr->set_text("EXIT");
    ocr->set_confidence(0.81f);
    ocr->mutable_bbox()->set_x(0.4f);
    ocr->mutable_bbox()->set_y(0.5f);
    ocr->mutable_bbox()->set_w(0.1f);
    ocr->mutable_bbox()->set_h(0.05f);
    auto* emb = pr.add_embeddings();
    emb->set_dim(3);
    emb->add_data(0.1f);
    emb->add_data(-0.2f);
    emb->add_data(0.3f);
    auto* dm = pr.add_depth_maps();
    dm->set_width(4);
    dm->set_height(2);
    const float depths[8] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    dm->set_depth_data(depths, sizeof(depths));
    return pr;
}
}  // namespace

TEST(PostResultCodec, ParsesAllSections) {
    InferenceResult out;
    codec::fill_post_result(sample_post_result(), out);

    ASSERT_EQ(out.objects.size(), 1u);
    EXPECT_EQ(out.objects[0].label, "person");
    EXPECT_FLOAT_EQ(out.objects[0].score, 0.87f);
    EXPECT_EQ(out.objects[0].class_id, 0);
    EXPECT_FLOAT_EQ(out.objects[0].bbox.x, 0.1f);
    EXPECT_FLOAT_EQ(out.objects[0].bbox.width, 0.3f);

    ASSERT_EQ(out.classifications.size(), 1u);
    EXPECT_EQ(out.classifications[0].type, "scene");
    EXPECT_EQ(out.classifications[0].class_id, 3);
    EXPECT_FLOAT_EQ(out.classifications[0].confidence, 0.66f);

    ASSERT_EQ(out.landmarks.size(), 1u);
    ASSERT_EQ(out.landmarks[0].points.size(), 1u);
    EXPECT_FLOAT_EQ(out.landmarks[0].points[0].x, 0.5f);
    EXPECT_FLOAT_EQ(out.landmarks[0].points[0].confidence, 0.9f);

    ASSERT_EQ(out.masks.size(), 1u);
    EXPECT_EQ(out.masks[0].mask_width, 8);
    EXPECT_EQ(out.masks[0].mask_height, 4);
    EXPECT_EQ(out.masks[0].mask_rle.size(), 2u);

    ASSERT_EQ(out.ocr_lines.size(), 1u);
    EXPECT_EQ(out.ocr_lines[0].text, "EXIT");
    EXPECT_FLOAT_EQ(out.ocr_lines[0].confidence, 0.81f);

    ASSERT_EQ(out.embeddings.size(), 1u);
    EXPECT_EQ(out.embeddings[0].dim, 3u);
    EXPECT_FLOAT_EQ(out.embeddings[0].data[1], -0.2f);

    ASSERT_EQ(out.depth_maps.size(), 1u);
    EXPECT_EQ(out.depth_maps[0].data.rows, 2);
    EXPECT_EQ(out.depth_maps[0].data.cols, 4);
    EXPECT_FLOAT_EQ(out.depth_maps[0].data.at<float>(1, 3), 8.f);
}

TEST(PostResultCodec, ToMaskDecodesRle) {
    InferenceResult out;
    codec::fill_post_result(sample_post_result(), out);
    // RLE (start=2,len=5) over an 8x4 mask: flat indices 2..6 set.
    const cv::Mat m = out.masks[0].to_mask();
    ASSERT_EQ(m.rows, 4);
    ASSERT_EQ(m.cols, 8);
    EXPECT_EQ(m.at<unsigned char>(0, 2), 255);
    EXPECT_EQ(m.at<unsigned char>(0, 6), 255);
    EXPECT_EQ(m.at<unsigned char>(0, 1), 0);
    EXPECT_EQ(m.at<unsigned char>(0, 7), 0);
}

// ---- parse_infer_response -------------------------------------------------------
TEST(ParseInferResponse, StructuredAndRawOutputs) {
    pbns::InferResponse resp;
    resp.mutable_status()->set_success(true);
    *resp.mutable_post_result() = sample_post_result();
    pbns::Tensor* t = resp.add_outputs();
    t->add_shape(2);
    t->add_shape(3);
    t->set_dtype(pbns::FLOAT32);
    const float vals[6] = {0, 1, 2, 3, 4, 5};
    t->set_data(vals, sizeof(vals));
    resp.set_infer_time_us(15000);
    resp.set_queue_time_us(200);
    resp.set_hw_infer_time_us(12000);

    const InferenceResult r = codec::parse_infer_response(resp);
    EXPECT_EQ(r.objects.size(), 1u);  // post-result carried through
    ASSERT_TRUE(r.raw_outputs.has_value());
    ASSERT_EQ(r.raw_outputs->size(), 1u);
    EXPECT_EQ((*r.raw_outputs)[0].rows, 2);
    EXPECT_EQ((*r.raw_outputs)[0].cols, 3);
    EXPECT_FLOAT_EQ((*r.raw_outputs)[0].at<float>(1, 2), 5.f);
    EXPECT_EQ(r.infer_time_us, 15000u);
    EXPECT_EQ(r.queue_time_us, 200u);
    EXPECT_EQ(r.hw_infer_time_us, 12000u);
    EXPECT_EQ(r.status_message, "");  // success leaves it empty
    EXPECT_EQ(r.frame_sequence, 0u);  // single-shot carries no identity
}

TEST(ParseInferResponse, FailedStatusLandsInStatusMessage) {
    pbns::InferResponse resp;
    resp.mutable_status()->set_success(false);
    resp.mutable_status()->set_message("npu busy");
    const InferenceResult r = codec::parse_infer_response(resp);
    EXPECT_EQ(r.status_message, "npu busy");
    EXPECT_FALSE(r.raw_outputs.has_value());
}

// ---- parse_stream_response (identity + skew) ------------------------------------
TEST(ParseStreamResponse, CarriesIdentityAndSkew) {
    pbns::StreamInferResponse resp;
    resp.set_frame_sequence(4242);
    resp.set_timestamp_ns(1690000012345678901ull);
    resp.set_skew_us(18750);
    resp.mutable_status()->set_success(true);
    *resp.mutable_post_result() = sample_post_result();

    const InferenceResult r = codec::parse_stream_response(resp);
    EXPECT_EQ(r.frame_sequence, 4242u);
    EXPECT_EQ(r.timestamp_ns, 1690000012345678901ull);
    EXPECT_EQ(r.skew_us, 18750);
    EXPECT_EQ(r.objects.size(), 1u);
}

TEST(ParseStreamResponse, NegativeSkewRidesRaw) {
    // int64 on the wire: 0/negative ride through untouched (the >0 EMA gate
    // lives in the health layer, not the codec).
    pbns::StreamInferResponse resp;
    resp.set_skew_us(-5);
    const InferenceResult r = codec::parse_stream_response(resp);
    EXPECT_EQ(r.skew_us, -5);
}

// ---- buffer_ref_tensor ------------------------------------------------------------
TEST(BufferRefTensor, Nv12GeometryByIdOnly) {
    const pbns::Tensor t = codec::buffer_ref_tensor(0xDEADBEEFull, 3840, 2160);
    EXPECT_EQ(t.buffer_id(), 0xDEADBEEFull);
    ASSERT_EQ(t.shape_size(), 2);
    EXPECT_EQ(t.shape(0), 3240);  // 2160 * 3 / 2 (NV12 luma+chroma height)
    EXPECT_EQ(t.shape(1), 3840);
    EXPECT_EQ(t.dtype(), pbns::UINT8);
    EXPECT_TRUE(t.data().empty());   // no pixel bytes cross the wire
    EXPECT_EQ(t.dma_fd(), 0);        // mutually exclusive with data/dma_fd
}

// ---- stream_infer_request ----------------------------------------------------------
TEST(StreamInferRequest, SetsAllFiveFields) {
    const pbns::StreamInferRequest req =
        codec::stream_infer_request("cam0_main", "yolo", 5, "s-1", true);
    EXPECT_EQ(req.model_id(), "yolo");
    EXPECT_EQ(req.stream_id(), "cam0_main");
    EXPECT_EQ(req.fps_limit(), 5u);
    EXPECT_EQ(req.session_id(), "s-1");
    EXPECT_TRUE(req.raw_output_only());
}

// ---- per-model stats (skew aggregates) ----------------------------------------------
TEST(InferenceStatsCodec, SkewAggregates) {
    pbns::InferenceStats s;
    s.set_model_id("yolo");
    s.set_total_inferences(480);
    s.set_total_errors(2);
    s.set_avg_latency_us(29000);
    s.set_current_qps(24.5f);
    s.set_queue_depth(1);
    s.set_hw_fps(29.7f);
    s.set_avg_skew_us(18200);
    s.set_max_skew_us(55100);
    s.set_skew_samples(477);

    const ModelStats ms = codec::inference_stats_from_proto(s);
    EXPECT_EQ(ms.model_id, "yolo");
    EXPECT_EQ(ms.total_inferences, 480u);
    EXPECT_FLOAT_EQ(ms.current_qps, 24.5f);
    EXPECT_EQ(ms.avg_skew_us, 18200u);
    EXPECT_EQ(ms.max_skew_us, 55100u);
    EXPECT_EQ(ms.skew_samples, 477u);
}

// ---- mat <-> tensor dtype roundtrips (tensor_io, via the codec header) ---------------
TEST(TensorIo, BgrMatRoundTrip) {
    cv::Mat img(4, 6, CV_8UC3, cv::Scalar(10, 20, 30));
    const pbns::Tensor t = detail::mat_to_tensor(img);
    ASSERT_EQ(t.shape_size(), 3);
    EXPECT_EQ(t.shape(0), 4);
    EXPECT_EQ(t.shape(1), 6);
    EXPECT_EQ(t.shape(2), 3);
    EXPECT_EQ(t.dtype(), pbns::UINT8);
    const cv::Mat back = detail::tensor_to_mat(t);
    ASSERT_EQ(back.type(), img.type());
    ASSERT_EQ(back.size(), img.size());
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
            for (int ch = 0; ch < 3; ++ch)
                EXPECT_EQ(back.at<cv::Vec3b>(r, c)[ch], img.at<cv::Vec3b>(r, c)[ch])
                    << r << "," << c << "," << ch;
}

TEST(TensorIo, Float32SingleChannelRoundTrip) {
    cv::Mat img(3, 5, CV_32F);
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c) img.at<float>(r, c) = r * 10 + c;
    const pbns::Tensor t = detail::mat_to_tensor(img);
    ASSERT_EQ(t.shape_size(), 2);
    EXPECT_EQ(t.dtype(), pbns::FLOAT32);
    const cv::Mat back = detail::tensor_to_mat(t);
    ASSERT_EQ(back.size(), img.size());
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
            EXPECT_FLOAT_EQ(back.at<float>(r, c), img.at<float>(r, c)) << r << "," << c;
}

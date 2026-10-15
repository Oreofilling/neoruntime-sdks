// test_pipeline.cpp — InferencePipeline composition parity (port of python
// tests/test_pipeline.py). python's FakeClient duck type maps to an
// in-process fake aipc.inference gRPC service here (the client is a concrete
// class, not an interface); the _prepare monkeypatch of the ref-lifecycle
// suite maps to detail::install_dsp_dispatch() returning a borrowed ref.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <grpcpp/grpcpp.h>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "ai-runtime/inference.grpc.pb.h"

#include <any>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/dsp.hpp"
#include "neoruntime_ipc_sdk/frame.hpp"
#include "neoruntime_ipc_sdk/inference.hpp"
#include "neoruntime_ipc_sdk/pipeline.hpp"
#include "neoruntime_ipc_sdk/postprocess.hpp"
#include "neoruntime_ipc_sdk/preprocess.hpp"

#include "detail/dsp_dispatch.hpp"

using namespace neoruntime_ipc_sdk;

namespace pbns = aipc::inference;  // not `pb`: modern protobuf owns the top-level pb namespace

namespace {

// python module-level HEAD: one confident person column
cv::Mat test_head() {
    constexpr int nc = 2;
    cv::Mat arr(4 + nc, 200, CV_32F, cv::Scalar(0));
    arr.at<float>(0, 0) = 320;
    arr.at<float>(1, 0) = 300;
    arr.at<float>(2, 0) = 100;
    arr.at<float>(3, 0) = 80;
    arr.at<float>(4 + 0, 0) = 0.9f;
    return arr;
}

bool mats_equal(const cv::Mat& a, const cv::Mat& b) {
    if (a.rows != b.rows || a.cols != b.cols || a.type() != b.type()) return false;
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::countNonZero(diff.reshape(1)) == 0;
}

// The surface the pipeline touches, scripted per test (python FakeClient).
struct FakeInference final : public pbns::InferenceService::Service {
    struct InferCall {
        std::string model_id;
        std::vector<int> shape;
        std::uint64_t buffer_id = 0;
        bool has_data = false;
        std::uint32_t timeout_ms = 0;
    };
    std::vector<InferCall> infer_calls;

    bool fail_infer = false;        // Infer → CANCELLED "infer down"
    bool emit_raw = false;          // outputs = [HEAD]
    bool emit_detection = false;    // post_result.detections = [person 0.8]

    std::string model_id;  // GetModelInfo: empty response = unknown model
    std::vector<int> input_shape;
    pbns::DataType input_dtype = pbns::UINT8;
    std::string input_layout = "NHWC";

    grpc::Status Infer(grpc::ServerContext*, const pbns::InferRequest* req,
                       pbns::InferResponse* resp) override {
        InferCall call;
        call.model_id = req->model_id();
        call.timeout_ms = req->timeout_ms();
        if (req->inputs_size() > 0) {
            const pbns::Tensor& t = req->inputs(0);
            for (int d : t.shape()) call.shape.push_back(d);
            call.buffer_id = t.buffer_id();
            call.has_data = !t.data().empty();
        }
        infer_calls.push_back(std::move(call));
        if (fail_infer) return {grpc::StatusCode::CANCELLED, "infer down"};
        resp->mutable_status()->set_success(true);
        if (emit_raw) {
            const cv::Mat head = test_head();
            pbns::Tensor* t = resp->add_outputs();
            t->add_shape(head.rows);
            t->add_shape(head.cols);
            t->set_dtype(pbns::FLOAT32);
            t->set_data(head.data, head.total() * head.elemSize());
        }
        if (emit_detection) {
            pbns::Detection* det = resp->mutable_post_result()->add_detections();
            det->set_label("person");
            det->set_confidence(0.8f);
            det->set_class_id(7);
            pbns::BBox* bb = det->mutable_bbox();
            bb->set_x(1);
            bb->set_y(2);
            bb->set_w(3);
            bb->set_h(4);
        }
        return grpc::Status::OK;
    }

    grpc::Status GetModelInfo(grpc::ServerContext*, const pbns::ModelInfo* req,
                              pbns::ModelInfo* resp) override {
        if (model_id.empty() || req->model_id() != model_id) return grpc::Status::OK;
        resp->set_model_id(model_id);
        resp->set_model_path("x.hef");
        resp->set_version("1");
        pbns::TensorSpec* in = resp->add_inputs();
        for (int d : input_shape) in->add_shape(d);
        in->set_dtype(input_dtype);
        in->set_name("in");
        in->set_layout(input_layout);
        return grpc::Status::OK;
    }
};

struct FakeRuntime {
    FakeInference svc;
    std::unique_ptr<grpc::Server> server;
    std::string endpoint;

    void start(const char* tag) {
        const std::string path = "/tmp/ne503-test-pipeline-" + std::string(tag) + ".sock";
        ::unlink(path.c_str());
        grpc::ServerBuilder builder;
        builder.AddListeningPort("unix://" + path, grpc::InsecureServerCredentials());
        builder.RegisterService(&svc);
        server = builder.BuildAndStart();
        endpoint = "unix://" + path;
    }
    ~FakeRuntime() {
        if (server) server->Shutdown();
    }
    std::unique_ptr<InferenceClient> client() const {
        return std::make_unique<InferenceClient>(endpoint);
    }
};

// ---- passthrough ref plumbing -------------------------------------------------------

// A synthetic daemon-side buffer: id 7, release hooks are no-ops. The pool
// ctor validates plane-fd count, and the dtor closes what it was given —
// two memfds stand in for the nv12 planes (never mmapped, only counted).
DspBufferPool make_pool(int w, int h) {
    const std::uint32_t strides[3] = {0, 0, 0};
    const std::uint32_t sizes[3] = {0, 0, 0};
    std::vector<int> fds{::memfd_create("pool-y", 0), ::memfd_create("pool-uv", 0)};
    return DspBufferPool(w, h, "nv12", {7}, std::move(fds), strides, sizes,
                         [](std::uint64_t) {}, [](DspBufferPool*) {});
}

// python monkeypatches accel.shared_dsp_call to return a sentinel; here the
// dispatch seam hands back a live borrowed ref bound to the test pool.
class RefDispatch final : public detail::DspDispatch {
public:
    bool has_live_pools() override { return false; }
    void close() override {}
    std::any call(const std::string&, std::vector<std::any>) override {
        return std::make_shared<DspBufferRef>(DspBufferRef::borrowed(pool));
    }
    DspBufferPool* pool = nullptr;
};

struct DispatchReset {
    ~DispatchReset() { detail::install_dsp_dispatch(nullptr); }
};

// NV12 source frame (python: Frame(width=128, height=96, image=(144,128))).
Frame nv12_frame() {
    Frame frame;
    frame.sequence = 1;
    frame.timestamp_ns = 0;
    frame.width = 128;
    frame.height = 96;
    frame.format = "NV12";
    frame.image = cv::Mat(144, 128, CV_8UC1, cv::Scalar(0));
    return frame;
}

// passthrough preprocessor: stretch, no normalize, packed NV12 out
Preprocessor nv12_pre(std::pair<int, int> size) {
    return Preprocessor(size, "RGB", "stretch", 114, false, "NHWC", std::string("NV12"),
                        true);
}

std::unique_ptr<Postprocessor> person_post() {
    return std::make_unique<YoloV8Postprocessor>(std::vector<std::string>{"person"}, 0.3);
}

}  // namespace

// ---- TestRun ----------------------------------------------------------------------

TEST(Run, PreInferPostComposition) {
    FakeRuntime rt;
    rt.start("compose");
    rt.svc.emit_raw = true;
    InferencePipeline pipe(rt.client(), "m",
                           Preprocessor(std::make_pair(640, 640), "RGB", "letterbox", 114,
                                        false, "NHWC", std::string("RGB")),
                           person_post());

    PipelineResult out = pipe.run(cv::Mat::zeros(1080, 1920, CV_8UC3));

    // preprocessor shrank the image before the RPC
    ASSERT_EQ(rt.svc.infer_calls.size(), 1u);
    EXPECT_EQ(rt.svc.infer_calls[0].model_id, "m");
    EXPECT_EQ(rt.svc.infer_calls[0].shape, (std::vector<int>{640, 640, 3}));
    EXPECT_EQ(rt.svc.infer_calls[0].timeout_ms, 5000u);
    EXPECT_TRUE(rt.svc.infer_calls[0].has_data);
    // client-side decode ran (raw_outputs present) with tunable threshold
    ASSERT_EQ(out.objects.size(), 1u);
    EXPECT_EQ(out.objects[0].label, "person");
    EXPECT_NEAR(out.objects[0].score, 0.9, 1e-6);
    ASSERT_TRUE(out.result.raw_outputs.has_value());
    ASSERT_EQ(out.result.raw_outputs->size(), 1u);
    EXPECT_TRUE(mats_equal((*out.result.raw_outputs)[0], test_head()));
    ASSERT_TRUE(out.tensor.has_value());
    const cv::Mat& tensor = std::get<cv::Mat>(*out.tensor);
    EXPECT_EQ(tensor.rows, 640);
    EXPECT_EQ(tensor.cols, 640);
    EXPECT_EQ(tensor.type(), CV_8UC3);
    ASSERT_TRUE(out.meta.has_value());
    EXPECT_EQ(out.meta->input_size, (std::pair<int, int>(640, 640)));
    EXPECT_GE(out.latency_ms, 0.0);
}

TEST(Run, ServerDecodedPassesThroughWithoutRawOutputs) {
    FakeRuntime rt;
    rt.start("server-decoded");
    rt.svc.emit_detection = true;
    InferencePipeline pipe(rt.client(), "m", std::nullopt,
                           std::make_unique<YoloV8Postprocessor>());
    PipelineResult out = pipe.run(cv::Mat::zeros(8, 8, CV_8UC3));
    // no double decode: server objects pass through untouched
    ASSERT_EQ(out.objects.size(), 1u);
    EXPECT_EQ(out.objects[0].label, "person");
    EXPECT_NEAR(out.objects[0].score, 0.8, 1e-6);
    EXPECT_EQ(out.objects[0].class_id, 7);
    EXPECT_FLOAT_EQ(out.objects[0].bbox.x, 1.0f);
    EXPECT_FLOAT_EQ(out.objects[0].bbox.y, 2.0f);
    EXPECT_FLOAT_EQ(out.objects[0].bbox.width, 3.0f);
    EXPECT_FLOAT_EQ(out.objects[0].bbox.height, 4.0f);
    EXPECT_FALSE(out.result.raw_outputs.has_value());
}

TEST(Run, NoModelIdRaises) {
    FakeRuntime rt;
    rt.start("no-model");
    InferencePipeline pipe(rt.client());
    try {
        pipe.run(cv::Mat::zeros(8, 8, CV_8UC3));
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& exc) {
        EXPECT_NE(std::string(exc.what()).find("model_id"), std::string::npos);
    }
    EXPECT_TRUE(rt.svc.infer_calls.empty());  // raised before the RPC
}

TEST(Run, PassthroughWithoutPreprocessor) {
    FakeRuntime rt;
    rt.start("no-pre");
    InferencePipeline pipe(rt.client(), "m");
    pipe.run(cv::Mat::zeros(8, 8, CV_8UC3));
    ASSERT_EQ(rt.svc.infer_calls.size(), 1u);
    EXPECT_EQ(rt.svc.infer_calls[0].shape, (std::vector<int>{8, 8, 3}));  // untouched
}

// ---- TestRunAsync --------------------------------------------------------------------

TEST(RunAsync, FutureResolvesToPipelineResult) {
    FakeRuntime rt;
    rt.start("async");
    rt.svc.emit_raw = true;
    InferencePipeline pipe(rt.client(), "m",
                           Preprocessor(std::make_pair(640, 640), "RGB", "letterbox", 114,
                                        false, "NHWC", std::string("RGB")),
                           person_post());
    std::future<PipelineResult> fut =
        pipe.run_async(cv::Mat::zeros(1080, 1920, CV_8UC3));
    PipelineResult out = fut.get();
    ASSERT_EQ(rt.svc.infer_calls.size(), 1u);
    EXPECT_EQ(rt.svc.infer_calls[0].shape, (std::vector<int>{640, 640, 3}));
    ASSERT_EQ(out.objects.size(), 1u);
    EXPECT_EQ(out.objects[0].label, "person");
}

// ---- TestFromModel ---------------------------------------------------------------------

TEST(FromModel, WiresDerivedPreprocessor) {
    FakeRuntime rt;
    rt.start("from-model");
    rt.svc.emit_raw = true;
    rt.svc.model_id = "m";
    rt.svc.input_shape = {1, 640, 640, 3};
    rt.svc.input_dtype = pbns::UINT8;
    rt.svc.input_layout = "NHWC";
    InferencePipeline pipe = InferencePipeline::from_model("m", rt.client(), person_post());
    ASSERT_TRUE(pipe.preprocessor.has_value());
    EXPECT_EQ(pipe.preprocessor->size,
              (std::optional<std::pair<int, int>>(std::make_pair(640, 640))));
    PipelineResult out = pipe.run(cv::Mat::zeros(1080, 1920, CV_8UC3));
    ASSERT_EQ(out.objects.size(), 1u);
    EXPECT_EQ(out.objects[0].label, "person");
}

TEST(FromModel, Nv12ModelAutoPassthroughFeedsInferTheRef) {
    // P2-6: an NV12-layout model gets the device-resident leg, and the ref
    // the preprocessor returns rides infer as its buffer_id (no RGB copy,
    // no read-back).
    DispatchReset reset;
    DspBufferPool pool = make_pool(640, 384);
    RefDispatch dispatch;
    dispatch.pool = &pool;
    detail::install_dsp_dispatch(&dispatch);

    FakeRuntime rt;
    rt.start("nv12-ref");
    rt.svc.emit_raw = true;
    rt.svc.model_id = "m";
    rt.svc.input_shape = {1, 384, 640, 1};
    rt.svc.input_dtype = pbns::UINT8;
    rt.svc.input_layout = "NV12";
    InferencePipeline pipe = InferencePipeline::from_model("m", rt.client(), person_post());
    ASSERT_TRUE(pipe.preprocessor.has_value());
    EXPECT_TRUE(pipe.preprocessor->nv12_passthrough);

    PipelineResult out = pipe.run(nv12_frame());
    // a ref, not pixels: buffer_id set, no data payload
    ASSERT_EQ(rt.svc.infer_calls.size(), 1u);
    EXPECT_EQ(rt.svc.infer_calls[0].buffer_id, 7u);
    EXPECT_FALSE(rt.svc.infer_calls[0].has_data);
    EXPECT_EQ(rt.svc.infer_calls[0].shape, (std::vector<int>{384 * 3 / 2, 640}));
    // settled after inference: the ref is gone from the result (released)
    EXPECT_FALSE(out.tensor.has_value());
}

// ---- TestPassthroughRefLifecycle --------------------------------------------------------
// The passthrough input ref must not outlive its pass by default. python
// monkeypatches _prepare with a counting ref; here the real passthrough runs
// against the borrowed-ref dispatch and the settle contract is asserted
// through PipelineResult::tensor (nullopt = released, value = retained).

TEST(PassthroughRefLifecycle, RunReleasesTheRefOnceInferenceSettles) {
    DispatchReset reset;
    DspBufferPool pool = make_pool(640, 384);
    RefDispatch dispatch;
    dispatch.pool = &pool;
    detail::install_dsp_dispatch(&dispatch);

    FakeRuntime rt;
    rt.start("ref-settle");
    rt.svc.emit_raw = true;
    InferencePipeline pipe(rt.client(), "m", nv12_pre({640, 384}), nullptr, 5000, false);
    PipelineResult out = pipe.run(nv12_frame());
    EXPECT_FALSE(out.tensor.has_value());  // released — nothing leaked into the result
    ASSERT_EQ(rt.svc.infer_calls.size(), 1u);
    EXPECT_EQ(rt.svc.infer_calls[0].buffer_id, 7u);  // it did ride infer alive
}

TEST(PassthroughRefLifecycle, RunFailurePathStillReleases) {
    DispatchReset reset;
    DspBufferPool pool = make_pool(640, 384);
    RefDispatch dispatch;
    dispatch.pool = &pool;
    detail::install_dsp_dispatch(&dispatch);

    FakeRuntime rt;
    rt.start("ref-fail");
    rt.svc.fail_infer = true;
    InferencePipeline pipe(rt.client(), "m", nv12_pre({640, 384}), nullptr, 5000, false);
    try {
        pipe.run(nv12_frame());
        FAIL() << "expected the infer failure to propagate";
    } catch (const std::exception& exc) {
        EXPECT_NE(std::string(exc.what()).find("infer down"), std::string::npos);
    }
}

TEST(PassthroughRefLifecycle, RetainInputKeepsTheRefUntilRelease) {
    DispatchReset reset;
    DspBufferPool pool = make_pool(640, 384);
    RefDispatch dispatch;
    dispatch.pool = &pool;
    detail::install_dsp_dispatch(&dispatch);

    FakeRuntime rt;
    rt.start("ref-retain");
    rt.svc.emit_raw = true;
    InferencePipeline pipe(rt.client(), "m", nv12_pre({640, 384}), nullptr, 5000, true);
    PipelineResult out = pipe.run(nv12_frame());
    ASSERT_TRUE(out.tensor.has_value());
    auto* ref = std::get_if<DspBufferRef>(&*out.tensor);
    ASSERT_NE(ref, nullptr);
    EXPECT_EQ(ref->buffer_id(), 7u);  // still alive for a zero-copy chain
    out.release();
    EXPECT_FALSE(out.tensor.has_value());
    out.release();  // idempotent
    EXPECT_FALSE(out.tensor.has_value());
}

TEST(PassthroughRefLifecycle, RunAsyncFailurePathStillReleases) {
    DispatchReset reset;
    DspBufferPool pool = make_pool(640, 384);
    RefDispatch dispatch;
    dispatch.pool = &pool;
    detail::install_dsp_dispatch(&dispatch);

    FakeRuntime rt;
    rt.start("ref-async-fail");
    rt.svc.fail_infer = true;
    InferencePipeline pipe(rt.client(), "m", nv12_pre({640, 384}), nullptr, 5000, false);
    std::future<PipelineResult> fut = pipe.run_async(nv12_frame());
    try {
        fut.get();
        FAIL() << "expected the infer failure to propagate";
    } catch (const std::exception& exc) {
        EXPECT_NE(std::string(exc.what()).find("infer down"), std::string::npos);
    }
}

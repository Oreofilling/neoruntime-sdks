// test_preprocess.cpp — Preprocessor + PreprocessMeta parity (port of
// python tests/test_preprocess.py). from_model rides an in-process fake
// InferenceService (the concrete InferenceClient has no interface seam);
// the nv12_passthrough device leg rides detail::install_dsp_dispatch()
// (python monkeypatches accel.shared_dsp_call) and set_route_policy for
// the policy variants.
#include <gtest/gtest.h>

#include <grpcpp/grpcpp.h>

#include <opencv2/core.hpp>

#include <atomic>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ai-runtime/inference.grpc.pb.h"

#include "neoruntime_ipc_sdk/accel.hpp"
#include "neoruntime_ipc_sdk/dsp.hpp"
#include "neoruntime_ipc_sdk/frame.hpp"
#include "neoruntime_ipc_sdk/inference.hpp"
#include "neoruntime_ipc_sdk/preprocess.hpp"

#include "detail/dsp_dispatch.hpp"

namespace pbns = aipc::inference;  // not `pb`: modern protobuf owns the top-level pb namespace
using namespace neoruntime_ipc_sdk;

namespace {

Frame make_rgb_frame(int w, int h) {
    Frame f;
    f.sequence = 1;
    f.width = w;
    f.height = h;
    f.format = "RGB";
    f.image = cv::Mat::zeros(h, w, CV_8UC3);
    return f;
}

Frame nv12_frame(int w, int h, std::uint8_t value = 128) {
    Frame f;
    f.sequence = 1;
    f.width = w;
    f.height = h;
    f.format = "NV12";
    f.image = cv::Mat(h * 3 / 2, w, CV_8UC1, cv::Scalar(value));
    return f;
}

bool same_bytes(const cv::Mat& a, const cv::Mat& b) {
    if (a.rows != b.rows || a.cols != b.cols || a.type() != b.type()) return false;
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::countNonZero(diff.reshape(1)) == 0;
}

// python's pytest.raises(match=...) — the thrown message, or "".
template <typename Fn>
std::string thrown_message(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& exc) {
        return exc.what();
    }
    return "";
}

// ---- from_model fake service --------------------------------------------------

struct FakeModel {
    std::vector<int> shape;
    pbns::DataType dtype = pbns::UINT8;
    std::string layout;
};

struct FakeInference final : pbns::InferenceService::Service {
    grpc::Status GetModelInfo(grpc::ServerContext*, const pbns::ModelInfo* req,
                              pbns::ModelInfo* resp) override {
        const auto it = models.find(req->model_id());
        if (it == models.end()) return grpc::Status::OK;  // empty id → nullopt
        resp->set_model_id(it->first);
        pbns::TensorSpec* inp = resp->add_inputs();
        for (int v : it->second.shape) inp->add_shape(v);
        inp->set_dtype(it->second.dtype);
        inp->set_name("in");
        inp->set_layout(it->second.layout);
        return grpc::Status::OK;
    }

    std::map<std::string, FakeModel> models;
};

// One in-process ai-runtime serving GetModelInfo over a UDS.
struct FakeRuntime {
    FakeInference svc;
    std::unique_ptr<grpc::Server> server;
    std::string endpoint;

    FakeInference& with(const std::string& id, FakeModel model) {
        svc.models[id] = std::move(model);
        return svc;
    }

    grpc::Server* start(const char* tag) {
        char sock[128];
        std::snprintf(sock, sizeof(sock), "/tmp/prep_%s_%d", tag, static_cast<int>(::getpid()));
        ::unlink(sock);
        endpoint = "unix://" + std::string(sock);  // triple-slash, the SDK's form
        grpc::ServerBuilder builder;
        builder.RegisterService(&svc);
        builder.AddListeningPort(endpoint, grpc::InsecureServerCredentials());
        server = builder.BuildAndStart();
        return server.get();
    }
    ~FakeRuntime() {
        if (server) server->Shutdown();
    }
};

// ---- passthrough dispatch seam --------------------------------------------------

struct CapturingDispatch final : detail::DspDispatch {
    std::vector<std::pair<std::string, std::vector<std::any>>> calls;
    std::function<std::any(const std::string&, const std::vector<std::any>&)> behavior;

    std::any call(const std::string& method, std::vector<std::any> args) override {
        calls.emplace_back(method, args);
        if (behavior) return behavior(method, args);
        return {};
    }
};

struct DispatchReset {
    ~DispatchReset() { detail::install_dsp_dispatch(nullptr); }
};

// RoutePolicy restoration around set_route_policy tests.
struct PolicyReset {
    explicit PolicyReset(AccelRouter& router) : original(router.policy()) {}
    ~PolicyReset() { set_route_policy(original); }
    RoutePolicy original;
};

}  // namespace

// ---- TestValidation -------------------------------------------------------------------

TEST(Validation, BadColorRejected) {
    const std::string msg = thrown_message(
        [] { Preprocessor(std::make_pair(64, 64), "YUV"); });
    EXPECT_NE(msg.find("color"), std::string::npos);
}

TEST(Validation, BadResizeModeRejected) {
    const std::string msg = thrown_message(
        [] { Preprocessor(std::make_pair(64, 64), "RGB", "pad"); });
    EXPECT_NE(msg.find("resize_mode"), std::string::npos);
}

TEST(Validation, BadLayoutRejected) {
    const std::string msg = thrown_message([] {
        Preprocessor(std::make_pair(64, 64), "RGB", "letterbox", 114, false, "CHW");
    });
    EXPECT_NE(msg.find("layout"), std::string::npos);
}

TEST(Validation, MissingSizeRejectedOnCall) {
    const std::string msg = thrown_message(
        [] { Preprocessor()(cv::Mat::zeros(8, 8, CV_8UC3)); });
    EXPECT_NE(msg.find("size"), std::string::npos);
}

TEST(Validation, TwoDArrayRequiresSourceFormat) {
    const std::string msg = thrown_message([] {
        Preprocessor(std::make_pair(64, 64))(cv::Mat::zeros(12, 8, CV_8UC1));
    });
    EXPECT_NE(msg.find("NV12"), std::string::npos);
    EXPECT_NE(msg.find("GRAY8"), std::string::npos);
}

TEST(Validation, UnsupportedSourceFormatRejectedAtConstruction) {
    const std::string msg = thrown_message([] {
        Preprocessor(std::make_pair(64, 64), "RGB", "letterbox", 114, false, "NHWC",
                     "YUYV");
    });
    EXPECT_NE(msg.find("source_format"), std::string::npos);
}

TEST(Validation, RgbaArrayRejectedWithStripAlphaHint) {
    // 4-channel input used to pass validation and blow up much later in
    // to_color; now it fails fast at the first call.
    const std::string msg = thrown_message([] {
        Preprocessor(std::make_pair(64, 64))(cv::Mat::zeros(48, 64, CV_8UC4));
    });
    EXPECT_NE(msg.find("RGBA"), std::string::npos);
}

// ---- TestCall ----------------------------------------------------------------------------

TEST(Call, LetterboxGeometryMatchesTransform) {
    Preprocessor pre(std::make_pair(640, 640));
    Preprocessor::Output out = pre(make_rgb_frame(1920, 1080));
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.rows, 640);
    EXPECT_EQ(tensor.cols, 640);
    EXPECT_EQ(tensor.type(), CV_8UC3);
    EXPECT_EQ(out.meta.original_size, std::make_pair(1920, 1080));
    EXPECT_EQ(out.meta.input_size, std::make_pair(640, 640));
    EXPECT_EQ(out.meta.origin, std::make_pair(0, 140));
    EXPECT_NEAR(out.meta.scale.first, 640.0 / 1920.0, 1e-9);
}

TEST(Call, IdentityWhenSizeMatches) {
    Preprocessor pre(std::make_pair(64, 48));
    Preprocessor::Output out = pre(make_rgb_frame(64, 48));
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.rows, 48);
    EXPECT_EQ(tensor.cols, 64);
    EXPECT_EQ(out.meta.scale, std::make_pair(1.0, 1.0));
    EXPECT_EQ(out.meta.origin, std::make_pair(0, 0));
}

TEST(Call, NdarrayInputWrapped) {
    Preprocessor pre(std::make_pair(32, 32), "RGB", "letterbox", 114, false, "NHWC",
                     "RGB");
    Preprocessor::Output out = pre(cv::Mat::zeros(48, 64, CV_8UC3));
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.rows, 32);
    EXPECT_EQ(tensor.cols, 32);
    EXPECT_EQ(out.meta.original_size, std::make_pair(64, 48));
}

TEST(Call, Nv12Source) {
    Preprocessor pre(std::make_pair(32, 32));
    Preprocessor::Output out = pre(nv12_frame(64, 48));
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.rows, 32);
    EXPECT_EQ(tensor.cols, 32);
    EXPECT_EQ(out.meta.origin, std::make_pair(0, 4));  // 32x24 content, chroma-aligned
}

TEST(Call, BgrSwap) {
    Preprocessor pre(std::make_pair(64, 48), "BGR", "letterbox", 114, false, "NHWC",
                     "RGB");
    cv::Mat src = cv::Mat::zeros(48, 64, CV_8UC3);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) src.at<cv::Vec3b>(y, x)[0] = 255;  // red in RGB
    Preprocessor::Output out = pre(src);
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.at<cv::Vec3b>(0, 0)[2], 255);  // now blue channel
    EXPECT_EQ(tensor.at<cv::Vec3b>(0, 0)[0], 0);
}

TEST(Call, NormalizeMakesFloat) {
    Preprocessor pre(std::make_pair(64, 48), "RGB", "letterbox", 114, true);
    Preprocessor::Output out = pre(make_rgb_frame(64, 48));
    // convertTo(CV_32F) keeps the 3 channels — CV_32FC3 (type 21)
    EXPECT_EQ(std::get<cv::Mat>(out.tensor).type(), CV_32FC3);
}

TEST(Call, NchwLayout) {
    Preprocessor pre(std::make_pair(64, 48), "RGB", "letterbox", 114, false, "NCHW");
    Preprocessor::Output out = pre(make_rgb_frame(64, 48));
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    // CHW is one continuous 3D Mat: dims (c, h, w) — rows/cols are -1
    EXPECT_EQ(tensor.dims, 3);
    EXPECT_EQ(tensor.size[0], 3);
    EXPECT_EQ(tensor.size[1], 48);
    EXPECT_EQ(tensor.size[2], 64);
    EXPECT_EQ(out.meta.layout, "NCHW");
}

// ---- TestFromModel -------------------------------------------------------------------------

namespace {

FakeModel nhwc_model() { return FakeModel{{1, 640, 640, 3}, pbns::UINT8, ""}; }

}  // namespace

TEST(FromModel, DerivesNhwcUint8) {
    FakeRuntime rt;
    rt.with("m", nhwc_model());
    ASSERT_NE(rt.start("nhwc"), nullptr);
    InferenceClient client(rt.endpoint);
    Preprocessor pre = Preprocessor::from_model(client, "m");
    EXPECT_EQ(pre.size, std::make_pair(640, 640));
    EXPECT_EQ(pre.layout, "NHWC");
    EXPECT_FALSE(pre.normalize);  // uint8 HEF: quantisation lives in the HEF
}

TEST(FromModel, DerivesNchwFromShape) {
    FakeRuntime rt;
    rt.with("m", FakeModel{{1, 3, 416, 416}, pbns::FLOAT32, ""});
    ASSERT_NE(rt.start("nchw"), nullptr);
    InferenceClient client(rt.endpoint);
    Preprocessor pre = Preprocessor::from_model(client, "m");
    EXPECT_EQ(pre.size, std::make_pair(416, 416));
    EXPECT_EQ(pre.layout, "NCHW");
    EXPECT_TRUE(pre.normalize);  // float32 input
}

TEST(FromModel, LayoutHintWins) {
    FakeRuntime rt;
    FakeModel model = nhwc_model();
    model.layout = "NHWC";
    rt.with("m", std::move(model));
    ASSERT_NE(rt.start("hint"), nullptr);
    InferenceClient client(rt.endpoint);
    Preprocessor pre = Preprocessor::from_model(client, "m");
    EXPECT_EQ(pre.layout, "NHWC");
}

TEST(FromModel, Overrides) {
    FakeRuntime rt;
    rt.with("m", nhwc_model());
    ASSERT_NE(rt.start("ovr"), nullptr);
    InferenceClient client(rt.endpoint);
    Preprocessor::Overrides overrides;
    overrides.color = "BGR";
    overrides.resize_mode = "stretch";
    Preprocessor pre = Preprocessor::from_model(client, "m", overrides);
    EXPECT_EQ(pre.color, "BGR");
    EXPECT_EQ(pre.resize_mode, "stretch");
}

TEST(FromModel, UnknownModelRaises) {
    FakeRuntime rt;  // serves nothing
    ASSERT_NE(rt.start("miss"), nullptr);
    InferenceClient client(rt.endpoint);
    const std::string msg = thrown_message(
        [&] { Preprocessor::from_model(client, "nope"); });
    EXPECT_NE(msg.find("not registered"), std::string::npos);
}

// ---- TestPreprocessMeta ----------------------------------------------------------------------

TEST(PreprocessMeta, RoundtripAgainstFrameTransform) {
    Frame f = make_rgb_frame(1920, 1080).resize(640, 640, "letterbox");
    ASSERT_TRUE(f.transform.has_value());
    const AffineTransform& t = *f.transform;
    PreprocessMeta meta;
    meta.original_size = {t.src_width, t.src_height};
    meta.input_size = {t.dst_width, t.dst_height};
    meta.scale = {t.scale_x, t.scale_y};
    meta.origin = {static_cast<int>(t.origin_x), static_cast<int>(t.origin_y)};
    const auto [x, y] = meta.to_source(320, 320);
    EXPECT_NEAR(x, 960, 1);
    EXPECT_NEAR(y, 540, 1);
}

TEST(PreprocessMeta, ToSourceBox) {
    PreprocessMeta meta;
    meta.original_size = {100, 100};
    meta.input_size = {50, 50};
    meta.scale = {0.5, 0.5};
    meta.origin = {0, 10};
    const std::array<double, 4> box = meta.to_source_box({10, 10, 30, 30});
    EXPECT_EQ(box, (std::array<double, 4>{20.0, 0.0, 60.0, 40.0}));
}

// ---- TestReviewFixes ---------------------------------------------------------------------------

TEST(ReviewFixes, IdentityPathInheritsExistingTransform) {
    // a frame already cropped to model size must not lose its transform —
    // postprocessing maps back to the ORIGINAL frame
    Frame cropped = make_rgb_frame(1920, 1080).crop(960, 220, 640, 640);
    Preprocessor pre(std::make_pair(640, 640));
    Preprocessor::Output out = pre(cropped);
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.rows, 640);
    EXPECT_EQ(tensor.cols, 640);
    EXPECT_EQ(out.meta.original_size, std::make_pair(1920, 1080));  // not (640, 640)
    const auto [x, y] = out.meta.to_source(320, 320);
    EXPECT_NEAR(x, 1280, 1);
    EXPECT_NEAR(y, 540, 1);
}

TEST(ReviewFixes, LowercaseParamsAccepted) {
    Preprocessor pre(std::make_pair(64, 48), "bgr", "letterbox", 114, false, "nchw",
                     "rgb");
    EXPECT_EQ(pre.color, "BGR");
    EXPECT_EQ(pre.layout, "NCHW");
    EXPECT_EQ(pre.source_format, "RGB");
    Preprocessor::Output out = pre(cv::Mat::zeros(48, 64, CV_8UC3));
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.dims, 3);
    EXPECT_EQ(tensor.size[0], 3);
    EXPECT_EQ(tensor.size[1], 48);
    EXPECT_EQ(tensor.size[2], 64);
}

TEST(ReviewFixes, LowercaseSourceFormatNv12) {
    Preprocessor pre(std::make_pair(32, 32), "RGB", "letterbox", 114, false, "NHWC",
                     "nv12");
    Preprocessor::Output out = pre(nv12_frame(64, 48).image);
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.rows, 32);
    EXPECT_EQ(tensor.cols, 32);
}

// ---- TestNv12Passthrough: constructor contract ---------------------------------------------

TEST(Nv12Passthrough, RequiresStretch) {
    // default resize_mode is letterbox; passthrough must refuse it
    const std::string msg = thrown_message([] {
        Preprocessor(std::make_pair(64, 48), "RGB", "letterbox", 114, false, "NHWC",
                     std::nullopt, true);
    });
    EXPECT_NE(msg.find("stretch"), std::string::npos);
}

TEST(Nv12Passthrough, RequiresNoNormalize) {
    const std::string msg = thrown_message([] {
        Preprocessor(std::make_pair(64, 48), "RGB", "stretch", 114, true, "NHWC",
                     std::nullopt, true);
    });
    EXPECT_NE(msg.find("normalize"), std::string::npos);
}

TEST(Nv12Passthrough, RequiresNhwc) {
    const std::string msg = thrown_message([] {
        Preprocessor(std::make_pair(64, 48), "RGB", "stretch", 114, false, "NCHW",
                     std::nullopt, true);
    });
    EXPECT_NE(msg.find("packed NV12"), std::string::npos);
}

// ---- TestNv12Passthrough: from_model auto-enable -------------------------------------------

namespace {

// configure + start in place: FakeRuntime has a user dtor (no implicit
// move), so it cannot be returned by value
void start_nv12_layout(FakeRuntime& rt, const char* tag,
                       pbns::DataType dtype = pbns::UINT8) {
    rt.with("m", FakeModel{{1, 384, 640, 1}, dtype, "NV12"});
    rt.start(tag);
}

}  // namespace

TEST(Nv12Passthrough, FromModelAutoEnablesForNv12Layout) {
    FakeRuntime rt;
    start_nv12_layout(rt, "auto");
    InferenceClient client(rt.endpoint);
    Preprocessor pre = Preprocessor::from_model(client, "m");
    EXPECT_TRUE(pre.nv12_passthrough);
    EXPECT_EQ(pre.resize_mode, "stretch");
    EXPECT_EQ(pre.size, std::make_pair(640, 384));
    EXPECT_EQ(pre.layout, "NHWC");
}

TEST(Nv12Passthrough, FromModelRespectsResizeModeOverride) {
    FakeRuntime rt;
    start_nv12_layout(rt, "keep");
    InferenceClient client(rt.endpoint);
    Preprocessor::Overrides overrides;
    overrides.resize_mode = "letterbox";
    Preprocessor pre = Preprocessor::from_model(client, "m", overrides);
    EXPECT_FALSE(pre.nv12_passthrough);
    EXPECT_EQ(pre.resize_mode, "letterbox");
}

TEST(Nv12Passthrough, FromModelRespectsExplicitPassthroughFalse) {
    FakeRuntime rt;
    start_nv12_layout(rt, "explicit");
    InferenceClient client(rt.endpoint);
    Preprocessor::Overrides overrides;
    overrides.nv12_passthrough = false;
    Preprocessor pre = Preprocessor::from_model(client, "m", overrides);
    EXPECT_FALSE(pre.nv12_passthrough);
}

TEST(Nv12Passthrough, FromModelNoAutoEnableForRgb) {
    FakeRuntime rt;
    rt.with("m", nhwc_model());
    ASSERT_NE(rt.start("rgb"), nullptr);
    InferenceClient client(rt.endpoint);
    Preprocessor pre = Preprocessor::from_model(client, "m");
    EXPECT_FALSE(pre.nv12_passthrough);
}

TEST(Nv12Passthrough, FromModelNoAutoEnableForFloatDtype) {
    // float implies normalize=true, which passthrough cannot serve
    FakeRuntime rt;
    start_nv12_layout(rt, "float", pbns::FLOAT32);
    InferenceClient client(rt.endpoint);
    Preprocessor pre = Preprocessor::from_model(client, "m");
    EXPECT_FALSE(pre.nv12_passthrough);
    EXPECT_TRUE(pre.normalize);
}

// ---- TestNv12Passthrough: device leg --------------------------------------------------------

TEST(Nv12Passthrough, DeviceLegReturnsRefAndMeta) {
    DispatchReset reset;
    CapturingDispatch dispatch;
    dispatch.behavior = [](const std::string&, const std::vector<std::any>&) {
        return std::any(std::make_shared<DspBufferRef>(DspBufferRef::detached()));
    };
    detail::install_dsp_dispatch(&dispatch);

    Preprocessor pre(std::make_pair(64, 32), "RGB", "stretch", 114, false, "NHWC",
                     std::nullopt, true);
    Frame frame = nv12_frame(128, 96);
    Preprocessor::Output out = pre(frame);

    EXPECT_TRUE(std::holds_alternative<DspBufferRef>(out.tensor));
    ASSERT_EQ(dispatch.calls.size(), 1u);
    EXPECT_EQ(dispatch.calls[0].first, "resize_hw");
    const std::vector<std::any>& args = dispatch.calls[0].second;
    ASSERT_EQ(args.size(), 4u);
    const auto* src = std::any_cast<Frame>(&args[0]);
    ASSERT_NE(src, nullptr);
    EXPECT_EQ(src->width, 128);
    EXPECT_EQ(src->height, 96);
    EXPECT_EQ(src->format, "NV12");
    EXPECT_EQ(std::any_cast<int>(args[1]), 64);
    EXPECT_EQ(std::any_cast<int>(args[2]), 32);
    const RouterDspOpts opts = std::any_cast<RouterDspOpts>(args[3]);
    EXPECT_EQ(opts.fmt, "nv12");
    EXPECT_EQ(opts.scaling, "stretch");
    EXPECT_FALSE(opts.cpu_fallback);
    EXPECT_TRUE(opts.out_ref);

    EXPECT_EQ(out.meta.color, "NV12");
    EXPECT_EQ(out.meta.layout, "NHWC");
    EXPECT_EQ(out.meta.original_size, std::make_pair(128, 96));
    EXPECT_EQ(out.meta.input_size, std::make_pair(64, 32));
    EXPECT_NEAR(out.meta.scale.first, 0.5, 1e-9);
    EXPECT_NEAR(out.meta.scale.second, 32.0 / 96.0, 1e-9);
    EXPECT_EQ(out.meta.origin, std::make_pair(0, 0));
}

TEST(Nv12Passthrough, DeviceLegComposesPriorTransform) {
    DispatchReset reset;
    CapturingDispatch dispatch;
    dispatch.behavior = [](const std::string&, const std::vector<std::any>&) {
        return std::any();  // not a DspBufferRef — CPU path, meta still composed
    };
    detail::install_dsp_dispatch(&dispatch);

    // mid already maps (128, 96) -> (64, 48); the leg's meta must map
    // model coords all the way back to (128, 96)
    Frame mid = nv12_frame(128, 96).resize(64, 48, "stretch");
    Preprocessor pre(std::make_pair(32, 24), "RGB", "stretch", 114, false, "NHWC",
                     std::nullopt, true);
    Preprocessor::Output out = pre(mid);
    EXPECT_EQ(out.meta.original_size, std::make_pair(128, 96));
    EXPECT_NEAR(out.meta.scale.first, 32.0 / 128.0, 1e-9);
    EXPECT_NEAR(out.meta.scale.second, 24.0 / 96.0, 1e-9);
}

// ---- TestNv12Passthrough: identity ----------------------------------------------------------

TEST(Nv12Passthrough, IdentityKeepfdFrameReturnsSource) {
    // dummy handle: the identity branch only checks it is set (the frame
    // itself is what infer's zero-copy path wants)
    Frame frame;
    frame.sequence = 1;
    frame.width = 64;
    frame.height = 48;
    frame.format = "NV12";
    frame.handle = std::make_shared<FrameHandle>(
        std::vector<int>{}, std::array<std::uint32_t, 3>{0, 0, 0},
        std::array<std::uint32_t, 3>{0, 0, 0}, 1, FrameHandle::ReleaseFn{}, 64, 48,
        "NV12", 0);
    Preprocessor pre(std::make_pair(64, 48), "RGB", "stretch", 114, false, "NHWC",
                     std::nullopt, true);
    Preprocessor::Output out = pre(frame);
    ASSERT_TRUE(std::holds_alternative<Frame>(out.tensor));
    const Frame& tensor = std::get<Frame>(out.tensor);
    EXPECT_EQ(tensor.handle, frame.handle);  // the source itself, not a copy
    EXPECT_EQ(out.meta.scale, std::make_pair(1.0, 1.0));
    EXPECT_EQ(out.meta.origin, std::make_pair(0, 0));
    EXPECT_EQ(out.meta.color, "NV12");
}

TEST(Nv12Passthrough, IdentityNdarrayReturnsArrayNotFrame) {
    // a handle-less source is not a valid infer input — the array must
    // come back out
    Preprocessor pre(std::make_pair(64, 48), "RGB", "stretch", 114, false, "NHWC",
                     "NV12", true);
    Preprocessor::Output out = pre(cv::Mat::zeros(72, 64, CV_8UC1));
    ASSERT_TRUE(std::holds_alternative<cv::Mat>(out.tensor));
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.rows, 72);
    EXPECT_EQ(tensor.cols, 64);
}

TEST(Nv12Passthrough, IdentityInheritsPriorTransform) {
    Frame mid = nv12_frame(128, 96).resize(64, 48, "stretch");
    Preprocessor pre(std::make_pair(64, 48), "RGB", "stretch", 114, false, "NHWC",
                     std::nullopt, true);
    Preprocessor::Output out = pre(mid);
    ASSERT_TRUE(std::holds_alternative<cv::Mat>(out.tensor));
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.rows, 72);
    EXPECT_EQ(tensor.cols, 64);
    EXPECT_EQ(out.meta.original_size, std::make_pair(128, 96));
    EXPECT_NEAR(out.meta.scale.first, 0.5, 1e-9);
    EXPECT_NEAR(out.meta.scale.second, 0.5, 1e-9);
}

// ---- TestNv12Passthrough: fallbacks ---------------------------------------------------------

TEST(Nv12Passthrough, NonNv12SourceRejected) {
    Preprocessor pre(std::make_pair(64, 48), "RGB", "stretch", 114, false, "NHWC",
                     std::nullopt, true);
    const std::string msg = thrown_message([&] { pre(make_rgb_frame(64, 64)); });
    EXPECT_NE(msg.find("NV12"), std::string::npos);
}

TEST(Nv12Passthrough, DspDownFallsBackToCpu) {
    DispatchReset reset;
    CapturingDispatch dispatch;
    dispatch.behavior = [](const std::string&, const std::vector<std::any>&) -> std::any {
        throw DspError("dsp down");
    };
    detail::install_dsp_dispatch(&dispatch);

    Frame frame = nv12_frame(128, 96);
    Preprocessor pre(std::make_pair(64, 32), "RGB", "stretch", 114, false, "NHWC",
                     std::nullopt, true);
    Preprocessor::Output out = pre(frame);

    Frame expected_frame = frame.resize(64, 32, "stretch");
    cv::Mat expected = expected_frame.to_array();
    ASSERT_TRUE(std::holds_alternative<cv::Mat>(out.tensor));
    const cv::Mat& tensor = std::get<cv::Mat>(out.tensor);
    EXPECT_EQ(tensor.rows, 48);
    EXPECT_EQ(tensor.cols, 64);
    EXPECT_TRUE(same_bytes(tensor, expected));
    EXPECT_EQ(out.meta.color, "NV12");
    EXPECT_EQ(out.meta.original_size, std::make_pair(128, 96));
    EXPECT_EQ(out.meta.origin, std::make_pair(0, 0));
    // the router recorded the degradation for health()
    const RouterHealth health = get_default_router().health();
    ASSERT_FALSE(health.recent_degradations.empty());
    EXPECT_EQ(health.recent_degradations.back().op, "resize_nv12");
}

TEST(Nv12Passthrough, SoftwareRouteTakesCpuWithoutCallingDsp) {
    DispatchReset reset;
    PolicyReset policy(get_default_router());
    set_route_policy("software_only");
    CapturingDispatch dispatch;  // must not be called
    detail::install_dsp_dispatch(&dispatch);

    Frame frame = nv12_frame(128, 96);
    Preprocessor pre(std::make_pair(64, 32), "RGB", "stretch", 114, false, "NHWC",
                     std::nullopt, true);
    Preprocessor::Output out = pre(frame);

    Frame expected_frame = frame.resize(64, 32, "stretch");
    cv::Mat expected = expected_frame.to_array();
    EXPECT_TRUE(same_bytes(std::get<cv::Mat>(out.tensor), expected));
    EXPECT_NEAR(out.meta.scale.first, 0.5, 1e-9);
    EXPECT_NEAR(out.meta.scale.second, 32.0 / 96.0, 1e-9);
    EXPECT_EQ(dispatch.calls.size(), 0u);  // router said software; DSP stayed idle
}

TEST(Nv12Passthrough, HardwareOnlyPolicyRaises) {
    DispatchReset reset;
    PolicyReset policy(get_default_router());
    set_route_policy(RoutePolicy::HardwareOnly);
    CapturingDispatch dispatch;
    dispatch.behavior = [](const std::string&, const std::vector<std::any>&) -> std::any {
        throw DspError("dsp down");
    };
    detail::install_dsp_dispatch(&dispatch);

    Preprocessor pre(std::make_pair(64, 32), "RGB", "stretch", 114, false, "NHWC",
                     std::nullopt, true);
    const std::string msg = thrown_message([&] { pre(nv12_frame(128, 96)); });
    EXPECT_NE(msg.find("hardware-only"), std::string::npos);
}

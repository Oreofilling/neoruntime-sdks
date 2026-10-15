// test_device_af.cpp — native-autofocus codec parity with device.py.
//
// Pure-function tests: pb responses filled field-by-field, run through
// detail::device_codec, compared against AfJob/AfStatus/AfMeasurement;
// plus the >3-windows client-side rejection of af_windows_request.
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/device.hpp"

#include "detail/device_codec.hpp"

using namespace neoruntime_ipc_sdk;
namespace codec = detail::device_codec;
namespace pbns = aipc::device;  // not `pb`: modern protobuf owns the top-level pb namespace

TEST(AfJobFrom, CarriesAcceptJobIdMessage) {
    pbns::AfJobResponse resp;
    resp.set_accepted(true);
    resp.set_job_id(42);
    resp.set_message("af started");
    AfJob job = codec::af_job_from(resp);
    EXPECT_TRUE(job.accepted);
    EXPECT_EQ(job.job_id, 42ull);
    EXPECT_EQ(job.message, "af started");

    pbns::AfJobResponse rejected;
    rejected.set_accepted(false);
    rejected.set_job_id(0);
    rejected.set_message("lens busy");
    AfJob j2 = codec::af_job_from(rejected);
    EXPECT_FALSE(j2.accepted);
    EXPECT_EQ(j2.job_id, 0ull);
    EXPECT_EQ(j2.message, "lens busy");
}

TEST(AfStatusFrom, MapsAllEighteenFields) {
    pbns::AfStatusResponse r;
    r.set_job_id(7);
    r.set_operation("oneshot");
    r.set_state("running");
    r.set_progress(0.42);
    r.set_busy(true);
    r.set_anchor_valid(false);
    r.set_requested_ratio(2.5f);
    r.set_effective_ratio(2.4f);
    r.set_zoom_pos(1234);
    r.set_focus_pos(567);
    r.set_best_focus(566);
    r.set_metric(0.81);
    r.set_confidence(0.77);
    r.set_reproducibility(0.93);
    r.set_estimated_distance_m(4.2f);
    r.set_elapsed_ms(250);
    r.set_error_code(0);
    r.set_message("ok");
    AfStatus st = codec::af_status_from(r);
    EXPECT_EQ(st.job_id, 7ull);
    EXPECT_EQ(st.operation, "oneshot");
    EXPECT_EQ(st.state, "running");
    EXPECT_DOUBLE_EQ(st.progress, 0.42);
    EXPECT_TRUE(st.busy);
    EXPECT_FALSE(st.anchor_valid);
    EXPECT_FLOAT_EQ(st.requested_ratio, 2.5f);
    EXPECT_FLOAT_EQ(st.effective_ratio, 2.4f);
    EXPECT_EQ(st.zoom_pos, 1234);
    EXPECT_EQ(st.focus_pos, 567);
    EXPECT_EQ(st.best_focus, 566);
    EXPECT_DOUBLE_EQ(st.metric, 0.81);
    EXPECT_DOUBLE_EQ(st.confidence, 0.77);
    EXPECT_DOUBLE_EQ(st.reproducibility, 0.93);
    EXPECT_FLOAT_EQ(st.estimated_distance_m, 4.2f);
    EXPECT_EQ(st.elapsed_ms, 250ull);
    EXPECT_EQ(st.error_code, 0);
    EXPECT_EQ(st.message, "ok");
}

TEST(AfMeasurementFrom, CarriesSeries) {
    pbns::AfMeasurementResponse r;
    r.add_focus_energy(10);
    r.add_focus_energy(20);
    r.add_focus_energy(30);
    r.add_mean_luma(50);
    r.add_mean_luma(60);
    r.set_frame_id(1024);
    AfMeasurement m = codec::af_measurement_from(r);
    EXPECT_EQ(m.focus_energy, (std::vector<std::uint32_t>{10, 20, 30}));
    EXPECT_EQ(m.mean_luma, (std::vector<std::uint32_t>{50, 60}));
    EXPECT_EQ(m.frame_id, 1024ull);
}

TEST(AfWindowsRequest, MapsEnabledWindowsStream) {
    std::vector<AfWindow> wins{{10, 20, 100, 80}, {200, 300, 50, 60}};
    pbns::SetAfWindowsRequest req = codec::af_windows_request(true, wins, "main");
    EXPECT_TRUE(req.enabled());
    EXPECT_EQ(req.stream_id(), "main");
    ASSERT_EQ(req.windows_size(), 2);
    EXPECT_EQ(req.windows(0).x(), 10);
    EXPECT_EQ(req.windows(0).y(), 20);
    EXPECT_EQ(req.windows(0).w(), 100);
    EXPECT_EQ(req.windows(0).h(), 80);
    EXPECT_EQ(req.windows(1).x(), 200);
    EXPECT_EQ(req.windows(1).h(), 60);
}

TEST(AfWindowsRequest, AcceptsAtMostThree) {
    std::vector<AfWindow> three{{0, 0, 10, 10}, {1, 1, 10, 10}, {2, 2, 10, 10}};
    EXPECT_EQ(codec::af_windows_request(false, three, "main").windows_size(), 3);

    std::vector<AfWindow> four = three;
    four.push_back({3, 3, 10, 10});
    try {
        (void)codec::af_windows_request(true, four, "main");
        FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "at most 3 AF windows are supported");
    }
}

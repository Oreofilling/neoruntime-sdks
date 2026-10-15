// test_camera_codec.cpp — camera codec + pip_dest parity with camera.py.
//
// Pure-function tests, no daemon: pb responses are filled field-by-field,
// run through detail::camera_codec, and compared against the expected
// public structs. pip_dest expectations are computed from the Python
// reference (corner string tests + even alignment + the four ValueError
// shapes).
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/camera.hpp"
#include "neoruntime_ipc_sdk/camera_types.hpp"

#include "detail/camera_codec.hpp"

using namespace neoruntime_ipc_sdk;
namespace codec = detail::camera_codec;
namespace pb = aipc::camera;

// ---- pip_dest ------------------------------------------------------------------
TEST(PipDest, CornersAndEvenAlignment) {
    // x = 1920-32-480 = 1408, y = 1080-32-270 = 778 (both already even).
    EXPECT_EQ(pip_dest(1920, 1080, 480, 270, "bottom-right", 32),
              (std::pair<std::uint32_t, std::uint32_t>{1408, 778}));
    EXPECT_EQ(pip_dest(1920, 1080, 480, 270, "top-left", 16),
              (std::pair<std::uint32_t, std::uint32_t>{16, 16}));
    EXPECT_EQ(pip_dest(1920, 1080, 480, 270, "top-right", 16),
              (std::pair<std::uint32_t, std::uint32_t>{1424, 16}));
    EXPECT_EQ(pip_dest(1920, 1080, 480, 270, "bottom-left", 16),
              (std::pair<std::uint32_t, std::uint32_t>{16, 794}));
    // Odd offsets round DOWN to even (NV12 chroma alignment): x = 1 -> 0.
    EXPECT_EQ(pip_dest(100, 100, 33, 33, "top-left", 1),
              (std::pair<std::uint32_t, std::uint32_t>{0, 0}));
    EXPECT_EQ(pip_dest(100, 100, 33, 33, "bottom-right", 0),
              (std::pair<std::uint32_t, std::uint32_t>{66, 66}));  // 67 -> 66
}

TEST(PipDest, RejectsBadArguments) {
    EXPECT_THROW(pip_dest(100, 100, 10, 10, "middle", 0), std::invalid_argument);
    EXPECT_THROW(pip_dest(100, 100, 0, 10, "top-left", 0), std::invalid_argument);
    EXPECT_THROW(pip_dest(100, 100, 10, 0, "top-left", 0), std::invalid_argument);
    EXPECT_THROW(pip_dest(100, 100, 10, 10, "top-left", -1), std::invalid_argument);
    // inset + margin larger than the frame cannot fit.
    EXPECT_THROW(pip_dest(100, 100, 480, 270, "top-left", 32), std::invalid_argument);
    EXPECT_THROW(pip_dest(100, 100, 90, 10, "top-left", 11), std::invalid_argument);
}

// ---- inject mode / push_frame_request ---------------------------------------------
TEST(InjectMode, MapsAndRejects) {
    EXPECT_EQ(codec::inject_mode_from_string("replace"), pb::INJECT_REPLACE);
    EXPECT_EQ(codec::inject_mode_from_string("overlay"), pb::INJECT_OVERLAY);
    EXPECT_THROW(codec::inject_mode_from_string("bogus"), std::invalid_argument);
    EXPECT_THROW(codec::inject_mode_from_string(""), std::invalid_argument);
}

TEST(PushFrameRequest, CarriesEveryField) {
    PushFrameSpec spec;
    spec.buffer_id = 42;
    spec.width = 480;
    spec.height = 270;
    spec.stride = 512;
    spec.mode = "overlay";
    spec.pts_ns = 1234567890;
    spec.dest_x = 64;
    spec.dest_y = 66;
    spec.stream_id = "main";
    spec.end_of_stream = false;
    spec.session_id = "s-1";
    pb::PushFrameRequest req = codec::push_frame_request(spec);
    EXPECT_EQ(req.buffer_id(), 42u);
    EXPECT_EQ(req.width(), 480u);
    EXPECT_EQ(req.height(), 270u);
    EXPECT_EQ(req.stride(), 512u);
    EXPECT_EQ(req.mode(), pb::INJECT_OVERLAY);
    EXPECT_EQ(req.pts_ns(), 1234567890ull);
    EXPECT_EQ(req.dest_x(), 64u);
    EXPECT_EQ(req.dest_y(), 66u);
    EXPECT_EQ(req.stream_id(), "main");
    EXPECT_FALSE(req.end_of_stream());
    EXPECT_EQ(req.session_id(), "s-1");
    // Bad mode must throw before anything is sent.
    spec.mode = "nope";
    EXPECT_THROW(codec::push_frame_request(spec), std::invalid_argument);
}

TEST(InjectionConverters, ResultAndStatus) {
    pb::PushFrameResponse resp;
    resp.set_success(true);
    resp.set_message("ok");
    resp.set_error_code(0);
    resp.set_injected_frame_id(7);
    resp.set_accepted_frame_count(3);
    resp.set_session_id("s-1");
    resp.add_in_flight_buffer_ids(11);
    resp.add_in_flight_buffer_ids(12);
    InjectionResult r = codec::injection_result_from(resp);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.injected_frame_id, 7u);
    EXPECT_EQ(r.accepted_frame_count, 3u);
    EXPECT_EQ(r.session_id, "s-1");
    EXPECT_EQ(r.in_flight_buffer_ids, (std::vector<std::uint64_t>{11, 12}));

    pb::InjectionStatusResponse st;
    st.set_success(true);
    st.set_active(true);
    st.set_mode(pb::INJECT_OVERLAY);
    st.set_frames_injected(100);
    st.set_frames_dropped(2);
    st.set_queue_depth(1);
    st.set_session_id("s-2");
    st.add_in_flight_buffer_ids(5);
    st.set_reports_in_flight_buffers(true);
    InjectionStatus s = codec::injection_status_from(st);
    EXPECT_TRUE(s.active);
    EXPECT_EQ(s.mode, "overlay");
    EXPECT_EQ(s.frames_injected, 100u);
    EXPECT_EQ(s.frames_dropped, 2u);
    EXPECT_EQ(s.queue_depth, 1u);
    EXPECT_TRUE(s.reports_in_flight_buffers);
    EXPECT_EQ(s.in_flight_buffer_ids, (std::vector<std::uint64_t>{5}));

    // Default (unset) enum maps back to "replace" — proto default is 0.
    pb::InjectionStatusResponse st2;
    EXPECT_EQ(codec::injection_status_from(st2).mode, "replace");
}

// ---- infrared / presets --------------------------------------------------------------
TEST(InfraredStatus, MapsAllFields) {
    pb::InfraredStatusResponse resp;
    resp.set_mode("infrared");
    resp.set_transition("night");
    resp.set_output_source("profile");
    resp.set_auto_follow(true);
    resp.set_follow_active(true);
    resp.set_manual_override(false);
    resp.set_degraded(false);
    resp.set_requested_near_pwm(10);
    resp.set_requested_far_pwm(20);
    resp.set_applied_near_pwm(11);
    resp.set_applied_far_pwm(21);
    resp.set_zoom_ratio(1.5f);
    resp.set_active_profile("night");
    resp.set_selected_mode("infrared");
    resp.set_light_percent(42);
    resp.set_light_mv(900);
    resp.set_light_milli(1200);
    resp.set_light_valid(true);
    resp.set_night_enter(600);
    resp.set_day_enter(700);
    InfraredStatus s = codec::infrared_status_from(resp);
    EXPECT_EQ(s.mode, "infrared");
    EXPECT_EQ(s.transition, "night");
    EXPECT_EQ(s.output_source, "profile");
    EXPECT_TRUE(s.auto_follow);
    EXPECT_TRUE(s.follow_active);
    EXPECT_FALSE(s.manual_override);
    EXPECT_EQ(s.requested_near_pwm, 10u);
    EXPECT_EQ(s.requested_far_pwm, 20u);
    EXPECT_EQ(s.applied_near_pwm, 11u);
    EXPECT_EQ(s.applied_far_pwm, 21u);
    EXPECT_FLOAT_EQ(s.zoom_ratio, 1.5f);
    EXPECT_EQ(s.active_profile, "night");
    EXPECT_EQ(s.selected_mode, "infrared");
    EXPECT_EQ(s.light_percent, 42);
    EXPECT_EQ(s.light_mv, 900u);
    EXPECT_EQ(s.light_milli, 1200);
    EXPECT_TRUE(s.light_valid);
    EXPECT_EQ(s.night_enter, 600);
    EXPECT_EQ(s.day_enter, 700);
}

TEST(IrPresets, MapsList) {
    pb::IrPresetListResponse resp;
    auto* a = resp.add_presets();
    a->set_name("day");
    a->set_zoom_ratio(1.0f);
    a->set_near_pwm(30);
    a->set_far_pwm(40);
    auto* b = resp.add_presets();
    b->set_name("night");
    b->set_zoom_ratio(2.5f);
    b->set_near_pwm(60);
    b->set_far_pwm(70);
    std::vector<IrPreset> presets = codec::ir_presets_from(resp);
    ASSERT_EQ(presets.size(), 2u);
    EXPECT_EQ(presets[0].name, "day");
    EXPECT_FLOAT_EQ(presets[0].zoom_ratio, 1.0f);
    EXPECT_EQ(presets[0].near_pwm, 30u);
    EXPECT_EQ(presets[1].name, "night");
    EXPECT_EQ(presets[1].far_pwm, 70u);
}

// ---- stream status (28 wire fields -> 25-field struct) --------------------------------
TEST(StreamStatus, MapsExtendedFields) {
    pb::StreamStatusInfo s;
    s.set_stream_id("main");
    s.set_status("active");
    s.set_has_encoder(true);
    s.set_codec("h264");
    s.set_width(1920);
    s.set_height(1080);
    s.set_fps(30);
    s.set_bitrate_bps(4000000);
    s.set_gop(30);
    // wire 10-12 (ms_since_last_frame/measured_fps/status_detail) are
    // deliberately NOT part of the public struct.
    s.set_packets_published(900);
    s.set_queue_overflow_drops(1);
    s.set_client_send_drops(2);
    s.set_client_send_failures(3);
    s.set_client_disconnects(4);
    s.set_last_packet_seq(900);
    s.set_publisher_clients(2);
    s.set_bake_skips(10);
    s.set_strict_locked(20);
    s.set_strict_degraded(30);
    s.set_strict_skips(40);
    s.set_stream_epoch(7);
    s.set_overlay_layer_count(3);
    s.set_overlay_late_commands(50);
    s.set_overlay_epoch_rejects(60);
    s.set_overlay_no_binding_drops(70);
    StreamStatus st = codec::stream_status_from(s);
    EXPECT_EQ(st.stream_id, "main");
    EXPECT_EQ(st.status, "active");
    EXPECT_TRUE(st.has_encoder);
    EXPECT_EQ(st.codec, "h264");
    EXPECT_EQ(st.width, 1920u);
    EXPECT_EQ(st.height, 1080u);
    EXPECT_EQ(st.fps, 30u);
    EXPECT_EQ(st.bitrate_bps, 4000000u);
    EXPECT_EQ(st.gop, 30u);
    EXPECT_EQ(st.packets_published, 900u);
    EXPECT_EQ(st.queue_overflow_drops, 1u);
    EXPECT_EQ(st.client_send_drops, 2u);
    EXPECT_EQ(st.client_send_failures, 3u);
    EXPECT_EQ(st.client_disconnects, 4u);
    EXPECT_EQ(st.last_packet_seq, 900u);
    EXPECT_EQ(st.publisher_clients, 2u);
    EXPECT_EQ(st.bake_skips, 10u);
    EXPECT_EQ(st.strict_locked, 20u);
    EXPECT_EQ(st.strict_degraded, 30u);
    EXPECT_EQ(st.strict_skips, 40u);
    EXPECT_EQ(st.stream_epoch, 7u);
    EXPECT_EQ(st.overlay_layer_count, 3u);
    EXPECT_EQ(st.overlay_late_commands, 50u);
    EXPECT_EQ(st.overlay_epoch_rejects, 60u);
    EXPECT_EQ(st.overlay_no_binding_drops, 70u);
}

// ---- privacy mask -----------------------------------------------------------------------
TEST(PrivacyMask, RoundTrip) {
    PrivacyMaskSettings settings;
    settings.color = 0x00FF0000;
    settings.blur_radius = 8;
    settings.enabled = true;
    settings.dpm_enabled = true;
    settings.dpm_labels = "person,vehicle";
    settings.dpm_mode = "mosaic";
    settings.dpm_color = 0x000000FF;
    PrivacyMaskRegion r1;
    r1.id = "r1";
    r1.name = "yard";
    r1.enabled = true;
    r1.points_x = {0.1f, 0.9f, 0.5f};
    r1.points_y = {0.1f, 0.1f, 0.9f};
    PrivacyMaskRegion r2;
    r2.id = "r2";
    r2.enabled = false;
    r2.points_x = {0.0f, 0.2f};
    r2.points_y = {0.0f, 0.2f};
    settings.regions = {r1, r2};

    pb::PrivacyMaskConfig cfg = codec::privacy_mask_config_from(settings);
    PrivacyMaskSettings back = codec::privacy_mask_from(cfg);
    EXPECT_EQ(back.color, settings.color);
    EXPECT_EQ(back.blur_radius, settings.blur_radius);
    EXPECT_TRUE(back.enabled);
    EXPECT_EQ(back.dpm_labels, "person,vehicle");
    ASSERT_EQ(back.regions.size(), 2u);
    EXPECT_EQ(back.regions[0].name, "yard");
    EXPECT_TRUE(back.regions[0].enabled);
    EXPECT_EQ(back.regions[0].points_x, r1.points_x);
    EXPECT_EQ(back.regions[1].id, "r2");
    EXPECT_FALSE(back.regions[1].enabled);
}

TEST(PrivacyMask, PartialMergePreservesUntouched) {
    // Current device state: color 1, blur 4, enabled, one region, dpm off.
    pb::PrivacyMaskConfig current;
    current.set_color(1);
    current.set_blur_radius(4);
    current.set_enabled(true);
    auto* r = current.add_regions();
    r->set_id("keep");
    r->set_name("zone");
    r->set_enabled(true);
    r->add_points_x(0.5f);
    r->add_points_y(0.5f);
    current.set_dpm_enabled(false);
    current.set_dpm_labels("old");
    current.set_dpm_mode("blur");
    current.set_dpm_color(9);

    // Partial: only blur + dpm_labels. Regions not given -> preserved.
    pb::PrivacyMaskConfig merged = codec::merged_privacy_mask(
        current, std::nullopt, 8, std::nullopt, std::nullopt, std::nullopt,
        std::optional<std::string>{"person"}, std::nullopt, std::nullopt);
    EXPECT_EQ(merged.color(), 1u);           // untouched
    EXPECT_EQ(merged.blur_radius(), 8);      // overridden
    EXPECT_TRUE(merged.enabled());
    ASSERT_EQ(merged.regions_size(), 1);     // preserved
    EXPECT_EQ(merged.regions(0).id(), "keep");
    EXPECT_EQ(merged.dpm_labels(), "person");
    EXPECT_EQ(merged.dpm_mode(), "blur");

    // regions given -> full replacement.
    PrivacyMaskRegion only;
    only.id = "new";
    only.points_x = {0.0f, 1.0f};
    only.points_y = {0.0f, 1.0f};
    pb::PrivacyMaskConfig replaced =
        codec::merged_privacy_mask(current, std::nullopt, std::nullopt, std::nullopt,
                                   std::optional<std::vector<PrivacyMaskRegion>>{{only}},
                                   std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    ASSERT_EQ(replaced.regions_size(), 1);
    EXPECT_EQ(replaced.regions(0).id(), "new");
    // Everything else still untouched.
    EXPECT_EQ(replaced.color(), 1u);
    EXPECT_EQ(replaced.dpm_labels(), "old");
}

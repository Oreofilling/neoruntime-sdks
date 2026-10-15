// camera.cpp — CameraClient implementation. See camera.hpp.
//
// Port of camera.py over the `aipc.camera` CameraControl service. Every RPC here
// is unary. Two status shapes appear: ISPUpdateResponse nests a Status sub-message
// (.status()), while most others carry a top-level bool success / string message.
// pb::Empty is camera.proto's own Empty (aipc.camera.Empty), NOT google's.
#include "neoruntime_ipc_sdk/camera.hpp"

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/config.hpp"

#include "detail/camera_codec.hpp"
#include "detail/grpc_channel.hpp"
#include "detail/transport.hpp"
#include "camera-daemon/camera.grpc.pb.h"
#include "camera-daemon/camera.pb.h"

namespace neoruntime_ipc_sdk {

namespace pb = aipc::camera;

// ---- pip_dest (camera.py module-level helper) --------------------------------
std::pair<std::uint32_t, std::uint32_t> pip_dest(std::uint32_t frame_w, std::uint32_t frame_h,
                                                 std::uint32_t inset_w, std::uint32_t inset_h,
                                                 const std::string& corner,
                                                 std::int32_t margin) {
    static const char* kCorners[] = {"top-left", "top-right", "bottom-left", "bottom-right"};
    bool known = false;
    for (const char* c : kCorners) known = known || corner == c;
    if (!known) {
        throw std::invalid_argument(
            "corner must be one of ('top-left', 'top-right', 'bottom-left', 'bottom-right'), "
            "got '" + corner + "'");
    }
    if (inset_w == 0 || inset_h == 0) {
        throw std::invalid_argument("inset dims must be positive, got " +
                                    std::to_string(inset_w) + "x" + std::to_string(inset_h));
    }
    if (margin < 0) {
        throw std::invalid_argument("margin must be >= 0, got " + std::to_string(margin));
    }
    if (inset_w + static_cast<std::uint32_t>(margin) > frame_w ||
        inset_h + static_cast<std::uint32_t>(margin) > frame_h) {
        throw std::invalid_argument("inset " + std::to_string(inset_w) + "x" +
                                    std::to_string(inset_h) + " + margin " +
                                    std::to_string(margin) + " does not fit inside " +
                                    std::to_string(frame_w) + "x" + std::to_string(frame_h));
    }
    // corner.endswith("left") / startswith("top") — same string tests as Python.
    const std::uint32_t x =
        corner.size() >= 4 && corner.compare(corner.size() - 4, 4, "left") == 0
            ? static_cast<std::uint32_t>(margin)
            : frame_w - static_cast<std::uint32_t>(margin) - inset_w;
    const std::uint32_t y =
        corner.compare(0, 3, "top") == 0 ? static_cast<std::uint32_t>(margin)
                                         : frame_h - static_cast<std::uint32_t>(margin) - inset_h;
    // Even alignment: NV12 UV plane is 2x2 subsampled per luma sample.
    return {x & ~1u, y & ~1u};
}

struct CameraClient::Impl : detail::GrpcClientBase<pb::CameraControl> {
    explicit Impl(std::string endpoint)
        : GrpcClientBase(std::move(endpoint)) {}
};

CameraClient::CameraClient(std::string endpoint)
    : impl_(std::make_unique<Impl>(
          endpoint.empty() ? Config::get_camera_control_endpoint() : std::move(endpoint))) {}

CameraClient::~CameraClient() = default;
CameraClient::CameraClient(CameraClient&&) noexcept = default;
CameraClient& CameraClient::operator=(CameraClient&&) noexcept = default;

void CameraClient::connect() { impl_->connect(); }

void CameraClient::close() { impl_->close(); }

bool CameraClient::connected() const noexcept { return impl_->connected(); }

// ---- ISP -------------------------------------------------------------------
void CameraClient::set_isp(const ISPConfig& cfg) {
    pb::ISPUpdateRequest req;
    if (cfg.brightness >= 0) req.set_brightness(cfg.brightness);
    if (cfg.contrast >= 0) req.set_contrast(cfg.contrast);
    if (cfg.saturation >= 0) req.set_saturation(cfg.saturation);
    if (cfg.sharpness >= 0) req.set_sharpness(cfg.sharpness);
    if (cfg.manual_mode) req.set_manual_mode(*cfg.manual_mode);
    if (cfg.auto_exposure) req.set_auto_exposure(*cfg.auto_exposure);
    if (cfg.backlight >= 0) req.set_backlight(cfg.backlight);
    if (cfg.exposure_time_us >= 0) req.set_exposure_time_us(cfg.exposure_time_us);
    if (cfg.gain >= 0) req.set_gain(cfg.gain);
    if (cfg.noise_reduction >= 0) req.set_noise_reduction(cfg.noise_reduction);
    if (cfg.wdr_value >= 0) req.set_wdr_value(cfg.wdr_value);
    if (cfg.powerline_freq >= 0) req.set_powerline_freq(cfg.powerline_freq);
    if (cfg.awb_index >= 0) req.set_awb_index(cfg.awb_index);

    pb::ISPUpdateResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().UpdateISPSettings(&ctx, req, &resp), "UpdateISPSettings");
    detail::require_success(resp.status().success(), resp.status().message(), "UpdateISPSettings");
}

ISPConfig CameraClient::get_isp() {
    pb::ISPConfigResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetISPConfig(&ctx, pb::Empty{}, &resp), "GetISPConfig");
    detail::require_success(resp.success(), resp.message(), "GetISPConfig");

    const auto& c = resp.current();
    ISPConfig cfg;
    cfg.brightness = c.brightness();
    cfg.contrast = c.contrast();
    cfg.saturation = c.saturation();
    cfg.sharpness = c.sharpness();
    cfg.manual_mode = c.has_manual_mode() ? std::optional<bool>{c.manual_mode()} : std::nullopt;
    cfg.auto_exposure = c.has_auto_exposure() ? std::optional<bool>{c.auto_exposure()} : std::nullopt;
    cfg.backlight = c.backlight();
    cfg.exposure_time_us = c.exposure_time_us();
    cfg.gain = c.gain();
    cfg.noise_reduction = c.noise_reduction();
    cfg.wdr_value = c.wdr_value();
    cfg.powerline_freq = c.powerline_freq();
    cfg.awb_index = c.awb_index();
    return cfg;
}

// ---- Transform -------------------------------------------------------------
TransformConfig CameraClient::get_transform() {
    pb::TransformConfig resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetTransformConfig(&ctx, pb::Empty{}, &resp),
                       "GetTransformConfig");
    return TransformConfig{
        static_cast<int>(resp.rotation()),
        static_cast<int>(resp.flip()),
        resp.dewarp(),
        resp.grayscale(),
    };
}

void CameraClient::set_transform(const TransformConfig& cfg) {
    pb::TransformConfig req;
    req.set_rotation(static_cast<std::uint32_t>(cfg.rotation));
    req.set_flip(static_cast<std::uint32_t>(cfg.flip));
    req.set_dewarp(cfg.dewarp);
    req.set_grayscale(cfg.grayscale);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetTransformConfig(&ctx, req, &resp), "SetTransformConfig");
    detail::require_success(resp.success(), resp.message(), "SetTransformConfig");
}

// ---- Encoder ---------------------------------------------------------------
void CameraClient::set_encoder(const std::string& stream_name, std::uint32_t bitrate_bps,
                               std::uint32_t framerate, std::uint32_t gop) {
    pb::EncoderConfigRequest req;
    req.set_stream_name(stream_name);
    req.set_bitrate_bps(bitrate_bps);
    req.set_framerate(framerate);
    req.set_gop(gop);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().UpdateEncoderConfig(&ctx, req, &resp), "UpdateEncoderConfig");
    detail::require_success(resp.success(), resp.message(), "UpdateEncoderConfig");
}

EncoderReconfigResult CameraClient::reconfigure_encoder(const std::string& stream_name,
                                                       std::uint32_t width, std::uint32_t height,
                                                       const std::string& codec,
                                                       std::uint32_t bitrate_bps,
                                                       std::uint32_t fps, std::uint32_t gop) {
    pb::EncoderReconfigRequest req;
    req.set_stream_name(stream_name);
    req.set_width(width);
    req.set_height(height);
    req.set_codec(codec);
    req.set_bitrate_bps(bitrate_bps);
    req.set_fps(fps);
    req.set_gop(gop);
    pb::EncoderReconfigResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().ReconfigureEncoder(&ctx, req, &resp), "ReconfigureEncoder");
    detail::require_success(resp.success(), resp.message(), "ReconfigureEncoder");
    return EncoderReconfigResult{resp.success(), resp.message(), resp.interrupt_ms()};
}

// ---- RTSP ------------------------------------------------------------------
void CameraClient::set_rtsp_enabled(bool enabled) {
    pb::RtspEnabledRequest req;
    req.set_enabled(enabled);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetRtspEnabled(&ctx, req, &resp), "SetRtspEnabled");
    detail::require_success(resp.success(), resp.message(), "SetRtspEnabled");
}

// ---- OSD -------------------------------------------------------------------
void CameraClient::set_osd(const std::vector<OsdStreamConfig>& streams) {
    pb::OsdConfigRequest req;
    for (const auto& s : streams) {
        auto* sc = req.add_streams();
        sc->set_stream_name(s.stream_name);
        for (const auto& t : s.text_overlays) {
            auto* tc = sc->add_text_overlays();
            tc->set_id(t.id);
            tc->set_text(t.text);
            tc->set_x(t.x);
            tc->set_y(t.y);
            tc->set_font_size(t.font_size);
            tc->set_text_color(t.text_color);
            tc->set_enabled(t.enabled);
            tc->set_h_align(t.h_align);
            tc->set_v_align(t.v_align);
        }
        for (const auto& d : s.datetime_overlays) {
            auto* dc = sc->add_datetime_overlays();
            dc->set_id(d.id);
            dc->set_x(d.x);
            dc->set_y(d.y);
            dc->set_format(d.format);
            dc->set_font_size(d.font_size);
            dc->set_text_color(d.text_color);
            dc->set_enabled(d.enabled);
            dc->set_h_align(d.h_align);
            dc->set_v_align(d.v_align);
        }
        // Python's dict path cannot carry image overlays; the typed struct
        // forwards them when populated (empty vector == Python behavior).
        for (const auto& i : s.image_overlays) {
            auto* ic = sc->add_image_overlays();
            ic->set_id(i.id);
            ic->set_image_path(i.image_path);
            ic->set_x(i.x);
            ic->set_y(i.y);
            ic->set_width(i.width);
            ic->set_height(i.height);
            ic->set_enabled(i.enabled);
            ic->set_h_align(i.h_align);
            ic->set_v_align(i.v_align);
        }
    }
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().UpdateOsdConfig(&ctx, req, &resp), "UpdateOsdConfig");
    detail::require_success(resp.success(), resp.message(), "UpdateOsdConfig");
}

// ---- OSD read-back ------------------------------------------------------------
std::vector<OsdStreamConfig> CameraClient::get_osd() {
    pb::OsdConfigResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetOsdConfig(&ctx, pb::Empty{}, &resp), "GetOsdConfig");
    std::vector<OsdStreamConfig> out;
    out.reserve(resp.streams_size());
    for (const auto& sc : resp.streams()) {
        OsdStreamConfig s;
        s.stream_name = sc.stream_name();
        for (const auto& tc : sc.text_overlays()) {
            OsdTextOverlay t;
            t.id = tc.id();
            t.text = tc.text();
            t.x = tc.x();
            t.y = tc.y();
            t.font_size = tc.font_size();
            t.text_color = tc.text_color();
            t.enabled = tc.enabled();
            t.h_align = tc.h_align();
            t.v_align = tc.v_align();
            s.text_overlays.push_back(std::move(t));
        }
        for (const auto& dc : sc.datetime_overlays()) {
            OsdDateTimeOverlay d;
            d.id = dc.id();
            d.x = dc.x();
            d.y = dc.y();
            d.format = dc.format();
            d.font_size = dc.font_size();
            d.text_color = dc.text_color();
            d.enabled = dc.enabled();
            d.h_align = dc.h_align();
            d.v_align = dc.v_align();
            s.datetime_overlays.push_back(std::move(d));
        }
        for (const auto& ic : sc.image_overlays()) {
            OsdImageOverlay i;
            i.id = ic.id();
            i.image_path = ic.image_path();
            i.x = ic.x();
            i.y = ic.y();
            i.width = ic.width();
            i.height = ic.height();
            i.enabled = ic.enabled();
            i.h_align = ic.h_align();
            i.v_align = ic.v_align();
            s.image_overlays.push_back(std::move(i));
        }
        out.push_back(std::move(s));
    }
    return out;
}

// ---- AI overlay ------------------------------------------------------------
void CameraClient::set_ai_overlay(bool enabled, bool show_label, bool show_confidence,
                                  int line_thickness) {
    pb::AiOverlayConfig req;
    req.set_enabled(enabled);
    req.set_show_label(show_label);
    req.set_show_confidence(show_confidence);
    req.set_line_thickness(line_thickness);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().UpdateAiOverlay(&ctx, req, &resp), "UpdateAiOverlay");
    detail::require_success(resp.success(), resp.message(), "UpdateAiOverlay");
}

// ---- Stream management -----------------------------------------------------
std::vector<StreamStatus> CameraClient::get_stream_status(std::optional<double> timeout_s) {
    pb::GetStreamStatusResponse resp;
    grpc::ClientContext ctx;
    detail::camera_codec::apply_timeout(ctx, timeout_s);
    detail::check_grpc(impl_->stub().GetStreamStatus(&ctx, pb::GetStreamStatusRequest{}, &resp),
                       "GetStreamStatus");
    std::vector<StreamStatus> out;
    out.reserve(resp.streams_size());
    for (const auto& s : resp.streams()) {
        out.push_back(detail::camera_codec::stream_status_from(s));
    }
    return out;
}

void CameraClient::add_stream(const std::string& stream_id, std::uint32_t width,
                             std::uint32_t height, std::uint32_t fps, const std::string& codec,
                             std::uint32_t bitrate, std::uint32_t gop) {
    pb::AddStreamRequest req;
    req.set_stream_id(stream_id);
    req.set_width(width);
    req.set_height(height);
    req.set_fps(fps);
    req.set_codec(codec);
    req.set_bitrate(bitrate);
    req.set_gop(gop);
    pb::StreamOperationResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().AddStream(&ctx, req, &resp), "AddStream");
    detail::require_success(resp.success(), resp.message(), "AddStream");
}

void CameraClient::remove_stream(const std::string& stream_name) {
    pb::RemoveStreamRequest req;
    req.set_stream_name(stream_name);
    pb::StreamOperationResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().RemoveStream(&ctx, req, &resp), "RemoveStream");
    detail::require_success(resp.success(), resp.message(), "RemoveStream");
}

// ---- Pipeline reconfiguration ----------------------------------------------
EncoderReconfigResult CameraClient::reconfigure_pipeline(
    const std::vector<PipelineStreamConfig>& streams) {
    pb::ReconfigurePipelineRequest req;
    for (const auto& s : streams) {
        auto* sc = req.add_streams();
        sc->set_stream_id(s.stream_id);
        sc->set_input_width(s.input_width);
        sc->set_input_height(s.input_height);
        sc->set_input_framerate(s.input_framerate);
        sc->set_codec(s.codec);
        sc->set_encoder_width(s.encoder_width);
        sc->set_encoder_height(s.encoder_height);
        sc->set_encoder_framerate(s.encoder_framerate);
        sc->set_encoder_bitrate(s.encoder_bitrate);
        sc->set_encoder_gop(s.encoder_gop);
    }
    pb::ReconfigurePipelineResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().ReconfigurePipeline(&ctx, req, &resp), "ReconfigurePipeline");
    detail::require_success(resp.success(), resp.message(), "ReconfigurePipeline");
    return EncoderReconfigResult{resp.success(), resp.message(), resp.interrupt_ms()};
}

// ---- Profiles --------------------------------------------------------------
std::string CameraClient::get_profile() {
    pb::GetProfileResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetProfile(&ctx, pb::Empty{}, &resp), "GetProfile");
    return resp.profile_name();
}

std::pair<std::vector<std::string>, std::string> CameraClient::list_profiles() {
    pb::ListProfilesResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().ListProfiles(&ctx, pb::Empty{}, &resp), "ListProfiles");
    std::vector<std::string> names;
    names.reserve(resp.profiles_size());
    for (const auto& p : resp.profiles()) names.push_back(p);
    return {std::move(names), resp.current_profile()};
}

EncoderReconfigResult CameraClient::switch_profile(const std::string& name) {
    pb::SwitchProfileRequest req;
    req.set_profile_name(name);
    pb::EncoderReconfigResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SwitchProfile(&ctx, req, &resp), "SwitchProfile");
    detail::require_success(resp.success(), resp.message(), "SwitchProfile");
    return EncoderReconfigResult{resp.success(), resp.message(), resp.interrupt_ms()};
}

void CameraClient::backup_profile(const std::string& path) {
    pb::BackupProfileRequest req;
    req.set_path(path);
    pb::BackupProfileResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().BackupProfile(&ctx, req, &resp), "BackupProfile");
    detail::require_success(resp.success(), resp.message(), "BackupProfile");
}

// ---- Sensor / capabilities / hardware --------------------------------------
SensorInfo CameraClient::get_sensor_info(std::uint32_t sensor_index) {
    pb::GetSensorInfoRequest req;
    req.set_sensor_index(sensor_index);
    pb::SensorInfoResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetSensorInfo(&ctx, req, &resp), "GetSensorInfo");
    return SensorInfo{
        resp.available(),
        resp.sensor_model(),
        resp.i2c_bus(),
        resp.i2c_address(),
        resp.pixel_format(),
    };
}

Capabilities CameraClient::get_capabilities() {
    pb::CapabilitiesResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetCapabilities(&ctx, pb::Empty{}, &resp), "GetCapabilities");
    return Capabilities{
        resp.has_video(),   resp.has_codec(),   resp.has_led(),  resp.has_sensor(),
        resp.has_mcu(),     resp.has_env_ctrl(), resp.has_alarm(), resp.has_rs485(),
        resp.has_osd(),     resp.has_draw(),    resp.has_audio(),
    };
}

HardwareStatus CameraClient::get_hardware_status() {
    pb::DeviceHardwareStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetDeviceHardwareStatus(&ctx, pb::Empty{}, &resp),
                       "GetDeviceHardwareStatus");
    detail::require_success(resp.success(), resp.message(), "GetDeviceHardwareStatus");
    return HardwareStatus{
        resp.light_sensor_mv(), resp.light_sensor_lux(), resp.mcu_temp_millic(),
        resp.ain_mv(),          resp.mcu_version(),      resp.white_light_duty(),
        resp.ir_led_duty(),     resp.ircut_mode(),
    };
}

// ---- LED -------------------------------------------------------------------
void CameraClient::set_led_duty(std::uint32_t led_id, std::uint32_t duty_percent) {
    pb::SetLedDutyRequest req;
    req.set_led_id(led_id);
    req.set_duty_percent(duty_percent);
    pb::LedStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetLedDuty(&ctx, req, &resp), "SetLedDuty");
    detail::require_success(resp.success(), resp.message(), "SetLedDuty");
}

std::uint32_t CameraClient::get_led_duty(std::uint32_t led_id) {
    pb::GetLedDutyRequest req;
    req.set_led_id(led_id);
    pb::LedStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetLedDuty(&ctx, req, &resp), "GetLedDuty");
    detail::require_success(resp.success(), resp.message(), "GetLedDuty");
    return resp.duty_percent();
}

// ---- IR-Cut ----------------------------------------------------------------
std::uint32_t CameraClient::set_ircut(std::uint32_t mode) {
    pb::SetIrCutRequest req;
    req.set_mode(mode);
    pb::SetIrCutResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetIrCut(&ctx, req, &resp), "SetIrCut");
    detail::require_success(resp.success(), resp.message(), "SetIrCut");
    return resp.current_mode();
}

std::uint32_t CameraClient::get_ircut() {
    pb::SetIrCutResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetIrCut(&ctx, pb::Empty{}, &resp), "GetIrCut");
    return resp.current_mode();
}

// ---- MCU raw ---------------------------------------------------------------
std::string CameraClient::mcu_raw_request(std::uint32_t cmd, const std::string& payload) {
    pb::McuRawRequestMessage req;
    req.set_cmd(cmd);
    req.set_payload(payload);
    pb::McuRawResponseMessage resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().McuRawRequest(&ctx, req, &resp), "McuRawRequest");
    detail::require_success(resp.success(), resp.message(), "McuRawRequest");
    return resp.payload();
}

// ---- Environment control ---------------------------------------------------
bool CameraClient::set_fan(bool enable) {
    pb::EnvCtrlRequest req;
    req.set_enable(enable);
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetFan(&ctx, req, &resp), "SetFan");
    detail::require_success(resp.success(), resp.message(), "SetFan");
    return resp.enabled();
}

EnvStatus CameraClient::get_fan() {
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetFan(&ctx, pb::Empty{}, &resp), "GetFan");
    return EnvStatus{resp.enabled()};
}

bool CameraClient::set_heat(bool enable) {
    pb::EnvCtrlRequest req;
    req.set_enable(enable);
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetHeat(&ctx, req, &resp), "SetHeat");
    detail::require_success(resp.success(), resp.message(), "SetHeat");
    return resp.enabled();
}

EnvStatus CameraClient::get_heat() {
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetHeat(&ctx, pb::Empty{}, &resp), "GetHeat");
    return EnvStatus{resp.enabled()};
}

bool CameraClient::set_radar(bool enable) {
    pb::EnvCtrlRequest req;
    req.set_enable(enable);
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetRadar(&ctx, req, &resp), "SetRadar");
    detail::require_success(resp.success(), resp.message(), "SetRadar");
    return resp.enabled();
}

EnvStatus CameraClient::get_radar() {
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetRadar(&ctx, pb::Empty{}, &resp), "GetRadar");
    return EnvStatus{resp.enabled()};
}

// ---- Alarm I/O -------------------------------------------------------------
bool CameraClient::set_alarm_out(std::uint32_t channel, bool enable) {
    pb::AlarmOutRequest req;
    req.set_channel(channel);
    req.set_enable(enable);
    pb::AlarmOutStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetAlarmOut(&ctx, req, &resp), "SetAlarmOut");
    detail::require_success(resp.success(), resp.message(), "SetAlarmOut");
    return resp.enabled();
}

bool CameraClient::get_alarm_out(std::uint32_t channel) {
    pb::AlarmOutRequest req;
    req.set_channel(channel);
    pb::AlarmOutStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetAlarmOut(&ctx, req, &resp), "GetAlarmOut");
    detail::require_success(resp.success(), resp.message(), "GetAlarmOut");
    return resp.enabled();
}

AlarmOutputs CameraClient::get_alarm_outputs() {
    pb::AlarmOutputsState resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetAlarmOutputs(&ctx, pb::Empty{}, &resp), "GetAlarmOutputs");
    detail::require_success(resp.success(), resp.message(), "GetAlarmOutputs");
    return AlarmOutputs{
        resp.alarm_out0(), resp.alarm_out1(), resp.wiegand0(), resp.wiegand1(),
    };
}

// ---- RS485 -----------------------------------------------------------------
void CameraClient::rs485_init(std::uint32_t baudrate, const std::string& config) {
    pb::Rs485InitRequest req;
    req.set_baudrate(baudrate);
    req.set_config(config);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().Rs485Init(&ctx, req, &resp), "Rs485Init");
    detail::require_success(resp.success(), resp.message(), "Rs485Init");
}

void CameraClient::rs485_deinit() {
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().Rs485Deinit(&ctx, pb::Empty{}, &resp), "Rs485Deinit");
    detail::require_success(resp.success(), resp.message(), "Rs485Deinit");
}

void CameraClient::rs485_tx(const std::string& data) {
    pb::Rs485TxRequest req;
    req.set_data(data);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().Rs485Tx(&ctx, req, &resp), "Rs485Tx");
    detail::require_success(resp.success(), resp.message(), "Rs485Tx");
}

// ---- Day/night imaging & infrared --------------------------------------------
InfraredStatus CameraClient::set_imaging_mode(const std::string& mode) {
    if (mode != "day" && mode != "infrared" && mode != "auto") {
        throw std::invalid_argument("mode must be day/infrared/auto, got '" + mode + "'");
    }
    pb::ImagingModeRequest req;
    req.set_mode(mode);
    pb::InfraredStatusResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetImagingMode(&ctx, req, &resp), "SetImagingMode");
    detail::require_success(resp.success(), resp.message(), "SetImagingMode");
    return detail::camera_codec::infrared_status_from(resp);
}

InfraredStatus CameraClient::get_infrared_status() {
    pb::InfraredStatusResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetInfraredStatus(&ctx, pb::Empty{}, &resp),
                       "GetInfraredStatus");
    detail::require_success(resp.success(), resp.message(), "GetInfraredStatus");
    return detail::camera_codec::infrared_status_from(resp);
}

InfraredStatus CameraClient::set_infrared_settings(std::optional<bool> auto_follow,
                                                   std::optional<std::uint32_t> near_pwm,
                                                   std::optional<std::uint32_t> far_pwm,
                                                   std::optional<std::int32_t> night_enter,
                                                   std::optional<std::int32_t> day_enter) {
    pb::InfraredSettingsRequest req;
    if (auto_follow.has_value()) req.set_auto_follow(*auto_follow);
    if (near_pwm.has_value()) req.set_near_pwm(*near_pwm);
    if (far_pwm.has_value()) req.set_far_pwm(*far_pwm);
    if (night_enter.has_value()) req.set_night_enter(*night_enter);
    if (day_enter.has_value()) req.set_day_enter(*day_enter);
    pb::InfraredStatusResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetInfraredSettings(&ctx, req, &resp), "SetInfraredSettings");
    detail::require_success(resp.success(), resp.message(), "SetInfraredSettings");
    return detail::camera_codec::infrared_status_from(resp);
}

InfraredStatus CameraClient::clear_infrared_manual() {
    pb::InfraredStatusResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().ClearInfraredManual(&ctx, pb::Empty{}, &resp),
                       "ClearInfraredManual");
    detail::require_success(resp.success(), resp.message(), "ClearInfraredManual");
    return detail::camera_codec::infrared_status_from(resp);
}

// ---- IR presets ----------------------------------------------------------------
std::vector<IrPreset> CameraClient::list_ir_presets() {
    pb::IrPresetListResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().ListIrPresets(&ctx, pb::Empty{}, &resp), "ListIrPresets");
    detail::require_success(resp.success(), resp.message(), "ListIrPresets");
    return detail::camera_codec::ir_presets_from(resp);
}

std::vector<IrPreset> CameraClient::save_ir_preset(const std::string& name, float zoom_ratio,
                                                   std::uint32_t near_pwm,
                                                   std::uint32_t far_pwm) {
    pb::IrPreset req;
    req.set_name(name);
    req.set_zoom_ratio(zoom_ratio);
    req.set_near_pwm(near_pwm);
    req.set_far_pwm(far_pwm);
    pb::IrPresetListResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SaveIrPreset(&ctx, req, &resp), "SaveIrPreset");
    detail::require_success(resp.success(), resp.message(), "SaveIrPreset");
    return detail::camera_codec::ir_presets_from(resp);
}

std::vector<IrPreset> CameraClient::delete_ir_preset(const std::string& name) {
    pb::DeleteIrPresetRequest req;
    req.set_name(name);
    pb::IrPresetListResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().DeleteIrPreset(&ctx, req, &resp), "DeleteIrPreset");
    detail::require_success(resp.success(), resp.message(), "DeleteIrPreset");
    return detail::camera_codec::ir_presets_from(resp);
}

// ---- Privacy mask ---------------------------------------------------------------
PrivacyMaskSettings CameraClient::get_privacy_mask() {
    pb::PrivacyMaskConfig cfg;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetPrivacyMaskConfig(&ctx, pb::Empty{}, &cfg),
                       "GetPrivacyMaskConfig");
    return detail::camera_codec::privacy_mask_from(cfg);
}

void CameraClient::set_privacy_mask(const PrivacyMaskSettings& settings) {
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetPrivacyMaskConfig(
                           &ctx, detail::camera_codec::privacy_mask_config_from(settings), &resp),
                       "SetPrivacyMaskConfig");
    detail::require_success(resp.success(), resp.message(), "SetPrivacyMaskConfig");
}

void CameraClient::set_privacy_mask(std::optional<std::uint32_t> color,
                                    std::optional<std::int32_t> blur_radius,
                                    std::optional<bool> enabled,
                                    std::optional<std::vector<PrivacyMaskRegion>> regions,
                                    std::optional<bool> dpm_enabled,
                                    std::optional<std::string> dpm_labels,
                                    std::optional<std::string> dpm_mode,
                                    std::optional<std::uint32_t> dpm_color) {
    // Merge semantics: read current config, overlay only the given fields —
    // partial updates never wipe existing regions unless `regions` is given.
    pb::PrivacyMaskConfig current;
    {
        grpc::ClientContext ctx;
        detail::check_grpc(impl_->stub().GetPrivacyMaskConfig(&ctx, pb::Empty{}, &current),
                           "GetPrivacyMaskConfig");
    }
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetPrivacyMaskConfig(
                           &ctx,
                           detail::camera_codec::merged_privacy_mask(
                               current, color, blur_radius, enabled, regions, dpm_enabled,
                               dpm_labels, dpm_mode, dpm_color),
                           &resp),
                       "SetPrivacyMaskConfig");
    detail::require_success(resp.success(), resp.message(), "SetPrivacyMaskConfig");
}

// ---- Raw config fields ------------------------------------------------------------
std::string CameraClient::get_config_field(const std::string& field_path) {
    pb::GetConfigFieldRequest req;
    req.set_field_path(field_path);
    pb::GetConfigFieldResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().GetConfigField(&ctx, req, &resp), "GetConfigField");
    detail::require_success(resp.success(), resp.message(), "GetConfigField");
    return resp.value();
}

void CameraClient::set_config_field(const std::string& field_path, const std::string& value) {
    // Python leaves `type` unset (proto default) — mirrored.
    pb::SetConfigFieldRequest req;
    req.set_field_path(field_path);
    req.set_value(value);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub().SetConfigField(&ctx, req, &resp), "SetConfigField");
    detail::require_success(resp.success(), resp.message(), "SetConfigField");
}

// ---- App frame injection -----------------------------------------------------------
InjectionResult CameraClient::push_frame(const PushFrameSpec& spec,
                                         std::optional<double> timeout_s) {
    // Client-side mode validation (throws std::invalid_argument) happens in
    // the codec, mirroring camera.py's pre-RPC _INJECT_MODES lookup.
    pb::PushFrameRequest req = detail::camera_codec::push_frame_request(spec);
    pb::PushFrameResponse resp;
    grpc::ClientContext ctx;
    detail::camera_codec::apply_timeout(ctx, timeout_s);
    detail::check_grpc(impl_->stub().PushFrame(&ctx, req, &resp), "PushFrame");
    InjectionResult result = detail::camera_codec::injection_result_from(resp);
    if (!result.success) {
        throw detail::RpcError("PushFrame failed: " + result.message +
                               " (error_code=" + std::to_string(result.error_code) + ")");
    }
    return result;
}

InjectionResult CameraClient::push_frame_stream(const std::vector<PushFrameSpec>& frames,
                                                std::optional<double> timeout_s) {
    // Validate + build every request up front: a bad mode must throw
    // client-side before any RPC starts (the generator in camera.py raises
    // into the stream before the daemon sees a byte).
    std::vector<pb::PushFrameRequest> reqs;
    reqs.reserve(frames.size());
    for (const auto& spec : frames) {
        reqs.push_back(detail::camera_codec::push_frame_request(spec));
    }
    pb::PushFrameResponse resp;
    grpc::ClientContext ctx;
    detail::camera_codec::apply_timeout(ctx, timeout_s);
    std::unique_ptr<grpc::ClientWriter<pb::PushFrameRequest>> writer(
        impl_->stub().PushFrameStream(&ctx, &resp));
    for (const auto& req : reqs) {
        if (!writer->Write(req)) break;  // stream broken; Finish() carries why
    }
    writer->WritesDone();
    detail::check_grpc(writer->Finish(), "PushFrameStream");
    InjectionResult result = detail::camera_codec::injection_result_from(resp);
    if (!result.success) {
        throw detail::RpcError(
            "PushFrameStream failed after " + std::to_string(result.accepted_frame_count) +
            " accepted frame(s): " + result.message + " (error_code=" +
            std::to_string(result.error_code) + ")");
    }
    return result;
}

InjectionStatus CameraClient::injection_status(std::optional<double> timeout_s) {
    pb::InjectionStatusResponse resp;
    grpc::ClientContext ctx;
    detail::camera_codec::apply_timeout(ctx, timeout_s);
    detail::check_grpc(impl_->stub().GetInjectionStatus(&ctx, pb::Empty{}, &resp),
                       "GetInjectionStatus");
    // Python: raise RuntimeError(f"GetInjectionStatus failed: {resp.message}")
    // — exactly require_success's format.
    detail::require_success(resp.success(), resp.message(), "GetInjectionStatus");
    return detail::camera_codec::injection_status_from(resp);
}

void CameraClient::stop_injection(std::optional<double> timeout_s) {
    pb::InjectionStatusResponse resp;
    grpc::ClientContext ctx;
    detail::camera_codec::apply_timeout(ctx, timeout_s);
    detail::check_grpc(impl_->stub().StopInjection(&ctx, pb::Empty{}, &resp), "StopInjection");
    detail::require_success(resp.success(), resp.message(), "StopInjection");
}

}  // namespace neoruntime_ipc_sdk

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

#include "detail/grpc_channel.hpp"
#include "camera-daemon/camera.grpc.pb.h"
#include "camera-daemon/camera.pb.h"

namespace neoruntime_ipc_sdk {

namespace pb = aipc::camera;

struct CameraClient::Impl {
    std::string endpoint;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<pb::CameraControl::Stub> stub;

    void ensure_connected() {
        if (!stub) {
            channel = detail::make_channel(endpoint);
            stub = pb::CameraControl::NewStub(channel);
        }
    }
};

CameraClient::CameraClient(std::string endpoint) : impl_(std::make_unique<Impl>()) {
    impl_->endpoint = endpoint.empty() ? Config::get_camera_control_endpoint()
                                       : std::move(endpoint);
}

CameraClient::~CameraClient() = default;
CameraClient::CameraClient(CameraClient&&) noexcept = default;
CameraClient& CameraClient::operator=(CameraClient&&) noexcept = default;

void CameraClient::connect() { impl_->ensure_connected(); }

void CameraClient::close() {
    impl_->stub.reset();
    impl_->channel.reset();
}

bool CameraClient::connected() const noexcept { return impl_->stub != nullptr; }

// ---- ISP -------------------------------------------------------------------
void CameraClient::set_isp(const ISPConfig& cfg) {
    impl_->ensure_connected();
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
    detail::check_grpc(impl_->stub->UpdateISPSettings(&ctx, req, &resp), "UpdateISPSettings");
    detail::require_success(resp.status().success(), resp.status().message(), "UpdateISPSettings");
}

ISPConfig CameraClient::get_isp() {
    impl_->ensure_connected();
    pb::ISPConfigResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetISPConfig(&ctx, pb::Empty{}, &resp), "GetISPConfig");
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
    impl_->ensure_connected();
    pb::TransformConfig resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetTransformConfig(&ctx, pb::Empty{}, &resp),
                       "GetTransformConfig");
    return TransformConfig{
        static_cast<int>(resp.rotation()),
        static_cast<int>(resp.flip()),
        resp.dewarp(),
        resp.grayscale(),
    };
}

void CameraClient::set_transform(const TransformConfig& cfg) {
    impl_->ensure_connected();
    pb::TransformConfig req;
    req.set_rotation(static_cast<std::uint32_t>(cfg.rotation));
    req.set_flip(static_cast<std::uint32_t>(cfg.flip));
    req.set_dewarp(cfg.dewarp);
    req.set_grayscale(cfg.grayscale);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->SetTransformConfig(&ctx, req, &resp), "SetTransformConfig");
    detail::require_success(resp.success(), resp.message(), "SetTransformConfig");
}

// ---- Encoder ---------------------------------------------------------------
void CameraClient::set_encoder(const std::string& stream_name, std::uint32_t bitrate_bps,
                               std::uint32_t framerate, std::uint32_t gop) {
    impl_->ensure_connected();
    pb::EncoderConfigRequest req;
    req.set_stream_name(stream_name);
    req.set_bitrate_bps(bitrate_bps);
    req.set_framerate(framerate);
    req.set_gop(gop);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->UpdateEncoderConfig(&ctx, req, &resp), "UpdateEncoderConfig");
    detail::require_success(resp.success(), resp.message(), "UpdateEncoderConfig");
}

EncoderReconfigResult CameraClient::reconfigure_encoder(const std::string& stream_name,
                                                       std::uint32_t width, std::uint32_t height,
                                                       const std::string& codec,
                                                       std::uint32_t bitrate_bps,
                                                       std::uint32_t fps, std::uint32_t gop) {
    impl_->ensure_connected();
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
    detail::check_grpc(impl_->stub->ReconfigureEncoder(&ctx, req, &resp), "ReconfigureEncoder");
    detail::require_success(resp.success(), resp.message(), "ReconfigureEncoder");
    return EncoderReconfigResult{resp.success(), resp.message(), resp.interrupt_ms()};
}

// ---- RTSP ------------------------------------------------------------------
void CameraClient::set_rtsp_enabled(bool enabled) {
    impl_->ensure_connected();
    pb::RtspEnabledRequest req;
    req.set_enabled(enabled);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->SetRtspEnabled(&ctx, req, &resp), "SetRtspEnabled");
    detail::require_success(resp.success(), resp.message(), "SetRtspEnabled");
}

// ---- OSD -------------------------------------------------------------------
void CameraClient::set_osd(const std::vector<OsdStreamConfig>& streams) {
    impl_->ensure_connected();
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
    }
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->UpdateOsdConfig(&ctx, req, &resp), "UpdateOsdConfig");
    detail::require_success(resp.success(), resp.message(), "UpdateOsdConfig");
}

// ---- AI overlay ------------------------------------------------------------
void CameraClient::set_ai_overlay(bool enabled, bool show_label, bool show_confidence,
                                  int line_thickness) {
    impl_->ensure_connected();
    pb::AiOverlayConfig req;
    req.set_enabled(enabled);
    req.set_show_label(show_label);
    req.set_show_confidence(show_confidence);
    req.set_line_thickness(line_thickness);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->UpdateAiOverlay(&ctx, req, &resp), "UpdateAiOverlay");
    detail::require_success(resp.success(), resp.message(), "UpdateAiOverlay");
}

// ---- Stream management -----------------------------------------------------
std::vector<StreamStatus> CameraClient::get_stream_status() {
    impl_->ensure_connected();
    pb::GetStreamStatusResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetStreamStatus(&ctx, pb::GetStreamStatusRequest{}, &resp),
                       "GetStreamStatus");
    std::vector<StreamStatus> out;
    out.reserve(resp.streams_size());
    for (const auto& s : resp.streams()) {
        StreamStatus st;
        st.stream_id = s.stream_id();
        st.status = s.status();
        st.has_encoder = s.has_encoder();
        st.codec = s.codec();
        st.width = s.width();
        st.height = s.height();
        st.fps = s.fps();
        st.bitrate_bps = s.bitrate_bps();
        st.gop = s.gop();
        out.push_back(std::move(st));
    }
    return out;
}

void CameraClient::add_stream(const std::string& stream_id, std::uint32_t width,
                             std::uint32_t height, std::uint32_t fps, const std::string& codec,
                             std::uint32_t bitrate, std::uint32_t gop) {
    impl_->ensure_connected();
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
    detail::check_grpc(impl_->stub->AddStream(&ctx, req, &resp), "AddStream");
    detail::require_success(resp.success(), resp.message(), "AddStream");
}

void CameraClient::remove_stream(const std::string& stream_name) {
    impl_->ensure_connected();
    pb::RemoveStreamRequest req;
    req.set_stream_name(stream_name);
    pb::StreamOperationResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->RemoveStream(&ctx, req, &resp), "RemoveStream");
    detail::require_success(resp.success(), resp.message(), "RemoveStream");
}

// ---- Pipeline reconfiguration ----------------------------------------------
EncoderReconfigResult CameraClient::reconfigure_pipeline(
    const std::vector<PipelineStreamConfig>& streams) {
    impl_->ensure_connected();
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
    detail::check_grpc(impl_->stub->ReconfigurePipeline(&ctx, req, &resp), "ReconfigurePipeline");
    detail::require_success(resp.success(), resp.message(), "ReconfigurePipeline");
    return EncoderReconfigResult{resp.success(), resp.message(), resp.interrupt_ms()};
}

// ---- Profiles --------------------------------------------------------------
std::string CameraClient::get_profile() {
    impl_->ensure_connected();
    pb::GetProfileResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetProfile(&ctx, pb::Empty{}, &resp), "GetProfile");
    return resp.profile_name();
}

std::pair<std::vector<std::string>, std::string> CameraClient::list_profiles() {
    impl_->ensure_connected();
    pb::ListProfilesResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->ListProfiles(&ctx, pb::Empty{}, &resp), "ListProfiles");
    std::vector<std::string> names;
    names.reserve(resp.profiles_size());
    for (const auto& p : resp.profiles()) names.push_back(p);
    return {std::move(names), resp.current_profile()};
}

EncoderReconfigResult CameraClient::switch_profile(const std::string& name) {
    impl_->ensure_connected();
    pb::SwitchProfileRequest req;
    req.set_profile_name(name);
    pb::EncoderReconfigResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->SwitchProfile(&ctx, req, &resp), "SwitchProfile");
    detail::require_success(resp.success(), resp.message(), "SwitchProfile");
    return EncoderReconfigResult{resp.success(), resp.message(), resp.interrupt_ms()};
}

void CameraClient::backup_profile(const std::string& path) {
    impl_->ensure_connected();
    pb::BackupProfileRequest req;
    req.set_path(path);
    pb::BackupProfileResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->BackupProfile(&ctx, req, &resp), "BackupProfile");
    detail::require_success(resp.success(), resp.message(), "BackupProfile");
}

// ---- Sensor / capabilities / hardware --------------------------------------
SensorInfo CameraClient::get_sensor_info(std::uint32_t sensor_index) {
    impl_->ensure_connected();
    pb::GetSensorInfoRequest req;
    req.set_sensor_index(sensor_index);
    pb::SensorInfoResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetSensorInfo(&ctx, req, &resp), "GetSensorInfo");
    return SensorInfo{
        resp.available(),
        resp.sensor_model(),
        resp.i2c_bus(),
        resp.i2c_address(),
        resp.pixel_format(),
    };
}

Capabilities CameraClient::get_capabilities() {
    impl_->ensure_connected();
    pb::CapabilitiesResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetCapabilities(&ctx, pb::Empty{}, &resp), "GetCapabilities");
    return Capabilities{
        resp.has_video(),   resp.has_codec(),   resp.has_led(),  resp.has_sensor(),
        resp.has_mcu(),     resp.has_env_ctrl(), resp.has_alarm(), resp.has_rs485(),
        resp.has_osd(),     resp.has_draw(),    resp.has_audio(),
    };
}

HardwareStatus CameraClient::get_hardware_status() {
    impl_->ensure_connected();
    pb::DeviceHardwareStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetDeviceHardwareStatus(&ctx, pb::Empty{}, &resp),
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
    impl_->ensure_connected();
    pb::SetLedDutyRequest req;
    req.set_led_id(led_id);
    req.set_duty_percent(duty_percent);
    pb::LedStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->SetLedDuty(&ctx, req, &resp), "SetLedDuty");
    detail::require_success(resp.success(), resp.message(), "SetLedDuty");
}

std::uint32_t CameraClient::get_led_duty(std::uint32_t led_id) {
    impl_->ensure_connected();
    pb::GetLedDutyRequest req;
    req.set_led_id(led_id);
    pb::LedStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetLedDuty(&ctx, req, &resp), "GetLedDuty");
    detail::require_success(resp.success(), resp.message(), "GetLedDuty");
    return resp.duty_percent();
}

// ---- IR-Cut ----------------------------------------------------------------
std::uint32_t CameraClient::set_ircut(std::uint32_t mode) {
    impl_->ensure_connected();
    pb::SetIrCutRequest req;
    req.set_mode(mode);
    pb::SetIrCutResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->SetIrCut(&ctx, req, &resp), "SetIrCut");
    detail::require_success(resp.success(), resp.message(), "SetIrCut");
    return resp.current_mode();
}

std::uint32_t CameraClient::get_ircut() {
    impl_->ensure_connected();
    pb::SetIrCutResponse resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetIrCut(&ctx, pb::Empty{}, &resp), "GetIrCut");
    return resp.current_mode();
}

// ---- MCU raw ---------------------------------------------------------------
std::string CameraClient::mcu_raw_request(std::uint32_t cmd, const std::string& payload) {
    impl_->ensure_connected();
    pb::McuRawRequestMessage req;
    req.set_cmd(cmd);
    req.set_payload(payload);
    pb::McuRawResponseMessage resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->McuRawRequest(&ctx, req, &resp), "McuRawRequest");
    detail::require_success(resp.success(), resp.message(), "McuRawRequest");
    return resp.payload();
}

// ---- Environment control ---------------------------------------------------
bool CameraClient::set_fan(bool enable) {
    impl_->ensure_connected();
    pb::EnvCtrlRequest req;
    req.set_enable(enable);
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->SetFan(&ctx, req, &resp), "SetFan");
    detail::require_success(resp.success(), resp.message(), "SetFan");
    return resp.enabled();
}

EnvStatus CameraClient::get_fan() {
    impl_->ensure_connected();
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetFan(&ctx, pb::Empty{}, &resp), "GetFan");
    return EnvStatus{resp.enabled()};
}

bool CameraClient::set_heat(bool enable) {
    impl_->ensure_connected();
    pb::EnvCtrlRequest req;
    req.set_enable(enable);
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->SetHeat(&ctx, req, &resp), "SetHeat");
    detail::require_success(resp.success(), resp.message(), "SetHeat");
    return resp.enabled();
}

EnvStatus CameraClient::get_heat() {
    impl_->ensure_connected();
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetHeat(&ctx, pb::Empty{}, &resp), "GetHeat");
    return EnvStatus{resp.enabled()};
}

bool CameraClient::set_radar(bool enable) {
    impl_->ensure_connected();
    pb::EnvCtrlRequest req;
    req.set_enable(enable);
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->SetRadar(&ctx, req, &resp), "SetRadar");
    detail::require_success(resp.success(), resp.message(), "SetRadar");
    return resp.enabled();
}

EnvStatus CameraClient::get_radar() {
    impl_->ensure_connected();
    pb::EnvCtrlStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetRadar(&ctx, pb::Empty{}, &resp), "GetRadar");
    return EnvStatus{resp.enabled()};
}

// ---- Alarm I/O -------------------------------------------------------------
bool CameraClient::set_alarm_out(std::uint32_t channel, bool enable) {
    impl_->ensure_connected();
    pb::AlarmOutRequest req;
    req.set_channel(channel);
    req.set_enable(enable);
    pb::AlarmOutStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->SetAlarmOut(&ctx, req, &resp), "SetAlarmOut");
    detail::require_success(resp.success(), resp.message(), "SetAlarmOut");
    return resp.enabled();
}

bool CameraClient::get_alarm_out(std::uint32_t channel) {
    impl_->ensure_connected();
    pb::AlarmOutRequest req;
    req.set_channel(channel);
    pb::AlarmOutStatus resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetAlarmOut(&ctx, req, &resp), "GetAlarmOut");
    detail::require_success(resp.success(), resp.message(), "GetAlarmOut");
    return resp.enabled();
}

AlarmOutputs CameraClient::get_alarm_outputs() {
    impl_->ensure_connected();
    pb::AlarmOutputsState resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->GetAlarmOutputs(&ctx, pb::Empty{}, &resp), "GetAlarmOutputs");
    detail::require_success(resp.success(), resp.message(), "GetAlarmOutputs");
    return AlarmOutputs{
        resp.alarm_out0(), resp.alarm_out1(), resp.wiegand0(), resp.wiegand1(),
    };
}

// ---- RS485 -----------------------------------------------------------------
void CameraClient::rs485_init(std::uint32_t baudrate, const std::string& config) {
    impl_->ensure_connected();
    pb::Rs485InitRequest req;
    req.set_baudrate(baudrate);
    req.set_config(config);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->Rs485Init(&ctx, req, &resp), "Rs485Init");
    detail::require_success(resp.success(), resp.message(), "Rs485Init");
}

void CameraClient::rs485_deinit() {
    impl_->ensure_connected();
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->Rs485Deinit(&ctx, pb::Empty{}, &resp), "Rs485Deinit");
    detail::require_success(resp.success(), resp.message(), "Rs485Deinit");
}

void CameraClient::rs485_tx(const std::string& data) {
    impl_->ensure_connected();
    pb::Rs485TxRequest req;
    req.set_data(data);
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(impl_->stub->Rs485Tx(&ctx, req, &resp), "Rs485Tx");
    detail::require_success(resp.success(), resp.message(), "Rs485Tx");
}

}  // namespace neoruntime_ipc_sdk

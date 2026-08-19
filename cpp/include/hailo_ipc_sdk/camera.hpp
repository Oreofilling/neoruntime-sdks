// camera.hpp — camera pipeline control client. 1:1 port of camera.py.
//
// Drives the camera-daemon CameraControl service (package `aipc.camera`): ISP,
// transform, encoder, RTSP, OSD, AI overlay, streams, profiles, sensor info,
// capabilities, hardware/LED/IR-cut/MCU/env/alarm/RS485 status. Transport: sync
// gRPC stubs over UDS (Config::get_camera_control_endpoint()). All RPCs here are
// unary; streaming camera RPCs (audio) live in audio.hpp.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hailo_ipc_sdk {

// ISP image-pipeline settings. Int fields use -1 = "no change" (mirrors app.py).
// manual_mode / auto_exposure are truly optional (proto3 optional bool).
struct ISPConfig {
    int brightness = -1;        // [0..100]
    int contrast = -1;
    int saturation = -1;
    int sharpness = -1;
    std::optional<bool> manual_mode;
    std::optional<bool> auto_exposure;
    int backlight = -1;         // [0..100]
    int exposure_time_us = -1;
    int gain = -1;
    int noise_reduction = -1;   // [0..100]
    int wdr_value = -1;         // [0..100]
    int powerline_freq = -1;    // 0=off, 1=50Hz, 2=60Hz
    int awb_index = -1;
};

struct TransformConfig {
    int rotation = 0;   // 0/1/2/3 => 0/90/180/270
    int flip = 0;       // 0=none, 1=H, 2=V, 3=both
    bool dewarp = false;
    bool grayscale = false;
};

struct EncoderReconfigResult {
    bool success = false;
    std::string message;
    std::uint32_t interrupt_ms = 0;
};

struct StreamStatus {
    std::string stream_id;
    std::string status;     // active, starting, stalled, stopped, error, unknown
    bool has_encoder = false;
    std::string codec;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t fps = 0;
    std::uint32_t bitrate_bps = 0;
    std::uint32_t gop = 0;
};

struct Capabilities {
    bool has_video = false;
    bool has_codec = false;
    bool has_led = false;
    bool has_sensor = false;
    bool has_mcu = false;
    bool has_env_ctrl = false;
    bool has_alarm = false;
    bool has_rs485 = false;
    bool has_osd = false;
    bool has_draw = false;
    bool has_audio = false;
};

struct SensorInfo {
    bool available = false;
    std::string sensor_model;
    int i2c_bus = -1;
    std::string i2c_address;
    int pixel_format = -1;
};

struct HardwareStatus {
    std::uint32_t light_sensor_mv = 0;
    int light_sensor_lux = 0;
    int mcu_temp_millic = 0;
    int ain_mv = 0;
    std::string mcu_version;
    std::uint32_t white_light_duty = 0;
    std::uint32_t ir_led_duty = 0;
    std::uint32_t ircut_mode = 0;  // 0=day, 1=night
};

struct PipelineStreamConfig {
    std::string stream_id;
    std::uint32_t input_width = 0;
    std::uint32_t input_height = 0;
    std::uint32_t input_framerate = 0;
    std::string codec = "h264";
    std::uint32_t encoder_width = 0;
    std::uint32_t encoder_height = 0;
    std::uint32_t encoder_framerate = 0;
    std::uint32_t encoder_bitrate = 0;
    std::uint32_t encoder_gop = 0;
};

struct EnvStatus {
    bool enabled = false;
};

// Result of GetAlarmOutputs: the four digital output lines.
struct AlarmOutputs {
    bool alarm_out0 = false;
    bool alarm_out1 = false;
    bool wiegand0 = false;
    bool wiegand1 = false;
};

// OSD overlay descriptors (typed replacement for camera.py's dict-based OSD API).
struct OsdTextOverlay {
    std::string id;
    std::string text;
    float x = 0.0f;
    float y = 0.0f;
    float font_size = 32.0f;
    std::uint32_t text_color = 0;  // RGBA
    bool enabled = true;
    int h_align = 0;  // 0=LEFT, 1=CENTER, 2=RIGHT
    int v_align = 0;  // 0=TOP, 1=CENTER, 2=BOTTOM
};

struct OsdDateTimeOverlay {
    std::string id;
    float x = 0.0f;
    float y = 0.0f;
    std::string format = "%Y-%m-%d %H:%M:%S";
    float font_size = 32.0f;
    std::uint32_t text_color = 0;  // RGBA
    bool enabled = true;
    int h_align = 0;
    int v_align = 0;
};

struct OsdStreamConfig {
    std::string stream_name = "main";
    std::vector<OsdTextOverlay> text_overlays;
    std::vector<OsdDateTimeOverlay> datetime_overlays;
};

class CameraClient {
public:
    // Empty endpoint => Config::get_camera_control_endpoint().
    explicit CameraClient(std::string endpoint = "");
    ~CameraClient();
    CameraClient(const CameraClient&) = delete;
    CameraClient& operator=(const CameraClient&) = delete;
    CameraClient(CameraClient&&) noexcept;
    CameraClient& operator=(CameraClient&&) noexcept;

    void connect();
    void close();
    bool connected() const noexcept;

    // -- ISP --
    void set_isp(const ISPConfig& config = ISPConfig{});
    ISPConfig get_isp();

    // -- Transform --
    TransformConfig get_transform();
    void set_transform(const TransformConfig& cfg);

    // -- Encoder --
    void set_encoder(const std::string& stream_name = "main",
                     std::uint32_t bitrate_bps = 0, std::uint32_t framerate = 0,
                     std::uint32_t gop = 0);
    EncoderReconfigResult reconfigure_encoder(const std::string& stream_name,
                                              std::uint32_t width = 0, std::uint32_t height = 0,
                                              const std::string& codec = "",
                                              std::uint32_t bitrate_bps = 0,
                                              std::uint32_t fps = 0, std::uint32_t gop = 0);

    // -- RTSP --
    void set_rtsp_enabled(bool enabled);

    // -- OSD --
    void set_osd(const std::vector<OsdStreamConfig>& streams);

    // -- AI overlay (convenience; full control via OverlayClient) --
    void set_ai_overlay(bool enabled, bool show_label = true,
                        bool show_confidence = true, int line_thickness = 2);

    // -- Stream management --
    std::vector<StreamStatus> get_stream_status();
    void add_stream(const std::string& stream_id, std::uint32_t width, std::uint32_t height,
                    std::uint32_t fps, const std::string& codec = "h264",
                    std::uint32_t bitrate = 4'000'000, std::uint32_t gop = 30);
    void remove_stream(const std::string& stream_name);

    // -- Pipeline reconfiguration --
    EncoderReconfigResult reconfigure_pipeline(const std::vector<PipelineStreamConfig>& streams);

    // -- Profiles --
    std::string get_profile();
    // Returns {profile_names, current_profile}.
    std::pair<std::vector<std::string>, std::string> list_profiles();
    EncoderReconfigResult switch_profile(const std::string& name);
    void backup_profile(const std::string& path = "");

    // -- Sensor / capabilities / hardware --
    SensorInfo get_sensor_info(std::uint32_t sensor_index = 0);
    Capabilities get_capabilities();
    HardwareStatus get_hardware_status();

    // -- LED --
    void set_led_duty(std::uint32_t led_id, std::uint32_t duty_percent);
    std::uint32_t get_led_duty(std::uint32_t led_id);

    // -- IR-Cut (0=day, 1=night) --
    std::uint32_t set_ircut(std::uint32_t mode);
    std::uint32_t get_ircut();

    // -- MCU raw --
    std::string mcu_raw_request(std::uint32_t cmd, const std::string& payload = "");

    // -- Environment control --
    bool set_fan(bool enable);
    EnvStatus get_fan();
    bool set_heat(bool enable);
    EnvStatus get_heat();
    bool set_radar(bool enable);
    EnvStatus get_radar();

    // -- Alarm I/O --
    bool set_alarm_out(std::uint32_t channel, bool enable);
    bool get_alarm_out(std::uint32_t channel);
    AlarmOutputs get_alarm_outputs();

    // -- RS485 --
    void rs485_init(std::uint32_t baudrate = 9600, const std::string& config = "8N1");
    void rs485_deinit();
    void rs485_tx(const std::string& data);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hailo_ipc_sdk

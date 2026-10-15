// camera.hpp — camera pipeline control client. 1:1 port of camera.py (0.8.1).
//
// Drives the camera-daemon CameraControl service (package `aipc.camera`): ISP,
// transform, encoder, RTSP, OSD, AI overlay, streams, profiles, sensor info,
// capabilities, hardware/LED/IR-cut/MCU/env/alarm/RS485 status, day/night
// imaging + IR presets, privacy mask, raw config fields, and app frame
// injection (PushFrame P0 / PushFrameStream P2). Transport: sync gRPC stubs
// over UDS (Config::get_camera_control_endpoint()). All RPCs here are unary
// except push_frame_stream (client-streaming); streaming camera RPCs (audio)
// live in audio.hpp.
//
// Data types live in camera_types.hpp (mirroring camera_types.py).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "camera_types.hpp"

namespace neoruntime_ipc_sdk {

// Position a picture-in-picture inset for mode="overlay" pushes.
//
// Returns {dest_x, dest_y} placing a inset_w×inset_h inset inside a
// frame_w×frame_h stream frame, margin pixels from the chosen corner (one of
// "top-left" / "top-right" / "bottom-left" / "bottom-right"). Coordinates are
// rounded down to even values — NV12 chroma is 2×2 subsampled, so an odd
// offset would misalign the inset's color plane against its luma.
//
// Throws std::invalid_argument for an unknown corner, non-positive inset
// dims, negative margin, or an inset that does not fit with its margin:
//
//   auto [x, y] = pip_dest(1920, 1080, 480, 270, "bottom-right", 32);
//   cam.push_frame({bid, 480, 270, stride, "overlay", 0, x, y, "main"});
std::pair<std::uint32_t, std::uint32_t> pip_dest(std::uint32_t frame_w, std::uint32_t frame_h,
                                                 std::uint32_t inset_w, std::uint32_t inset_h,
                                                 const std::string& corner = "bottom-right",
                                                 std::int32_t margin = 16);

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

    // -- Day/night imaging & infrared --
    // mode: "day", "infrared" or "auto" (light-sensor driven). Anything else
    // throws std::invalid_argument client-side.
    InfraredStatus set_imaging_mode(const std::string& mode);
    InfraredStatus get_infrared_status();
    // Omitted (nullopt) fields are left unchanged on the device.
    InfraredStatus set_infrared_settings(
        std::optional<bool> auto_follow = std::nullopt,
        std::optional<std::uint32_t> near_pwm = std::nullopt,
        std::optional<std::uint32_t> far_pwm = std::nullopt,
        std::optional<std::int32_t> night_enter = std::nullopt,
        std::optional<std::int32_t> day_enter = std::nullopt);
    // Drop manual IR overrides and return to profile-driven output.
    InfraredStatus clear_infrared_manual();

    // -- IR presets --
    std::vector<IrPreset> list_ir_presets();
    // Save (or overwrite) an IR preset and return the new list.
    std::vector<IrPreset> save_ir_preset(const std::string& name, float zoom_ratio,
                                         std::uint32_t near_pwm, std::uint32_t far_pwm);
    // Delete an IR preset by name and return the remaining list.
    std::vector<IrPreset> delete_ir_preset(const std::string& name);

    // -- Privacy mask --
    PrivacyMaskSettings get_privacy_mask();
    // Apply a full PrivacyMaskSettings wholesale.
    void set_privacy_mask(const PrivacyMaskSettings& settings);
    // Partial update with merge semantics: only the fields given are
    // changed; everything else (including existing regions, unless
    // `regions` is given) is read back from the device and preserved, so
    // partial updates never wipe masks.
    void set_privacy_mask(std::optional<std::uint32_t> color,
                          std::optional<std::int32_t> blur_radius,
                          std::optional<bool> enabled,
                          std::optional<std::vector<PrivacyMaskRegion>> regions,
                          std::optional<bool> dpm_enabled = std::nullopt,
                          std::optional<std::string> dpm_labels = std::nullopt,
                          std::optional<std::string> dpm_mode = std::nullopt,
                          std::optional<std::uint32_t> dpm_color = std::nullopt);

    // -- OSD --
    void set_osd(const std::vector<OsdStreamConfig>& streams);
    // Read back OSD overlay config, shaped like set_osd() input: each
    // stream carries text / datetime / image overlays.
    std::vector<OsdStreamConfig> get_osd();

    // -- Raw config fields --
    // Read one camera-daemon config field by dotted path (e.g.
    // "frontend.hailort.use-hailort-service"); returns the value as a string.
    std::string get_config_field(const std::string& field_path);
    // Write one config field; `value` is encoded as a string ("true", "42", …).
    void set_config_field(const std::string& field_path, const std::string& value);

    // -- AI overlay (convenience; full control via OverlayClient) --
    void set_ai_overlay(bool enabled, bool show_label = true,
                        bool show_confidence = true, int line_thickness = 2);

    // -- Stream management --
    // timeout_s bounds the RPC (nullopt waits indefinitely, gRPC default).
    std::vector<StreamStatus> get_stream_status(
        std::optional<double> timeout_s = std::nullopt);
    void add_stream(const std::string& stream_id, std::uint32_t width, std::uint32_t height,
                    std::uint32_t fps, const std::string& codec = "h264",
                    std::uint32_t bitrate = 4'000'000, std::uint32_t gop = 30);
    void remove_stream(const std::string& stream_name);

    // -- Pipeline reconfiguration --
    EncoderReconfigResult reconfigure_pipeline(const std::vector<PipelineStreamConfig>& streams);

    // -- App frame injection (PushFrame P0) --------------------------------
    //
    // Push one frame into the daemon's encoder feed. spec.buffer_id must be
    // a DSP-registry buffer id (see DspBufferPool / DspClient::import_frame)
    // — a raw fd number never crosses gRPC. The registered buffer's
    // geometry must equal spec.width / spec.height.
    //
    // Modes: "replace" — full-frame NV12 content copy over one stream's
    // encode output (empty stream_id keeps the P0 any-matching-stream
    // semantics; an explicit stream_id requires dims equal to its encode
    // resolution). "overlay" — compose on top of the live stream; requires
    // stream_id; dest_x/dest_y (even, NV12 chroma alignment) position the
    // inset; NV12 insets paste opaquely, ARGB32 insets alpha-blend on the
    // CPU.
    //
    // pts_ns paces the bake (device CLOCK_MONOTONIC domain): 0 = due
    // immediately; a future value holds the frame until the encoder
    // frontier passes it, and a newer due frame supersedes (drops) older
    // due frames for the same target. end_of_stream=true (buffer_id 0)
    // flushes the session and restores the ISP path at the next IDR.
    // session_id tags the injection session (P2-13): correlation and
    // observability only — the daemon anchors the session on the buffer
    // owner's UDS connection and closes it on that client's disconnect
    // regardless of the tag.
    //
    // Throws std::invalid_argument for an unknown mode, std::runtime_error
    // (detail::RpcError) carrying "PushFrame failed: <msg>
    // (error_code=<n>)" when the daemon rejects the frame.
    InjectionResult push_frame(const PushFrameSpec& spec,
                               std::optional<double> timeout_s = std::nullopt);

    // Push a run of frames over one client-streaming RPC. Per-request
    // semantics are identical to push_frame; a spec with end_of_stream=true
    // closes the injection session server-side before the RPC returns,
    // while ending the vector without one leaves the session open (close
    // later via stop_injection). The first server-side rejection ends the
    // stream and re-raises here as RpcError ("PushFrameStream failed after
    // N accepted frame(s): …"). The returned InjectionResult's
    // accepted_frame_count reports how many frames the daemon accepted;
    // in_flight_buffer_ids is a final-response snapshot only (client
    // streaming has no per-frame acks).
    InjectionResult push_frame_stream(const std::vector<PushFrameSpec>& frames,
                                      std::optional<double> timeout_s = std::nullopt);

    // Injection session snapshot: active, mode, counters, queue depth, the
    // live session's lifecycle tag (empty when untagged/closed), and the
    // write-lease set (in_flight_buffer_ids + reports_in_flight_buffers).
    InjectionStatus injection_status(std::optional<double> timeout_s = std::nullopt);

    // Close the injection session and restore the ISP path (next IDR).
    void stop_injection(std::optional<double> timeout_s = std::nullopt);

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

}  // namespace neoruntime_ipc_sdk

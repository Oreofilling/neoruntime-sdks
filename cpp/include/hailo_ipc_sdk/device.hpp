// device.hpp — DeviceControl gRPC client. 1:1 port of device.py.
//
// Transport: sync gRPC stubs over UDS (the default endpoint). The Python SDK
// relies on a wire-format coincidence for set_ir_led / control_iris that the
// strongly-typed C++ stubs cannot reproduce; those two methods therefore
// follow the .proto signatures (SetIrLed(LightLevelRequest), IrisRequest.speed)
// rather than device.py's buggy Python wrappers. See device.cpp notes.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace hailo_ipc_sdk {

// IR-cut filter mode. Values match aipc.device.IrCutMode (IRCUT_AUTO/DAY/NIGHT).
enum class IrCutMode : int {
    Auto  = 0,
    Day   = 1,
    Night = 2,
};

struct LensLimit {
    std::int32_t min_pos = 0;
    std::int32_t max_pos = 0;
};

struct LensStatus {
    std::uint32_t zoom_state = 0;   // motor: 0=NoCfg 1=Stopped 2=Running 3=ResetZero 4=Error
    std::uint32_t focus_state = 0;
    bool zoom_rz_done = false;
    bool focus_rz_done = false;
    std::int32_t zoom_pos = 0;
    std::int32_t focus_pos = 0;
    std::uint32_t iris_adc = 0;
    bool autofocus_enabled = false;
    LensLimit zoom_limit{};
    LensLimit focus_limit{};
};

struct DeviceStatus {
    float soc_temp_c = 0.0f;
    float mcu_temp_c = 0.0f;
    std::uint32_t light_sensor = 0;
    std::int32_t ptz_pan_pos = 0;
    std::int32_t ptz_tilt_pos = 0;
    std::int32_t zoom_pos = 0;
    std::int32_t focus_pos = 0;
    bool autofocus_enabled = false;
    IrCutMode ircut_mode = IrCutMode::Auto;
    std::uint32_t white_light_level = 0;
    std::uint32_t ir_led_level = 0;   // proto field 41 (device.py wrongly read ir_led_on)
    std::string mcu_version;
    std::uint64_t mcu_uptime_ms = 0;
};

struct DeviceEvent {
    enum class Type : int {
        GpioChange        = 0,
        LightSensorChange = 1,
        TemperatureAlert  = 2,
        PtzMoveComplete   = 3,
        FocusComplete     = 4,
    };
    Type type = Type::GpioChange;
    std::uint64_t timestamp_ns = 0;
    // Payload — only the field(s) relevant to `type` are populated.
    std::uint32_t gpio_pin = 0;
    bool gpio_value = false;
    std::uint32_t light_sensor_value = 0;
    float temperature = 0.0f;
};

// Pull-based equivalent of Python's `for ev in dev.subscribe_events():` generator.
// Must not outlive the DeviceClient that produced it (it borrows the channel).
class DeviceEventStream {
public:
    ~DeviceEventStream();
    DeviceEventStream(DeviceEventStream&&) noexcept;
    DeviceEventStream& operator=(DeviceEventStream&&) noexcept;
    DeviceEventStream(const DeviceEventStream&) = delete;
    DeviceEventStream& operator=(const DeviceEventStream&) = delete;

    // Next event, or std::nullopt when the stream ends (server closed / broken).
    std::optional<DeviceEvent> next();

private:
    DeviceEventStream();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class DeviceClient;
};

class DeviceClient {
public:
    // Empty endpoint => Config::get_device_control_endpoint().
    explicit DeviceClient(std::string endpoint = "");
    ~DeviceClient();
    DeviceClient(const DeviceClient&) = delete;
    DeviceClient& operator=(const DeviceClient&) = delete;
    DeviceClient(DeviceClient&&) noexcept;
    DeviceClient& operator=(DeviceClient&&) noexcept;

    void connect();
    void close();
    bool connected() const noexcept;

    // ---- Light control ----
    void set_white_light(std::uint32_t level);
    void set_ir_led(std::uint32_t level);        // proto: SetIrLed(LightLevelRequest)
    void set_ircut(IrCutMode mode);

    // ---- PTZ ----
    void pan_left(std::uint32_t speed = 50);
    void pan_right(std::uint32_t speed = 50);
    void pan_stop();
    void tilt_up(std::uint32_t speed = 50);
    void tilt_down(std::uint32_t speed = 50);
    void tilt_stop();
    void ptz_stop();
    void save_preset(std::uint32_t preset_id);
    void call_preset(std::uint32_t preset_id);

    // ---- Lens ----
    void zoom_in(std::int32_t speed = 50);
    void zoom_out(std::int32_t speed = 50);
    void zoom_stop();
    void zoom(std::int32_t speed);
    void set_zoom_level(float level);
    void focus_in(std::int32_t speed = 50);
    void focus_out(std::int32_t speed = 50);
    void focus_stop();
    void focus(std::int32_t speed);
    void focus_auto(bool enable = true);
    void set_focus_level(float level);
    LensStatus get_lens_status();
    void set_lens_limits(std::optional<LensLimit> zoom_limit = std::nullopt,
                         std::optional<LensLimit> focus_limit = std::nullopt);
    void oneshot_autofocus(double timeout_seconds = 20.0);
    void lens_reset_zero(bool zoom = true, bool focus = true);
    void control_iris(std::int32_t speed);        // proto: IrisRequest{int32 speed}
    void set_iris_target(std::uint32_t target);
    void lens_init();
    void lens_goto_ratio_distance(float zoom_ratio, float focus_distance_m);

    // ---- Alarm I/O ----
    void set_wiegand_out(std::uint32_t channel, bool enable);
    bool get_wiegand_out(std::uint32_t channel);

    // ---- RS485 ----
    void rs485_init(std::uint32_t baudrate, std::string config = "");
    void rs485_deinit();
    void rs485_tx(std::string_view data);

    // ---- GPIO ----
    void gpio_set(std::uint32_t pin, bool value);
    bool gpio_get(std::uint32_t pin);

    // ---- Status / events ----
    DeviceStatus get_device_status();
    DeviceEventStream subscribe_events();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hailo_ipc_sdk

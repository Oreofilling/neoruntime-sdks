"""Data types returned by the camera control client."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Capabilities",
    "EncoderReconfigResult",
    "EnvStatus",
    "HardwareStatus",
    "InfraredStatus",
    "InjectionResult",
    "InjectionStatus",
    "IrPreset",
    "ISPConfig",
    "PipelineStreamConfig",
    "PrivacyMaskSettings",
    "SensorInfo",
    "StreamStatus",
    "TransformConfig",
]


@dataclass
class ISPConfig:
    brightness: int = -1  # [0..100], -1 = no change
    contrast: int = -1  # [0..100]
    saturation: int = -1  # [0..100]
    sharpness: int = -1  # [0..100]
    manual_mode: bool | None = None
    auto_exposure: bool | None = None
    backlight: int = -1  # [0..100]
    exposure_time_us: int = -1
    gain: int = -1
    noise_reduction: int = -1  # [0..100]
    wdr_value: int = -1  # [0..100]
    powerline_freq: int = -1  # 0=off, 1=50Hz, 2=60Hz
    awb_index: int = -1


@dataclass
class TransformConfig:
    rotation: int = 0  # 0/1/2/3 => 0/90/180/270
    flip: int = 0  # 0=none, 1=H, 2=V, 3=both
    dewarp: bool = False
    grayscale: bool = False


@dataclass
class EncoderReconfigResult:
    success: bool
    message: str
    interrupt_ms: int = 0


@dataclass
class InjectionResult:
    """Result of one CameraClient.push_frame call (PushFrameResponse)."""

    success: bool
    message: str
    error_code: int = 0
    injected_frame_id: int = 0
    # PushFrameStream only: frames accepted before the stream returned
    # (0 for unary PushFrame).
    accepted_frame_count: int = 0
    # Lifecycle session tag echoed by the daemon (P2-13): correlation
    # and observability only — ownership is fd-anchored daemon-side.
    session_id: str = ""


@dataclass
class InjectionStatus:
    """Snapshot of the daemon-side injection session (GetInjectionStatus)."""

    success: bool
    message: str
    active: bool = False
    mode: str = "replace"  # "replace" (P0) | "overlay" (P1, DSP blend)
    frames_injected: int = 0
    frames_dropped: int = 0  # drop-oldest overflow, never backpressure
    queue_depth: int = 0
    # Latest non-empty lifecycle tag of the live session (P2-13); empty
    # when the session is untagged or closed.
    session_id: str = ""


@dataclass
class StreamStatus:
    stream_id: str
    status: str
    has_encoder: bool
    codec: str
    width: int
    height: int
    fps: int
    bitrate_bps: int
    gop: int
    # Unified drop/throughput observability (wire fields 13-23). All
    # default to 0 and stay 0 when the layer is absent (publisher
    # disabled, stream never through the overlay bake site) or the
    # server predates the fields — the response shape is stable.
    # Publisher side (packets keyed by seq, assigned at enqueue):
    packets_published: int = 0      # packets assigned a seq (== last_packet_seq)
    queue_overflow_drops: int = 0   # shallow-queue evictions before send
    client_send_drops: int = 0      # per-client non-blocking send skips
    client_send_failures: int = 0   # per-client hard send failures (client dropped)
    client_disconnects: int = 0     # subscribers lost via control-poll EOF/ERR
    last_packet_seq: int = 0        # newest seq assigned (0 = none yet)
    publisher_clients: int = 0      # connected encoded-stream subscribers
    # Overlay bake side (frames shipped clean when no fresh result):
    bake_skips: int = 0             # frames shipped clean: no result / TTL expired
    strict_locked: int = 0          # strict-gate draws that waited and matched
    strict_degraded: int = 0        # strict-gate frames degraded (cap/hopeless)
    strict_skips: int = 0           # strict-gate frames with no result at all
    # Behavior decoupling + frame sync (wire fields 24-28). stream_epoch is
    # the stream generation counter — ReconfigureEncoder / full transform
    # reinit bumps it and purges held layers; an app pinning overlay events
    # to an epoch taken from an earlier snapshot learns of the restart by
    # comparing, instead of publishing silently-rejected overlays. 0 = the
    # server predates the fields (epoch is never 0 on a current daemon).
    stream_epoch: int = 0           # stream generation (restart bumps)
    overlay_layer_count: int = 0    # overlay layers currently held for the stream
    overlay_late_commands: int = 0  # app overlay events rejected: frame already passed
    overlay_epoch_rejects: int = 0  # app overlay events rejected: stale stream_epoch
    overlay_no_binding_drops: int = 0  # platform result events dropped: no binding


@dataclass
class Capabilities:
    has_video: bool = False
    has_codec: bool = False
    has_led: bool = False
    has_sensor: bool = False
    has_mcu: bool = False
    has_env_ctrl: bool = False
    has_alarm: bool = False
    has_rs485: bool = False
    has_osd: bool = False
    has_draw: bool = False
    has_audio: bool = False


@dataclass
class SensorInfo:
    available: bool
    sensor_model: str
    i2c_bus: int
    i2c_address: str
    pixel_format: int


@dataclass
class HardwareStatus:
    light_sensor_mv: int
    light_sensor_lux: int
    mcu_temp_millic: int
    ain_mv: int
    mcu_version: str
    white_light_duty: int
    ir_led_duty: int
    ircut_mode: int  # 0=day, 1=night


@dataclass
class PipelineStreamConfig:
    stream_id: str
    input_width: int = 0
    input_height: int = 0
    input_framerate: int = 0
    codec: str = "h264"
    encoder_width: int = 0
    encoder_height: int = 0
    encoder_framerate: int = 0
    encoder_bitrate: int = 0
    encoder_gop: int = 0


@dataclass
class EnvStatus:
    enabled: bool


@dataclass
class InfraredStatus:
    """Day/night imaging state reported by the camera pipeline."""

    mode: str
    transition: str
    output_source: str
    auto_follow: bool
    follow_active: bool
    manual_override: bool
    degraded: bool
    requested_near_pwm: int
    requested_far_pwm: int
    applied_near_pwm: int
    applied_far_pwm: int
    zoom_ratio: float
    active_profile: str
    selected_mode: str
    light_percent: int
    light_mv: int
    light_milli: int
    light_valid: bool
    night_enter: int
    day_enter: int


@dataclass
class IrPreset:
    """Saved IR-light profile bound to a zoom ratio."""

    name: str
    zoom_ratio: float
    near_pwm: int
    far_pwm: int


@dataclass
class PrivacyMaskSettings:
    """Static and dynamic (AI) privacy-mask configuration.

    regions is a list of dicts: {id, name, enabled, points_x, points_y}
    with normalized [0.0-1.0] polygon coordinates (up to 8 points).
    """

    color: int
    blur_radius: int
    enabled: bool
    regions: list[dict]
    dpm_enabled: bool
    dpm_labels: str
    dpm_mode: str
    dpm_color: int

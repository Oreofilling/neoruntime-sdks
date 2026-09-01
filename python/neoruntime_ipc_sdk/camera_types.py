"""Data types returned by the camera control client."""

from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    "Capabilities",
    "EncoderReconfigResult",
    "EnvStatus",
    "HardwareStatus",
    "InfraredStatus",
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
    brightness: int = -1       # [0..100], -1 = no change
    contrast: int = -1         # [0..100]
    saturation: int = -1       # [0..100]
    sharpness: int = -1        # [0..100]
    manual_mode: Optional[bool] = None
    auto_exposure: Optional[bool] = None
    backlight: int = -1        # [0..100]
    exposure_time_us: int = -1
    gain: int = -1
    noise_reduction: int = -1  # [0..100]
    wdr_value: int = -1        # [0..100]
    powerline_freq: int = -1   # 0=off, 1=50Hz, 2=60Hz
    awb_index: int = -1


@dataclass
class TransformConfig:
    rotation: int = 0   # 0/1/2/3 => 0/90/180/270
    flip: int = 0       # 0=none, 1=H, 2=V, 3=both
    dewarp: bool = False
    grayscale: bool = False


@dataclass
class EncoderReconfigResult:
    success: bool
    message: str
    interrupt_ms: int = 0


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
    regions: List[dict]
    dpm_enabled: bool
    dpm_labels: str
    dpm_mode: str
    dpm_color: int

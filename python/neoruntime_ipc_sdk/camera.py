"""
Camera Control Client

Comprehensive camera pipeline control: ISP, encoder, RTSP, OSD,
stream management, profiles, capabilities, and hardware status.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import grpc

from .config import Config
from .proto import camera_pb2, camera_pb2_grpc

logger = logging.getLogger("neoruntime_ipc_sdk.camera")


# -- Data classes --

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


def _check_status(resp, label: str) -> None:
    """Check a Status-bearing response."""
    s = resp.status if hasattr(resp, "status") and hasattr(resp.status, "success") else resp
    if hasattr(s, "success") and not s.success:
        msg = s.message if hasattr(s, "message") else "unknown error"
        raise RuntimeError(f"{label} failed: {msg}")


class CameraClient:
    """
    Camera pipeline control client.

    Usage::

        cam = CameraClient()

        # ISP
        cam.set_isp(brightness=60, contrast=50)
        isp = cam.get_isp()

        # Encoder
        cam.set_encoder("main", bitrate_bps=8_000_000)

        # RTSP
        cam.set_rtsp_enabled(True)

        # Streams
        streams = cam.get_stream_status()
        cam.add_stream("third", 1920, 1080, 30, "h264", 4_000_000, 30)

        # Profiles
        cam.switch_profile("night")
        cam.backup_profile()

        # Capabilities
        caps = cam.get_capabilities()
    """

    def __init__(self, endpoint: Optional[str] = None):
        if endpoint is None:
            endpoint = Config.get_camera_control_endpoint()
        self.endpoint = endpoint
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[camera_pb2_grpc.CameraControlStub] = None

    def _connect(self) -> camera_pb2_grpc.CameraControlStub:
        if self._stub is not None:
            return self._stub
        self._channel = grpc.insecure_channel(self.endpoint)
        self._stub = camera_pb2_grpc.CameraControlStub(self._channel)
        return self._stub

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *args):
        self.close()

    # -- ISP --

    def set_isp(self, config: Optional[ISPConfig] = None, **kwargs) -> None:
        """Update ISP image pipeline settings."""
        cfg = config or ISPConfig()
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        req = camera_pb2.ISPUpdateRequest()
        if cfg.brightness >= 0:
            req.brightness = cfg.brightness
        if cfg.contrast >= 0:
            req.contrast = cfg.contrast
        if cfg.saturation >= 0:
            req.saturation = cfg.saturation
        if cfg.sharpness >= 0:
            req.sharpness = cfg.sharpness
        if cfg.manual_mode is not None:
            req.manual_mode = cfg.manual_mode
        if cfg.auto_exposure is not None:
            req.auto_exposure = cfg.auto_exposure
        if cfg.backlight >= 0:
            req.backlight = cfg.backlight
        if cfg.exposure_time_us >= 0:
            req.exposure_time_us = cfg.exposure_time_us
        if cfg.gain >= 0:
            req.gain = cfg.gain
        if cfg.noise_reduction >= 0:
            req.noise_reduction = cfg.noise_reduction
        if cfg.wdr_value >= 0:
            req.wdr_value = cfg.wdr_value
        if cfg.powerline_freq >= 0:
            req.powerline_freq = cfg.powerline_freq
        if cfg.awb_index >= 0:
            req.awb_index = cfg.awb_index

        stub = self._connect()
        resp = stub.UpdateISPSettings(req)
        _check_status(resp, "UpdateISPSettings")

    def get_isp(self) -> ISPConfig:
        """Get current ISP configuration."""
        stub = self._connect()
        resp = stub.GetISPConfig(camera_pb2.Empty())
        if not resp.success:
            raise RuntimeError(f"GetISPConfig failed: {resp.message}")
        c = resp.current
        return ISPConfig(
            brightness=c.brightness,
            contrast=c.contrast,
            saturation=c.saturation,
            sharpness=c.sharpness,
            manual_mode=c.manual_mode if c.HasField("manual_mode") else None,
            auto_exposure=c.auto_exposure if c.HasField("auto_exposure") else None,
            backlight=c.backlight,
            exposure_time_us=c.exposure_time_us,
            gain=c.gain,
            noise_reduction=c.noise_reduction,
            wdr_value=c.wdr_value,
            powerline_freq=c.powerline_freq,
            awb_index=c.awb_index,
        )

    # -- Transform --

    def get_transform(self) -> TransformConfig:
        stub = self._connect()
        resp = stub.GetTransformConfig(camera_pb2.Empty())
        return TransformConfig(
            rotation=resp.rotation,
            flip=resp.flip,
            dewarp=resp.dewarp,
            grayscale=resp.grayscale,
        )

    def set_transform(self, cfg: TransformConfig) -> None:
        stub = self._connect()
        resp = stub.SetTransformConfig(camera_pb2.TransformConfig(
            rotation=cfg.rotation,
            flip=cfg.flip,
            dewarp=cfg.dewarp,
            grayscale=cfg.grayscale,
        ))
        _check_status(resp, "SetTransformConfig")

    # -- Encoder --

    def set_encoder(self, stream_name: str = "main", bitrate_bps: int = 0,
                    framerate: int = 0, gop: int = 0) -> None:
        """Dynamic encoder config (no restart)."""
        stub = self._connect()
        resp = stub.UpdateEncoderConfig(camera_pb2.EncoderConfigRequest(
            stream_name=stream_name,
            bitrate_bps=bitrate_bps,
            framerate=framerate,
            gop=gop,
        ))
        _check_status(resp, "UpdateEncoderConfig")

    def reconfigure_encoder(self, stream_name: str, width: int = 0, height: int = 0,
                            codec: str = "", bitrate_bps: int = 0,
                            fps: int = 0, gop: int = 0) -> EncoderReconfigResult:
        """Full encoder reconfiguration (brief restart, ~100ms)."""
        stub = self._connect()
        resp = stub.ReconfigureEncoder(camera_pb2.EncoderReconfigRequest(
            stream_name=stream_name,
            width=width, height=height,
            codec=codec,
            bitrate_bps=bitrate_bps,
            fps=fps, gop=gop,
        ))
        if not resp.success:
            raise RuntimeError(f"ReconfigureEncoder failed: {resp.message}")
        return EncoderReconfigResult(
            success=resp.success,
            message=resp.message,
            interrupt_ms=resp.interrupt_ms,
        )

    # -- RTSP --

    def set_rtsp_enabled(self, enabled: bool) -> None:
        stub = self._connect()
        resp = stub.SetRtspEnabled(camera_pb2.RtspEnabledRequest(enabled=enabled))
        _check_status(resp, "SetRtspEnabled")

    # -- OSD --

    def set_osd(self, streams: List[dict]) -> None:
        """Update OSD text and datetime overlays per stream.

        Args:
            streams: list of dicts with keys:
                stream_name, text_overlays (list of dicts), datetime_overlays (list of dicts)
        """
        req = camera_pb2.OsdConfigRequest()
        for s in streams:
            sc = req.streams.add()
            sc.stream_name = s.get("stream_name", "main")
            for t in s.get("text_overlays", []):
                tc = sc.text_overlays.add()
                for k, v in t.items():
                    setattr(tc, k, v)
            for d in s.get("datetime_overlays", []):
                dc = sc.datetime_overlays.add()
                for k, v in d.items():
                    setattr(dc, k, v)
        stub = self._connect()
        resp = stub.UpdateOsdConfig(req)
        _check_status(resp, "UpdateOsdConfig")

    # -- AI Overlay (convenience) --

    def set_ai_overlay(self, enabled: bool, show_label: bool = True,
                       show_confidence: bool = True, line_thickness: int = 2) -> None:
        stub = self._connect()
        resp = stub.UpdateAiOverlay(camera_pb2.AiOverlayConfig(
            enabled=enabled,
            show_label=show_label,
            show_confidence=show_confidence,
            line_thickness=line_thickness,
        ))
        _check_status(resp, "UpdateAiOverlay")

    # -- Day/night imaging & infrared --

    def _to_infrared_status(self, resp: "camera_pb2.InfraredStatusResponse") -> InfraredStatus:
        return InfraredStatus(
            mode=resp.mode, transition=resp.transition,
            output_source=resp.output_source, auto_follow=resp.auto_follow,
            follow_active=resp.follow_active,
            manual_override=resp.manual_override, degraded=resp.degraded,
            requested_near_pwm=resp.requested_near_pwm,
            requested_far_pwm=resp.requested_far_pwm,
            applied_near_pwm=resp.applied_near_pwm,
            applied_far_pwm=resp.applied_far_pwm,
            zoom_ratio=resp.zoom_ratio,
            active_profile=resp.active_profile,
            selected_mode=resp.selected_mode,
            light_percent=resp.light_percent, light_mv=resp.light_mv,
            light_milli=resp.light_milli, light_valid=resp.light_valid,
            night_enter=resp.night_enter, day_enter=resp.day_enter)

    def set_imaging_mode(self, mode: str) -> InfraredStatus:
        """Switch day/night imaging mode.

        Args:
            mode: "day", "infrared" or "auto" (light-sensor driven).

        Returns: the resulting InfraredStatus.
        """
        if mode not in ("day", "infrared", "auto"):
            raise ValueError(f"mode must be day/infrared/auto, got {mode!r}")
        stub = self._connect()
        resp = stub.SetImagingMode(camera_pb2.ImagingModeRequest(mode=mode))
        _check_status(resp, "SetImagingMode")
        return self._to_infrared_status(resp)

    def get_infrared_status(self) -> InfraredStatus:
        """Current day/night imaging state (mode, PWMs, light sensor)."""
        stub = self._connect()
        resp = stub.GetInfraredStatus(camera_pb2.Empty())
        _check_status(resp, "GetInfraredStatus")
        return self._to_infrared_status(resp)

    def set_infrared_settings(self, auto_follow: Optional[bool] = None,
                              near_pwm: Optional[int] = None,
                              far_pwm: Optional[int] = None,
                              night_enter: Optional[int] = None,
                              day_enter: Optional[int] = None) -> InfraredStatus:
        """Update IR light settings; omitted fields (None) are left unchanged.

        Args:
            auto_follow: tie IR output to optical zoom ratio.
            near_pwm / far_pwm: manual IR intensities (0-100).
            night_enter / day_enter: light-sensor thresholds for auto mode.
        """
        req = camera_pb2.InfraredSettingsRequest()
        if auto_follow is not None:
            req.auto_follow = auto_follow
        if near_pwm is not None:
            req.near_pwm = near_pwm
        if far_pwm is not None:
            req.far_pwm = far_pwm
        if night_enter is not None:
            req.night_enter = night_enter
        if day_enter is not None:
            req.day_enter = day_enter
        stub = self._connect()
        resp = stub.SetInfraredSettings(req)
        _check_status(resp, "SetInfraredSettings")
        return self._to_infrared_status(resp)

    def clear_infrared_manual(self) -> InfraredStatus:
        """Drop manual IR overrides and return to profile-driven output."""
        stub = self._connect()
        resp = stub.ClearInfraredManual(camera_pb2.Empty())
        _check_status(resp, "ClearInfraredManual")
        return self._to_infrared_status(resp)

    # -- IR presets --

    def list_ir_presets(self) -> List[IrPreset]:
        """Saved IR-light profiles (per zoom ratio)."""
        stub = self._connect()
        resp = stub.ListIrPresets(camera_pb2.Empty())
        _check_status(resp, "ListIrPresets")
        return [IrPreset(name=p.name, zoom_ratio=p.zoom_ratio,
                         near_pwm=p.near_pwm, far_pwm=p.far_pwm)
                for p in resp.presets]

    def save_ir_preset(self, name: str, zoom_ratio: float,
                       near_pwm: int, far_pwm: int) -> List[IrPreset]:
        """Save (or overwrite) an IR preset and return the new list."""
        stub = self._connect()
        resp = stub.SaveIrPreset(camera_pb2.IrPreset(
            name=name, zoom_ratio=zoom_ratio,
            near_pwm=near_pwm, far_pwm=far_pwm))
        _check_status(resp, "SaveIrPreset")
        return [IrPreset(name=p.name, zoom_ratio=p.zoom_ratio,
                         near_pwm=p.near_pwm, far_pwm=p.far_pwm)
                for p in resp.presets]

    def delete_ir_preset(self, name: str) -> List[IrPreset]:
        """Delete an IR preset by name and return the remaining list."""
        stub = self._connect()
        resp = stub.DeleteIrPreset(camera_pb2.DeleteIrPresetRequest(name=name))
        _check_status(resp, "DeleteIrPreset")
        return [IrPreset(name=p.name, zoom_ratio=p.zoom_ratio,
                         near_pwm=p.near_pwm, far_pwm=p.far_pwm)
                for p in resp.presets]

    # -- Privacy mask --

    def get_privacy_mask(self) -> PrivacyMaskSettings:
        """Current static + dynamic (AI) privacy-mask configuration."""
        stub = self._connect()
        cfg = stub.GetPrivacyMaskConfig(camera_pb2.Empty())
        return PrivacyMaskSettings(
            color=cfg.color, blur_radius=cfg.blur_radius,
            enabled=cfg.enabled,
            regions=[{"id": r.id, "name": r.name, "enabled": r.enabled,
                      "points_x": list(r.points_x),
                      "points_y": list(r.points_y)}
                     for r in cfg.regions],
            dpm_enabled=cfg.dpm_enabled, dpm_labels=cfg.dpm_labels,
            dpm_mode=cfg.dpm_mode, dpm_color=cfg.dpm_color)

    def set_privacy_mask(self, settings: Optional[PrivacyMaskSettings] = None, *,
                         color: Optional[int] = None,
                         blur_radius: Optional[int] = None,
                         enabled: Optional[bool] = None,
                         regions: Optional[List[dict]] = None,
                         dpm_enabled: Optional[bool] = None,
                         dpm_labels: Optional[str] = None,
                         dpm_mode: Optional[str] = None,
                         dpm_color: Optional[int] = None) -> None:
        """Update privacy-mask configuration with merge semantics.

        Only the fields given as arguments are changed; everything else
        (including existing regions) is read back from the device and
        preserved, so partial updates never wipe masks.

        Args:
            settings: full PrivacyMaskSettings to apply wholesale (optional).
            color: 0x00RRGGBB fill color (used when blur_radius=0).
            blur_radius: pixelation block size 2-64; 0 = solid color.
            enabled: global static-mask on/off.
            regions: list of {id, name, enabled, points_x, points_y} with
                normalized polygon coordinates (up to 8 points).
            dpm_*: dynamic (AI) privacy-mask options.
        """
        stub = self._connect()
        if settings is not None:
            cfg = camera_pb2.PrivacyMaskConfig(
                color=settings.color, blur_radius=settings.blur_radius,
                enabled=settings.enabled,
                dpm_enabled=settings.dpm_enabled,
                dpm_labels=settings.dpm_labels,
                dpm_mode=settings.dpm_mode, dpm_color=settings.dpm_color)
            regions = settings.regions
            if regions is not None:
                for region in regions:
                    self._add_mask_region(cfg, region)
        else:
            current = stub.GetPrivacyMaskConfig(camera_pb2.Empty())
            cfg = camera_pb2.PrivacyMaskConfig()
            cfg.CopyFrom(current)
            if color is not None:
                cfg.color = color
            if blur_radius is not None:
                cfg.blur_radius = blur_radius
            if enabled is not None:
                cfg.enabled = enabled
            if regions is not None:
                del cfg.regions[:]
                for region in regions:
                    self._add_mask_region(cfg, region)
            if dpm_enabled is not None:
                cfg.dpm_enabled = dpm_enabled
            if dpm_labels is not None:
                cfg.dpm_labels = dpm_labels
            if dpm_mode is not None:
                cfg.dpm_mode = dpm_mode
            if dpm_color is not None:
                cfg.dpm_color = dpm_color
        resp = stub.SetPrivacyMaskConfig(cfg)
        _check_status(resp, "SetPrivacyMaskConfig")

    @staticmethod
    def _add_mask_region(cfg: "camera_pb2.PrivacyMaskConfig",
                         region: dict) -> None:
        r = cfg.regions.add()
        r.id = region.get("id", "")
        r.name = region.get("name", "")
        r.enabled = region.get("enabled", True)
        r.points_x.extend(region.get("points_x", []))
        r.points_y.extend(region.get("points_y", []))

    # -- OSD read-back --

    def get_osd(self) -> List[dict]:
        """Read back OSD overlay config, shaped like set_osd() input.

        Returns: list of {stream_name, text_overlays, datetime_overlays,
        image_overlays}; each overlay is a plain dict of its proto fields,
        so the result can be edited and fed straight back into set_osd().
        """
        stub = self._connect()
        resp = stub.GetOsdConfig(camera_pb2.Empty())
        streams = []
        for sc in resp.streams:
            entry = {
                "stream_name": sc.stream_name,
                "text_overlays": [],
                "datetime_overlays": [],
                "image_overlays": [],
            }
            for tc in sc.text_overlays:
                entry["text_overlays"].append({
                    "id": tc.id, "text": tc.text, "x": tc.x, "y": tc.y,
                    "font_size": tc.font_size, "text_color": tc.text_color,
                    "enabled": tc.enabled, "h_align": tc.h_align,
                    "v_align": tc.v_align})
            for dc in sc.datetime_overlays:
                entry["datetime_overlays"].append({
                    "id": dc.id, "x": dc.x, "y": dc.y, "format": dc.format,
                    "font_size": dc.font_size, "text_color": dc.text_color,
                    "enabled": dc.enabled, "h_align": dc.h_align,
                    "v_align": dc.v_align})
            for ic in sc.image_overlays:
                entry["image_overlays"].append({
                    "id": ic.id, "image_path": ic.image_path, "x": ic.x,
                    "y": ic.y, "width": ic.width, "height": ic.height,
                    "enabled": ic.enabled, "h_align": ic.h_align,
                    "v_align": ic.v_align})
            streams.append(entry)
        return streams

    # -- Raw config fields --

    def get_config_field(self, field_path: str) -> str:
        """Read one camera-daemon config field by dotted path.

        Args:
            field_path: e.g. "frontend.hailort.use-hailort-service".

        Returns: the current value encoded as a string.
        """
        stub = self._connect()
        resp = stub.GetConfigField(
            camera_pb2.GetConfigFieldRequest(field_path=field_path))
        _check_status(resp, "GetConfigField")
        return resp.value

    def set_config_field(self, field_path: str, value) -> None:
        """Write one camera-daemon config field by dotted path.

        Args:
            field_path: dotted config path.
            value: value encoded as a string (e.g. "true", "42", "1.5");
                non-str values are converted with str().
        """
        stub = self._connect()
        resp = stub.SetConfigField(camera_pb2.SetConfigFieldRequest(
            field_path=field_path,
            value=value if isinstance(value, str) else str(value)))
        _check_status(resp, "SetConfigField")

    # -- Stream management --

    def get_stream_status(self) -> List[StreamStatus]:
        stub = self._connect()
        resp = stub.GetStreamStatus(camera_pb2.GetStreamStatusRequest())
        return [
            StreamStatus(
                stream_id=s.stream_id,
                status=s.status,
                has_encoder=s.has_encoder,
                codec=s.codec,
                width=s.width,
                height=s.height,
                fps=s.fps,
                bitrate_bps=s.bitrate_bps,
                gop=s.gop,
            )
            for s in resp.streams
        ]

    def add_stream(self, stream_id: str, width: int, height: int, fps: int,
                   codec: str = "h264", bitrate: int = 4_000_000,
                   gop: int = 30) -> None:
        stub = self._connect()
        resp = stub.AddStream(camera_pb2.AddStreamRequest(
            stream_id=stream_id,
            width=width, height=height, fps=fps,
            codec=codec, bitrate=bitrate, gop=gop,
        ))
        if not resp.success:
            raise RuntimeError(f"AddStream failed: {resp.message}")

    def remove_stream(self, stream_name: str) -> None:
        stub = self._connect()
        resp = stub.RemoveStream(camera_pb2.RemoveStreamRequest(stream_name=stream_name))
        if not resp.success:
            raise RuntimeError(f"RemoveStream failed: {resp.message}")

    # -- Pipeline reconfiguration --

    def reconfigure_pipeline(self, streams: List[PipelineStreamConfig]) -> EncoderReconfigResult:
        req = camera_pb2.ReconfigurePipelineRequest()
        for s in streams:
            sc = req.streams.add()
            sc.stream_id = s.stream_id
            sc.input_width = s.input_width
            sc.input_height = s.input_height
            sc.input_framerate = s.input_framerate
            sc.codec = s.codec
            sc.encoder_width = s.encoder_width
            sc.encoder_height = s.encoder_height
            sc.encoder_framerate = s.encoder_framerate
            sc.encoder_bitrate = s.encoder_bitrate
            sc.encoder_gop = s.encoder_gop

        stub = self._connect()
        resp = stub.ReconfigurePipeline(req)
        if not resp.success:
            raise RuntimeError(f"ReconfigurePipeline failed: {resp.message}")
        return EncoderReconfigResult(
            success=resp.success,
            message=resp.message,
            interrupt_ms=resp.interrupt_ms,
        )

    # -- Profiles --

    def get_profile(self) -> str:
        stub = self._connect()
        resp = stub.GetProfile(camera_pb2.Empty())
        return resp.profile_name

    def list_profiles(self) -> tuple[list[str], str]:
        """Returns (profile_names, current_profile)."""
        stub = self._connect()
        resp = stub.ListProfiles(camera_pb2.Empty())
        return list(resp.profiles), resp.current_profile

    def switch_profile(self, name: str) -> EncoderReconfigResult:
        stub = self._connect()
        resp = stub.SwitchProfile(camera_pb2.SwitchProfileRequest(profile_name=name))
        if not resp.success:
            raise RuntimeError(f"SwitchProfile failed: {resp.message}")
        return EncoderReconfigResult(
            success=resp.success,
            message=resp.message,
            interrupt_ms=resp.interrupt_ms,
        )

    def backup_profile(self, path: str = "") -> None:
        stub = self._connect()
        resp = stub.BackupProfile(camera_pb2.BackupProfileRequest(path=path))
        if not resp.success:
            raise RuntimeError(f"BackupProfile failed: {resp.message}")

    # -- Sensor --

    def get_sensor_info(self, sensor_index: int = 0) -> SensorInfo:
        stub = self._connect()
        resp = stub.GetSensorInfo(camera_pb2.GetSensorInfoRequest(sensor_index=sensor_index))
        return SensorInfo(
            available=resp.available,
            sensor_model=resp.sensor_model,
            i2c_bus=resp.i2c_bus,
            i2c_address=resp.i2c_address,
            pixel_format=resp.pixel_format,
        )

    # -- Capabilities --

    def get_capabilities(self) -> Capabilities:
        stub = self._connect()
        resp = stub.GetCapabilities(camera_pb2.Empty())
        return Capabilities(
            has_video=resp.has_video,
            has_codec=resp.has_codec,
            has_led=resp.has_led,
            has_sensor=resp.has_sensor,
            has_mcu=resp.has_mcu,
            has_env_ctrl=resp.has_env_ctrl,
            has_alarm=resp.has_alarm,
            has_rs485=resp.has_rs485,
            has_osd=resp.has_osd,
            has_draw=resp.has_draw,
            has_audio=resp.has_audio,
        )

    # -- Hardware status --

    def get_hardware_status(self) -> HardwareStatus:
        stub = self._connect()
        resp = stub.GetDeviceHardwareStatus(camera_pb2.Empty())
        if not resp.success:
            raise RuntimeError(f"GetDeviceHardwareStatus failed: {resp.message}")
        return HardwareStatus(
            light_sensor_mv=resp.light_sensor_mv,
            light_sensor_lux=resp.light_sensor_lux,
            mcu_temp_millic=resp.mcu_temp_millic,
            ain_mv=resp.ain_mv,
            mcu_version=resp.mcu_version,
            white_light_duty=resp.white_light_duty,
            ir_led_duty=resp.ir_led_duty,
            ircut_mode=resp.ircut_mode,
        )

    # -- LED --

    def set_led_duty(self, led_id: int, duty_percent: int) -> None:
        stub = self._connect()
        resp = stub.SetLedDuty(camera_pb2.SetLedDutyRequest(
            led_id=led_id, duty_percent=duty_percent,
        ))
        if not resp.success:
            raise RuntimeError(f"SetLedDuty failed: {resp.message}")

    def get_led_duty(self, led_id: int) -> int:
        stub = self._connect()
        resp = stub.GetLedDuty(camera_pb2.GetLedDutyRequest(led_id=led_id))
        if not resp.success:
            raise RuntimeError(f"GetLedDuty failed: {resp.message}")
        return resp.duty_percent

    # -- IR-Cut --

    def set_ircut(self, mode: int) -> int:
        """Set IR-cut filter. mode: 0=day, 1=night. Returns current mode."""
        stub = self._connect()
        resp = stub.SetIrCut(camera_pb2.SetIrCutRequest(mode=mode))
        if not resp.success:
            raise RuntimeError(f"SetIrCut failed: {resp.message}")
        return resp.current_mode

    def get_ircut(self) -> int:
        stub = self._connect()
        resp = stub.GetIrCut(camera_pb2.Empty())
        return resp.current_mode

    # -- MCU raw --

    def mcu_raw_request(self, cmd: int, payload: bytes = b"") -> bytes:
        stub = self._connect()
        resp = stub.McuRawRequest(camera_pb2.McuRawRequestMessage(cmd=cmd, payload=payload))
        if not resp.success:
            raise RuntimeError(f"McuRawRequest failed: {resp.message}")
        return resp.payload

    # -- Environment control --

    def set_fan(self, enable: bool) -> bool:
        stub = self._connect()
        resp = stub.SetFan(camera_pb2.EnvCtrlRequest(enable=enable))
        if not resp.success:
            raise RuntimeError(f"SetFan failed: {resp.message}")
        return resp.enabled

    def get_fan(self) -> EnvStatus:
        stub = self._connect()
        resp = stub.GetFan(camera_pb2.Empty())
        return EnvStatus(enabled=resp.enabled)

    def set_heat(self, enable: bool) -> bool:
        stub = self._connect()
        resp = stub.SetHeat(camera_pb2.EnvCtrlRequest(enable=enable))
        if not resp.success:
            raise RuntimeError(f"SetHeat failed: {resp.message}")
        return resp.enabled

    def get_heat(self) -> EnvStatus:
        stub = self._connect()
        resp = stub.GetHeat(camera_pb2.Empty())
        return EnvStatus(enabled=resp.enabled)

    def set_radar(self, enable: bool) -> bool:
        stub = self._connect()
        resp = stub.SetRadar(camera_pb2.EnvCtrlRequest(enable=enable))
        if not resp.success:
            raise RuntimeError(f"SetRadar failed: {resp.message}")
        return resp.enabled

    def get_radar(self) -> EnvStatus:
        stub = self._connect()
        resp = stub.GetRadar(camera_pb2.Empty())
        return EnvStatus(enabled=resp.enabled)

    # -- Alarm I/O --

    def set_alarm_out(self, channel: int, enable: bool) -> bool:
        stub = self._connect()
        resp = stub.SetAlarmOut(camera_pb2.AlarmOutRequest(channel=channel, enable=enable))
        if not resp.success:
            raise RuntimeError(f"SetAlarmOut failed: {resp.message}")
        return resp.enabled

    def get_alarm_out(self, channel: int) -> bool:
        stub = self._connect()
        resp = stub.GetAlarmOut(camera_pb2.AlarmOutRequest(channel=channel))
        if not resp.success:
            raise RuntimeError(f"GetAlarmOut failed: {resp.message}")
        return resp.enabled

    def get_alarm_outputs(self) -> dict:
        stub = self._connect()
        resp = stub.GetAlarmOutputs(camera_pb2.Empty())
        if not resp.success:
            raise RuntimeError(f"GetAlarmOutputs failed: {resp.message}")
        return {
            "alarm_out0": resp.alarm_out0,
            "alarm_out1": resp.alarm_out1,
            "wiegand0": resp.wiegand0,
            "wiegand1": resp.wiegand1,
        }

    # -- RS485 --

    def rs485_init(self, baudrate: int = 9600, config: str = "8N1") -> None:
        stub = self._connect()
        resp = stub.Rs485Init(camera_pb2.Rs485InitRequest(baudrate=baudrate, config=config))
        _check_status(resp, "Rs485Init")

    def rs485_deinit(self) -> None:
        stub = self._connect()
        resp = stub.Rs485Deinit(camera_pb2.Empty())
        _check_status(resp, "Rs485Deinit")

    def rs485_tx(self, data: bytes) -> None:
        stub = self._connect()
        resp = stub.Rs485Tx(camera_pb2.Rs485TxRequest(data=data))
        _check_status(resp, "Rs485Tx")

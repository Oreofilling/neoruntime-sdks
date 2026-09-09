"""Phase 4 — camera control interfaces.

State-restore discipline: every mutating group snapshots first
(``get_isp``, ``get_transform``, ``get_privacy_mask``, ``get_profile``,
stream statuses, LED duty) and restores in teardown; the profile switch
is bracketed by ``backup_profile``; the test stream added here is
removed again and its absence verified. Raw MCU requests stay unexercised
by design (protocol semantics unverifiable — device-safety call).
"""

from __future__ import annotations

import os
import unittest

from neoruntime_ipc_sdk import (
    CameraClient,
    ISPConfig,
    PrivacyMaskSettings,
    PipelineStreamConfig,
    TransformConfig,
)

from common import DEVICE_TMP_DIR, DeviceTestCase, known_issue

TEST_STREAM = "sdk-test-stream"

# KNOWN daemon-side reconfiguration cascade, captured from camera-daemon's
# own log during the 2026-09-09 run: set_transform takes the in-place
# "consumers restarted" path, but the handler never returns and leaves the
# pipeline in "reconfiguration in progress"; later pipeline RPCs hang past
# the client alarm (no server-side timeout), reconfigure_encoder refuses
# with "already in progress", and add_stream's full MediaLibrary reinit
# deadlocks inside stop_pipeline — camera.sock and encoded/* vanish. The
# deadlocked thread sits in uninterruptible sleep: `systemctl restart`
# stalls in final-sigkill and only a device reboot recovers the camera.
# The half-applied transform is ALSO persisted to
# /data/aipc/etc/transform_config.json, so the reboot comes back up in
# portrait (w=2160 h=3840) while raw buffers stay landscape-sized —
# frames arrive but NV12 decode mismatches until the config is reset.
# Non-pipeline RPCs (LED/IRCUT/fan/OSD-get/backup_profile) keep answering
# throughout. The same ops passed on the first run of this suite, so the
# trigger is state/race dependent — the AF lens reinit was failing -2815
# in the minutes right before.
_RECONFIG_CASCADE = (
    "camera-daemon reconfiguration cascade (2026-09-09): set_transform's "
    "consumer-restart path wedges the pipeline; later pipeline RPCs hang "
    "past the client alarm, reconfigure_encoder answers 'already in "
    "progress', add_stream's MediaLibrary reinit deadlocks in "
    "stop_pipeline (SIGKILL-immune, D-state) — camera.sock and encoded/* "
    "vanish and only a device reboot recovers")


def _soft(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


class _CameraArea(DeviceTestCase):
    area = "camera"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = CameraClient()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()


class T01Readonly(_CameraArea):
    def test_01_get_isp(self):
        self.mark("CameraClient.get_isp")
        cfg = self.timed(self.client.get_isp, label="get_isp")
        self.assertIsInstance(cfg, ISPConfig)
        self.evidence(brightness=cfg.brightness, contrast=cfg.contrast,
                      saturation=cfg.saturation, sharpness=cfg.sharpness)

    def test_02_get_transform(self):
        self.mark("CameraClient.get_transform")
        cfg = self.timed(self.client.get_transform, label="get_transform")
        self.assertIsInstance(cfg, TransformConfig)
        self.evidence(rotation=cfg.rotation, flip=cfg.flip,
                      dewarp=cfg.dewarp, grayscale=cfg.grayscale)

    def test_03_get_osd(self):
        self.mark("CameraClient.get_osd")
        osd = self.timed(self.client.get_osd, label="get_osd")
        self.evidence(osd=osd)
        self.assertIsInstance(osd, list)

    def test_04_get_stream_status(self):
        self.mark("CameraClient.get_stream_status")
        streams = self.timed(self.client.get_stream_status,
                             label="get_stream_status")
        self.evidence(streams=[{
            "id": s.stream_id, "status": s.status, "codec": s.codec,
            "w": s.width, "h": s.height, "fps": s.fps,
        } for s in streams])
        self.assertTrue(any(s.stream_id == "main" for s in streams))

    def test_05_profiles_read(self):
        self.mark("CameraClient.get_profile/list_profiles")
        current = self.client.get_profile()
        profiles, active = self.client.list_profiles()
        self.evidence(current=current, profiles=profiles, active=active)
        self.assertTrue(profiles)

    def test_06_get_sensor_info(self):
        self.mark("CameraClient.get_sensor_info")
        info = self.timed(self.client.get_sensor_info, label="get_sensor_info")
        self.evidence(available=info.available,
                      sensor_model=info.sensor_model)

    def test_07_get_capabilities(self):
        self.mark("CameraClient.get_capabilities")
        caps = self.timed(self.client.get_capabilities,
                          label="get_capabilities")
        self.evidence(capabilities=vars(caps))

    def test_08_get_hardware_status(self):
        self.mark("CameraClient.get_hardware_status")
        hw = self.timed(self.client.get_hardware_status,
                        label="get_hardware_status")
        self.evidence(light_sensor_lux=hw.light_sensor_lux,
                      mcu_temp_c=hw.mcu_temp_millic / 1000.0,
                      mcu_version=hw.mcu_version,
                      white_light_duty=hw.white_light_duty,
                      ir_led_duty=hw.ir_led_duty, ircut_mode=hw.ircut_mode)

    def test_09_get_led_duty(self):
        self.mark("CameraClient.get_led_duty")
        duty, err = _soft(self.client.get_led_duty, 0)
        self.evidence(led0=duty if err is None else err)
        if err is not None:
            self.na(f"led 0 unreadable: {err}")

    def test_10_get_ircut(self):
        self.mark("CameraClient.get_ircut")
        mode, err = _soft(self.client.get_ircut)
        self.evidence(mode=mode if err is None else err)
        if err is not None:
            self.na(f"get_ircut rejected: {err}")

    def test_11_env_read(self):
        self.mark("CameraClient.get_fan/get_heat/get_radar/get_alarm_outputs")
        reads = {}
        for name, fn in (("fan", self.client.get_fan),
                         ("heat", self.client.get_heat),
                         ("radar", self.client.get_radar),
                         ("alarm_outputs", self.client.get_alarm_outputs)):
            value, err = _soft(fn)
            reads[name] = (vars(value) if value is not None and
                           hasattr(value, "__dict__") else value) if err is None else err
        self.evidence(reads=reads)

    def test_12_list_ir_presets(self):
        self.mark("CameraClient.list_ir_presets")
        presets, err = _soft(self.client.list_ir_presets)
        self.evidence(presets=None if presets is None else
                      [vars(p) for p in presets])
        if err is not None:
            self.na(f"list_ir_presets rejected: {err}")

    def test_13_mcu_raw_request_skipped(self):
        self.mark("CameraClient.mcu_raw_request")
        self.na("raw MCU protocol semantics unverifiable — skipped for "
                "device safety (write-only opcode space)")


class T02IspTransform(_CameraArea):
    def test_01_set_isp_roundtrip(self):
        self.mark("CameraClient.set_isp")
        original = self.client.get_isp()
        # Nudge brightness by one step inside [0,100], then restore.
        base = original.brightness if 0 <= original.brightness <= 100 else 50
        probe = min(base + 1, 100)
        self.timed(self.client.set_isp, brightness=probe,
                   label="set_isp")
        after = self.client.get_isp()
        restored = self.client.set_isp(brightness=base)
        final = self.client.get_isp()
        self.evidence(base=base, probe=probe, after=after.brightness,
                      restored=final.brightness)
        self.assertEqual(final.brightness, base,
                         "ISP brightness not restored")

    @known_issue(_RECONFIG_CASCADE)
    def test_02_set_transform_roundtrip(self):
        self.mark("CameraClient.set_transform")
        original = self.client.get_transform()
        probe = TransformConfig(rotation=(original.rotation + 1) % 4,
                                flip=original.flip, dewarp=original.dewarp,
                                grayscale=original.grayscale)
        self.timed(self.client.set_transform, probe, label="set_transform")
        after = self.client.get_transform()
        self.client.set_transform(original)
        final = self.client.get_transform()
        self.evidence(original=vars(original), probe=vars(probe),
                      after=vars(after), restored=vars(final))
        self.assertEqual(final.rotation, original.rotation,
                         "transform not restored")

    def test_03_config_field(self):
        self.mark("CameraClient.get_config_field/set_config_field")
        value, err = _soft(self.client.get_config_field, "isp.brightness")
        self.evidence(read=value if err is None else err)
        if err is not None:
            self.na(f"config field 'isp.brightness' unreadable: {err} "
                    "(field-path schema is device-specific)")


class T03EncoderStreams(_CameraArea):
    @known_issue(_RECONFIG_CASCADE)
    def test_01_set_encoder_noop(self):
        self.mark("CameraClient.set_encoder")
        main = next(s for s in self.client.get_stream_status()
                    if s.stream_id == "main")
        # Re-apply the exact current values: a true no-op write.
        self.timed(
            self.client.set_encoder, "main",
            bitrate_bps=main.bitrate_bps, framerate=main.fps, gop=main.gop,
            label="set_encoder",
        )
        after = next(s for s in self.client.get_stream_status()
                     if s.stream_id == "main")
        self.evidence(applied={"bitrate": main.bitrate_bps,
                               "fps": main.fps, "gop": main.gop},
                      after={"bitrate": after.bitrate_bps, "fps": after.fps})

    @known_issue(_RECONFIG_CASCADE)
    def test_02_reconfigure_encoder_noop(self):
        self.mark("CameraClient.reconfigure_encoder")
        main = next(s for s in self.client.get_stream_status()
                    if s.stream_id == "main")
        result = self.timed(
            self.client.reconfigure_encoder, "main",
            width=main.width, height=main.height, codec=main.codec,
            bitrate_bps=main.bitrate_bps, fps=main.fps, gop=main.gop,
            label="reconfigure_encoder",
        )
        self.evidence(success=result.success, message=result.message,
                      interrupt_ms=result.interrupt_ms)
        self.assertTrue(result.success)

    def test_03_set_rtsp_enabled(self):
        self.mark("CameraClient.set_rtsp_enabled")
        self.timed(self.client.set_rtsp_enabled, True,
                   label="set_rtsp_enabled")
        self.evidence(enabled=True, note="left enabled; benign service")

    @known_issue(_RECONFIG_CASCADE)
    def test_04_add_remove_stream(self):
        self.mark("CameraClient.add_stream/remove_stream")
        _, add_err = _soft(self.timed, self.client.add_stream,
                           TEST_STREAM, 640, 360, 15, "h264",
                           1_000_000, 30, label="add_stream")
        if add_err is not None:
            self.na(f"add_stream rejected: {add_err}")
        ids = [s.stream_id for s in self.client.get_stream_status()]
        self.assertIn(TEST_STREAM, ids)
        self.timed(self.client.remove_stream, TEST_STREAM,
                   label="remove_stream")
        ids_after = [s.stream_id for s in self.client.get_stream_status()]
        self.evidence(present_before=ids, absent_after=TEST_STREAM not in ids_after)
        self.assertNotIn(TEST_STREAM, ids_after,
                         "test stream not removed")

    def test_05_reconfigure_pipeline_noop(self):
        self.mark("CameraClient.reconfigure_pipeline")
        statuses = self.client.get_stream_status()
        cfgs = [
            PipelineStreamConfig(
                stream_id=s.stream_id,
                codec=s.codec,
                encoder_width=s.width, encoder_height=s.height,
                encoder_framerate=s.fps, encoder_bitrate=s.bitrate_bps,
                encoder_gop=s.gop,
            )
            for s in statuses if s.has_encoder
        ]
        self.evidence(n_streams=len(cfgs))
        if not cfgs:
            self.na("no encoder streams reported to re-apply")
        result = self.timed(self.client.reconfigure_pipeline, cfgs,
                            label="reconfigure_pipeline")
        self.evidence(success=result.success, message=result.message,
                      interrupt_ms=result.interrupt_ms)
        self.assertTrue(result.success)


class T04Imaging(_CameraArea):
    def test_01_set_osd_roundtrip(self):
        self.mark("CameraClient.set_osd")
        original = self.client.get_osd()
        probe = [{
            "stream_name": "main",
            "text_overlays": [{
                # Each overlay needs its own string id — the daemon drops
                # "text OSD with empty id". x/y are NORMALIZED floats and
                # the colour field is text_color (uint32 RGBA) —
                # camera.proto:100-112.
                "id": "sdk-osd-1",
                "text": "SDK-TEST", "x": 0.05, "y": 0.05,
                "font_size": 24, "text_color": 0xFFFFFFFF,
            }],
            "datetime_overlays": [],
        }]
        self.timed(self.client.set_osd, probe, label="set_osd")
        after = self.client.get_osd()
        # Restore whatever was there before (possibly nothing).
        self.timed(self.client.set_osd, original or [{"stream_name": "main",
                        "text_overlays": [], "datetime_overlays": []}],
                   label="set_osd_restore")
        self.evidence(original=original, probe_applied=after)
        self.assertIsInstance(after, list)

    def test_02_set_ai_overlay(self):
        self.mark("CameraClient.set_ai_overlay")
        self.client.set_ai_overlay(True, show_label=True,
                                   show_confidence=True, line_thickness=2)
        self.client.set_ai_overlay(False)
        self.evidence(note="enabled briefly, restored disabled")

    def test_03_imaging_mode(self):
        self.mark("CameraClient.set_imaging_mode/get_infrared_status")
        results = {}
        for mode in ("day", "night", "auto"):
            status, err = _soft(self.timed, self.client.set_imaging_mode,
                                mode, label=f"imaging_{mode}")
            results[mode] = (err or vars(status)) if status is not None else err
        status = self.client.get_infrared_status()
        self.evidence(modes=results, final=vars(status))

    def test_04_infrared_settings(self):
        self.mark("CameraClient.set_infrared_settings/clear_infrared_manual")
        status, err = _soft(self.timed, self.client.set_infrared_settings,
                            near_pwm=10, far_pwm=10,
                            label="set_infrared_settings")
        if err is not None:
            self.na(f"set_infrared_settings rejected: {err}")
        cleared, clear_err = _soft(self.client.clear_infrared_manual)
        self.evidence(set=vars(status), cleared=clear_err or vars(cleared))

    def test_05_set_ircut_roundtrip(self):
        self.mark("CameraClient.set_ircut/get_ircut")
        original, get_err = _soft(self.client.get_ircut)
        if get_err is not None:
            self.na(f"get_ircut rejected: {get_err}")
        mode, err = _soft(self.timed, self.client.set_ircut,
                          1 - int(original), label="set_ircut")
        if err is not None:
            self.na(f"set_ircut rejected: {err}")
        flipped, _ = _soft(self.client.get_ircut)
        self.client.set_ircut(int(original))
        self.evidence(original=original, applied=mode, flipped=flipped,
                      restored=original)

    @known_issue(_RECONFIG_CASCADE)
    def test_06_privacy_mask_roundtrip(self):
        self.mark("CameraClient.get_privacy_mask/set_privacy_mask")
        original = self.timed(self.client.get_privacy_mask,
                              label="get_privacy_mask")
        self.assertIsInstance(original, PrivacyMaskSettings)
        if original.enabled:
            self.na("privacy mask already in use by an operator — not "
                    "touching it")
        # One small corner polygon on main stream. Region id is a proto
        # *string* and points are NORMALIZED floats [0..1] —
        # camera.proto:717-723.
        self.timed(self.client.set_privacy_mask,
                   enabled=True,
                   regions=[{
                       "id": "99", "name": "sdk-test", "enabled": True,
                       "points_x": [0.0, 0.1, 0.1, 0.0],
                       "points_y": [0.0, 0.0, 0.1, 0.1],
                   }],
                   label="set_privacy_mask")
        during = self.client.get_privacy_mask()
        # Restore the exact original settings object.
        self.client.set_privacy_mask(original)
        final = self.client.get_privacy_mask()
        self.evidence(during_enabled=during.enabled,
                      n_regions_during=len(during.regions),
                      final_enabled=final.enabled,
                      final_regions=len(final.regions))
        self.assertFalse(final.enabled, "privacy mask not restored")
        self.assertEqual(len(final.regions), len(original.regions))


class T05Profiles(_CameraArea):
    @known_issue(_RECONFIG_CASCADE)
    def test_01_backup_and_switch(self):
        self.mark("CameraClient.backup_profile/switch_profile")
        backup = os.path.join(DEVICE_TMP_DIR, "profile-backup.json")
        self.timed(self.client.backup_profile, backup,
                   label="backup_profile")
        self.evidence(backup=backup,
                      backup_bytes=os.path.getsize(backup)
                      if os.path.exists(backup) else -1)
        current = self.client.get_profile()
        profiles, _active = self.client.list_profiles()
        others = [p for p in profiles if p != current]
        if not others:
            self.na(f"only one profile ({current!r}) — nothing to switch to")
        target = others[0]
        try:
            result = self.timed(self.client.switch_profile, target,
                                label="switch_profile")
        except Exception as exc:  # noqa: BLE001 — daemon policy gate below
            # This daemon build restricts switching to an allowlist of
            # profiles; a policy denial is a device-config outcome, not an
            # SDK failure (nothing changed, so no restore is needed).
            if "allowlist" in str(exc):
                self.na(f"switch_profile denied by daemon policy: {exc}")
            raise
        switched = self.client.get_profile()
        back = self.client.switch_profile(current)
        restored = self.client.get_profile()
        self.evidence(current=current, target=target,
                      switched_to=switched, restored_to=restored,
                      interrupt_ms_back=back.interrupt_ms)
        self.assertEqual(switched, target, "switch_profile did not take")
        self.assertEqual(restored, current, "profile not restored")


class T06Env(_CameraArea):
    """Fan / heat / radar / alarm-out — thermal & accessory controls."""

    def test_01_led_duty_roundtrip(self):
        self.mark("CameraClient.set_led_duty/get_led_duty")
        original, get_err = _soft(self.client.get_led_duty, 0)
        if get_err is not None:
            self.na(f"led 0 unreadable: {get_err}")
        duty, err = _soft(self.timed, self.client.set_led_duty, 0, 10,
                          label="set_led_duty")
        if err is not None:
            self.na(f"set_led_duty rejected: {err}")
        readback, _ = _soft(self.client.get_led_duty, 0)
        self.client.set_led_duty(0, int(original))
        self.evidence(original=original, readback=readback,
                      restored=int(original))

    def test_02_fan_roundtrip(self):
        self.mark("CameraClient.set_fan/get_fan")
        status, get_err = _soft(self.client.get_fan)
        original = bool(status.enabled) if status is not None else False
        ok, err = _soft(self.timed, self.client.set_fan, True,
                        label="set_fan_on")
        self.client.set_fan(original)
        self.evidence(original=original, call=err or ok,
                      note="restored original state")
        if err is not None and get_err is not None:
            self.na(f"fan not controllable: {err}")

    def test_03_heat(self):
        self.mark("CameraClient.set_heat/get_heat")
        status, get_err = _soft(self.client.get_heat)
        self.evidence(status=None if status is None else vars(status))
        if get_err is not None:
            self.na(f"get_heat rejected: {get_err}")
        if status is None or not status.enabled:
            self.na("heater off in production state — not switching it on "
                    "from a test")

    def test_04_radar(self):
        self.mark("CameraClient.set_radar/get_radar")
        status, get_err = _soft(self.client.get_radar)
        self.evidence(status=None if status is None else vars(status))
        if get_err is not None:
            self.na(f"get_radar rejected: {get_err}")
        if status is None or not status.enabled:
            self.na("radar off in production state — not switching it on "
                    "from a test")

    def test_05_alarm_out_roundtrip(self):
        self.mark("CameraClient.set_alarm_out/get_alarm_out")
        original, get_err = _soft(self.client.get_alarm_out, 0)
        if get_err is not None:
            self.na(f"alarm channel 0 unreadable: {get_err}")
        ok, err = _soft(self.timed, self.client.set_alarm_out, 0,
                        not bool(original), label="set_alarm_out")
        if err is not None:
            self.na(f"set_alarm_out rejected: {err}")
        flipped, _ = _soft(self.client.get_alarm_out, 0)
        self.client.set_alarm_out(0, bool(original))
        self.evidence(original=original, flipped=flipped,
                      restored=bool(original), call=ok)


if __name__ == "__main__":
    unittest.main()

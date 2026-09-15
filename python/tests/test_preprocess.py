"""Preprocessor + PreprocessMeta (tier-3 client-side preprocessing)."""

import numpy as np
import pytest

from neoruntime_ipc_sdk import Frame, ModelInfo, PreprocessMeta, Preprocessor


def make_rgb_frame(w, h):
    return Frame(sequence=1, timestamp_ns=0, width=w, height=h, format="RGB",
                 image=np.zeros((h, w, 3), np.uint8))


class TestValidation:
    def test_bad_color_rejected(self):
        with pytest.raises(ValueError, match="color"):
            Preprocessor(size=(64, 64), color="YUV")

    def test_bad_resize_mode_rejected(self):
        with pytest.raises(ValueError, match="resize_mode"):
            Preprocessor(size=(64, 64), resize_mode="pad")

    def test_bad_layout_rejected(self):
        with pytest.raises(ValueError, match="layout"):
            Preprocessor(size=(64, 64), layout="CHW")

    def test_missing_size_rejected_on_call(self):
        with pytest.raises(ValueError, match="size"):
            Preprocessor()(np.zeros((8, 8, 3), np.uint8))

    def test_2d_array_requires_source_format(self):
        with pytest.raises(ValueError, match="NV12.*GRAY8"):
            Preprocessor(size=(64, 64))(np.zeros((12, 8), np.uint8))

    def test_unsupported_source_format_rejected_at_construction(self):
        with pytest.raises(ValueError, match="source_format"):
            Preprocessor(size=(64, 64), source_format="YUYV")

    def test_rgba_array_rejected_with_strip_alpha_hint(self):
        # 4-channel input used to pass validation and blow up much later
        # in _to_color; now it fails fast at the first call.
        with pytest.raises(ValueError, match="RGBA"):
            Preprocessor(size=(64, 64))(np.zeros((48, 64, 4), np.uint8))


class TestCall:
    def test_letterbox_geometry_matches_transform(self):
        pre = Preprocessor(size=(640, 640))
        tensor, meta = pre(make_rgb_frame(1920, 1080))
        assert tensor.shape == (640, 640, 3) and tensor.dtype == np.uint8
        assert meta.original_size == (1920, 1080)
        assert meta.input_size == (640, 640)
        assert meta.origin == (0, 140)
        assert meta.scale[0] == pytest.approx(640 / 1920)

    def test_identity_when_size_matches(self):
        pre = Preprocessor(size=(64, 48))
        tensor, meta = pre(make_rgb_frame(64, 48))
        assert tensor.shape == (48, 64, 3)
        assert meta.scale == (1.0, 1.0) and meta.origin == (0, 0)

    def test_ndarray_input_wrapped(self):
        pre = Preprocessor(size=(32, 32), source_format="RGB")
        tensor, meta = pre(np.zeros((48, 64, 3), np.uint8))
        assert tensor.shape == (32, 32, 3)
        assert meta.original_size == (64, 48)

    def test_nv12_source(self):
        pre = Preprocessor(size=(32, 32))
        nv12 = Frame(sequence=1, timestamp_ns=0, width=64, height=48, format="NV12",
                     image=np.zeros((72, 64), np.uint8))
        tensor, meta = pre(nv12)
        assert tensor.shape == (32, 32, 3)
        assert meta.origin == (0, 4)  # 32x24 content, chroma-aligned

    def test_bgr_swap(self):
        pre = Preprocessor(size=(64, 48), color="BGR", source_format="RGB")
        src = np.zeros((48, 64, 3), np.uint8)
        src[:, :, 0] = 255  # all red in RGB terms
        tensor, _ = pre(src)
        assert tensor[0, 0, 2] == 255 and tensor[0, 0, 0] == 0  # now blue channel

    def test_normalize_makes_float(self):
        pre = Preprocessor(size=(64, 48), normalize=True)
        tensor, _ = pre(make_rgb_frame(64, 48))
        assert tensor.dtype == np.float32

    def test_nchw_layout(self):
        pre = Preprocessor(size=(64, 48), layout="NCHW")
        tensor, meta = pre(make_rgb_frame(64, 48))
        assert tensor.shape == (3, 48, 64)
        assert meta.layout == "NCHW"


class TestFromModel:
    @staticmethod
    def _client(shape, dtype=0, layout=""):
        class _Fake:
            def get_model_info(self, model_id):
                return ModelInfo(
                    model_id=model_id, model_path="x.hef", version="1",
                    inputs=[{"shape": shape, "dtype": dtype, "name": "in",
                             "layout": layout}],
                )

        return _Fake()

    def test_derives_nhwc_uint8(self):
        pre = Preprocessor.from_model(self._client([1, 640, 640, 3], dtype=0), "m")
        assert pre.size == (640, 640) and pre.layout == "NHWC"
        assert pre.normalize is False  # uint8 HEF: quantisation lives in the HEF

    def test_derives_nchw_from_shape(self):
        pre = Preprocessor.from_model(self._client([1, 3, 416, 416], dtype=5), "m")
        assert pre.size == (416, 416) and pre.layout == "NCHW"
        assert pre.normalize is True  # float32 input

    def test_layout_hint_wins(self):
        pre = Preprocessor.from_model(
            self._client([1, 640, 640, 3], layout="NHWC"), "m"
        )
        assert pre.layout == "NHWC"

    def test_overrides(self):
        pre = Preprocessor.from_model(self._client([1, 640, 640, 3]), "m",
                                      color="BGR", resize_mode="stretch")
        assert pre.color == "BGR" and pre.resize_mode == "stretch"

    def test_unknown_model_raises(self):
        class _Missing:
            def get_model_info(self, model_id):
                return None

        with pytest.raises(ValueError, match="not registered"):
            Preprocessor.from_model(_Missing(), "nope")


class TestPreprocessMeta:
    def test_roundtrip_against_frame_transform(self):
        f = make_rgb_frame(1920, 1080).resize(640, 640, mode="letterbox")
        t = f.metadata["transform"]
        meta = PreprocessMeta(original_size=t["src_size"], input_size=t["dst_size"],
                              scale=t["scale"], origin=t["origin"])
        x, y = meta.to_source(320, 320)
        assert x == pytest.approx(960, abs=1)
        assert y == pytest.approx(540, abs=1)

    def test_to_source_box(self):
        meta = PreprocessMeta(original_size=(100, 100), input_size=(50, 50),
                              scale=(0.5, 0.5), origin=(0, 10))
        assert meta.to_source_box((10, 10, 30, 30)) == (20.0, 0.0, 60.0, 40.0)


class TestReviewFixes:
    """Regressions from the pre-merge review."""

    def test_identity_path_inherits_existing_transform(self):
        # Review P2: a frame already cropped/resize'd to model size must not
        # lose its transform — postprocessing maps back to the ORIGINAL frame.
        cropped = make_rgb_frame(1920, 1080).crop(960, 220, 640, 640)
        pre = Preprocessor(size=(640, 640))
        tensor, meta = pre(cropped)

        assert tensor.shape == (640, 640, 3)
        assert meta.original_size == (1920, 1080)  # not (640, 640)
        x, y = meta.to_source(320, 320)
        assert x == pytest.approx(1280, abs=1)
        assert y == pytest.approx(540, abs=1)

    def test_lowercase_params_accepted(self):
        # Review P3: "rgb"/"nchw"/"nv12" style values normalise instead of
        # failing.
        pre = Preprocessor(size=(64, 48), color="bgr", layout="nchw",
                           source_format="rgb")
        assert pre.color == "BGR" and pre.layout == "NCHW"
        assert pre.source_format == "RGB"
        tensor, _ = pre(np.zeros((48, 64, 3), np.uint8))
        assert tensor.shape == (3, 48, 64)

    def test_lowercase_source_format_nv12(self):
        pre = Preprocessor(size=(32, 32), source_format="nv12")
        nv12 = Frame(sequence=1, timestamp_ns=0, width=64, height=48,
                     format="NV12", image=np.zeros((72, 64), np.uint8))
        tensor, _ = pre(nv12)
        assert tensor.shape == (32, 32, 3)


class TestNv12Passthrough:
    """P2-6: the device-resident NV12 preprocessing leg."""

    @staticmethod
    def _nv12_frame(w, h, value=128):
        return Frame(sequence=1, timestamp_ns=0, width=w, height=h,
                     format="NV12",
                     image=np.full((h * 3 // 2, w), value, np.uint8))

    @staticmethod
    def _fake_router(backend="hardware", policy=None):
        class _Route:
            def __init__(self, backend):
                self.backend = backend

        class _Router:
            def __init__(self):
                self.policy = policy
                self.notes = []

            def route(self, op):
                return _Route(backend)

            def note_degradation(self, op, reason):
                self.notes.append((op, reason))

        return _Router()

    # -- constructor contract ------------------------------------------------

    def test_requires_stretch(self):
        # default resize_mode is letterbox; passthrough must refuse it
        with pytest.raises(ValueError, match="stretch"):
            Preprocessor(size=(64, 48), nv12_passthrough=True)

    def test_requires_no_normalize(self):
        with pytest.raises(ValueError, match="normalize"):
            Preprocessor(size=(64, 48), resize_mode="stretch", normalize=True,
                         nv12_passthrough=True)

    def test_requires_nhwc(self):
        with pytest.raises(ValueError, match="packed NV12"):
            Preprocessor(size=(64, 48), resize_mode="stretch", layout="NCHW",
                         nv12_passthrough=True)

    # -- from_model auto-enable ----------------------------------------------

    def test_from_model_auto_enables_for_nv12_layout(self):
        pre = Preprocessor.from_model(
            TestFromModel._client([1, 384, 640, 1], layout="NV12"), "m"
        )
        assert pre.nv12_passthrough is True
        assert pre.resize_mode == "stretch"
        assert pre.size == (640, 384) and pre.layout == "NHWC"

    def test_from_model_respects_resize_mode_override(self):
        # a caller who pinned letterbox keeps the old behaviour
        pre = Preprocessor.from_model(
            TestFromModel._client([1, 384, 640, 1], layout="NV12"), "m",
            resize_mode="letterbox",
        )
        assert pre.nv12_passthrough is False
        assert pre.resize_mode == "letterbox"

    def test_from_model_respects_explicit_passthrough_false(self):
        pre = Preprocessor.from_model(
            TestFromModel._client([1, 384, 640, 1], layout="NV12"), "m",
            nv12_passthrough=False,
        )
        assert pre.nv12_passthrough is False

    def test_from_model_no_auto_enable_for_rgb(self):
        pre = Preprocessor.from_model(TestFromModel._client([1, 640, 640, 3]), "m")
        assert pre.nv12_passthrough is False

    def test_from_model_no_auto_enable_for_float_dtype(self):
        # float implies normalize=True, which passthrough cannot serve
        pre = Preprocessor.from_model(
            TestFromModel._client([1, 384, 640, 1], dtype=5, layout="NV12"), "m"
        )
        assert pre.nv12_passthrough is False and pre.normalize is True

    # -- device leg ------------------------------------------------------------

    def test_device_leg_returns_ref_and_meta(self, monkeypatch):
        import neoruntime_ipc_sdk.accel as accel

        calls = []
        sentinel = object()

        def fake_call(method, *args, **kwargs):
            calls.append((method, args, kwargs))
            return sentinel

        monkeypatch.setattr(accel, "shared_dsp_call", fake_call)
        monkeypatch.setattr(
            accel, "get_default_router", lambda: self._fake_router()
        )

        pre = Preprocessor(size=(64, 32), resize_mode="stretch",
                           nv12_passthrough=True)
        frame = self._nv12_frame(128, 96)
        tensor, meta = pre(frame)

        assert tensor is sentinel
        assert len(calls) == 1
        method, args, kwargs = calls[0]
        assert method == "resize_hw"
        assert args == (frame, 64, 32)
        assert kwargs == {"fmt": "nv12", "scaling": "stretch",
                          "out": "ref", "cpu_fallback": False}
        assert meta.color == "NV12" and meta.layout == "NHWC"
        assert meta.original_size == (128, 96)
        assert meta.input_size == (64, 32)
        assert meta.scale == (0.5, 32 / 96) and meta.origin == (0, 0)

    def test_device_leg_composes_prior_transform(self, monkeypatch):
        import neoruntime_ipc_sdk.accel as accel

        monkeypatch.setattr(
            accel, "shared_dsp_call", lambda *a, **k: object()
        )
        monkeypatch.setattr(
            accel, "get_default_router", lambda: self._fake_router()
        )

        # mid already maps (128, 96) -> (64, 48); the leg's meta must map
        # model coords all the way back to (128, 96)
        mid = self._nv12_frame(128, 96).resize(64, 48, mode="stretch")
        pre = Preprocessor(size=(32, 24), resize_mode="stretch",
                           nv12_passthrough=True)
        _, meta = pre(mid)
        assert meta.original_size == (128, 96)
        assert meta.scale == (32 / 128, 24 / 96)

    # -- identity ---------------------------------------------------------------

    def test_identity_keepfd_frame_returns_source(self):
        # dummy handle: the identity branch only checks it is set (the
        # frame itself is what infer's zero-copy path wants)
        frame = Frame(sequence=1, timestamp_ns=0, width=64, height=48,
                      format="NV12", image=None, handle=object())
        pre = Preprocessor(size=(64, 48), resize_mode="stretch",
                           nv12_passthrough=True)
        tensor, meta = pre(frame)
        assert tensor is frame
        assert meta.scale == (1.0, 1.0) and meta.origin == (0, 0)
        assert meta.color == "NV12"

    def test_identity_ndarray_returns_array_not_frame(self):
        # a handle-less Frame is not a valid infer input — the array must
        # come back out
        pre = Preprocessor(size=(64, 48), resize_mode="stretch",
                           nv12_passthrough=True, source_format="NV12")
        tensor, meta = pre(np.zeros((72, 64), np.uint8))
        assert isinstance(tensor, np.ndarray) and tensor.shape == (72, 64)
        assert not hasattr(tensor, "handle")

    def test_identity_inherits_prior_transform(self):
        pre = Preprocessor(size=(64, 48), resize_mode="stretch",
                           nv12_passthrough=True)
        mid = self._nv12_frame(128, 96).resize(64, 48, mode="stretch")
        tensor, meta = pre(mid)
        assert isinstance(tensor, np.ndarray) and tensor.shape == (72, 64)
        assert meta.original_size == (128, 96)
        assert meta.scale == (0.5, 0.5)

    # -- fallbacks ----------------------------------------------------------------

    def test_non_nv12_source_rejected(self):
        pre = Preprocessor(size=(64, 48), resize_mode="stretch",
                           nv12_passthrough=True)
        with pytest.raises(ValueError, match="NV12"):
            pre(make_rgb_frame(64, 64))

    def test_dsp_down_falls_back_to_cpu(self, monkeypatch):
        import neoruntime_ipc_sdk.accel as accel
        from neoruntime_ipc_sdk.dsp import DspError

        router = self._fake_router()

        def boom(*a, **k):
            raise DspError("dsp down")

        monkeypatch.setattr(accel, "shared_dsp_call", boom)
        monkeypatch.setattr(accel, "get_default_router", lambda: router)

        frame = self._nv12_frame(128, 96)
        pre = Preprocessor(size=(64, 32), resize_mode="stretch",
                           nv12_passthrough=True)
        tensor, meta = pre(frame)

        expected = frame.resize(64, 32, mode="stretch").to_array()
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == expected.shape == (48, 64)
        assert tensor.tobytes() == expected.tobytes()
        assert meta.color == "NV12"
        assert meta.original_size == (128, 96) and meta.origin == (0, 0)
        assert router.notes and router.notes[0][0] == "resize_nv12"

    def test_software_route_takes_cpu_without_calling_dsp(self, monkeypatch):
        import neoruntime_ipc_sdk.accel as accel

        def must_not_run(*a, **k):
            raise AssertionError("router said software; DSP must not be called")

        monkeypatch.setattr(accel, "shared_dsp_call", must_not_run)
        monkeypatch.setattr(
            accel, "get_default_router", lambda: self._fake_router(backend="software")
        )

        frame = self._nv12_frame(128, 96)
        pre = Preprocessor(size=(64, 32), resize_mode="stretch",
                           nv12_passthrough=True)
        tensor, meta = pre(frame)
        expected = frame.resize(64, 32, mode="stretch").to_array()
        assert tensor.tobytes() == expected.tobytes()
        assert meta.scale == (0.5, 32 / 96)

    def test_hardware_only_policy_raises(self, monkeypatch):
        import neoruntime_ipc_sdk.accel as accel
        from neoruntime_ipc_sdk.accel import HardwareUnavailable, RoutePolicy
        from neoruntime_ipc_sdk.dsp import DspError

        def boom(*a, **k):
            raise DspError("dsp down")

        monkeypatch.setattr(accel, "shared_dsp_call", boom)
        monkeypatch.setattr(
            accel, "get_default_router",
            lambda: self._fake_router(policy=RoutePolicy.HARDWARE_ONLY),
        )

        pre = Preprocessor(size=(64, 32), resize_mode="stretch",
                           nv12_passthrough=True)
        with pytest.raises(HardwareUnavailable, match="hardware-only"):
            pre(self._nv12_frame(128, 96))

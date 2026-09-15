"""Shared fixtures for the SDK unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _reset_resident_dsp(monkeypatch):
    """Give every test a fresh process-resident DSP client.

    The accel router keeps one DspClient for the whole process
    (:func:`neoruntime_ipc_sdk.accel.shared_dsp_call`); without this
    reset, a client built under one test's patched ``dsp.DspClient``
    factory would leak into the next test and serve it stale results.
    """
    import neoruntime_ipc_sdk.accel as accel

    monkeypatch.setattr(accel, "_dsp_shared", None)
    yield

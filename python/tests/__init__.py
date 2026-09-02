"""Test suite for neoruntime_ipc_sdk.

Importing submodules here keeps legacy ``import tests`` / unittest
discovery working; pytest collects the files directly either way.
"""

from . import (
    test_config,  # noqa: F401
    test_device,  # noqa: F401
    test_events,  # noqa: F401
    test_inference,  # noqa: F401
    test_media,  # noqa: F401
    test_plugin,  # noqa: F401
)

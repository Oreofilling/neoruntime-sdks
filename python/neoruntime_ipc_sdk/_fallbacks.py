"""One-shot warnings for optional-dependency fallbacks.

The SDK's numpy legs keep every operation working without cv2, but they
are typically 10-50x slower at full-resolution frame rates. These
helpers make that degradation visible once per operation instead of
hoping the app notices its CPU usage.
"""

from __future__ import annotations

import logging
import threading

__all__ = ["warn_numpy_fallback", "reset_fallback_warnings"]

_logger = logging.getLogger("neoruntime_ipc_sdk.fallbacks")
_lock = threading.Lock()
_warned: set[str] = set()


def warn_numpy_fallback(operation: str) -> None:
    """Emit one warning per ``operation`` when a pure-numpy path replaces cv2."""
    with _lock:
        if operation in _warned:
            return
        _warned.add(operation)
    _logger.warning(
        "%s: cv2 not installed — using the pure-numpy fallback (10-50x slower "
        "on full-resolution frames). Installing opencv (cv2) is strongly "
        "recommended for performance headroom.",
        operation,
    )


def reset_fallback_warnings() -> None:
    """Forget which warnings were emitted (test helper)."""
    with _lock:
        _warned.clear()

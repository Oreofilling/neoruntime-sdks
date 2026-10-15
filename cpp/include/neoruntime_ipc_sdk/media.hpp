// media.hpp — media clients (compatibility facade).
//
// The implementation lives in frame.hpp (frame types, pixel formats,
// crop/resize), encoded.hpp (encoded-stream UDS client) and fd_client.hpp
// (zero-copy DMA-BUF fd client). This header re-exports the full historical
// surface so `#include "neoruntime_ipc_sdk/media.hpp"` keeps working
// unchanged — new code should include the specific module it needs.
#pragma once

#include "neoruntime_ipc_sdk/encoded.hpp"
#include "neoruntime_ipc_sdk/fd_client.hpp"
#include "neoruntime_ipc_sdk/frame.hpp"

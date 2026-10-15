// version.hpp — NE503 AIPC C++ SDK version.
//
// The C++ SDK is versioned independently while it tracks the Python SDK API
// surface. The 0.8.0 rewrite mirrors python/neoruntime_ipc_sdk 0.8.1:
// core domains (media/camera/device/overlay/inference/events/app/audio)
// plus the DSP domain, frame injection, and the inference tool layer
// (preprocess/postprocess/pipeline/runner/stream_pipeline/color/draw/
// diagnostics/accel). Web (MJPEG), recording, and plugin are out of scope;
// plugin.hpp keeps only its deprecation marker.
#pragma once

namespace neoruntime_ipc_sdk {

inline constexpr char kVersion[] = "0.8.0";
inline constexpr char kMirroredPythonSdk[] = "0.8.1";

}  // namespace neoruntime_ipc_sdk

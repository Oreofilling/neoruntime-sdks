# tests/CMakeLists.txt — GoogleTest mirror of python/tests/.
#
# Scope: these are the daemon-independent, x86-runnable logic tests. They cover
# env-var configuration (Config), inference value-type logic (BoundingBox /
# InferenceResult), the packed media/audio wire structs (fd_protocol.hpp), and
# AudioFrame decoding. Anything that needs a live platform daemon (a real
# ai-runtime.sock / camera.sock / DMA-BUF fd) is an on-device integration test
# and is NOT built here — see the plan's Verification section.
#
# Tests link the full ne503::aipc_sdk static lib so they exercise the real
# compiled objects (config.cpp, audio_stream.cpp, …) rather than a parallel
# re-implementation. proto-gen + gRPC + OpenCV are therefore required on the
# host, which matches the native x64 CI image.

# ---- GoogleTest -------------------------------------------------------------
# Prefer the system package (apt libgtest-dev ships GTestConfig.cmake → the
# GTest::* imported targets the executables below link). Fall back to a source
# download only when no system package is present, so this works in both bare
# CI images and vcpkg hosts.
find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        # v1.14.0 has no release assets; the auto-generated source archive is
        # the only downloadable artifact. URL_HASH pins the exact tree.
        URL "https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
    )
    # On Windows / newer toolchains gtest needs this for shared CRT; harmless elsewhere.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# ---- Test helpers ----------------------------------------------------------
# fd_protocol.hpp is a PRIVATE header (lives under src/detail/); the wire-struct
# test needs it on its include path. The public headers (Config, inference,
# AudioFrame) ride the SDK target's PUBLIC include dir.
set(NE503_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")

# ---- test_config -----------------------------------------------------------
add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_config PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_config)

# ---- test_inference --------------------------------------------------------
# BoundingBox + InferenceResult logic is header-inline, but link the SDK anyway
# for symmetry and so a future proto-backed assertion drops in cleanly.
add_executable(test_inference test_inference.cpp)
target_link_libraries(test_inference PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_inference PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_inference)

# ---- test_wire_protocol ----------------------------------------------------
add_executable(test_wire_protocol test_wire_protocol.cpp)
target_include_directories(test_wire_protocol PRIVATE "${NE503_SRC_DIR}")
target_link_libraries(test_wire_protocol PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_wire_protocol PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_wire_protocol)

# ---- test_audio_frame ------------------------------------------------------
add_executable(test_audio_frame test_audio_frame.cpp)
target_link_libraries(test_audio_frame PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_audio_frame PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_audio_frame)

# ---- test_channel_options --------------------------------------------------
# transport.hpp is a PRIVATE header and instantiates GrpcClientBase over a real
# generated service type, so it needs both the src/ and proto-gen include dirs.
add_executable(test_channel_options test_channel_options.cpp)
target_include_directories(test_channel_options PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_channel_options PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_channel_options PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_channel_options)

# ---- test_frame_geometry -----------------------------------------------------
# Phase 2 media/frame parity: detail/geometry.hpp is a PRIVATE header on the
# src/ include path.
add_executable(test_frame_geometry test_frame_geometry.cpp)
target_include_directories(test_frame_geometry PRIVATE "${NE503_SRC_DIR}")
target_link_libraries(test_frame_geometry PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_frame_geometry PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_frame_geometry)

# ---- test_fd_media_lifecycle -------------------------------------------------
# In-process fake FdPublisher (UDS listener + memfd planes via SCM_RIGHTS)
# exercising the real FdMediaClient copy/keep_fd/decimation/close protocol.
add_executable(test_fd_media_lifecycle test_fd_media_lifecycle.cpp)
target_include_directories(test_fd_media_lifecycle PRIVATE "${NE503_SRC_DIR}")
target_link_libraries(test_fd_media_lifecycle PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_fd_media_lifecycle PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_fd_media_lifecycle)

# ---- test_encoded_v3 -----------------------------------------------------------
# In-process fake EncodedPublisher (30B EncHeader + V3 seq extension) covering
# gap/restart observability and ctor path resolution.
add_executable(test_encoded_v3 test_encoded_v3.cpp)
target_include_directories(test_encoded_v3 PRIVATE "${NE503_SRC_DIR}")
target_link_libraries(test_encoded_v3 PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_encoded_v3 PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_encoded_v3)

# ---- test_camera_codec ---------------------------------------------------------
# Phase 3 camera parity: detail/camera_codec.hpp is PRIVATE and includes the
# generated camera.pb.h, so both src/ and proto-gen include dirs are needed.
add_executable(test_camera_codec test_camera_codec.cpp)
target_include_directories(test_camera_codec PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_camera_codec PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_camera_codec PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_camera_codec)

# ---- test_device_af ------------------------------------------------------------
# Phase 3 native-autofocus codec parity (detail/device_codec.hpp + device.pb.h).
add_executable(test_device_af test_device_af.cpp)
target_include_directories(test_device_af PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_device_af PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_device_af PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_device_af)

# ---- test_overlay_strict ---------------------------------------------------------
# Phase 3 overlay strict-mode wire shape (ai_overlay_config + camera.pb.h),
# ported from python/tests/test_overlay_strict.py.
add_executable(test_overlay_strict test_overlay_strict.cpp)
target_include_directories(test_overlay_strict PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_overlay_strict PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_overlay_strict PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_overlay_strict)

# ---- test_overlay_annotate -------------------------------------------------------
# Phase 3 annotate payload/metadata parity (overlay_codec + camera.pb.h),
# ported from python/tests/test_overlay_annotate.py.
add_executable(test_overlay_annotate test_overlay_annotate.cpp)
target_include_directories(test_overlay_annotate PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_overlay_annotate PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_overlay_annotate PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_overlay_annotate)

# ---- test_inference_codec ---------------------------------------------------------
# Phase 4 inference parity: detail/inference_codec.hpp is PRIVATE and includes
# the generated inference.pb.h, so both src/ and proto-gen include dirs are
# needed (same shape as test_channel_options).
add_executable(test_inference_codec test_inference_codec.cpp)
target_include_directories(test_inference_codec PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_inference_codec PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_inference_codec PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_inference_codec)

# ---- test_subscribe_health ----------------------------------------------------------
# Phase 4 subscribe health: bounded drop-oldest queue, EMA seeding, failure
# limiter, exact raise message (detail/subscribe_health.hpp, src/ path only).
add_executable(test_subscribe_health test_subscribe_health.cpp)
target_include_directories(test_subscribe_health PRIVATE "${NE503_SRC_DIR}")
target_link_libraries(test_subscribe_health PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_subscribe_health PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_subscribe_health)

# ---- test_dsp_wire --------------------------------------------------------------------
# Phase 5 DSP wire parity: detail/dsp_wire.hpp includes camera.pb.h (enum
# maps), so both src/ and proto-gen include dirs are needed.
add_executable(test_dsp_wire test_dsp_wire.cpp)
target_include_directories(test_dsp_wire PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_dsp_wire PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_dsp_wire PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_dsp_wire)

# ---- test_dsp_fallback ------------------------------------------------------------------
# Phase 5 CPU-fallback parity: detail/dsp_format.hpp is PRIVATE and pb-free,
# so only the src/ include path is needed.
add_executable(test_dsp_fallback test_dsp_fallback.cpp)
target_include_directories(test_dsp_fallback PRIVATE "${NE503_SRC_DIR}")
target_link_libraries(test_dsp_fallback PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_dsp_fallback PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_dsp_fallback)

# ---- test_dsp_lifecycle -----------------------------------------------------------------
# Phase 5 wire lifecycle: in-process fake camera.sock daemon (memfd planes via
# SCM_RIGHTS) exercising the real DspClient alloc/import/release exchanges,
# stride-padded pool write/read, and close poisoning. detail/dsp_wire.hpp
# includes camera.pb.h, so the proto-gen include dir is needed too.
add_executable(test_dsp_lifecycle test_dsp_lifecycle.cpp)
target_include_directories(test_dsp_lifecycle PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_dsp_lifecycle PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_dsp_lifecycle PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_dsp_lifecycle)

# ---- test_injection_slots ---------------------------------------------------------------
# Phase 5 FramePublisher parity: in-process fake camera.sock daemon (memfd
# planes) + in-process gRPC CameraControl service exercising the real
# publish/publish_stream/publish_rgb/push_slot protocol, the write-lease
# state machine and the ctor validation ladder (ported from python
# test_injection.py). detail/dsp_wire.hpp + camera.grpc.pb.h need both
# include dirs.
add_executable(test_injection_slots test_injection_slots.cpp)
target_include_directories(test_injection_slots PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_injection_slots PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_injection_slots PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_injection_slots)

# ---- test_color --------------------------------------------------------------------
# Phase 6 tool-layer parity: NV12 <-> RGB/BGR conversion and NV12 resize
# against python tests/test_color.py's numeric expectations (BT.601 limited
# range). Public header only (color.hpp).
add_executable(test_color test_color.cpp)
target_link_libraries(test_color PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_color PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_color)

# ---- test_draw ---------------------------------------------------------------------
# Phase 6 draw parity (draw.hpp public header): boxes/text/detections/
# polygons raster expectations ported from python tests/test_draw.py.
add_executable(test_draw test_draw.cpp)
target_link_libraries(test_draw PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_draw PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_draw)

# ---- test_diagnostics ----------------------------------------------------------------
# Phase 6 diagnostics parity (diagnostics.hpp public header): offline env
# isolation, encoded-stream listing and check() grading ported from python
# tests/test_diagnostics.py.
add_executable(test_diagnostics test_diagnostics.cpp)
target_link_libraries(test_diagnostics PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_diagnostics PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_diagnostics)

# ---- test_accel -----------------------------------------------------------------------
# Phase 6 accel-router parity (accel.hpp): route/run/degrade/health/probe
# plus the shared-DSP-resident retirement machinery via the
# detail::install_dsp_dispatch() seam (detail/dsp_dispatch.hpp is pb-free,
# so only the src/ include path is needed).
add_executable(test_accel test_accel.cpp)
target_include_directories(test_accel PRIVATE "${NE503_SRC_DIR}")
target_link_libraries(test_accel PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_accel PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_accel)

# ---- test_preprocess -------------------------------------------------------------------
# Phase 6 preprocess parity: ctor ladder, geometry/meta, from_model via an
# in-process fake InferenceService (ai-runtime/inference.grpc.pb.h needs
# the proto-gen include dir) and the nv12_passthrough device leg via the
# detail::install_dsp_dispatch() seam.
add_executable(test_preprocess test_preprocess.cpp)
target_include_directories(test_preprocess PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_preprocess PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_preprocess PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_preprocess)

# ---- test_postprocess ------------------------------------------------------------------
# Phase 6 NMS parity (postprocess.hpp public header): greedy suppression,
# class separation, validation ladder, ported from python
# tests/test_postprocess.py.
add_executable(test_postprocess test_postprocess.cpp)
target_link_libraries(test_postprocess PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_postprocess PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_postprocess)

# ---- test_yolo_decode --------------------------------------------------------------------
# Phase 6 YOLO decode adapter parity (postprocess.hpp public header):
# V8/V5 export-shape decoding, meta mapping and pad clipping, ported from
# python tests/test_yolo_decode.py.
add_executable(test_yolo_decode test_yolo_decode.cpp)
target_link_libraries(test_yolo_decode PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_yolo_decode PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_yolo_decode)

# ---- test_pipeline -----------------------------------------------------------------------
# Phase 6 pipeline composition parity: run/run_async/from_model against an
# in-process fake aipc.inference gRPC service (python's FakeClient duck
# type), plus the passthrough ref settle/retain lifecycle via the
# detail::install_dsp_dispatch() borrowed-ref seam. ai-runtime/
# inference.grpc.pb.h needs the proto-gen include dir, dsp_dispatch.hpp the
# src/ one (same shape as test_preprocess).
add_executable(test_pipeline test_pipeline.cpp)
target_include_directories(test_pipeline PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_pipeline PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_pipeline PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_pipeline)

# ---- test_runner ------------------------------------------------------------------------
# Phase 6 PipelineRunner parity (runner.hpp public header): latest-wins
# queue, graceful stop, error policy and the bounded sink-less capture,
# ported from python tests/test_runner.py via FrameHandle release counters.
add_executable(test_runner test_runner.cpp)
target_link_libraries(test_runner PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_runner PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_runner)

# ---- test_stream_pipeline -----------------------------------------------------------------
# Phase 6 StreamPipeline parity: whole-pipeline facade against three
# in-process fakes on one server — aipc.inference (scripted server-streaming
# StreamInfer), aipc.event (Publish recorder) and aipc.camera
# (UpdateAiOverlay recorder) — ported from python
# tests/test_p211_stream_pipeline.py. The generated inference/event/camera
# grpc pb headers need the proto-gen include dir.
add_executable(test_stream_pipeline test_stream_pipeline.cpp)
target_include_directories(test_stream_pipeline PRIVATE
    "${NE503_SRC_DIR}" "${NE503_PROTO_GEN_DIR}")
target_link_libraries(test_stream_pipeline PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_stream_pipeline PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_stream_pipeline)

set_target_properties(test_config test_inference test_wire_protocol test_audio_frame
    test_channel_options test_frame_geometry test_fd_media_lifecycle test_encoded_v3
    test_camera_codec test_device_af test_overlay_strict test_overlay_annotate
    test_inference_codec test_subscribe_health test_dsp_wire test_dsp_fallback
    test_dsp_lifecycle test_injection_slots
    test_color test_draw test_diagnostics test_accel test_preprocess
    test_postprocess test_yolo_decode test_pipeline test_runner test_stream_pipeline
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)

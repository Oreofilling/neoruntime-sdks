# examples/CMakeLists.txt — 1:1 ports of python/examples/*.py.
#
# Each example is a standalone aarch64 binary linked against the SDK static lib.
# They run on-device inside an app container; they are not built as tests.

set(NE503_EXAMPLES
    connectivity_smoke   # read-only gRPC round-trip smoke (safe on live hardware)
    phase3_smoke         # Phase 3 domains on-device: imaging/IR/status + inject wire + AF + overlay strict
    phase4_smoke         # Phase 4 inference on-device: register/infer variants/subscribe health/CPU gate
    phase5_smoke         # Phase 5 DSP+injection on-device: alloc/resize/convert/blend/jpeg + FramePublisher e2e
    phase6_smoke         # Phase 6 tool layer on-device: StreamPipeline e2e + bus-side overlay observer
    media_smoke          # raw-UDS media path (encoded H.264/H.265 + zero-copy DMA-BUF)
    api_tour             # every client module, one core call each, PASS/SKIP/FAIL
    dsp_resize           # DSP domain tour: alloc/resize/convert/encode-jpeg + fallback visibility
    frame_injection      # FramePublisher on 'sub': nv12 publish + publish_rgb, clean EOS
    stream_pipeline      # StreamPipeline: zoned subscribe -> overlay -> result loop
    person_detection
    event_subscriber
    perimeter_guard
    video_processor)

# Examples that exercise the raw video path and therefore need OpenCV's
# imgcodecs/imgproc (imwrite / cvtColor) directly in addition to the SDK link.
set(NE503_EXAMPLES_NEED_OPENCV video_processor)

foreach(example ${NE503_EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE ne503::aipc_sdk)

    if(example IN_LIST NE503_EXAMPLES_NEED_OPENCV)
        target_link_libraries(${example} PRIVATE
            opencv_imgcodecs opencv_imgproc)
    endif()

    # The device's runtime libstdc++ (Yocto) lags the cross toolchain's
    # (Ubuntu: GLIBCXX_3.4.30); pin the C++ runtime into the binary so
    # examples run on either. Host builds keep dynamic linking.
    if(CMAKE_CROSSCOMPILING)
        target_link_options(${example} PRIVATE -static-libstdc++ -static-libgcc)
    endif()

    set_target_properties(${example} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
endforeach()

# Install the example binaries (useful when packaging an SDK image).
install(TARGETS ${NE503_EXAMPLES} RUNTIME DESTINATION bin)

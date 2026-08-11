# GenerateProtos.cmake
# -----------------------------------------------------------------------------
# Generates C++ protobuf message sources (.pb.{h,cc}) AND gRPC service stubs
# (.grpc.pb.{h,cc}) from the shared repo-root proto/ directory into a build-time
# output dir. This is the C++ analogue of scripts/sync_platform_protos.sh's
# `grpc_tools.protoc` step. Output is gitignored and regenerated every build;
# the shared proto/*.proto files remain the single source of truth.
#
# Usage:
#   include(GenerateProtos.cmake)
#   ne503_generate_protos(
#       PROTO_ROOT <abs path to proto dir>
#       OUTPUT_DIR <abs path to gen output>
#       PROTOC     <protoc executable or generator expression>
#       PLUGIN     <grpc_cpp_plugin executable or generator expression>
#       PROTOS     <relative proto paths, e.g. "device-control/device.proto">)
#   # -> ${ne503_proto_generated_srcs} is set in the caller's scope (PARENT_SCOPE).
# -----------------------------------------------------------------------------

function(ne503_generate_protos)
    set(options)
    set(oneValueArgs PROTO_ROOT OUTPUT_DIR PROTOC PLUGIN)
    set(multiValueArgs PROTOS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_PROTO_ROOT OR NOT ARG_OUTPUT_DIR OR NOT ARG_PROTOC OR NOT ARG_PLUGIN)
        message(FATAL_ERROR "ne503_generate_protos: PROTO_ROOT/OUTPUT_DIR/PROTOC/PLUGIN are required")
    endif()

    # protoc < 3.15 rejects proto3 `optional` fields unless passed
    # --experimental_allow_proto3_optional (default ever since). The shared
    # proto/*.proto (e.g. camera-daemon/camera.proto) uses optional fields, so an
    # older system protoc (3.12.x from apt) needs the flag. Detect the protoc
    # version once when the binary is a plain path; a generator-expression protoc
    # (vcpkg config-mode) is always modern, so skip detection there.
    if(DEFINED NE503_PROTOC_EXTRA_FLAGS)
        set(_proto_extra ${NE503_PROTOC_EXTRA_FLAGS})
    elseif(NOT "${ARG_PROTOC}" MATCHES [[\$<]])
        execute_process(
            COMMAND "${ARG_PROTOC}" --version
            OUTPUT_VARIABLE _pv OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_pv MATCHES "libprotoc ([0-9]+)\\.([0-9]+)")
            if((CMAKE_MATCH_1 EQUAL 3 AND CMAKE_MATCH_2 LESS 15) OR CMAKE_MATCH_1 LESS 3)
                set(_proto_extra "--experimental_allow_proto3_optional")
            else()
                set(_proto_extra)
            endif()
        else()
            set(_proto_extra)
        endif()
    else()
        set(_proto_extra)
    endif()

    set(_all_generated)
    set(_all_outputs)
    foreach(proto ${ARG_PROTOS})
        get_filename_component(_dir "${proto}" DIRECTORY)
        get_filename_component(_name "${proto}" NAME_WE)

        if(_dir)
            set(_out_sub "${ARG_OUTPUT_DIR}/${_dir}")
        else()
            set(_out_sub "${ARG_OUTPUT_DIR}")
        endif()
        file(MAKE_DIRECTORY "${_out_sub}")

        set(_pb_h    "${_out_sub}/${_name}.pb.h")
        set(_pb_cc   "${_out_sub}/${_name}.pb.cc")
        set(_grpc_h  "${_out_sub}/${_name}.grpc.pb.h")
        set(_grpc_cc "${_out_sub}/${_name}.grpc.pb.cc")

        # A single protoc invocation emits both message + gRPC sources via the
        # grpc plugin. -I<PROTO_ROOT> makes import paths resolve as they do in
        # the Python generator (e.g. "import camera-daemon/lens_hal.proto").
        add_custom_command(
            OUTPUT  "${_pb_cc}" "${_pb_h}" "${_grpc_cc}" "${_grpc_h}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${_out_sub}"
            COMMAND "${ARG_PROTOC}"
                    ${_proto_extra}
                    "--cpp_out=${ARG_OUTPUT_DIR}"
                    "--grpc_out=${ARG_OUTPUT_DIR}"
                    "--plugin=protoc-gen-grpc=${ARG_PLUGIN}"
                    "-I${ARG_PROTO_ROOT}"
                    "${ARG_PROTO_ROOT}/${proto}"
            DEPENDS "${ARG_PROTO_ROOT}/${proto}"
            COMMENT "protoc (cpp+grpc) -> ${proto}"
            VERBATIM)

        list(APPEND _all_generated "${_pb_cc}" "${_grpc_cc}")
        list(APPEND _all_outputs   "${_pb_cc}" "${_pb_h}" "${_grpc_cc}" "${_grpc_h}")
    endforeach()

    # Custom target guarantees the codegen runs even when the static lib has no
    # hand-written sources yet (early scaffolding), and lets other targets
    # depend on a single name.
    if(NOT TARGET ne503_proto_gen)
        add_custom_command(
            OUTPUT  "${ARG_OUTPUT_DIR}/.ne503_proto_stamp"
            COMMAND "${CMAKE_COMMAND}" -E touch "${ARG_OUTPUT_DIR}/.ne503_proto_stamp"
            DEPENDS ${_all_outputs}
            COMMENT "NE503 proto codegen complete")
        add_custom_target(ne503_proto_gen ALL
            SOURCES ${_all_outputs}
            DEPENDS "${ARG_OUTPUT_DIR}/.ne503_proto_stamp")
    endif()

    set(ne503_proto_generated_srcs ${_all_generated} PARENT_SCOPE)
endfunction()

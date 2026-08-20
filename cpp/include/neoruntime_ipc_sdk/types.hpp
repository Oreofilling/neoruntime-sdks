// types.hpp — cross-module value types shared by multiple clients.
//
// Shared enums/structs that more than one module needs live here. Where a value
// mirrors a protobuf enum (e.g. DataType below), this header owns a decoupled
// copy so the public headers never need to include generated *.pb.h; the .cpp
// layer translates between the two.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace neoruntime_ipc_sdk {

// Pixel formats used by the video media path. Mirrors
// python/neoruntime_ipc_sdk/media.py PixelFormat exactly (do not renumber).
enum class PixelFormat : int {
    NV12  = 0,
    NV21  = 1,
    RGB   = 2,
    BGR   = 3,
    RGBA  = 4,
    BGRA  = 5,
    GRAY8 = 6,
    YUYV  = 7,
};

inline constexpr const char* pixel_format_name(PixelFormat fmt) noexcept {
    switch (fmt) {
        case PixelFormat::NV12:  return "NV12";
        case PixelFormat::NV21:  return "NV21";
        case PixelFormat::RGB:   return "RGB";
        case PixelFormat::BGR:   return "BGR";
        case PixelFormat::RGBA:  return "RGBA";
        case PixelFormat::BGRA:  return "BGRA";
        case PixelFormat::GRAY8: return "GRAY8";
        case PixelFormat::YUYV:  return "YUYV";
    }
    return "UNKNOWN";
}

// Axis-aligned box in normalized [0,1] image coordinates, stored as the
// top-left corner plus width/height. Mirrors inference.py::BoundingBox.
struct BoundingBox {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;

    // (x1, y1, x2, y2) top-left / bottom-right.
    constexpr std::array<float, 4> to_xyxy() const noexcept {
        return {x, y, x + width, y + height};
    }
    // (x, y, width, height) — identity accessor for API parity with Python.
    constexpr std::array<float, 4> to_xywh() const noexcept {
        return {x, y, width, height};
    }
};

// Tensor element type. Values match aipc.inference.DataType (do not renumber);
// kept here as a plain enum so the public inference header stays proto-free.
enum class DataType : int {
    Uint8   = 0,
    Int8    = 1,
    Uint16  = 2,
    Int16   = 3,
    Float16 = 4,
    Float32 = 5,
    Int32   = 6,
    Uint32  = 7,
};

inline constexpr const char* data_type_name(DataType dt) noexcept {
    switch (dt) {
        case DataType::Uint8:   return "uint8";
        case DataType::Int8:    return "int8";
        case DataType::Uint16:  return "uint16";
        case DataType::Int16:   return "int16";
        case DataType::Float16: return "float16";
        case DataType::Float32: return "float32";
        case DataType::Int32:   return "int32";
        case DataType::Uint32:  return "uint32";
    }
    return "unknown";
}

}  // namespace neoruntime_ipc_sdk

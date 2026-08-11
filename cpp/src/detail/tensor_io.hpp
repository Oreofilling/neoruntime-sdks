// tensor_io.hpp — cv::Mat <-> aipc::inference::Tensor conversion.
//
// INTERNAL helper (lives under the PRIVATE src include path). It includes the
// generated proto header, so it must never be pulled into a public header.
//
// Faithfulness to the Python SDK (inference.py::_numpy_to_tensor /
// _tensor_to_numpy):
//   * mat_to_tensor emits shape {H,W} (single-channel) or {H,W,C} — matching
//     numpy ndarray.shape for cv2 images — and the C-contiguous byte payload.
//     Non-continuous Mats are cloned first so the bytes are truly row-major.
//   * tensor_to_mat reconstructs an image-shaped Mat for 2D/3D (HWC) tensors;
//     anything else (1D, 4D NCHW, or a shape/data-size mismatch) collapses to
//     a flat 1xN Mat, mirroring numpy's reshape() fallback to a flat array.
//
// dtype caveat: OpenCV has no unsigned 32-bit depth. UINT32 tensors map to
// CV_32S (same 4-byte layout); callers that need true unsigned semantics must
// reinterpret. All other dtypes map 1:1.
#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "ai-runtime/inference.pb.h"
#include "hailo_ipc_sdk/types.hpp"

namespace hailo_ipc_sdk::detail {

// ---- public DataType <-> proto DataType (values are identical 0..7) ----------
inline aipc::inference::DataType data_type_to_pb(DataType dt) noexcept {
    return static_cast<aipc::inference::DataType>(static_cast<int>(dt));
}
inline DataType pb_to_data_type(int value) noexcept {
    return static_cast<DataType>(value);
}

// ---- proto DataType <-> OpenCV depth ---------------------------------------
inline int pb_dtype_to_cv_depth(aipc::inference::DataType dt) noexcept {
    switch (dt) {
        case aipc::inference::UINT8:   return CV_8U;
        case aipc::inference::INT8:    return CV_8S;
        case aipc::inference::UINT16:  return CV_16U;
        case aipc::inference::INT16:   return CV_16S;
        case aipc::inference::FLOAT16: return CV_16F;
        case aipc::inference::FLOAT32: return CV_32F;
        case aipc::inference::INT32:   return CV_32S;
        case aipc::inference::UINT32:  return CV_32S;  // no native CV_32U
        default:                       return CV_32F;
    }
}

inline aipc::inference::DataType cv_depth_to_pb(int depth) noexcept {
    switch (depth) {
        case CV_8U:  return aipc::inference::UINT8;
        case CV_8S:  return aipc::inference::INT8;
        case CV_16U: return aipc::inference::UINT16;
        case CV_16S: return aipc::inference::INT16;
        case CV_32S: return aipc::inference::INT32;
        case CV_32F: return aipc::inference::FLOAT32;
        case CV_16F: return aipc::inference::FLOAT16;
        default:     return aipc::inference::FLOAT32;
    }
}

inline std::size_t cv_depth_elem_size(int depth) noexcept {
    switch (depth) {
        case CV_8U:  case CV_8S:                    return 1;
        case CV_16U: case CV_16S: case CV_16F:      return 2;
        case CV_32S: case CV_32F:                   return 4;
        default:                                    return 1;
    }
}

// Lowercase "uint8"/"float32"/... -> DataType (default FLOAT32, matching
// inference.py::_dtype_str_to_enum).
inline DataType data_type_from_str(std::string_view s) {
    auto eq = [s](const char* lit) {
        return s.size() == std::strlen(lit) &&
               std::equal(s.begin(), s.end(), lit,
                          [](char c, char l) { return std::tolower(c) == l; });
    };
    if (eq("uint8"))   return DataType::Uint8;
    if (eq("int8"))    return DataType::Int8;
    if (eq("uint16"))  return DataType::Uint16;
    if (eq("int16"))   return DataType::Int16;
    if (eq("float16")) return DataType::Float16;
    if (eq("int32"))   return DataType::Int32;
    if (eq("uint32"))  return DataType::Uint32;
    return DataType::Float32;
}

// ---- cv::Mat -> Tensor ------------------------------------------------------
inline aipc::inference::Tensor mat_to_tensor(const cv::Mat& mat) {
    // Clone ROIs/non-continuous Mats so the byte payload is C-contiguous.
    const cv::Mat& cont = mat.isContinuous() ? mat
                                              : static_cast<const cv::Mat&>(cv::Mat(mat).clone());
    aipc::inference::Tensor t;
    t.add_shape(cont.rows);
    t.add_shape(cont.cols);
    if (cont.channels() > 1) t.add_shape(cont.channels());
    t.set_dtype(cv_depth_to_pb(cont.depth()));
    t.set_data(cont.data, cont.total() * cont.elemSize());
    return t;
}

// ---- Tensor -> cv::Mat ------------------------------------------------------
inline cv::Mat tensor_to_mat(const aipc::inference::Tensor& tensor) {
    const int depth = pb_dtype_to_cv_depth(tensor.dtype());
    const std::size_t elem = cv_depth_elem_size(depth);
    const auto& shape = tensor.shape();
    const std::string& data = tensor.data();

    std::size_t shape_elems = 1;
    for (int s : shape) shape_elems *= static_cast<std::size_t>(s);
    const std::size_t data_elems = elem ? (data.size() / elem) : 0;

    const auto copy_into = [&](cv::Mat& mat) {
        const std::size_t bytes = std::min(data.size(), mat.total() * mat.elemSize());
        if (bytes) std::memcpy(mat.data, data.data(), bytes);
    };

    // 3D HxWxC with 1..4 channels -> natural image form (channels folded in).
    if (shape.size() == 3 && shape[2] >= 1 && shape[2] <= 4 &&
        shape_elems == data_elems && data_elems > 0) {
        cv::Mat mat(shape[0], shape[1], CV_MAKETYPE(depth, shape[2]));
        copy_into(mat);
        return mat;
    }
    // 2D HxW -> single-channel image.
    if (shape.size() == 2 && shape_elems == data_elems && data_elems > 0) {
        cv::Mat mat(shape[0], shape[1], CV_MAKETYPE(depth, 1));
        copy_into(mat);
        return mat;
    }
    // Otherwise: flat 1xN single-channel (numpy "return flat array" fallback).
    cv::Mat mat(1, static_cast<int>(data_elems), CV_MAKETYPE(depth, 1));
    copy_into(mat);
    return mat;
}

}  // namespace hailo_ipc_sdk::detail

// frame.cpp — Frame color conversions and persistence.
// Port of media.py Frame.to_rgb() / Frame.save().
#include "neoruntime_ipc_sdk/media.hpp"

#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace neoruntime_ipc_sdk {

cv::Mat Frame::to_rgb() const {
    if (image.empty()) {
        throw std::invalid_argument("Frame::to_rgb(): empty image");
    }

    if (format == "RGB") {
        return image.clone();
    }
    if (format == "BGR") {
        cv::Mat out;
        cv::cvtColor(image, out, cv::COLOR_BGR2RGB);
        return out;
    }
    if (format == "NV12") {
        // image is H*3/2 x W single-channel (Y plane + interleaved UV).
        cv::Mat out;
        cv::cvtColor(image, out, cv::COLOR_YUV2RGB_NV12);
        return out;
    }
    if (format == "NV21") {
        cv::Mat out;
        cv::cvtColor(image, out, cv::COLOR_YUV2RGB_NV21);
        return out;
    }
    if (format == "GRAY8") {
        cv::Mat out;
        cv::cvtColor(image, out, cv::COLOR_GRAY2RGB);
        return out;
    }
    if (format == "RGBA") {
        cv::Mat out;
        cv::cvtColor(image, out, cv::COLOR_RGBA2RGB);
        return out;
    }
    if (format == "BGRA") {
        cv::Mat out;
        cv::cvtColor(image, out, cv::COLOR_BGRA2RGB);
        return out;
    }
    if (format == "YUYV") {
        // YUYV is packed 4:2:2: convert to BGR then to RGB.
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_YUV2BGR_YUYV);
        cv::Mat out;
        cv::cvtColor(bgr, out, cv::COLOR_BGR2RGB);
        return out;
    }

    throw std::invalid_argument("Frame::to_rgb(): unsupported format '" + format + "'");
}

void Frame::save(const std::string& path) const {
    cv::Mat rgb = to_rgb();
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(path, bgr)) {
        throw std::runtime_error("Frame::save(): failed to write '" + path + "'");
    }
}

cv::Mat Frame::data() const {
    if (image.empty()) return cv::Mat();
    // A flat, contiguous byte view — preserves the original pixel layout.
    return image.isContinuous() ? image : image.clone();
}

// ---- EncodedFrame ----------------------------------------------------------
std::string EncodedFrame::codec_name() const {
    switch (codec) {
        case 0: return "h264";
        case 1: return "h265";
        default: return "unknown";
    }
}

}  // namespace neoruntime_ipc_sdk

// geometry.hpp — resize geometry shared by Frame::resize (CPU path), the DSP
// fast path (Phase 5) and preprocessing (Python frame._resize_geometry). The
// single source of placement truth: every leg must compute the same content
// box (rw, rh) and placement (ox, oy) or the same frame would place pixels
// differently depending on which leg served it.
#pragma once

#include <algorithm>
#include <cmath>
#include <string_view>

namespace neoruntime_ipc_sdk::detail {

// Round down to the nearest even number, minimum 2 (YUV plane safety).
inline int even_down(int value) { return std::max(2, value - (value % 2)); }

struct ResizeBox {
    int rw = 0;  // content box size
    int rh = 0;
    int ox = 0;  // placement of the content box
    int oy = 0;
};

// Content box and placement for the resize modes:
//   "stretch"    — content fills dst exactly.
//   "letterbox"  — content scaled to (rw, rh) and placed at (ox, oy) inside
//                  the dst canvas.
//   "crop"       — content scaled to cover at (rw, rh); a dst-sized window is
//                  read out of it at (ox, oy).
// `yuv` adjusts the box/offsets to even, chroma-aligned values.
inline ResizeBox resize_geometry(int sw, int sh, int dw, int dh,
                                 std::string_view mode, bool yuv) {
    if (mode == "stretch") return {dw, dh, 0, 0};
    if (mode == "letterbox") {
        double scale = std::min(static_cast<double>(dw) / sw,
                                static_cast<double>(dh) / sh);
        int rw = std::max(1, static_cast<int>(std::lround(sw * scale)));
        int rh = std::max(1, static_cast<int>(std::lround(sh * scale)));
        if (yuv) {
            rw = even_down(rw);
            rh = even_down(rh);
        }
        int ox = (dw - rw) / 2;
        int oy = (dh - rh) / 2;
        if (yuv) {
            ox &= ~1;  // chroma-aligned pad offsets
            oy &= ~1;
        }
        return {rw, rh, ox, oy};
    }
    // crop: scale to cover, read a dst-sized window out of the middle
    double scale = std::max(static_cast<double>(dw) / sw,
                            static_cast<double>(dh) / sh);
    int rw = std::max(dw, static_cast<int>(std::lround(sw * scale)));
    int rh = std::max(dh, static_cast<int>(std::lround(sh * scale)));
    if (yuv) {
        rw = even_down(rw);
        rh = even_down(rh);
    }
    int ox = (rw - dw) / 2;
    int oy = (rh - dh) / 2;
    return {rw, rh, ox, oy};
}

}  // namespace neoruntime_ipc_sdk::detail

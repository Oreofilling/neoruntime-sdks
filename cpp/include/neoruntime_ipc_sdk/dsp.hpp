// dsp.hpp — hardware DSP surface: resize/crop/multi-crop/convert/blend on
// the camera-daemon DSP service, plus JPEG encoding (port of python
// dsp.py, 0.8.1).
//
// Two transports, one client (see src/detail/dsp_wire.hpp for the wire):
//   * gRPC job plane   — camera-control.sock: SubmitDspJob / Async /
//     Wait / EncodeImage
//   * UDS buffer plane — camera.sock: ALLOC / IMPORT / BUF_RELEASE with
//     SCM_RIGHTS fd passing
//
// Sources for the *_hw jobs arrive as DspJobSource: a cv::Mat (copied
// in), a keep-fd Frame/FrameHandle (dma-bufs imported zero-copy), or a
// DspBufferRef (its daemon buffer used directly — borrowed). When the
// daemon lacks the DSP surface, array-source calls fall back to the CPU
// legs (dsp_format) with a stderr warning and last_used_hw() = false;
// zero-copy sources refuse the silent fallback instead, exactly like
// python.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "frame.hpp"

namespace neoruntime_ipc_sdk {

class DspClient;
class DspBufferPool;
class SharedImportPool;
class PendingDspJob;
class DspBufferRef;

// DSP job or buffer error; `code` mirrors the daemon error codes -1..-7
// (see detail/dsp_wire.hpp) when one rode the response.
class DspError : public std::runtime_error {
public:
    explicit DspError(const std::string& message,
                      std::optional<std::int32_t> code = std::nullopt)
        : std::runtime_error(message), code_(code) {}

    std::optional<std::int32_t> code() const { return code_; }

private:
    std::optional<std::int32_t> code_;
};

// What a *_hw call takes as its pixel source (python JobSource).
using DspJobSource =
    std::variant<cv::Mat, Frame, std::shared_ptr<FrameHandle>, DspBufferRef*>;

// One blend overlay: an (h, w, 4) uint8 RGBA Mat plus its 1:1 position on
// the NV12 base.
struct DspOverlay {
    cv::Mat rgba;
    int x = 0;
    int y = 0;
};

// One multi-crop window (python's (x, y, w, h, dst_w, dst_h) tuple).
struct DspWindow {
    int x = 0, y = 0, width = 0, height = 0, dst_width = 0, dst_height = 0;
};

// Shared knobs of the *_hw job methods (python keyword tail).
// `out_ref` is python's out="ref": the sync no-read-back form returning a
// DspBufferRef. `wait=false` submits without blocking and returns a
// PendingDspJob instead.
struct DspJobParams {
    std::string interpolation = "bilinear";
    std::string scaling = "stretch";
    std::string priority = "normal";
    double timeout_s = 5.0;
    bool cpu_fallback = true;
    bool wait = true;
    bool out_ref = false;
};

// blend_hw extras: caller-owned destination slot (out=pool — the blend
// lands in dst_pool->write(dst_slot) territory and the call returns
// monostate) and the device-side-base escape hatch.
struct DspBlendParams : DspJobParams {
    bool zero_copy = false;
    DspBufferPool* dst_pool = nullptr;
    int dst_slot = 0;
};

// ---- DspBufferPool -----------------------------------------------------------
// Daemon-allocated dma-buf buffers sharing one geometry (one wire ALLOC).
// Every buffer exposes plane_count(fmt) dma-buf fds; plane rows may be
// padded (strides > row bytes) — write/read copy row-by-row so padding is
// preserved. release() returns the buffers to the daemon and closes every
// fd; the destructor releases too, and the client's close() poisons every
// still-live pool (the daemon reclaims buffers on UDS disconnect).
class DspBufferPool {
public:
    ~DspBufferPool();
    DspBufferPool(DspBufferPool&&) noexcept;
    DspBufferPool& operator=(DspBufferPool&&) noexcept;
    DspBufferPool(const DspBufferPool&) = delete;
    DspBufferPool& operator=(const DspBufferPool&) = delete;

    int count() const noexcept { return static_cast<int>(ids.size()); }
    std::uint64_t buffer_id(int index) const;

    // Copy `arr` into buffer `index`. nv12: (h*3/2, w) 8UC1; rgb24:
    // (h, w, 3) 8UC3; argb: (h, w, 4) 8UC4 (wire byte order [A,R,G,B]);
    // gray8: (h, w) 8UC1. Throws DspError on shape/type mismatch.
    void write(int index, const cv::Mat& arr);
    // Read buffer `index` back in the same layout.
    cv::Mat read(int index);

    // Return all buffers to the daemon (idempotent).
    void release() noexcept;
    bool released() const noexcept { return released_; }
    // Internal: poison without wire traffic — the client is closing and
    // the daemon reclaims the ids on disconnect.
    void mark_dead() noexcept;

    int width = 0;
    int height = 0;
    std::string fmt;
    std::vector<std::uint64_t> ids;               // daemon registry ids
    std::array<std::uint32_t, 3> strides{};       // per plane (<= 3 used)
    std::array<std::uint32_t, 3> plane_sizes{};
    std::vector<int> plane_fds;                   // count * plane_count fds

    // Internal: constructed by DspClient::alloc_buffers only.
    DspBufferPool(int w, int h, std::string fmt, std::vector<std::uint64_t> ids,
                  std::vector<int> fds, const std::uint32_t strides[3],
                  const std::uint32_t sizes[3],
                  std::function<void(std::uint64_t)> send_release,
                  std::function<void(DspBufferPool*)> on_gone);

private:
    bool released_ = false;
    // Hooks the client wires in at construction (public ctor: python's
    // pool attributes are public the same way — only DspClient builds one).
    std::function<void(std::uint64_t)> send_release_;  // daemon wire
    std::function<void(DspBufferPool*)> on_gone_;      // registry cleanup
};

// ---- SharedImportPool ----------------------------------------------------------
// A persistent round-robin ring of shared ARGB32 memfd imports the daemon
// blends straight from (some deployed HALs refuse ARGB32 dma-buf pool
// allocation, but the bake path maps memfd USERPTR planes fine — both
// processes share the same pages, so write() lands in exactly the memory
// the daemon reads). Slots recycle round-robin; do not rewrite a slot
// while its frame may still sit in a daemon-side queue (queue cap 3, so a
// ring depth >= 4 paces faster-than-bake publishing).
class SharedImportPool {
public:
    ~SharedImportPool();
    SharedImportPool(SharedImportPool&&) noexcept;
    SharedImportPool& operator=(SharedImportPool&&) noexcept;
    SharedImportPool(const SharedImportPool&) = delete;
    SharedImportPool& operator=(const SharedImportPool&) = delete;

    int count() const noexcept { return static_cast<int>(ids.size()); }
    // ARGB32 is one width*4 plane.
    std::array<std::uint32_t, 1> strides() const noexcept {
        return {static_cast<std::uint32_t>(width * 4)};
    }
    std::uint64_t buffer_id(int index) const;

    // Copy one (h, w, 4) uint8 ARGB frame into slot `index`'s shared pages.
    void write(int index, const cv::Mat& arr);

    // Return every import and close the shared pages (idempotent).
    void release() noexcept;
    bool released() const noexcept { return released_; }

    int width = 0;
    int height = 0;
    std::string fmt = "argb";
    std::vector<std::uint64_t> ids;

    // Internal: constructed by DspClient::import_shared_buffers only.
    SharedImportPool(int w, int h, std::vector<std::uint64_t> ids,
                     std::vector<void*> maps, std::vector<std::size_t> map_sizes,
                     std::vector<int> fds,
                     std::function<void(std::uint64_t)> send_release);

private:
    bool released_ = false;
    std::vector<void*> maps_;
    std::vector<std::size_t> map_sizes_;
    std::vector<int> fds_;
    std::function<void(std::uint64_t)> send_release_;
};

// ---- PendingDspJob --------------------------------------------------------------
// The wait=false return of the *_hw job methods: one submitted job plus
// every buffer it needs, owned until consumed.
//   wait()       — block, read the destination, release everything.
//   wait_result() — block but keep the result device-side; chain
//                  buffer_id() into encode_jpeg_hw for a zero read-back
//                  encode, then release().
//   done()       — poll; timeout_s=0 is a pure non-blocking check.
//   release()    — drop the result and free the buffers (idempotent; a
//                  still-pending job is reaped with one bounded wait).
class PendingDspJob {
public:
    ~PendingDspJob();
    PendingDspJob(PendingDspJob&&) noexcept;
    PendingDspJob& operator=(PendingDspJob&&) noexcept;
    PendingDspJob(const PendingDspJob&) = delete;
    PendingDspJob& operator=(const PendingDspJob&) = delete;

    // Daemon id of the (first) destination buffer.
    std::uint64_t buffer_id() const;
    // Poll for completion without consuming the result. A job that failed
    // still counts as done — the error surfaces from wait()/wait_result().
    bool done(double timeout_s = 0.0);
    // Block until the job completes, the result staying device-side. A
    // timed-out job throws but stays pending in the daemon — wait again
    // with a longer timeout. A failed job throws its error.
    PendingDspJob& wait_result(std::optional<double> timeout_s = std::nullopt);
    // Block, read the single destination, release the buffers.
    cv::Mat wait(std::optional<double> timeout_s = std::nullopt);
    // Block, read every destination (multi-crop), release the buffers.
    std::vector<cv::Mat> wait_all(std::optional<double> timeout_s = std::nullopt);
    void release() noexcept;

    // Daemon job id (empty when the daemon lacked the async rpc and the
    // sync fallback already executed the job — born done).
    std::optional<std::uint64_t> job_id() const;

    struct Impl;
    // Internal: built by the *_hw methods (defined in dsp.cpp — an inline
    // ctor would instantiate unique_ptr's deleter on the incomplete Impl in
    // every including TU).
    explicit PendingDspJob(std::unique_ptr<Impl> impl);

private:
    std::unique_ptr<Impl> impl_;
};

// ---- DspBufferRef ---------------------------------------------------------------
// The out_ref=true return of the sync *_hw job methods: one completed
// job's destination buffer, device-side, plus everything the call owned
// to produce it. Chain buffer_id() into another *_hw call or
// encode_jpeg_hw without touching the pixels; read() brings them back on
// demand; release() returns the owned buffers (idempotent — a
// caller-supplied dst_pool is never owned, but the ref is dead either
// way). A ref is also a valid job source (borrowed: keep it alive until
// the consuming job completes, then release it).
class DspBufferRef {
public:
    ~DspBufferRef();
    DspBufferRef(DspBufferRef&&) noexcept;
    DspBufferRef& operator=(DspBufferRef&&) noexcept;
    DspBufferRef(const DspBufferRef&) = delete;
    DspBufferRef& operator=(const DspBufferRef&) = delete;

    std::uint64_t buffer_id() const;  // throws when released/dead
    int width() const noexcept { return pool_->width; }
    int height() const noexcept { return pool_->height; }
    const std::string& fmt() const noexcept { return pool_->fmt; }
    // A moved-from ref counts as released (its impl is gone).
    bool released() const noexcept { return released_ || impl_ == nullptr; }

    cv::Mat read();      // throws when released/dead
    void release() noexcept;

    struct Impl;
    // Internal: built by the *_hw methods (out-of-line for the same
    // incomplete-type reason as PendingDspJob's). Note the unique_ptr<Impl>
    // parameter is destroyed in the CALLER's TU, so this ctor is only
    // callable from inside the SDK build — outsiders use detached().
    DspBufferRef(std::unique_ptr<Impl> impl, DspBufferPool* pool);

    // A ref with no backing buffer (impl/pool null): released() is true,
    // accessors throw, destructor is a no-op. For tests and edge callers
    // that exercise ref-carrying surfaces without a live pool.
    static DspBufferRef detached();

    // A ref bound to a caller-owned pool but owning no buffers (empty impl):
    // released() is false, buffer_id()/width()/height()/fmt() read the pool,
    // and release() is a safe no-op. For tests and zero-copy dry runs that
    // exercise ref-carrying surfaces with a synthetic pool — the pipeline's
    // passthrough settle path, e.g. (python monkeypatches _prepare; C++
    // injects one of these through the dsp-dispatch seam).
    static DspBufferRef borrowed(DspBufferPool* pool);

private:
    std::unique_ptr<Impl> impl_;
    DspBufferPool* pool_ = nullptr;  // borrowed; Impl keeps it alive
    bool released_ = false;
};

// ---- DspClient --------------------------------------------------------------------
class DspClient {
public:
    // Empty sock_path = env CAMERA_SOCK_PATH or /run/aipc/camera.sock;
    // empty endpoint = the camera-control gRPC endpoint from Config.
    explicit DspClient(std::string sock_path = "", std::string endpoint = "");
    ~DspClient();
    DspClient(const DspClient&) = delete;
    DspClient& operator=(const DspClient&) = delete;

    void connect();
    // Close both transports; the daemon releases our DSP buffers. Every
    // still-live pool is poisoned first so later use fails client-side.
    void close() noexcept;
    bool connected() const noexcept;

    // Null until the first *_hw call; then true = hardware, false = CPU
    // fallback (health reporting stays truthful about the path used).
    std::optional<bool> last_used_hw() const noexcept;

    // True while a pool this client allocated is still usable.
    bool has_live_pools() const noexcept;

    // Allocate `count` daemon-side DSP buffers of one geometry.
    std::unique_ptr<DspBufferPool> alloc_buffers(int width, int height,
                                                 const std::string& fmt = "nv12",
                                                 int count = 1);
    // Import `count` shared ARGB32 memfd buffers as a writable ring.
    std::unique_ptr<SharedImportPool> import_shared_buffers(
        int width, int height, const std::string& fmt = "argb", int count = 2,
        double timeout_s = 5.0);
    // Import a frame's dma-bufs into the daemon registry (DSP_IMPORT) —
    // the zero-copy entry point for inference and jobs. The daemon dups
    // the fds, so the id outlives the handle; free it with
    // release_buffer().
    std::uint64_t import_frame(FrameHandle& handle);
    // Free a registry id (import id or pool buffer).
    void release_buffer(std::uint64_t buffer_id);

    // Scale src to (width, height) on the DSP.
    std::variant<cv::Mat, PendingDspJob, DspBufferRef> resize_hw(
        const DspJobSource& src, int width, int height, std::string fmt = "",
        const DspJobParams& params = {}, DspBufferPool* src_pool = nullptr,
        DspBufferPool* dst_pool = nullptr);

    // Crop (x, y, width, height) and scale to the destination size
    // (dst_* default to the crop size).
    std::variant<cv::Mat, PendingDspJob, DspBufferRef> crop_hw(
        const DspJobSource& src, int x, int y, int width, int height,
        int dst_width = -1, int dst_height = -1, std::string fmt = "",
        const DspJobParams& params = {}, DspBufferPool* src_pool = nullptr,
        DspBufferPool* dst_pool = nullptr);

    // Crop/resize many windows in one job. Destination buffers are
    // grouped by geometry (one pool per distinct output size); results
    // come back in rect order.
    std::variant<std::vector<cv::Mat>, PendingDspJob> multi_crop_hw(
        const DspJobSource& src, const std::vector<DspWindow>& rects,
        std::string fmt = "", const DspJobParams& params = {},
        DspBufferPool* src_pool = nullptr,
        const std::vector<DspBufferPool*>& dst_pools = {});

    // Convert src to dst_fmt (nv12/rgb24/gray8), keeping the dimensions.
    // Supported pairs are firmware-dependent (hailo15: rgb24 <-> nv12
    // only); a refused gray8 pair warns once and later pairs go straight
    // to the CPU leg.
    std::variant<cv::Mat, PendingDspJob, DspBufferRef> convert_hw(
        const DspJobSource& src, const std::string& dst_fmt, std::string fmt = "",
        const DspJobParams& params = {}, DspBufferPool* src_pool = nullptr,
        DspBufferPool* dst_pool = nullptr);

    // Composite ARGB32 overlays onto an NV12 base on the DSP, in place on
    // a daemon pool copy (the input is never modified). Overlays smaller
    // than 16x16 are padded with transparent pixels to 16; odd overlay
    // heights gain one transparent bottom row (device-ruled). Wire order
    // is ARGB32; the RGBA -> ARGB pack is internal. Device-side bases are
    // refused unless zero_copy=true (the import->resize->blend chain has
    // wedged the DSP device-wide in the field; the array path is the
    // proven one). dst_pool (DspBlendParams) blends straight into the
    // caller's slot and returns monostate.
    std::variant<cv::Mat, PendingDspJob, DspBufferRef, std::monostate> blend_hw(
        const DspJobSource& base, const std::vector<DspOverlay>& overlays,
        std::string fmt = "", const DspBlendParams& params = {});

    // Encode src as one JPEG frame on the daemon (EncodeImage rpc — no
    // destination buffer; the bytes ride the response). gray8 arrays
    // up-convert to rgb24 client-side; gray8 frames raise (accept the
    // copy yourself with frame.to_array()). src_buffer_id != 0 encodes
    // straight from a daemon-side buffer — no CPU fallback on that leg.
    // Pass src = nullptr together with src_buffer_id.
    std::vector<std::uint8_t> encode_jpeg_hw(const DspJobSource* src, int quality = 85,
                                             std::string fmt = "", double timeout_s = 5.0,
                                             DspBufferPool* src_pool = nullptr,
                                             bool cpu_fallback = true,
                                             std::uint64_t src_buffer_id = 0);

    struct Impl;
    // Their Impls reach impl_ for the WaitDspJob rpc (nested classes of a
    // friend share the friendship).
    friend class PendingDspJob;
    friend class DspBufferRef;
    std::unique_ptr<Impl> impl_;
};

}  // namespace neoruntime_ipc_sdk

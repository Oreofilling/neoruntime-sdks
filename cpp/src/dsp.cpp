// dsp.cpp — see include/neoruntime_ipc_sdk/dsp.hpp (port of python dsp.py).
//
// Layout: the DspClient pimpl owns both transports (gRPC job plane via
// GrpcClientBase, raw UDS buffer plane), the pool/ref/pending-job types own
// the buffers their producing call allocated, and the *_hw methods tie them
// together with the python CPU-fallback contract (warn + compute for array
// sources, refuse for zero-copy sources).
#include "neoruntime_ipc_sdk/dsp.hpp"

#include <opencv2/imgcodecs.hpp>

#include <grpcpp/grpcpp.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "neoruntime_ipc_sdk/config.hpp"

#include "camera-daemon/camera.grpc.pb.h"
#include "camera-daemon/camera.pb.h"
#include "detail/dsp_format.hpp"
#include "detail/dsp_wire.hpp"
#include "detail/raw_socket.hpp"
#include "detail/transport.hpp"

namespace neoruntime_ipc_sdk {

namespace pb = aipc::camera;
namespace wire = detail::dsp_wire;
namespace fmtx = detail::dsp_format;

namespace {

// DMA_BUF_IOCTL_SYNC (same constants as media/frame.cpp): _IOW('b', 0, u64).
constexpr unsigned long kDmaBufIoctlSync = 0x40086200;
constexpr std::uint64_t kSyncRead = 1 << 0;
constexpr std::uint64_t kSyncWrite = 1 << 1;
constexpr std::uint64_t kSyncStart = 1 << 2;
constexpr std::uint64_t kSyncEnd = 2 << 2;

void dma_sync(int fd, std::uint64_t flags) noexcept { ::ioctl(fd, kDmaBufIoctlSync, &flags); }

void warn(const std::string& message) {
    std::fprintf(stderr, "neoruntime_ipc_sdk UserWarning: %s\n", message.c_str());
}

// python _DspUnavailable: the daemon lacks the DSP surface (UNIMPLEMENTED or
// service down). Caught by the *_hw tails to engage the CPU fallback.
struct DspUnavailable : DspError {
    explicit DspUnavailable(const std::string& m) : DspError(m) {}
};

// One complete UDS message (python _recv_one_msg). Every byte — the header
// included — must come from recvmsg: on a stream socket SCM_RIGHTS rides
// with the first byte of the sender's sendmsg, and a plain recv consuming
// that byte silently drops the ancillary record. Received fds are the
// caller's to close.
struct UdsMsg {
    std::uint32_t type = 0;
    std::vector<std::uint8_t> bytes;
    std::vector<int> fds;
};

UdsMsg recv_one_msg(int fd, bool* would_block = nullptr) {
    if (would_block) *would_block = false;
    UdsMsg m;
    std::uint8_t hdr[8];
    std::size_t got = 0;
    while (got < 8) {
        const std::size_t fds_before = m.fds.size();
        ssize_t n = detail::recvmsg_with_fds(fd, hdr + got, 8 - got, m.fds,
                                             static_cast<int>(wire::kDspMaxFds));
        if (n == 0 && m.fds.size() == fds_before)
            throw DspError("camera socket closed waiting for a message");
        if (n < 0) {
            // EAGAIN with a would_block flag is a reported timeout, not an
            // error: return the empty message so the caller's flag check
            // replaces the generic socket error with the real diagnostic
            // (e.g. the pre-DSP_IMPORT daemon hint in import_planes).
            if (would_block && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                *would_block = true;
                return m;
            }
            throw DspError("camera socket error waiting for a message");
        }
        got += static_cast<std::size_t>(n);
    }
    const std::uint32_t mtype = wire::get_u32(hdr);
    const std::uint32_t msize = wire::get_u32(hdr + 4);
    if (msize < 8 || msize > (1u << 20))
        throw DspError("corrupt camera-sock header: type=" + std::to_string(mtype) +
                       " size=" + std::to_string(msize));
    m.bytes.assign(hdr, hdr + 8);
    m.bytes.resize(msize);
    std::size_t body = 8;
    while (body < msize) {
        const std::size_t fds_before = m.fds.size();
        ssize_t n = detail::recvmsg_with_fds(fd, m.bytes.data() + body, msize - body, m.fds,
                                             static_cast<int>(wire::kDspMaxFds));
        if (n == 0 && m.fds.size() == fds_before)
            throw DspError("camera socket closed mid-message");
        if (n < 0) {
            if (would_block && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                *would_block = true;
                return m;  // reported timeout, see the header loop
            }
            throw DspError("camera socket error mid-message");
        }
        body += static_cast<std::size_t>(n);
    }
    m.type = mtype;
    return m;
}

// sendmsg with SCM_RIGHTS fds attached (python's sendmsg([payload], anc)).
bool sendmsg_with_fds(int fd, const void* buf, std::size_t len, const std::vector<int>& fds) {
    iovec iov;
    iov.iov_base = const_cast<void*>(buf);
    iov.iov_len = len;
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    std::vector<char> cbuf(CMSG_SPACE(fds.size() * sizeof(int)));
    msg.msg_control = cbuf.data();
    msg.msg_controllen = cbuf.size();
    cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(fds.size() * sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), fds.data(), fds.size() * sizeof(int));
    while (::sendmsg(fd, &msg, 0) < 0) {
        if (errno == EINTR) continue;
        return false;
    }
    return true;
}

// Stride-respecting row copy between mapped plane memory and a packed buffer
// (python DspBufferPool._copy_rows).
void copy_rows(std::uint8_t* mem, std::uint32_t stride, std::uint8_t* packed,
               std::uint32_t row_bytes, std::uint32_t rows, bool to_mem) {
    if (stride == row_bytes) {  // fast path: tightly packed
        std::memcpy(to_mem ? mem : packed, to_mem ? packed : mem,
                    static_cast<std::size_t>(rows) * row_bytes);
        return;
    }
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::size_t off = static_cast<std::size_t>(r) * stride;
        const std::size_t src = static_cast<std::size_t>(r) * row_bytes;
        if (to_mem)
            std::memcpy(mem + off, packed + src, row_bytes);
        else
            std::memcpy(packed + src, mem + off, row_bytes);
    }
}

// Mapped RAII guard for one dma-buf/memfd plane.
struct Mapped {
    void* addr = MAP_FAILED;
    std::size_t size = 0;

    Mapped(int fd, std::size_t size) : size(size) {
        addr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }
    ~Mapped() {
        if (addr != MAP_FAILED) ::munmap(addr, size);
    }
    Mapped(const Mapped&) = delete;
    Mapped& operator=(const Mapped&) = delete;
    std::uint8_t* data() { return static_cast<std::uint8_t*>(addr); }
};

void close_fds(std::vector<int>& fds) noexcept {
    for (int fd : fds) ::close(fd);
    fds.clear();
}

std::string describe_mat(const cv::Mat& m) {
    return "type " + std::to_string(m.type()) + " (" + std::to_string(m.rows) + ", " +
           std::to_string(m.cols) + ")";
}

// python JobSource duck-typing (buffer_id()); release rides the owns list.
struct JobSourceHandle {
    enum class Kind { kPool, kImport, kRef };
    Kind kind = Kind::kPool;
    DspBufferPool* pool = nullptr;
    std::uint64_t import_id = 0;
    DspBufferRef* ref = nullptr;

    std::uint64_t buffer_id(int index = 0) const {
        switch (kind) {
            case Kind::kPool: return pool->buffer_id(index);
            case Kind::kImport: return import_id;
            case Kind::kRef: return ref->buffer_id();
        }
        return 0;
    }
};

// python's `own` list entries: released when the pending job / ref dies.
struct Ownee {
    virtual ~Ownee() = default;
    virtual void release() noexcept = 0;
};

struct PoolOwnee : Ownee {
    std::unique_ptr<DspBufferPool> pool;
    void release() noexcept override {
        if (pool) pool->release();
    }
};

// python _ImportedSource: one DSP_IMPORT id, released once.
struct ImportOwnee : Ownee {
    std::function<void(std::uint64_t)> send;
    std::uint64_t id = 0;
    bool done = false;
    void release() noexcept override {
        if (!done) {
            if (send) send(id);
            done = true;
        }
    }
};

void release_owned(std::vector<std::unique_ptr<Ownee>>& own) noexcept {
    for (auto& o : own) o->release();
    own.clear();
}

// python _resolve_source.
struct ResolvedSource {
    int width = 0;
    int height = 0;
    std::string fmt;
    FrameHandle* handle = nullptr;             // zero-copy dma-buf source
    DspBufferRef* ref = nullptr;               // daemon-side buffer source
    std::shared_ptr<FrameHandle> keep_handle;  // keeps a handle source alive
    Frame keep_frame;                          // keeps a Frame source alive
    cv::Mat pixels;                            // array sources (handle/ref null)
};

ResolvedSource resolve_source(const DspJobSource& src, const std::string& fmt) {
    ResolvedSource r;
    if (std::holds_alternative<DspBufferRef*>(src)) {
        DspBufferRef* ref = std::get<DspBufferRef*>(src);
        if (ref->released())
            throw DspError(
                "dsp buffer ref is released — its buffers are back with the daemon; "
                "keep the ref alive across the job");
        r.fmt = ref->fmt();
        if (!fmt.empty() && fmt != r.fmt)
            throw DspError("format mismatch: source is '" + r.fmt + "', fmt='" + fmt + "'");
        r.width = ref->width();
        r.height = ref->height();
        r.ref = ref;
        return r;
    }
    if (std::holds_alternative<std::shared_ptr<FrameHandle>>(src)) {
        r.keep_handle = std::get<std::shared_ptr<FrameHandle>>(src);
        r.handle = r.keep_handle.get();
    } else if (std::holds_alternative<Frame>(src)) {
        r.keep_frame = std::get<Frame>(src);
        const Frame& f = r.keep_frame;
        if (!f.handle) {
            if (f.image.empty())
                throw DspError(
                    "frame has neither pixels nor a dma-buf handle — "
                    "subscribe/receive with keep_fd=true to use it as a zero-copy "
                    "source");
            // the frame's format metadata outranks shape inference — a 2D NV12
            // array is indistinguishable from gray8 by shape alone
            const std::string frame_fmt = fmtx::frame_fmt_to_dsp(f.format);
            if (!fmt.empty() && !frame_fmt.empty() && fmt != frame_fmt)
                throw DspError("format mismatch: source is '" + frame_fmt + "', fmt='" +
                               fmt + "'");
            r.fmt = !fmt.empty() ? fmtx::infer_fmt(f.image, fmt) : frame_fmt;
            const cv::Size dims = fmtx::src_dims(f.image, r.fmt);
            r.width = dims.width;
            r.height = dims.height;
            r.pixels = f.image;
            return r;
        }
        r.handle = f.handle.get();
    } else {
        r.pixels = std::get<cv::Mat>(src);
        r.fmt = fmtx::infer_fmt(r.pixels, fmt);
        const cv::Size dims = fmtx::src_dims(r.pixels, r.fmt);
        r.width = dims.width;
        r.height = dims.height;
        return r;
    }

    // a retained handle source (from either variant arm)
    FrameHandle* handle = r.handle;
    if (handle->closed())
        throw DspError(
            "frame handle is closed — its dma-bufs are gone; keep the "
            "Frame/FrameHandle alive across the call");
    const std::string src_fmt = fmtx::frame_fmt_to_dsp(handle->format);
    if (src_fmt.empty())
        throw DspError((handle->format.empty() ? std::string("unknown-format")
                                               : handle->format) +
                       " frames cannot be imported as a DSP source "
                       "(supported: NV12/RGB/BGR/GRAY8)");
    if (handle->width <= 0 || handle->height <= 0)
        throw DspError(
            "frame handle carries no geometry — it predates SDK 0.6.0; re-fetch the "
            "frame");
    if (!fmt.empty() && fmt != src_fmt)
        throw DspError("format mismatch: source is '" + src_fmt + "', fmt='" + fmt + "'");
    r.fmt = src_fmt;
    r.width = handle->width;
    r.height = handle->height;
    return r;
}

// python _as_pixels: pixels behind an array source (never a zero-copy one).
cv::Mat as_pixels(const DspJobSource& src) {
    if (std::holds_alternative<Frame>(src)) {
        Frame f = std::get<Frame>(src);  // to_array() materializes+mutates; Mat is refcounted
        if (f.image.empty()) return f.to_array();
        return f.image;
    }
    if (std::holds_alternative<cv::Mat>(src)) return std::get<cv::Mat>(src);
    throw DspError("no CPU pixels behind this source");
}

// python _check_out_mode.
void check_out_mode(const DspJobParams& p) {
    if (p.out_ref && !p.wait)
        throw DspError(
            "out= is the sync no-read-back form; wait=false already returns a "
            "PendingDspJob whose wait_result() keeps the result device-side");
}

// python raises this shape for zero-copy sources when the DSP surface is
// unavailable (refusing the silent CPU fallback).
[[noreturn]] void refuse_zero_copy_fallback(const DspUnavailable& e, const char* what) {
    throw DspError(std::string(what) +
                   " with a zero-copy frame source — refusing the silent CPU fallback "
                   "(the source holds fds, not pixels; use frame.to_array()/ref.read() "
                   "to accept the copy)");
}

void warn_cpu_fallback(const std::string& reason) {
    warn(reason + "; CPU fallback engaged (client.last_used_hw records the path used)");
}

// CPU jpeg encode (python frame._encode_jpeg).
std::vector<std::uint8_t> encode_jpeg_cpu(const cv::Mat& rgb, int quality) {
    std::vector<std::uint8_t> out;
    std::vector<int> par{cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", rgb, out, par))
        throw DspError("jpeg encode failed on the CPU leg");
    return out;
}

// RGBA (h, w, 4) -> wire ARGB32 [A, R, G, B] (python rgba[:, :, [3, 0, 1, 2]]).
cv::Mat rgba_to_argb_wire(const cv::Mat& rgba) {
    cv::Mat out(rgba.rows, rgba.cols, CV_8UC4);
    const int from_to[] = {3, 0, 0, 1, 1, 2, 2, 3};
    cv::mixChannels(&rgba, 1, &out, 1, from_to, 4);
    return out;
}

// Build a PendingDspJob from a submitted job (python's wait=false tail).
PendingDspJob make_pending(DspClient* client, std::vector<std::pair<DspBufferPool*, int>> reads,
                           const std::optional<std::uint64_t>& job_id,
                           std::vector<std::unique_ptr<Ownee>> owns, double timeout_s,
                           bool multi);

DspBufferRef make_ref(DspBufferPool* pool, std::vector<std::unique_ptr<Ownee>> owns);

}  // namespace

// ---- DspBufferPool ------------------------------------------------------------

DspBufferPool::DspBufferPool(int w, int h, std::string f, std::vector<std::uint64_t> ids_,
                             std::vector<int> fds, const std::uint32_t strides_in[3],
                             const std::uint32_t sizes_in[3],
                             std::function<void(std::uint64_t)> send_release,
                             std::function<void(DspBufferPool*)> on_gone)
    : width(w),
      height(h),
      fmt(std::move(f)),
      ids(std::move(ids_)),
      plane_fds(std::move(fds)),
      send_release_(std::move(send_release)),
      on_gone_(std::move(on_gone)) {
    for (int i = 0; i < 3; ++i) {
        strides[i] = strides_in[i];
        plane_sizes[i] = sizes_in[i];
    }
    if (static_cast<int>(plane_fds.size()) !=
        static_cast<int>(ids.size()) * wire::plane_count(fmt)) {
        std::string msg = "alloc returned " + std::to_string(plane_fds.size()) + " fds for " +
                          std::to_string(ids.size()) + " " + fmt + " buffers (need " +
                          std::to_string(ids.size() * wire::plane_count(fmt)) + ")";
        close_fds(plane_fds);
        throw DspError(msg);
    }
}

DspBufferPool::~DspBufferPool() {
    release();
    if (on_gone_) on_gone_(this);
}

DspBufferPool::DspBufferPool(DspBufferPool&& o) noexcept
    : width(o.width),
      height(o.height),
      fmt(std::move(o.fmt)),
      ids(std::move(o.ids)),
      strides(o.strides),
      plane_sizes(o.plane_sizes),
      plane_fds(std::move(o.plane_fds)),
      released_(o.released_),
      send_release_(std::move(o.send_release_)),
      on_gone_(std::move(o.on_gone_)) {
    // the moved-from shell must not release or unregister (the client's
    // registry still holds the OLD address until its dtor fires); the live
    // pool at this address keeps working through send_release_, and the
    // daemon-side cleanup covers disconnect
    o.released_ = true;
    o.on_gone_ = nullptr;
}

DspBufferPool& DspBufferPool::operator=(DspBufferPool&& o) noexcept {
    if (this != &o) {
        release();
        if (on_gone_) on_gone_(this);
        width = o.width;
        height = o.height;
        fmt = std::move(o.fmt);
        ids = std::move(o.ids);
        strides = o.strides;
        plane_sizes = o.plane_sizes;
        plane_fds = std::move(o.plane_fds);
        released_ = o.released_;
        send_release_ = std::move(o.send_release_);
        on_gone_ = std::move(o.on_gone_);
        o.released_ = true;
        o.on_gone_ = nullptr;
    }
    return *this;
}

std::uint64_t DspBufferPool::buffer_id(int index) const {
    if (index < 0 || index >= static_cast<int>(ids.size()))
        throw DspError("buffer index " + std::to_string(index) + " outside the pool's " +
                       std::to_string(ids.size()) + " buffers");
    return ids[static_cast<std::size_t>(index)];
}

void DspBufferPool::write(int index, const cv::Mat& arr) {
    if (released_) throw DspError("write on released pool");
    if (index < 0 || index >= count())
        throw DspError("buffer index " + std::to_string(index) + " outside the pool's " +
                       std::to_string(count()) + " buffers");
    // expected shape (python _expected_shape)
    bool ok = true;
    if (fmt == "nv12")
        ok = arr.type() == CV_8UC1 && arr.dims == 2 && arr.rows == height * 3 / 2 &&
             arr.cols == width;
    else if (fmt == "rgb24")
        ok = arr.type() == CV_8UC3 && arr.rows == height && arr.cols == width;
    else if (fmt == "argb")
        ok = arr.type() == CV_8UC4 && arr.rows == height && arr.cols == width;
    else
        ok = arr.type() == CV_8UC1 && arr.dims == 2 && arr.rows == height && arr.cols == width;
    if (!ok)
        throw DspError("write expects uint8 " + fmt + " " + std::to_string(height) + "x" +
                       std::to_string(width) + ", got " + describe_mat(arr));

    // contiguous source planes
    cv::Mat packed = arr.isContinuous() ? arr : arr.clone();
    const int h = height;
    const std::uint8_t* y = nullptr;
    const std::uint8_t* uv = nullptr;
    cv::Mat flat;
    if (fmt == "nv12") {
        y = packed.ptr<std::uint8_t>(0);
        uv = packed.ptr<std::uint8_t>(h);
    } else if (fmt == "rgb24" || fmt == "argb") {
        flat = packed.reshape(1, h);  // (h, w*{3,4}) 8UC1
    }
    const auto rows_spec = wire::plane_rows(fmt, width, height);

    const int base = index * wire::plane_count(fmt);
    auto do_plane = [&](int p, const std::uint8_t* plane) {
        const int fd = plane_fds[static_cast<std::size_t>(base + p)];
        dma_sync(fd, kSyncWrite | kSyncStart);
        {
            Mapped m(fd, plane_sizes[static_cast<std::size_t>(p)]);
            if (m.addr == MAP_FAILED) {
                dma_sync(fd, kSyncWrite | kSyncEnd);
                throw DspError("cannot map dsp buffer plane " + std::to_string(p));
            }
            copy_rows(m.data(), strides[static_cast<std::size_t>(p)], const_cast<std::uint8_t*>(plane),
                      rows_spec[static_cast<std::size_t>(p)].row_bytes,
                      rows_spec[static_cast<std::size_t>(p)].rows, true);
        }
        dma_sync(fd, kSyncWrite | kSyncEnd);
    };
    if (fmt == "nv12") {
        do_plane(0, y);
        do_plane(1, uv);
    } else if (fmt == "gray8") {
        do_plane(0, packed.ptr<std::uint8_t>(0));
    } else {
        do_plane(0, flat.ptr<std::uint8_t>(0));
    }
}

cv::Mat DspBufferPool::read(int index) {
    if (released_) throw DspError("read on released pool");
    if (index < 0 || index >= count())
        throw DspError("buffer index " + std::to_string(index) + " outside the pool's " +
                       std::to_string(count()) + " buffers");
    const int h = height;
    const int base = index * wire::plane_count(fmt);
    const auto rows_spec = wire::plane_rows(fmt, width, height);

    cv::Mat out;
    if (fmt == "nv12")
        out.create(h * 3 / 2, width, CV_8UC1);
    else if (fmt == "rgb24" || fmt == "argb")
        out.create(h, static_cast<int>(rows_spec[0].row_bytes), CV_8UC1);
    else
        out.create(h, width, CV_8UC1);

    std::size_t plane_off = 0;  // byte offset of this plane inside `out`
    for (std::size_t p = 0; p < rows_spec.size(); ++p) {
        const int fd = plane_fds[static_cast<std::size_t>(base + static_cast<int>(p))];
        dma_sync(fd, kSyncRead | kSyncStart);
        {
            Mapped m(fd, plane_sizes[p]);
            if (m.addr == MAP_FAILED) {
                dma_sync(fd, kSyncRead | kSyncEnd);
                throw DspError("cannot map dsp buffer plane " + std::to_string(p));
            }
            copy_rows(m.data(), strides[p], out.data + plane_off, rows_spec[p].row_bytes,
                      rows_spec[p].rows, false);
        }
        dma_sync(fd, kSyncRead | kSyncEnd);
        plane_off += static_cast<std::size_t>(rows_spec[p].row_bytes) * rows_spec[p].rows;
    }
    if (fmt == "rgb24") return out.reshape(3, h);
    if (fmt == "argb") return out.reshape(4, h);
    return out;
}

void DspBufferPool::release() noexcept {
    if (released_) return;
    released_ = true;
    if (send_release_) {
        for (std::uint64_t bid : ids) send_release_(bid);
    }
    close_fds(plane_fds);
}

void DspBufferPool::mark_dead() noexcept {
    if (released_) return;
    released_ = true;
    close_fds(plane_fds);
}

// ---- SharedImportPool ----------------------------------------------------------

SharedImportPool::SharedImportPool(int w, int h, std::vector<std::uint64_t> ids_,
                                   std::vector<void*> maps, std::vector<std::size_t> map_sizes,
                                   std::vector<int> fds,
                                   std::function<void(std::uint64_t)> send_release)
    : width(w),
      height(h),
      ids(std::move(ids_)),
      maps_(std::move(maps)),
      map_sizes_(std::move(map_sizes)),
      fds_(std::move(fds)),
      send_release_(std::move(send_release)) {}

SharedImportPool::~SharedImportPool() { release(); }

SharedImportPool::SharedImportPool(SharedImportPool&& o) noexcept
    : width(o.width),
      height(o.height),
      fmt(std::move(o.fmt)),
      ids(std::move(o.ids)),
      released_(o.released_),
      maps_(std::move(o.maps_)),
      map_sizes_(std::move(o.map_sizes_)),
      fds_(std::move(o.fds_)),
      send_release_(std::move(o.send_release_)) {
    o.released_ = true;
    o.maps_.clear();
    o.map_sizes_.clear();
    o.fds_.clear();
}

SharedImportPool& SharedImportPool::operator=(SharedImportPool&& o) noexcept {
    if (this != &o) {
        release();
        width = o.width;
        height = o.height;
        fmt = std::move(o.fmt);
        ids = std::move(o.ids);
        released_ = o.released_;
        maps_ = std::move(o.maps_);
        map_sizes_ = std::move(o.map_sizes_);
        fds_ = std::move(o.fds_);
        send_release_ = std::move(o.send_release_);
        o.released_ = true;
        o.maps_.clear();
        o.map_sizes_.clear();
        o.fds_.clear();
    }
    return *this;
}

std::uint64_t SharedImportPool::buffer_id(int index) const {
    if (index < 0 || index >= count())
        throw DspError("buffer index " + std::to_string(index) + " outside the pool's " +
                       std::to_string(count()) + " slots");
    return ids[static_cast<std::size_t>(index)];
}

void SharedImportPool::write(int index, const cv::Mat& arr) {
    if (released_) throw DspError("write on released pool");
    if (index < 0 || index >= count())
        throw DspError("buffer index " + std::to_string(index) + " outside the pool's " +
                       std::to_string(count()) + " slots");
    if (arr.type() != CV_8UC4 || arr.rows != height || arr.cols != width)
        throw DspError("write expects uint8 (" + std::to_string(height) + ", " +
                       std::to_string(width) + ", 4), got " + describe_mat(arr));
    const cv::Mat packed = arr.isContinuous() ? arr : arr.clone();
    std::memcpy(maps_[static_cast<std::size_t>(index)], packed.data,
                static_cast<std::size_t>(height) * width * 4);
}

void SharedImportPool::release() noexcept {
    if (released_) return;
    released_ = true;
    if (send_release_) {
        for (std::uint64_t bid : ids) send_release_(bid);
    }
    for (std::size_t i = 0; i < maps_.size(); ++i) {
        if (maps_[i] != MAP_FAILED) ::munmap(maps_[i], map_sizes_[i]);
    }
    maps_.clear();
    close_fds(fds_);
}

// ---- PendingDspJob --------------------------------------------------------------

struct PendingDspJob::Impl {
    DspClient* client = nullptr;
    std::vector<std::pair<DspBufferPool*, int>> reads;
    std::vector<std::unique_ptr<Ownee>> owns;
    double timeout_s = 0.0;
    bool multi = false;
    std::optional<std::uint64_t> job_id;  // empty = sync fallback already ran
    enum class St { kPending, kDone, kReleased } state = St::kPending;
    std::optional<std::string> error;
    std::optional<std::int32_t> error_code;

    // One WaitDspJob round-trip. Returns true when the job completed, false
    // while still pending; throws for rpc failures, unknown ids and failed
    // jobs.
    bool wait_one(int timeout_ms);
};

PendingDspJob::~PendingDspJob() { release(); }
PendingDspJob::PendingDspJob(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PendingDspJob::PendingDspJob(PendingDspJob&&) noexcept = default;
PendingDspJob& PendingDspJob::operator=(PendingDspJob&& o) noexcept {
    if (this != &o) {
        if (impl_) release();  // free the job being overwritten (its buffers too)
        impl_ = std::move(o.impl_);
    }
    return *this;
}

std::optional<std::uint64_t> PendingDspJob::job_id() const {
    return impl_ ? impl_->job_id : std::nullopt;
}

std::uint64_t PendingDspJob::buffer_id() const {
    if (!impl_) throw DspError("pending job already consumed");
    const auto& entry = impl_->reads.at(0);
    return entry.first->buffer_id(entry.second);
}

bool PendingDspJob::done(double timeout_s) {
    if (!impl_) return true;  // nothing pending in a consumed/moved-from handle
    if (impl_->state != Impl::St::kPending) return impl_->state == Impl::St::kDone;
    try {
        if (!impl_->wait_one(static_cast<int>(std::max(timeout_s, 0.0) * 1000))) return false;
        impl_->state = Impl::St::kDone;
        return true;
    } catch (const DspError& e) {
        // a job that failed still counts as done; the error surfaces from
        // wait()/wait_result()
        impl_->state = Impl::St::kDone;
        impl_->error = e.what();
        impl_->error_code = e.code();
        return true;
    }
}

PendingDspJob& PendingDspJob::wait_result(std::optional<double> timeout_s) {
    if (!impl_ || impl_->state == Impl::St::kReleased)
        throw DspError("pending job already released");
    if (impl_->state == Impl::St::kPending) {
        const double limit = timeout_s.has_value() ? *timeout_s : impl_->timeout_s;
        if (!impl_->wait_one(static_cast<int>(std::max(limit, 0.0) * 1000)))
            throw DspError("dsp job " + std::to_string(impl_->job_id.value_or(0)) +
                           " still pending after " + std::to_string(limit) +
                           "s — the daemon keeps it; wait again with a longer timeout");
        impl_->state = Impl::St::kDone;
    }
    if (impl_->error) throw DspError(*impl_->error, impl_->error_code);
    return *this;
}

cv::Mat PendingDspJob::wait(std::optional<double> timeout_s) {
    wait_result(timeout_s);
    if (impl_->multi) throw DspError("multi-crop job: use wait_all()");
    const auto& entry = impl_->reads.at(0);
    cv::Mat out = entry.first->read(entry.second);
    release();
    return out;
}

std::vector<cv::Mat> PendingDspJob::wait_all(std::optional<double> timeout_s) {
    wait_result(timeout_s);
    std::vector<cv::Mat> out;
    out.reserve(impl_->reads.size());
    for (const auto& entry : impl_->reads) out.push_back(entry.first->read(entry.second));
    release();
    return out;
}

void PendingDspJob::release() noexcept {
    if (!impl_ || impl_->state == Impl::St::kReleased) return;
    if (impl_->state == Impl::St::kPending) {
        try {
            impl_->wait_one(static_cast<int>(std::max(impl_->timeout_s, 0.0) * 1000));
        } catch (const DspError&) {
            // callers that care surface errors from wait_result
        }
    }
    impl_->state = Impl::St::kReleased;
    release_owned(impl_->owns);
    impl_->reads.clear();
}

// ---- DspBufferRef -----------------------------------------------------------------

struct DspBufferRef::Impl {
    std::vector<std::unique_ptr<Ownee>> owns;
};

DspBufferRef::~DspBufferRef() { release(); }
DspBufferRef::DspBufferRef(std::unique_ptr<Impl> impl, DspBufferPool* pool)
    : impl_(std::move(impl)), pool_(pool) {}

DspBufferRef DspBufferRef::detached() {
    // built here, where Impl is complete — the public ctor's by-value
    // unique_ptr cannot be destroyed in an outside TU
    return DspBufferRef(nullptr, nullptr);
}

DspBufferRef DspBufferRef::borrowed(DspBufferPool* pool) {
    // alive (non-null impl, released_ false) but owning nothing, so
    // release() is a safe no-op and every accessor reads the pool
    return DspBufferRef(std::make_unique<Impl>(), pool);
}
DspBufferRef::DspBufferRef(DspBufferRef&&) noexcept = default;
DspBufferRef& DspBufferRef::operator=(DspBufferRef&& o) noexcept {
    if (this != &o) {
        if (impl_ && !released_) release();  // return the buffers being overwritten
        impl_ = std::move(o.impl_);
        pool_ = o.pool_;
        released_ = o.released_;
    }
    return *this;
}

namespace {
void ref_ensure_alive(const DspBufferRef* ref, const DspBufferPool* pool) {
    if (ref->released())
        throw DspError(
            "dsp buffer ref is released — its buffers are back with the daemon (or "
            "reused); keep the ref alive across jobs that read it, like a keep-fd "
            "frame");
    if (pool->released())
        throw DspError(
            "dsp buffer ref's pool is gone — released explicitly, or its client "
            "closed and the daemon reclaimed the buffers on disconnect; keep the "
            "owning client open across jobs that read the ref");
}
}  // namespace

std::uint64_t DspBufferRef::buffer_id() const {
    ref_ensure_alive(this, pool_);
    return pool_->buffer_id(0);
}

cv::Mat DspBufferRef::read() {
    ref_ensure_alive(this, pool_);
    return pool_->read(0);
}

void DspBufferRef::release() noexcept {
    if (!impl_ || released_) return;
    released_ = true;
    release_owned(impl_->owns);
}

// ---- DspClient::Impl ---------------------------------------------------------------

struct DspClient::Impl : detail::GrpcClientBase<pb::CameraControl> {
    std::string sock_path;
    int sock = -1;
    // Serializes the buffer-plane UDS exchanges (ALLOC / IMPORT /
    // BUF_RELEASE send+recv cycles). The job plane (gRPC) is thread-safe on
    // its own; the raw socket is not.
    std::mutex plane_lock;
    std::mutex pools_lock;
    std::vector<DspBufferPool*> live_pools;
    std::optional<bool> last_used_hw;
    bool gray8_refused = false;

    Impl(std::string endpoint, std::string sock_path)
        : GrpcClientBase(std::move(endpoint)), sock_path(std::move(sock_path)) {}

    ~Impl() { close_sock(); }

    int ensure_sock() {
        if (sock < 0) {
            try {
                sock = detail::connect_unix(sock_path);
            } catch (const std::runtime_error& e) {
                throw DspError("cannot connect to camera socket " + sock_path + ": " +
                               e.what());
            }
        }
        return sock;
    }

    void close_sock() noexcept {
        if (sock >= 0) {
            ::close(sock);
            sock = -1;
        }
    }

    void register_pool(DspBufferPool* p) {
        std::lock_guard<std::mutex> g(pools_lock);
        live_pools.push_back(p);
    }

    void unregister_pool(DspBufferPool* p) noexcept {
        std::lock_guard<std::mutex> g(pools_lock);
        live_pools.erase(std::remove(live_pools.begin(), live_pools.end(), p),
                         live_pools.end());
    }

    void mark_pools_dead() noexcept {
        std::lock_guard<std::mutex> g(pools_lock);
        for (DspBufferPool* p : live_pools) p->mark_dead();
        live_pools.clear();
    }

    // python _send_release: fire-and-forget DSP_BUF_RELEASE.
    void send_release(std::uint64_t buffer_id) noexcept {
        std::lock_guard<std::mutex> g(plane_lock);
        if (sock < 0) return;
        const auto req = wire::release_request(buffer_id);
        if (!detail::send_all(sock, req.data(), req.size()))
            std::fprintf(stderr, "neoruntime_ipc_sdk: DSP release send failed\n");
    }

    // python _exchange_alloc: DSP_ALLOC out, RESP (with fds) back.
    wire::AllocResp exchange_alloc(std::uint32_t width, std::uint32_t height,
                                   std::uint32_t fmt_wire, std::uint32_t count,
                                   std::vector<int>& fds) {
        std::lock_guard<std::mutex> g(plane_lock);
        const int fd = ensure_sock();
        const auto req = wire::alloc_request(width, height, fmt_wire, count);
        if (!detail::send_all(fd, req.data(), req.size()))
            throw DspError("camera socket closed during DSP alloc");

        std::vector<std::uint8_t> buf(wire::kAllocRespSize);
        std::size_t got = 0;
        while (got < wire::kAllocRespSize) {
            const std::size_t before = fds.size();
            ssize_t n = detail::recvmsg_with_fds(
                fd, buf.data() + got, wire::kAllocRespSize - got, fds,
                static_cast<int>(wire::kDspMaxFds));
            if (n == 0 && fds.size() == before)
                throw DspError("camera socket closed during DSP alloc");
            if (n < 0) throw DspError("camera socket error during DSP alloc");
            got += static_cast<std::size_t>(n);
        }
        wire::AllocResp resp = wire::parse_alloc_resp(buf.data(), buf.size());
        if (resp.code != 0) {
            close_fds(fds);
            throw DspError(wire::error_text(resp.code), resp.code);
        }
        if (resp.count != count || fds.size() != count * resp.num_planes) {
            std::string msg = "alloc returned " + std::to_string(resp.count) +
                              " buffers / " + std::to_string(fds.size()) +
                              " fds, requested " + std::to_string(count);
            close_fds(fds);
            throw DspError(msg);
        }
        return resp;
    }

    // python _import_planes: geometry + fds out, import id back.
    std::uint64_t import_planes(std::uint32_t width, std::uint32_t height,
                                const std::string& fmt, std::uint32_t num_planes,
                                const std::uint32_t strides[3], const std::uint32_t sizes[3],
                                const std::vector<int>& fds, double timeout_s) {
        const auto payload = wire::import_request(width, height, wire::hal_pixel_format(fmt),
                                                  num_planes, strides, sizes);
        std::lock_guard<std::mutex> g(plane_lock);
        const int fd = ensure_sock();
        if (!sendmsg_with_fds(fd, payload.data(), payload.size(), fds))
            throw DspError("camera socket send failed during DSP import");

        detail::set_recv_timeout(fd, static_cast<int>(timeout_s * 1000));
        bool would_block = false;
        try {
            for (int drain = 0; drain < 64; ++drain) {
                UdsMsg m = recv_one_msg(fd, &would_block);
                close_fds(m.fds);  // the reply itself never carries fds; also
                                   // drops any a timed-out partial read got
                if (would_block)
                    throw DspError("no DSP_IMPORT response in " +
                                   std::to_string(timeout_s) +
                                   "s — the daemon may predate DSP_IMPORT (needs "
                                   "platform a94ee007+); close this client, the socket "
                                   "may hold a partial message");
                if (m.type == wire::kMsgDspImportResp) {
                    const wire::ImportResp resp =
                        wire::parse_import_resp(m.bytes.data(), m.bytes.size());
                    if (resp.code != 0)
                        throw DspError("buffer import rejected: " + wire::error_text(resp.code),
                                       resp.code);
                    return resp.import_id;
                }
                if (m.type == wire::kMsgOk || m.type == wire::kMsgError)
                    continue;  // control acks from an earlier request
                throw DspError("unexpected camera-sock message type " +
                               std::to_string(m.type) +
                               " while awaiting DSP import response");
            }
        } catch (...) {
            detail::set_recv_timeout(fd, 0);
            throw;
        }
        detail::set_recv_timeout(fd, 0);
        throw DspError("too many control messages before import response");
    }

    // python _import_source: a frame handle's dma-bufs as a job source.
    std::uint64_t import_source(FrameHandle& handle, int width, int height,
                                const std::string& fmt, double timeout_s) {
        const auto num_planes = static_cast<std::uint32_t>(handle.fds.size());
        if (num_planes != static_cast<std::uint32_t>(wire::plane_count(fmt)))
            throw DspError(fmt + " source carries " + std::to_string(num_planes) +
                           " dma-buf fd(s), expected " +
                           std::to_string(wire::plane_count(fmt)));
        std::uint32_t strides[3] = {0, 0, 0}, sizes[3] = {0, 0, 0};
        for (std::size_t i = 0; i < handle.fds.size() && i < 3; ++i) {
            strides[i] = handle.strides[i];
            sizes[i] = handle.plane_sizes[i];
        }
        return import_planes(static_cast<std::uint32_t>(width),
                             static_cast<std::uint32_t>(height), fmt, num_planes, strides,
                             sizes, handle.fds, timeout_s);
    }

    // python _import_memfd: client-owned plane bytes as a single-plane buffer.
    std::uint64_t import_memfd(const std::uint8_t* bytes, std::size_t len, int width,
                               int height, const std::string& fmt, std::uint32_t stride,
                               double timeout_s) {
        int fd = ::memfd_create("dsp-import", 0);
        if (fd < 0)
            throw DspError(std::string("memfd_create failed: ") + std::strerror(errno));
        try {
            if (::ftruncate(fd, static_cast<off_t>(len)) != 0)
                throw DspError(std::string("ftruncate failed: ") + std::strerror(errno));
            std::size_t off = 0;
            while (off < len) {  // write(2) may be partial on large buffers
                ssize_t n = ::write(fd, bytes + off, len - off);
                if (n < 0) {
                    if (errno == EINTR) continue;
                    throw DspError(std::string("memfd write failed: ") +
                                   std::strerror(errno));
                }
                off += static_cast<std::size_t>(n);
            }
            const std::uint32_t strides[3] = {stride, 0, 0};
            const std::uint32_t sizes[3] = {static_cast<std::uint32_t>(len), 0, 0};
            const std::vector<int> fds{fd};
            const std::uint64_t id =
                import_planes(static_cast<std::uint32_t>(width),
                              static_cast<std::uint32_t>(height), fmt, 1, strides, sizes,
                              fds, timeout_s);
            ::close(fd);  // the daemon holds its own dup from SCM_RIGHTS
            return id;
        } catch (...) {
            ::close(fd);
            throw;
        }
    }

    // python _submit_job. `rects` carry (x, y, w, h, dw, dh) 6-tuples.
    struct SubmitResult {
        std::uint64_t elapsed_ms = 0;         // sync form
        std::optional<std::uint64_t> job_id;  // async form; empty = shadowed sync
    };

    SubmitResult submit_job(std::int32_t op, std::uint64_t src_id,
                            const std::vector<std::uint64_t>& dst_ids,
                            const std::vector<std::array<std::int32_t, 6>>& rects,
                            const std::string& interpolation, const std::string& scaling,
                            const std::string& priority, double timeout_s, bool wait) {
        pb::DspJobRequest req;
        req.set_op(static_cast<pb::DspOp>(op));
        req.set_src_buffer_id(src_id);
        for (std::uint64_t id : dst_ids) req.add_dst_buffer_ids(id);
        for (const auto& r : rects) {
            pb::DspRect* rect = req.add_rects();
            rect->set_x(static_cast<std::uint32_t>(r[0]));
            rect->set_y(static_cast<std::uint32_t>(r[1]));
            rect->set_width(static_cast<std::uint32_t>(r[2]));
            rect->set_height(static_cast<std::uint32_t>(r[3]));
            rect->set_dst_width(static_cast<std::uint32_t>(r[4]));
            rect->set_dst_height(static_cast<std::uint32_t>(r[5]));
        }
        // ALWAYS explicit: proto default 0 = NEAREST, which the vendor
        // MULTI_CROP path rejects (HAL -2801)
        req.set_interpolation(wire::interp_wire(interpolation));
        req.set_scaling_mode(wire::scaling_wire(scaling));
        req.set_priority(wire::priority_wire(priority));

        const auto apply_deadline = [timeout_s](grpc::ClientContext& ctx) {
            ctx.set_deadline(std::chrono::system_clock::now() +
                             std::chrono::milliseconds(static_cast<long>(timeout_s * 1000)));
        };

        SubmitResult out;
        pb::DspJobResponse resp;
        grpc::Status st;
        bool shadowed = false;
        if (wait) {
            grpc::ClientContext ctx;
            apply_deadline(ctx);
            st = stub().SubmitDspJob(&ctx, req, &resp);
        } else {
            grpc::ClientContext ctx;
            apply_deadline(ctx);
            st = stub().SubmitDspJobAsync(&ctx, req, &resp);
            if (st.ok()) {
                // accepted for async execution; job success surfaces on wait
                out.job_id = resp.job_id();
                return out;
            }
            if (st.error_code() != grpc::StatusCode::UNIMPLEMENTED)
                throw DspError("SubmitDspJobAsync rpc failed: " + st.error_message());
            // old daemon: run the job synchronously instead — the pending
            // handle is born already done
            grpc::ClientContext ctx2;
            apply_deadline(ctx2);
            pb::DspJobResponse sync_resp;
            st = stub().SubmitDspJob(&ctx2, req, &sync_resp);
            resp = std::move(sync_resp);
            shadowed = true;
        }
        if (!st.ok()) {
            if (st.error_code() == grpc::StatusCode::UNIMPLEMENTED)
                throw DspUnavailable("SubmitDspJob not in daemon");
            throw DspError("SubmitDspJob rpc failed: " + st.error_message());
        }
        if (!resp.success()) {
            if (resp.error_code() == wire::kSvcUnavailable)
                throw DspUnavailable("dsp service not running");
            throw DspError("dsp job failed: " +
                           (resp.message().empty() ? wire::error_text(resp.error_code())
                                                   : resp.message()),
                           resp.error_code());
        }
        if (shadowed) {
            out.job_id = std::nullopt;  // sync fallback ran it; born done
            return out;
        }
        out.elapsed_ms = resp.elapsed_ms();
        return out;
    }

    // python _wait_rpc. Returns true when the job completed, false while
    // still pending (-4 timeout).
    bool wait_job(std::uint64_t job_id, int timeout_ms) {
        pb::DspWaitRequest req;
        req.set_job_id(job_id);
        req.set_timeout_ms(static_cast<std::uint32_t>(timeout_ms));
        grpc::ClientContext ctx;
        // the server holds its reply for up to timeout_ms — keep the rpc
        // deadline comfortably past it
        ctx.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(timeout_ms + 2000));
        pb::DspJobResponse resp;
        grpc::Status st = stub().WaitDspJob(&ctx, req, &resp);
        if (!st.ok()) {
            if (st.error_code() == grpc::StatusCode::UNIMPLEMENTED)
                throw DspUnavailable("WaitDspJob not in daemon");
            throw DspError("WaitDspJob rpc failed: " + st.error_message());
        }
        if (!resp.success()) {
            if (resp.error_code() == wire::kErrTimeout) return false;  // re-wait works
            throw DspError("dsp job failed: " +
                           (resp.message().empty() ? wire::error_text(resp.error_code())
                                                   : resp.message()),
                           resp.error_code());
        }
        return true;
    }

    // python _encode_rpc: one EncodeImage round-trip.
    std::vector<std::uint8_t> encode_rpc(std::uint64_t src_id, int quality, double timeout_s) {
        pb::EncodeImageRequest req;
        req.set_src_buffer_id(src_id);
        req.set_quality(static_cast<std::uint32_t>(quality));
        grpc::ClientContext ctx;
        ctx.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(static_cast<long>(timeout_s * 1000)));
        pb::EncodeImageResponse resp;
        grpc::Status st = stub().EncodeImage(&ctx, req, &resp);
        if (!st.ok()) {
            if (st.error_code() == grpc::StatusCode::UNIMPLEMENTED)
                throw DspUnavailable("EncodeImage not in daemon");
            throw DspError("EncodeImage rpc failed: " + st.error_message());
        }
        if (!resp.success()) {
            if (resp.error_code() == wire::kSvcUnavailable)
                throw DspUnavailable("dsp service not running");
            throw DspError("jpeg encode failed: " +
                           (resp.message().empty() ? wire::error_text(resp.error_code())
                                                   : resp.message()),
                           resp.error_code());
        }
        if (resp.jpeg().empty())
            throw DspError("EncodeImage returned success with no jpeg payload");
        return {resp.jpeg().begin(), resp.jpeg().end()};
    }
};

bool PendingDspJob::Impl::wait_one(int timeout_ms) {
    return client->impl_->wait_job(job_id.value_or(0), timeout_ms);
}

// ---- DspClient -------------------------------------------------------------------

DspClient::DspClient(std::string sock_path, std::string endpoint)
    : impl_(std::make_unique<Impl>(
          endpoint.empty() ? Config::get_camera_control_endpoint() : std::move(endpoint),
          std::move(sock_path))) {
    if (impl_->sock_path.empty()) {
        const char* env = std::getenv("CAMERA_SOCK_PATH");
        impl_->sock_path = env ? env : "/run/aipc/camera.sock";
    }
}

DspClient::~DspClient() = default;

void DspClient::connect() { impl_->connect(); }

void DspClient::close() noexcept {
    impl_->mark_pools_dead();
    impl_->close_sock();
    impl_->close();
}

bool DspClient::connected() const noexcept { return impl_->connected(); }

std::optional<bool> DspClient::last_used_hw() const noexcept { return impl_->last_used_hw; }

bool DspClient::has_live_pools() const noexcept {
    std::lock_guard<std::mutex> g(impl_->pools_lock);
    for (const DspBufferPool* p : impl_->live_pools)
        if (!p->released()) return true;
    return false;
}

std::unique_ptr<DspBufferPool> DspClient::alloc_buffers(int width, int height,
                                                        const std::string& fmt, int count) {
    wire::validate_geometry(width, height, fmt, "alloc");
    if (count < 1) throw DspError("count must be >= 1");
    if (count * wire::plane_count(fmt) > static_cast<int>(wire::kDspMaxFds))
        throw DspError("count*num_planes exceeds the " +
                       std::to_string(wire::kDspMaxFds) + "-fd UDS response cap");
    std::vector<int> fds;
    wire::AllocResp resp =
        impl_->exchange_alloc(static_cast<std::uint32_t>(width),
                              static_cast<std::uint32_t>(height),
                              wire::hal_pixel_format(fmt), static_cast<std::uint32_t>(count),
                              fds);
    auto pool = std::make_unique<DspBufferPool>(
        width, height, fmt, std::move(resp.ids), std::move(fds), resp.strides, resp.sizes,
        [impl = impl_.get()](std::uint64_t id) { impl->send_release(id); },
        [impl = impl_.get()](DspBufferPool* p) { impl->unregister_pool(p); });
    impl_->register_pool(pool.get());
    return pool;
}

std::unique_ptr<SharedImportPool> DspClient::import_shared_buffers(
    int width, int height, const std::string& fmt, int count, double timeout_s) {
    if (fmt != "argb") throw DspError("shared import supports ARGB32 only");
    wire::validate_geometry(width, height, fmt, "shared import");
    if (count < 1) throw DspError("count must be >= 1");
    const std::uint32_t stride = static_cast<std::uint32_t>(width) * 4;
    const std::size_t size = static_cast<std::size_t>(stride) * height;
    std::vector<std::uint64_t> ids;
    std::vector<void*> maps;
    std::vector<std::size_t> map_sizes;
    std::vector<int> fds;
    int fd = -1;
    try {
        for (int slot = 0; slot < count; ++slot) {
            fd = ::memfd_create("dsp-shared-import", 0);
            if (fd < 0)
                throw DspError(std::string("memfd_create failed: ") + std::strerror(errno));
            if (::ftruncate(fd, static_cast<off_t>(size)) != 0)
                throw DspError(std::string("ftruncate failed: ") + std::strerror(errno));
            const std::uint32_t strides[3] = {stride, 0, 0};
            const std::uint32_t sizes[3] = {static_cast<std::uint32_t>(size), 0, 0};
            ids.push_back(impl_->import_planes(static_cast<std::uint32_t>(width),
                                               static_cast<std::uint32_t>(height), fmt, 1,
                                               strides, sizes, std::vector<int>{fd},
                                               timeout_s));
            void* mapped = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (mapped == MAP_FAILED) throw DspError("cannot map shared import slot");
            maps.push_back(mapped);
            map_sizes.push_back(size);
            fds.push_back(fd);
            fd = -1;
        }
    } catch (...) {
        for (std::size_t i = 0; i < maps.size(); ++i) ::munmap(maps[i], map_sizes[i]);
        if (fd >= 0) ::close(fd);  // the failing slot's own memfd, never in fds
        close_fds(fds);
        for (std::uint64_t id : ids) impl_->send_release(id);  // earlier imports back
        throw;
    }
    return std::make_unique<SharedImportPool>(
        width, height, std::move(ids), std::move(maps), std::move(map_sizes), std::move(fds),
        [impl = impl_.get()](std::uint64_t id) { impl->send_release(id); });
}

std::uint64_t DspClient::import_frame(FrameHandle& handle) {
    // aliasing shared_ptr: borrows the handle without owning it
    ResolvedSource r = resolve_source(
        DspJobSource{std::shared_ptr<FrameHandle>(std::shared_ptr<FrameHandle>(), &handle)},
        "");
    return impl_->import_source(handle, r.width, r.height, r.fmt, 5.0);
}

void DspClient::release_buffer(std::uint64_t buffer_id) {
    if (buffer_id == 0) throw DspError("invalid buffer id: " + std::to_string(buffer_id));
    impl_->send_release(buffer_id);
}

namespace {

// python _prep: resolve the source to a daemon buffer and allocate/collect
// destination pools. `dst_specs` are (dw, dh, count) per distinct geometry.
struct Prepared {
    JobSourceHandle source;
    std::vector<DspBufferPool*> pools;  // per dst spec
    std::vector<std::unique_ptr<Ownee>> owns;
};

PendingDspJob make_pending(DspClient* client,
                           std::vector<std::pair<DspBufferPool*, int>> reads,
                           const std::optional<std::uint64_t>& job_id,
                           std::vector<std::unique_ptr<Ownee>> owns, double timeout_s,
                           bool multi) {
    auto impl = std::make_unique<PendingDspJob::Impl>();
    impl->client = client;
    impl->reads = std::move(reads);
    impl->owns = std::move(owns);
    impl->timeout_s = timeout_s;
    impl->multi = multi;
    impl->job_id = job_id;
    // empty job id = the sync fallback already executed the job
    impl->state = job_id.has_value() ? PendingDspJob::Impl::St::kPending
                                     : PendingDspJob::Impl::St::kDone;
    return PendingDspJob(std::move(impl));
}

DspBufferRef make_ref(DspBufferPool* pool, std::vector<std::unique_ptr<Ownee>> owns) {
    auto impl = std::make_unique<DspBufferRef::Impl>();
    impl->owns = std::move(owns);
    return DspBufferRef(std::move(impl), pool);
}

}  // namespace

// The three legs of prep live as DspClient members so alloc_buffers (a
// public method) stays in reach.
static Prepared client_prep(DspClient& self, DspClient::Impl& impl, const DspJobSource& src,
                            ResolvedSource& r, const std::vector<std::array<int, 3>>& dst_specs,
                            DspBufferPool* src_pool,
                            const std::vector<DspBufferPool*>& dst_pools, double timeout_s);

std::variant<cv::Mat, PendingDspJob, DspBufferRef> DspClient::resize_hw(
    const DspJobSource& src, int width, int height, std::string fmt,
    const DspJobParams& params, DspBufferPool* src_pool, DspBufferPool* dst_pool) {
    check_out_mode(params);
    ResolvedSource r = resolve_source(src, fmt);
    wire::validate_geometry(width, height, r.fmt, "destination");
    try {
        Prepared p = client_prep(*this, *impl_, src, r, {{width, height, 1}}, src_pool,
                                 dst_pool ? std::vector<DspBufferPool*>{dst_pool}
                                          : std::vector<DspBufferPool*>{},
                                 params.timeout_s);
        try {
            Impl::SubmitResult j = impl_->submit_job(
                wire::kOpResize, p.source.buffer_id(0), {p.pools[0]->buffer_id(0)}, {},
                params.interpolation, params.scaling, params.priority, params.timeout_s,
                params.wait);
            impl_->last_used_hw = true;
            if (!params.wait)
                return make_pending(this, {{p.pools[0], 0}}, j.job_id, std::move(p.owns),
                                    params.timeout_s, false);
            if (params.out_ref) return make_ref(p.pools[0], std::move(p.owns));
            cv::Mat out = p.pools[0]->read(0);
            release_owned(p.owns);
            return out;
        } catch (...) {
            release_owned(p.owns);
            throw;
        }
    } catch (const DspUnavailable& e) {
        if (r.handle || r.ref) refuse_zero_copy_fallback(e, "DSP unavailable");
        if (!params.cpu_fallback) throw;
        warn_cpu_fallback(e.what());
        impl_->last_used_hw = false;
        return fmtx::cpu_resize(as_pixels(src), r.fmt, width, height, params.scaling,
                                params.interpolation);
    }
}

std::variant<cv::Mat, PendingDspJob, DspBufferRef> DspClient::crop_hw(
    const DspJobSource& src, int x, int y, int width, int height, int dst_width,
    int dst_height, std::string fmt, const DspJobParams& params, DspBufferPool* src_pool,
    DspBufferPool* dst_pool) {
    check_out_mode(params);
    ResolvedSource r = resolve_source(src, fmt);
    if (dst_width < 0) dst_width = width;
    if (dst_height < 0) dst_height = height;
    const fmtx::CropRect rect = fmtx::validated_rect(r.width, r.height, r.fmt, x, y, width,
                                                     height, dst_width, dst_height);
    try {
        Prepared p = client_prep(*this, *impl_, src, r, {{dst_width, dst_height, 1}}, src_pool,
                                 dst_pool ? std::vector<DspBufferPool*>{dst_pool}
                                          : std::vector<DspBufferPool*>{},
                                 params.timeout_s);
        try {
            Impl::SubmitResult j = impl_->submit_job(
                wire::kOpCropAndResize, p.source.buffer_id(0), {p.pools[0]->buffer_id(0)},
                {{rect.x, rect.y, rect.w, rect.h, rect.dw, rect.dh}}, params.interpolation,
                params.scaling, params.priority, params.timeout_s, params.wait);
            impl_->last_used_hw = true;
            if (!params.wait)
                return make_pending(this, {{p.pools[0], 0}}, j.job_id, std::move(p.owns),
                                    params.timeout_s, false);
            if (params.out_ref) return make_ref(p.pools[0], std::move(p.owns));
            cv::Mat out = p.pools[0]->read(0);
            release_owned(p.owns);
            return out;
        } catch (...) {
            release_owned(p.owns);
            throw;
        }
    } catch (const DspUnavailable& e) {
        if (r.handle || r.ref) refuse_zero_copy_fallback(e, "DSP unavailable");
        if (!params.cpu_fallback) throw;
        warn_cpu_fallback(e.what());
        impl_->last_used_hw = false;
        cv::Mat out = fmtx::cpu_crop(as_pixels(src), r.fmt, x, y, width, height);
        if (dst_width != width || dst_height != height)
            out = fmtx::cpu_resize(out, r.fmt, dst_width, dst_height, "stretch",
                                   params.interpolation);
        return out;
    }
}

std::variant<std::vector<cv::Mat>, PendingDspJob> DspClient::multi_crop_hw(
    const DspJobSource& src, const std::vector<DspWindow>& windows, std::string fmt,
    const DspJobParams& params, DspBufferPool* src_pool,
    const std::vector<DspBufferPool*>& dst_pools) {
    ResolvedSource r = resolve_source(src, fmt);
    if (windows.empty()) throw DspError("multi_crop needs at least one rect");
    if (windows.size() > static_cast<std::size_t>(wire::kMaxBatch))
        throw DspError(std::to_string(windows.size()) + " rects exceed daemon batch cap " +
                       std::to_string(wire::kMaxBatch));
    std::vector<fmtx::CropRect> rects;
    rects.reserve(windows.size());
    for (const DspWindow& w : windows)
        rects.push_back(fmtx::validated_rect(r.width, r.height, r.fmt, w.x, w.y, w.width,
                                             w.height, w.dst_width, w.dst_height));

    // one pool per distinct destination geometry, sized by multiplicity
    std::vector<std::pair<int, int>> order;
    std::map<std::pair<int, int>, int> per_geom;
    for (const auto& rc : rects) {
        const std::pair<int, int> geom{rc.dw, rc.dh};
        if (per_geom.find(geom) == per_geom.end()) order.push_back(geom);
        ++per_geom[geom];
    }
    std::vector<std::array<int, 3>> specs;
    for (const auto& geom : order) specs.push_back({geom.first, geom.second, per_geom[geom]});

    try {
        Prepared p = client_prep(*this, *impl_, src, r, specs, src_pool, dst_pools,
                                 params.timeout_s);
        try {
            std::map<std::pair<int, int>, int> slots;
            for (const auto& geom : order) slots[geom] = 0;
            std::vector<std::uint64_t> dst_ids;
            std::vector<std::array<std::int32_t, 6>> wire_rects;
            for (const auto& rc : rects) {
                const std::pair<int, int> geom{rc.dw, rc.dh};
                const int idx = static_cast<int>(std::find(order.begin(), order.end(), geom) -
                                                 order.begin());
                dst_ids.push_back(
                    p.pools[static_cast<std::size_t>(idx)]->buffer_id(slots[geom]++));
                wire_rects.push_back({rc.x, rc.y, rc.w, rc.h, rc.dw, rc.dh});
            }
            Impl::SubmitResult j = impl_->submit_job(wire::kOpMultiCrop, p.source.buffer_id(0),
                                                     dst_ids, wire_rects, params.interpolation,
                                                     params.scaling, params.priority,
                                                     params.timeout_s, params.wait);
            impl_->last_used_hw = true;
            if (!params.wait) {
                std::map<std::pair<int, int>, int> slot_of;
                for (const auto& geom : order) slot_of[geom] = 0;
                std::vector<std::pair<DspBufferPool*, int>> reads;
                for (const auto& rc : rects) {
                    const std::pair<int, int> geom{rc.dw, rc.dh};
                    const int idx =
                        static_cast<int>(std::find(order.begin(), order.end(), geom) -
                                         order.begin());
                    reads.emplace_back(p.pools[static_cast<std::size_t>(idx)],
                                       slot_of[geom]++);
                }
                return make_pending(this, std::move(reads), j.job_id, std::move(p.owns),
                                    params.timeout_s, true);
            }
            std::vector<cv::Mat> out;
            std::map<std::pair<int, int>, int> done;
            for (const auto& geom : order) done[geom] = 0;
            for (const auto& rc : rects) {
                const std::pair<int, int> geom{rc.dw, rc.dh};
                const int idx =
                    static_cast<int>(std::find(order.begin(), order.end(), geom) -
                                     order.begin());
                out.push_back(
                    p.pools[static_cast<std::size_t>(idx)]->read(done[geom]++));
            }
            release_owned(p.owns);
            return out;
        } catch (...) {
            release_owned(p.owns);
            throw;
        }
    } catch (const DspUnavailable& e) {
        if (r.handle || r.ref) refuse_zero_copy_fallback(e, "DSP unavailable");
        if (!params.cpu_fallback) throw;
        warn_cpu_fallback(e.what());
        impl_->last_used_hw = false;
        std::vector<cv::Mat> out;
        for (const auto& rc : rects) out.push_back(fmtx::cpu_crop_resize(as_pixels(src), r.fmt, rc));
        return out;
    }
}

std::variant<cv::Mat, PendingDspJob, DspBufferRef> DspClient::convert_hw(
    const DspJobSource& src, const std::string& dst_fmt, std::string fmt,
    const DspJobParams& params, DspBufferPool* src_pool, DspBufferPool* dst_pool) {
    check_out_mode(params);
    ResolvedSource r = resolve_source(src, fmt);
    if (std::holds_alternative<Frame>(src) && std::get<Frame>(src).format == "BGR")
        std::fprintf(stderr,
                     "neoruntime_ipc_sdk: convert_hw: source frame is BGR — the DSP wire "
                     "only carries rgb24, so R and B come back swapped; use "
                     "frame.to_rgb() first or stay on the CPU leg\n");
    wire::validate_geometry(r.width, r.height, r.fmt, "source");
    wire::validate_geometry(r.width, r.height, dst_fmt, "destination");
    if (dst_fmt == r.fmt)
        throw DspError("CONVERT needs differing formats (src is '" + r.fmt +
                       "'); dimensions stay equal — resize_hw scales");
    if (src_pool && (src_pool->width != r.width || src_pool->height != r.height ||
                     src_pool->fmt != r.fmt))
        throw DspError("src_pool is " + std::to_string(src_pool->width) + "x" +
                       std::to_string(src_pool->height) + " " + src_pool->fmt +
                       ", source is " + std::to_string(r.width) + "x" +
                       std::to_string(r.height) + " " + r.fmt);
    if (dst_pool && (dst_pool->width != r.width || dst_pool->height != r.height ||
                     dst_pool->fmt != dst_fmt))
        throw DspError("dst_pool is " + std::to_string(dst_pool->width) + "x" +
                       std::to_string(dst_pool->height) + " " + dst_pool->fmt +
                       ", job needs " + std::to_string(r.width) + "x" +
                       std::to_string(r.height) + " " + dst_fmt + " (CONVERT keeps dims)");
    if ((r.fmt == "gray8" || dst_fmt == "gray8") && impl_->gray8_refused) {
        // the daemon already refused a gray8 pair this session (the firmware
        // has no gray8 leg) — skip the doomed submit and the repeat warning
        if (r.handle || r.ref)
            throw DspError(
                "gray8 conversions are firmware-refused on this device — refusing the "
                "silent CPU fallback for a zero-copy frame source; use frame.to_array() "
                "to accept the copy");
        impl_->last_used_hw = false;
        return fmtx::cpu_convert(as_pixels(src), r.fmt, dst_fmt);
    }
    try {
        // bespoke prep: _prep assumes one fmt for src AND dst pools, but
        // CONVERT needs the dst pool in dst_fmt at the source geometry
        std::vector<std::unique_ptr<Ownee>> owns;
        JobSourceHandle source;
        if (r.ref) {
            if (src_pool)
                throw DspError("src_pool applies to numpy sources; a DspBufferRef is "
                               "already a daemon buffer");
            source = JobSourceHandle{JobSourceHandle::Kind::kRef, nullptr, 0, r.ref};
        } else if (r.handle) {
            if (src_pool)
                throw DspError("src_pool applies to numpy sources; a frame handle imports "
                               "its own dma-bufs");
            auto own = std::make_unique<ImportOwnee>();
            own->id = impl_->import_source(*r.handle, r.width, r.height, r.fmt,
                                           params.timeout_s);
            own->send = [impl = impl_.get()](std::uint64_t id) { impl->send_release(id); };
            source =
                JobSourceHandle{JobSourceHandle::Kind::kImport, nullptr, own->id, nullptr};
            owns.push_back(std::move(own));
        } else {
            DspBufferPool* pool = src_pool;
            if (!pool) {
                auto owned = std::make_unique<PoolOwnee>();
                owned->pool = alloc_buffers(r.width, r.height, r.fmt, 1);
                pool = owned->pool.get();
                owns.push_back(std::move(owned));
            }
            pool->write(0, r.pixels.empty() ? as_pixels(src) : r.pixels);
            source = JobSourceHandle{JobSourceHandle::Kind::kPool, pool, 0, nullptr};
        }
        DspBufferPool* out_pool = dst_pool;
        if (!out_pool) {
            auto owned = std::make_unique<PoolOwnee>();
            owned->pool = alloc_buffers(r.width, r.height, dst_fmt, 1);
            out_pool = owned->pool.get();
            owns.push_back(std::move(owned));
        }
        try {
            Impl::SubmitResult j;
            try {
                j = impl_->submit_job(wire::kOpConvertFormat, source.buffer_id(0),
                                      {out_pool->buffer_id(0)},
                                      {},  // CONVERT takes no rects
                                      "bilinear", "stretch", params.priority,
                                      params.timeout_s, params.wait);
            } catch (const DspError& e) {
                // the daemon took the job but the hardware refused it — on
                // hailo15 firmware every gray8 pair lands here
                if (r.fmt == "gray8" || dst_fmt == "gray8") impl_->gray8_refused = true;
                if (r.handle || r.ref)
                    throw DspError(
                        std::string(
                            "DSP rejected the conversion with a zero-copy frame source — "
                            "refusing the silent CPU fallback (the frame holds fds, not "
                            "pixels; use frame.to_array() to accept the copy)"));
                if (!params.cpu_fallback) throw;
                warn("DSP rejected " + r.fmt + "->" + dst_fmt + " (" + e.what() +
                     "); CPU fallback engaged (client.last_used_hw records the path used)");
                impl_->last_used_hw = false;
                release_owned(owns);
                return fmtx::cpu_convert(as_pixels(src), r.fmt, dst_fmt);
            }
            impl_->last_used_hw = true;
            if (!params.wait)
                return make_pending(this, {{out_pool, 0}}, j.job_id, std::move(owns),
                                    params.timeout_s, false);
            if (params.out_ref) return make_ref(out_pool, std::move(owns));
            cv::Mat out = out_pool->read(0);
            release_owned(owns);
            return out;
        } catch (...) {
            release_owned(owns);
            throw;
        }
    } catch (const DspUnavailable& e) {
        if (r.handle || r.ref) refuse_zero_copy_fallback(e, "DSP unavailable");
        if (!params.cpu_fallback) throw;
        warn_cpu_fallback(e.what());
        impl_->last_used_hw = false;
        return fmtx::cpu_convert(as_pixels(src), r.fmt, dst_fmt);
    }
}

std::variant<cv::Mat, PendingDspJob, DspBufferRef, std::monostate> DspClient::blend_hw(
    const DspJobSource& base, const std::vector<DspOverlay>& overlays, std::string fmt,
    const DspBlendParams& params) {
    if (!fmt.empty() && fmt != "nv12")
        throw DspError("BLEND base must be nv12 (daemon contract), got '" + fmt + "'");
    check_out_mode(params);
    // the base is nv12 by definition — never leave it to shape inference (a
    // 2D nv12 array would ambiguously infer gray8)
    ResolvedSource r = resolve_source(base, "nv12");
    if ((r.handle || r.ref) && !params.zero_copy)
        throw DspError(
            "blend_hw refuses device-side bases (keep-fd frame/handle or DspBufferRef) "
            "by default: the import->resize->blend chain has wedged the DSP device-wide "
            "until reboot in the field (state-dependent, root cause open). Pass "
            "frame.to_array()/ref.read() — the array path is the proven one — or "
            "zero_copy=true to force the chain at your own risk.");
    wire::validate_geometry(r.width, r.height, r.fmt, "base");
    DspBufferPool* dst_pool = params.dst_pool;
    if (dst_pool) {
        if (dst_pool->width != r.width || dst_pool->height != r.height ||
            dst_pool->fmt != "nv12")
            throw DspError("dst_pool is " + std::to_string(dst_pool->width) + "x" +
                           std::to_string(dst_pool->height) + " " + dst_pool->fmt +
                           ", job needs " + std::to_string(r.width) + "x" +
                           std::to_string(r.height) + " nv12 (blend keeps dims)");
        if (params.dst_slot < 0 || params.dst_slot >= dst_pool->count())
            throw DspError("dst_slot " + std::to_string(params.dst_slot) +
                           " outside the dst_pool's " + std::to_string(dst_pool->count()) +
                           " slots");
    }
    if (overlays.empty()) throw DspError("BLEND needs at least one overlay");
    if (overlays.size() > static_cast<std::size_t>(wire::kMaxBatch))
        throw DspError("too many overlays (" + std::to_string(overlays.size()) +
                       "); max is " + std::to_string(wire::kMaxBatch));

    // validate + pad overlays before any wire work: the daemon floor is
    // 16x16 and the DSP blend rejects odd overlay heights (-2801). Both
    // pads add fully transparent pixels — semantic no-ops — but they grow
    // the rect, so bounds are enforced on the padded dims.
    std::vector<fmtx::RgbaOverlay> prepared;
    for (std::size_t i = 0; i < overlays.size(); ++i) {
        cv::Mat rgba = overlays[i].rgba;
        const int x = overlays[i].x, y = overlays[i].y;
        if (rgba.type() != CV_8UC4 || rgba.dims != 2)
            throw DspError("overlay " + std::to_string(i) +
                           " must be an (h, w, 4) uint8 rgba array, got " +
                           describe_mat(rgba));
        int oh = rgba.rows, ow = rgba.cols;
        bool padded = false;
        if (ow < wire::kMinDim || oh < wire::kMinDim) {
            cv::Mat canvas(std::max(oh, wire::kMinDim), std::max(ow, wire::kMinDim), CV_8UC4,
                           cv::Scalar(0, 0, 0, 0));
            rgba(cv::Rect(0, 0, ow, oh)).copyTo(canvas(cv::Rect(0, 0, ow, oh)));
            rgba = canvas;
            oh = rgba.rows;
            ow = rgba.cols;
            padded = true;
        }
        if (oh % 2) {
            // one transparent bottom row; composites as a no-op. Odd
            // width/x/y are fine on-device — only height is not.
            cv::Mat canvas(oh + 1, ow, CV_8UC4, cv::Scalar(0, 0, 0, 0));
            rgba.copyTo(canvas(cv::Rect(0, 0, ow, oh)));
            rgba = canvas;
            oh += 1;
            padded = true;
        }
        if (x < 0 || y < 0 || x + ow > r.width || y + oh > r.height)
            throw DspError("overlay " + std::to_string(i) + " (" + std::to_string(ow) + "x" +
                           std::to_string(oh) + " at (" + std::to_string(x) + "," +
                           std::to_string(y) + ")) exceeds the " + std::to_string(r.width) +
                           "x" + std::to_string(r.height) + " base" +
                           (padded ? " (includes transparent padding)" : "") +
                           " — clamp or clip before blending");
        wire::validate_geometry(ow, oh, "argb", "overlay " + std::to_string(i));
        prepared.push_back({rgba, x, y});
    }

    std::vector<std::unique_ptr<Ownee>> owns;
    try {
        DspBufferPool* base_pool = dst_pool;
        int base_slot = params.dst_slot;
        if (!base_pool) {
            auto owned = std::make_unique<PoolOwnee>();
            owned->pool = alloc_buffers(r.width, r.height, "nv12", 1);
            base_pool = owned->pool.get();
            base_slot = 0;
            owns.push_back(std::move(owned));
        }
        try {
            if (r.handle || r.ref) {
                // device-side base: a DspBufferRef rides its daemon id
                // directly (no import, borrowed); a keep-fd frame is
                // imported zero-copy. Either way the DSP copies it into the
                // pool with a 1:1 RESIZE — the blend then composites in
                // place on that copy (imported BLEND *destinations* are
                // rejected and camera dma-bufs must never be written).
                JobSourceHandle src;
                if (r.ref) {
                    src = JobSourceHandle{JobSourceHandle::Kind::kRef, nullptr, 0, r.ref};
                } else {
                    auto own = std::make_unique<ImportOwnee>();
                    own->id = impl_->import_source(*r.handle, r.width, r.height, r.fmt,
                                                   params.timeout_s);
                    own->send = [impl = impl_.get()](std::uint64_t id) {
                        impl->send_release(id);
                    };
                    src = JobSourceHandle{JobSourceHandle::Kind::kImport, nullptr, own->id,
                                          nullptr};
                    owns.push_back(std::move(own));
                }
                impl_->submit_job(wire::kOpResize, src.buffer_id(0),
                                  {base_pool->buffer_id(base_slot)}, {}, "bilinear",
                                  "stretch", params.priority, params.timeout_s, true);
            } else {
                base_pool->write(base_slot, r.pixels.empty() ? as_pixels(base) : r.pixels);
            }
            // overlays travel as memfd imports, not pool allocs (deployed
            // HALs reject ARGB32 pool allocation; the daemon maps the memfd
            // as USERPTR planes). Hardware ARGB32 is [A, R, G, B].
            std::vector<std::uint64_t> ov_ids;
            for (const auto& ov : prepared) {
                cv::Mat wire_mat = rgba_to_argb_wire(ov.rgba);
                const int ow = ov.rgba.cols, oh = ov.rgba.rows;
                auto own = std::make_unique<ImportOwnee>();
                own->id = impl_->import_memfd(
                    wire_mat.data, static_cast<std::size_t>(oh) * ow * 4, ow, oh, "argb",
                    static_cast<std::uint32_t>(ow) * 4, params.timeout_s);
                own->send = [impl = impl_.get()](std::uint64_t id) {
                    impl->send_release(id);
                };
                ov_ids.push_back(own->id);
                owns.push_back(std::move(own));
            }
            std::vector<std::array<std::int32_t, 6>> rects;
            for (const auto& ov : prepared)
                rects.push_back({ov.x, ov.y, ov.rgba.cols, ov.rgba.rows, ov.rgba.cols,
                                 ov.rgba.rows});
            try {
                Impl::SubmitResult j = impl_->submit_job(
                    wire::kOpBlend, base_pool->buffer_id(base_slot), ov_ids, rects,
                    "bilinear", "stretch", params.priority, params.timeout_s, params.wait);
                impl_->last_used_hw = true;
                if (!params.wait)
                    return make_pending(this, {{base_pool, base_slot}}, j.job_id,
                                        std::move(owns), params.timeout_s, false);
                if (dst_pool) return std::monostate{};  // pixels ARE the caller's slot
                if (params.out_ref) return make_ref(base_pool, std::move(owns));
                cv::Mat out = base_pool->read(base_slot);
                release_owned(owns);
                return out;
            } catch (const DspError& e) {
                if (r.handle || r.ref)
                    throw DspError(
                        std::string("DSP rejected the blend with a device-resident base — "
                                    "refusing the silent CPU fallback (the base holds a "
                                    "buffer id, not pixels; use frame.to_array()/ref.read() "
                                    "to accept the copy)"));
                if (!params.cpu_fallback) throw;
                warn("DSP rejected the blend (" + std::string(e.what()) +
                     "); CPU fallback engaged (client.last_used_hw records the path used)");
                impl_->last_used_hw = false;
                cv::Mat blended = fmtx::cpu_blend(
                    r.pixels.empty() ? as_pixels(base) : r.pixels, r.fmt, prepared);
                if (dst_pool) {
                    dst_pool->write(params.dst_slot, blended);
                    release_owned(owns);
                    return std::monostate{};
                }
                release_owned(owns);
                return blended;
            }
        } catch (...) {
            release_owned(owns);
            throw;
        }
    } catch (const DspUnavailable& e) {
        if (r.handle || r.ref) refuse_zero_copy_fallback(e, "DSP unavailable");
        if (!params.cpu_fallback) throw;
        warn_cpu_fallback(e.what());
        impl_->last_used_hw = false;
        cv::Mat blended =
            fmtx::cpu_blend(r.pixels.empty() ? as_pixels(base) : r.pixels, r.fmt, prepared);
        if (dst_pool) {
            dst_pool->write(params.dst_slot, blended);
            return std::monostate{};
        }
        return blended;
    }
}

std::vector<std::uint8_t> DspClient::encode_jpeg_hw(const DspJobSource* src, int quality,
                                                    std::string fmt, double timeout_s,
                                                    DspBufferPool* src_pool,
                                                    bool cpu_fallback,
                                                    std::uint64_t src_buffer_id) {
    if (src_buffer_id != 0) {
        if (src)
            throw DspError("pass either src or src_buffer_id, not both");
        if (quality < 1 || quality > 100)
            throw DspError("quality must be 1..100, got " + std::to_string(quality));
        try {
            std::vector<std::uint8_t> jpeg =
                impl_->encode_rpc(src_buffer_id, quality, timeout_s);
            impl_->last_used_hw = true;
            return jpeg;
        } catch (const DspUnavailable&) {
            throw DspError(
                "EncodeImage unavailable for a daemon-side buffer — no client pixels to "
                "fall back on; pass the array instead");
        }
    }
    ResolvedSource r = resolve_source(*src, fmt);
    wire::validate_geometry(r.width, r.height, r.fmt, "source");
    if (quality < 1 || quality > 100)
        throw DspError("quality must be 1..100, got " + std::to_string(quality));
    cv::Mat pixels = r.pixels;
    if (r.fmt == "gray8") {
        if (r.handle || r.ref)
            throw DspError(
                "gray8 frames cannot ride the hardware jpeg encoder without a copy (the "
                "daemon feeds nv12/rgb24); use frame.to_array() (or ref.read()) and pass "
                "the array");
        // no hardware gray leg: replicate to rgb24 (R=G=B=gray) and let the
        // daemon's DSP convert take it from there
        pixels = fmtx::cpu_convert(as_pixels(*src), "gray8", "rgb24");
        r.fmt = "rgb24";
    }
    if (src_pool && (src_pool->width != r.width || src_pool->height != r.height ||
                     src_pool->fmt != r.fmt))
        throw DspError("src_pool is " + std::to_string(src_pool->width) + "x" +
                       std::to_string(src_pool->height) + " " + src_pool->fmt +
                       ", source is " + std::to_string(r.width) + "x" +
                       std::to_string(r.height) + " " + r.fmt);
    try {
        std::vector<std::unique_ptr<Ownee>> owns;
        JobSourceHandle source;
        if (r.ref) {
            if (src_pool)
                throw DspError("src_pool applies to numpy sources; a DspBufferRef is "
                               "already a daemon buffer");
            source = JobSourceHandle{JobSourceHandle::Kind::kRef, nullptr, 0, r.ref};
        } else if (r.handle) {
            if (src_pool)
                throw DspError("src_pool applies to numpy sources; a frame handle imports "
                               "its own dma-bufs");
            auto own = std::make_unique<ImportOwnee>();
            own->id = impl_->import_source(*r.handle, r.width, r.height, r.fmt, timeout_s);
            own->send = [impl = impl_.get()](std::uint64_t id) { impl->send_release(id); };
            source =
                JobSourceHandle{JobSourceHandle::Kind::kImport, nullptr, own->id, nullptr};
            owns.push_back(std::move(own));
        } else {
            DspBufferPool* pool = src_pool;
            if (!pool) {
                auto owned = std::make_unique<PoolOwnee>();
                owned->pool = alloc_buffers(r.width, r.height, r.fmt, 1);
                pool = owned->pool.get();
                owns.push_back(std::move(owned));
            }
            pool->write(0, pixels.empty() ? as_pixels(*src) : pixels);
            source = JobSourceHandle{JobSourceHandle::Kind::kPool, pool, 0, nullptr};
        }
        try {
            std::vector<std::uint8_t> jpeg =
                impl_->encode_rpc(source.buffer_id(0), quality, timeout_s);
            impl_->last_used_hw = true;
            release_owned(owns);
            return jpeg;
        } catch (...) {
            release_owned(owns);
            throw;
        }
    } catch (const DspUnavailable& e) {
        if (r.handle || r.ref) refuse_zero_copy_fallback(e, "EncodeImage unavailable");
        if (!cpu_fallback) throw;
        warn_cpu_fallback(e.what());
        impl_->last_used_hw = false;
        cv::Mat px = pixels.empty() ? as_pixels(*src) : pixels;
        if (r.fmt != "rgb24") px = fmtx::cpu_convert(px, r.fmt, "rgb24");
        return encode_jpeg_cpu(px, quality);
    }
}

// ---- prep (file-local; needs alloc_buffers, hence this placement) -----------------

static Prepared client_prep(DspClient& self, DspClient::Impl& impl, const DspJobSource& src,
                            ResolvedSource& r, const std::vector<std::array<int, 3>>& dst_specs,
                            DspBufferPool* src_pool,
                            const std::vector<DspBufferPool*>& dst_pools, double timeout_s) {
    wire::validate_geometry(r.width, r.height, r.fmt, "source");
    for (const auto& spec : dst_specs)
        wire::validate_geometry(spec[0], spec[1], r.fmt, "destination");
    Prepared out;
    if (r.ref) {
        if (src_pool)
            throw DspError("src_pool applies to numpy sources; a DspBufferRef is already a "
                           "daemon buffer");
        out.source = JobSourceHandle{JobSourceHandle::Kind::kRef, nullptr, 0, r.ref};
    } else if (r.handle) {
        if (src_pool)
            throw DspError("src_pool applies to numpy sources; a frame handle imports its "
                           "own dma-bufs");
        auto own = std::make_unique<ImportOwnee>();
        own->id = impl.import_source(*r.handle, r.width, r.height, r.fmt, timeout_s);
        DspClient::Impl* implp = &impl;  // value-capture: the lambda outlives this frame
        own->send = [implp](std::uint64_t id) { implp->send_release(id); };
        out.source = JobSourceHandle{JobSourceHandle::Kind::kImport, nullptr, own->id,
                                     nullptr};
        out.owns.push_back(std::move(own));
    } else {
        DspBufferPool* pool = src_pool;
        if (!pool) {
            auto owned = std::make_unique<PoolOwnee>();
            owned->pool = self.alloc_buffers(r.width, r.height, r.fmt, 1);
            pool = owned->pool.get();
            out.owns.push_back(std::move(owned));
        }
        pool->write(0, r.pixels.empty() ? as_pixels(src) : r.pixels);
        out.source = JobSourceHandle{JobSourceHandle::Kind::kPool, pool, 0, nullptr};
    }
    if (dst_pools.empty()) {
        for (const auto& spec : dst_specs) {
            auto owned = std::make_unique<PoolOwnee>();
            owned->pool = self.alloc_buffers(spec[0], spec[1], r.fmt, spec[2]);
            out.pools.push_back(owned->pool.get());
            out.owns.push_back(std::move(owned));
        }
    } else {
        out.pools = dst_pools;
    }
    return out;
}

}  // namespace neoruntime_ipc_sdk

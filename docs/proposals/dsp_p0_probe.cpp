/**
 * @file dsp_p0_probe.cpp
 * @brief P0 experiments for the HAL DSP context-model decision
 *        (multi-context vs single-context multiplexed by the daemon).
 *
 * Static source analysis of libhailodsp 1.12.0 predicts:
 *   - dsp_create_device == open("/dev/dsp0", O_RDWR) with no O_EXCL
 *     -> multiple contexts should be creatable.
 *   - ALL ops funnel through PriorityQueueSingleton (one thread per process)
 *     -> in-process multi-context should show ZERO parallel speedup.
 *
 * Experiments (run on the real device, daemons running):
 *   e1   double-init: create two HAL contexts in-process, verify both work,
 *        count /dev/dsp0 fds in /proc/self/fd.
 *   e2a  in-process concurrency: serial N ops on ctx1 vs N/2+N/2 concurrent
 *        ops on ctx1/ctx2 (two threads). Also same-ctx two-thread variant.
 *   e2b  cross-process contention: hammer resize for K seconds, sample
 *        dsp utilization out-of-band via a private vendor device handle.
 *   e3   single-context throughput: sync loop vs async submit/wait pipeline
 *        (depth D), latency percentiles.
 *   e4   HAL-1/2/3 validation (2026-09-01):
 *        A: multi_crop_and_resize via HAL ops table, N=1/2/4/7/16/64. N>7
 *           exercises the hal_v2 wrapper fix (2026-09-01): dynamic storage
 *           sized to output_count, batches above HAL_DSP_MULTI_CROP_MAX_OUTPUTS
 *           (128) rejected with HAL_ERR_INVALID_ARG instead of truncated.
 *           Before the fix the wrapper used fixed stack storage of 7 with an
 *           unclamped count (OOB read inside the vendor lib for N>7).
 *        B: vendor-direct dsp_multi_crop_and_resize, N=1/7/16/64/260 (vendor
 *           documents max 260 crops per job) — per-rect marginal cost curve.
 *        C: blend. Base must be NV12 (only supported format), overlays only
 *           A420/ARGB (alpha from the overlay's alpha channel), base is
 *           modified in place. Verifies alpha=0 passthrough / alpha=255
 *           blend / outside untouched, then 1-overlay vs 8-overlay timing.
 *        All sub-tests honor --mem, so `--mem dmabuf --mode e4` doubles as
 *        the HAL_MEM_DMABUF path validation (HAL-3).
 *
 * Usage:
 *   dsp_p0_probe --mode all|e1|e2a|e2b|e3|e4 [--w 1920] [--h 1080]
 *                [--ow 640] [--oh 360] [--iters 120] [--seconds 10]
 *                [--depth 4] [--mem auto|userptr|dmabuf]
 *
 * Build (poky cross):
 *   aarch64-poky-linux-g++ -std=c++17 -O2 -pthread \
 *     -I<hal_v2>/include --sysroot=<poky sysroot> \
 *     dsp_p0_probe.cpp <hal_v2>/platforms/hailo15/dsp/hailo15_dsp_impl.cpp \
 *     -o dsp_p0_probe -lhailodsp
 */

#include "dsp/hal_dsp.h"

#include <hailo/hailodsp.h>

/* Declared in vendor-internal utilization.hpp only (not in public headers);
 * symbol is C++-mangled in libhailodsp, so this redeclaration links. */
dsp_status dsp_get_utilization(dsp_device device, uint32_t &utilization);

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <mutex>
#include <numeric>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{

/* ------------------------------------------------------------------ */
/* timing helpers                                                      */
/* ------------------------------------------------------------------ */

double now_us()
{
    using namespace std::chrono;
    return duration_cast<duration<double, std::micro>>(
               steady_clock::now().time_since_epoch())
        .count();
}

struct LatencyStats
{
    size_t count = 0;
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double max_us = 0.0;
};

LatencyStats summarize(std::vector<double> v)
{
    LatencyStats s;
    s.count = v.size();
    if (v.empty()) {
        return s;
    }
    std::sort(v.begin(), v.end());
    auto pick = [&](double q) {
        size_t idx = std::min(v.size() - 1,
                              static_cast<size_t>(q * (v.size() - 1)));
        return v[idx];
    };
    s.mean_us = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    s.p50_us = pick(0.50);
    s.p95_us = pick(0.95);
    s.p99_us = pick(0.99);
    s.max_us = v.back();
    return s;
}

void print_stats(const char *label, const LatencyStats &s, double mpix_src)
{
    std::printf("    %-22s n=%zu mean=%.0fus p50=%.0fus p95=%.0fus "
                "p99=%.0fus max=%.0fus\n",
                label, s.count, s.mean_us, s.p50_us, s.p95_us, s.p99_us,
                s.max_us);
    if (mpix_src > 0.0 && s.mean_us > 0.0) {
        std::printf("    -> throughput=%.1f ops/s (%.2f MPix/s src)\n",
                    1e6 / s.mean_us, mpix_src / s.mean_us);
    }
}

/* ------------------------------------------------------------------ */
/* NV12 buffers: userptr (malloc) or dmabuf (CMA dma-heap)             */
/* ------------------------------------------------------------------ */

struct Nv12Buffer
{
    HalFrameBuffer fb{};
    std::vector<unsigned char> heap; /* userptr backing            */
    int dma_fds[HAL_MAX_PLANES] = {-1, -1, -1};
    void *mapped[HAL_MAX_PLANES] = {nullptr, nullptr, nullptr};
    size_t mapped_len[HAL_MAX_PLANES] = {0, 0, 0};
    bool using_dmabuf = false;
};

int dma_heap_alloc(size_t len, int *fd_out, void **map_out, size_t *len_out)
{
    int heap_fd = open("/dev/dma_heap/linux,cma", O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        return -1;
    }
    struct dma_heap_allocation_data data{};
    data.len = len;
    data.fd_flags = O_RDWR | O_CLOEXEC;
    int ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data);
    close(heap_fd);
    if (ret < 0) {
        return -1;
    }
    void *mem = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, data.fd, 0);
    if (mem == MAP_FAILED) {
        close(data.fd);
        return -1;
    }
    *fd_out = (int)data.fd;
    *map_out = mem;
    *len_out = len;
    return 0;
}

void sync_dmabuf_write(const HalFrameBuffer *fb); /* e4 helper, below */

void fill_nv12_pattern(HalFrameBuffer *fb)
{
    const uint32_t w = fb->width;
    const uint32_t h = fb->height;
    unsigned char *y = static_cast<unsigned char *>(fb->planes[0]);
    unsigned char *uv = static_cast<unsigned char *>(fb->planes[1]);
    for (uint32_t r = 0; r < h; ++r) {
        unsigned char *row = y + (size_t)r * fb->strides[0];
        for (uint32_t c = 0; c < w; ++c) {
            row[c] = static_cast<unsigned char>((r + c) & 0xFF);
        }
    }
    for (uint32_t r = 0; r < h / 2; ++r) {
        unsigned char *row = uv + (size_t)r * fb->strides[1];
        for (uint32_t c = 0; c < w; c += 2) {
            row[c] = static_cast<unsigned char>(((c / 2) + r) & 0xFF);
            row[c + 1] = 128;
        }
    }
    sync_dmabuf_write(fb); /* flush CPU writes before any DSP read */
}

bool alloc_nv12(Nv12Buffer *b, uint32_t w, uint32_t h, bool use_dmabuf)
{
    std::memset(b, 0, sizeof(*b));
    const size_t y_size = (size_t)w * h;
    const size_t uv_size = y_size / 2;

    b->fb.width = w;
    b->fb.height = h;
    b->fb.format = HAL_PIX_FMT_NV12;
    b->fb.num_planes = 2;
    b->fb.strides[0] = w;
    b->fb.strides[1] = w;
    b->fb.sizes[0] = (uint32_t)y_size;
    b->fb.sizes[1] = (uint32_t)uv_size;
    for (int i = 0; i < HAL_MAX_PLANES; ++i) {
        b->fb.dma_fds[i] = -1;
        b->dma_fds[i] = -1;
    }
    b->using_dmabuf = use_dmabuf;

    if (!use_dmabuf) {
        b->heap.resize(y_size + uv_size + 4096);
        uintptr_t addr = reinterpret_cast<uintptr_t>(b->heap.data());
        addr = (addr + 4095) & ~uintptr_t(4095);
        b->fb.planes[0] = reinterpret_cast<void *>(addr);
        b->fb.planes[1] = reinterpret_cast<void *>(addr + y_size);
        b->fb.mem_type = HAL_MEM_MALLOC;
        return true;
    }
    if (dma_heap_alloc(y_size, &b->dma_fds[0], &b->mapped[0],
                       &b->mapped_len[0]) != 0 ||
        dma_heap_alloc(uv_size, &b->dma_fds[1], &b->mapped[1],
                       &b->mapped_len[1]) != 0) {
        return false;
    }
    b->fb.dma_fds[0] = b->dma_fds[0];
    b->fb.dma_fds[1] = b->dma_fds[1];
    b->fb.planes[0] = b->mapped[0];
    b->fb.planes[1] = b->mapped[1];
    b->fb.mem_type = HAL_MEM_DMABUF;
    return true;
}

void free_nv12(Nv12Buffer *b)
{
    for (int i = 0; i < HAL_MAX_PLANES; ++i) {
        if (b->mapped[i]) {
            munmap(b->mapped[i], b->mapped_len[i]);
            b->mapped[i] = nullptr;
        }
        if (b->dma_fds[i] >= 0) {
            close(b->dma_fds[i]);
            b->dma_fds[i] = -1;
        }
    }
}

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

int do_resize(void *ctx, const HalFrameBuffer *src, HalFrameBuffer *dst,
              double *lat_us)
{
    HalDspResizeParams p{};
    p.src = src;
    p.dst = dst;
    p.interpolation = HAL_DSP_INTERPOLATION_BILINEAR;
    const double t0 = now_us();
    const int rc = HAL_DSP_OPS.resize(ctx, &p);
    if (lat_us) {
        *lat_us = now_us() - t0;
    }
    return rc;
}

int probe_resize_once(void *ctx, Nv12Buffer *src, Nv12Buffer *dst,
                      const char *tag)
{
    double lat = 0.0;
    const int rc = do_resize(ctx, &src->fb, &dst->fb, &lat);
    if (rc != 0) {
        std::printf("    RESIZE FAILED (%s) rc=%d\n", tag, rc);
        return rc;
    }
    const unsigned char *y =
        static_cast<const unsigned char *>(dst->fb.planes[0]);
    uint32_t acc = 0;
    for (size_t i = 0; i < dst->fb.sizes[0]; i += 1024) {
        acc += y[i];
    }
    std::printf("    resize ok (%s) lat=%.0fus dst_y_checksum=%u\n", tag, lat,
                acc);
    return 0;
}

int count_dsp0_fds()
{
    int n = 0;
    for (int fd = 0; fd < 256; ++fd) {
        char link[64];
        char target[256];
        std::snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
        ssize_t len = readlink(link, target, sizeof(target) - 1);
        if (len > 0) {
            target[len] = '\0';
            if (std::strstr(target, "dsp0")) {
                ++n;
            }
        }
    }
    return n;
}

/* private vendor handle for out-of-band utilization sampling */
struct VendorDev
{
    dsp_device dev = nullptr;
    bool ok = false;
    VendorDev() : ok(dsp_create_device(&dev) == DSP_SUCCESS) {}
    ~VendorDev()
    {
        if (ok) {
            dsp_release_device(dev);
        }
    }
};

int errors = 0;

void resize_loop(void *ctx, const HalFrameBuffer *src, HalFrameBuffer *dst,
                 uint32_t iters, std::vector<double> *out,
                 std::atomic<bool> *go)
{
    out->reserve(iters);
    while (!go->load()) {
        std::this_thread::yield();
    }
    for (uint32_t i = 0; i < iters; ++i) {
        double lat = 0.0;
        if (do_resize(ctx, src, dst, &lat) != 0) {
            ++errors;
        }
        out->push_back(lat);
    }
}

/* ------------------------------------------------------------------ */
/* experiments                                                         */
/* ------------------------------------------------------------------ */

struct ProbeCfg
{
    uint32_t w = 1920;
    uint32_t h = 1080;
    uint32_t ow = 640;
    uint32_t oh = 360;
    uint32_t iters = 120;
    uint32_t seconds = 10;
    uint32_t depth = 4;
    std::string mem = "auto";
};

int run_e1(Nv12Buffer *src, Nv12Buffer *dst)
{
    std::printf("\n[E1] double-init (in-process 2nd context; camera-daemon "
                "already holds one cross-process)\n");
    void *ctx1 = nullptr;
    void *ctx2 = nullptr;
    HalDspConfig c{};

    int rc = HAL_DSP_OPS.init(&c, &ctx1);
    std::printf("    init ctx1 rc=%d (%s)\n", rc, rc == 0 ? "OK" : "FAIL");
    if (rc != 0) {
        return rc;
    }
    rc = HAL_DSP_OPS.init(&c, &ctx2);
    std::printf("    init ctx2 rc=%d (%s)\n", rc, rc == 0 ? "OK" : "FAIL");
    std::printf("    /dev/dsp0 fds held by this process: %d\n",
                count_dsp0_fds());

    if (rc == 0) {
        probe_resize_once(ctx1, src, dst, "ctx1");
        probe_resize_once(ctx2, src, dst, "ctx2");
        HAL_DSP_OPS.deinit(ctx2);
    }
    HAL_DSP_OPS.deinit(ctx1);
    std::printf("    E1 RESULT: %s\n",
                rc == 0 ? "multi-context creatable + functional"
                        : "second context REFUSED");
    return 0;
}

int run_e2a(const ProbeCfg &cfg, Nv12Buffer *src, Nv12Buffer *dst)
{
    const double mpix = (double)cfg.w * cfg.h / 1e6;
    std::printf("\n[E2a] in-process concurrency (src %ux%u -> %ux%u, %u ops "
                "per phase)\n",
                cfg.w, cfg.h, cfg.ow, cfg.oh, cfg.iters);
    void *ctx1 = nullptr;
    void *ctx2 = nullptr;
    HalDspConfig c{};
    if (HAL_DSP_OPS.init(&c, &ctx1) != 0 || HAL_DSP_OPS.init(&c, &ctx2) != 0) {
        std::printf("    init failed\n");
        return -1;
    }

    /* separate dst per thread to avoid write races */
    Nv12Buffer dstA{}, dstB{};
    alloc_nv12(&dstA, cfg.ow, cfg.oh, src->using_dmabuf);
    alloc_nv12(&dstB, cfg.ow, cfg.oh, src->using_dmabuf);

    /* phase S: serial baseline on one thread */
    std::vector<double> latS;
    std::atomic<bool> goS(true);
    const double t0 = now_us();
    resize_loop(ctx1, &src->fb, &dstA.fb, cfg.iters, &latS, &goS);
    const double wallS = now_us() - t0;
    print_stats("serial 1ctx", summarize(latS), mpix);
    std::printf("    serial wall=%.0fus ops/s=%.1f\n", wallS,
                cfg.iters / (wallS / 1e6));

    /* phase P: two contexts, two threads, half the ops each */
    const uint32_t half = cfg.iters / 2;
    double wallP = 0.0;
    {
        std::atomic<bool> go(false);
        std::vector<double> latA, latB;
        std::thread thA(resize_loop, ctx1, &src->fb, &dstA.fb, half, &latA,
                        &go);
        std::thread thB(resize_loop, ctx2, &src->fb, &dstB.fb, half, &latB,
                        &go);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const double tp0 = now_us();
        go.store(true);
        thA.join();
        thB.join();
        wallP = now_us() - tp0;
        latA.insert(latA.end(), latB.begin(), latB.end());
        print_stats("2ctx 2thr", summarize(latA), mpix);
        std::printf("    parallel wall=%.0fus ops/s=%.1f speedup=%.2fx "
                    "(~1.0x == vendor singleton serializes)\n",
                    wallP, cfg.iters / (wallP / 1e6), wallS / wallP);
    }

    /* phase Q: ONE context, two threads (same-ctx sync safety) */
    {
        std::atomic<bool> go(false);
        std::vector<double> latA, latB;
        std::thread thC(resize_loop, ctx1, &src->fb, &dstA.fb, half, &latA,
                        &go);
        std::thread thD(resize_loop, ctx1, &src->fb, &dstB.fb, half, &latB,
                        &go);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const double tq0 = now_us();
        go.store(true);
        thC.join();
        thD.join();
        const double wallQ = now_us() - tq0;
        latA.insert(latA.end(), latB.begin(), latB.end());
        print_stats("1ctx 2thr", summarize(latA), mpix);
        std::printf("    wall=%.0fus ops/s=%.1f errors=%d\n", wallQ,
                    cfg.iters / (wallQ / 1e6), errors);
    }

    free_nv12(&dstA);
    free_nv12(&dstB);
    HAL_DSP_OPS.deinit(ctx2);
    HAL_DSP_OPS.deinit(ctx1);
    return 0;
}

int run_e2b(const ProbeCfg &cfg, Nv12Buffer *src, Nv12Buffer *dst)
{
    const double mpix = (double)cfg.w * cfg.h / 1e6;
    std::printf("\n[E2b] cross-process contention: hammer %us while sampling "
                "utilization (daemons keep running)\n",
                cfg.seconds);
    VendorDev vd;
    if (!vd.ok) {
        std::printf("    dsp_create_device for sampler FAILED\n");
        return -1;
    }
    void *ctx = nullptr;
    HalDspConfig c{};
    if (HAL_DSP_OPS.init(&c, &ctx) != 0) {
        return -1;
    }

    std::atomic<bool> stop_all(false);
    std::vector<double> lats;
    unsigned long long ops = 0;
    std::mutex lat_mtx;

    auto hammer = [&]() {
        std::vector<double> local;
        while (!stop_all.load()) {
            double lat = 0.0;
            if (do_resize(ctx, &src->fb, &dst->fb, &lat) == 0) {
                ++ops;
            } else {
                ++errors;
            }
            local.push_back(lat);
        }
        std::lock_guard<std::mutex> lk(lat_mtx);
        lats.insert(lats.end(), local.begin(), local.end());
    };

    std::vector<uint32_t> util_samples;
    std::mutex util_mtx;
    auto sampler = [&]() {
        while (!stop_all.load()) {
            uint32_t util = 0;
            if (dsp_get_utilization(vd.dev, util) == DSP_SUCCESS) {
                std::lock_guard<std::mutex> lk(util_mtx);
                util_samples.push_back(util);
            } else {
                std::lock_guard<std::mutex> lk(util_mtx);
                util_samples.push_back(UINT32_MAX);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    };

    const double t0 = now_us();
    std::thread hammer_th(hammer);
    std::thread sampler_th(sampler);
    std::this_thread::sleep_for(std::chrono::seconds(cfg.seconds));
    stop_all.store(true);
    hammer_th.join();
    sampler_th.join();
    const double wall = now_us() - t0;

    size_t fails = 0, ok = 0;
    unsigned long long acc = 0;
    uint32_t peak = 0;
    for (uint32_t x : util_samples) {
        if (x == UINT32_MAX) {
            ++fails;
        } else {
            ++ok;
            acc += x;
            peak = std::max(peak, x);
        }
    }
    std::printf("    ops=%llu wall=%.0fus ops/s=%.1f (%.2f MPix/s) "
                "resize_errors=%d\n",
                ops, wall, ops / (wall / 1e6), ops * mpix / (wall / 1e6),
                errors);
    print_stats("hammer lat", summarize(lats), mpix);
    std::printf("    util: samples=%zu ok=%zu failed=%zu avg=%u%% peak=%u%%\n",
                util_samples.size(), ok, fails,
                ok ? (uint32_t)(acc / ok) : 0, peak);
    HAL_DSP_OPS.deinit(ctx);
    return 0;
}

int run_e3(const ProbeCfg &cfg, Nv12Buffer *src, Nv12Buffer *dst)
{
    const double mpix = (double)cfg.w * cfg.h / 1e6;
    std::printf("\n[E3] single-context throughput: sync vs async pipeline "
                "(depth %u, %u jobs)\n",
                cfg.depth, cfg.iters);
    void *ctx = nullptr;
    HalDspConfig c{};
    if (HAL_DSP_OPS.init(&c, &ctx) != 0) {
        return -1;
    }

    /* sync loop */
    std::vector<double> lat_sync;
    std::atomic<bool> go(true);
    const double t0 = now_us();
    resize_loop(ctx, &src->fb, &dst->fb, cfg.iters, &lat_sync, &go);
    const double wall_sync = now_us() - t0;
    print_stats("sync loop", summarize(lat_sync), mpix);
    std::printf("    sync wall=%.0fus jobs/s=%.1f\n", wall_sync,
                cfg.iters / (wall_sync / 1e6));

    /* async pipeline */
    HalDspResizeParams p{};
    p.src = &src->fb;
    p.dst = &dst->fb;
    p.interpolation = HAL_DSP_INTERPOLATION_BILINEAR;

    std::vector<double> lat_async;
    std::vector<HalDspJobHandle> ring;
    std::vector<double> submit_t;
    uint32_t submitted = 0;
    uint32_t completed = 0;
    int submit_errs = 0;
    const double t1 = now_us();
    while (completed < cfg.iters) {
        while (ring.size() < cfg.depth && submitted < cfg.iters) {
            HalDspJobHandle job = nullptr;
            if (HAL_DSP_OPS.submit(ctx, HAL_DSP_OP_RESIZE, &p, &job) != 0) {
                ++submit_errs;
                ++submitted;
                continue;
            }
            ring.push_back(job);
            submit_t.push_back(now_us());
            ++submitted;
        }
        if (ring.empty()) {
            break; /* nothing in flight and nothing left to submit */
        }
        HalDspJobResult res{};
        if (HAL_DSP_OPS.wait(ctx, ring.front(), 1000, &res) != 0) {
            ++errors;
            HAL_DSP_OPS.job_release(ctx, ring.front());
            ring.erase(ring.begin());
            submit_t.erase(submit_t.begin());
            ++completed;
            continue;
        }
        if (res.status == HAL_DSP_JOB_PENDING) {
            continue; /* timeout while pending: keep pipeline full */
        }
        lat_async.push_back(now_us() - submit_t.front());
        if (res.status != HAL_DSP_JOB_COMPLETED) {
            ++errors;
        }
        HAL_DSP_OPS.job_release(ctx, ring.front());
        ring.erase(ring.begin());
        submit_t.erase(submit_t.begin());
        ++completed;
    }
    const double wall_async = now_us() - t1;
    print_stats("async pipe", summarize(lat_async), mpix);
    std::printf("    async wall=%.0fus jobs/s=%.1f (sync jobs/s=%.1f) "
                "submit_errs=%d\n",
                wall_async, cfg.iters / (wall_async / 1e6),
                cfg.iters / (wall_sync / 1e6), submit_errs);

    HAL_DSP_OPS.deinit(ctx);
    return 0;
}

/* ------------------------------------------------------------------ */
/* e4 helpers                                                          */
/* ------------------------------------------------------------------ */

/* CPU-read barrier for dmabuf-backed planes: without DMA_BUF_IOCTL_SYNC the
 * cache may serve stale lines after the DSP wrote via DMA. */
static void dmabuf_sync(const HalFrameBuffer *fb, uint64_t extra_flags)
{
#ifdef DMA_BUF_SYNC_START
    for (int i = 0; i < HAL_MAX_PLANES; ++i) {
        if (fb->dma_fds[i] < 0) {
            continue;
        }
        struct dma_buf_sync s{};
        s.flags = DMA_BUF_SYNC_START | extra_flags;
        ioctl(fb->dma_fds[i], DMA_BUF_IOCTL_SYNC, &s);
        s.flags = DMA_BUF_SYNC_END | extra_flags;
        ioctl(fb->dma_fds[i], DMA_BUF_IOCTL_SYNC, &s);
    }
#endif
}

void sync_dmabuf_read(const HalFrameBuffer *fb)
{
    dmabuf_sync(fb, DMA_BUF_SYNC_READ);
}

void sync_dmabuf_write(const HalFrameBuffer *fb)
{
    dmabuf_sync(fb, DMA_BUF_SYNC_WRITE);
}

uint32_t checksum_y(const HalFrameBuffer *fb)
{
    sync_dmabuf_read(fb);
    const unsigned char *y = static_cast<const unsigned char *>(fb->planes[0]);
    uint32_t acc = 0;
    for (uint32_t r = 0; r < fb->height; ++r) {
        const unsigned char *row = y + (size_t)r * fb->strides[0];
        for (uint32_t c = 0; c < fb->width; c += 17) {
            acc += row[c];
        }
    }
    return acc;
}

double mean_y_region(const HalFrameBuffer *fb, uint32_t x0, uint32_t y0,
                     uint32_t x1, uint32_t y1)
{
    sync_dmabuf_read(fb);
    const unsigned char *y = static_cast<const unsigned char *>(fb->planes[0]);
    unsigned long long acc = 0;
    size_t n = 0;
    for (uint32_t r = y0; r < y1; ++r) {
        const unsigned char *row = y + (size_t)r * fb->strides[0];
        for (uint32_t c = x0; c < x1; c += 3) {
            acc += row[c];
            ++n;
        }
    }
    return n ? (double)acc / (double)n : 0.0;
}

/* tile the source into n even-aligned ROIs on a near-square grid
 * (HalDspRoi and dsp_roi_t share field names) */
template <typename RoiT>
void make_roi_grid(RoiT *rois, uint32_t n, uint32_t w, uint32_t h)
{
    uint32_t cols = 1;
    while (cols * cols < n) {
        ++cols;
    }
    const uint32_t rows = (n + cols - 1) / cols;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t cx = i % cols;
        const uint32_t cy = i / cols;
        rois[i].start_x = (cx * (w / cols)) & ~1u;
        rois[i].end_x = (cx + 1 == cols) ? w : (((cx + 1) * (w / cols)) & ~1u);
        rois[i].start_y = (cy * (h / rows)) & ~1u;
        rois[i].end_y = (cy + 1 == rows || (cy + 1) * (h / rows) > h)
                            ? h
                            : (((cy + 1) * (h / rows)) & ~1u);
    }
}

/* ARGB32 overlay: left half alpha=0 (passthrough), right half alpha=255 */
struct ArgbBuffer
{
    HalFrameBuffer fb{};
    std::vector<unsigned char> heap;
};

void fill_argb_overlay(HalFrameBuffer *fb)
{
    const uint32_t w = fb->width;
    unsigned char *p = static_cast<unsigned char *>(fb->planes[0]);
    for (uint32_t r = 0; r < fb->height; ++r) {
        unsigned char *row = p + (size_t)r * fb->strides[0];
        for (uint32_t c = 0; c < w; ++c) {
            unsigned char *px = row + (size_t)c * 4;
            px[0] = (c < w / 2) ? 0 : 255; /* A */
            px[1] = 255;                   /* R */
            px[2] = 0;                     /* G */
            px[3] = 0;                     /* B */
        }
    }
}

/* vendor-side image view of an Nv12Buffer (for direct vendor calls) */
struct Vimg
{
    dsp_image_properties_t img{};
    dsp_data_plane_t pl[HAL_MAX_PLANES]{};
};

void nv12_to_vendor_image(const Nv12Buffer *b, Vimg *v)
{
    v->img.width = b->fb.width;
    v->img.height = b->fb.height;
    v->img.format = DSP_IMAGE_FORMAT_NV12;
    v->img.planes_count = 2;
    v->img.planes = v->pl;
    for (int i = 0; i < 2; ++i) {
        if (b->fb.mem_type == HAL_MEM_DMABUF) {
            v->pl[i].fd = b->fb.dma_fds[i];
        } else {
            v->pl[i].userptr = b->fb.planes[i];
        }
        v->pl[i].bytesperline = b->fb.strides[i];
        v->pl[i].bytesused = b->fb.sizes[i];
    }
    v->img.memory = (b->fb.mem_type == HAL_MEM_DMABUF) ? DSP_MEMORY_TYPE_DMABUF
                                                       : DSP_MEMORY_TYPE_USERPTR;
}

/* ------------------------------------------------------------------ */
/* e4: HAL-1/2/3 validation                                            */
/* ------------------------------------------------------------------ */

int run_e4(const ProbeCfg &cfg, Nv12Buffer *src)
{
    std::printf("\n[E4] HAL-1/2/3 validation: multi-crop capacity, blend, "
                "mem=%s\n",
                src->using_dmabuf ? "dmabuf" : "userptr");
    void *ctx = nullptr;
    HalDspConfig c{};
    if (HAL_DSP_OPS.init(&c, &ctx) != 0) {
        return -1;
    }
    VendorDev vd;
    if (!vd.ok) {
        std::printf("    vendor device for E4-B FAILED\n");
        HAL_DSP_OPS.deinit(ctx);
        return -1;
    }

    /* ---- E4-A: HAL ops-table multi_crop_and_resize, N <= 64 ---- */
    std::printf("\n[E4-A] HAL multi_crop_and_resize (ops table; wrapper "
                "dynamic storage, N>7 exercises the HAL-7 fix)\n");
    {
        const uint32_t Ns[] = {1, 2, 4, 7, 16, 64};
        double mean_1 = 0.0;
        for (uint32_t N : Ns) {
            std::vector<Nv12Buffer> dsts(N);
            bool ok = true;
            for (auto &d : dsts) {
                ok = ok && alloc_nv12(&d, cfg.ow, cfg.oh, src->using_dmabuf);
            }
            if (!ok) {
                std::printf("    N=%u dst alloc FAILED\n", N);
                HAL_DSP_OPS.deinit(ctx);
                return -1;
            }
            std::vector<HalDspRoi> rois(N);
            std::vector<HalDspMultiCropOutput> outs(N);
            make_roi_grid(rois.data(), N, cfg.w, cfg.h);
            for (uint32_t i = 0; i < N; ++i) {
                outs[i].crop = rois[i];
                outs[i].dst = &dsts[i].fb;
                outs[i].scaling_mode = HAL_DSP_SCALING_STRETCH;
                outs[i].letterbox_color = HalDspColor{};
            }
            HalDspMultiCropResizeParams p{};
            p.src = &src->fb;
            p.outputs = outs.data();
            p.output_count = N;
            p.interpolation = HAL_DSP_INTERPOLATION_BILINEAR;

            /* correctness: zero outputs, run twice, deterministic + nonzero */
            for (auto &d : dsts) {
                std::memset(d.fb.planes[0], 0, d.fb.sizes[0]);
                sync_dmabuf_write(&d.fb);
            }
            int rc = HAL_DSP_OPS.multi_crop_and_resize(ctx, &p);
            std::vector<uint32_t> cs(N, 0);
            for (uint32_t i = 0; i < N; ++i) {
                cs[i] = checksum_y(&dsts[i].fb);
            }
            rc = HAL_DSP_OPS.multi_crop_and_resize(ctx, &p);
            bool det = (rc == 0);
            bool nonzero = true;
            uint32_t maxdelta = 0;
            for (uint32_t i = 0; i < N; ++i) {
                const uint32_t c2 = checksum_y(&dsts[i].fb);
                if (c2 != cs[i]) {
                    det = false;
                    const uint32_t dd = (c2 > cs[i]) ? (c2 - cs[i])
                                                     : (cs[i] - c2);
                    if (dd > maxdelta) {
                        maxdelta = dd;
                    }
                }
                if (cs[i] == 0) {
                    nonzero = false;
                }
            }

            std::vector<double> lat;
            const uint32_t iters = (N >= 64) ? std::max(30u, cfg.iters / 2)
                                             : cfg.iters;
            lat.reserve(iters);
            for (uint32_t k = 0; k < iters; ++k) {
                const double t0 = now_us();
                if (HAL_DSP_OPS.multi_crop_and_resize(ctx, &p) != 0) {
                    ++errors;
                }
                lat.push_back(now_us() - t0);
            }
            const LatencyStats st = summarize(lat);
            if (N == 1) {
                mean_1 = st.mean_us;
            }
            std::printf(
                "    HAL N=%-3u mean=%7.0fus p95=%7.0fus rects/s=%8.1f "
                "marginal=%7.1fus/rect det=%d maxd=%u nonzero=%d rc=%d\n",
                N, st.mean_us, st.p95_us, (double)N * 1e6 / st.mean_us,
                (N > 1 && mean_1 > 0.0) ? (st.mean_us - mean_1) / (N - 1)
                                        : 0.0,
                det ? 1 : 0, maxdelta, nonzero ? 1 : 0, rc);
            for (auto &d : dsts) {
                free_nv12(&d);
            }
        }

        /* HAL-7 validation edge: a batch above HAL_DSP_MULTI_CROP_MAX_OUTPUTS
         * must be rejected, not truncated (rejection happens before any
         * outputs[] dereference, so no DSP work runs here). */
        {
            const uint32_t over = HAL_DSP_MULTI_CROP_MAX_OUTPUTS + 1;
            std::vector<Nv12Buffer> dsts(over);
            bool ok = true;
            for (auto &d : dsts) {
                ok = ok && alloc_nv12(&d, cfg.ow, cfg.oh, false);
            }
            if (!ok) {
                std::printf("    N=%u dst alloc FAILED\n", over);
            } else {
                std::vector<HalDspRoi> rois(over);
                std::vector<HalDspMultiCropOutput> outs(over);
                make_roi_grid(rois.data(), over, cfg.w, cfg.h);
                for (uint32_t i = 0; i < over; ++i) {
                    outs[i].crop = rois[i];
                    outs[i].dst = &dsts[i].fb;
                    outs[i].scaling_mode = HAL_DSP_SCALING_STRETCH;
                    outs[i].letterbox_color = HalDspColor{};
                }
                HalDspMultiCropResizeParams p{};
                p.src = &src->fb;
                p.outputs = outs.data();
                p.output_count = over;
                p.interpolation = HAL_DSP_INTERPOLATION_BILINEAR;
                const int rc = HAL_DSP_OPS.multi_crop_and_resize(ctx, &p);
                const bool rejected = (rc != 0);
                std::printf("    HAL N=%u (cap+1) rc=%d rejected=%d %s\n",
                            over, rc, rejected ? 1 : 0,
                            rejected ? "OK" : "FAIL");
                if (!rejected) {
                    ++errors;
                }
            }
            for (auto &d : dsts) {
                free_nv12(&d);
            }
        }
    }

    /* ---- E4-B: vendor-direct multi_crop_and_resize, N up to 260 ---- */
    std::printf("\n[E4-B] vendor-direct dsp_multi_crop_and_resize "
                "(heap arrays; dsts userptr to bound memory)\n");
    {
        Vimg s;
        nv12_to_vendor_image(src, &s);
        const uint32_t Ns[] = {1, 7, 16, 64, 128, 260};
        double mean_1 = 0.0;
        for (uint32_t N : Ns) {
            std::vector<Nv12Buffer> dsts(N);
            bool ok = true;
            for (auto &d : dsts) {
                ok = ok && alloc_nv12(&d, cfg.ow, cfg.oh, false);
            }
            if (!ok) {
                std::printf("    N=%u dst alloc FAILED\n", N);
                continue;
            }
            std::vector<Vimg> vimgs(N);
            std::vector<dsp_roi_t> rois(N);
            std::vector<dsp_crop_resize_params_t> cps(N);
            make_roi_grid(rois.data(), N, cfg.w, cfg.h);
            for (uint32_t i = 0; i < N; ++i) {
                nv12_to_vendor_image(&dsts[i], &vimgs[i]);
                cps[i].crop = &rois[i];
                for (uint32_t j = 0; j < DSP_MULTI_RESIZE_OUTPUTS_COUNT; ++j) {
                    cps[i].dst[j] = nullptr;
                }
                cps[i].dst[0] = &vimgs[i].img;
                cps[i].scaling_params[0].scaling_mode = DSP_SCALING_MODE_STRETCH;
            }
            dsp_multi_crop_resize_params_t m{};
            m.src = &s.img;
            m.crop_resize_params = cps.data();
            m.crop_resize_params_count = N;
            m.interpolation = INTERPOLATION_TYPE_BILINEAR;

            for (auto &d : dsts) {
                std::memset(d.fb.planes[0], 0, d.fb.sizes[0]);
                sync_dmabuf_write(&d.fb);
            }
            int rc = dsp_multi_crop_and_resize(vd.dev, &m);
            uint32_t zeros = 0;
            int first_zero = -1, last_zero = -1;
            for (uint32_t i = 0; i < N; ++i) {
                if (checksum_y(&dsts[i].fb) == 0) {
                    ++zeros;
                    if (first_zero < 0) {
                        first_zero = (int)i;
                    }
                    last_zero = (int)i;
                }
            }

            const uint32_t iters = (N >= 64) ? std::max(30u, cfg.iters / 2)
                                             : cfg.iters;
            std::vector<double> lat;
            lat.reserve(iters);
            for (uint32_t k = 0; k < iters; ++k) {
                const double t0 = now_us();
                if (dsp_multi_crop_and_resize(vd.dev, &m) != DSP_SUCCESS) {
                    ++errors;
                }
                lat.push_back(now_us() - t0);
            }
            const LatencyStats st = summarize(lat);
            if (N == 1) {
                mean_1 = st.mean_us;
            }
            std::printf(
                "    VND N=%-3u mean=%7.0fus p95=%7.0fus rects/s=%8.1f "
                "marginal=%7.1fus/rect zeros=%u%s rc=%d\n",
                N, st.mean_us, st.p95_us, (double)N * 1e6 / st.mean_us,
                (N > 1 && mean_1 > 0.0) ? (st.mean_us - mean_1) / (N - 1)
                                        : 0.0,
                zeros,
                zeros ? (" first=" + std::to_string(first_zero) +
                         " last=" + std::to_string(last_zero))
                            .c_str()
                      : "",
                rc);
            for (auto &d : dsts) {
                free_nv12(&d);
            }
        }
    }

    /* ---- E4-C: blend (base NV12 in-place, overlay ARGB) ---- */
    std::printf("\n[E4-C] HAL blend: base %ux%u NV12 in-place + ARGB "
                "256x128 overlay (left half a=0, right half a=255)\n",
                cfg.w, cfg.h);
    {
        Nv12Buffer base{};
        if (!alloc_nv12(&base, cfg.w, cfg.h, src->using_dmabuf)) {
            std::printf("    base alloc FAILED\n");
            HAL_DSP_OPS.deinit(ctx);
            return -1;
        }
        fill_nv12_pattern(&base.fb);

        const uint32_t ow = 256, oh = 128, ox = 64, oy = 96;
        ArgbBuffer ovb{};
        ovb.fb.width = ow;
        ovb.fb.height = oh;
        ovb.fb.format = HAL_PIX_FMT_ARGB32;
        ovb.fb.num_planes = 1;
        ovb.fb.strides[0] = ow * 4;
        ovb.fb.sizes[0] = ow * oh * 4;
        for (int i = 0; i < HAL_MAX_PLANES; ++i) {
            ovb.fb.dma_fds[i] = -1;
        }
        ovb.heap.resize((size_t)ow * oh * 4);
        ovb.fb.planes[0] = ovb.heap.data();
        ovb.fb.mem_type = HAL_MEM_MALLOC;
        fill_argb_overlay(&ovb.fb);

        HalDspOverlay hov{};
        hov.overlay = &ovb.fb;
        hov.x_offset = (int32_t)ox;
        hov.y_offset = (int32_t)oy;
        HalDspBlendParams bp{};
        bp.base = &base.fb;
        bp.overlays = &hov;
        bp.overlay_count = 1;

        const double out_b = mean_y_region(&base.fb, 4, 4, 24, 24);
        const double a0_b = mean_y_region(&base.fb, ox + 4, oy + 4,
                                          ox + ow / 2 - 4, oy + oh - 4);
        const double a1_b = mean_y_region(&base.fb, ox + ow / 2 + 4, oy + 4,
                                          ox + ow - 4, oy + oh - 4);
        const int rc = HAL_DSP_OPS.blend(ctx, &bp);
        const double d_out = mean_y_region(&base.fb, 4, 4, 24, 24) - out_b;
        const double d_a0 = mean_y_region(&base.fb, ox + 4, oy + 4,
                                          ox + ow / 2 - 4, oy + oh - 4) - a0_b;
        const double d_a1 = mean_y_region(&base.fb, ox + ow / 2 + 4, oy + 4,
                                          ox + ow - 4, oy + oh - 4) - a1_b;
        const bool pass = (rc == 0) && (d_out < 0.5 && d_out > -0.5) &&
                          (d_a0 < 0.5 && d_a0 > -0.5) &&
                          (d_a1 > 2.0 || d_a1 < -2.0);
        std::printf("    blend rc=%d dY(outside)=%+.2f dY(alpha0)=%+.2f "
                    "dY(alpha255)=%+.2f -> %s\n",
                    rc, d_out, d_a0, d_a1,
                    pass ? "PASS (a=0 passthrough, a=255 blends)"
                         : "FAIL");

        if (rc == 0) {
            /* timing: 1 overlay vs 8 overlays */
            std::vector<double> lat1;
            lat1.reserve(cfg.iters);
            for (uint32_t k = 0; k < cfg.iters; ++k) {
                const double t0 = now_us();
                if (HAL_DSP_OPS.blend(ctx, &bp) != 0) {
                    ++errors;
                }
                lat1.push_back(now_us() - t0);
            }
            const LatencyStats st1 = summarize(lat1);

            HalDspOverlay ov8[8];
            const int32_t xs[8] = {0, 320, 640, 960, 1280, 1600, 200, 800};
            const int32_t ys[8] = {0, 0, 0, 0, 0, 0, 540, 540};
            for (int i = 0; i < 8; ++i) {
                ov8[i].overlay = &ovb.fb;
                ov8[i].x_offset = xs[i];
                ov8[i].y_offset = ys[i];
            }
            HalDspBlendParams bp8{};
            bp8.base = &base.fb;
            bp8.overlays = ov8;
            bp8.overlay_count = 8;
            std::vector<double> lat8;
            lat8.reserve(cfg.iters);
            for (uint32_t k = 0; k < cfg.iters; ++k) {
                const double t0 = now_us();
                if (HAL_DSP_OPS.blend(ctx, &bp8) != 0) {
                    ++errors;
                }
                lat8.push_back(now_us() - t0);
            }
            const LatencyStats st8 = summarize(lat8);
            std::printf("    blend 1 ov : mean=%7.0fus p95=%7.0fus\n",
                        st1.mean_us, st1.p95_us);
            std::printf("    blend 8 ov : mean=%7.0fus p95=%7.0fus "
                        "marginal=%7.1fus/ov\n",
                        st8.mean_us, st8.p95_us,
                        (st8.mean_us - st1.mean_us) / 7.0);
        }

        /* unsupported overlay format must be rejected (run LAST) */
        Nv12Buffer bad{};
        if (alloc_nv12(&bad, 64, 64, false)) {
            HalDspOverlay bov{};
            bov.overlay = &bad.fb;
            bov.x_offset = 0;
            bov.y_offset = 0;
            HalDspBlendParams bb{};
            bb.base = &base.fb;
            bb.overlays = &bov;
            bb.overlay_count = 1;
            const int rcb = HAL_DSP_OPS.blend(ctx, &bb);
            std::printf("    NV12 overlay (unsupported per docs) rc=%d -> %s\n",
                        rcb, rcb != 0 ? "rejected as expected"
                                      : "ACCEPTED (contract wider than docs)");
            free_nv12(&bad);
        }
        free_nv12(&base);
    }

    HAL_DSP_OPS.deinit(ctx);
    return 0;
}

/* mem-mode auto probe: try userptr first, fall back to dmabuf */
bool alloc_auto(Nv12Buffer *src, Nv12Buffer *dst, const ProbeCfg &cfg,
                std::string *used)
{
    bool try_dmabuf = (cfg.mem == "dmabuf");
    if (cfg.mem == "auto" || cfg.mem == "userptr") {
        if (alloc_nv12(src, cfg.w, cfg.h, false) &&
            alloc_nv12(dst, cfg.ow, cfg.oh, false)) {
            fill_nv12_pattern(&src->fb);
            fill_nv12_pattern(&dst->fb);
            void *probe_ctx = nullptr;
            HalDspConfig c{};
            *used = "userptr";
            if (HAL_DSP_OPS.init(&c, &probe_ctx) == 0) {
                const int rc = probe_resize_once(probe_ctx, src, dst, "auto");
                HAL_DSP_OPS.deinit(probe_ctx);
                if (rc == 0) {
                    return true;
                }
            }
            free_nv12(src);
            free_nv12(dst);
            if (cfg.mem == "userptr") {
                return false;
            }
            std::printf("    userptr failed -> trying dmabuf\n");
            try_dmabuf = true;
        }
    }
    if (!alloc_nv12(src, cfg.w, cfg.h, try_dmabuf) ||
        !alloc_nv12(dst, cfg.ow, cfg.oh, try_dmabuf)) {
        return false;
    }
    *used = try_dmabuf ? "dmabuf" : "userptr";
    fill_nv12_pattern(&src->fb);
    fill_nv12_pattern(&dst->fb);
    void *probe_ctx = nullptr;
    HalDspConfig c{};
    if (HAL_DSP_OPS.init(&c, &probe_ctx) != 0) {
        return false;
    }
    const int rc = probe_resize_once(probe_ctx, src, dst, "mem-probe");
    HAL_DSP_OPS.deinit(probe_ctx);
    return rc == 0;
}

} /* namespace */

int main(int argc, char **argv)
{
    ProbeCfg cfg;
    std::string mode = "all";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if (a == "--mode") {
            mode = next();
        } else if (a == "--w") {
            cfg.w = (uint32_t)std::stoul(next());
        } else if (a == "--h") {
            cfg.h = (uint32_t)std::stoul(next());
        } else if (a == "--ow") {
            cfg.ow = (uint32_t)std::stoul(next());
        } else if (a == "--oh") {
            cfg.oh = (uint32_t)std::stoul(next());
        } else if (a == "--iters") {
            cfg.iters = (uint32_t)std::stoul(next());
        } else if (a == "--seconds") {
            cfg.seconds = (uint32_t)std::stoul(next());
        } else if (a == "--depth") {
            cfg.depth = (uint32_t)std::stoul(next());
        } else if (a == "--mem") {
            cfg.mem = next();
        }
    }

    std::printf("dsp_p0_probe mode=%s src=%ux%u dst=%ux%u iters=%u mem=%s\n",
                mode.c_str(), cfg.w, cfg.h, cfg.ow, cfg.oh, cfg.iters,
                cfg.mem.c_str());
    std::printf("HAL version: %s\n", HAL_DSP_OPS.get_version());

    Nv12Buffer src{}, dst{};
    std::string used;
    if (!alloc_auto(&src, &dst, cfg, &used)) {
        std::printf("FATAL: no working buffer mode\n");
        return 2;
    }
    std::printf("buffer mode: %s\n", used.c_str());

    if (mode == "e1" || mode == "all") {
        run_e1(&src, &dst);
    }
    if (mode == "e2a" || mode == "all") {
        run_e2a(cfg, &src, &dst);
    }
    if (mode == "e2b" || mode == "all") {
        run_e2b(cfg, &src, &dst);
    }
    if (mode == "e3" || mode == "all") {
        run_e3(cfg, &src, &dst);
    }
    if (mode == "e4") { /* not in "all": e4 allocates its own dsts */
        run_e4(cfg, &src);
    }

    free_nv12(&src);
    free_nv12(&dst);
    std::printf("\nprobe done (errors=%d)\n", errors);
    return errors ? 1 : 0;
}

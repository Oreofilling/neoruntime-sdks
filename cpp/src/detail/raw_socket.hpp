// raw_socket.hpp — thin POSIX Unix-domain-socket helpers shared by the raw-UDS
// transports (FdMediaClient, EncodedStreamClient, and the audio clients).
//
// INTERNAL (PRIVATE include path). No public API depends on this header.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cerrno>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

namespace neoruntime_ipc_sdk::detail {

// Connect a blocking SOCK_STREAM to a Unix domain socket `path`. Throws
// std::runtime_error on failure. Returns the fd (caller owns it).
inline int connect_unix(const std::string& path) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("connect_unix: socket() failed");
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        throw std::runtime_error("connect_unix: path too long: " + path);
    }
    path.copy(addr.sun_path, path.size());
    addr.sun_path[path.size()] = '\0';

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        throw std::runtime_error("connect_unix: connect() failed for " + path);
    }
    return fd;
}

// Write exactly n bytes. Returns false on EOF/error.
inline bool send_all(int fd, const void* buf, std::size_t n) {
    auto p = static_cast<const char*>(buf);
    std::size_t sent = 0;
    while (sent < n) {
        ssize_t r = ::send(fd, p + sent, n - sent, 0);
        if (r <= 0) {
            if (r < 0 && errno == EINTR) continue;
            return false;
        }
        sent += static_cast<std::size_t>(r);
    }
    return true;
}

// Read exactly n bytes. Returns false on EOF or short read (the peer closed).
inline bool recv_exact(int fd, void* buf, std::size_t n) {
    auto p = static_cast<char*>(buf);
    std::size_t got = 0;
    while (got < n) {
        ssize_t r = ::recv(fd, p + got, n - got, 0);
        if (r == 0) return false;  // peer closed
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        got += static_cast<std::size_t>(r);
    }
    return true;
}

// Set SO_RCVTIMEO so recv_exact returns false instead of blocking forever.
inline bool set_recv_timeout(int fd, int timeout_ms) {
    if (timeout_ms <= 0) return true;
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    return ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
}

// Receive up to `len` bytes plus any SCM_RIGHTS file descriptors in one
// recvmsg call (Python _transport.recvmsg_with_fds). Returns the byte count
// (0 on clean EOF, -1 on error/timeout); received fds are appended to
// `fds_out`. `max_fds` bounds the ancillary buffer — the kernel silently
// truncates anything beyond it.
inline ssize_t recvmsg_with_fds(int fd, void* buf, std::size_t len,
                                std::vector<int>& fds_out, int max_fds = 3) {
    iovec iov;
    iov.iov_base = buf;
    iov.iov_len = len;

    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    std::vector<char> cbuf(CMSG_SPACE(max_fds * sizeof(int)));
    msg.msg_control = cbuf.data();
    msg.msg_controllen = cbuf.size();

    ssize_t n = ::recvmsg(fd, &msg, 0);
    if (n <= 0) return n;  // 0 = EOF, -1 = error/timeout

    for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr;
         cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
            int nf = static_cast<int>((cmsg->cmsg_len - sizeof(cmsghdr)) / sizeof(int));
            auto fd_data = reinterpret_cast<int*>(CMSG_DATA(cmsg));
            for (int i = 0; i < nf; ++i) fds_out.push_back(fd_data[i]);
        }
    }
    return n;
}

// Send exactly n bytes (no ancillary data). Returns false on failure.
inline bool sendmsg_plain(int fd, const void* buf, std::size_t n) {
    return send_all(fd, buf, n);
}

}  // namespace neoruntime_ipc_sdk::detail

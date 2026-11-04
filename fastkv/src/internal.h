#pragma once
#include "fastkv/fastkv.h"
#include <condition_variable>
#include <deque>
#include <thread>

namespace fastkv {
using Deadline = std::chrono::steady_clock::time_point;
struct Socket {
    int fd = -1;
    const std::atomic<bool> *stopped = nullptr;
    explicit Socket(int fd = -1, const std::atomic<bool> *stopped = nullptr) : fd(fd), stopped(stopped) {}
    ~Socket();
    Socket(Socket &&s) noexcept : fd(s.fd), stopped(s.stopped) { s.fd = -1; }
    Socket &operator=(Socket &&s) noexcept;
    Socket(const Socket &) = delete;
    void Send(const void *p, size_t n, Deadline deadline) const;
    void Receive(void *p, size_t n, Deadline deadline) const;
    void SendFrame(const Bytes &bytes, Deadline deadline) const;
    Bytes ReceiveFrame(size_t limit, Deadline deadline) const;
    void CheckRunning(Deadline deadline) const;
};

class Pool {
  public:
    Pool(int count, int capacity);
    ~Pool();
    void Enqueue(std::function<void()> task);
    void Stop();

  private:
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::function<void()>> queue;
    std::vector<std::thread> workers;
    size_t capacity;
    bool stop = false;
};

class RdmaConnection {
  public:
    virtual ~RdmaConnection() = default;
    virtual void Send(const Socket &, const std::vector<Buffer> &, Deadline) = 0;
    virtual void Receive(const Socket &, const std::vector<Buffer> &, Deadline) = 0;
};
std::unique_ptr<RdmaConnection> ConnectRdma(const Socket &, const Options &, Deadline);
bool RdmaAvailable();
void Require(bool condition, const char *message);
size_t CheckedAdd(size_t a, size_t b);
size_t CheckedMul(size_t a, size_t b);
} // namespace fastkv

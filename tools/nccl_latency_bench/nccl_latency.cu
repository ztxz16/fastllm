#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        cudaError_t error = (cmd);                                             \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error " << cudaGetErrorString(error)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define NCCL_CHECK(cmd)                                                        \
    do {                                                                       \
        ncclResult_t result = (cmd);                                           \
        if (result != ncclSuccess) {                                           \
            std::cerr << "NCCL error " << ncclGetErrorString(result)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    int device0 = 0;
    int device1 = 1;
    int warmup = 100;
    int iterations = 1000;
    int batch_iterations = 2000;
    std::vector<size_t> byte_sizes = {4096, 16384, 2 * 1024 * 1024};
};

struct Stats {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

double ElapsedUs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

Stats Summarize(std::vector<double> values) {
    if (values.empty()) {
        return {};
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    std::sort(values.begin(), values.end());
    auto percentile = [&](double p) {
        const size_t index = static_cast<size_t>(
            std::ceil(p * static_cast<double>(values.size())) - 1.0);
        return values[std::min(index, values.size() - 1)];
    };
    return {
        sum / static_cast<double>(values.size()),
        percentile(0.50),
        percentile(0.95),
        values.front(),
        values.back(),
    };
}

size_t ParseSize(const std::string &text) {
    if (text.empty()) {
        throw std::invalid_argument("empty byte size");
    }
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str()) {
        throw std::invalid_argument("invalid byte size: " + text);
    }
    unsigned long long multiplier = 1;
    if (*end != '\0') {
        if (end[1] != '\0') {
            throw std::invalid_argument("invalid byte suffix: " + text);
        }
        switch (*end) {
            case 'k':
            case 'K': multiplier = 1024; break;
            case 'm':
            case 'M': multiplier = 1024 * 1024; break;
            case 'g':
            case 'G': multiplier = 1024ULL * 1024ULL * 1024ULL; break;
            default: throw std::invalid_argument("invalid byte suffix: " + text);
        }
    }
    if (value > std::numeric_limits<size_t>::max() / multiplier) {
        throw std::overflow_error("byte size is too large: " + text);
    }
    return static_cast<size_t>(value * multiplier);
}

std::vector<size_t> ParseSizes(const std::string &text) {
    std::vector<size_t> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const size_t bytes = ParseSize(item);
        if (bytes == 0 || bytes % sizeof(__half) != 0) {
            throw std::invalid_argument("sizes must be non-zero multiples of 2 bytes");
        }
        result.push_back(bytes);
    }
    if (result.empty()) {
        throw std::invalid_argument("--bytes needs at least one size");
    }
    return result;
}

int ParsePositiveInt(const char *name, const char *text) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return static_cast<int>(value);
}

void PrintUsage(const char *program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --devices D0,D1       CUDA device ids (default: 0,1)\n"
        << "  --bytes LIST          Comma-separated sizes; K/M/G suffixes allowed\n"
        << "                        (default: 4K,16K,2M)\n"
        << "  --warmup N            Synchronous warmup collectives (default: 100)\n"
        << "  --iters N             Individually synchronized samples (default: 1000)\n"
        << "  --batch-iters N       Collectives queued before one sync (default: 2000)\n";
}

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--devices") {
            const std::string value = next("--devices");
            const size_t comma = value.find(',');
            if (comma == std::string::npos || value.find(',', comma + 1) != std::string::npos) {
                throw std::invalid_argument("--devices must be D0,D1");
            }
            options.device0 = std::stoi(value.substr(0, comma));
            options.device1 = std::stoi(value.substr(comma + 1));
            if (options.device0 == options.device1) {
                throw std::invalid_argument("the two CUDA devices must be different");
            }
        } else if (arg == "--bytes") {
            options.byte_sizes = ParseSizes(next("--bytes"));
        } else if (arg == "--warmup") {
            options.warmup = ParsePositiveInt("--warmup", next("--warmup"));
        } else if (arg == "--iters") {
            options.iterations = ParsePositiveInt("--iters", next("--iters"));
        } else if (arg == "--batch-iters") {
            options.batch_iterations =
                ParsePositiveInt("--batch-iters", next("--batch-iters"));
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    return options;
}

class NcclPair {
public:
    explicit NcclPair(const Options &options)
        : devices_{options.device0, options.device1} {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        for (int device : devices_) {
            if (device < 0 || device >= device_count) {
                throw std::invalid_argument("CUDA device id is out of range");
            }
        }

        int can_access_01 = 0;
        int can_access_10 = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_01, devices_[0], devices_[1]));
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_10, devices_[1], devices_[0]));
        std::cout << "CUDA peer access: " << devices_[0] << "->" << devices_[1]
                  << "=" << can_access_01 << ", " << devices_[1] << "->"
                  << devices_[0] << "=" << can_access_10 << "\n";

        NCCL_CHECK(ncclCommInitAll(comms_, 2, devices_));
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(devices_[rank]));
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[rank], cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreate(&start_events_[rank]));
            CUDA_CHECK(cudaEventCreate(&stop_events_[rank]));
        }
    }

    ~NcclPair() {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(devices_[rank]));
            if (send_[rank] != nullptr) CUDA_CHECK(cudaFree(send_[rank]));
            if (recv_[rank] != nullptr) CUDA_CHECK(cudaFree(recv_[rank]));
            CUDA_CHECK(cudaEventDestroy(start_events_[rank]));
            CUDA_CHECK(cudaEventDestroy(stop_events_[rank]));
            CUDA_CHECK(cudaStreamDestroy(streams_[rank]));
            NCCL_CHECK(ncclCommDestroy(comms_[rank]));
        }
    }

    void Allocate(size_t bytes) {
        if (bytes <= allocated_bytes_) return;
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(devices_[rank]));
            if (send_[rank] != nullptr) CUDA_CHECK(cudaFree(send_[rank]));
            if (recv_[rank] != nullptr) CUDA_CHECK(cudaFree(recv_[rank]));
            CUDA_CHECK(cudaMalloc(&send_[rank], bytes));
            CUDA_CHECK(cudaMalloc(&recv_[rank], bytes));
            CUDA_CHECK(cudaMemsetAsync(send_[rank], 0, bytes, streams_[rank]));
            CUDA_CHECK(cudaMemsetAsync(recv_[rank], 0, bytes, streams_[rank]));
        }
        Synchronize();
        allocated_bytes_ = bytes;
    }

    void EnqueueAllReduce(size_t bytes) {
        const size_t elements = bytes / sizeof(__half);
        NCCL_CHECK(ncclGroupStart());
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(devices_[rank]));
            NCCL_CHECK(ncclAllReduce(send_[rank], recv_[rank], elements, ncclHalf,
                                     ncclSum, comms_[rank], streams_[rank]));
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    void Synchronize() {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(devices_[rank]));
            CUDA_CHECK(cudaStreamSynchronize(streams_[rank]));
        }
    }

    void RecordStarts() {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(devices_[rank]));
            CUDA_CHECK(cudaEventRecord(start_events_[rank], streams_[rank]));
        }
    }

    void RecordStops() {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(devices_[rank]));
            CUDA_CHECK(cudaEventRecord(stop_events_[rank], streams_[rank]));
        }
    }

    double WaitStopsAndGetMaxUs() {
        double maximum_us = 0.0;
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(devices_[rank]));
            CUDA_CHECK(cudaEventSynchronize(stop_events_[rank]));
            float milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_events_[rank],
                                            stop_events_[rank]));
            maximum_us = std::max(maximum_us,
                                  static_cast<double>(milliseconds) * 1000.0);
        }
        return maximum_us;
    }

private:
    int devices_[2];
    ncclComm_t comms_[2]{};
    cudaStream_t streams_[2]{};
    cudaEvent_t start_events_[2]{};
    cudaEvent_t stop_events_[2]{};
    void *send_[2]{};
    void *recv_[2]{};
    size_t allocated_bytes_ = 0;
};

void PrintStats(const char *name, const Stats &stats) {
    std::cout << std::left << std::setw(20) << name << std::right << std::fixed
              << std::setprecision(2) << " mean=" << std::setw(8) << stats.mean
              << " us  p50=" << std::setw(8) << stats.p50
              << " us  p95=" << std::setw(8) << stats.p95
              << " us  min=" << std::setw(8) << stats.minimum
              << " us  max=" << std::setw(8) << stats.maximum << " us\n";
}

void BenchmarkSize(NcclPair &pair, const Options &options, size_t bytes) {
    pair.Allocate(bytes);
    for (int i = 0; i < options.warmup; ++i) {
        pair.EnqueueAllReduce(bytes);
        pair.Synchronize();
    }

    std::vector<double> enqueue_samples;
    std::vector<double> device_samples;
    std::vector<double> end_to_end_samples;
    enqueue_samples.reserve(options.iterations);
    device_samples.reserve(options.iterations);
    end_to_end_samples.reserve(options.iterations);

    for (int i = 0; i < options.iterations; ++i) {
        pair.RecordStarts();
        const auto total_begin = Clock::now();
        const auto enqueue_begin = Clock::now();
        pair.EnqueueAllReduce(bytes);
        const auto enqueue_end = Clock::now();
        pair.RecordStops();
        const double device_us = pair.WaitStopsAndGetMaxUs();
        const auto total_end = Clock::now();
        enqueue_samples.push_back(ElapsedUs(enqueue_begin, enqueue_end));
        device_samples.push_back(device_us);
        end_to_end_samples.push_back(ElapsedUs(total_begin, total_end));
    }

    pair.Synchronize();
    pair.RecordStarts();
    const auto batch_begin = Clock::now();
    for (int i = 0; i < options.batch_iterations; ++i) {
        pair.EnqueueAllReduce(bytes);
    }
    const auto batch_enqueue_end = Clock::now();
    pair.RecordStops();
    const double batch_device_us = pair.WaitStopsAndGetMaxUs();
    const auto batch_end = Clock::now();

    std::cout << "\nbytes=" << bytes << " (" << bytes / sizeof(__half)
              << " fp16 elements)\n";
    PrintStats("host enqueue", Summarize(std::move(enqueue_samples)));
    PrintStats("CUDA event complete", Summarize(std::move(device_samples)));
    PrintStats("host enqueue+wait", Summarize(std::move(end_to_end_samples)));
    std::cout << std::fixed << std::setprecision(2)
              << "batched / operation  enqueue="
              << ElapsedUs(batch_begin, batch_enqueue_end) /
                     static_cast<double>(options.batch_iterations)
              << " us  CUDA-event="
              << batch_device_us / static_cast<double>(options.batch_iterations)
              << " us  wall="
              << ElapsedUs(batch_begin, batch_end) /
                     static_cast<double>(options.batch_iterations)
              << " us\n";
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        int runtime_version = 0;
        int driver_version = 0;
        int nccl_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        NCCL_CHECK(ncclGetVersion(&nccl_version));
        std::cout << "CUDA runtime=" << runtime_version
                  << " driver=" << driver_version
                  << " NCCL=" << nccl_version << "\n";
        std::cout << "devices=" << options.device0 << "," << options.device1
                  << " warmup=" << options.warmup
                  << " synchronized_iters=" << options.iterations
                  << " batched_iters=" << options.batch_iterations << "\n";
        const char *protocol = std::getenv("NCCL_PROTO");
        const char *shm_disabled = std::getenv("NCCL_SHM_DISABLE");
        std::cout << "NCCL_PROTO=" << (protocol == nullptr ? "<auto>" : protocol)
                  << " NCCL_SHM_DISABLE="
                  << (shm_disabled == nullptr ? "<unset>" : shm_disabled) << "\n";

        NcclPair pair(options);
        for (size_t bytes : options.byte_sizes) {
            BenchmarkSize(pair, options, bytes);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }
}

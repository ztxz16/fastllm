// ELF --wrap stubs: reject before the CUDA driver/runtime can touch a GPU.
// No CUDA headers, API calls, dlopen, model initialization, or bypass fallback.
#include <atomic>
static std::atomic<unsigned long long> rejected{0};
extern "C" unsigned long long P0SnapshotCudaRejectedCalls() { return rejected.load(); }
#define P0_CUDA_REJECT(symbol) extern "C" int __wrap_##symbol(...) { ++rejected; return 801; }
P0_CUDA_REJECT(cuInit)
P0_CUDA_REJECT(cudaGetDeviceCount)
P0_CUDA_REJECT(cudaGetDevice)
P0_CUDA_REJECT(cudaSetDevice)
P0_CUDA_REJECT(cudaGetDeviceProperties)
P0_CUDA_REJECT(cudaGetDeviceProperties_v2)
P0_CUDA_REJECT(cudaMalloc)
P0_CUDA_REJECT(cudaMallocAsync)
P0_CUDA_REJECT(cudaFree)
P0_CUDA_REJECT(cudaHostAlloc)
P0_CUDA_REJECT(cudaHostRegister)
P0_CUDA_REJECT(cudaStreamCreateWithFlags)
P0_CUDA_REJECT(cudaStreamCreateWithPriority)
P0_CUDA_REJECT(cudaEventCreateWithFlags)
P0_CUDA_REJECT(cudaMemcpy)
P0_CUDA_REJECT(cudaMemcpyAsync)
P0_CUDA_REJECT(cudaLaunchKernel)
#undef P0_CUDA_REJECT

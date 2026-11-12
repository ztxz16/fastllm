// Included after the planar launchers. Geometry candidates reuse the existing
// GGUF dot products and rounding; selection belongs to the common registry.
namespace fastllm_gguf_planar {
namespace {
const fastllm::kernels::Family projectionFamily{14, "gguf.planar.projection", "1", {
    {"cuda.planar.reference", json11::Json::object{{"backend","cuda"},{"algorithm","existing planar projection"}}},
    {"cuda.planar.w2.r1.full", json11::Json::object{{"backend","cuda"},{"warps",2},{"outputs_per_warp",1},{"grid","all output tiles"},{"formats","Q4_K/IQ4_XS"},{"rows","arbitrary; physical tiles up to 8"},{"architecture","CUDA; no tensor cores"}}},
    {"cuda.planar.w4.r1.occupancy", json11::Json::object{{"backend","cuda"},{"warps",4},{"outputs_per_warp",1},{"grid","device occupancy"},{"formats","Q4_K/IQ4_XS"},{"rows","arbitrary; physical tiles up to 8"},{"architecture","CUDA; no tensor cores"}}},
    {"cuda.planar.w8.r2.occupancy", json11::Json::object{{"backend","cuda"},{"warps",8},{"outputs_per_warp",2},{"grid","device occupancy"},{"formats","Q4_K/IQ4_XS"},{"rows","arbitrary; physical tiles up to 8"},{"architecture","CUDA; no tensor cores"}}}
}};
const bool projectionRegistered = [] { fastllm::kernels::Register(&projectionFamily); return true; }();
struct ProjectionCase {
    const void *weight, *workspace;
    half *output;
    int type, tokens, k, n, stride;
    cudaStream_t stream;
};
template<ggml_type Type, int T, int Warps, int Rows, bool Occupancy>
void LaunchMeasured(const ProjectionCase &c, const int8_t *q, const half2 *d,
                    const short2 *s, half *y) {
    int blocks = (c.n + Warps * Rows - 1) / (Warps * Rows);
    if constexpr (Occupancy) {
        static thread_local std::map<int,int> limits;
        const int device = FastllmCudaGetDevice();
        auto found = limits.find(device);
        if (found == limits.end()) {
            int active = 0, sms = 0;
            fastllm::kernels::CudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, Project<Type,T,0,0,false,Warps,Rows>, Warps * 32, 0));
            fastllm::kernels::CudaCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            found = limits.emplace(device, std::max(1,active*sms)).first;
        }
        blocks = std::min(blocks,found->second);
    }
    Project<Type,T,0,0,false,Warps,Rows><<<blocks,Warps*32,0,c.stream>>>(
        c.weight,nullptr,q,d,s,y,c.n,c.stride,c.k);
}
template<ggml_type Type, int T>
void LaunchMeasuredTile(const ProjectionCase &c, int impl, const int8_t *q,
                        const half2 *d, const short2 *s, half *y) {
    switch (impl) {
    case 1: LaunchMeasured<Type,T,2,1,false>(c,q,d,s,y); break;
    case 2: LaunchMeasured<Type,T,4,1,true>(c,q,d,s,y); break;
    case 3: LaunchMeasured<Type,T,8,2,true>(c,q,d,s,y); break;
    }
}
template<ggml_type Type>
void RunMeasured(const ProjectionCase &c,int impl) {
    if (!impl) {
        DispatchRows<Type,0>(c.weight,nullptr,c.workspace,c.output,c.tokens,c.k,c.n,c.stride,c.stream);
        return;
    }
    const auto *q = static_cast<const int8_t *>(c.workspace);
    const auto *d = reinterpret_cast<const half2 *>(q+size_t(c.tokens)*c.k);
    const auto *s = reinterpret_cast<const short2 *>(d+size_t(c.tokens)*(c.k/32));
    for (int64_t first=0;first<c.tokens;first+=8) {
        const auto *qt=q+first*c.k; const auto *dt=d+first*(c.k/32); const auto *st=s+first*(c.k/32);
        auto *y=c.output+first*c.stride;
#define TILE(T) case T: LaunchMeasuredTile<Type,T>(c,impl,qt,dt,st,y); break
        switch(std::min<int64_t>(8,c.tokens-first)) {
            TILE(1); TILE(2); TILE(3); TILE(4); TILE(5); TILE(6); TILE(7); TILE(8);
        }
#undef TILE
    }
}
bool RunProjection(void *opaque,int impl) {
    const auto &c=*static_cast<ProjectionCase *>(opaque);
    if(c.type==GGML_TYPE_Q4_K) RunMeasured<GGML_TYPE_Q4_K>(c,impl);
    else RunMeasured<GGML_TYPE_IQ4_XS>(c,impl);
    return true;
}
void SelectProjection(ProjectionCase operation) {
    using namespace fastllm;
    if(!kernels::Enabled()) { RunProjection(&operation,0); return; }
    kernels::Signature signature{{14,FastllmCudaRuntimeArch(),FastllmCudaGetDevice(),
        operation.tokens,operation.k,operation.n,operation.type,operation.stride}};
    cudaStreamCaptureStatus status;
    kernels::CudaCheck(cudaStreamIsCapturing(operation.stream,&status));
    const bool capturing=status!=cudaStreamCaptureStatusNone || FastllmCudaGraphIsCapturing() ||
        FastllmCudaGraphMemoryPoolActive();
    ProjectionCase trial=operation;
    kernels::CudaOutputsCase benchmark{{&trial,RunProjection,
        [](void *,int){return std::string{};},nullptr,0,false},{}};
    void *scratch=nullptr;
    int selected=0;
    try {
        if(!capturing && kernels::NeedsPreparation(signature)) {
            // Inputs may have been produced on a caller-owned stream. Trials
            // use private outputs and the benchmark stream, never live output.
            kernels::CudaCheck(cudaStreamSynchronize(operation.stream));
            kernels::CudaCheck(cudaMalloc(&scratch,size_t(operation.tokens)*operation.stride*sizeof(half)));
            trial.output=static_cast<half *>(scratch); trial.stream=cudaStreamPerThread;
            benchmark.outputs.push_back({scratch,size_t(operation.tokens)*operation.stride,FLOAT16});
        }
        selected=kernels::Select({&projectionFamily,signature,&benchmark,capturing,
            kernels::CudaOutputsUnsupported,
            [](void *p,int i,int repeats) { return kernels::CudaMeasure(
                &static_cast<kernels::CudaOutputsCase *>(p)->operation,i,repeats); },
            kernels::CudaOutputsValidate,kernels::CudaHardware,0});
    } catch(...) { if(scratch)cudaFree(scratch); throw; }
    if(scratch)kernels::CudaCheck(cudaFree(scratch));
    RunProjection(&operation,selected<0?0:selected);
}
} // namespace
} // namespace fastllm_gguf_planar

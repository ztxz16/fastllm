namespace {
bool HyperReadDense(const fastllm::Data &d) {
    uint64_t stride = 1;
    if (d.dims.empty() || d.strides.size() != d.dims.size()) return false;
    for (int i = int(d.dims.size()) - 1; i >= 0; --i) {
        if (d.dims[i] <= 0 || (d.dims[i] > 1 && d.strides[i] != stride)) return false;
        stride *= d.dims[i];
    }
    return d.dataDevice == fastllm::CUDA && d.cudaData;
}
bool HyperReadOverlap(const fastllm::Data &a, const fastllm::Data &b) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a.cudaData), y = reinterpret_cast<uintptr_t>(b.cudaData);
    return x < y + b.GetBytes() && y < x + a.GetBytes();
}
}

bool FastllmCudaQwen4HyperRead(const fastllm::Data &input,
        const fastllm::Data *block, const fastllm::Data *oldInjection,
        const fastllm::Data &normWeight, const fastllm::Data &down,
        const fastllm::Data &up, const fastllm::Data *inject, fastllm::Data *residual,
        fastllm::Data &mixed, fastllm::Data *injection, int groups, float eps, int rowTile, int columns, bool planar) {
    using namespace fastllm;
    if (groups <= 0 || !HyperReadDense(input) || input.dataType != FLOAT32 ||
        input.dims.back() <= 0 || input.dims.back() % groups || !std::isfinite(eps) || eps < 0 ||
        (rowTile != 1 && rowTile != 4) || (columns != 2 && columns != 4 && columns != 8 && columns != 16)) return false;
    const int channels = input.dims.back(), width = channels / groups;
    const uint64_t rows = input.Count(0) / channels;
    if (!rows || rows > INT_MAX || !HyperReadDense(down) || down.dataType != FLOAT16 ||
        down.dims.size() != 2 || down.dims[1] != channels ||
        !HyperReadDense(normWeight) || normWeight.dataType != FLOAT32 || normWeight.Count(0) != channels ||
        !HyperReadDense(up) || up.dataType != FLOAT16 || up.dims != std::vector<int>({channels,down.dims[0]}) ||
        bool(block) != bool(oldInjection) || bool(block) != bool(residual) || bool(inject) != bool(injection)) return false;
    const int rank = down.dims[0];
    if (groups != 4 || width != 2560 || rank != 320) return false;
    std::vector<const Data *> reads{&input, &normWeight, &down, &up}, writes{&mixed};
    if (block) {
        if (!HyperReadDense(*block) || !Qwen4CudaActivationType(block->dataType) || block->Count(0) != rows * width ||
            !HyperReadDense(*oldInjection) || !Qwen4CudaActivationType(oldInjection->dataType) || oldInjection->Count(0) != rows * groups ||
            residual->dims != input.dims) return false;
        reads.push_back(block); reads.push_back(oldInjection); writes.push_back(residual);
    }
    if (inject) {
        if (!HyperReadDense(*inject) || inject->dataType != FLOAT16 || inject->dims != std::vector<int>({groups, channels}) ||
            injection->Count(0) != rows * groups) return false;
        reads.push_back(inject); writes.push_back(injection);
    }
    if (!Qwen4CudaActivationType(mixed.dataType) || mixed.Count(0) != rows * width) return false;
    // Vector loads require aligned dense inputs. Views that violate this
    // contract are handled by the original implementation before any write.
    for (const Data *data : {&input, &normWeight, &down, &up})
        if ((reinterpret_cast<uintptr_t>(data->cudaData) & 15) ||
            data->dataDeviceIds != input.dataDeviceIds) return false;
    if (inject && ((reinterpret_cast<uintptr_t>(inject->cudaData) & 15) ||
                   inject->dataDeviceIds != input.dataDeviceIds)) return false;
    if (block && (block->dataDeviceIds != input.dataDeviceIds ||
                  oldInjection->dataDeviceIds != input.dataDeviceIds ||
                  (block->dataType == FLOAT32 && (reinterpret_cast<uintptr_t>(block->cudaData) & 15)))) return false;
    for (size_t i = 0; i < writes.size(); ++i) {
        if (!HyperReadDense(*writes[i]) || (writes[i] != &mixed && writes[i]->dataType != FLOAT32) ||
            writes[i]->dataDeviceIds != input.dataDeviceIds) return false;
        for (const Data *r : reads) if (HyperReadOverlap(*writes[i], *r)) return false;
        for (size_t j = 0; j < i; ++j) if (HyperReadOverlap(*writes[i], *writes[j])) return false;
    }
    int capacity = 0;
    if (cudaDeviceGetAttribute(&capacity, cudaDevAttrMaxSharedMemoryPerBlock, FastllmCudaGetDevice()) != cudaSuccess ||
        capacity < int(std::min<uint64_t>(rows, rowTile) * width * sizeof(float) + 8 * 4 * sizeof(float))) return false;
    // Four is a physical tile, not a limit on logical rows. Scratch stays bounded.
    Data scratch(FLOAT32,{rowTile,groups,rank+groups+1});
    scratch.ToDevice(CUDA,input.dataDeviceIds,false); scratch.Allocate(false);
    using namespace fastllm_strata_hc;
    for (uint64_t first = 0; first < rows; first += rowTile) {
        GrMulti m{}; m.T = int(std::min<uint64_t>(rows-first,rowTile));
        for (int t=0;t<m.T;++t) {
            auto &a=m.a[t]; const uint64_t row=first+t;
            a.R=(const float*)input.cudaData+row*channels;
            a.w_norm=(const float*)normWeight.cudaData;
            a.w_down=(const uint16_t*)down.cudaData; a.w_up=(const uint16_t*)up.cudaData;
            a.w_inject=inject?(const uint16_t*)inject->cudaData:nullptr;
            a.mixed=(char*)mixed.cudaData+row*width*mixed.unitSize; a.mixedType=int(mixed.dataType);
            a.inject_out=injection?(float*)injection->cudaData+row*groups:nullptr;
            a.apply=bool(block); a.eps=eps;
            if(block) {
                a.bo_prev=(const char*)block->cudaData+row*width*block->unitSize;
                a.inj_prev=(const char*)oldInjection->cudaData+row*groups*oldInjection->unitSize;
                a.blockType=int(block->dataType); a.injectionType=int(oldInjection->dataType);
                a.R_out=(float*)residual->cudaData+row*channels;
            }
        }
        float *part=(float*)scratch.cudaData,*ssg=part+rowTile*groups*(rank+groups);
        auto dispatch = [&](auto tile, auto layout) {
            constexpr bool P = decltype(layout)::value;
            constexpr int C = decltype(tile)::value;
            switch(m.T) {case 1:launch<1,C,P>(m,part,ssg);break;case 2:launch<2,C,P>(m,part,ssg);break;
                case 3:launch<3,C,P>(m,part,ssg);break;default:launch<4,C,P>(m,part,ssg);}
        };
        auto selectColumns = [&](auto layout) {
            switch(columns) {case 2:dispatch(std::integral_constant<int,2>{},layout);break;
                case 4:dispatch(std::integral_constant<int,4>{},layout);break;
                case 8:dispatch(std::integral_constant<int,8>{},layout);break;
                default:dispatch(std::integral_constant<int,16>{},layout);}
        };
        if (planar) selectColumns(std::true_type{}); else selectColumns(std::false_type{});

    }
    // Once launched, CUDA failures are errors, never a partial-write fallback.
    checkCudaErrors("HC fused launch", cudaGetLastError());
    return true;
}

namespace {
const fastllm::kernels::Family hyperReadFamily{11,"qwen4.hyper_read","3",{
    {"cuda.hc.reference",json11::Json::object{{"backend","cuda"},{"algorithm","existing norm/projection/mix"},{"rows","arbitrary"}}},
    {"cuda.hc.strata.cols4",json11::Json::object{{"backend","cuda"},{"source","Strata MIT, 9f7d977d"},{"geometry","2560 x 4, rank 320"},{"weights","FP16"},{"row_tile",4},{"column_tile",4},{"rows","arbitrary"},{"architecture","CUDA, no tensor cores"}}},
    {"cuda.hc.strata.cols8",json11::Json::object{{"backend","cuda"},{"source","Strata MIT, 9f7d977d"},{"geometry","2560 x 4, rank 320"},{"weights","FP16"},{"row_tile",4},{"column_tile",8},{"rows","arbitrary"},{"architecture","CUDA, no tensor cores"}}},
    {"cuda.hc.planar.cols4",json11::Json::object{{"backend","cuda"},{"source","Strata MIT, 9f7d977d"},{"geometry","2560 x 4, rank 320"},{"weights","FP16"},{"row_tile",4},{"column_tile",4},{"rows","arbitrary"},{"shared_layout","bank-conflict-free float4 planes"},{"architecture","CUDA, no tensor cores"}}},
    {"cuda.hc.planar.cols8",json11::Json::object{{"backend","cuda"},{"source","Strata MIT, 9f7d977d"},{"geometry","2560 x 4, rank 320"},{"weights","FP16"},{"row_tile",4},{"column_tile",8},{"rows","arbitrary"},{"shared_layout","bank-conflict-free float4 planes"},{"architecture","CUDA, no tensor cores"}}}
}};
const bool hyperReadRegistered=[] {fastllm::kernels::Register(&hyperReadFamily);return true;}();
}
bool FastllmCudaQwen4SelectHyperRead(const fastllm::Data &input,
    const fastllm::Data *block, const fastllm::Data *oldInjection,
    const fastllm::Data &norm, const fastllm::Data &down, const fastllm::Data &up,
    const fastllm::Data *inject, fastllm::Data *residual, fastllm::Data &mixed,
    fastllm::Data *injection, int groups, float eps, bool referenceMode,
    const std::function<void()> &reference) {
    using namespace fastllm;
    if(!kernels::Enabled())return false;
    const int rows=input.Count(0)/input.dims.back();
    std::function<bool(int)> run=[&](int implementation) {
        if(!implementation){reference();return true;}
        return FastllmCudaQwen4HyperRead(input,block,oldInjection,norm,down,up,inject,
            residual,mixed,injection,groups,eps,4,(implementation==1 || implementation==3)?4:8,implementation>=3);
    };
    int capacity=0;
    checkCudaErrors("HC shared memory capacity",cudaDeviceGetAttribute(&capacity,
        cudaDevAttrMaxSharedMemoryPerBlock,FastllmCudaGetDevice()));
    struct Case { std::function<bool(int)> *run; bool supported; } operation{&run,
        groups==4 && input.dims.back()==10240 && down.dims==std::vector<int>({320,10240}) &&
        up.dims==std::vector<int>({10240,320}) && capacity>=std::min(rows,4)*2560*4+128};
    kernels::CudaOutputsCase benchmark{{&operation,
        [](void *p,int i){return (*static_cast<Case*>(p)->run)(i);},
        [](void *p,int i){return i && !static_cast<Case*>(p)->supported ?
            std::string("unsupported HC geometry/shared memory capacity") : std::string{};},nullptr,0,false},
        {{mixed.cudaData,mixed.Count(0),mixed.dataType}}};
    if(residual)benchmark.outputs.push_back({residual->cudaData,residual->Count(0),FLOAT32});
    if(injection)benchmark.outputs.push_back({injection->cudaData,injection->Count(0),FLOAT32});
    uint32_t epsilon;std::memcpy(&epsilon,&eps,sizeof(epsilon));
    kernels::Signature signature{{11,FastllmCudaRuntimeArch(),FastllmCudaGetDevice(),rows,
        input.dims.back(),down.dims[0],groups,block?int(block->dataType):-1,
        oldInjection?int(oldInjection->dataType):-1,bool(inject),int(mixed.dataType) | (int(referenceMode)<<8),epsilon}};
    const int selected=kernels::SelectCudaOutputs(hyperReadFamily,signature,benchmark,0);
    return run(selected<0?0:selected);
}

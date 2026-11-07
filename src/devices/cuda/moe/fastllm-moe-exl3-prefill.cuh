#pragma once

namespace {
__global__ void Exl3PrefillGather(const uint16_t *input, uint16_t *output,
        const int32_t *rows, int hidden) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < hidden) output[size_t(blockIdx.y)*hidden+c] = input[size_t(rows[blockIdx.y])*hidden+c];
}
__global__ void Exl3PrefillScatter(const uint16_t *input, float *output,
        const int32_t *rows, const float *scores, int hidden) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < hidden) output[size_t(rows[blockIdx.y])*hidden+c] += scores[blockIdx.y] *
        __uint_as_float(uint32_t(input[size_t(blockIdx.y)*hidden+c]) << 16);
}

// One layer owns at most two temporary expert pairs. Cache hits and admissions
// borrow the canonical compact record directly; misses use the existing
// bounded pageable-weight uploader. Copy the next expert while this one runs.
bool RunExl3CachedPrefill(const fastllm::Data &input, const fastllm::Data &index,
        const fastllm::Data &score, fastllm::Data &output, fastllm::Data **weights,
        int weightsBatch, const std::function<void()> &launchParallel) {
    using namespace fastllm;
    const int rows = input.dims[0], hidden = input.dims[1];
    if (input.dataDevice != CUDA || input.dataType != BFLOAT16 || !input.cudaData ||
        !PackedCacheRows(input) || index.dims.size() != 2 || index.dims != score.dims ||
        index.dims[0] != rows || index.dims[1] < 1 || index.dims[1] > kMaxTopK ||
        index.dataType != INT32 || score.dataType != FLOAT32 ||
        !PackedCacheRows(index) || !PackedCacheRows(score)) return false;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return false;
    const int experts = weightsBatch/2-1, topk = index.dims[1];
    std::vector<int32_t> ids(size_t(rows)*topk);
    std::vector<float> scales(ids.size());
    if (index.dataDevice == CUDA) FastllmCudaCopyFromDeviceToHost(ids.data(), index.cudaData, ids.size()*sizeof(int32_t));
    else std::memcpy(ids.data(), index.cpuData, ids.size()*sizeof(int32_t));
    if (score.dataDevice == CUDA) FastllmCudaCopyFromDeviceToHost(scales.data(), score.cudaData, scales.size()*sizeof(float));
    else std::memcpy(scales.data(), score.cpuData, scales.size()*sizeof(float));
    std::vector<std::vector<int32_t>> bins(experts);
    std::vector<std::vector<float>> binScales(experts);
    std::unordered_set<int> selected;
    for (int row=0; row<rows; ++row) for (int j=0; j<topk; ++j) {
        const int at=row*topk+j, e=ids[at];
        if (e<0 || e>=experts || !std::isfinite(scales[at])) return false;
        // The scatter is deterministic because each expert owns unique rows.
        // Unexpected duplicate router entries retain the general fallback.
        for (int previous=0; previous<j; ++previous) if (ids[row*topk+previous]==e) return false;
        bins[e].push_back(row); binScales[e].push_back(scales[at]); selected.insert(e+1);
    }
    FastllmCudaMoePrefillCache cache;
    if (!cache.Begin(weights, experts, ids.data(), scales.data(), rows, topk, selected, true)) return false;
    struct Transfers {
        cudaStream_t stream=nullptr;
        cudaEvent_t ready[2]{}, done=nullptr;
        Transfers() {
            checkCudaErrors("EXL3 prefill copy stream", cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            for(auto &event:ready) checkCudaErrors("EXL3 prefill ready event",cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
            checkCudaErrors("EXL3 prefill compute event",cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
        }
        ~Transfers() {
            cudaStreamSynchronize(stream); cudaStreamSynchronize(cudaStreamPerThread);
            for(auto event:ready) cudaEventDestroy(event);
            cudaEventDestroy(done); cudaStreamDestroy(stream);
        }
    } transfer;
    const int device=FastllmCudaGetDevice(), mid=weights[2]->dims[0]/2;
    std::vector<int> active, offsets;
    std::vector<int32_t> gatherRows;
    std::vector<float> gatherScales;
    int capacity=0;
    for(int e=0;e<experts;++e) if(!bins[e].empty()) {
        active.push_back(e); offsets.push_back(gatherRows.size());
        capacity=std::max(capacity,int(bins[e].size()));
        gatherRows.insert(gatherRows.end(),bins[e].begin(),bins[e].end());
        gatherScales.insert(gatherScales.end(),binScales[e].begin(),binScales[e].end());
    }
    Data rowIds(INT32,{int(gatherRows.size())}), routeScales(FLOAT32,{int(gatherScales.size())});
    rowIds.Allocate(false); routeScales.Allocate(false);
    std::memcpy(rowIds.cpuData,gatherRows.data(),gatherRows.size()*sizeof(int32_t));
    std::memcpy(routeScales.cpuData,gatherScales.data(),gatherScales.size()*sizeof(float));
    rowIds.ToDevice(CUDA,std::vector<int>{device}); routeScales.ToDevice(CUDA,std::vector<int>{device});
    if(output.dataType!=FLOAT32) output.FreeSpace();
    output.dataType=FLOAT32; output.UpdateUnitSize(); output.Resize(input.dims);
    output.ToDevice(CUDA,{device},false); output.Allocate(false);
    checkCudaErrors("EXL3 prefill zero",cudaMemsetAsync(output.cudaData,0,output.GetBytes(),cudaStreamPerThread));
    Data gathered(BFLOAT16,{capacity,hidden}), middle(BFLOAT16,{capacity,2*mid}),
         activation(BFLOAT16,{capacity,mid}), result(BFLOAT16,{capacity,hidden}), empty;
    for(auto *tensor:{&gathered,&middle,&activation,&result}) {
        tensor->ToDevice(CUDA,{device},false); tensor->Allocate(false);
    }
    auto upload=[&](int i) {
        cache.Upload(2+2*active[i],transfer.stream); cache.Upload(3+2*active[i],transfer.stream);
        checkCudaErrors("EXL3 prefill weight ready",cudaEventRecord(transfer.ready[i%2],transfer.stream));
    };
    if(launchParallel) launchParallel();
    if(!active.empty()) upload(0);
    auto table=cache.Weights();
    for(int i=0;i<int(active.size());++i) {
        const int e=active[i], count=bins[e].size(), offset=offsets[i];
        checkCudaErrors("EXL3 prefill wait weights",cudaStreamWaitEvent(cudaStreamPerThread,transfer.ready[i%2],0));
        gathered.Resize({count,hidden}); middle.Resize({count,2*mid});
        activation.Resize({count,mid}); result.Resize({count,hidden});
        auto *selectedRows=static_cast<int32_t *>(rowIds.cudaData)+offset;
        auto *selectedScales=static_cast<float *>(routeScales.cudaData)+offset;
        Exl3PrefillGather<<<dim3((hidden+255)/256,count),256,0,cudaStreamPerThread>>>(
            static_cast<const uint16_t *>(input.cudaData),static_cast<uint16_t *>(gathered.cudaData),selectedRows,hidden);
        AssertInFastLLM(FastllmCudaExl3Linear(gathered,*table[2+2*e],empty,middle),"EXL3 prefill gate failed");
        AssertInFastLLM(FastllmCudaDeepSeekV41SharedSwiglu(middle,
            FindGroup(weights,weightsBatch)->layout.swigluLimit,activation),"EXL3 prefill activation failed");
        AssertInFastLLM(FastllmCudaExl3Linear(activation,*table[3+2*e],empty,result),"EXL3 prefill down failed");
        Exl3PrefillScatter<<<dim3((hidden+255)/256,count),256,0,cudaStreamPerThread>>>(
            static_cast<const uint16_t *>(result.cudaData),static_cast<float *>(output.cudaData),selectedRows,selectedScales,hidden);
        checkCudaErrors("EXL3 prefill compute",cudaEventRecord(transfer.done,cudaStreamPerThread));
        if(i+1<int(active.size())) upload(i+1);
        checkCudaErrors("EXL3 prefill temporary lifetime",cudaEventSynchronize(transfer.done));
        cache.Store(e); cache.Release(2+2*e); cache.Release(3+2*e);
    }
    cache.Finish();
    return true;
}
}

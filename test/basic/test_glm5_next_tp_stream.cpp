#ifdef USE_CUDA
#include "models/glm5_next.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <stdexcept>

namespace fastllm {
struct Glm5NextGGUFTestAccess {
    static void SetMtp(Glm5NextModel &m, bool enabled) { m.mtpEnabled = enabled; }
    static void Init(Glm5NextModel &m) {
        m.block_cnt = 0;
        m.embed_dim = 4096;
        m.hcMult = 4;
        m.kdaHeads = m.num_attention_heads = 4;
        m.denseMlpLayers.clear();
        m.kdaLayers.clear();
        m.mtpEnabled = true;
        m.deviceMap = {{"cuda:0", 1}};
        m.moeDeviceMap = {{"cpu", 1}};
        m.weight.dicts["gguf_architecture"] = "glm5next";
        for (const auto &kv : std::map<std::string,std::string>{
            {"hidden_size","4096"},{"num_hidden_layers","45"},{"num_attention_heads","64"},
            {"linear_attn_config.num_heads","64"},{"linear_attn_config.head_dim","128"},
            {"linear_attn_config.short_conv_kernel_size","4"},{"linear_attn_config.gate_lower_bound","-5"},
            {"q_lora_rank","1536"},{"kv_lora_rank","512"},{"qk_nope_head_dim","256"},
            {"qk_rope_head_dim","0"},{"v_head_dim","256"},{"intermediate_size","12288"},
            {"moe_intermediate_size","2048"},{"first_k_dense_replace","3"},{"swiglu_limit","7"},
            {"hc_mult","4"},{"hc_sinkhorn_iters","20"},{"hc_eps","0.00001"},{"rms_norm_eps","0.00001"},
            {"index_topk","2048"},{"index_n_heads","32"},{"index_head_dim","128"},{"index_kpool","4"},
            {"max_position_embeddings","32768"},{"num_nextn_predict_layers","1"},
            {"n_routed_experts","288"},{"num_experts_per_tok","8"},{"n_shared_experts","1"},
            {"routed_scaling_factor","2.5"}}) m.weight.dicts[kv.first]=kv.second;
        std::string kda;
        for (int i=0;i<45;++i) if(i%4!=3) kda+=std::to_string(i)+",";
        m.weight.dicts["linear_attn_config.kda_layers"]=kda;
        std::vector<float> matrix(8 * 4096, 0), norm(4096, 1);
        for (int i = 0; i < 8; ++i) matrix[i * 4096 + i] = 1;
        m.weight["model.language_model.embed_tokens.weight"].CopyFrom(Data(BFLOAT16, {8,4096}, matrix));
        m.weight["lm_head.weight"].CopyFrom(Data(BFLOAT16, {8,4096}, matrix));
        m.weight["model.language_model.norm.weight"].CopyFrom(Data(FLOAT32, {4096}, norm));
        m.InitThreadTp();
    }
    static int Forward(Glm5NextModel &m, const Data &ids) {
        std::vector<std::pair<Data,Data>> cache;
        GenerationConfig config;
        config.top_k = 1;
        LastTokensManager history;
        const int result = m.ForwardThreadTp(ids, cache, config, history, nullptr);
        m.RemoveThreadTpRequest(&cache);
        return result;
    }
    static void CheckCoordinatorEmbedding(Glm5NextModel &m) {
        // The MTP draft uses the coordinator's embedding after rank zero has
        // borrowed its CPU storage. Enabling CUDA embedding must not move the
        // owner and invalidate that rank's view before the next target step.
        auto &weight = m.weight["model.language_model.embed_tokens.weight"];
        auto *storage = weight.cpuData;
        SetCudaEmbedding(true);
        Data ids(FLOAT32, {1,1}, {3}), hidden;
        m.ForwardEmbedding(ids, hidden);
        if (weight.dataDevice != DataDevice::CPU || weight.cpuData != storage)
            throw std::runtime_error("coordinator embedding invalidated TP CPU storage");
        if (Forward(m, ids) != 3)
            throw std::runtime_error("TP embedding changed after coordinator lookup");
        std::cout << "PASS: coordinator embedding retains TP CPU storage" << std::endl;
    }
};
}
static void CUDART_CB Delay(void *) { std::this_thread::sleep_for(std::chrono::milliseconds(80)); }
static void CheckCuda(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
int main() {
    using namespace fastllm;
    if (FastllmCudaGetDeviceCount() < 2) return 77;
#ifdef _WIN32
    _putenv_s("FASTLLM_TP", "0,1");
#else
    setenv("FASTLLM_TP", "0,1", 1);
#endif
    SetThreads(2);
    SetCudaSharedExpert(true);
    SetCudaEmbedding(false);
    Glm5NextModel model;
    Glm5NextGGUFTestAccess::Init(model);
    Data ids(FLOAT32, {1,1}, {0});
    ids.ToDevice(CUDA, std::vector<int>{0});
    if (Glm5NextGGUFTestAccess::Forward(model, ids) != 0) throw std::runtime_error("toy model warmup failed");
    Glm5NextGGUFTestAccess::CheckCoordinatorEmbedding(model);
    float *pinned = nullptr;
    CheckCuda(cudaMallocHost(reinterpret_cast<void **>(&pinned), sizeof(float)));
    for (bool enabled : {false, true}) {
        Glm5NextGGUFTestAccess::SetMtp(model, enabled);
        int correct = 0;
        for (int i = 1; i <= 4; ++i) {
            CheckCuda(cudaSetDevice(0));
            *pinned = 0;
            CheckCuda(cudaMemcpy(ids.cudaData, pinned, sizeof(float), cudaMemcpyHostToDevice));
            CheckCuda(cudaDeviceSynchronize());
            *pinned = float(i);
            CheckCuda(cudaLaunchHostFunc(cudaStreamPerThread, Delay, nullptr));
            CheckCuda(cudaMemcpyAsync(ids.cudaData, pinned, sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
            const int token = Glm5NextGGUFTestAccess::Forward(model, ids);
            CheckCuda(cudaSetDevice(0));
            CheckCuda(cudaDeviceSynchronize());
            correct += token == i;
            std::cout << "HANDOFF mtp=" << enabled << " expected=" << i << " got=" << token << std::endl;
        }
        if (correct != 4) throw std::runtime_error("event handoff failed");
    }
    CheckCuda(cudaFreeHost(pinned));
}

#else
int main() { return 77; }
#endif

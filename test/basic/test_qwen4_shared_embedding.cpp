#include "models/qwen4_shared_embedding.h"

#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace fastllm;
using namespace fastllm::qwen4_pipeline;

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> static void Reject(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception &) { rejected = true; }
    Check(rejected, "unsafe embedding operation accepted");
}

static float Expand(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float output; std::memcpy(&output,&bits,sizeof(output)); return output;
}

int main() {
    try {
        constexpr int vocab=17, hidden=2560;
        std::vector<uint16_t> payload(vocab*hidden);
        for (size_t i=0;i<payload.size();++i) payload[i]=0x3d00+(i%127);
        const auto original=payload;
        auto embedding=SharedTokenEmbedding::FromOwnedBf16(vocab,hidden,std::move(payload),1024);
        const std::vector<float> ids{16,0,8,8,1};
        auto rows=embedding.GatherTokenRows(ids,42);
        Check(rows.request_epoch==42,"request epoch lost");
        Check(rows.bf16_rows->dims==std::vector<int>({1,5,hidden}),"embedding shape changed");
        Check(rows.bf16_rows->dataType==BFLOAT16 && rows.fp32_rows->dataType==FLOAT32,
              "embedding source/calculation types changed");
        Check(rows.bf16_rows->dataDevice==DataDevice::CPU && rows.fp32_rows->dataDevice==DataDevice::CPU,
              "row lookup moved to CUDA");
        const auto *actual=reinterpret_cast<const uint16_t *>(rows.bf16_rows->cpuData);
        const auto *expanded=reinterpret_cast<const float *>(rows.fp32_rows->cpuData);
        for (int r=0;r<5;++r) for (int h=0;h<hidden;++h) {
            const auto expected=original[static_cast<int>(ids[r])*hidden+h];
            Check(actual[r*hidden+h]==expected,"BF16 source row byte mismatch");
            Check(expanded[r*hidden+h]==Expand(expected),"BF16 to FP32 row mismatch");
        }
        Check(rows.fp32_rows->GetBytes()==5*hidden*sizeof(float),"entire table converted/uploaded");
        Reject([&]{embedding.GatherTokenRows({-1},42);});
        Reject([&]{embedding.GatherTokenRows({17},42);});
        Reject([&]{embedding.GatherTokenRows({0.5f},42);});
        Reject([&]{embedding.GatherTokenRows({std::numeric_limits<float>::quiet_NaN()},42);});
        Reject([&]{embedding.GatherTokenRows({std::numeric_limits<float>::infinity()},42);});
        Reject([&]{embedding.GatherTokenRows(std::vector<float>(1025,0),42);});
        Reject([&]{embedding.GatherTokenRows({},42);});
        Reject([&]{SharedTokenEmbedding::FromOwnedBf16(17,2560,{},1024);});
        SetLowMemMode(true);
        Reject([&]{embedding.GatherTokenRows(ids,42);});
        Check(GetLowMemMode(),"embedding changed process low-memory mode");
        SetLowMemMode(false);
        std::vector<std::future<void>> readers;
        for (int stage=0;stage<2;++stage) readers.emplace_back(std::async(std::launch::async,[&,stage]{
            for (int turn=0;turn<20;++turn) {
                auto local=embedding.GatherTokenRows(ids,static_cast<uint64_t>(stage+turn));
                Check(local.bf16_rows.get()!=rows.bf16_rows.get(),"calls share mutable Data metadata");
                Check(std::memcmp(local.bf16_rows->cpuData,rows.bf16_rows->cpuData,
                                  rows.bf16_rows->GetBytes())==0,"concurrent lookup corrupted rows");
            }
        }));
        for (auto &reader:readers) reader.get();
        Check(embedding.PayloadBytes()==static_cast<uint64_t>(vocab)*hidden*2,"payload was replicated");
        std::cout<<"PASS CPU EmbeddingDirect, exact BF16 rows, FP32 expansion, bounds, row budget and concurrency\n";
        return 0;
    } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}

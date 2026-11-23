#include "models/naive_n05_flash.h"
#include "json11.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime_api.h>

using namespace fastllm;

// Checkpoint-backed oracle test. Fixtures are produced by the original Torch
// draft implementation, independently of FastLLM's attention/cache code.
class NaiveDraftFixture : public NaiveN05FlashModel {
public:
    double referenceRounding = 0;
    void Read(const std::string &directory) {
        std::ifstream stream(directory + "/manifest.json");
        std::string text((std::istreambuf_iterator<char>(stream)), {}), error;
        auto manifest = json11::Json::parse(text, error);
        if (!error.empty()) throw std::runtime_error(error);
        embed_dim = manifest["hidden_size"].int_value();
        draftLayers = manifest["layers"].int_value();
        draftHeads = manifest["heads"].int_value();
        draftKvHeads = manifest["kv_heads"].int_value();
        draftHeadDim = manifest["head_dim"].int_value();
        draftWindow = manifest["window"].int_value();
        draftBlock = manifest["block"].int_value();
        draftEps = manifest["eps"].number_value();
        draftTheta = manifest["theta"].number_value();
        referenceRounding = manifest["torch_bf16_vs_fp32_rmse"].number_value();
        deviceMap = {{"cuda:0", 1}};
        block_cnt = 48;
        for (const auto &entry : manifest["tensors"].array_items()) {
            std::string name = entry["name"].string_value();
            std::vector<int> shape;
            for (auto &d : entry["shape"].array_items()) shape.push_back(d.int_value());
            auto type = entry["dtype"].string_value() == "float32" ? FLOAT32 : BFLOAT16;
            weight.AddEmptyWeight(name, shape, type);
            Data &data = weight[name];
            data.Allocate();
            std::ifstream input(directory + "/" + entry["file"].string_value(), std::ios::binary);
            if (!input.read((char *)data.cpuData, data.GetBytes()))
                throw std::runtime_error("Cannot read tensor: " + name);
        }
    }
    void RunBatch() {
        SetCudaEmbedding(true); SetThreads(16); maxBatch = 64;
        setenv("FASTLLM_DSPARK_TP", "8", 1);
        AssertInFastLLM(PrepareDraftTP(), "fixture TP preparation failed");
        auto &input = weight["test.context"];
        auto &reference = weight["test.expected"];
        AssertInFastLLM(input.dims.size() == 3 && input.dims[0] >= 64 && input.dims[2] == embed_dim &&
            reference.dims == std::vector<int>({input.dims[0], draftBlock, embed_dim}),
            "batch fixture requires at least 64 contexts and matching reference rows");
        const int capacity = input.dims[0], length = input.dims[1];
        input.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        std::vector<std::shared_ptr<DraftContext>> owners;
        for(int i=0;i<capacity;++i) {
            auto context=std::make_shared<DraftContext>();Data view;
            view.Resize({1,length,embed_dim});view.FakeFrom(input,size_t(i)*length*embed_dim*sizeof(uint16_t));
            AppendDraftContext(view,0,*context);owners.push_back(context);
        }
        // Establish each request's retained scalar result against the independent
        // oracle before admitting a wider batch. Chained BF16 rounding is distinct
        // from the individual operator checks in the CUDA tests.
        std::vector<float> scalar(size_t(capacity)*draftBlock*embed_dim);
        for(int i=0;i<capacity;++i) {
            Data single;preparingGraphs=true;
            AssertInFastLLM(RunDraftBatch({i},{owners[i].get()},single),"scalar draft fixture failed");
            preparingGraphs=false;ToDataType(single,FLOAT32);single.ToDevice(DataDevice::CPU);
            std::memcpy(scalar.data()+size_t(i)*draftBlock*embed_dim,single.cpuData,size_t(draftBlock)*embed_dim*sizeof(float));
        }
        for(int count:{8,16,32,64,8}) {
            std::vector<DraftContext*> contexts;std::vector<int> anchors;
            for(int i=0;i<count;++i){contexts.push_back(owners[i].get());anchors.push_back(i);}
            Data actual;
            preparingGraphs=true;
            AssertInFastLLM(RunDraftBatch(anchors,contexts,actual),"batch draft rejected fixture");
            preparingGraphs=false;
            ToDataType(actual,FLOAT32);actual.ToDevice(DataDevice::CPU);
            double error=0,scale=0,worst=0,scalarWorst=0,deltaWorst=0;
            for(int row=0;row<count*draftBlock;++row) {
                double e=0,b=0,se=0,sb=0,delta=0;
                for(int col=0;col<embed_dim;++col) {
                    const int i=row*embed_dim+col;
                    const double got=((float*)actual.cpuData)[i],want=((float*)reference.cpuData)[i],base=scalar[i];
                    e+=(got-want)*(got-want); b+=want*want;
                    se+=(base-want)*(base-want); sb+=base*base;
                    delta+=(got-base)*(got-base);
                }
                const double oracle=std::sqrt(e/std::max(1e-30,b));
                const double baseline=std::sqrt(se/std::max(1e-30,b));
                const double change=std::sqrt(delta/std::max(1e-30,sb));
                if(!std::isfinite(oracle) || !std::isfinite(baseline) || !std::isfinite(change) || baseline>.05 || oracle>1.5*baseline+.003 || change>.04)
                    throw std::runtime_error("wide body exceeded scalar/oracle error bounds");
                worst=std::max(worst,oracle);scalarWorst=std::max(scalarWorst,baseline);deltaWorst=std::max(deltaWorst,change);error+=e;scale+=b;
            }
            std::cout<<"BATCH_ORACLE count="<<count<<" l2="<<std::sqrt(error/scale)<<" max_row_l2="<<worst<<" scalar_oracle_max="<<scalarWorst<<" wide_scalar_max="<<deltaWorst<<std::endl;
            Data again;
            AssertInFastLLM(RunDraftBatch(anchors,contexts,again),"batch replay rejected fixture");
            ToDataType(again,FLOAT32);again.ToDevice(DataDevice::CPU);
            if(std::memcmp(actual.cpuData,again.cpuData,actual.GetBytes()))throw std::runtime_error("batch replay changed immutable context");
        }
    }

    void Run() {
        Data &hidden = weight["test.context"];
        const int length = hidden.dims[1], split = length / 2;
        DraftContext context;
        Data first, last;
        Split(hidden, 1, 0, split, first);
        Split(hidden, 1, split, length, last);
        AppendDraftContext(first, 0, context);
        AppendDraftContext(last, split, context);
        if (context.committed != length || context.kv[0].first.dims[1] != std::min(length, draftWindow - 1))
            throw std::runtime_error("Draft context append/trim failed");
        Data actual = RunDraft(0, context);
        ToDataType(actual, FLOAT32);
        actual.ToDevice(DataDevice::CPU);
        Data &expected = weight["test.expected"];
        double squared = 0, dot = 0, aa = 0, bb = 0, maximum = 0;
        for (int i = 0; i < actual.Count(0); ++i) {
            double a = ((float *)actual.cpuData)[i], b = ((float *)expected.cpuData)[i];
            maximum = std::max(maximum, std::abs(a - b));
            squared += (a - b) * (a - b); dot += a * b; aa += a * a; bb += b * b;
        }
        double rmse = std::sqrt(squared / actual.Count(0)), cosine = dot / std::sqrt(aa * bb);
        std::cout << "draft_hidden_rmse=" << rmse << " cosine=" << cosine << " max_abs=" << maximum << '\n';
        // Quantify the BF16 error with the independent FP32 pass, rather than
        // imposing an absolute threshold on checkpoint-specific norm scales.
        if (!std::isfinite(rmse) || rmse > std::max(.02, 1.5 * referenceRounding) || cosine < .998)
            throw std::runtime_error("Draft differs from original Torch implementation");
        Data repeat = RunDraft(0, context);
        ToDataType(repeat, FLOAT32); repeat.ToDevice(DataDevice::CPU);
        if (std::memcmp(actual.cpuData, repeat.cpuData, actual.GetBytes()) != 0)
            throw std::runtime_error("Draft proposal contaminated the committed KV cache");
    }
};

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "batch")) {
        std::cerr << "Usage: naive_n05_draft_test FIXTURE_DIRECTORY [batch]\n"; return 2;
    }
    const bool batch = argc == 3;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < (batch ? 8 : 1)) return 77;
    try {
        NaiveDraftFixture model;
        model.Read(argv[1]);
        if(batch) model.RunBatch(); else model.Run();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}

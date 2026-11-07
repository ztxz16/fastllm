#include "utils.h"
#include "../../src/weights/safetensors_source.h"
#include "../../src/weights/row_concat.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>

using namespace fastllm;
struct Item {
    std::string tensorName, fileName, dtype;
    std::vector<uint64_t> shape, data_offsets;
    std::vector<int> intShape;
    uint64_t bytes = 0, len = 0;
    int blockK = -1, blockM = -1;
    std::shared_ptr<const TensorSource> source;
};
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("fastllm-weight-source-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    struct Cleanup { std::filesystem::path path; ~Cleanup(){ std::filesystem::remove_all(path); } } cleanup{root};
    try {
        for (int bits : {2,3,4,5,6,7,8,10,12,14,16}) {
            std::map<std::string, Item> items;
            std::vector<uint8_t> expected(Exl3PartBytes(128,128,bits));
            for (size_t i=0;i<expected.size();++i) expected[i] = uint8_t(i*17+bits);
            auto add = [&](const std::string &name, std::vector<int> shape, const char *dtype,
                           const uint8_t *data, size_t size) {
                auto file = root / name;
                std::ofstream stream(file, std::ios::binary);
                // Exercise offsets and separate component files, not only one contiguous range.
                stream.write("padding",7); stream.write((const char*)data,size); stream.close();
                Item item; item.tensorName=name; item.fileName=file.string(); item.dtype=dtype;
                item.intShape=shape; item.shape.assign(shape.begin(),shape.end()); item.len=1;
                for (int n:shape) item.len*=n;
                item.bytes=size; item.data_offsets={7,7+size}; items[name]=item;
            };
            const size_t trellis=expected.size()-512; const uint32_t marker=0x83dcd12d;
            add("head.trellis",{8,8,bits*8},"I16",expected.data(),trellis);
            add("head.suh",{128},"F16",expected.data()+trellis,256);
            add("head.svh",{128},"F16",expected.data()+trellis+256,256);
            add("head.mul1",{},"I32",(const uint8_t*)&marker,4);
            NormalizeSafeTensorSources("exl3",items);
            Check(items.size()==1 && items.count("head.weight"),"logical source directory");
            auto source=items.at("head.weight").source;
            items.clear(); // Reader must own its ranges after inventory temporaries go away.
            Check(source && source->StoredType()==EXL3,"retained encoding");
            std::unique_ptr<uint8_t[]> actual(source->Read(EXL3));
            Check(std::memcmp(actual.get(),expected.data(),expected.size())==0,"source byte order");
            Data first(EXL3), second(EXL3), merged(EXL3);
            source->Configure(first);source->Configure(second);source->Configure(merged);
            first.Resize({128,128});second.Resize({128,128});merged.Resize({256,128});
            Check(CanConcatWeightRows(first,second),"compatible packed projections");
            ConfigureWeightRowConcat(merged,{&first,&second});
            Check(merged.blockM==2 && merged.GetBytes()==2*expected.size(),"merged payload geometry");
            second.blockK=bits==2?4:2;
            Check(!CanConcatWeightRows(first,second),"mixed bitrate concat accepted");
            second.blockK=bits;second.Resize({256,128});
            Check(!CanConcatWeightRows(first,second),"unequal projection concat accepted");
        }
        Data dense(FLOAT16,{128,128}), other(FLOAT16,{256,128});
        Check(CanConcatWeightRows(dense,other),"ordinary row concat");
        Data blocked(FP8_E4M3_BLOCK_128,{129,128}); blocked.blockK=128;blocked.blockM=128;
        Check(!CanConcatWeightRows(blocked,blocked),"partial quantization block concat accepted");
        std::cout << "PASS: compound weight sources and common row concatenation\n";
        return 0;
    } catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}

#include "fastllm.h"
#include "executor.h"
#include "utils/autotune.h"
#include "json11.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>
using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    const std::string path = "/tmp/fastllm-autotune-test-"+std::to_string(getpid())+".jsonl";
    try {
        SetThreads(2);
        Executor executor;
        executor.SetFirstDevice("cpu");
        auto previous = GetExecutor();
        SetCurrentThreadExecutor(&executor);
        Data input(FLOAT32,{2,32}), weight(FLOAT32,{32,32}), bias(FLOAT32), a, b;
        input.Allocate(); weight.Allocate();
        for (int i=0;i<64;++i) ((float*)input.cpuData)[i]=std::sin(i*.1f);
        for (int i=0;i<1024;++i) ((float*)weight.cpuData)[i]=(i%7-3)*.01f;
        Linear(input,weight,bias,a);
        Check(!AutotuneRecording(),"recording enabled by default");
        Check(!StartAutotuneRecording(path+"/missing"),"invalid path accepted");
        Check(StartAutotuneRecording(path),"capture start failed");
        Check(!StartAutotuneRecording(path),"overlapping capture accepted");
        Linear(input,weight,bias,b);
        Data ids(INT32,{2,1}); ids.Allocate();
        ((int*)ids.cpuData)[0]=0; ((int*)ids.cpuData)[1]=1;
        std::vector<Data*> weights{nullptr,nullptr,&weight,&weight,&weight,&weight};
        // Exercise the same metadata/route boundary used by Executor, with a
        // nested scope that must not double-count the enclosing operator.
        {
            DataDict data{{"input",&input},{"index",&ids},{"weights",(Data*)weights.data()}};
            AutotuneOpScope outer("MergeMOE","cpu",data,{{"layer",3},{"weights___batch",6}});
            AutotuneOpScope nested("nested","cpu",{},{});
        }
        std::thread first([] { for(int i=0;i<16;++i) RecordAutotuneSample("lane",{1},2,3); });
        std::thread second([] { for(int i=0;i<16;++i) RecordAutotuneSample("lane",{2},2,3); });
        first.join();second.join();
        Check(StopAutotuneRecording(),"capture close failed");
        Check(!StopAutotuneRecording(),"duplicate close accepted");
        for(int i=0;i<64;++i) Check(((float*)a.cpuData)[i]==((float*)b.cpuData)[i],"capture changed inference output");
        int routes=0,lanes=0,complete=0;
        std::ifstream file(path); std::string line,error;
        while(std::getline(file,line)) {
            auto value=json11::Json::parse(line,error);
            Check(error.empty(),"interleaved/invalid capture JSON");
            Check(value["op"].string_value()!="nested","nested operator double-counted");
            if(value["op"].string_value()=="MergeMOE") {
                Check(value["ids"].array_items().size()==2 && value["ids"][1].int_value()==1,"routes changed");
                Check(value["bytes"].array_items().size()==2,"expert byte sizes missing");
                ++routes;
            }
            lanes += value["kind"].string_value()=="lane";
            complete += value["kind"].string_value()=="complete";
        }
        Check(routes==1 && lanes==32 && complete==1,"missing capture records");
        file.close();
        Check(StartAutotuneRecording(path),"bounded capture start failed");
        {
            AutotuneOpScope outer("overflow","cpu",{},{});
            AutotuneOpScope nested("nested","cpu",{},{});
            const std::string payload(1024*1024, 'x');
            for(int i=0;i<65 && AutotuneRecording();++i)
                RecordAutotuneSample(payload,{},1,1);
            Check(!AutotuneRecording(),"capture exceeded byte limit");
        }
        Check(!StopAutotuneRecording(),"truncated capture reported complete");
        std::ifstream bounded(path,std::ios::binary|std::ios::ate);
        Check(bounded.tellg()<=64*1024*1024,"capture file exceeded byte limit");
        bounded.close();
        Check(StartAutotuneRecording(path),"capture could not restart after overflow");
        { AutotuneOpScope recovered("recovered","cpu",{},{}); }
        Check(StopAutotuneRecording(),"restarted capture failed");
        std::ifstream recovered(path);
        std::getline(recovered,line);
        Check(json11::Json::parse(line,error)["op"].string_value()=="recovered","nested depth leaked after overflow");
        SetCurrentThreadExecutor(previous);
        std::remove(path.c_str());
        puts("PASS: bounded opt-in autotune capture, routes, threads and unchanged inference");
    } catch(const std::exception &error) {
        std::remove(path.c_str());fprintf(stderr,"%s\n",error.what());return 1;
    }
}

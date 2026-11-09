#include "models/basellm.h"
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
class Fixture : public basellm {
public:
    using basellm::ShutdownRuntime;
    std::string MakeInput(const std::string &, int, const std::string &) override { return {}; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &) override { return {}; }
    int Create(int count, bool ending = true, ResponseContextError error = ResponseContextErrorNone) {
        std::lock_guard<std::mutex> lock(dictLocker);
        int h = responseContextDict.CreateHandle();
        auto *c = responseContextDict.GetHandle(h);
        c->inputTokens = 100; c->cacheLen = 80; c->curTokens = count;
        c->isEnding = ending; c->error = error;
        for (int i=0; i<count; ++i) c->resultTokenQueue.push(10+i);
        return h;
    }
    void Finish(int h) {
        std::lock_guard<std::mutex> lock(dictLocker);
        responseContextDict.GetHandle(h)->isEnding = true;
        NotifyResponseReady();
    }
    uint64_t Replace(int h) {
        std::lock_guard<std::mutex> lock(dictLocker);
        RemoveResponseContext(h); responseContextDict.nextHandle = h;
        Check(responseContextDict.CreateHandle()==h, "reuse failed");
        auto *c = responseContextDict.GetHandle(h);
        c->resultTokenQueue.push(123); c->isEnding = true;
        NotifyResponseReady(); return c->generation;
    }
};
int main() {
    try {
        Fixture m;
        for (int concurrency : {1,4,16,128,256,1024}) for (int width : {1,3,5,7,65}) {
            std::vector<int> handles, tokens(concurrency*3), records(concurrency*7), received(concurrency), done(concurrency);
            std::vector<uint64_t> generations;
            for (int i=0; i<concurrency; ++i) {
                handles.push_back(m.Create(width+i%3)); generations.push_back(m.GetResponseGeneration(handles.back()));
            }
            uint64_t version=0; int remaining=concurrency;
            for (int pass=0; remaining && pass<100; ++pass) {
                int n=m.FetchResponses(handles.data(),generations.data(),concurrency,3,
                    tokens.data(),tokens.size(),records.data(),concurrency,0,version);
                Check(n>0,"batch did not make progress");
                for (int j=0;j<n;++j) {
                    int *r=records.data()+j*7, i=r[0];
                    if (done[i]) { Check(r[2]==0 && r[3]==-1,"retired request returned tokens"); continue; }
                    for(int k=0;k<r[2];++k) Check(tokens[r[1]+k]==10+received[i]++,"token order/count changed");
                    Check(r[4]==80 && r[5]==20 && r[6]==width+i%3,"statistics changed");
                    if(r[3]) { Check(r[3]==-1 && received[i]==width+i%3,"terminal lost tail"); done[i]=1;--remaining; }
                }
            }
            Check(!remaining,"requests did not finish");
        }
        int token[64], record[7]; uint64_t version=0;
        for(auto error:{ResponseContextErrorPromptTooLong,ResponseContextErrorRuntime}) {
            int h=m.Create(2,true,error);uint64_t g=m.GetResponseGeneration(h);
            Check(m.FetchResponses(&h,&g,1,64,token,64,record,1,0,version)==1,"missing error record");
            Check(record[2]==2 && record[3]==(error==ResponseContextErrorPromptTooLong?-2:-3),"lost error/tail ordering");
            Check(record[4]==80 && record[6]==2,"terminal stats lost");
        }
        int h=m.Create(0,false);uint64_t g=m.GetResponseGeneration(h);
        m.WakeResponseOutput();
        auto start=std::chrono::steady_clock::now();
        Check(m.FetchResponses(&h,&g,1,64,token,64,record,1,1000,version)==0,"control wake returned data");
        Check(std::chrono::steady_clock::now()-start<std::chrono::milliseconds(200),"wake before read lost");
        auto waiter=std::async(std::launch::async,[&]{return m.FetchResponses(&h,&g,1,64,token,64,record,1,2000,version);});
        Check(waiter.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout,"empty read did not block");
        m.Finish(h);
        Check(waiter.wait_for(std::chrono::seconds(1))==std::future_status::ready,"publication did not wake reader");
        waiter.get();
        Check(m.FetchResponses(&h,&g,1,64,token,64,record,1,0,version)==1 && record[3]==-1,"finish missing");
        h=m.Create(0,false);g=m.GetResponseGeneration(h);uint64_t replacement=m.Replace(h);
        m.AbortResponse(h,g);
        Check(m.FetchResponses(&h,&g,1,64,token,64,record,1,0,version)==1 && record[2]==0 && record[3]==-1,"old identity reached replacement");
        Check(m.FetchResponses(&h,&replacement,1,64,token,64,record,1,0,version)==1 && token[0]==123,"replacement was consumed/cancelled");
        h=m.Create(2,false);g=m.GetResponseGeneration(h);m.AbortResponse(h,g);
        Check(m.FetchResponses(&h,&g,1,64,token,64,record,1,0,version)==1 && record[2]==0 && record[4]==-1,"cancelled data escaped");
        m.RemoveResponseContext(h);
        h=m.Create(0,false);g=m.GetResponseGeneration(h);
        m.FetchResponses(&h,&g,1,64,token,64,record,1,0,version);
        waiter=std::async(std::launch::async,[&]{return m.FetchResponses(&h,&g,1,64,token,64,record,1,2000,version);});
        Check(waiter.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout,"shutdown read did not block");
        m.ShutdownRuntime();
        Check(waiter.wait_for(std::chrono::seconds(1))==std::future_status::ready,"shutdown did not wake reader");
        waiter.get();
        Check(m.FetchResponses(&h,&g,1,64,token,64,record,1,0,version)==-1,"shutdown accepted a read");
        std::cout<<"ALL_PASS multi-request output, partial blocks, terminal stats, errors, cancellation, ABA and wake/shutdown\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}

#include "devices/disk/disk-weight-buffer-pool.h"
#include <stdexcept>
#define Require(...) do { if (!(__VA_ARGS__)) throw std::runtime_error(#__VA_ARGS__); } while (0)
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include <unistd.h>
#include <iostream>
using namespace fastllm;
static uint64_t limit = 1 << 20;
static uint64_t Budget() { return limit; }
struct Counters {
    uint64_t cpuWorkspaceBytes=0,cpuBufferReuses=0,cpuBufferAllocations=0;
    uint64_t cpuBufferSplits=0,cpuBufferCoalesces=0,cpuBufferMmaps=0,cpuBufferMunmaps=0;
};
static DiskWeightAllocation Mapped(size_t bytes) {
    void *p=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    Require(p!=MAP_FAILED);
    return DiskWeightAllocation(static_cast<uint8_t *>(p),DiskWeightDeleter{bytes});
}
int main() {
    const size_t page=sysconf(_SC_PAGESIZE);
    DiskWeightBufferPool pool(Budget);
    auto whole=Mapped(8*page); auto *base=whole.get();
    Require(pool.Retain(whole,8*page));
    auto a=pool.Acquire(3*page,true), b=pool.Acquire(5*page,true);
    Require(a.get()==base && b.get()==base+3*page);
    memset(a.get(),0x17,3*page); memset(b.get(),0x29,5*page);
    Require(a[3*page-1]==0x17 && b[0]==0x29);
    Require(pool.Retain(a,3*page) && pool.Retain(b,5*page));
    Counters s;pool.Stats(s);
    Require(s.cpuWorkspaceBytes==8*page && s.cpuBufferSplits==1 && s.cpuBufferCoalesces==1);
    whole=pool.Acquire(8*page,true);Require(whole.get()==base);
    Require(whole[0]==0x17 && whole[3*page]==0x29);
    Require(pool.Retain(whole,8*page));
    // Heap records retain exact-size semantics and never join mmap records.
    DiskWeightAllocation heap(new uint8_t[page],DiskWeightDeleter{});
    Require(pool.Retain(heap,page));Require(!pool.Acquire(page/2,false));
    heap=pool.Acquire(page,false);Require(heap && !heap.get_deleter().mappedBytes);
    // Concurrent mixed sizes exercise split ownership, non-overlap, trimming
    // and re-coalescing. Each writer verifies its whole range after yielding.
    std::vector<std::thread> workers;
    for(int t=0;t<8;++t) workers.emplace_back([&,t] {
        std::mt19937 rng(103+t);
        for(int i=0;i<500;++i) {
            size_t n=(1+rng()%16)*page;
            auto p=pool.Acquire(n,true);if(!p)p=Mapped(n);
            Require(p.get_deleter().mappedBytes==n);
            memset(p.get(),t+1,n);std::this_thread::yield();
            for(size_t j=0;j<n;++j)Require(p[j]==t+1);
            pool.Retain(p,n);
        }
    });
    for(auto &t:workers)t.join();
    pool.Stats(s);Require(s.cpuWorkspaceBytes<=limit);
    limit=0;pool.Trim();pool.Stats(s);Require(s.cpuWorkspaceBytes==0);
    auto p=Mapped(page);Require(!pool.Retain(p,page));
    pool.Clear();
    std::cout<<"PASS: disk mapped buffer split/coalesce, concurrency and budget"<<std::endl;
}

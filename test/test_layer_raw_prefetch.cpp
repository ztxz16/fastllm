#include "utils/layer_raw_prefetch.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
using namespace fastllm;
int main(int argc,char **argv) {
 if(argc!=2)return 2;
 const std::string file=argv[1];
 const std::vector<uint8_t> bytes={0,255,1,128,42,3,4,5,6,7};
 {std::ofstream f(file,std::ios::binary);f.write((char*)bytes.data(),bytes.size());}
 StreamingLoadBudget budget(32);
 BoundedLayerRawPrefetch prefetch(budget,16);
 const RawPrefetchRange wanted{"target",file,2,6};
 assert(prefetch.Start({wanted}));
 assert(budget.InFlight()==6);
 assert(!prefetch.Take(wanted)); // No access to buffers still being written.
 prefetch.Join();
 assert(!prefetch.Take({"target",file,1,6})); // Offset is part of the identity.
 auto payload=prefetch.Take(wanted);assert(payload);
 assert(!prefetch.Take(wanted)); // Ownership can be transferred only once.
 {
  ScopedPrefetchedRawTensor scope(payload);
  assert(ScopedPrefetchedRawTensor::Take("scale",file,2,6)==nullptr);
  auto raw=ScopedPrefetchedRawTensor::Take("target",file,2,6);assert(raw);
  for(int i=0;i<6;i++)assert(raw[i]==bytes[i+2]);
  assert(ScopedPrefetchedRawTensor::Take("target",file,2,6)==nullptr);
  delete[]raw;
 }
 payload.lease.Release();assert(budget.InFlight()==0);
 {auto held=budget.Acquire(30);assert(!prefetch.Start({wanted}));assert(budget.InFlight()==30);}
 assert(budget.InFlight()==0);
 {
  BoundedLayerRawPrefetch parallel(budget,16,4);
  std::vector<RawPrefetchRange> parts;
  for(int i=0;i<4;i++)parts.push_back({"part"+std::to_string(i),file,(uint64_t)i*2,2});
  assert(parallel.Start(parts));parallel.Join();
  for(int i=0;i<4;i++) {
   auto owned=parallel.Take(parts[i]);assert(owned);ScopedPrefetchedRawTensor scope(owned);
   auto raw=ScopedPrefetchedRawTensor::Take(parts[i].name,file,parts[i].offset,2);assert(raw);
   assert(raw[0]==bytes[i*2]&&raw[1]==bytes[i*2+1]);delete[]raw;
  }
  assert(parallel.Stats().readerWorkers==4&&parallel.Stats().adoptedCalls==4);
 }
 assert(budget.InFlight()==0);
 {
  BoundedLayerRawPrefetch concurrent(budget,16);
  assert(concurrent.Start({wanted}));concurrent.Join();
  std::atomic<int> owners{0};std::vector<std::thread> consumers;
  for(int j=0;j<8;j++)consumers.emplace_back([&]{
   auto owned=concurrent.Take(wanted);
   if(owned){ScopedPrefetchedRawTensor scope(owned);
    auto raw=ScopedPrefetchedRawTensor::Take("target",file,2,6);assert(raw);
    for(int i=0;i<6;i++)assert(raw[i]==bytes[i+2]);delete[]raw;owners++;
   }
  });
  for(auto &consumer:consumers)consumer.join();assert(owners==1);
 }
 assert(budget.InFlight()==0);
 assert(!prefetch.Start({{"large",file,0,17}})); // Window limit.
 assert(!prefetch.Start({wanted,wanted})); // Ambiguous tensor identity.
 assert(!prefetch.Start({{"zero",file,0,0}}));
 assert(prefetch.Start({{"short",file,8,6}}));prefetch.Join();
 assert(!prefetch.Take({"short",file,8,6})); // Read failure permits original path.
 assert(budget.InFlight()==0);
 {
  BoundedLayerRawPrefetch unfinished(budget,16);
  assert(unfinished.Start({wanted})); // Destructor joins and frees unused data.
 }
 assert(budget.InFlight()==0);
 auto stats=prefetch.Stats();
 assert(stats.adoptedCalls==1&&stats.adoptedBytes==6&&stats.readFailures==1);
 std::cout<<"PASS: exact ranges, raw byte identity, single ownership, hard budget, bounded window, failed read fallback, join/destruction"<<std::endl;
}

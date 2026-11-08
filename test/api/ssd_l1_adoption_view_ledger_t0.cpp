#include "utils/prefix_cache_page_ledger.h"
#include <cassert>
#include <cstdio>
#include <stdexcept>
using namespace fastllm;
class ViewLedger : public PrefixCachePageLedger {
public:
 using PrefixCachePageLedger::PrefixCachePageLedger;
 bool CommitL1Batch(const std::vector<L1Adoption> &a,const std::vector<PrefixCachePageIdentity> *expected=nullptr) {
  std::vector<L1AdoptionView> views;views.reserve(a.size());
  for(const auto &v:a)views.push_back({&v.identity,v.lease,v.manager,v.page,v.generation,v.h2dEvent,v.producerEvent});
  return CommitL1BatchView(views,expected);
 }
};

#define PrefixCachePageLedger ViewLedger
#define main RunOriginalLedgerControls
#include "../basic/test_prefix_cache_page_ledger.cpp"
#undef main
#undef PrefixCachePageLedger
static void TestViewLifetimeAndNull() {
 PrefixCachePageLedger ledger;int manager=0;
 auto first=Page("view",0,"","view-0"),second=Page("view",1,"view-0","view-1");
 std::weak_ptr<int> event;
 {
  std::vector<PrefixCachePageIdentity> ids={first,second};auto producer=std::make_shared<int>(42);event=producer;
  std::vector<PrefixCachePageLedger::L1AdoptionView> views={{&ids[0],{},&manager,0,2,producer,producer},{nullptr,{},&manager,1,2,producer,producer}};
  assert(!ledger.CommitL1BatchView(views,&ids));PrefixCachePageSnapshot snap;assert(!ledger.Snapshot(first,snap));
  views[1].identity=&ids[1];assert(ledger.CommitL1BatchView(views,&ids));
  ids[0].model="changed-source-after-commit";ids.clear();
 }
 assert(!event.expired());PrefixCachePageSnapshot snap;assert(ledger.Snapshot(first,snap));
 assert(snap.identity==first&&snap.producerEvent&&*static_cast<int*>(snap.producerEvent.get())==42);
 assert(ledger.Snapshot(second,snap)&&snap.identity==second);
 assert(!ledger.CommitL1BatchView({},nullptr));
 std::puts("PASS null view atomic rejection and owning identity/event lifetime after source destruction");
}

int main(){TestViewLifetimeAndNull();return RunOriginalLedgerControls();}

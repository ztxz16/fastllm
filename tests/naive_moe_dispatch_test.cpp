#undef NDEBUG
#include "models/naive_moe_dispatch.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>
using namespace fastllm;
int main() {
    using B = NaiveMoeBackend;
    // Available cubins, SM/SM-count/P2P probes are supplied by the native
    // admission functions. Tests exercise policy for all possible probe results.
    assert(NaiveMoeGeometrySupported(4096,256,8,48,true,1.0f));
    assert(!NaiveMoeGeometrySupported(4096,256,4,48,true,1.0f));
    assert(!NaiveMoeGeometrySupported(4096,128,8,48,true,1.0f));
    assert(!NaiveMoeGeometrySupported(2048,256,8,48,true,1.0f));
    assert(!NaiveMoeGeometrySupported(4096,256,8,49,true,1.0f));
    assert(!NaiveMoeGeometrySupported(4096,256,8,0,true,1.0f));
    assert(!NaiveMoeGeometrySupported(4096,256,8,48,false,1.0f));
    assert(!NaiveMoeGeometrySupported(4096,256,8,48,true,0.5f));
    int cases = 0;
    for (int ep : {1,4,8}) for (int bits=0; bits<128; ++bits) {
        bool shared=bits&1, old=bits&2, mega=bits&4, flex=bits&8;
        bool shape=bits&16, format=bits&32, enabled=bits&64;
        B b=SelectNaiveMoeBackend(ep,shared,old,mega,flex,shape,format,enabled);
        if(!shape||!format||!enabled) assert(b==B::LegacyTP);
        if(ep>1&&!mega) assert(b==B::LegacyTP);
        if(ep==1&&!flex) assert(b==B::LegacyTP);
        if(b==B::FlexEP) assert(ep==8&&shared&&mega&&flex&&shape&&format&&enabled);
        if(b==B::FlexTP) assert(ep==1&&shared&&flex&&shape&&format&&enabled);
        if(b==B::LegacyFlexTP) assert(ep==1&&!shared&&old&&flex&&shape&&format&&enabled);
        if(b==B::MegaEP) assert(ep>1&&mega&&shape&&format&&enabled);
        ++cases;
    }
    assert(SelectNaiveMoeBackend(8,true,false,true,false,true,true,true)==B::MegaEP); // Hopper/SM100
    assert(SelectNaiveMoeBackend(8,true,false,false,false,true,true,true)==B::LegacyTP); // SM89/120/missing cubin
    assert(SelectNaiveMoeBackend(1,true,true,true,true,true,true,true)==B::FlexTP); // no two Flex owners
    assert(SelectNaiveMoeBackend(4,true,false,true,true,true,true,true)==B::MegaEP);
    assert(SelectNaiveMoeBackend(8,true,false,true,true,true,true,true)==B::FlexEP);
    assert(SelectNaiveMoeBackend(1,false,true,false,true,true,true,true)==B::LegacyFlexTP);
    std::printf("POLICY_PASS %d combinations + 6 selection/fallback + 8 geometry cases\n",cases);
}

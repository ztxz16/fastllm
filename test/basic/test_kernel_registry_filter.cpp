#include "utils/kernel_registry.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
using namespace fastllm::kernels;
int main(){Family a{901,"test.a","1",{{"test.a.ref",{}},{"test.a.opt",{}}}},b{902,"test.b","1",{{"test.b.ref",{}},{"test.b.opt",{}}}};
Register(&a);Register(&b);std::string e;assert(Configure(R"({"mode":"quick","families":["test.a"]})",e));
assert(CandidateAllowed("test.a.opt"));assert(!CandidateAllowed("test.b.opt"));assert(!NeedsPreparation(Signature{{902}}));assert(NeedsPreparation(Signature{{901}}));
Call c{};c.family=&b;c.fallback=1;assert(Select(c)==1);
assert(!Configure(R"({"families":["unknown"]})",e));assert(!Configure(R"({"families":["test.a"],"force":{"test.b":"test.b.opt"}})",e));
assert(Configure(R"({"mode":"quick","families":[]})",e));assert(!CandidateAllowed("test.a.opt"));
assert(Configure(R"({"mode":"quick"})",e));assert(CandidateAllowed("test.a.opt")&&CandidateAllowed("test.b.opt"));puts("PASS family filter preserves excluded defaults and validates policy");}

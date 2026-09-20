#ifdef NDEBUG
#undef NDEBUG
#endif
#include "utils/disk_prefix_cache.h"
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>

using Cache = fastllm::DiskPrefixCache;
using Json = Cache::Json;
namespace fs = std::filesystem;

static Json save(Cache &cache, const std::string &key, const std::string &data,
                 int length, const Json &shared = Json()) {
    auto write = cache.BeginWrite(data.size() + (1 << 20));
    auto chunks = write.StoreBytes(data.size(), [&](size_t offset,char *out,size_t n) {
        memcpy(out,data.data()+offset,n);
    });
    Json state = Json::object{{"kind","prefix"},{"length",length},{"chunks",chunks},
        {"shared",shared},{"kv_dtype","fp8_e4m3"}};
    write.Commit(key,state);
    return state;
}
static std::string load(Cache &cache, const std::string &key, size_t bytes) {
    auto read = cache.BeginRead(); auto state = read.Load(key);
    std::string data(bytes,'\0');
    read.ReadBytes(state["chunks"],bytes,[&](size_t offset,const char *p,size_t n) {
        memcpy(data.data()+offset,p,n);
    });
    return data;
}
template<class F> static bool fails(F fn) {
    try { fn(); return false; } catch (const std::exception &) { return true; }
}
int main() {
    char pattern[] = "/tmp/fastllm-ssd-test-XXXXXX";
    char *name = mkdtemp(pattern); assert(name);
    fs::path root(name);
    try {
        const auto identity = Cache::Digest("test-model");
        const auto family = Cache::Digest("test-family");
        const auto keyA = Cache::Digest("branch-A"), keyB = Cache::Digest("branch-B");
        std::string data((2 << 20) + 7,'a');
        for (size_t i = 0; i < data.size(); ++i) data[i] = char((i*17+3)%251);
        Json parent;
        {
            Cache cache(root.string(),identity,uint64_t(128)<<20,family);
            parent = save(cache,keyA,data,8192);
            assert(load(cache,keyA,data.size()) == data);
            auto before = cache.ListCheckpoints(identity,"",20000);
            assert(before.size() == 1);
            save(cache,keyA,data,8192);
            auto after = cache.ListCheckpoints(identity,"",20000);
            assert(after.size() == 1 && before[0].createdNs == after[0].createdNs);
            save(cache,keyB,"branch B",16384,parent["chunks"]);
            assert(load(cache,keyB,8) == "branch B");
            auto writer = cache.BeginWrite(2<<20);
            auto chunks = writer.StoreBytes(4,[](size_t,char *p,size_t){ memcpy(p,"next",4); });
            writer.Commit(Cache::Digest("checkpoint1"),Json::object{{"length",2048},{"chunks",chunks}});
            writer.Commit(Cache::Digest("checkpoint2"),Json::object{{"length",4096},{"chunks",chunks}});
        }
        // A new process knows neither the old in-memory index nor its buffers.
        pid_t child = fork(); assert(child >= 0);
        if (child == 0) {
            try { Cache cache(root.string(),identity,uint64_t(128)<<20,family);
                _exit(load(cache,keyA,data.size()) == data ? 0 : 2); }
            catch (...) { _exit(3); }
        }
        int status = 0; assert(waitpid(child,&status,0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        fs::remove(root/"v2/index.sqlite3"); fs::remove(root/"v2/index.sqlite3-wal"); fs::remove(root/"v2/index.sqlite3-shm");
        {
            Cache cache(root.string(),identity,uint64_t(128)<<20,family);
            assert(cache.ListCheckpoints(identity,"",20000).size() == 4);
            assert(load(cache,keyA,data.size()) == data);
            const auto hash = parent["chunks"][0]["sha256"].string_value();
            fs::path object;
            for (auto &entry : fs::directory_iterator(root/"v2/objects"))
                if (entry.path().filename().string().find(hash) != std::string::npos) object = entry.path();
            assert(!object.empty());
            { std::fstream f(object,std::ios::binary|std::ios::in|std::ios::out);
              f.seekp(-1,std::ios::end); char value = 0x7f; f.write(&value,1); }
            assert(fails([&] { load(cache,keyA,data.size()); }));
            save(cache,keyA,data,8192); // Existing corrupt payload is repaired.
            assert(load(cache,keyA,data.size()) == data);
            assert(fails([&] { save(cache,keyA,"different payload",8192); }));
        }
        // Kill a writer after one durable object and before its next object.
        // No shutdown handler can rescue its reservation or partial checkpoint.
        const auto abandoned = Cache::Digest("abandoned");
        int ready[2]; assert(pipe(ready) == 0);
        child = fork(); assert(child >= 0);
        if (child == 0) {
            close(ready[0]);
            try { Cache cache(root.string(),identity,uint64_t(128)<<20,family);
                auto writer = cache.BeginWrite(20<<20);
                writer.StoreBytes(Cache::ChunkBytes+16,[&](size_t offset,char *p,size_t n){
                    if (offset > 0) { assert(write(ready[1],"r",1) == 1); for (;;) pause(); }
                    memset(p,42,n);
                });
                _exit(5); } catch (...) { _exit(4); }
        }
        close(ready[1]); char message = 0;
        assert(read(ready[0],&message,1) == 1 && message == 'r'); close(ready[0]);
        assert(kill(child,SIGKILL) == 0);
        assert(waitpid(child,&status,0) == child && WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
        {
            Cache cache(root.string(),identity,uint64_t(128)<<20,family);
            assert(fails([&] { auto read=cache.BeginRead(); read.Load(abandoned); }));
            assert(load(cache,keyA,data.size()) == data);
            // A descendant is self-contained even if its ancestor commit goes away.
            fs::remove(root/"v2/commits"/identity/(keyA+".commit"));
            cache.RebuildIndex();
            auto reader=cache.BeginRead(); auto descendant=reader.Load(keyB);
            std::string trunk(data.size(),'\0');
            reader.ReadBytes(descendant["shared"],trunk.size(),[&](size_t offset,const char *p,size_t n){
                memcpy(trunk.data()+offset,p,n);
            });
            assert(trunk == data && load(cache,keyB,8) == "branch B");
        }
        // Both processes construct their backends before either starts writing,
        // so object publication and duplicate commit handling truly overlap.
        int initialized[2], start[2];
        assert(pipe(initialized) == 0 && pipe(start) == 0);
        pid_t children[2];
        for (int i=0;i<2;++i) {
            children[i]=fork(); assert(children[i]>=0);
            if (children[i]==0) {
                close(initialized[0]); close(start[1]);
                try {
                    Cache cache((root/"concurrent").string(),identity,uint64_t(128)<<20,family);
                    assert(write(initialized[1],"r",1)==1);
                    char go=0; assert(read(start[0],&go,1)==1);
                    save(cache,keyA,data,8192);
                    _exit(0);
                } catch (...) { _exit(6); }
            }
        }
        close(initialized[1]); close(start[0]);
        for (int i=0;i<2;++i) assert(read(initialized[0],&message,1)==1);
        assert(write(start[1],"gg",2)==2);
        close(initialized[0]); close(start[1]);
        for (auto pid:children) {
            assert(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0);
        }
        {
            Cache cache((root/"concurrent").string(),identity,uint64_t(128)<<20,family);
            assert(cache.ListCheckpoints(identity,"",20000).size()==1);
            assert(load(cache,keyA,data.size())==data);
        }
        // Small capacity forces FIFO. Reading the first entry must not renew it.
        {
            Cache cache((root/"fifo").string(),identity,uint64_t(6)<<20,family);
            std::string block(2<<20,'x');
            save(cache,keyA,block,2048);
            assert(load(cache,keyA,block.size()) == block);
            block[0]='y'; save(cache,keyB,block,4096);
            block[0]='z'; save(cache,Cache::Digest("C"),block,6144);
            auto remaining=cache.ListCheckpoints(identity,"",20000);
            bool hasA=false,hasC=false;
            for (auto &entry:remaining) { hasA |= entry.key==keyA; hasC |= entry.key==Cache::Digest("C"); }
            assert(!hasA && hasC);
        }
        fs::remove_all(root);
        std::cout << "SSD storage: cross-process, branches, shared refs, concurrent commits, SIGKILL, corruption, index recovery, FIFO: PASS\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "SSD test failed; preserved " << root << ": " << e.what() << '\n';
        return 1;
    }
}

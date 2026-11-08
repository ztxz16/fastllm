#!/usr/bin/env python3
"""Prepare CPU-only A/B probes from the actual import/publication source.
Invoke through the project action logger. This script only generates a build
tree; it does not execute tests, touch a cache, or signal any service.
"""
from pathlib import Path
import argparse, subprocess, json, hashlib
ap=argparse.ArgumentParser()
ap.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[2])
ap.add_argument("--baseline",default="88afa59422aca5e203e253b8b9f5dfe600da829b")
ap.add_argument("--build",type=Path,required=True)
args=ap.parse_args()
repo=args.source.resolve()
R=repo.parent
B=args.build.resolve()
if not B.is_relative_to((R/"build").resolve()):
    raise SystemExit("REFUSED: generated build tree must stay below workspace/build")
if B.exists():
    raise SystemExit("REFUSED: choose a fresh probe build directory")
(B/"base").mkdir(parents=True)
(B/"tmp").mkdir()
for name in ["src/models/qwen3_5_persistent.inc","src/models/qwen3_5_spill.inc"]:
    text=subprocess.check_output(["git","-C",str(repo),"show",args.baseline+":"+name],text=True)
    p=B/"base"/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(text)
(B/"assembly_t0.cpp").write_bytes((repo/"test/api/ssd_import_assembly_local_t0.cpp").read_bytes())
(B/"publication_t0.cpp").write_bytes((repo/"test/api/ssd_direct_publication_local_t0.cpp").read_bytes())
(B/"head_mapping_t0.cpp").write_bytes((repo/"test/api/ssd_import_head_mapping_t0.cpp").read_bytes())
for variant,root in [("base",B/"base"),("candidate",repo)]:
 s=(root/"src/models/qwen3_5_persistent.inc").read_text()
 if variant=="base" and "uint64_t h2dAssembleBytes" not in s:
  s=s.replace("uint64_t h2dSpanCopies = 0, h2dHeadCopies = 0;","uint64_t h2dSpanCopies = 0, h2dHeadCopies = 0;\n        uint64_t h2dAssembleBytes=0,h2dCompareBytes=0,h2dSegmentVisits=0;",1)
  s=s.replace("for (const auto &item : loaded) for (const auto &entry : item.segments) {","for (const auto &item : loaded) for (const auto &entry : item.segments) {\n                    ++h2dSegmentVisits;",1)
  s=s.replace("memcpy(outputPtr, source, (size_t)tokens * entryHeads * rowBytes);","memcpy(outputPtr, source, (size_t)tokens * entryHeads * rowBytes);\n                        h2dAssembleBytes += (size_t)tokens*entryHeads*rowBytes;",1)
  marker="                                    for (int head = 0; head < spanHeads; ++head)\n                                        filled[targetHeadStart+head] = 1;"
  s=s.replace(marker,"                                    h2dAssembleBytes += (uint64_t)tokens * spanBytes;\n"+marker,1)
  s=s.replace("                                            Check(memcmp(destination,sourceRow,rowBytes) == 0,","                                            h2dCompareBytes += rowBytes;\n                                            Check(memcmp(destination,sourceRow,rowBytes) == 0,",1)
 start=s.index("        struct LoadedBundle",s.index("bool Qwen35PersistentCache::Import("))
 stop=s.index("                const auto copyStart = ",start)
 body=s[start:stop]
 function=("RunBase" if variant=="base" else "RunCandidate")
 text=f"""Result {function}(TestCase &tc,bool full) {{
    Result result;
    FakeModel modelStorage; modelStorage.num_key_value_heads=tc.heads;
    auto *model=&modelStorage;
    const int length=tc.length;
    const bool convert=false;
    const auto &pools=tc.pools;
    const auto &bundlesByBegin=tc.bundles;
    FakeReader reader(tc.bytes);
    std::function<bool()> keepGoing=[&] {{return !tc.cancel;}};
    const auto started=std::chrono::steady_clock::now();
    auto importMark=[&] {{return std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-started).count();}};
    [[maybe_unused]] double tKv=0;
    try {{
"""+body+"""
                Observe(outputPtr,outputBytes,result,full);
                coverage[targetKey]+=tokens;
            }
        }
        for(const auto &p:pools) Check(coverage[p.first]==length,"ssd_incomplete_kv");
        result.assemble_ms=h2dAssembleMs;result.readmany_ms=h2dReadManyMs;
        result.memcpy_bytes=h2dAssembleBytes;result.compare_bytes=h2dCompareBytes;
        result.segment_visits=h2dSegmentVisits;result.span_copies=h2dSpanCopies;
    } catch(const std::exception&e) { result.error=e.what(); }
    result.wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    return result;
}
"""
 (B/("assembly_"+variant+".inc")).write_text(text)
 (B/variant).mkdir(exist_ok=True)
 s=(root/"src/models/qwen3_5_spill.inc").read_text()
 start=s.index("bool Qwen3_5Model::PublishDirectPagedPrefix(")
 stop=s.index("\nvoid Qwen3_5Model::PreparePagedSpillPrefix(",start)
 func=s[start:stop].replace("bool Qwen3_5Model::PublishDirectPagedPrefix(","bool ProbeModel::PublishDirectPagedPrefix(",1)
 func=func.replace("std::lock_guard<std::mutex> lock(plan.manager->pageIndexLocker);","std::lock_guard<std::mutex> lock(plan.manager->pageIndexLocker);\n            ++plan.manager->pageLocks;")
 func=func.replace("std::lock_guard<std::mutex> lock(manager->pageIndexLocker);","std::lock_guard<std::mutex> lock(manager->pageIndexLocker);\n        ++manager->pageLocks;")
 (B/variant/"publication_path.inc").write_text(func)
cmake=f"""cmake_minimum_required(VERSION 3.20)
project(b21_local_opt_t0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
add_executable(assembly_t0 assembly_t0.cpp {repo}/third_party/json11/json11.cpp)
target_include_directories(assembly_t0 PRIVATE {B} {repo}/third_party/json11)
target_compile_options(assembly_t0 PRIVATE -O3 -march=native -Wall -Wextra)
add_executable(head_mapping_t0 head_mapping_t0.cpp {repo}/third_party/json11/json11.cpp)
target_include_directories(head_mapping_t0 PRIVATE {B} {repo}/third_party/json11)
target_compile_options(head_mapping_t0 PRIVATE -O3 -march=native -Wall -Wextra)
"""
for variant in ["base","candidate"]:
 cmake+=f"""add_executable(publication_{variant} publication_t0.cpp)
target_include_directories(publication_{variant} PRIVATE {B}/{variant} {repo}/include)
target_compile_options(publication_{variant} PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(publication_{variant} PRIVATE Threads::Threads)
"""
(B/"CMakeLists.txt").write_text(cmake)
print(json.dumps({"build":str(B),"cpp_sources":["assembly_t0.cpp","publication_t0.cpp"],"source_extractions":[{"path":str(f),"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(B.glob("assembly_*.inc"))]+[{"path":str(f),"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(B.glob("*/publication_path.inc"))]}))

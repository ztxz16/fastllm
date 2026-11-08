"""Exercise the actual CUDA snapshot finder on CPU-only metadata stubs.

The code under test is extracted unchanged from qwen3_5.cpp. No model, GPU,
SSD store or copied implementation of the matching policy is used.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
BUILD = Path(os.environ.get("FASTLLM_SNAPSHOT_ACCEPTANCE_BUILD",
    str(WORKSPACE / "build/request-observability-20261016/host-probes/source-finder")))
LOGS = WORKSPACE / "logs"


def record(**fields):
    with (LOGS / "actions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(), **fields}, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def function(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    state = "code"
    end = opening + 1
    while depth:
        char, pair = source[end], source[end:end + 2]
        if state == "line":
            if char == "\n":
                state = "code"
        elif state == "comment":
            if pair == "*/":
                state, end = "code", end + 1
        elif state in ('"', "'"):
            if char == "\\":
                end += 1
            elif char == state:
                state = "code"
        elif pair == "//":
            state, end = "line", end + 1
        elif pair == "/*":
            state, end = "comment", end + 1
        elif char in ('"', "'"):
            state = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        end += 1
    return source[start:end]


PREFIX = r'''
#include "utils/multimodal_prefix_cache.h"
#include "utils/snapshot_prefix_identity.h"
#include "json11.hpp"
#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <utility>
#include <stdexcept>
namespace fastllm {
using Json = json11::Json;
static void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
int GetPageLen() { return 128; }
enum class DataDevice { CPU, CUDA };
struct Data {
    std::vector<int> dims;
    DataDevice dataDevice = DataDevice::CPU;
    const void *cpuData = reinterpret_cast<void *>(1);
};
struct ResponseContext {
    std::vector<int> allTokens, currentTokens;
    std::vector<std::string> prefixCachePageKeys, prefixCacheBasePageKeys;
    std::string prefixCacheProvenance;
    std::vector<int> prefixCachePositionIds;
    int inputTokens = 0;
    struct { bool output_logits = false; } generationConfig;
    std::vector<MultimodalPrefixSpan> multimodalPrefixSpans;
    std::map<std::string, int> multimodalInput;
'''

SNAPSHOT = r'''
};
struct Qwen3_5Model {};
struct Qwen35LinearPrefixSnapshot {
    int cachedLen = 0;
    long long timestamp = 1;
    std::vector<int> tokens;
    std::vector<std::string> mediaPageKeys;
    SnapshotPrefixIdentity matchIdentity;
    bool mtpValid = false, dflashValid = false;
    int mtpTokens = 0, dflashTokens = 0;
    Data mtpKey, mtpValue;
    std::vector<std::pair<Data, Data>> dflashKeyValues;
};
auto &Qwen35LinearPrefixSnapshots() {
    static std::map<const Qwen3_5Model *,
        std::vector<std::shared_ptr<Qwen35LinearPrefixSnapshot>>> snapshots;
    return snapshots;
}
'''

CASES = r'''
}
int main() {
    using namespace fastllm;
    Qwen3_5Model model;
    ResponseContext text;
    for (int i=0; i<384; ++i) text.allTokens.push_back(i+1000);
    text.currentTokens = text.allTokens;
    text.prefixCachePageKeys = {"text-hash-0", "text-hash-1", "text-hash-2"};
    text.prefixCacheBasePageKeys = text.prefixCachePageKeys;
    auto snapshot = std::make_shared<Qwen35LinearPrefixSnapshot>();
    snapshot->cachedLen = 256;
    snapshot->tokens.assign(text.allTokens.begin(), text.allTokens.begin()+256);
    snapshot->mediaPageKeys.assign(text.prefixCachePageKeys.begin(), text.prefixCachePageKeys.begin()+2);
    assert(Qwen35SetSnapshotMatchIdentity(&text, *snapshot));
    Qwen35LinearPrefixSnapshots()[&model] = {snapshot};
    auto find = [&](ResponseContext &ctx, bool mtp=false, bool dflash=false) {
        return Qwen35FindLinearPrefixSnapshotLocked(&model, ctx.currentTokens,
            &ctx, 256, -1, mtp, dflash, 1, 2, 4);
    };
    assert(find(text) == snapshot.get());
    std::map<std::string,std::string> diagnostic;
    assert(Qwen35FindLinearPrefixSnapshotLocked(&model,text.currentTokens,&text,128,-1,
        false,false,1,2,4,&diagnostic)==nullptr);
    assert(diagnostic.at("snapshot_reject_length")=="1");
    diagnostic.clear();
    auto changed=text;changed.currentTokens[7]++;
    assert(Qwen35FindLinearPrefixSnapshotLocked(&model,changed.currentTokens,&changed,256,-1,
        false,false,1,2,4,&diagnostic)==nullptr);
    assert(diagnostic.at("snapshot_reject_token_prefix")=="1");
    diagnostic.clear();
    assert(Qwen35FindLinearPrefixSnapshotLocked(&model,text.currentTokens,&text,256,-1,
        false,true,1,2,4,&diagnostic)==nullptr);
    assert(diagnostic.at("snapshot_reject_dflash_state")=="1");
    ResponseContext image = text;
    image.multimodalInput["image_frames"] = 1;
    MultimodalPrefixSpan span; span.begin=256; span.end=320; span.digest.fill(7);
    image.multimodalPrefixSpans = {span};
    std::vector<int> positions;
    for (int axis=0; axis<3; ++axis)
        for (int i=0; i<384; ++i) positions.push_back(i);
    image.prefixCachePageKeys = BuildMultimodalPrefixPageKeys(image.allTokens, positions,
        image.multimodalPrefixSpans, 128);
    image.prefixCacheBasePageKeys = image.prefixCachePageKeys;
    assert(find(image) == snapshot.get()); // actual finder, not just predicate
    // Native tensors can already be consumed while persistent media identity
    // remains in spans. The production eligibility and finder must retain it.
    image.multimodalInput.clear();
    assert(image.HasMultimodalPrefixIdentity());
    assert(image.CanUsePagedPrefixCache());
    assert(image.PrefixCachePageKeys() == &image.prefixCachePageKeys);
    assert(find(image) == snapshot.get());
    image.prefixCachePositionIds = positions;
    image.generationConfig.output_logits = true;
    const Json saved = SavePrefetchContext(&image);
    assert(saved["media"].bool_value());
    assert(saved["spans"].array_items().size() == 1);
    std::string parseError;
    const Json persisted = Json::parse(saved.dump(), parseError);
    assert(parseError.empty());
    ResponseContext replay;
    LoadPrefetchContext(persisted, replay);
    assert(replay.multimodalPrefixSpans.size() == 1);
    assert(replay.multimodalPrefixSpans[0].digest == span.digest);
    assert(replay.prefixCachePageKeys == image.prefixCachePageKeys);
    assert(replay.prefixCacheBasePageKeys == image.prefixCacheBasePageKeys);
    assert(replay.prefixCachePositionIds == positions);
    assert(replay.generationConfig.output_logits);
    replay.multimodalInput.clear(); // replay must not rely on synthetic tensor data
    assert(replay.HasMultimodalPrefixIdentity());
    assert(find(replay) == snapshot.get());
    // A prefix trimmed before the trailing image retains media=true but does
    // not invent an in-prefix image span or break the two text pages.
    ResponseContext trimmed;
    LoadPrefetchContext(SavePrefetchContext(&image, 256), trimmed);
    assert(trimmed.multimodalPrefixSpans.empty());
    assert(trimmed.prefixCachePageKeys.size() == 2);
    assert(find(trimmed) == snapshot.get());
    image.currentTokens[0]++;
    assert(find(image) == nullptr);
    image.currentTokens = image.allTokens;
    image.prefixCacheBasePageKeys[0] = "changed-position";
    assert(find(image) == nullptr);
    image.prefixCacheBasePageKeys = image.prefixCachePageKeys;
    image.prefixCacheProvenance = "other-conversion";
    assert(find(image) == nullptr);
    image.prefixCacheProvenance.clear();
    assert(find(image, true) == nullptr);
    snapshot->mtpValid=true; snapshot->mtpTokens=256;
    snapshot->mtpKey.dims={1,256,4}; snapshot->mtpValue.dims={1,256,4};
    assert(find(image, true) == snapshot.get());
    snapshot->mtpValue.dims[1]=128;
    assert(find(image, true) == nullptr);
    assert(find(image, false, true) == nullptr);
    snapshot->dflashValid=true; snapshot->dflashTokens=256;
    Data key; key.dims={2,256,4};
    snapshot->dflashKeyValues={{key,key}};
    assert(find(image, false, true) == snapshot.get());
    snapshot->dflashKeyValues[0].second.cpuData=nullptr;
    assert(find(image, false, true) == nullptr);

    // An image inside the candidate remains a distinct snapshot identity.
    span.begin=64; span.end=80; image.multimodalPrefixSpans={span};
    image.prefixCachePageKeys=BuildMultimodalPrefixPageKeys(image.allTokens, positions,
        image.multimodalPrefixSpans,128);
    image.prefixCacheBasePageKeys=image.prefixCachePageKeys;
    assert(find(image) == nullptr);
    auto media=std::make_shared<Qwen35LinearPrefixSnapshot>();
    media->cachedLen=256; media->tokens=snapshot->tokens;
    media->mediaPageKeys.assign(image.prefixCachePageKeys.begin(),image.prefixCachePageKeys.begin()+2);
    assert(Qwen35SetSnapshotMatchIdentity(&image,*media));
    Qwen35LinearPrefixSnapshots()[&model]={media};
    assert(find(image) == media.get());
    image.multimodalInput.clear();
    ResponseContext mediaReplay;
    LoadPrefetchContext(SavePrefetchContext(&image), mediaReplay);
    mediaReplay.multimodalInput.clear();
    assert(mediaReplay.multimodalPrefixSpans.size() == 1);
    assert(find(mediaReplay) == media.get());
    assert(find(text) == nullptr); // reverse-direction image-to-text safety
    image.multimodalPrefixSpans[0].digest.fill(8);
    image.prefixCacheBasePageKeys=BuildMultimodalPrefixPageKeys(image.allTokens,positions,
        image.multimodalPrefixSpans,128);
    assert(find(image) == nullptr);
    return 0;
}
'''


def main():
    paths = [ROOT / "src/models/qwen3_5.cpp", ROOT / "include/models/basellm.h",
             ROOT / "src/models/qwen3_5_persistent.inc"]
    record(action="snapshot_source_t0_read", phase="before",
           paths=[str(path) for path in paths], expected="Extract actual production matching functions")
    source, context, persistent = [path.read_text() for path in paths]
    record(action="snapshot_source_t0_read", phase="after", exit_code=0,
           sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    methods = "\n".join(function(context, marker) for marker in (
        "        bool HasMultimodalPrefixIdentity() const",
        "        bool CanUsePagedPrefixCache() const",
        "        const std::vector<std::string> *PrefixCachePageKeys() const"))
    helpers = source[source.index("        static const std::vector<std::string> *Qwen35SnapshotBaseKeys("):
                     source.index("        struct Qwen35LoadProfile {")]
    roundtrip = "\n".join(function(persistent, marker) for marker in (
        "    static std::vector<int> Ints(", "    static Json IntArray(",
        "    static std::string EncodeKey(", "    static std::string DecodeKey(",
        "    static Json SavePrefetchContext(", "    static void LoadPrefetchContext("))
    code = PREFIX + methods + SNAPSHOT + helpers + "\n" + roundtrip + "\n" + function(
        source, "        static bool Qwen35DFlashSnapshotMatches(") + "\n" + function(
        source, "        static const Qwen35LinearPrefixSnapshot *Qwen35FindLinearPrefixSnapshotLocked(") + CASES
    tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    build = BUILD / tag
    record(action="snapshot_source_t0_stage", phase="before", build=str(build),
           expected="Write host-only extracted code and Makefile inside fixed build")
    build.mkdir(parents=True, exist_ok=False)
    (build / "probe.cpp").write_text(code)
    (build / "Makefile").write_text(
        "CXX := /usr/bin/c++\nFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -I" +
        str(ROOT / "include") + " -I" + str(ROOT / "third_party/json11") +
        "\nall: probe\nprobe: probe.o json11.o\n\t$(CXX) $^ -o $@\n"
        "probe.o: probe.cpp\n\t$(CXX) $(FLAGS) -c $< -o $@\njson11.o: " +
        str(ROOT / "third_party/json11/json11.cpp") + "\n\t$(CXX) $(filter-out -Werror,$(FLAGS)) -c $< -o $@\n")
    record(action="snapshot_source_t0_stage", phase="after", exit_code=0, build=str(build),
           generated_cpp_sha256=hashlib.sha256(code.encode()).hexdigest())
    for label, command in (("compile", ["make", "-j36"]), ("run", ["./probe"])):
        target = LOGS / ("t0-snapshot-source-" + label + "-" + tag + ".log")
        record(action="snapshot_source_t0_" + label, phase="before", cwd=str(build),
               command=command, log=str(target), expected="Actual finder metadata assertions pass without GPU")
        with target.open("xb") as output:
            result = subprocess.run(command, cwd=build, stdout=output, stderr=subprocess.STDOUT)
        record(action="snapshot_source_t0_" + label, phase="after", command=command,
               exit_code=result.returncode, log=str(target))
        if result.returncode:
            raise SystemExit(result.returncode)
    print("Actual snapshot finder T0 passed")


if __name__ == "__main__":
    main()

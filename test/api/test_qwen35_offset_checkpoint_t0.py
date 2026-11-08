"""Exercise unchanged production checkpoint admission code on CPU metadata."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest
import uuid

ROOT = Path(os.environ.get("FASTLLM_TEST_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "fastllm-test")))
(ROOT / 'logs').mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[2]


def record(event, **fields):
    with (ROOT / 'logs/actions.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            'task': 'offset-checkpoint-t0', 'event': event, **fields}) + '\n')
        f.flush()
        os.fsync(f.fileno())


class OffsetCheckpointT0(unittest.TestCase):
    def test_production_admission_accepts_restored_chunks_preserves_safety(self):
        source = (REPO / 'src/models/qwen3_5.cpp').read_text()
        begin = source.index('        if (!context->multimodalInput.empty()) {',
                             source.index('bool Qwen3_5Model::TryRecordPagedPrefixCacheExtra('))
        end = source.index('        // Also cover skip-warmup', begin)
        guard = source[begin:end]
        tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
        build = ROOT / 'build/request-observability-20261016/host-probes' / ('offset-checkpoint-' + tag)
        logs = [ROOT / 'logs' / ('offset-checkpoint-' + tag + suffix) for suffix in ('.compile.log', '.run.log')]
        record('prepare', build_dir=str(build), logs=list(map(str, logs)), source=str(REPO / 'src/models/qwen3_5.cpp'),
               source_sha256=hashlib.sha256(source.encode()).hexdigest(), gpu_mapping='none',
               expected='production guard accepts completed restored-offset chunks; rejects partial/page/input violations')
        body = r"""
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>
struct Context {
    std::vector<int> multimodalInput{1};
    std::vector<int> allTokens = std::vector<int>(112000);
    int inputTokens=112000, cacheLen=0;
};
int configuredChunk=2048;
bool requireDraft=true;
constexpr int QWEN35_DFLASH_LONG_PREFILL_CHUNK_SIZE=2048;
int GetChunkedPrefillSize(){return configuredChunk;}
bool RequiresDFlashPrefixSnapshot(Context*){return requireDraft;}
bool admitted(Context *context,int currentLen){const int pageLen=128;
""" + guard + r"""
return true;}
int main(){
    Context context;
    int accepted=0;
    for(bool draft : {false,true}) {
        requireDraft=draft;
        for(int configured : {256,2048,4096}) {
            configuredChunk=configured;
            const int chunk=draft?std::min(configured,2048):configured;
            for(int restored : {0,128,1024,2048,3072,39936}) {
                context.cacheLen=restored;
                assert(!admitted(&context,restored));
                for(int step=1; step<=16; ++step) {
                    const int boundary=restored+step*chunk;
                    assert(admitted(&context,boundary));++accepted;
                    assert(!admitted(&context,boundary-128));
                    assert(!admitted(&context,boundary-1));
                }
                assert(!admitted(&context,context.inputTokens+chunk));
                if(restored>0) assert(!admitted(&context,restored-128));
            }
        }
    }
    configuredChunk=0;context.cacheLen=39936;assert(!admitted(&context,41984));
    configuredChunk=-1;assert(!admitted(&context,41984));
    configuredChunk=2048;context.cacheLen=39936;context.inputTokens=108333;
    int restoredAccepted=0;
    for(int n=39936+2048;n<=108333;n+=2048)restoredAccepted+=admitted(&context,n);
    assert(restoredAccepted==33);assert(!admitted(&context,108288));
    // Text requests retain their pre-existing page-only admission behavior.
    context.multimodalInput.clear();assert(admitted(&context,108288));
    assert(!admitted(&context,108289));
    std::printf("OFFSET_CHECKPOINT_T0_OK variants_accepted=%d restored_39936_accepted=%d\n",accepted,restoredAccepted);
}
"""
        build.mkdir(parents=True, exist_ok=False)
        (build / 'probe.cpp').write_text(body)
        (build / 'Makefile').write_text('all: probe\nprobe: probe.cpp\n\t/usr/bin/c++ -std=c++17 -O2 -Wall -Wextra -Werror $< -o $@\n')
        for cmd, log in zip((['make', '-j36'], [str(build / 'probe')]), logs):
            record('command_start', command=cmd, cwd=str(build), log_path=str(log), gpu_mapping='none', expected='exit0')
            with log.open('xb') as out:
                result = subprocess.run(cmd, cwd=build, stdout=out, stderr=subprocess.STDOUT)
            record('command_end', command=cmd, exit_code=result.returncode, log_path=str(log), resources='no GPU/service used')
            output=log.read_text();print(output,end='',flush=True)
            self.assertEqual(result.returncode,0,output)
        self.assertIn('restored_39936_accepted=33',logs[1].read_text())


if __name__ == '__main__':
    unittest.main()

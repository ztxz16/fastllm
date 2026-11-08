#!/usr/bin/env python3
"""Compile actual metric helpers with a tiny context; no GPU/model required."""
import datetime,hashlib,json,os,pathlib,subprocess,unittest,uuid
ROOT = pathlib.Path(os.environ.get("FASTLLM_TEST_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "fastllm-test")))
(ROOT / 'logs').mkdir(parents=True, exist_ok=True)
REPO=pathlib.Path(__file__).resolve().parents[2]

def record(event,**data):
    with (ROOT/'logs/actions.jsonl').open('a') as f:
        f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'native-metrics-t0-20261017','event':event,**data})+'\n');f.flush();os.fsync(f.fileno())

class NativeMetricsT0(unittest.TestCase):
    def test_actual_helpers_and_post_filter_production_hooks(self):
        tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
        build=pathlib.Path(os.environ.get('FASTLLM_METRICS_ACCEPTANCE_BUILD',str(ROOT/'build/request-observability-20261016/host-probes')))/('native-metrics-'+tag)
        compile_log=ROOT/'logs'/('native-metrics-t0-'+tag+'.compile.log');run_log=ROOT/'logs'/('native-metrics-t0-'+tag+'.run.log')
        record('source_probe_prepare',cwd=str(REPO),build_dir=str(build),logs=[str(compile_log),str(run_log)],gpu_mapping='none',environment={'FASTLLM_OBSERVABILITY_MODE':'debug'},expected='compile production native metrics on CPU; verify exact counts/windows and no placeholder rates')
        base=(REPO/'src/models/basellm.cpp').read_text();qwen=(REPO/'src/models/qwen3_5.cpp').read_text()
        functions=base[base.index('    void RecordRequestNativeCommit('):base.index('    void RecordRequestPrefillProgress(')]
        qstart=qwen.index('                    const bool nativePromptStep = selectedIsPrompt || ctx->curTokens == 0;')
        qblock=qwen[qstart:qwen.index('                model->dictCV.notify_all();',qstart)]
        for earlier in ('if (!partialPrefill)','curRet == model->eos_token_id','generationConfig.stop_token_ids.find','resultTokenQueue.push(curRet)','ctx->curTokens++'):
            self.assertLess(qblock.index(earlier),qblock.index('RecordRequestNativeCommit(ctx, nativePromptStep)'))
        self.assertLess(qblock.index('RecordRequestNativeCommit'),qblock.index('ctx->curTokens == ctx->generationConfig.output_token_limit'))
        self.assertIn('if (ctx->isEnding) {\n                            break;',qblock)
        self.assertNotIn('RecordRequestDecodeProgress(ctx, decodeTokens)',qwen)
        body=r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include "utils/native_generation_metrics.h"
#include "utils/prefix_cache_logging.h"
namespace fastllm {
struct ResponseContext {
    uint64_t requestId = 7;
    NativeGenerationMetrics nativeGeneration;
    int inputTokens = 1024, cacheLen = 256, progressDecodeTokens = 0;
    bool isEnding = false;
};
void RecordRequestDecodeProgress(ResponseContext *c, int n) { c->progressDecodeTokens += n; }
"""+functions+r"""
}
int main() {
    using namespace fastllm;
    NativeGenerationMetrics m;
    assert(!m.HasDecodeRate()); assert(m.DecodeTokensPerSecond() < 0);
    m.Prefill(768, 1000000000ULL, 2000000000ULL, 200000000ULL);
    assert(m.prefillTokens == 768 && m.prefillElapsedNs == 800000000ULL);
    assert(m.excludedVisionNs == 200000000ULL);
    m.Commit(true, 2000000000ULL);m.Ready(2000000000ULL);
    m.Commit(false);m.Commit(false);m.DecodeEnd(4000000000ULL);
    assert(m.generatedTokens == 3 && m.prefillGeneratedTokens == 1 && m.decodeTokens == 2);
    assert(m.HasDecodeRate() && std::fabs(m.DecodeTokensPerSecond() - 1.0) < 1e-12);
    m.Prefill(1,4000000000ULL,5000000000ULL,0);assert(!m.HasDecodeRate());
    NativeGenerationMetrics bad;bad.Prefill(1,20,10,0);assert(!bad.prefillClockValid);
    NativeGenerationMetrics zero;zero.Ready(10);zero.Commit(false);zero.DecodeEnd(10);assert(!zero.HasDecodeRate());
    ResponseContext c;RegisterFastllmPublicRequestId(7,"req-native-t0");
    const uint64_t now=FastllmObservationTimestampNs();c.nativeGeneration.createdNs=now-1000000000ULL;
    RecordRequestNativeCommit(&c,true);
    RecordRequestNativeStep(&c,true,false,768,now-1000000ULL,now,200000ULL);
    assert(c.nativeGeneration.prefillReported && c.nativeGeneration.firstTokenReported);
    assert(c.nativeGeneration.prefillElapsedNs==800000ULL);
    RecordRequestNativeCommit(&c,false);RecordRequestNativeCommit(&c,false);
    c.isEnding=true;
    const uint64_t end=FastllmObservationTimestampNs();
    RecordRequestNativeStep(&c,false,false,0,end-1,end,0);
    assert(c.nativeGeneration.finishedReported && c.nativeGeneration.HasDecodeRate());
    assert(c.nativeGeneration.generatedTokens==3 && c.nativeGeneration.decodeTokens==2);
    assert(c.progressDecodeTokens==3);
    std::puts("NATIVE_METRICS_T0_OK");return 0;
}
"""
        build.mkdir(parents=True,exist_ok=False);(build/'probe.cpp').write_text(body)
        (build/'Makefile').write_text('CXX := /usr/bin/c++\nCXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -pthread -I'+str(REPO/'include')+'\nall: native_metrics_probe\nnative_metrics_probe: probe.cpp\n\t$(CXX) $(CXXFLAGS) $< -o $@\n')
        results=[]
        for cmd,cwd,env,path in [(['make','-j36'],build,os.environ.copy(),compile_log),([str(build/'native_metrics_probe')],build,{**os.environ,'FASTLLM_OBSERVABILITY_MODE':'debug'},run_log)]:
            record('probe_command_start',command=cmd,cwd=str(cwd),build_dir=str(build),log_path=str(path),gpu_mapping='none',environment={'FASTLLM_OBSERVABILITY_MODE':env.get('FASTLLM_OBSERVABILITY_MODE')},expected='exit 0')
            with path.open('xb') as out:p=subprocess.run(cmd,cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT)
            record('probe_command_end',command=cmd,exit_code=p.returncode,log_path=str(path),resource_cleanup='no service/GPU resources started')
            text=path.read_text();print(text,end='',flush=True);self.assertEqual(p.returncode,0,text);results.append({'command':cmd,'exit_code':p.returncode,'log':str(path)})
        output=run_log.read_text();self.assertIn('generated_tokens=3 prefill_generated_tokens=1 native_decode_tokens=2',output)
        self.assertIn('missed_input_tokens=768',output);self.assertIn('event=native_decode',output);self.assertIn('NATIVE_METRICS_T0_OK',output)
        receipt={'passed':True,'build_dir':str(build),'results':results,'production_source_sha256':hashlib.sha256(base.encode()).hexdigest(),'scope':'actual production helper functions, fake-clock counter properties and post-filter hook locations; no GPU inference'}
        path=ROOT/'logs'/('native-metrics-t0-'+tag+'.json');record('write_probe_receipt',path=str(path),receipt=receipt);path.write_text(json.dumps(receipt,indent=2)+'\n');print('T0_RECEIPT='+str(path))

if __name__=='__main__':unittest.main()

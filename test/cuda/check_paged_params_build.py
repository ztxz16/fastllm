from pathlib import Path
import argparse,json,subprocess,time,tempfile
parser=argparse.ArgumentParser()
parser.add_argument('--nvcc',default='nvcc')
parser.add_argument('--output',type=Path)
args=parser.parse_args()
S=Path(__file__).resolve().parents[2]
R=args.output or Path(tempfile.mkdtemp(prefix='fastllm-paged-compile-'))
R.mkdir(parents=True,exist_ok=True)
nvcc=args.nvcc
includes=['include','include/utils','include/models','include/blocks','include/devices/cpu','include/devices/disk','third_party/json11','third_party/gguf','third_party/flashinfer','include/devices/cuda','src/devices/cuda/linear/marlin_dense_fp8','third_party/turbomind','include/devices/multicuda','include/devices/numas']
common=[nvcc,'-std=c++17','-DUSE_CUDA','--default-stream=per-thread','--expt-relaxed-constexpr','--expt-extended-lambda']
tu=S/'src/devices/cuda/attention/fastllm-attention.cu'
t=tu.read_text();start=t.index('constexpr int kFastllmPagedIntParamsMaxSmall');end=t.index('// CUDA kernel for batch copying data from input to paged KV cache',start)
probe=R/'paged-params-production.cu';probe.write_text('#include <cuda_runtime.h>\n#include <cstdint>\n#include <algorithm>\nvoid DeviceSync();\n'+t[start:end])
commands={
 'full_attention_sm60':common+['-arch=sm_60','-DFASTLLM_CUDA_LEGACY_ONLY']+['-I'+str(S/i) for i in includes]+['-c',str(tu),'-o',str(R/'attention-sm60.o')],
 'paged_params_mixed_sm60_sm75':common+['-gencode','arch=compute_60,code=sm_60','-gencode','arch=compute_75,code=sm_75','-c',str(probe),'-o',str(R/'paged-mixed.o')]
}
results=[]
for name,cmd in commands.items():
 start=time.time()
 with (R/(name+'.log')).open('w') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
 results.append(dict(name=name,command=cmd,exit=code,seconds=time.time()-start))
 (R/'compile-checks.json').write_text(json.dumps(results,indent=2)+'\n')
 print(name,code,flush=True)
assert all(r['exit']==0 for r in results),results

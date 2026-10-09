"""Hopper decode projections (rows 1..128): FP32 oracle, bias/zero inputs and graph replay.

Uses the production Data dispatch bridge, not a copied kernel implementation.
"""
import ctypes as C
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import torch

ROOT=Path(__file__).resolve().parents[2]
LIB=ROOT/'build-fastllm/tools/ftllm/libfastllm_tools.so'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional JSON results file')
    parser.add_argument('--library',type=Path,default=LIB,help='Native library to test')
    args_cli=parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    with tempfile.TemporaryDirectory() as tmp:
        bridge=Path(tmp)/'bridge.so'
        subprocess.run(['g++','-std=c++17','-shared','-fPIC','-O2',
            '-I'+str(ROOT/'include'),'-I'+str(ROOT/'third_party/json11'),'-I/usr/local/cuda/include',
            str(ROOT/'test/ops/hopperDecodeTestBridge.cpp'),str(args_cli.library),
            '-Wl,-rpath,'+str(args_cli.library.parent),'-o',str(bridge)],check=True)
        lib=C.CDLL(str(bridge))
        fn=lib.Fp8Sm90TestLinear
        fn.argtypes=[C.c_void_p]*5+[C.c_int]*3+[C.c_bool,C.c_int];fn.restype=C.c_bool
        lib.HopperTestSetExactThreshold.argtypes=[C.c_int]
        lib.HopperTestSetExactThreshold.restype=C.c_int
        lib.Fp8Sm90TestStream.restype=C.c_void_p
        for name,args in [('BeginCapture',[]),('EndCapture',[C.POINTER(C.c_void_p)]),('Instantiate',[C.c_void_p,C.POINTER(C.c_void_p)]),('Launch',[C.c_void_p]),('Destroy',[C.c_void_p]),('ExecDestroy',[C.c_void_p])]:
            api=getattr(lib,'FastllmCudaGraph'+name);api.argtypes=args
            api.restype=None if name.endswith('Destroy') else C.c_bool
        graphs=[]
        with torch.cuda.stream(torch.cuda.ExternalStream(lib.Fp8Sm90TestStream())):
            # Small scratch first; later shapes must not invalidate its captured pointers.
            cases=[(rows,N,K) for rows in [*range(1,32),32,33,63,64,65,100,127,128,129]
                   for N,K in [(5120,6144),(34816,5120),(5120,17408),
                               (16384,5120),(14336,5120)]]
            for rows,N,K in cases:
                for with_bias in [False,True]:
                    x=torch.randn(rows,K,device='cuda',dtype=torch.float16)*.3
                    x.mul_(torch.linspace(.2,5,rows,device='cuda').view(-1,1))  # Independent row scales.
                    w=torch.randn(N,K,device='cuda').clamp(-3,3).to(torch.float8_e4m3fn)
                    s=torch.rand(N//128,K//128,device='cuda')*.1+.002
                    bias=torch.randn(N,device='cuda')*.02 if with_bias else None
                    storage=torch.full((rows+1,N),17.,device='cuda',dtype=torch.float16)
                    y=storage[:rows]
                    y.fill_(float('nan'))
                    args=[x.data_ptr(),w.data_ptr(),s.data_ptr(),bias.data_ptr() if bias is not None else 0,y.data_ptr(),rows,K,N,False,0]
                    assert fn(*args)
                    torch.cuda.synchronize()
                    g,ex=C.c_void_p(),C.c_void_p()
                    assert lib.FastllmCudaGraphBeginCapture()
                    assert fn(*args)
                    assert lib.FastllmCudaGraphEndCapture(C.byref(g))
                    assert lib.FastllmCudaGraphInstantiate(g,C.byref(ex))
                    lib.FastllmCudaGraphDestroy(g)
                    graphs.append((ex,x,w,s,bias,y,args,storage[rows:]))
            records=[]
            for ex,x,w,s,bias,y,args,guard in graphs:
                N,K=w.shape
                rows=x.shape[0]
                modes=['random','changed']+(['row_zero'] if rows>1 else [])+['zero']
                for mode in modes:
                    if mode=='changed':x[0].mul_(.7)
                    if mode=='row_zero':x[0].zero_();x[-1].mul_(-.5)
                    if mode=='zero':x.zero_()
                    y.fill_(float('nan'))
                    assert lib.FastllmCudaGraphLaunch(ex)
                    torch.cuda.synchronize()
                    xf=x.float().view(rows,K//128,128)
                    xs=(xf.abs().amax(-1,keepdim=True)/448).clamp_min(1e-10)
                    xq=((xf/xs).to(torch.float8_e4m3fn).float()*xs).reshape(rows,K)
                    wf=w.float()*s.repeat_interleave(128,0).repeat_interleave(128,1)
                    ref=(xq@wf.T).half().float()
                    native=x.float()@wf.T
                    if bias is not None:
                        ref=(ref+bias).half().float();native+=bias
                    diff=y.float()-ref
                    nrmse=(diff.norm()/ref.norm().clamp_min(1e-8)).item()
                    row_nrmse=(diff.norm(dim=1)/ref.norm(dim=1).clamp_min(1e-8)).tolist()
                    max_relative=(diff.abs().max()/ref.abs().max().clamp_min(1e-8)).item()
                    quant_nrmse=((y.float()-native).norm()/native.norm().clamp_min(1e-8)).item()
                    assert torch.isfinite(y).all()
                    assert torch.all(guard==17), 'TMA store wrote beyond the last row'
                    assert nrmse<.002 and max_relative<.006,(N,K,mode,nrmse,max_relative)
                    assert max(row_nrmse)<.002,(rows,N,K,mode,row_nrmse)
                    assert quant_nrmse<.04,(N,K,mode,quant_nrmse)
                    record=dict(rows=rows,N=N,K=K,bias=bias is not None,mode=mode,nrmse=nrmse,row_nrmse=row_nrmse,max_relative=max_relative,quant_nrmse=quant_nrmse)
                    records.append(record);print(json.dumps(record),flush=True)
                y.fill_(17)
                for invalid in [1,2,3,4,5,6,8,9]:
                    bad=args.copy();bad[-1]=invalid
                    assert not fn(*bad),invalid
                for index,value in [(5,0),(6,K-1),(7,N-1)]+([(8,True)] if rows<32 else []):
                    bad=args.copy();bad[index]=value
                    assert not fn(*bad),(index,value)
                if bias is not None:
                    bad=args.copy();bad[-1]=7
                    assert not fn(*bad), 'invalid bias dtype accepted'
                previous=lib.HopperTestSetExactThreshold(rows+1)
                try:
                    assert not fn(*args), 'exact verification entered W8A8 decode'
                finally:
                    lib.HopperTestSetExactThreshold(previous)
                assert torch.all(y==17), 'rejected calls modified output'
                lib.FastllmCudaGraphExecDestroy(ex)
            if args_cli.output:
                args_cli.output.write_text(json.dumps(records,indent=2))
        print('PASS: production FP8 decode, graph scratch growth/replay, changed and zero inputs, bias, exact-row exclusion and rejection guards',flush=True)

if __name__=='__main__':main()

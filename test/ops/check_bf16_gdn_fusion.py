"""Hopper BF16 GDN/gated RMSNorm: production fallback, FP32 oracle and graph replay."""
import argparse,ctypes as C,hashlib,json,math,subprocess,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',type=Path,default=ROOT/'build-fastllm/tools/ftllm/libfastllm_tools.so')
p.add_argument('--output',type=Path,required=True)
args=p.parse_args();records=[];checks=0
torch.backends.cuda.matmul.allow_tf32=False
def digest(*tensors):return hashlib.sha256(b''.join(t.view(torch.uint8).cpu().numpy().tobytes() for t in tensors)).hexdigest()
def error(x,ref):return ((x.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-30)).item()
with tempfile.TemporaryDirectory() as tmp:
    bridge=Path(tmp)/'bridge.so'
    cmd=['g++','-std=c++17','-O2','-fPIC','-shared','-I'+str(ROOT/'include'),
         '-I'+str(ROOT/'include/devices/cuda'),'-I'+str(ROOT/'third_party/json11'),'-I/usr/local/cuda/include',
         str(ROOT/'test/ops/bf16GdnFusionTestBridge.cpp'),str(args.library),
         '-Wl,-rpath,'+str(args.library.parent),'-L/usr/local/cuda/lib64','-lcudart','-o',str(bridge)]
    subprocess.run(cmd,check=True);lib=C.CDLL(str(bridge))
    lib.GdnTestStream.restype=C.c_void_p
    run=lib.GdnTestRun;run.argtypes=[C.c_int]*4+[C.c_void_p]*11+[C.c_float]*2;run.restype=C.c_bool
    gate=lib.GdnTestGate;gate.argtypes=[C.c_int]*5+[C.c_void_p]*5+[C.c_float];gate.restype=C.c_bool
    lib.GdnTestExact.argtypes=[C.c_int];lib.GdnTestExact.restype=C.c_int
    for name,params in [('BeginCapture',[]),('EndCapture',[C.POINTER(C.c_void_p)]),('Instantiate',[C.c_void_p,C.POINTER(C.c_void_p)]),('Launch',[C.c_void_p]),('Destroy',[C.c_void_p]),('ExecDestroy',[C.c_void_p])]:
        fn=getattr(lib,'FastllmCudaGraph'+name);fn.argtypes=params;fn.restype=None if name.endswith('Destroy') else C.c_bool
    def capture(fn):
        graph,ex=C.c_void_p(),C.c_void_p()
        assert lib.FastllmCudaGraphBeginCapture();assert fn()
        assert lib.FastllmCudaGraphEndCapture(C.byref(graph))
        assert lib.FastllmCudaGraphInstantiate(graph,C.byref(ex));lib.FastllmCudaGraphDestroy(graph)
        return ex
    with torch.cuda.stream(torch.cuda.ExternalStream(lib.GdnTestStream())):
        for hk,hv,mode in [(16,48,'random'),(8,24,'random'),(2,4,'random'),(1,1,'random'),
                           (16,48,'wide'),(16,48,'zero'),(16,48,'gates')]:
            torch.manual_seed(42+hk)
            q,k=[torch.randn(hk,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            v=torch.randn(hv,128,device='cuda',dtype=torch.bfloat16)*.1
            a,b=[torch.randn(hv,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            w=torch.full((128,),1/math.sqrt(128),device='cuda')
            l=torch.randn(hv,device='cuda')*.3;d=torch.randn(hv,device='cuda')*.1
            initial=torch.randn(hv,128,128,device='cuda',dtype=torch.bfloat16)*.02
            if mode=='wide':initial.mul_(1e6);v.mul_(1e6)
            if mode=='zero':initial.zero_();q.zero_();k.zero_();v.zero_()
            if mode=='gates':a.copy_(torch.linspace(-40,40,hv,device='cuda'));b.copy_(torch.linspace(-40,40,hv,device='cuda'))
            storage=torch.full((hv+1,128,128),17.,device='cuda',dtype=torch.bfloat16);state=storage[:hv];state.copy_(initial)
            legacy=initial.clone();ref_state=initial.clone()
            out=torch.full((hv+1,128),17.,device='cuda',dtype=torch.bfloat16);y=out[:hv];y_old=torch.empty_like(y)
            scratch=torch.empty(2*hk*128+2*hv,device='cuda',dtype=torch.bfloat16)
            def call(fused,state,y,variant=0):return run(fused,hk,hv,variant,*[t.data_ptr() for t in [q,k,v,a,b,w,l,d,state,y,scratch]],1e-6,1/math.sqrt(128))
            ex=capture(lambda:call(1,state,y))
            max_e=max_s=max_legacy_e=max_legacy_s=0.
            for step in range(32):
                if step==7:q.mul_(-.7);v.mul_(.5)
                if step==15:b.mul_(-1)
                qn=(q.float()*torch.rsqrt(q.float().square().mean(-1,keepdim=True)+1e-6)*w).bfloat16()
                qn=(qn.float()/math.sqrt(128)).bfloat16().float().repeat_interleave(hv//hk,0)
                kn=(k.float()*torch.rsqrt(k.float().square().mean(-1,keepdim=True)+1e-6)*w).bfloat16().float().repeat_interleave(hv//hk,0)
                gn=(-l.exp()*torch.nn.functional.softplus(a.float()+d)).bfloat16().float().exp()
                bn=torch.sigmoid(b.float()).bfloat16().float()
                dec=(ref_state.float()*gn[:,None,None]).bfloat16().float()
                delta=(v.float()-(dec*kn[:,:,None]).sum(1))*bn[:,None]
                ref_state=(dec+kn[:,:,None]*delta[:,None,:]).bfloat16()
                ref=(ref_state.float()*qn[:,:,None]).sum(1).bfloat16()
                assert lib.FastllmCudaGraphLaunch(ex);assert call(0,legacy,y_old);torch.cuda.synchronize()
                e,se=error(y,ref),error(state,ref_state);le,ls=error(y,y_old),error(state,legacy)
                max_e=max(max_e,e);max_s=max(max_s,se);max_legacy_e=max(max_legacy_e,le);max_legacy_s=max(max_legacy_s,ls)
                assert max(e,se,le,ls)<.008,(hk,hv,mode,step,e,se,le,ls)
                assert torch.isfinite(y).all() and torch.isfinite(state).all()
                assert torch.all(out[hv:]==17) and torch.all(storage[hv:]==17)
                checks+=4
            lib.FastllmCudaGraphExecDestroy(ex)
            rec=dict(op='gdn',hk=hk,hv=hv,mode=mode,max_nrmse=max_e,max_state_nrmse=max_s,
                     legacy_nrmse=max_legacy_e,legacy_state_nrmse=max_legacy_s,legacy_sha256=digest(legacy,y_old))
            records.append(rec);print(json.dumps(rec),flush=True)
            saved=state.clone();y.fill_(17)
            for variant in range(1,9):assert not call(1,state,y,variant)
            previous=lib.GdnTestExact(2)
            try:assert not call(1,state,y)
            finally:lib.GdnTestExact(previous)
            torch.cuda.synchronize();assert torch.equal(state,saved) and torch.all(y==17)
        for dtype in [torch.float16,torch.bfloat16]:
            for rows in [1,4,24,48,96,1536]:
                for mode in ['random','zero','wide'] if dtype==torch.bfloat16 else ['random']:
                    torch.manual_seed(123+rows)
                    x=torch.randn(rows,128,device='cuda',dtype=dtype)
                    z=torch.randn_like(x)*3;w=torch.randn(128,device='cuda')*.2+1
                    if mode=='zero':x.zero_();z.zero_()
                    if mode=='wide':x.mul_(1e15);z.mul_(1e3)
                    y=torch.empty_like(x);scratch=torch.empty_like(x);legacy=torch.empty_like(x)
                    def gcall(fused,x,z,y,variant=0,cols=128):return gate(fused,1 if dtype==torch.float16 else 2,rows,cols,variant,
                        x.data_ptr(),w.data_ptr(),z.data_ptr(),y.data_ptr(),scratch.data_ptr(),1e-6)
                    assert gcall(0,x,z,legacy)
                    fused=2 if dtype==torch.float16 else 1
                    ex=capture(lambda:gcall(fused,x,z,y));assert lib.FastllmCudaGraphLaunch(ex);torch.cuda.synchronize()
                    if dtype==torch.bfloat16:
                        assert torch.equal(y,legacy),('gated norm differs from unfused BF16',rows,mode,error(y,legacy))
                        rms=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*w).bfloat16().float()
                        silu=torch.nn.functional.silu(z.float()).bfloat16().float()
                        assert error(y,(rms*silu).bfloat16())<.008
                    else:assert error(y,legacy)<.002
                    lib.FastllmCudaGraphExecDestroy(ex)
                    if dtype==torch.bfloat16:
                        xi=x.clone();zi=z.clone();assert gcall(1,xi,zi,xi);assert torch.equal(xi,legacy)
                        xi=x.clone();zi=z.clone();assert gcall(1,xi,zi,zi);assert torch.equal(zi,legacy)
                        y.fill_(17)
                        for variant in range(1,6):assert not gcall(1,x,z,y,variant)
                        assert not gcall(1,x,z,y,cols=64)
                        previous=lib.GdnTestExact(2)
                        try:assert not gcall(1,x,z,y)
                        finally:lib.GdnTestExact(previous)
                        assert torch.all(y==17)
                    checks+=1
                    rec=dict(op='gated_norm',dtype=str(dtype),rows=rows,mode=mode,legacy_sha256=digest(legacy))
                    records.append(rec);print(json.dumps(rec),flush=True)
args.output.write_text(json.dumps(dict(checks=checks,records=records),indent=2))
print('PASS: recurrent FP32 oracle/legacy comparison, graph replay, BF16 intermediate rounding, in-place output and rejection guards',flush=True)

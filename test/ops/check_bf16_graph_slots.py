"""BF16 graph slots: replay with changed inputs, sparse/reordered/recycled slots."""
import argparse,ctypes as C,json,math,subprocess,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
library=ROOT/'build-fastllm/tools/ftllm/libfastllm_tools.so';records=[];checks=0
with tempfile.TemporaryDirectory() as tmp:
    bridge=Path(tmp)/'bridge.so'
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared',
        '-I'+str(ROOT/'include'),'-I'+str(ROOT/'include/devices/cuda'),'-I'+str(ROOT/'third_party/json11'),
        '-I/usr/local/cuda/include',str(ROOT/'test/ops/bf16GdnFusionTestBridge.cpp'),str(library),
        '-Wl,-rpath,'+str(library.parent),'-L/usr/local/cuda/lib64','-lcudart','-o',str(bridge)],check=True)
    lib=C.CDLL(str(bridge));lib.GdnTestStream.restype=C.c_void_p
    slots=lib.GraphTestSlots;slots.argtypes=[C.c_int]*5+[C.c_void_p]*8;slots.restype=C.c_bool
    eager=lib.GdnTestRun;eager.argtypes=[C.c_int]*4+[C.c_void_p]*11+[C.c_float]*2;eager.restype=C.c_bool
    conv=lib.GraphTestConv;conv.argtypes=[C.c_int]*4+[C.c_void_p]*6;conv.restype=C.c_bool
    for name,params in [('BeginCapture',[]),('EndCapture',[C.POINTER(C.c_void_p)]),('Instantiate',[C.c_void_p,C.POINTER(C.c_void_p)]),('Launch',[C.c_void_p]),('Destroy',[C.c_void_p]),('ExecDestroy',[C.c_void_p])]:
        fn=getattr(lib,'FastllmCudaGraph'+name);fn.argtypes=params;fn.restype=None if name.endswith('Destroy') else C.c_bool
    def capture(fn):
        g,ex=C.c_void_p(),C.c_void_p();assert lib.FastllmCudaGraphBeginCapture();assert fn()
        assert lib.FastllmCudaGraphEndCapture(C.byref(g));assert lib.FastllmCudaGraphInstantiate(g,C.byref(ex))
        lib.FastllmCudaGraphDestroy(g);return ex
    def ptrs(ts):return [t.data_ptr() for t in ts]
    with torch.cuda.stream(torch.cuda.ExternalStream(lib.GdnTestStream())):
        for batch,hk,hv in [(1,16,48),(2,16,48),(4,16,48),(8,16,48),(16,16,48),(31,16,48),(4,1,1),(4,2,4)]:
            torch.manual_seed(719+batch+hk);capacity=batch+7
            c=torch.randn(batch,(2*hk+hv)*128,device='cuda',dtype=torch.bfloat16)
            ba=torch.randn(batch,2*hv,device='cuda',dtype=torch.bfloat16)
            w=torch.full((128,),1/math.sqrt(128),device='cuda');l=torch.randn(hv,device='cuda')*.3;d=torch.randn(hv,device='cuda')*.1
            s=torch.randn(capacity,hv,128,128,device='cuda',dtype=torch.bfloat16)*.01;reference=s.clone()
            ids=torch.arange(batch,device='cuda',dtype=torch.int32);y=torch.empty(batch,hv,128,device='cuda',dtype=torch.bfloat16);refy=torch.empty_like(y)
            scratch=torch.empty(2*hk*128+2*hv,device='cuda',dtype=torch.bfloat16)
            def call(variant=0):return slots(batch,hk,hv,capacity,variant,*ptrs([c,ba,w,l,d,s,ids,y]))
            assert call();s.copy_(reference);torch.cuda.synchronize();ex=capture(call)
            for step in range(32):
                selected=torch.randperm(capacity)[:batch].tolist();ids.copy_(torch.tensor(selected,device='cuda',dtype=torch.int32))
                c.normal_();ba.normal_()
                if step==16:
                    # A released slot is reinitialized for a different request.
                    s[selected[0]].zero_();reference[selected[0]].zero_()
                for row,slot in enumerate(selected):
                    tensors=[c[row,:hk*128],c[row,hk*128:hk*256],c[row,hk*256:],ba[row,hv:],ba[row,:hv],w,l,d,reference[slot],refy[row],scratch]
                    assert eager(1,hk,hv,0,*ptrs(tensors),1e-6,1/math.sqrt(128))
                assert lib.FastllmCudaGraphLaunch(ex);torch.cuda.synchronize()
                assert torch.equal(s,reference),('state',batch,hk,hv,step)
                assert torch.equal(y,refy),('output',batch,hk,hv,step)
                checks+=2
            lib.FastllmCudaGraphExecDestroy(ex)
            saved=s.clone();y.fill_(17)
            for variant in range(1,8):assert not call(variant)
            assert torch.equal(s,saved) and torch.all(y==17)
            records.append(dict(op='gdn_slots',batch=batch,hk=hk,hv=hv,steps=32,bitwise_equal=True))
            print(json.dumps(records[-1]),flush=True)
        for batch,channels,bias in [(1,10240,0),(2,10240,1),(4,640,1),(31,384,0)]:
            capacity=batch+7;torch.manual_seed(138+batch)
            s=torch.randn(capacity,channels,4,device='cuda',dtype=torch.bfloat16);ref=s.clone()
            ids=torch.arange(batch,device='cuda',dtype=torch.int32)
            x=torch.randn(batch,channels,device='cuda',dtype=torch.bfloat16);w=torch.randn(channels,4,device='cuda')*.2
            b=torch.randn(channels,device='cuda')*.1;y=torch.empty_like(x)
            def call():return conv(batch,channels,capacity,bias,*ptrs([s,ids,x,w,b,y]))
            assert call();s.copy_(ref);torch.cuda.synchronize();ex=capture(call);worst=0.
            for step in range(32):
                selected=torch.randperm(capacity)[:batch].tolist();ids.copy_(torch.tensor(selected,device='cuda',dtype=torch.int32));x.normal_()
                shifted=torch.cat((ref[selected,:,1:],x[:,:,None]),-1);ref[selected]=shifted
                z=(shifted.float()*w).sum(-1)+(b if bias else 0);z=z.bfloat16().float()
                target=torch.nn.functional.silu(z).bfloat16()
                assert lib.FastllmCudaGraphLaunch(ex);torch.cuda.synchronize();assert torch.equal(s,ref)
                err=((y.float()-target.float()).norm()/target.float().norm().clamp_min(1e-20)).item();worst=max(worst,err)
                assert err<.003,err;checks+=2
            lib.FastllmCudaGraphExecDestroy(ex)
            records.append(dict(op='conv_slots',batch=batch,channels=channels,bias=bool(bias),steps=32,max_nrmse=worst))
            print(json.dumps(records[-1]),flush=True)
args.output.write_text(json.dumps(dict(passed=True,checks=checks,records=records),indent=2))
print('PASS',checks,flush=True)

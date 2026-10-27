"""Production BF16 pointwise/GDN dispatch against FP32 references and graph replay."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import torch

root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',type=Path,default=root/'build-fastllm/tools/ftllm/libfastllm_tools.so')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--legacy-only',action='store_true')
args=p.parse_args()
torch.backends.cuda.matmul.allow_tf32=False
records=[]

def check(y,ref,label,dtype):
    assert torch.isfinite(y).all(),label
    err=(y.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-20)
    tolerance={torch.float32:2e-5,torch.float16:.003,torch.bfloat16:.008}[dtype]
    assert err<tolerance,(label,err.item(),tolerance)
    return err.item()

with tempfile.TemporaryDirectory() as temp:
    bridge=Path(temp)/'bridge.so'
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared',
                    '-I'+str(root/'include'),'-I'+str(root/'include/devices/cuda'),
                    '-I'+str(root/'third_party/json11'),'-I/usr/local/cuda/include',
                    str(root/'test/ops/bf16CompatTestBridge.cpp'),str(args.library),
                    '-Wl,-rpath,'+str(args.library.parent),'-o',str(bridge)],check=True)
    lib=C.CDLL(str(bridge))
    lib.Bf16CompatStream.restype=C.c_void_p
    conv=lib.Bf16CompatConv;conv.argtypes=[C.c_int]*7+[C.c_void_p]*4;conv.restype=C.c_bool
    mask=lib.Bf16CompatDecayMask;mask.argtypes=[C.c_int]*3+[C.c_void_p]*2;mask.restype=None
    point=lib.Bf16CompatPointwise;point.argtypes=[C.c_int]*4+[C.c_void_p]*5+[C.c_float];point.restype=None
    gdn=lib.Bf16CompatGdn;gdn.argtypes=[C.c_int]*6+[C.c_void_p]*8+[C.c_float];gdn.restype=C.c_bool
    for name,types in [('BeginCapture',[]),('EndCapture',[C.POINTER(C.c_void_p)]),
                       ('Instantiate',[C.c_void_p,C.POINTER(C.c_void_p)]),('Launch',[C.c_void_p]),
                       ('Destroy',[C.c_void_p]),('ExecDestroy',[C.c_void_p])]:
        fn=getattr(lib,'FastllmCudaGraph'+name);fn.argtypes=types
        fn.restype=None if name.endswith('Destroy') else C.c_bool
    def capture(fn):
        fn();torch.cuda.synchronize()
        graph,ex=C.c_void_p(),C.c_void_p()
        assert lib.FastllmCudaGraphBeginCapture();fn()
        assert lib.FastllmCudaGraphEndCapture(C.byref(graph))
        assert lib.FastllmCudaGraphInstantiate(graph,C.byref(ex));lib.FastllmCudaGraphDestroy(graph)
        return ex

    with torch.cuda.stream(torch.cuda.ExternalStream(lib.Bf16CompatStream())):
        dtypes=[torch.float32,torch.float16]+([] if args.legacy_only else [torch.bfloat16])
        for code,dtype in enumerate(dtypes):
            # Independent seed keeps existing operator regression hashes stable.
            torch.manual_seed(900+code)
            for batch,channels,length,kernel,stride,pad in [(1,7,4,4,1,0),(31,16,4,4,1,0),(2,16,2048,4,1,3),(3,7,17,3,2,1)]:
                for bias in [False,True]:
                    x=torch.randn(batch,channels,length,device='cuda',dtype=dtype)
                    w=torch.randn(channels,1,kernel,device='cuda');b=torch.randn(channels,device='cuda') if bias else None
                    n=(length+2*pad-kernel)//stride+1
                    storage=torch.full((batch+1,channels,n),17,device='cuda',dtype=dtype);y=storage[:batch]
                    assert conv(code,batch,channels,length,kernel,stride,pad,x.data_ptr(),w.data_ptr(),b.data_ptr() if bias else 0,y.data_ptr())
                    ref=torch.nn.functional.conv1d(x.float(),w,b,stride,pad,groups=channels).to(dtype)
                    err=check(y,ref,('conv',batch,length,bias),dtype)
                    assert torch.all(storage[batch:]==17)
                    record=dict(op='conv',dtype=str(dtype),batch=batch,length=length,bias=bias,nrmse=err,
                                sha256=hashlib.sha256(y.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                    records.append(record);print(json.dumps(record),flush=True)
            torch.manual_seed(100+code)
            for rows,cols in [(1,7),(31,64),(100,128)]:
                x=(-torch.rand(rows,cols,device='cuda').cumsum(-1)*.1).to(dtype)
                storage=torch.full((rows+1,cols,cols),17,device='cuda',dtype=dtype);y=storage[:rows]
                ref=torch.exp(x.float().unsqueeze(-1)-x.float().unsqueeze(-2)).tril().to(dtype)
                mask(code,rows,cols,x.data_ptr(),y.data_ptr());torch.cuda.synchronize()
                err=check(y,ref,('decay_mask',rows,cols),dtype)
                assert torch.all(storage[rows:]==17)
                record=dict(op='decay_mask',dtype=str(dtype),rows=rows,cols=cols,nrmse=err,
                            sha256=hashlib.sha256(y.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                records.append(record);print(json.dumps(record),flush=True)
            for rows,channels in [(1,7),(31,48),(100,80)]:
                x=torch.randn(rows,channels,device='cuda',dtype=dtype)
                a=torch.linspace(-30,30,rows*channels,device='cuda').reshape(rows,channels).to(dtype)
                log=torch.linspace(-2,2,channels,device='cuda');dt=torch.linspace(-1,1,channels,device='cuda')
                storage=torch.full((rows+1,channels),17,device='cuda',dtype=dtype);y=storage[:rows]
                for op in range(3):
                    scale=-.5 if op==1 else 1.
                    call=lambda:point(op,code,rows,channels,x.data_ptr(),a.data_ptr(),y.data_ptr(),log.data_ptr(),dt.data_ptr(),scale)
                    ex=capture(call)
                    for mode in ['changed','zero']:
                        x.copy_(torch.randn_like(x)*3 if mode=='changed' else torch.zeros_like(x))
                        if dtype == torch.float16 and op == 0:x.clamp_(-8,8)
                        if mode=='zero':a.zero_()
                        ref=torch.exp(x.float()) if op==0 else -torch.exp(log)*torch.nn.functional.softplus(a.float()+dt)*(scale if op==1 else 1.)
                        sigmoid_ref=torch.sigmoid(x.float()).to(dtype)
                        ref=ref.to(dtype);y.fill_(float('nan'))
                        assert lib.FastllmCudaGraphLaunch(ex);torch.cuda.synchronize()
                        err=check(y,ref,('pointwise',op,rows,mode),dtype)
                        if op==2:check(x,sigmoid_ref,('sigmoid',rows,mode),dtype)
                        assert torch.all(storage[rows:]==17)
                        digest=hashlib.sha256(y.view(torch.uint8).cpu().numpy().tobytes()+x.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                        record=dict(op=op,dtype=str(dtype),rows=rows,channels=channels,mode=mode,nrmse=err,sha256=digest)
                        records.append(record);print(json.dumps(record),flush=True)
                    lib.FastllmCudaGraphExecDestroy(ex)
            for batch,kd,vd in [(1,17,33),(1,128,128),(2,128,128),(8,128,128),(31,128,128),(32,128,128),(64,128,128),(100,128,128)]:
                hk,hv=2,4
                shape=(batch,hv,kd,vd)
                q=torch.nn.functional.normalize(torch.randn(batch,hk,kd,device='cuda'),dim=-1).to(dtype)
                k=torch.nn.functional.normalize(torch.randn_like(q).float(),dim=-1).to(dtype)
                v=torch.randn(batch,hv,vd,device='cuda',dtype=dtype)*.1
                g=(-torch.rand(batch,hv,device='cuda')*.3-.1).to(dtype)
                b=(torch.rand(batch,hv,device='cuda')*.1).to(dtype)
                initial=torch.randn(shape,device='cuda',dtype=dtype)*.02
                for use_pointers in [False,True]:
                    states=[torch.empty(1,hv,kd,vd,device='cuda',dtype=dtype) for _ in range(batch)] if use_pointers else [torch.empty_like(initial)]
                    pointers=torch.tensor([s.data_ptr() for s in states],device='cuda',dtype=torch.int64) if use_pointers else None
                    storage=torch.full((batch+1,hv,vd),17,device='cuda',dtype=dtype);y=storage[:batch]
                    scale=.37
                    call=lambda:gdn(code,batch,hk,hv,kd,vd,q.data_ptr(),k.data_ptr(),v.data_ptr(),g.data_ptr(),b.data_ptr(),states[0].data_ptr(),y.data_ptr(),pointers.data_ptr() if use_pointers else 0,scale)
                    ex=capture(lambda:call() or (_ for _ in ()).throw(AssertionError('GDN dispatch rejected input')))
                    for i,state in enumerate(states):state.copy_(initial[i:i+1] if use_pointers else initial)
                    ref_state=initial.clone()
                    for step in range(8):
                        if step==3:q.mul_(-.7);v.mul_(.5)
                        kk=k.float().repeat_interleave(hv//hk,dim=1)
                        qq=(q.float()*scale).to(dtype).float().repeat_interleave(hv//hk,dim=1)
                        decayed=(ref_state.float()*g.float().exp()[...,None,None]).to(dtype).float()
                        delta=(v.float()-(decayed*kk[...,None]).sum(-2))*b.float()[...,None]
                        ref_state=(decayed+kk[...,None]*delta[...,None,:]).to(dtype)
                        ref=(ref_state.float()*qq[...,None]).sum(-2).to(dtype)
                        y.fill_(float('nan'))
                        assert lib.FastllmCudaGraphLaunch(ex);torch.cuda.synchronize()
                        actual_state=torch.cat(states,dim=0) if use_pointers else states[0]
                        err=check(y,ref,('gdn',batch,use_pointers,step),dtype)
                        state_err=check(actual_state,ref_state,('state',batch,use_pointers,step),dtype)
                        assert torch.all(storage[batch:]==17)
                        record=dict(op='gdn',dtype=str(dtype),batch=batch,kd=kd,vd=vd,pointers=use_pointers,step=step,nrmse=err,state_nrmse=state_err,
                                    sha256=hashlib.sha256(y.view(torch.uint8).cpu().numpy().tobytes()+actual_state.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                        records.append(record);print(json.dumps(record),flush=True)
                    lib.FastllmCudaGraphExecDestroy(ex)
args.output.write_text(json.dumps(records,indent=2))
print('PASS: pointwise and recurrent GDN references, graph replay, state updates and output bounds.',flush=True)

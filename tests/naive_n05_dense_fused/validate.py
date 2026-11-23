"""Eight-device dense SwiGLU/TP reference, tail/chunk, guards and graph replay."""
import argparse, ctypes as C, concurrent.futures as F, threading, json
import torch
p=argparse.ArgumentParser();p.add_argument('library');a=p.parse_args()
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
lib=C.CDLL(a.library);rt=C.CDLL('libcudart.so.12')
P=C.c_void_p;I=C.c_int
lib.fastshared_tp_create.argtypes=[C.POINTER(I),I,C.POINTER(P)]
lib.fastshared_tp_destroy.argtypes=[P]
lib.fastshared_tp_dense_supported.argtypes=[P]
lib.fastshared_tp_run_dense.argtypes=[P,I,I]+[P]*5+[C.c_float]+[P]*2
lib.fastshared_tp_dense_admit.argtypes=[P,I,I,I]
rt.cudaStreamBeginCapture.argtypes=[P,I]
rt.cudaStreamEndCapture.argtypes=[P,C.POINTER(P)]
rt.cudaGraphInstantiateWithFlags.argtypes=[C.POINTER(P),P,C.c_ulonglong]
rt.cudaGraphLaunch.argtypes=[P,P]
rt.cudaGraphDestroy.argtypes=[P];rt.cudaGraphExecDestroy.argtypes=[P]
def ck(status):
    assert status==0,status
state=P();ck(lib.fastshared_tp_create((I*8)(*range(8)),1,C.byref(state)))
assert lib.fastshared_tp_dense_supported(state)==1
bar=threading.Barrier(8);pool=F.ThreadPoolExecutor(8)
H=4096;D=2048;G=8
def parallel(fn):
    fs=[pool.submit(fn,r) for r in range(8)]
    return [f.result() for f in fs]
def initialize(r):
    torch.cuda.set_device(r)
    gen=torch.Generator(device=r).manual_seed(199+r)
    weights=[(torch.randn(shape,device=r,generator=gen)*.025).bfloat16() for shape in [(D,H),(D,H),(H,D)]]
    x=torch.empty((64,H),device=r,dtype=torch.bfloat16)
    mid=torch.full((64*D+G*2,),71.,device=r,dtype=torch.bfloat16)
    out=torch.full((64*H+G*2,),73.,device=r,dtype=torch.bfloat16)
    torch.cuda.synchronize(r)
    gamma=(torch.arange(H,device=r).float()%11)/100+1
    return dict(weights=weights,x=x,mid=mid,out=out,gamma=gamma)
data=parallel(initialize);max_mid=max_out=0.;cases=0
def invoke(r,rows):
    d=data[r]
    return lib.fastshared_tp_run_dense(state,r,rows,d['x'].data_ptr(),*[t.data_ptr() for t in d['weights']],d['gamma'].data_ptr(),1.e-5,d['mid'].data_ptr()+G*2,d['out'].data_ptr()+G*2)
def run_case(rows,amp,graph=False):
    global max_mid,max_out,cases
    cpu=(torch.randn((rows,H),generator=torch.Generator().manual_seed(991+cases))*amp).bfloat16()
    def worker(r):
        torch.cuda.set_device(r);d=data[r]
        d['x'][:rows].copy_(cpu);d['mid'].fill_(71);d['out'].fill_(73)
        gate,up,down=d['weights']
        def normalized():
            raw=d['x'][:rows].float();inv=torch.rsqrt(raw.square().mean(-1,keepdim=True)+1.e-5)
            return ((raw*inv).bfloat16().float()*d['gamma']).bfloat16()
        x=normalized()
        # cuBLAS BF16 GEMMs, preserving the existing BF16 nonlinear boundaries.
        ref_mid=torch.nn.functional.silu(x@gate.T)*(x@up.T)
        ref_out=ref_mid@down.T
        torch.cuda.synchronize(r);bar.wait()
        if graph:
            g=P();exe=P();ck(rt.cudaStreamBeginCapture(P(2),1));ck(invoke(r,rows))
            ck(rt.cudaStreamEndCapture(P(2),C.byref(g)))
            ck(rt.cudaGraphInstantiateWithFlags(C.byref(exe),g,0));bar.wait()
            for replay in range(3):
                # Change inputs after capture; the same graph must observe each change.
                d['x'][:rows].copy_(cpu*(replay+1));torch.cuda.synchronize(r);bar.wait()
                ck(rt.cudaGraphLaunch(exe,P(2)));ck(rt.cudaDeviceSynchronize());bar.wait()
            x=normalized();ref_mid=torch.nn.functional.silu(x@gate.T)*(x@up.T)
            ref_out=ref_mid@down.T
            ck(rt.cudaGraphExecDestroy(exe));ck(rt.cudaGraphDestroy(g))
        else:
            ck(invoke(r,rows));ck(rt.cudaDeviceSynchronize())
        torch.cuda.synchronize(r)
        assert bool((d['mid'][:G]==71).all()) and bool((d['mid'][G+rows*D:]==71).all())
        assert bool((d['out'][:G]==73).all()) and bool((d['out'][G+rows*H:]==73).all())
        got_mid=d['mid'][G:G+rows*D].reshape(rows,D).float().cpu()
        got_out=d['out'][G:G+rows*H].reshape(rows,H).float().cpu()
        return got_mid,got_out,ref_mid.float().cpu(),ref_out.float().cpu()
    results=parallel(worker)
    summed=torch.zeros_like(results[0][3])
    for row in results:summed+=row[3]
    ref=summed.bfloat16().float()
    def l2(x,y):return float(((x-y).norm(dim=1)/y.norm(dim=1).clamp_min(1.e-20)).max())
    for gm,go,rm,ro in results:
        assert torch.isfinite(gm).all() and torch.isfinite(go).all()
        em,eo=l2(gm,rm),l2(go,ref);max_mid=max(max_mid,em);max_out=max(max_out,eo)
        assert em<.006 and eo<.006,(rows,amp,graph,em,eo)
        assert torch.equal(go,results[0][1]),'rank disagreement'
    cases+=1
    print(json.dumps(dict(rows=rows,amplitude=amp,graph=graph,mid_l2=max_mid,out_l2=max_out)),flush=True)
try:
    # Rejection is before any launch and leaves both output buffers untouched.
    def rejects(r):
        torch.cuda.set_device(r);d=data[r];d['mid'].fill_(71);d['out'].fill_(73);torch.cuda.synchronize(r)
        args=[state,r,8,d['x'].data_ptr(),*[t.data_ptr() for t in d['weights']],d['gamma'].data_ptr(),1.e-5,d['mid'].data_ptr()+16,d['out'].data_ptr()+16]
        for index,value in [(0,None),(1,-1),(1,8),(2,0),(2,65),(3,None),(3,args[3]+2),(10,args[3]),(9,args[10]),(7,None),(8,0.),(8,float('nan'))]:
            bad=args.copy();bad[index]=value;assert lib.fastshared_tp_run_dense(*bad)!=0
        ck(rt.cudaDeviceSynchronize());assert bool((d['mid']==71).all()) and bool((d['out']==73).all())
    parallel(rejects)
    for bad_rank in [-1,0,3,7]:
        got=parallel(lambda r:lib.fastshared_tp_dense_admit(state,r,8,r!=bad_rank))
        assert got==[int(bad_rank<0)]*8,got
    assert parallel(lambda r:lib.fastshared_tp_dense_admit(state,r,7 if r==3 else 8,1))==[0]*8
    assert parallel(lambda r:lib.fastshared_tp_dense_admit(state,r,8,1))==[1]*8
    for rows in [1,2,3,4,5,6,7,8,9,13,16,31,33,64]:
        for amp in [0.,.01,1.,10.]:run_case(rows,amp)
    for rows in [1,7,8,13,33,64]:run_case(rows,.1,True)
    print('ALL_PASS',json.dumps(dict(cases=cases,max_mid_l2=max_mid,max_out_l2=max_out)),flush=True)
finally:
    pool.shutdown();lib.fastshared_tp_destroy(state)

from fixture import *

lib.FastllmDraftFusedReduceSupported.argtypes=[P,C.c_int];lib.FastllmDraftFusedReduceSupported.restype=C.c_bool
lib.FastllmDraftFusedReduce.argtypes=[P,C.c_int,C.c_int,P,C.c_int,P]
lib.FastllmDraftFusedReduce.restype=C.c_bool
cuda=C.CDLL('libcudart.so')
cuda.cudaStreamBeginCapture.argtypes=[P,C.c_int]
cuda.cudaStreamEndCapture.argtypes=[P,C.POINTER(P)]
cuda.cudaGraphInstantiateWithFlags.argtypes=[C.POINTER(P),P,C.c_ulonglong]
cuda.cudaGraphLaunch.argtypes=[P,P]
cuda.cudaGraphDestroy.argtypes=[P];cuda.cudaGraphExecDestroy.argtypes=[P]
assert lib.FastllmDraftFusedPrepare(state,64),lib.FastllmDraftFusedError()
results=[]
for count in (8,15,16,31,32,63,64,8):
 size=count*a.rows*4096
 values=[torch.empty(size+8,device=r) for r in range(8)]
 graphs=[];execs=[]
 for r in range(8):
  torch.cuda.set_device(r);g=P();e=P()
  assert cuda.cudaStreamBeginCapture(streams[r].cuda_stream,1)==0
  assert lib.FastllmDraftFusedReduce(state,r,count,values[r].data_ptr(),size,streams[r].cuda_stream),lib.FastllmDraftFusedError()
  assert cuda.cudaStreamEndCapture(streams[r].cuda_stream,C.byref(g))==0
  assert cuda.cudaGraphInstantiateWithFlags(C.byref(e),g,0)==0
  graphs.append(g);execs.append(e)
 for replay in range(3):
  inputs=[torch.randn(size).bfloat16().float() for _ in range(8)]
  expected=torch.zeros(size)
  for source in inputs:expected+=source
  for r in range(8):
   with torch.cuda.device(r),torch.cuda.stream(streams[r]):
    values[r].fill_(-173.)
    values[r][:size].copy_(inputs[r])
    assert cuda.cudaGraphLaunch(execs[r],streams[r].cuda_stream)==0
  sync()
  for r in range(8):
   assert torch.equal(values[r][:size].cpu(),expected),('rank-ordered FP32 reduction',count,replay,r)
   assert torch.all(values[r][size:]==-173),('guard',count,replay,r)
  assert not lib.FastllmDraftFusedWatchdog(state)
  print('REDUCE_PASS',count,a.rows,replay,flush=True)
 for r in range(8):
  torch.cuda.set_device(r)
  assert not lib.FastllmDraftFusedReduce(state,r,count,values[r].data_ptr(),size-1,streams[r].cuda_stream)
  cuda.cudaGraphExecDestroy(execs[r]);cuda.cudaGraphDestroy(graphs[r])
 results.append(dict(batch=count,rows=a.rows,exact=True,replays=3))
Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
lib.FastllmDraftFusedDestroy(state)
# Growing a state must reject the wide collective before any writes when its
# original contiguous slab is too small. Model dispatch then selects NCCL.
grown=lib.FastllmDraftFusedCreate(devices,8,a.layers,flat,a.rows,a.window,1e-5,10000.)
assert grown,lib.FastllmDraftFusedError()
assert lib.FastllmDraftFusedPrepare(grown,4)
assert lib.FastllmDraftFusedPrepare(grown,64)
assert lib.FastllmDraftFusedReduceSupported(grown,4)
assert not lib.FastllmDraftFusedReduceSupported(grown,8)
assert not lib.FastllmDraftFusedReduceSupported(grown,65)
assert not lib.FastllmDraftFusedReduceSupported(None,1)
lib.FastllmDraftFusedDestroy(grown)
print('REDUCE_ALL_PASS',flush=True)

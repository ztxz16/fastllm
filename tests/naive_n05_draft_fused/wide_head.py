from fixture import *
import numpy as np

# Independent vectorized Philox4x32-10 oracle: subsequence = vocabulary ID,
# offset = draft step. This also checks that batching preserves request seeds.
def noise(seed,step):
 mask=np.uint64(0xffffffff);ids=np.arange(152576,dtype=np.uint64)
 c0=np.full_like(ids,step//4);c1=np.zeros_like(ids);c2=ids;c3=np.zeros_like(ids)
 k0=np.uint64(seed & 0xffffffff);k1=np.uint64(seed>>32)
 for _ in range(10):
  p0=c0*np.uint64(0xd2511f53);p1=c2*np.uint64(0xcd9e8d57)
  c0,c1,c2,c3=((p1>>np.uint64(32))^c1^k0)&mask,p1&mask,((p0>>np.uint64(32))^c3^k1)&mask,p0&mask
  k0=(k0+np.uint64(0x9e3779b9))&mask;k1=(k1+np.uint64(0xbb67ae85))&mask
 u=(( [c0,c1,c2,c3][step%4].astype(np.float64)+.5)*2**-32).astype(np.float32)
 u=np.minimum(u,np.nextafter(np.float32(1),np.float32(0)))
 return torch.from_numpy(-np.log(-np.log(u))).to(0)

lib.FastllmDraftFusedReduce.argtypes=[P,C.c_int,C.c_int,P,C.c_int,P];lib.FastllmDraftFusedReduce.restype=C.c_bool
lib.FastllmDraftFusedHeadPrepare.argtypes=[P,P,P,P,P,P,C.c_int];lib.FastllmDraftFusedHeadPrepare.restype=C.c_bool
lib.FastllmDraftFusedHeadProbe.argtypes=[P,C.c_int,C.c_int,P,P,P,P,P,C.c_int,P];lib.FastllmDraftFusedHeadProbe.restype=C.c_bool
lib.FastllmDraftFusedHeadSubmit.argtypes=[P,C.c_int,P,C.c_float,P,P,P,C.c_int,P];lib.FastllmDraftFusedHeadSubmit.restype=C.c_bool
lib.FastllmDraftFusedHeadSubmitBatch.argtypes=[P,C.c_int,P,P,P,P,P,P,P];lib.FastllmDraftFusedHeadSubmitBatch.restype=C.c_bool
lib.FastllmDraftFusedVerificationCreate.argtypes=[P,C.c_int,C.c_int,P,P,C.c_int,P,P,C.c_int];lib.FastllmDraftFusedVerificationCreate.restype=P
lib.FastllmDraftFusedVerificationUpload.argtypes=[P,P,C.c_int,P];lib.FastllmDraftFusedVerificationUpload.restype=C.c_bool
lib.FastllmDraftFusedVerificationExec.argtypes=[P,C.c_bool];lib.FastllmDraftFusedVerificationExec.restype=P
lib.FastllmDraftFusedVerificationDestroy.argtypes=[P]
class Request(C.Structure):
 _fields_=[('anchor',C.c_int),('position',C.c_int),('seed',C.c_uint64),('temperature',C.c_float)]
cuda=C.CDLL('libcudart.so')
cuda.cudaGraphCreate.argtypes=[C.POINTER(P),C.c_uint];cuda.cudaGraphDestroy.argtypes=[P];cuda.cudaGraphLaunch.argtypes=[P,P]
cuda.cudaEventCreateWithFlags.argtypes=[C.POINTER(P),C.c_uint];cuda.cudaEventDestroy.argtypes=[P]
assert lib.FastllmDraftFusedPrepare(state,64),lib.FastllmDraftFusedError()
with safe_open(a.draft,framework='pt',device='cpu') as f:
 w1=f.get_tensor('markov_head.markov_w1.weight').bfloat16().to(0)
 w2=f.get_tensor('markov_head.markov_w2.weight').bfloat16().to(0)
 cw=f.get_tensor('confidence_head.proj.weight').float().to(0).flatten()
 cb=f.get_tensor('confidence_head.proj.bias').float().to(0).flatten()
with safe_open(a.target_head,framework='pt',device='cpu') as f:h=f.get_tensor('lm_head.weight').bfloat16()
heads=[h[r*19072:(r+1)*19072].contiguous().to(r) for r in range(8)];del h
ptrs=(P*8)(*(x.data_ptr() for x in heads));sync()
assert lib.FastllmDraftFusedHeadPrepare(state,ptrs,w1.data_ptr(),w2.data_ptr(),cw.data_ptr(),cb.data_ptr(),0),lib.FastllmDraftFusedError()
print('PREPARED',flush=True)
results=[]
for count in [int(x) for x in (a.requests or '8,15,16,31,32,63,64').split(',')]:
 plans=[];children=[];ready=[];max_head_l2=0.;max_q_l2=0.
 xx=torch.randn(count*7,4096).bfloat16();xs=[xx.to(r) for r in range(8)]
 packed=[torch.full((3,count*8+3),-17.,device=r) for r in range(8)]
 for r in range(8):
  torch.cuda.set_device(r);event=P();child=P()
  assert cuda.cudaEventCreateWithFlags(C.byref(event),2)==0
  assert cuda.cudaGraphCreate(C.byref(child),0)==0
  plan=lib.FastllmDraftFusedVerificationCreate(state,r,count,xs[r].data_ptr(),packed[r].data_ptr(),count*8+3,event,child,8)
  assert plan,lib.FastllmDraftFusedError()
  plans.append(plan);children.append(child);ready.append(event)
 arena_guard=ReclaimedMemoryGuard()
 for replay in range(2):
  # Interleave the body collective with head replay on the same scratch/epochs.
  reductions=[torch.zeros(count*7*4096,device=r) for r in range(8)];sync()
  for r in range(8):
   torch.cuda.set_device(r)
   assert lib.FastllmDraftFusedReduce(state,r,count,reductions[r].data_ptr(),reductions[r].numel(),streams[r].cuda_stream)
  sync()
  for value in reductions:assert torch.count_nonzero(value)==0
  anchors=[(19071+i*2309+replay*991)%152576 for i in range(count)]
  seeds=[1749+i*719+replay*973 for i in range(count)]
  temps=[(.4,.7,1.,1.5)[i%4] for i in range(count)]
  requests=(Request*count)(*(Request(anchors[i],8192+replay*200+i*31,seeds[i],temps[i]) for i in range(count)))
  for r in range(8):
   torch.cuda.set_device(r)
   assert lib.FastllmDraftFusedVerificationUpload(plans[r],requests,count,streams[r].cuda_stream)
   assert cuda.cudaGraphLaunch(lib.FastllmDraftFusedVerificationExec(plans[r],True),streams[r].cuda_stream)==0
  sync();torch.cuda.set_device(0)
  assert not lib.FastllmDraftFusedWatchdog(state),"head watchdog"
  torch.cuda.set_device(0)
  tokens=torch.full((count,7),-17,dtype=torch.int64,pin_memory=True)
  scores=torch.full((count,7),-17.,pin_memory=True)
  q=torch.full((count,7,152576),-17.,device=0)
  ts=(C.c_float*count)(*temps);prefix=[7 if replay==0 else 1+i%7 for i in range(count)];ns=(C.c_int*count)(*prefix)
  assert lib.FastllmDraftFusedHeadSubmitBatch(state,count,xs[0].data_ptr(),ts,ns,q.data_ptr(),tokens.data_ptr(),scores.data_ptr(),streams[0].cuda_stream),lib.FastllmDraftFusedError()
  streams[0].synchronize()
  all_corrected=torch.empty(count,7,152576,device=0);head_l2=0.
  for r in range(8):
   torch.cuda.set_device(r)
   base=torch.empty(count,7,19072,device=r);corrected=torch.empty_like(base);ids=torch.empty(count,8,dtype=torch.int64,device=r)
   biases=torch.empty(count,7,19072,dtype=torch.bfloat16,device=r)
   for i in range(count):
    prev=(C.c_int64*7)(*([anchors[i]]+tokens[i,:6].tolist()))
    assert lib.FastllmDraftFusedHeadProbe(state,r,i,base[i].data_ptr(),corrected[i].data_ptr(),ids[i].data_ptr(),biases[i].data_ptr(),prev,7,streams[r].cuda_stream)
   streams[r].synchronize()
   ref=gemm(xs[r],heads[r]).reshape_as(base)
   m=metrics(base,ref,torch.zeros_like(ref));assert m['max_row_l2']<.006,m;head_l2=max(head_l2,m['max_row_l2'])
   assert torch.equal(corrected,(base+biases.float()).bfloat16().float()),('boundary',count,r,replay)
   expected_ids=torch.cat([torch.tensor(anchors)[:,None],tokens],-1)
   assert torch.equal(ids.cpu(),expected_ids),('rank_ids',count,r)
   all_corrected[:,:,r*19072:(r+1)*19072]=corrected.to(0)
   expected=torch.full_like(packed[r],-17.)
   expected[0,:count*8]=ids.flatten().float()
   pos=torch.tensor([requests[i].position+j for i in range(count) for j in range(8)],device=r)
   expected[1,:count*8]=pos.float();expected.view(torch.int32)[2,:count*8]=pos+1
   assert torch.equal(expected.view(torch.int32),packed[r].view(torch.int32)),('packed',count,r)
  torch.cuda.set_device(0)
  q_l2=0.
  # Every request's distribution, confidence, row guards; independently check
  # RNG law for distinct boundary/centre lanes to bound test runtime.
  for i in range(count):
   n=prefix[i];inverse=float(np.float32(1)/np.float32(temps[i]));refq=torch.softmax(all_corrected[i,:n]*inverse,-1)
   m=metrics(q[i,:n],refq,torch.zeros_like(refq));assert m['max_row_l2']<5e-6,m;q_l2=max(q_l2,m['max_row_l2'])
   # Bitwise parity with the retained exporter is mandatory. The separate
   # Torch softmax allows five ppm for its different FP32 exp/reduction order.
   legacy=torch.empty_like(q[i]);legacy_tokens=torch.empty(7,dtype=torch.int64,pin_memory=True)
   assert lib.FastllmDraftFusedHeadSubmit(state,i,xs[0][i*7:].data_ptr(),temps[i],legacy.data_ptr(),legacy_tokens.data_ptr(),None,n,streams[0].cuda_stream)
   streams[0].synchronize()
   assert torch.equal(q[i,:n],legacy[:n]),('export scalar parity',count,replay,i)
   assert torch.all(q[i,n:]==-17),('q_guard',count,i)
   prev=[anchors[i]]+tokens[i,:6].tolist()
   f=torch.cat([xs[0][i*7:i*7+n].float(),w1[prev[:n]].float()],-1)
   refc=(f@cw+cb).cpu().sigmoid();got=scores[i,:n].sigmoid()
   assert float((refc-got).norm()/refc.norm().clamp_min(1e-20))<.006
  for i in sorted(set([0,count//2,count-1])):
   for step in range(7):
    expected=int((all_corrected[i,step]*(1./temps[i])+noise(seeds[i],step)).argmax())
    assert expected==int(tokens[i,step]),('Philox_law',count,replay,i,step,expected,int(tokens[i,step]))
  assert not lib.FastllmDraftFusedWatchdog(state)
  for r in range(8):
   with torch.cuda.device(r),torch.cuda.stream(streams[r]):
    packed[r].fill_(-17)
    assert cuda.cudaGraphLaunch(lib.FastllmDraftFusedVerificationExec(plans[r],False),streams[r].cuda_stream)==0
  sync()
  for r in range(8):assert torch.all(packed[r]==-17)
  print('CASE_PASS',count,replay,'head_l2',head_l2,'q_l2',q_l2,flush=True)
  max_head_l2=max(max_head_l2,head_l2);max_q_l2=max(max_q_l2,q_l2)
 # The full-budget projection is shared by DS1/DS3; their graph prefixes
 # must stop at the requested number of Markov steps and pack the right stride.
 if count in (8,15,64):
  for short_rows in (2,4):
   short_plans=[];short_children=[];short_ready=[]
   short_packed=[torch.full((3,count*short_rows+3),-17.,device=r) for r in range(8)]
   for r in range(8):
    torch.cuda.set_device(r);event=P();child=P()
    assert cuda.cudaEventCreateWithFlags(C.byref(event),2)==0
    assert cuda.cudaGraphCreate(C.byref(child),0)==0
    plan=lib.FastllmDraftFusedVerificationCreate(state,r,count,xs[r].data_ptr(),short_packed[r].data_ptr(),count*short_rows+3,event,child,short_rows)
    assert plan,lib.FastllmDraftFusedError()
    short_plans.append(plan);short_children.append(child);short_ready.append(event)
   for r in range(8):
    torch.cuda.set_device(r)
    assert lib.FastllmDraftFusedVerificationUpload(short_plans[r],requests,count,streams[r].cuda_stream)
    assert cuda.cudaGraphLaunch(lib.FastllmDraftFusedVerificationExec(short_plans[r],True),streams[r].cuda_stream)==0
   sync()
   for r in range(8):
    expected=torch.full_like(short_packed[r],-17.)
    expected[0,:count*short_rows]=torch.cat([torch.tensor(anchors)[:,None],tokens[:,:short_rows-1]],-1).flatten().to(r).float()
    pos=torch.tensor([requests[i].position+j for i in range(count) for j in range(short_rows)],device=r)
    expected[1,:count*short_rows]=pos.float();expected.view(torch.int32)[2,:count*short_rows]=pos+1
    assert torch.equal(expected.view(torch.int32),short_packed[r].view(torch.int32)),('short prefix',count,short_rows,r)
   assert not lib.FastllmDraftFusedWatchdog(state)
   for r in range(8):
    torch.cuda.set_device(r);lib.FastllmDraftFusedVerificationDestroy(short_plans[r]);cuda.cudaGraphDestroy(short_children[r]);cuda.cudaEventDestroy(short_ready[r])
   print('SHORT_PREFIX_PASS',count,short_rows,flush=True)
 timings=[]
 for repeat in range(6 if a.bench else 0):
  begin=[torch.cuda.Event(enable_timing=True) for _ in range(8)];end=[torch.cuda.Event(enable_timing=True) for _ in range(8)]
  for r in range(8):
   with torch.cuda.device(r),torch.cuda.stream(streams[r]):
    begin[r].record()
    for _ in range(20):assert cuda.cudaGraphLaunch(lib.FastllmDraftFusedVerificationExec(plans[r],True),streams[r].cuda_stream)==0
    end[r].record()
  sync();timings.append(max(b.elapsed_time(e) for b,e in zip(begin,end))*1000/20)
 results.append(dict(batch=count,head_l2=max_head_l2,q_l2=max_q_l2,head_us=timings))
 print('TIMING',results[-1],flush=True)
 arena_guard.close()
 for r in range(8):
  torch.cuda.set_device(r);lib.FastllmDraftFusedVerificationDestroy(plans[r]);cuda.cudaGraphDestroy(children[r]);cuda.cudaEventDestroy(ready[r])
Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
lib.FastllmDraftFusedDestroy(state)
print('WIDE_HEAD_PASS',flush=True)

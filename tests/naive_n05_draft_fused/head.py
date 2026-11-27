from fixture import *

lib.FastllmDraftFusedHeadPrepare.argtypes=[P,P,P,P,P,P,C.c_int];lib.FastllmDraftFusedHeadPrepare.restype=C.c_bool
lib.FastllmDraftFusedHeadRun.argtypes=[P,C.c_int,C.c_int,P,C.c_int,C.c_uint64,C.c_float,C.c_int,P];lib.FastllmDraftFusedHeadRun.restype=C.c_bool
lib.FastllmDraftFusedHeadFinish.argtypes=[P,C.c_int,P,C.c_float,P,P,P,C.c_int,P];lib.FastllmDraftFusedHeadFinish.restype=C.c_bool
lib.FastllmDraftFusedHeadProbe.argtypes=[P,C.c_int,C.c_int,P,P,P,P,P,C.c_int,P];lib.FastllmDraftFusedHeadProbe.restype=C.c_bool
lib.FastllmDraftFusedHeadTokens.argtypes=[P,C.c_int,C.c_int];lib.FastllmDraftFusedHeadTokens.restype=P
lib.FastllmDraftFusedVerificationInput.argtypes=[P,P,C.c_int,C.c_int,C.c_int,C.c_int,P]
lib.FastllmDraftFusedVerificationInput.restype=C.c_bool
with safe_open(a.draft,framework='pt',device='cpu') as f:
 w1=f.get_tensor('markov_head.markov_w1.weight').bfloat16().to(0)
 w2=f.get_tensor('markov_head.markov_w2.weight').bfloat16().to(0)
 cw=f.get_tensor('confidence_head.proj.weight').float().to(0).flatten()
 cb=f.get_tensor('confidence_head.proj.bias').float().to(0).flatten()
with safe_open(a.target_head,framework='pt',device='cpu') as f:
 h=f.get_tensor('lm_head.weight').bfloat16()
heads=[h[r*19072:(r+1)*19072].contiguous().to(r) for r in range(8)];del h
ptrs=(P*8)(*(x.data_ptr() for x in heads));sync()
# A rejected budget must leave the same state usable for a later full preparation.
budgeted=lib.FastllmDraftFusedHeadPrepareWithBudget
budgeted.argtypes=lib.FastllmDraftFusedHeadPrepare.argtypes+[C.c_double]
budgeted.restype=C.c_bool
assert not budgeted(state,ptrs,w1.data_ptr(),w2.data_ptr(),cw.data_ptr(),cb.data_ptr(),0,1e-6)
start=time.monotonic()
assert lib.FastllmDraftFusedHeadPrepare(state,ptrs,w1.data_ptr(),w2.data_ptr(),cw.data_ptr(),cb.data_ptr(),0),lib.FastllmDraftFusedError()
print('head prepared seconds',time.monotonic()-start,flush=True)
baseouts=[torch.empty(7,19072,device=r) for r in range(8)]
corrected=[torch.empty_like(x) for x in baseouts]
ids=[torch.empty(8,dtype=torch.int64,device=r) for r in range(8)]
q=torch.empty(7,152576,device=0);tokens=(C.c_int64*7)();scores=(C.c_float*7)()
assert lib.FastllmDraftFusedPrepare(state,8),lib.FastllmDraftFusedError()
results=[];cases_data=[]
lib.FastllmDraftFusedHeadSubmit.argtypes=lib.FastllmDraftFusedHeadFinish.argtypes
lib.FastllmDraftFusedHeadSubmit.restype=C.c_bool
for case in range(8):
 lane=case;anchor=[1,19071,75000,152575][case%4];seed=1749+case;temperature=[.4,1.,.7,1.5][case%4]
 xx=torch.randn(7,4096).bfloat16();xs=[xx.to(r) for r in range(8)]
 def launch(r,lane=lane,seed=seed):
  assert lib.FastllmDraftFusedHeadRun(state,r,lane,xs[r].data_ptr(),anchor,seed,temperature,7,streams[r].cuda_stream),lib.FastllmDraftFusedError()
 sync()
 for r in range(8):
  with torch.cuda.device(r),torch.cuda.stream(streams[r]):launch(r)
 sync()
 torch.cuda.set_device(0)
 assert lib.FastllmDraftFusedHeadFinish(state,lane,xs[0].data_ptr(),temperature,q.data_ptr(),tokens,scores,7,streams[0].cuda_stream),lib.FastllmDraftFusedError()
 for r in range(8):
  with torch.cuda.device(r),torch.cuda.stream(streams[r]):
   assert lib.FastllmDraftFusedHeadProbe(state,r,lane,baseouts[r].data_ptr(),corrected[r].data_ptr(),ids[r].data_ptr(),None,None,0,streams[r].cuda_stream)
 sync()
 previous=[anchor]+list(tokens)[:-1]
 prev=(C.c_int64*7)(*previous)
 biases=[torch.empty(7,19072,dtype=torch.bfloat16,device=r) for r in range(8)]
 for r in range(8):
  torch.cuda.set_device(r)
  assert lib.FastllmDraftFusedHeadProbe(state,r,lane,None,None,None,biases[r].data_ptr(),prev,7,streams[r].cuda_stream)
 sync()
 for r in range(8):
  headref=gemm(xs[r],heads[r]);biasref=gemm(w1[previous],w2[r*19072:(r+1)*19072]).to(r)
  bm=metrics(baseouts[r],headref,torch.zeros_like(headref));mm=metrics(biases[r],biasref,torch.zeros_like(biasref))
  assert bm['max_row_l2']<=.006,('head',bm);assert mm['max_row_l2']<=.006,('markov',mm)
  # Arithmetic contract: corrected logits round after the BF16 head + BF16 bias sum.
  expected=(baseouts[r]+biases[r].float()).bfloat16().float()
  assert torch.equal(corrected[r],expected),('corrected boundary',case,r)
  assert ids[r].cpu().tolist()==[anchor]+list(tokens),('rank token disagreement',case,r)
  # Device handoff keeps the packed input ABI, integer lengths and guard rows.
  packed=torch.full((3,23),-17.,device=r);expected=packed.clone();position=8192+case*1031
  verify_rows=2+case%7
  expected[0,5:5+verify_rows]=ids[r][:verify_rows].float()
  expected[1,5:5+verify_rows]=torch.arange(position,position+verify_rows,device=r)
  expected.view(torch.int32)[2,5:5+verify_rows]=torch.arange(position+1,position+verify_rows+1,device=r,dtype=torch.int32)
  with torch.cuda.device(r),torch.cuda.stream(streams[r]):
   token_ptr=lib.FastllmDraftFusedHeadTokens(state,r,lane)
   assert token_ptr
   for invalid in (0,1,9):
    assert not lib.FastllmDraftFusedVerificationInput(token_ptr,packed.data_ptr(),23,5,position,invalid,streams[r].cuda_stream)
   assert lib.FastllmDraftFusedVerificationInput(token_ptr,packed.data_ptr(),23,5,position,verify_rows,streams[r].cuda_stream)
  streams[r].synchronize()
  assert torch.equal(packed.view(torch.int32),expected.view(torch.int32)),('verification input',case,r)
  results.append(dict(case=case,rank=r,head=bm,markov=mm))
 merged=torch.cat([x.to(0) for x in corrected],-1)
 refq=torch.softmax(merged/temperature,-1)
 qm=metrics(q,refq,torch.zeros_like(refq));assert qm['max_row_l2']<2e-6,qm
 assert torch.allclose(q.sum(-1),torch.ones(7,device=0),atol=2e-6,rtol=0)
 features=torch.cat([xs[0].float(),w1[previous].float()],-1)
 refc=(features@cw+cb).cpu();gotc=torch.tensor(list(scores))
 conf_l2=float((gotc.sigmoid()-refc.sigmoid()).norm()/refc.sigmoid().norm())
 assert conf_l2<.006,(gotc,refc,conf_l2)
 saved=list(tokens);savedq=q.clone()
 cases_data.append((xs,anchor,seed,temperature,saved,savedq,gotc.clone()))
 for r in range(8):
  with torch.cuda.device(r):launch(r)
 sync()
 torch.cuda.set_device(0)
 assert lib.FastllmDraftFusedHeadFinish(state,lane,xs[0].data_ptr(),temperature,q.data_ptr(),tokens,scores,7,streams[0].cuda_stream)
 assert list(tokens)==saved and torch.equal(q,savedq),'seed replay changed'
 print('head case',case,'head L2',max(v['head']['max_row_l2'] for v in results if v['case']==case),
       'bias L2',max(v['markov']['max_row_l2'] for v in results if v['case']==case),'q L2',qm['max_row_l2'],'confidence probability L2',conf_l2,flush=True)
 assert not lib.FastllmDraftFusedWatchdog(state)
# Async batch output must remain isolated until one owner-stream join.
for count in [1,2,4]:
 pinned_tokens=torch.empty(count,7,dtype=torch.int64,pin_memory=True)
 pinned_scores=torch.empty(count,7,dtype=torch.float32,pin_memory=True)
 batch_q=torch.empty(count,7,152576,device=0)
 for rank in range(8):
  with torch.cuda.device(rank):
   for lane in range(count):
    inputs,anchor0,seed0,temp0,_,_,_=cases_data[lane]
    assert lib.FastllmDraftFusedHeadRun(state,rank,lane,inputs[rank].data_ptr(),anchor0,seed0,temp0,7,streams[rank].cuda_stream)
 sync();torch.cuda.set_device(0)
 for lane in range(count):
  inputs,_,_,temp0,_,_,_=cases_data[lane]
  assert lib.FastllmDraftFusedHeadSubmit(state,lane,inputs[0].data_ptr(),temp0,batch_q[lane].data_ptr(),pinned_tokens[lane].data_ptr(),pinned_scores[lane].data_ptr(),7,streams[0].cuda_stream)
 streams[0].synchronize()
 for lane in range(count):
  _,_,_,_,expected_tokens,expected_q,expected_scores=cases_data[lane]
  assert pinned_tokens[lane].tolist()==expected_tokens
  assert torch.equal(batch_q[lane],expected_q)
  assert torch.equal(pinned_scores[lane],expected_scores)
 print('ASYNC_BATCH_EXACT',count,flush=True)

# Short Markov chains must preserve each full-chain prefix and leave the
# caller's unused output rows untouched. Confidence export is optional.
for steps in range(1,8):
 lane=steps-1
 inputs,anchor0,seed0,temp0,expected_tokens,expected_q,expected_scores=cases_data[lane]
 for rank in range(8):
  with torch.cuda.device(rank):
   assert lib.FastllmDraftFusedHeadRun(state,rank,lane,inputs[rank].data_ptr(),anchor0,seed0,temp0,steps,streams[rank].cuda_stream)
 sync();torch.cuda.set_device(0)
 for confidence in (False,True):
  out_q=torch.full_like(q,-17.)
  out_tokens=torch.full((7,),-17,dtype=torch.int64,pin_memory=True)
  out_scores=torch.full((7,),-17.,pin_memory=True)
  assert lib.FastllmDraftFusedHeadFinish(state,lane,inputs[0].data_ptr(),temp0,out_q.data_ptr(),
       out_tokens.data_ptr(),out_scores.data_ptr() if confidence else None,steps,streams[0].cuda_stream)
  assert out_tokens[:steps].tolist()==expected_tokens[:steps]
  assert torch.equal(out_q[:steps],expected_q[:steps])
  assert torch.all(out_tokens[steps:]==-17) and torch.all(out_q[steps:]==-17)
  if confidence:assert torch.equal(out_scores[:steps],expected_scores[:steps])
  assert torch.all(out_scores[steps if confidence else 0:]==-17)
 print('PREFIX_EXACT',steps,flush=True)

# Composite head/input graphs retain dynamic per-request parameters and publish
# a head completion event to a different consumer stream (as in the model).
class HeadRequest(C.Structure):
 _fields_=[('anchor',C.c_int),('position',C.c_int),('seed',C.c_uint64),('temperature',C.c_float)]
lib.FastllmDraftFusedVerificationCreate.argtypes=[P,C.c_int,C.c_int,P,P,C.c_int,P,P,C.c_int]
lib.FastllmDraftFusedVerificationCreate.restype=P
lib.FastllmDraftFusedVerificationUpload.argtypes=[P,P,C.c_int,P];lib.FastllmDraftFusedVerificationUpload.restype=C.c_bool
lib.FastllmDraftFusedVerificationExec.argtypes=[P,C.c_bool];lib.FastllmDraftFusedVerificationExec.restype=P
lib.FastllmDraftFusedVerificationDestroy.argtypes=[P]
cuda=C.CDLL('libcudart.so')
cuda.cudaGraphCreate.argtypes=[C.POINTER(P),C.c_uint]
cuda.cudaGraphDestroy.argtypes=[P];cuda.cudaGraphLaunch.argtypes=[P,P]
cuda.cudaEventCreateWithFlags.argtypes=[C.POINTER(P),C.c_uint]
cuda.cudaEventDestroy.argtypes=[P];cuda.cudaStreamWaitEvent.argtypes=[P,P,C.c_uint]
cuda.cudaMemsetAsync.argtypes=[P,C.c_int,C.c_size_t,P]
cuda.cudaStreamBeginCapture.argtypes=[P,C.c_int]
cuda.cudaStreamEndCapture.argtypes=[P,C.POINTER(P)]
composite_cases=0
composite_shapes=list(dict.fromkeys(
 [(n,r) for n in (1,2) for r in range(2,9)]+[(n,2+n%7) for n in range(3,9)]+
 [(n,r) for n in (4,8) for r in (2,4,8)]))
for count,verify_rows in composite_shapes:
 packed=[torch.full((3,count*verify_rows+3),-17.,device=r) for r in range(8)]
 batch_hidden=[torch.cat([cases_data[row][0][r] for row in range(count)]) for r in range(8)]
 child_hidden=[torch.cat([cases_data[row][0][r] for row in range(count)]).clone() for r in range(8)]
 child_cache=[[torch.randn(1032,128,device=r).bfloat16() for _ in range(a.layers*count*2)] for r in range(8)]
 child_tables=[torch.tensor([x.data_ptr() for x in child_cache[r]],dtype=torch.int64,device=r) for r in range(8)]
 child_lives=[torch.full((count,),8193,dtype=torch.int32,device=r) for r in range(8)]
 def child_run(rank):
  for repeat in range(5):
   assert lib.FastllmDraftFusedRun(state,rank,count,child_hidden[rank].data_ptr(),child_lives[rank].data_ptr(),child_tables[rank].data_ptr(),streams[rank].cuda_stream),lib.FastllmDraftFusedError()
 sync()
 for rank in range(8):
  torch.cuda.set_device(rank);child_run(rank)
 sync()
 ready=[];children=[];plans=[];sync()
 for rank in range(8):
  torch.cuda.set_device(rank);event=P();child=P()
  assert cuda.cudaEventCreateWithFlags(C.byref(event),2)==0
  assert cuda.cudaStreamBeginCapture(streams[rank].cuda_stream,1)==0
  child_run(rank)
  assert cuda.cudaStreamEndCapture(streams[rank].cuda_stream,C.byref(child))==0
  plan=lib.FastllmDraftFusedVerificationCreate(state,rank,count,batch_hidden[rank].data_ptr(),
       packed[rank].data_ptr(),count*verify_rows+3,event,child,verify_rows)
  assert plan,lib.FastllmDraftFusedError()
  ready.append(event);children.append(child);plans.append(plan)
 owner_stream=torch.cuda.Stream(device=0)
 # Exercise pane reuse while an independent target graph occupies the GPU.
 # A missing reader join can sporadically corrupt a whole 16-column head tile.
 for replay in range(64 if count in (4,8) else 8):
  requests=(HeadRequest*count)(*(HeadRequest(cases_data[row][1],8192+replay*200+row*31,
       cases_data[row][2],cases_data[row][3]) for row in range(count)))
  for rank in range(8):
   torch.cuda.set_device(rank)
   for row in range(count):
    assert cuda.cudaMemsetAsync(lib.FastllmDraftFusedHeadTokens(state,rank,row),0,64,streams[rank].cuda_stream)==0
   assert lib.FastllmDraftFusedVerificationUpload(plans[rank],requests,count,streams[rank].cuda_stream)
   assert cuda.cudaGraphLaunch(lib.FastllmDraftFusedVerificationExec(plans[rank],True),streams[rank].cuda_stream)==0
  torch.cuda.set_device(0)
  for event in ready:assert cuda.cudaStreamWaitEvent(owner_stream.cuda_stream,event,0)==0
  steps=verify_rows-1;confidence=replay%2==0
  out_tokens=torch.full((count,7),-17,dtype=torch.int64,pin_memory=True)
  out_scores=torch.full((count,7),-17.,pin_memory=True);out_q=torch.full((count,7,152576),-17.,device=0)
  for row in range(count):
   assert lib.FastllmDraftFusedHeadSubmit(state,row,batch_hidden[0][row*7:].data_ptr(),cases_data[row][3],
       out_q[row].data_ptr(),out_tokens[row].data_ptr(),
       out_scores[row].data_ptr() if confidence else None,steps,owner_stream.cuda_stream)
  owner_stream.synchronize()
  for row in range(count):
   assert out_tokens[row,:steps].tolist()==cases_data[row][4][:steps],('composite token',count,replay,row)
   assert torch.equal(out_q[row,:steps],cases_data[row][5][:steps]),('composite q',count,replay,row)
   if confidence:assert torch.equal(out_scores[row,:steps],cases_data[row][6][:steps]),('composite confidence',count,replay,row)
  assert torch.all(out_tokens[:,steps:]==-17) and torch.all(out_q[:,steps:]==-17)
  assert torch.all(out_scores[:,steps if confidence else 0:]==-17)
  for rank in range(8):
   expected=torch.full_like(packed[rank],-17.)
   for row in range(count):
    start,end=row*verify_rows,(row+1)*verify_rows
    expected[0,start:end]=torch.tensor(([cases_data[row][1]]+cases_data[row][4])[:verify_rows],device=rank)
    pos=requests[row].position
    expected[1,start:end]=torch.arange(pos,pos+verify_rows,device=rank)
    expected.view(torch.int32)[2,start:end]=torch.arange(pos+1,pos+verify_rows+1,device=rank,dtype=torch.int32)
   assert torch.equal(packed[rank].view(torch.int32),expected.view(torch.int32)),('composite input',count,replay,rank)
  sync()
  # Switch the same executable to ordinary target-only replay. The disabled
  # head/input prefix must leave IDs, positions and output guards untouched.
  for rank in range(8):
   with torch.cuda.device(rank),torch.cuda.stream(streams[rank]):
    packed[rank].fill_(-17)
    assert cuda.cudaGraphLaunch(lib.FastllmDraftFusedVerificationExec(plans[rank],False),streams[rank].cuda_stream)==0
  sync()
  for rank in range(8):assert torch.all(packed[rank]==-17)
  composite_cases+=1
 for rank in range(8):
  torch.cuda.set_device(rank);streams[rank].synchronize()
  lib.FastllmDraftFusedVerificationDestroy(plans[rank])
  assert cuda.cudaGraphDestroy(children[rank])==0 and cuda.cudaEventDestroy(ready[rank])==0
 print('COMPOSITE_EXACT',count,'verify_rows',verify_rows,flush=True)

# Repeat independent lanes through graph replay; changing seeds/anchors were tested above.
timings={}
for count in [1,2,4]:
 graphs=[]
 for r in range(8):
  with torch.cuda.device(r):
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g,stream=streams[r]):
    for _ in range(10):
     for lane in range(count):launch(r,lane=lane)
   graphs.append(g)
 ts=[]
 for rep in range(5):
  begin=[torch.cuda.Event(enable_timing=True) for _ in range(8)];end=[torch.cuda.Event(enable_timing=True) for _ in range(8)]
  for r in range(8):
   with torch.cuda.device(r),torch.cuda.stream(streams[r]):
    begin[r].record()
    for _ in range(10):graphs[r].replay()
    end[r].record()
  sync();ts.append(max(b.elapsed_time(e) for b,e in zip(begin,end))*1000/100)
  assert not lib.FastllmDraftFusedWatchdog(state)
 timings[count]=ts;print('head graph requests',count,'us',ts,flush=True)
 del graphs
# Time the actual verification composite entry with an empty target graph.
# The existing eager timing above intentionally remains a fallback reference.
composite_timings={}
if a.bench:
 for count,verify_rows in [(n,r) for n in (1,2,3,4,8) for r in (4,8)]:
  plans=[];children=[];ready=[]
  batch_hidden=[torch.cat([cases_data[row][0][rank] for row in range(count)]) for rank in range(8)]
  packed=[torch.empty(3,count*verify_rows,device=rank) for rank in range(8)]
  requests=(HeadRequest*count)(*(HeadRequest(cases_data[row][1],8192+row*17,cases_data[row][2],cases_data[row][3]) for row in range(count)))
  for rank in range(8):
   torch.cuda.set_device(rank);event=P();child=P()
   assert cuda.cudaEventCreateWithFlags(C.byref(event),2)==0
   assert cuda.cudaGraphCreate(C.byref(child),0)==0
   plan=lib.FastllmDraftFusedVerificationCreate(state,rank,count,batch_hidden[rank].data_ptr(),packed[rank].data_ptr(),count*verify_rows,event,child,verify_rows)
   assert plan,lib.FastllmDraftFusedError()
   assert lib.FastllmDraftFusedVerificationUpload(plan,requests,count,streams[rank].cuda_stream)
   plans.append(plan);children.append(child);ready.append(event)
  sync();ts=[]
  for rep in range(8):
   begin=[torch.cuda.Event(enable_timing=True) for _ in range(8)];end=[torch.cuda.Event(enable_timing=True) for _ in range(8)]
   for rank in range(8):
    with torch.cuda.device(rank),torch.cuda.stream(streams[rank]):
     begin[rank].record()
     for _ in range(50):assert cuda.cudaGraphLaunch(lib.FastllmDraftFusedVerificationExec(plans[rank],True),streams[rank].cuda_stream)==0
     end[rank].record()
   sync();ts.append(max(b.elapsed_time(e) for b,e in zip(begin,end))*1000/50)
   assert not lib.FastllmDraftFusedWatchdog(state)
  composite_timings[str((count,verify_rows-1))]=ts;print('head composite requests',count,'steps',verify_rows-1,'us',ts,flush=True)
  for rank in range(8):
   torch.cuda.set_device(rank);lib.FastllmDraftFusedVerificationDestroy(plans[rank]);assert cuda.cudaGraphDestroy(children[rank])==0;assert cuda.cudaEventDestroy(ready[rank])==0
Path(a.out).write_text(json.dumps(dict(results=results,timing_us=timings,composite_exact_cases=composite_cases,composite_timing_us=composite_timings),indent=2)+'\n')
lib.FastllmDraftFusedDestroy(state)
print('HEAD_PASS',flush=True)

from fixture import *

lib.FastllmDraftFusedContextPrepare.argtypes=[P,P,P,C.c_int]
lib.FastllmDraftFusedContextPrepare.restype=C.c_bool
lib.FastllmDraftFusedContextRun.argtypes=[P,C.c_int,C.c_int,P,C.c_int,C.c_int,C.c_int,P,C.c_int,C.c_int,P]
lib.FastllmDraftFusedContextRun.restype=C.c_bool
class ContextRequest(C.Structure):
    _fields_=[('features',P),('cacheTable',P)]+[(n,C.c_int) for n in
        ['rows','position','oldLength','tableRequests','tableRequest']]
lib.FastllmDraftFusedContextSharedSupported.argtypes=[P,C.c_int];lib.FastllmDraftFusedContextSharedSupported.restype=C.c_bool
lib.FastllmDraftFusedContextSharedUpload.argtypes=[P,P,C.c_int];lib.FastllmDraftFusedContextSharedUpload.restype=C.c_bool
lib.FastllmDraftFusedContextSharedRun.argtypes=[P,C.c_int,C.c_int,P];lib.FastllmDraftFusedContextSharedRun.restype=C.c_bool
lib.FastllmDraftFusedContextSharedUploadAccepted.argtypes=[P,P,C.c_int,P,P]
lib.FastllmDraftFusedContextSharedUploadAccepted.restype=C.c_bool
class NextInputs(C.Structure):
    _fields_=[('tokens',P),('stride',C.c_int),('anchors',P*8),('lengths',P*8),('seed',C.c_uint64),('temperature',C.c_float)]
lib.FastllmDraftFusedContextUploadNext.argtypes=[P,P,C.c_int,P,C.POINTER(NextInputs),P]
lib.FastllmDraftFusedContextUploadNext.restype=C.c_bool
with safe_open(a.draft,framework='pt',device='cpu') as f:
    fc=f.get_tensor('fc.weight').bfloat16().to(0)
    hn=f.get_tensor('hidden_norm.weight').float().to(0)
sync()
assert lib.FastllmDraftFusedContextPrepare(state,fc.data_ptr(),hn.data_ptr(),0), lib.FastllmDraftFusedError()
assert all(lib.FastllmDraftFusedContextSharedSupported(state,n) for n in range(1,5)),lib.FastllmDraftFusedError()
assert not lib.FastllmDraftFusedContextSharedSupported(state,prepared_capacity+1)
request_counts=[int(n) for n in a.requests.split(',')] if a.requests else [2,3,4,5,8]
assert request_counts and all(1<=n<=64 for n in request_counts)
assert lib.FastllmDraftFusedPrepare(state,max(4,max(request_counts))),lib.FastllmDraftFusedError()
arena_guard=ReclaimedMemoryGuard()
prepared_cases=0
prepared_exact_cases=0
prepared_results=[]
def prepared_matches_eager(features,caches,before,tables,ns,starts,references):
    global prepared_cases,prepared_exact_cases
    count=len(ns)
    expected=[[x.clone() for x in row] for row in caches]
    if count >= 4 and count < 8 and torch.cuda.get_device_capability(0) == (9, 0):
        for r in range(8):
            for dst,src in zip(caches[r],before[r]): dst.copy_(src)
        sync()
        for j in range(count):
            single=(ContextRequest*8)(*[ContextRequest(features[r][j].data_ptr(),tables[r].data_ptr(),
                ns[j],starts[j],min(starts[j],1023),count,j) for r in range(8)])
            assert lib.FastllmDraftFusedContextSharedUpload(state,single,1)
            for r in range(8):
                with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                    assert lib.FastllmDraftFusedContextSharedRun(state,r,1,streams[r].cuda_stream)
            sync()
        scalar=[[x.clone() for x in row] for row in caches]
    else:
        scalar=None
    stride=8
    tokens=torch.arange(32,32+count*stride,dtype=torch.int32,device=0).view(count,stride)
    anchors=[torch.empty(count+1,dtype=torch.float32,device=r) for r in range(8)]
    lengths=[torch.empty(count+1,dtype=torch.int32,device=r) for r in range(8)]
    next_inputs=NextInputs(tokens.data_ptr(),stride,(P*8)(*(x.data_ptr() for x in anchors)),
        (P*8)(*(x.data_ptr() for x in lengths)))
    for device_counts,bind_next in ((True,True),(True,False),(False,False)):
        for anchor_out,length_out in zip(anchors,lengths):
            anchor_out.fill_(-1)
            length_out.fill_(-1)
        for r in range(8):
            for dst,src in zip(caches[r],before[r]):dst.copy_(src)
        sync()
        descriptors=(ContextRequest*(8*count))(*[ContextRequest(features[r][j].data_ptr(),tables[r].data_ptr(),
            ns[j],starts[j],min(starts[j],1023),count,j) for r in range(8) for j in range(count)])
        if device_counts:
            # Delay a producer-stream count update so the consumer must honor its event.
            accepted=torch.zeros(count,dtype=torch.int32,device=0)
            source=torch.tensor([n-1 for n in ns],dtype=torch.int32,device=0)
            padded=[[torch.nn.functional.pad(x,(0,0,0,8-x.shape[0])) for x in rank] for rank in features]
            sync()
            for r in range(8):
                for j in range(count):
                    descriptors[r*count+j].rows=8
                    descriptors[r*count+j].features=padded[r][j].data_ptr()
            with torch.cuda.device(0),torch.cuda.stream(streams[0]):
                torch.cuda._sleep(100000)
                accepted.copy_(source)
                if bind_next:
                    assert lib.FastllmDraftFusedContextUploadNext(state,descriptors,count,accepted.data_ptr(),
                        C.byref(next_inputs),streams[0].cuda_stream),lib.FastllmDraftFusedError()
                else:
                    assert lib.FastllmDraftFusedContextSharedUploadAccepted(state,descriptors,count,accepted.data_ptr(),streams[0].cuda_stream),lib.FastllmDraftFusedError()
        else:
            assert lib.FastllmDraftFusedContextSharedUpload(state,descriptors,count),lib.FastllmDraftFusedError()
        for r in range(8):
            with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                assert lib.FastllmDraftFusedContextSharedRun(state,r,count,streams[r].cuda_stream),lib.FastllmDraftFusedError()
        sync()
        for anchor_out,length_out in zip(anchors,lengths):
            expected_anchors=[32+j*stride+n-1 for j,n in enumerate(ns)] if bind_next else [-1]*count
            expected_lengths=[start+n+1 for start,n in zip(starts,ns)] if bind_next else [-1]*count
            assert anchor_out.tolist()==expected_anchors+[-1]
            assert length_out.tolist()==expected_lengths+[-1]
        exact=True
        for r in range(8):
            for i,(got,want) in enumerate(zip(caches[r],expected[r])):
                request=(i//2)%count
                keep=min(starts[request],1023-ns[request]);end=keep+ns[request]
                # Parallel SSQ reduction may change rounding in newly projected KV;
                # compacted history, guard rows and padding must remain bitwise equal.
                assert torch.equal(got[:keep],want[:keep]),('shared retained',count,r,i)
                assert torch.equal(got[end:],want[end:]),('shared guard/padding',count,r,i)
                if scalar is not None:
                    assert torch.equal(got,scalar[r][i]), ('scalar prepared KV',count,r,i,device_counts)
                exact=exact and torch.equal(got,want)
                ref=references[r][i]
                m=metrics(got[keep:end],ref,torch.zeros_like(ref))
                assert m['max_row_l2']<=.006,('shared reference',count,r,i,m)
                delta=metrics(got[keep:end],want[keep:end],torch.zeros_like(ref))
                prepared_results.append(dict(device_counts=device_counts,bind_next=bind_next,batch=count,rows=ns[request],start=starts[request],
                    rank=r,layer=i//(2*count),request=request,kind=i%2,eager_row_l2=delta['max_row_l2'],**m))
        assert not lib.FastllmDraftFusedWatchdog(state)
        prepared_cases+=1
        prepared_exact_cases+=int(exact)
def rope(x,pos):
    inv=10000.**(-torch.arange(64,device=x.device).float()/64)
    ang=pos[:,None].float()*inv[None,:]
    c,s=ang.cos().bfloat16().float(),ang.sin().bfloat16().float()
    p,q=x[...,:64].float(),x[...,64:].float()
    return torch.cat([(p*c).bfloat16().float()-(q*s).bfloat16().float(),
        (q*c).bfloat16().float()+(p*s).bfloat16().float()],-1).bfloat16()

results=[]; timings=[]
for start in [0,1000,8192,65529]:
    for rows in [1,3,7,8]:
        lane=(rows-1)%4
        features=[torch.randn(rows,4096,device=r).bfloat16() for r in range(8)]
        cat=torch.cat([x.to(0) for x in features],-1)
        hidden=norm(gemm(cat,fc),hn)
        old=min(start,1023);keep=min(old,1023-rows)
        caches=[];before=[];tables=[];refs=[]
        for r in range(8):
            cache=[torch.randn(1032,128,device=r).bfloat16() for _ in range(10)]
            caches.append(cache);before.append([x.clone() for x in cache])
            tables.append(torch.tensor([x.data_ptr() for x in cache],dtype=torch.int64,device=r))
            h=hidden.to(r); pos=torch.arange(start,start+rows,device=r)
            per=[]
            for w in weights[r]:
                raw=gemm(h,w['qkv'][512:])
                per.extend([rope(norm(raw[:,:128],w['kNorm']),pos),raw[:,128:]])
            refs.append(per)
        sync()
        def launch(r):
            assert lib.FastllmDraftFusedContextRun(state,r,lane,features[r].data_ptr(),rows,start,old,
                tables[r].data_ptr(),1,0,streams[r].cuda_stream), lib.FastllmDraftFusedError()
        for r in range(8):
            with torch.cuda.device(r),torch.cuda.stream(streams[r]): launch(r)
        sync()
        vals=[]
        for r in range(8):
            for i,(cache,original,ref) in enumerate(zip(caches[r],before[r],refs[r])):
                assert torch.equal(cache[:keep],original[old-keep:old]), ('trim',start,rows,r,i)
                assert torch.equal(cache[1023:1024],original[1023:1024]), ('guard row',start,rows,r,i)
                got=cache[keep:keep+rows]
                m=metrics(got,ref,torch.zeros_like(ref)); vals.append(m['max_row_l2'])
                results.append(dict(start=start,rows=rows,lane=lane,rank=r,layer=i//2,kind='key' if i%2==0 else 'value',**m))
        assert not lib.FastllmDraftFusedWatchdog(state)
        print('context',start,rows,'max row L2',max(vals),flush=True)
        prepared_matches_eager([[x] for x in features],caches,before,tables,[rows],[start],refs)
        if start==8192 and rows==8:
            graphs=[]
            for r in range(8):
                with torch.cuda.device(r):
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g,stream=streams[r]):
                        for _ in range(10):launch(r)
                    graphs.append(g)
            for rep in range(5):
                begin=[torch.cuda.Event(enable_timing=True) for _ in range(8)]
                end=[torch.cuda.Event(enable_timing=True) for _ in range(8)]
                for r in range(8):
                    with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                        begin[r].record()
                        for _ in range(10):graphs[r].replay()
                        end[r].record()
                sync()
                timings.append(max(x.elapsed_time(y) for x,y in zip(begin,end))*1000/100)
                for r in range(8):
                    for i,ref in enumerate(refs[r]):
                        got=caches[r][i][keep:keep+rows]
                        m=metrics(got,ref,torch.zeros_like(ref))
                        assert m['max_row_l2']<=.006, ('replay',m)
                assert not lib.FastllmDraftFusedWatchdog(state)
            print('context us',timings,flush=True)
            del graphs
# Ragged requests share one dynamic pointer table, but keep distinct mailboxes,
# positions and cache allocations. Validate every retained row as well as new KV.
for count in request_counts:
    ns=[[1,3,7,8][j%4] for j in range(count)]; starts=[[17,1000,8192,65529][j%4] for j in range(count)]
    features=[[torch.randn(n,4096,device=r).bfloat16() for n in ns] for r in range(8)]
    hidden=[norm(gemm(torch.cat([features[r][j].to(0) for r in range(8)],-1),fc),hn) for j in range(count)]
    caches=[[torch.randn(1031,128,device=r).bfloat16() for _ in range(10*count)] for r in range(8)]
    before=[[x.clone() for x in row] for row in caches]
    tables=[torch.tensor([x.data_ptr() for x in row],dtype=torch.int64,device=r) for r,row in enumerate(caches)]
    sync()
    refs=[[None]*(10*count) for _ in range(8)]
    for r in range(8):
        with torch.cuda.device(r),torch.cuda.stream(streams[r]):
            for j in range(count):
                assert lib.FastllmDraftFusedContextRun(state,r,j,features[r][j].data_ptr(),ns[j],starts[j],min(starts[j],1023),
                    tables[r].data_ptr(),count,j,streams[r].cuda_stream),lib.FastllmDraftFusedError()
    sync()
    for r in range(8):
        for j in range(count):
            old=min(starts[j],1023);keep=min(old,1023-ns[j]);h=hidden[j].to(r)
            pos=torch.arange(starts[j],starts[j]+ns[j],device=r)
            for layer,w in enumerate(weights[r]):
                raw=gemm(h,w['qkv'][512:])
                for kind,ref in enumerate([rope(norm(raw[:,:128],w['kNorm']),pos),raw[:,128:]]):
                    idx=(layer*count+j)*2+kind;cache=caches[r][idx];original=before[r][idx]
                    refs[r][idx]=ref
                    assert torch.equal(cache[:keep],original[old-keep:old]),('ragged retained',count,r,j,layer,kind)
                    assert torch.equal(cache[1023:1024],original[1023:1024]),'ragged guard'
                    m=metrics(cache[keep:keep+ns[j]],ref,torch.zeros_like(ref))
                    results.append(dict(batch=count,start=starts[j],rows=ns[j],lane=j,rank=r,layer=layer,kind=kind,**m))
                    assert m['max_row_l2']<=.006,('ragged',m)
    assert not lib.FastllmDraftFusedWatchdog(state)
    print('ragged context requests',count,'PASS',flush=True)
    prepared_matches_eager(features,caches,before,tables,ns,starts,refs)
# Compare paired/four-request alias fallbacks against ordered single-request
# launches, including a feature/cache dependency in the final request.
alias_cases=0
for count,feature_alias in [(n,alias) for n in sorted(set([2,4]+[n for n in request_counts if n in (8,16,64)])) for alias in (False,True)]:
 ns=[3]*(count-1)+[1 if feature_alias else 7];old=1023 if not feature_alias else 17
 caches=[[torch.randn(1032,128,device=rank).bfloat16() for _ in range(10)] for rank in range(8)]
 before=[[x.clone() for x in row] for row in caches]
 tables=[torch.tensor([x.data_ptr() for x in row],dtype=torch.int64,device=rank) for rank,row in enumerate(caches)]
 features=[[torch.randn(ns[j],4096,device=rank).bfloat16() for j in range(count)] for rank in range(8)]
 if feature_alias:
  for rank in range(8):features[rank][-1]=caches[rank][0].flatten()[:4096].view(1,4096)
 sync()
 for request in range(count):
  descriptors=(ContextRequest*8)(*[ContextRequest(features[rank][request].data_ptr(),tables[rank].data_ptr(),ns[request],8192+request*8,old,1,0) for rank in range(8)])
  assert lib.FastllmDraftFusedContextSharedUpload(state,descriptors,1)
  for rank in range(8):
   with torch.cuda.device(rank),torch.cuda.stream(streams[rank]):assert lib.FastllmDraftFusedContextSharedRun(state,rank,1,streams[rank].cuda_stream)
  sync()
 expected=[[x.clone() for x in row] for row in caches]
 for rank in range(8):
  for dst,src in zip(caches[rank],before[rank]):dst.copy_(src)
 sync()
 descriptors=(ContextRequest*(8*count))(*[ContextRequest(features[rank][request].data_ptr(),tables[rank].data_ptr(),ns[request],8192+request*8,old,1,0) for rank in range(8) for request in range(count)])
 assert lib.FastllmDraftFusedContextSharedUpload(state,descriptors,count)
 for rank in range(8):
  with torch.cuda.device(rank),torch.cuda.stream(streams[rank]):assert lib.FastllmDraftFusedContextSharedRun(state,rank,count,streams[rank].cuda_stream)
 sync()
 for rank in range(8):
  for got,want in zip(caches[rank],expected[rank]):assert torch.equal(got,want),('alias fallback changed output',feature_alias,rank)
 assert not lib.FastllmDraftFusedWatchdog(state)
 alias_cases+=1
 print('CONTEXT_ALIAS_EXACT',count,feature_alias,flush=True)
prepared_timings={}
if a.bench:
 for count in sorted(set([1,2,3,4]+request_counts)):
  features=[[torch.randn(8,4096,device=rank).bfloat16() for _ in range(count)] for rank in range(8)]
  caches=[[torch.randn(1032,128,device=rank).bfloat16() for _ in range(10*count)] for rank in range(8)]
  tables=[torch.tensor([x.data_ptr() for x in row],dtype=torch.int64,device=rank) for rank,row in enumerate(caches)]
  descriptors=(ContextRequest*(8*count))(*[ContextRequest(features[rank][request].data_ptr(),tables[rank].data_ptr(),8,8192+request*17,1023,count,request) for rank in range(8) for request in range(count)])
  sync();assert lib.FastllmDraftFusedContextSharedUpload(state,descriptors,count)
  ts=[]
  for rep in range(8):
   begin=[torch.cuda.Event(enable_timing=True) for _ in range(8)];end=[torch.cuda.Event(enable_timing=True) for _ in range(8)]
   for rank in range(8):
    with torch.cuda.device(rank),torch.cuda.stream(streams[rank]):
     begin[rank].record()
     for _ in range(50):assert lib.FastllmDraftFusedContextSharedRun(state,rank,count,streams[rank].cuda_stream)
     end[rank].record()
   sync();ts.append(max(b.elapsed_time(e) for b,e in zip(begin,end))*1000/50)
   assert not lib.FastllmDraftFusedWatchdog(state)
  prepared_timings[count]=ts;print('context prepared requests',count,'us',ts,flush=True)
Path(a.out).write_text(json.dumps(dict(results=results,timing_us=timings,
    prepared_cases=prepared_cases,alias_exact_cases=alias_cases,prepared_timing_us=prepared_timings,prepared_eager_exact_cases=prepared_exact_cases,prepared_results=prepared_results,
    prepared_max_row_l2=max(v['max_row_l2'] for v in prepared_results),
    prepared_eager_max_row_l2=max(v['eager_row_l2'] for v in prepared_results),
    max_row_l2=max(v['max_row_l2'] for v in results)),indent=2)+'\n')
assert all(v['max_row_l2']<=.006 for v in results), 'per-row relative L2 limit 0.006'
arena_guard.close()
lib.FastllmDraftFusedDestroy(state)
print('CONTEXT_PASS',flush=True)

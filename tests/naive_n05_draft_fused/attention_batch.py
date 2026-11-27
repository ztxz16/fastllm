from fixture import *

lib.FastllmDraftFusedAttentionBatchProbe.argtypes=[P,C.c_int,C.c_int,C.c_int,P,P,P,P,P]
lib.FastllmDraftFusedAttentionBatchProbe.restype=C.c_bool
counts=[int(v) for v in a.requests.split(',')] if a.requests else [1,2,3,4,8,16]
assert counts and min(counts)>0 and max(counts)<=64
assert lib.FastllmDraftFusedPrepare(state,max(counts))
assert not lib.FastllmDraftFusedPrepare(state,65)

def rope(x,pos):
    inv=10000.**(-torch.arange(64,device=x.device).float()/64)
    ang=pos[:,None].float()*inv[None,:]
    c,s=ang.cos().bfloat16().float()[:,None,:],ang.sin().bfloat16().float()[:,None,:]
    p,q=x[...,:64].float(),x[...,64:].float()
    return torch.cat([(p*c).bfloat16().float()-(q*s).bfloat16().float(),
        (q*c).bfloat16().float()+(p*s).bfloat16().float()],-1).bfloat16()

results=[];timings=[]
for count in counts:
    initial=torch.randn(count*a.rows,4096).bfloat16()
    xs=[initial.to(r) for r in range(8)]
    guards=[torch.full((count*a.rows+2,4096),17.,dtype=torch.bfloat16,device=r) for r in range(8)]
    outs=[x[1:-1] for x in guards]
    positions=[int(v) for v in a.positions.split(',')]
    assert positions and min(positions)>0
    lengths=[positions[i%len(positions)] for i in range(count)]
    caches=[];saved=[];tables=[];lives=[]
    for r in range(8):
        cache=[torch.randn(a.window+8,128,device=r).bfloat16() for _ in range(a.layers*count*2)]
        caches.append(cache);saved.append([v.clone() for v in cache])
        tables.append(torch.tensor([v.data_ptr() for v in cache],dtype=torch.int64,device=r))
        lives.append(torch.tensor(lengths,dtype=torch.int32,device=r))
    for layer in range(a.layers):
        refs=[];kvref={}
        for request,length in enumerate(lengths):
            past=min(length-1,(a.window-1));partial=[]
            for r in range(8):
                w=weights[r][layer];x=xs[r][request*a.rows:(request+1)*a.rows]
                k,v=caches[r][(layer*count+request)*2:(layer*count+request+1)*2]
                qkv=gemm(norm(x,w['inputNorm']),w['qkv'])
                pos=torch.arange(length-1,length+a.rows-1,device=r)
                q=rope(norm(qkv[:,:512].reshape(a.rows,4,128),w['qNorm']),pos)
                kb=rope(norm(qkv[:,512:640].reshape(a.rows,1,128),w['kNorm']),pos).squeeze(1)
                vb=qkv[:,640:];kvref[r,request]=(kb,vb)
                keys=torch.cat([k[:past],kb]);values=torch.cat([v[:past],vb])
                score=torch.einsum('rhd,kd->rhk',q.float(),keys.float()).bfloat16().float()
                score=(score*(1./(128**.5))).bfloat16().float()
                cols=torch.arange(past+a.rows,device=r)[None,None,:]
                begin=torch.arange(a.rows,device=r)[:,None,None]+past-(a.window-1)
                prob=torch.softmax(score.masked_fill(cols<begin,float('-inf')),-1).bfloat16()
                attn=torch.einsum('rhk,kd->rhd',prob.float(),values.float()).bfloat16().reshape(a.rows,512)
                partial.append(gemm(attn,w['o']))
            refs.append((reduce(partial).float()+initial[request*a.rows:(request+1)*a.rows].to(0).float()).bfloat16())
        ref=torch.cat(refs);sync()
        def run(r):
            assert lib.FastllmDraftFusedAttentionBatchProbe(state,r,count,layer,xs[r].data_ptr(),outs[r].data_ptr(),lives[r].data_ptr(),tables[r].data_ptr(),streams[r].cuda_stream),lib.FastllmDraftFusedError()
        for r in range(8):
            with torch.cuda.device(r):run(r)
        sync()
        batch_kv=[[v.clone() for v in cache] for cache in caches]
        control_out=[torch.empty_like(x) for x in outs];sync()
        for rank in range(8):
            with torch.cuda.device(rank):
                for req in range(count):
                    off=req*a.rows*4096*2
                    assert lib.FastllmDraftFusedAttentionProbe(state,rank,req,layer,
                        xs[rank].data_ptr()+off,control_out[rank].data_ptr()+off,lives[rank].data_ptr()+req*4,
                        tables[rank].data_ptr(),count,streams[rank].cuda_stream)
        sync()
        for r in range(8):
            m=metrics(outs[r],ref,initial)
            control=metrics(control_out[r],ref,initial)
            regression=metrics(outs[r],control_out[r],initial)
            if count >= 4 and torch.cuda.get_device_capability(r) == (9, 0):
                assert torch.equal(outs[r],control_out[r]), ('SM90 scalar attention',count,layer,r)
                assert all(torch.equal(x,y) for x,y in zip(batch_kv[r],caches[r])), ('SM90 scalar KV',count,layer,r)
            assert regression['max_row_l2']<=.006,regression
            # Large random suites expose pre-existing BF16-boundary outliers in
            # the scalar path. Retain them in the report, and require the batch
            # path to match the scalar path and not worsen each reference row.
            z=ref.float().cpu();den=z.norm(dim=-1).clamp_min(1e-20)
            err=(outs[r].float().cpu()-z).norm(dim=-1)/den
            olderr=(control_out[r].float().cpu()-z).norm(dim=-1)/den
            assert torch.all(err<=torch.maximum(torch.full_like(olderr,.006),olderr+1.e-4)),(m,control)
            m.update(scalar_reference_max_row_l2=control['max_row_l2'],
                batch_scalar_max_row_l2=regression['max_row_l2'])
            assert torch.all(guards[r][[0,-1]]==17.)
            results.append(dict(op='attention',requests=count,layer=layer,rank=r,**m))
            for request,length in enumerate(lengths):
                past=min(length-1,(a.window-1))
                for kind in range(2):
                    index=(layer*count+request)*2+kind
                    got=batch_kv[r][index];old=saved[r][index]
                    assert torch.equal(got[:past],old[:past]),'retained KV changed'
                    assert torch.equal(got[past+a.rows:],old[past+a.rows:]),'KV guard changed'
                    m=metrics(got[past:past+a.rows],kvref[r,request][kind],torch.zeros_like(kvref[r,request][kind]))
                    assert m['max_row_l2']<=.006,m
                    results.append(dict(op='key' if kind==0 else 'value',requests=count,layer=layer,rank=r,request=request,**m))
        print('attention',count,layer,'max_row_l2',max(v['max_row_l2'] for v in results),flush=True)
        if layer==0:
            expected=[x.clone() for x in outs];sync();graphs=[]
            for r in range(8):
                with torch.cuda.device(r):
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g,stream=streams[r]):
                        for _ in range(10):run(r)
                    graphs.append(g)
            samples=[]
            for rep in range(5):
                starts=[torch.cuda.Event(enable_timing=True) for _ in range(8)]
                ends=[torch.cuda.Event(enable_timing=True) for _ in range(8)]
                for r in range(8):
                    with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                        starts[r].record()
                        for _ in range(10):graphs[r].replay()
                        ends[r].record()
                sync();samples.append(max(s.elapsed_time(e) for s,e in zip(starts,ends))*10)
                for x,y in zip(outs,expected):assert torch.equal(x,y),'graph replay mismatch'
            timings.append(dict(requests=count,samples_us=samples,median_us=sorted(samples)[2]))
            print(timings[-1],flush=True)
            # Reuse the same graph with new hidden rows and positions. Compare
            # against the retained scalar kernel on the newly committed prefix.
            for r in range(8):
                with torch.cuda.device(r):
                    xs[r].add_(.125);lives[r].add_(17)
            sync()
            for r in range(8):
                with torch.cuda.device(r):graphs[r].replay()
            sync()
            for r in range(8):
                with torch.cuda.device(r):
                    for req in range(count):
                        off=req*a.rows*4096*2
                        assert lib.FastllmDraftFusedAttentionProbe(state,r,req,layer,
                            xs[r].data_ptr()+off,control_out[r].data_ptr()+off,lives[r].data_ptr()+req*4,
                            tables[r].data_ptr(),count,streams[r].cuda_stream)
            sync()
            for r in range(8):
                if count >= 4 and torch.cuda.get_device_capability(r) == (9, 0):
                    assert torch.equal(outs[r],control_out[r]), ('SM90 scalar replay',count,r)
                m=metrics(outs[r],control_out[r],xs[r]);assert m['max_row_l2']<=.006,m
                results.append(dict(op='changing_graph',requests=count,rank=r,**m))
                # Later independent-reference cases start from their original
                # tensors and cache images rather than the graph mutation.
                xs[r].copy_(initial.to(r));lives[r].copy_(torch.tensor(lengths,device=r,dtype=torch.int32))
                for cache,original in zip(caches[r],saved[r]):cache.copy_(original)
            sync();del graphs
        assert not lib.FastllmDraftFusedWatchdog(state)
Path(a.out).write_text(json.dumps(dict(results=results,timings=timings),indent=2)+'\n')
lib.FastllmDraftFusedDestroy(state)
print('PASS',flush=True)

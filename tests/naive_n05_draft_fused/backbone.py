from fixture import *

results=[]
x0=torch.randn(a.rows,4096).bfloat16()
xs=[x0.to(r) for r in range(8)]
outs=[torch.empty_like(x) for x in xs]
for layer in range(a.layers):
    partial=[]
    for r in range(8):
        w=weights[r][layer]
        gu=gemm(norm(xs[r],w['postNorm']),w['gateUp'])
        gate,up=gu[:,:512],gu[:,512:]
        inter=(torch.nn.functional.silu(gate.float()).bfloat16().float()*up.float()).bfloat16()
        partial.append(gemm(inter,w['down']))
    ref=(reduce(partial).float()+xs[0].float()).bfloat16()
    sync()
    for r in range(8):
        with torch.cuda.device(r):
            assert lib.FastllmDraftFusedMlpProbe(state,r,0,layer,xs[r].data_ptr(),outs[r].data_ptr(),streams[r].cuda_stream)
    sync()
    for r in range(8):
        m=metrics(outs[r],ref,x0);results.append(dict(op='mlp',layer=layer,rank=r,**m))
    print('mlp',layer,'max_l2',max(v['l2'] for v in results if v['layer']==layer),flush=True)
    assert not lib.FastllmDraftFusedWatchdog(state)

def rope(x,pos):
    inv=10000.**(-torch.arange(64,device=x.device).float()/64)
    ang=pos[:,None].float()*inv[None,:]
    c,s=ang.cos().bfloat16().float()[:,None,:],ang.sin().bfloat16().float()[:,None,:]
    p,q=x[...,:64].float(),x[...,64:].float()
    return torch.cat([(p*c).bfloat16().float()-(q*s).bfloat16().float(),
        (q*c).bfloat16().float()+(p*s).bfloat16().float()],-1).bfloat16()

if not a.mlp_only:
    for length in [1,17,256,1024,8193,65529]:
        past=min(length-1,1023)
        # Distinct caches and positions per lane catch accidental shared request state.
        for layer in range(a.layers):
            lane=layer%4
            caches=[]; tables=[]; lives=[]; partial=[]; kvref=[]; saved=[]
            for r in range(8):
                w=weights[r][layer]
                k=torch.randn(1032,128,device=r).bfloat16()
                v=torch.randn(1032,128,device=r).bfloat16()
                caches.append((k,v)); saved.append((k.clone(),v.clone()))
                table=torch.tensor([k.data_ptr(),v.data_ptr()]*(a.layers*4),dtype=torch.int64,device=r)
                tables.append(table);lives.append(torch.tensor([length],dtype=torch.int32,device=r))
                qkv=gemm(norm(xs[r],w['inputNorm']),w['qkv'])
                pos=torch.arange(length-1,length+a.rows-1,device=r)
                q=rope(norm(qkv[:,:512].reshape(a.rows,4,128),w['qNorm']),pos)
                kb=rope(norm(qkv[:,512:640].reshape(a.rows,1,128),w['kNorm']),pos).squeeze(1)
                vb=qkv[:,640:]; kvref.append((kb,vb))
                keys=torch.cat([k[:past],kb]);values=torch.cat([v[:past],vb])
                score=torch.einsum('rhd,kd->rhk',q.float(),keys.float()).bfloat16().float()
                score=(score*(1./(128**.5))).bfloat16().float()
                cols=torch.arange(past+a.rows,device=r)[None,None,:]
                begin=torch.arange(a.rows,device=r)[:,None,None]+past-1023
                score=score.masked_fill(cols<begin,float('-inf'))
                prob=torch.softmax(score,-1).bfloat16()
                attn=torch.einsum('rhk,kd->rhd',prob.float(),values.float()).bfloat16().reshape(a.rows,512)
                partial.append(gemm(attn,w['o']))
            ref=(reduce(partial).float()+xs[0].float()).bfloat16()
            sync()
            for r in range(8):
                with torch.cuda.device(r):
                    assert lib.FastllmDraftFusedAttentionProbe(state,r,lane,layer,xs[r].data_ptr(),outs[r].data_ptr(),
                        lives[r].data_ptr(),tables[r].data_ptr(),4,streams[r].cuda_stream)
            sync()
            vals=[]
            for r in range(8):
                m=metrics(outs[r],ref,x0);vals.append(m['l2'])
                results.append(dict(op='attention',length=length,lane=lane,layer=layer,rank=r,**m))
                for kind,(cache,original,expected) in enumerate(zip(caches[r],saved[r],kvref[r])):
                    assert torch.equal(cache[:past],original[:past]),'attention modified retained KV'
                    assert torch.equal(cache[past+a.rows:],original[past+a.rows:]),'attention wrote padded KV'
                    km=metrics(cache[past:past+a.rows],expected,torch.zeros_like(expected))
                    results.append(dict(op='attention_key' if kind==0 else 'attention_value',length=length,lane=lane,layer=layer,rank=r,**km))
            print('attention',length,lane,'max_l2',max(vals),flush=True)
            assert not lib.FastllmDraftFusedWatchdog(state)

timings=[]
if a.bench:
    for count in [1,2,4]:
        init=[]; hidden=[]; cache=[]; tables=[]; lives=[]
        initial=torch.randn(count*a.rows,4096).bfloat16()
        for r in range(8):
            init.append(initial.to(r))
            hidden.append(torch.empty_like(init[-1]))
            cache.append([torch.randn(1032,128,device=r).bfloat16() for _ in range(a.layers*count*2)])
            tables.append(torch.tensor([x.data_ptr() for x in cache[-1]],dtype=torch.int64,device=r))
            lives.append(torch.tensor([8193,17,1024,65529][:count],dtype=torch.int32,device=r))
        sync()
        def run(r):
            hidden[r].copy_(init[r])
            assert lib.FastllmDraftFusedRun(state,r,count,hidden[r].data_ptr(),lives[r].data_ptr(),
                tables[r].data_ptr(),streams[r].cuda_stream)
        for r in range(8):
            with torch.cuda.device(r),torch.cuda.stream(streams[r]):run(r)
        sync()
        eager=[x.clone() for x in hidden]
        graphs=[]
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
            sync()
            samples.append(max(s.elapsed_time(e) for s,e in zip(starts,ends))*1000/100)
            for x,y in zip(hidden,eager):assert torch.equal(x,y), 'replay/request isolation mismatch'
            assert not lib.FastllmDraftFusedWatchdog(state)
        row=dict(requests=count,full_backbone_us=samples,median_us=sorted(samples)[2])
        timings.append(row);print('timing',row,flush=True)
        del graphs
Path(a.out).write_text(json.dumps(dict(results=results,max_l2=max(v['l2'] for v in results),timings=timings),indent=2)+'\n')
assert all(v['max_row_l2']<=.006 for v in results), 'per-row relative L2 limit 0.006'
lib.FastllmDraftFusedDestroy(state)
print('PASS',flush=True)

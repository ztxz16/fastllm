from fixture import *

# Exercise the same batched entry as serving; the retained scalar probe provides
# a separate kernel path, and FP32 matmuls provide an independent oracle.
lib.FastllmDraftFusedMlpBatchProbe.argtypes=[P,C.c_int,C.c_int,C.c_int,P,P,P]
lib.FastllmDraftFusedMlpBatchProbe.restype=C.c_bool
counts=[int(v) for v in a.requests.split(',')] if a.requests else [1,2,3,4,8,16]
assert counts and min(counts)>0 and max(counts)<=64
assert lib.FastllmDraftFusedPrepare(state,max(counts))
assert not lib.FastllmDraftFusedPrepare(state,65)
results=[];timings=[]
for count in counts:
    initial=torch.randn(count*a.rows,4096).bfloat16()
    xs=[initial.to(r) for r in range(8)]
    guarded=[torch.full((count*a.rows+2,4096),17.,dtype=torch.bfloat16,device=r) for r in range(8)]
    outs=[x[1:-1] for x in guarded]
    for layer in range(a.layers):
        partial=[]
        for r in range(8):
            w=weights[r][layer]
            gu=gemm(norm(xs[r],w['postNorm']),w['gateUp'])
            gate,up=gu[:,:512],gu[:,512:]
            inter=(torch.nn.functional.silu(gate.float()).bfloat16().float()*up.float()).bfloat16()
            partial.append(gemm(inter,w['down']))
        ref=(reduce(partial).float()+initial.to(0).float()).bfloat16()
        sync()
        def run(r):
            assert lib.FastllmDraftFusedMlpBatchProbe(state,r,count,layer,xs[r].data_ptr(),outs[r].data_ptr(),streams[r].cuda_stream),lib.FastllmDraftFusedError()
        for r in range(8):
            with torch.cuda.device(r):run(r)
        sync()
        for r in range(8):
            m=metrics(outs[r],ref,initial)
            assert m['max_row_l2']<=.006,m
            assert torch.all(guarded[r][[0,-1]]==17.)
            results.append(dict(requests=count,layer=layer,rank=r,**m))
        print('mlp',count,layer,'max_row_l2',max(x['max_row_l2'] for x in results[-8:]),flush=True)
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
                sync()
                samples.append(max(s.elapsed_time(e) for s,e in zip(starts,ends))*10)
                for x,y in zip(outs,expected):assert torch.equal(x,y),'graph replay changed output'
            timings.append(dict(requests=count,samples_us=samples,median_us=sorted(samples)[2]))
            print(timings[-1],flush=True)
            for r in range(8):xs[r].mul_(.75)
            sync()
            for r in range(8):
                with torch.cuda.device(r):graphs[r].replay()
            sync()
            controls=[torch.empty_like(v) for v in outs]
            for r in range(8):
                with torch.cuda.device(r):
                    for req in range(count):
                        off=req*a.rows*4096*2
                        assert lib.FastllmDraftFusedMlpProbe(state,r,req,layer,
                            xs[r].data_ptr()+off,controls[r].data_ptr()+off,streams[r].cuda_stream)
            sync()
            for r in range(8):
                m=metrics(outs[r],controls[r],xs[r]);assert m['max_row_l2']<=.006,m
                results.append(dict(op='changing_graph',requests=count,layer=layer,rank=r,**m))
                xs[r].copy_(initial.to(r))
            sync();del graphs
        assert not lib.FastllmDraftFusedWatchdog(state)
Path(a.out).write_text(json.dumps(dict(results=results,timings=timings),indent=2)+'\n')
lib.FastllmDraftFusedDestroy(state)
print('PASS',flush=True)

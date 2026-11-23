"""Fused body writeback/norm against retained body and independent FP32 norm."""
from fixture import *

lib.FastllmDraftFusedRunNormalized.argtypes=[P,C.c_int,C.c_int,P,P,P,P,P,P]
lib.FastllmDraftFusedRunNormalized.restype=C.c_bool
assert lib.FastllmDraftFusedPrepare(state,8)
with safe_open(a.draft,framework='pt',device='cpu') as f:
    gamma=f.get_tensor('norm.weight').float()
gammas=[gamma.to(r) for r in range(8)]
results=[]
cuda=C.CDLL('/usr/local/cuda-12.9/lib64/libcudart.so')
cuda.cudaStreamBeginCapture.argtypes=[P,C.c_int]
cuda.cudaStreamEndCapture.argtypes=[P,C.POINTER(P)]
cuda.cudaGraphInstantiateWithFlags.argtypes=[C.POINTER(P),P,C.c_uint64]
cuda.cudaGraphLaunch.argtypes=[P,P]
cuda.cudaGraphGetEdges_v2.argtypes=[P,P,P,P,C.POINTER(C.c_size_t)]
cuda.cudaGraphDestroy.argtypes=[P];cuda.cudaGraphExecDestroy.argtypes=[P]
class Edge(C.Structure):
    _fields_=[('from_port',C.c_ubyte),('to_port',C.c_ubyte),('type',C.c_ubyte),('reserved',C.c_ubyte*5)]
embedded = lib.FastllmDraftFusedRunEmbedded
embedded.argtypes = [P, C.c_int, C.c_int] + [P] * 9
embedded.restype = C.c_bool
weights = torch.randn(19, 4096).bfloat16()
mask = torch.randn(4096).bfloat16()
embeddings = [weights.to(rank) for rank in range(8)]
masks = [mask.to(rank) for rank in range(8)]
for count in [1,2,4,8]:
    initial=torch.randn(count*a.rows,4096).bfloat16()
    old=[initial.to(r) for r in range(8)];new=[v.clone() for v in old]
    normalized=[torch.full_like(v,float('nan')) for v in old]
    caches=[[torch.randn(1032,128,device=r).bfloat16() for _ in range(a.layers*count*2)] for r in range(8)]
    tables=[torch.tensor([v.data_ptr() for v in caches[r]],device=r,dtype=torch.int64) for r in range(8)]
    lives=[torch.tensor([1,17,256,1024,8193,65529,97,1023][:count],device=r,dtype=torch.int32) for r in range(8)]
    sync()
    for r in range(8):
        torch.cuda.set_device(r)
        assert lib.FastllmDraftFusedRun(state,r,count,old[r].data_ptr(),lives[r].data_ptr(),tables[r].data_ptr(),streams[r].cuda_stream)
    sync()
    # Body reads committed prefixes; speculative suffixes are overwritten by
    # each layer, so the same cache is a valid identical-input replay.
    for r in range(8):
        torch.cuda.set_device(r)
        assert lib.FastllmDraftFusedRunNormalized(state,r,count,new[r].data_ptr(),lives[r].data_ptr(),tables[r].data_ptr(),gammas[r].data_ptr(),normalized[r].data_ptr(),streams[r].cuda_stream)
    sync()
    for r in range(8):
        assert torch.equal(old[r],new[r]),('raw body changed',count,r)
        ref=norm(old[r],gammas[r]);m=metrics(normalized[r],ref,torch.zeros_like(ref))
        assert m['max_row_l2']<=.006,m
        results.append(dict(requests=count,rank=r,**m))
    saved=[v.clone() for v in normalized];graphs=[];edge_counts=[]
    for r in range(8):
        torch.cuda.set_device(r)
        init=initial.to(r);sync()
        graph=P();executable=P()
        assert cuda.cudaStreamBeginCapture(streams[r].cuda_stream,1)==0
        with torch.cuda.stream(streams[r]):
            new[r].copy_(init)
            assert lib.FastllmDraftFusedRunNormalized(state,r,count,new[r].data_ptr(),lives[r].data_ptr(),tables[r].data_ptr(),gammas[r].data_ptr(),normalized[r].data_ptr(),streams[r].cuda_stream)
        assert cuda.cudaStreamEndCapture(streams[r].cuda_stream,C.byref(graph))==0
        edges=C.c_size_t()
        assert cuda.cudaGraphGetEdges_v2(graph,None,None,None,C.byref(edges))==0
        sources=(P*edges.value)();destinations=(P*edges.value)();data=(Edge*edges.value)()
        assert cuda.cudaGraphGetEdges_v2(graph,sources,destinations,data,C.byref(edges))==0
        programmatic=sum(e.type==1 for e in data)
        # Either operation may independently fall back to the scalar tile.
        expected={0}|{a.layers*((count//2)*nodes+(count%2)*2) for nodes in (2,3,4)}
        expected|={a.layers*((count//4)*nodes+(count%4//2)*2+(count%2)*2) for nodes in (2,3,5)}
        assert programmatic in expected,(programmatic,expected)
        edge_counts.append(programmatic)
        assert cuda.cudaGraphInstantiateWithFlags(C.byref(executable),graph,0)==0
        graphs.append((graph,executable,init))
    for repeat in range(3):
        for r,(_,executable,_) in enumerate(graphs):
            torch.cuda.set_device(r)
            assert cuda.cudaGraphLaunch(executable,streams[r].cuda_stream)==0
        sync()
        for r in range(8):assert torch.equal(normalized[r],saved[r]),('replay',count,repeat,r)
    assert not lib.FastllmDraftFusedWatchdog(state)
    print('NORMALIZED_PASS',a.rows,count,max(r['max_row_l2'] for r in results if r['requests']==count),'PDL_EDGES',edge_counts,flush=True)
    for rank,(graph,executable,_) in enumerate(graphs):
        torch.cuda.set_device(rank)
        assert cuda.cudaGraphExecDestroy(executable)==0
        assert cuda.cudaGraphDestroy(graph)==0
    if count < 8:
        # Replay the same graph after replacing its token IDs. Compare both
        # outputs to the normalized body with an independently assembled input.
        ids = [torch.zeros(count, device=r) for r in range(8)]
        embedded_graphs = []
        for r in range(8):
            with torch.cuda.device(r):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=streams[r]):
                    assert embedded(state, r, count, new[r].data_ptr(), ids[r].data_ptr(),
                        embeddings[r].data_ptr(), masks[r].data_ptr(), lives[r].data_ptr(),
                        tables[r].data_ptr(), gammas[r].data_ptr(), normalized[r].data_ptr(),
                        streams[r].cuda_stream)
                embedded_graphs.append(graph)
        for offset in (1, 5):
            tokens = torch.arange(count) + offset
            inputs = mask.expand(count, a.rows, 4096).clone()
            inputs[:, 0] = weights[tokens]
            for r in range(8):
                with torch.cuda.device(r), torch.cuda.stream(streams[r]):
                    old[r].copy_(inputs.reshape(-1, 4096))
                    ids[r].copy_(tokens)
                    assert lib.FastllmDraftFusedRunNormalized(state, r, count, old[r].data_ptr(),
                        lives[r].data_ptr(), tables[r].data_ptr(), gammas[r].data_ptr(),
                        saved[r].data_ptr(), streams[r].cuda_stream)
                    embedded_graphs[r].replay()
            sync()
            for r in range(8):
                assert torch.equal(old[r], new[r]) and torch.equal(saved[r], normalized[r]), (count, r)
        del embedded_graphs
    del graphs
Path(a.out).write_text(json.dumps(results,indent=2))
lib.FastllmDraftFusedDestroy(state)
print('PASS',flush=True)

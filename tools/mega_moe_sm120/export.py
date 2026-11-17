"""Build-time exporter; output contains CUDA binaries and relocatable graph recipes.

Kernel provenance: FlashInfer SM120 split CuTeDSL MegaMoE (Apache-2.0).
No Python objects, live pointers, or executable host code are deployed.
"""
import argparse, ctypes as C, hashlib, json, struct
from pathlib import Path
import torch
from cuda.bindings import driver as cuda
from prepare import GraphProvider
from benchmark import benchmark, select_combine

def check(result):
    if int(result[0]): raise RuntimeError(str(result))
    return result[1:]

def cubins(obj):
    data = obj.dump_to_object('fastllm_export')
    pos = 0
    while True:
        start = data.find(b'\x7fELF',pos)
        if start < 0:return
        pos = start+4
        if data[start+4:start+6] != b'\x02\x01' or struct.unpack_from('<H',data,start+18)[0] != 190:continue
        shoff=struct.unpack_from('<Q',data,start+40)[0]
        entsize,count=struct.unpack_from('<HH',data,start+58)
        end=shoff+entsize*count
        for i in range(count):
            h=struct.unpack_from('<IIQQQQIIQQ',data,start+shoff+i*entsize)
            if h[1]!=8:end=max(end,h[4]+h[5])
        assert start+end <= len(data)
        yield data[start:start+end]

driver=C.CDLL('libcuda.so.1')
driver.cuGraphNodeGetDependencies_v2.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p,C.POINTER(C.c_size_t)]
driver.cuTensorMapReplaceAddress.argtypes=[C.c_void_p,C.c_void_p]

def dependencies(node):
    count=C.c_size_t()
    assert driver.cuGraphNodeGetDependencies_v2(int(node),None,None,C.byref(count))==0
    if not count.value:return []
    data=(C.c_void_p*count.value)()
    edges=(C.c_ubyte*(8*count.value))()
    status=driver.cuGraphNodeGetDependencies_v2(int(node),data,edges,C.byref(count))
    assert status==0,(status,count.value,int(node))
    return [(p,list(edges[i*8:i*8+8])) for i,p in enumerate(data)]

def kernel_args(params):
    out=[]
    for i in range(256):
        ret=cuda.cuFuncGetParamInfo(params.func,i)
        if int(ret[0]):break
        ptr=C.c_void_p.from_address(params.kernelParams+i*C.sizeof(C.c_void_p)).value
        out.append(C.string_at(ptr,ret[2]))
    return out

class Exporter:
    def __init__(self,provider,out):
        self.provider=provider;self.out=out;out.mkdir(parents=True,exist_ok=True)
        self.binaries={};self.symbols={}
    def binary(self,data):
        key=hashlib.sha256(data).hexdigest()
        (self.out/(key+'.cubin')).write_bytes(data)
        self.binaries[key]=len(data)
        return key
    def compiled(self,obj):
        keys=[self.binary(x) for x in cubins(obj)]
        for name in obj.kernel_info:
            candidates=[k for k in keys if name.encode()+b'\0' in (self.out/(k+'.cubin')).read_bytes()]
            assert len(candidates)==1,(name,keys)
            self.symbols[name]=candidates[0]
    def symbol(self,name,function):
        for obj in vars(self.provider.staging).values():
            caches=getattr(obj,'device_caches',{})
            for entry in caches.values():
                for kernel in entry[0].values():
                    if kernel.function==int(function):return self.binary(kernel.asm['cubin'])
        if name in self.symbols:return self.symbols[name]
        raise RuntimeError('No captured compiler artifact for kernel: '+name)
    def register_buffers(self,bucket):
        self.buffers=[];self.tensor_ptrs={};self.specs=[]
        for rank in range(len(self.provider.devices)):
            w=self.provider.workspaces[rank][bucket]
            f=next(iter(w._frontends.values()));e=f._compiled.execution
            x,ids,scores,_=self.provider.inputs[rank][bucket]
            weights=self.provider.weights[rank][0]
            named={'heap':self.provider.arena.heaps[rank,bucket],
                   'local':e.local_workspace,'input':x,'input_ids':ids,'input_scores':scores,
                   'w13':weights[0][0],'s13':weights[0][1],'w2':weights[1][0],'s2':weights[1][1]}
            for i,t in enumerate(e.epilogue_args):named['alpha'+str(i)]=t
            spec={}
            for name,t in named.items():
                size=t.numel()*t.element_size();ptr=t.data_ptr()
                self.buffers.append((ptr,size,rank,name))
                spec[name]={'bytes':size,'init':'float_one' if name.startswith('alpha') else 'zero'}
            self.specs.append(spec)
            # TMA descriptors can point to interior views of the heap.
            for t in [w.x,w.x_scale,w.topk_ids,w.topk_weights,w.output,e.combine_output,
                      e.shared_workspace,e.rank_combine_ready,*named.values()]:
                if t is not None:self.tensor_ptrs[t.data_ptr()]=self.pointer(t.data_ptr())
    def pointer(self,value):
        matches=[dict(rank=r,buffer=n,offset=value-p) for p,s,r,n in self.buffers if p<=value<p+s]
        assert len(matches)==1,('unknown or overlapping pointer',hex(value),matches)
        return matches[0]
    def pointer_or_none(self,value):
        for p,s,r,n in self.buffers:
            if p<=value<p+s:return dict(rank=r,buffer=n,offset=value-p)
        return None
    def mapper_locations(self,obj,runtime):
        from src.sym_buffer import SymBufferHost
        original=runtime.get('peer_rank_ptr_mapper_host')
        if original is None:return {}
        markers=tuple(0x1234000000100000+i*4096 for i in range(len(self.provider.devices)))
        changed=dict(runtime,peer_rank_ptr_mapper_host=SymBufferHost(
            base_addr=original.base_addr,offsets=markers,rank_idx=original.rank_idx,num_max_ranks=original.num_max_ranks))
        stream=torch.cuda.Stream();changed['stream']=cuda.CUstream(stream.cuda_stream)
        g=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g,stream=stream,capture_error_mode='thread_local'):
            obj.to(None)(**changed)
        graph=cuda.CUgraph(g.raw_cuda_graph())
        _,count=check(cuda.cuGraphGetNodes(graph,0));nodes,_=check(cuda.cuGraphGetNodes(graph,count))
        result={}
        for node in nodes:
            typ,=check(cuda.cuGraphNodeGetType(node))
            if typ!=cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:continue
            params,=check(cuda.cuGraphKernelNodeGetParams(node));name,=check(cuda.cuFuncGetName(params.func));name=name.decode()
            found={}
            for a,data in enumerate(kernel_args(params)):
                for off in range(0,len(data)-7,8):
                    val=struct.unpack_from('<Q',data,off)[0]
                    if val in markers:found[a,off]=markers.index(val)
            assert len(found)==len(markers),(name,found)
            result[name]=found
        return result
    def relocate(self,data,arg,name,rank):
        raw=bytearray(data);relocs=[]
        if len(data)==128:
            # Ask the CUDA API to identify the descriptor's address field;
            # never depend on its opaque binary layout.
            storage=C.create_string_buffer(256);aligned=(C.addressof(storage)+63)&~63
            found=[]
            for ptr,ref in self.tensor_ptrs.items():
                C.memmove(aligned,data,128)
                if driver.cuTensorMapReplaceAddress(aligned,ptr)==0 and C.string_at(aligned,128)==data:
                    found.append(ref)
            if found:
                assert len(found)==1,(name,arg,found)
                # Strip the captured process address from the deployed recipe.
                C.memmove(aligned,data,128)
                assert driver.cuTensorMapReplaceAddress(aligned,0x100000)==0
                raw[:]=C.string_at(aligned,128)
                return {'hex':raw.hex(),'relocations':[dict(kind='tma',arg_offset=0,**found[0])]}
        for off in range(0,len(data)-7,8):
            mapper=self.mapper.get(name,{}).get((arg,off))
            if mapper is not None:
                relocs.append(dict(kind='peer_offset',arg_offset=off,rank=mapper,local_rank=rank))
                raw[off:off+8]=b'\0'*8
                continue
            ref=self.pointer_or_none(struct.unpack_from('<Q',data,off)[0])
            if ref:
                relocs.append(dict(kind='pointer',arg_offset=off,**ref));raw[off:off+8]=b'\0'*8
        return {'hex':raw.hex(),'relocations':relocs}
    def graph(self,g,rank):
        _,count=check(cuda.cuGraphGetNodes(g,0));nodes,_=check(cuda.cuGraphGetNodes(g,count))
        index={int(n):i for i,n in enumerate(nodes)};out=[]
        for node in nodes:
            typ,=check(cuda.cuGraphNodeGetType(node))
            v={'dependencies':[dict(node=index[x],edge=e) for x,e in dependencies(node)]}
            if typ==cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH:
                child,=check(cuda.cuGraphChildGraphNodeGetGraph(node));v.update(type='graph',nodes=self.graph(child,rank))
            elif typ==cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                p,=check(cuda.cuGraphKernelNodeGetParams(node));name,=check(cuda.cuFuncGetName(p.func));name=name.decode()
                args=kernel_args(p)
                if 'FillFunctorIhE' in name or 'FillFunctorIiE' in name:
                    size=1 if 'FillFunctorIhE' in name else 4
                    assert [len(a) for a in args]==[4,size,8],(name,args)
                    v.update(type='memset',destination=self.pointer(struct.unpack('<Q',args[2])[0]),
                             value=int.from_bytes(args[1],'little'),element_size=size,width=struct.unpack('<I',args[0])[0])
                else:
                    v.update(type='kernel',name=name,binary=self.symbol(name,p.func),
                             grid=[p.gridDimX,p.gridDimY,p.gridDimZ],block=[p.blockDimX,p.blockDimY,p.blockDimZ],
                             shared=p.sharedMemBytes,green=self.green.get(name,-1),
                             args=[self.relocate(d,i,name,rank) for i,d in enumerate(args)])
            elif typ==cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EMPTY:v['type']='empty'
            else:raise RuntimeError('Unsupported graph node '+str(typ))
            out.append(v)
        return out
    def bucket(self,bucket,meta):
        self.register_buffers(bucket);ranks=[]
        for rank in range(len(self.provider.devices)):
            with torch.cuda.device(self.provider.devices[rank]):
                front=next(iter(self.provider.workspaces[rank][bucket]._frontends.values()));c=front._compiled
                self.mapper={};self.green={}
                for key,ctx in [('k1',0),('k2',1),('k2_drain',0),('k2_finalizer',-1),('k3',-1)]:
                    obj=getattr(c,key)
                    if obj is None:continue
                    self.compiled(obj)
                    for name in obj.kernel_info:self.green[name]=ctx
                    runtime=getattr(c,'runtime_'+key,None)
                    if runtime:self.mapper.update(self.mapper_locations(obj,runtime))
                entry=dict(buffers=self.specs[rank],green_sms=[c.graph.k1_sm_count,c.graph.k2_sm_count],
                           nodes=self.graph(cuda.CUgraph(int(meta[rank]['graph'],16)),rank),
                           io={name:self.pointer(int(meta[rank][source],16)) for name,source in
                               [('input','x'),('indices','indices'),('scores','scores'),('output','output')]})
                ranks.append(entry)
        return dict(capacity=bucket,ranks=ranks)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--capacities',type=int,nargs='+',default=[1,2,4,8,16,32,64,128])
    ap.add_argument('--ep',type=int,choices=[4,8],default=8)
    ap.add_argument('--hidden',type=int,default=4096)
    ap.add_argument('--intermediate',type=int,default=2048)
    ap.add_argument('--experts',type=int,default=256)
    ap.add_argument('--topk',type=int,default=8)
    ap.add_argument('--combine',choices=['auto','direct','local'],default='auto',
                    help='Build-time output-combine selection; auto benchmarks each capacity and deploys only the winner')
    args=ap.parse_args()
    h,mid=args.hidden,args.intermediate
    if h<=0 or mid<=0 or h%128 or mid%64 or args.experts%args.ep or not 0<args.topk<=args.experts:
        ap.error('Hidden must be a positive multiple of 128, intermediate of 64; experts divisible by EP; valid top-k required')
    if not args.capacities or any(c<=0 for c in args.capacities) or len(set(args.capacities))!=len(args.capacities):
        ap.error('Capacities must be unique positive integers')
    torch.set_num_threads(2);holders=[];layers=[]
    for rank in range(args.ep):
        with torch.cuda.device(rank):
            local=[]
            for n,k in ((2*mid,h),(h,mid)):
                generator=torch.Generator(device='cuda').manual_seed(1927+rank+k)
                value=torch.randint(0,256,(args.experts//args.ep,n*k//32*17),dtype=torch.uint8,device='cuda',generator=generator)
                value[:,n*k//2:].fill_(122);holders.append(value)
                local.append([value[e].data_ptr() for e in range(args.experts//args.ep)])
            layers.append([list(map(list,zip(*local)))])
    exporter=Exporter(None,args.output)
    manifest=dict(version=1,sm=120,sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
                  hidden=h,intermediate=mid,experts=args.experts,topk=args.topk,ep=args.ep,buckets=[],
                  source='FlashInfer SM120 split CuTeDSL MegaMoE, flashinfer-python 0.7.2, Apache-2.0')
    for cap in sorted(args.capacities):
        providers=[]
        def prepare(local):
            provider=GraphProvider(list(range(args.ep)),layers,hidden=h,intermediate=mid,
                                   experts=args.experts,topk=args.topk,rank_local_combine=local)
            providers.append(provider)
            return provider,provider.prepare(0,cap)
        try:
            selected,meta=prepare(args.combine=='local')
            selection={'mode':'local' if args.combine=='local' else 'direct'}
            if args.combine=='auto':
                direct=benchmark(selected,cap)
                try:
                    candidate,candidate_meta=prepare(True)
                except ValueError as error:
                    # Unsupported upstream combinations retain the direct graph.
                    selection['reason']=str(error)
                else:
                    use_local,comparison=select_combine(direct,benchmark(candidate,cap))
                    selection['benchmark']=comparison
                    if use_local:
                        selected,meta=candidate,candidate_meta
                        selection['mode']='local'
                    else:
                        selection['reason']='Requires L2 <= 0.005, >= 5% aggregate gain and <= 5% regression per workload'
            exporter.provider=selected
            bucket=exporter.bucket(cap,meta)
            bucket['selection']=selection
            manifest['buckets'].append(bucket)
            manifest['binaries']=exporter.binaries
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print('EXPORTED',cap,json.dumps(selection),'binary_count',len(exporter.binaries),flush=True)
        finally:
            for provider in reversed(providers):provider.close()
    # A regenerated package keeps only artifacts selected by this manifest.
    # Restrict cleanup to the exporter's SHA256-named CUDA binaries.
    for path in args.output.glob('*.cubin'):
        key=path.stem
        if len(key)==64 and all(c in '0123456789abcdef' for c in key) and key not in exporter.binaries:
            path.unlink()

if __name__=='__main__':main()

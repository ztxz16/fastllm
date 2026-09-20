import argparse,ctypes,json,struct
from pathlib import Path
import numpy as np
from gguf import GGUFReader,GGMLQuantizationType,GGML_QUANT_SIZES
parser=argparse.ArgumentParser(description='Generate independent llama.cpp CPU reference fixtures for direct GGUF GEMV')
parser.add_argument('model');parser.add_argument('llama_library');parser.add_argument('output_dir')
parser.add_argument('--extra-columns', type=int, nargs='*', default=[],
                    help='Additional widths, formed by repeating valid quantization blocks')
args=parser.parse_args()
work=Path(args.output_dir);work.mkdir(parents=True,exist_ok=True)
ref=ctypes.CDLL(args.llama_library)
r=GGUFReader(args.model)
names=['Q4_0','Q4_1','Q8_0','Q2_K','Q3_K','Q4_K','Q5_K','Q6_K','IQ2_XXS','IQ2_XS','IQ2_S','IQ3_XXS','IQ3_S','IQ1_S','IQ1_M','IQ4_NL','IQ4_XS']
cases=[];seen=set();rng=np.random.default_rng(91019)
def add(name,columns,data,label):
    tp=GGMLQuantizationType[name];qk,bs=GGML_QUANT_SIZES[tp]
    data=np.ascontiguousarray(data,dtype=np.uint8);rows=data.shape[0]
    out=np.empty((rows,columns),np.float32)
    func=getattr(ref,'dequantize_row_'+name.lower().replace('_k','_K'))
    func.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64];func.restype=None
    func(data.ctypes.data,out.ctypes.data,rows*columns)
    assert np.isfinite(out).all(),label
    cases.append((int(tp),rows,columns,data.tobytes(),out.tobytes(),label))
for t in r.tensors:
    name=t.tensor_type.name
    if name not in names or len(t.shape)!=2:continue
    columns,rows=map(int,t.shape);key=(name,columns)
    if key in seen:continue
    seen.add(key);qk,bs=GGML_QUANT_SIZES[t.tensor_type]
    raw=t.data.view(np.uint8).reshape(rows,columns//qk*bs)
    chosen=np.linspace(0,rows-1,min(33,rows),dtype=int)
    for width in sorted({columns,256,768}|({32,96} if qk==32 else set())):
        if width>columns:continue
        add(name,width,raw[chosen,:width//qk*bs],t.name)
for name in names:
    if any(k[0]==name for k in seen):continue
    tp=GGMLQuantizationType[name];qk,bs=GGML_QUANT_SIZES[tp]
    for columns in ([32,96,256,768,5120] if qk==32 else [256,768,5120]):
        raw=rng.integers(0,256,(33,columns//qk,bs),dtype=np.uint8)
        dpos=bs-2 if name=='Q3_K' else 0
        raw[:,:,dpos:dpos+2]=np.frombuffer(np.float16(.002).tobytes(),np.uint8)
        if name in ('Q4_1','Q5_K'):raw[:,:,2:4]=np.frombuffer(np.float16(.001).tobytes(),np.uint8)
        add(name,columns,raw.reshape(33,-1),'synthetic valid blocks')
# Decode the extended packed rows with the independent CPU implementation,
# including widths beyond the original model and incomplete CUDA input tiles.
for name in names:
    tp=GGMLQuantizationType[name];qk,bs=GGML_QUANT_SIZES[tp]
    source=next(c for c in cases if c[0]==int(tp))
    _,rows,columns,data,_,label=source
    raw=np.frombuffer(data,np.uint8).reshape(rows,-1)
    for width in sorted(set(args.extra_columns)):
        if width<=0 or width%qk:raise ValueError(f'{name}: width must be a positive multiple of {qk}')
        if any(c[0]==int(tp) and c[2]==width for c in cases):continue
        expanded=np.tile(raw,(1,(width+columns-1)//columns))[:,:width//qk*bs]
        add(name,width,expanded,label+' (repeated blocks)')
with (work/'gemv-cases.bin').open('wb') as f:
    f.write(struct.pack('<I',len(cases)))
    for tp,rows,cols,data,reference,_ in cases:
        f.write(struct.pack('<IIII',tp,rows,cols,len(data)));f.write(data);f.write(reference)
(work/'gemv-cases.json').write_text(json.dumps([dict(type=GGMLQuantizationType(x[0]).name,rows=x[1],columns=x[2],source=x[5]) for x in cases],indent=2))
print('CASES',len(cases),'types',len({x[0] for x in cases}),flush=True)

"""Native-only world acceptance capture; no tensor framework is imported."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
for key in ('library','weights','text','output'):p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--second-action',default='forward',choices=('forward','follow_prompt'))
args=p.parse_args()
path=Path(__file__).resolve().parents[2]/'tools/fastllm_pytools/nomas_server.py'
spec=importlib.util.spec_from_file_location('nomas_server',path)
server=importlib.util.module_from_spec(spec);sys.modules[spec.name]=server;spec.loader.exec_module(server)
backend=server.NativeBackend(args.library,args.weights,args.text)
report={'torch_loaded':False,'steps':[]}
try:
    for i,action in enumerate(('stay',args.second_action)):
        request=server.WorldGenerationRequest(prompt='A red ball rolls slowly across a wooden floor.',width=128,height=128,world_id='acceptance',action=action)
        start=time.monotonic()
        rgb=backend.generate(request,lambda stage,current,total:print(f'{i}: stage={stage} {current}/{total}',flush=True) or False)
        (args.output/f'step_{i}_served.rgb').write_bytes(rgb)
        report['steps'].append({'action':action,'seconds':time.monotonic()-start,'rgb_bytes':len(rgb)})
        print(json.dumps(report['steps'][-1]),flush=True)
    report['torch_loaded']='torch' in sys.modules
    assert not report['torch_loaded']
    (args.output/'native_world_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
finally:backend.close()

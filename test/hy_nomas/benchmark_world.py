"""Native-only, sequential world benchmark, including a bounded-history run."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
for key in ('library', 'weights', 'text', 'report'):
    p.add_argument('--'+key, type=Path, required=True)
p.add_argument('--sizes', default='128x128,256x160,384x224,512x288')
p.add_argument('--segments', type=int, default=4)
args = p.parse_args()
path = Path(__file__).resolve().parents[2]/'tools/fastllm_pytools/nomas_server.py'
spec = importlib.util.spec_from_file_location('nomas_server', path)
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)
report = {'precision': 'fp32', 'steps': 8, 'frames_per_segment': 20, 'cases': []}
backend = server.NativeBackend(args.library, args.weights, args.text)
try:
    for size in args.sizes.split(','):
        width, height = map(int, size.split('x'))
        for segment in range(args.segments):
            action = ('stay', 'forward', 'look_right', 'stay')[segment % 4]
            request = server.WorldGenerationRequest(
                prompt='First-person view inside a spacious wooden cabin, a table and chairs in the center, large windows overlooking a green forest, warm daylight, realistic details.',
                width=width, height=height, world_id='benchmark', action=action)
            started = time.monotonic()
            rgb = backend.generate(request, lambda *unused: False)
            elapsed = time.monotonic()-started
            assert len(rgb) == width*height*20*3
            case = dict(size=size, segment=segment+1, action=action,
                        seconds=elapsed, fps=20/elapsed, cold=segment==0)
            report['cases'].append(case)
            args.report.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(case), flush=True)
        backend.drop_world('benchmark')
    report['torch_loaded'] = 'torch' in sys.modules
    assert not report['torch_loaded']
    args.report.write_text(json.dumps(report, indent=2)+'\n')
finally:
    backend.close()

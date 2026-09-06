"""Extract small GGUF weight slices for rocmGgufMmvq.hip without Python packages."""
import argparse
import json
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--rows', type=int, default=37, help='Number of original weight rows per fixture')
args = parser.parse_args()
assert args.rows > 0, '--rows must be positive'
out = args.output
out.mkdir(parents=True, exist_ok=True)
path = args.model
f = path.open('rb')

def u64():
    return struct.unpack('<Q', f.read(8))[0]

def string():
    return f.read(u64()).decode()

def skip(kind):
    if kind == 8:
        f.seek(u64(), 1)
    elif kind == 9:
        elem = struct.unpack('<I', f.read(4))[0]
        count = u64()
        for _ in range(count):
            skip(elem)
    else:
        f.seek({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[kind], 1)

assert f.read(4) == b'GGUF'
version = struct.unpack('<I', f.read(4))[0]
nt, nk = u64(), u64()
alignment = 32
for _ in range(nk):
    key = string()
    kind = struct.unpack('<I', f.read(4))[0]
    if key == 'general.alignment':
        assert kind == 4
        alignment = struct.unpack('<I', f.read(4))[0]
    else:
        skip(kind)
tensors = []
for _ in range(nt):
    name = string()
    nd = struct.unpack('<I', f.read(4))[0]
    dims = [u64() for _ in range(nd)]
    kind = struct.unpack('<I', f.read(4))[0]
    offset = u64()
    tensors.append(dict(name=name, dims=dims, type=kind, offset=offset))
base = (f.tell() + alignment - 1) // alignment * alignment
sizes = {8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
         13: (256, 176), 14: (256, 210), 17: (256, 74), 18: (256, 98),
         20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136)}
manifest = []
for kind, (block, size) in sizes.items():
    eligible = [t for t in tensors if t['type'] == kind and len(t['dims']) == 2 and t['dims'][1] >= args.rows]
    if not eligible:
        continue
    # Test two reduction lengths where available, including long FFN rows.
    selected = {}
    for t in eligible:
        selected.setdefault(t['dims'][0], t)
    for m in sorted(selected)[::max(1, len(selected)-1)]:
        t = selected[m]
        row_bytes = m // block * size
        f.seek(base + t['offset'])
        data = f.read(row_bytes * args.rows)
        assert len(data) == row_bytes * args.rows
        file = f'type{kind}-m{m}.bin'
        (out / file).write_bytes(data)
        manifest.append(dict(type=kind, m=m, rows=args.rows, file=file, tensor=t['name']))
assert manifest, 'No supported quantized matrices found'
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
(out / 'manifest.txt').write_text(''.join(f"{t['type']} {t['m']} {t['rows']} {t['file']}\n" for t in manifest))
print('fixtures', len(manifest), 'data_offset', base)

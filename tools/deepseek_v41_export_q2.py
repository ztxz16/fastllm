#!/usr/bin/env python3
"""Export a self-contained FastLLM Q2_K/Q4_K DeepSeek-V4.1 checkpoint.

Requantizes main routed experts, keeps protected/dense weights at high
precision, copies Engram and DSpark tensors unchanged, and records the source
activation-quantization boundaries. Each completed shard can be resumed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time

DEFAULT_RULES = Path(__file__).resolve().parents[1] / 'example/quant/deepseekv41/Q2_K_MIXED.json'
EXPERT = re.compile(r'^layers\.\d+\.ffn\.experts\.\d+\.w([123])\.weight$')
ACTIVATION_NAMES = 'fastllm_activation_quantized_linears'


def read_header(path):
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 128 * 2**20:
            raise ValueError(f'Invalid safetensors header length: {path}')
        header = json.loads(stream.read(length))
    entries = {k: v for k, v in header.items() if k != '__metadata__'}
    end = max((v['data_offsets'][1] for v in entries.values()), default=0)
    if path.stat().st_size != 8 + length + end:
        raise ValueError(f'Truncated or invalid safetensors file: {path}')
    return entries, 8 + length


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def copy_tensors(source, destination, names, header, data_start):
    if set(names) == set(header):
        temporary = destination.with_suffix(destination.suffix + '.partial')
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        return
    offsets, current = {}, 0
    for name in sorted(names):
        item = dict(header[name])
        length = item['data_offsets'][1] - item['data_offsets'][0]
        item['data_offsets'] = [current, current + length]
        offsets[name] = item
        current += length
    metadata = json.dumps(offsets, separators=(',', ':')).encode()
    metadata += b' ' * (-len(metadata) % 8)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    with source.open('rb') as reader, temporary.open('wb') as writer:
        writer.write(struct.pack('<Q', len(metadata)))
        writer.write(metadata)
        for name in sorted(names):
            begin, end = header[name]['data_offsets']
            reader.seek(data_start + begin)
            remaining = end - begin
            while remaining:
                block = reader.read(min(remaining, 8 * 2**20))
                if not block:
                    raise IOError(f'Unexpected EOF copying {name}')
                writer.write(block)
                remaining -= len(block)
    temporary.replace(destination)


def verify_quantized_shard(path, source_header):
    header, data_start = read_header(path)
    expected_experts = {name for name in source_header if EXPERT.fullmatch(name)}
    if not expected_experts.issubset(header):
        raise ValueError(f'Export dropped routed experts: {sorted(expected_experts - header)[:8]}')
    counts = {'q2_k': 0, 'q4_k': 0}
    with path.open('rb') as reader:
        for name, item in header.items():
            match = EXPERT.fullmatch(name)
            if not match:
                continue
            # Standard GGML enum values: Q2_K=10 (84 B/256), Q4_K=12 (144 B/256).
            qtype, block_bytes = (12, 144) if match[1] == '2' else (10, 84)
            if item['dtype'] != 'fastllm':
                raise ValueError(f'Expert was not quantized: {name}')
            reader.seek(data_start + item['data_offsets'][0])
            version, dtype, actual_qtype = struct.unpack('<iii', reader.read(12))
            if version != 1 or dtype != 9999 or actual_qtype != qtype:
                raise ValueError(f'Wrong exported expert format: {name}: {(version, dtype, actual_qtype)}')
            rows, columns = item['shape']
            original = source_header[name]['shape']
            if [rows, columns] != [original[0], original[1] * 2]:
                raise ValueError(f'Wrong unpacked expert shape: {name}')
            expected = 12 + rows * columns // 256 * block_bytes
            if columns % 256 or item['data_offsets'][1] - item['data_offsets'][0] != expected:
                raise ValueError(f'Wrong expert payload size: {name}')
            counts['q4_k' if qtype == 12 else 'q2_k'] += 1
    return header, counts


def export(args):
    source, output = args.model.expanduser().resolve(), args.output.expanduser().resolve()
    if source == output or source in output.parents:
        raise ValueError('Output must be separate from the source checkpoint')
    config = json.loads((source / 'config.json').read_text())
    if config.get('model_type') != 'deepseek_v41':
        raise ValueError('Expected model_type=deepseek_v41')
    rules_text = args.dtype_config.read_text()
    rules = json.loads(rules_text)
    if rules != json.loads(DEFAULT_RULES.read_text()):
        raise ValueError('This exporter currently validates the supplied Q2_K/Q4_K profile only')
    source_index_path = source / 'model.safetensors.index.json'
    if source_index_path.is_file():
        source_index = json.loads(source_index_path.read_text())['weight_map']
        shards = sorted(set(source_index.values()))
    else:
        shards = ['model.safetensors']
        source_index = {k: shards[0] for k in read_header(source / shards[0])[0]}
    headers = {name: read_header(source / name) for name in shards}
    signature = {
        'source': str(source),
        'config_sha256': hashlib.sha256((source / 'config.json').read_bytes()).hexdigest(),
        'rules_sha256': hashlib.sha256(rules_text.encode()).hexdigest(),
        'shards': {name: {'bytes': (source / name).stat().st_size,
                           'mtime_ns': (source / name).stat().st_mtime_ns} for name in shards},
    }
    state_path = output / 'q2_export_state.json'
    if output.exists():
        if not args.resume or not state_path.is_file():
            raise ValueError('Output exists; use --resume only for this exporter\'s checkpoint')
        state = json.loads(state_path.read_text())
        if state['signature'] != signature:
            raise ValueError('Resume source or quantization profile changed')
    else:
        output.mkdir(parents=True)
        state = {'signature': signature, 'shards': {}, 'complete': False}
        atomic_json(state_path, state)
    from ftllm import llm
    llm.set_cpu_threads(args.threads)
    library = Path(llm.__file__).with_name('libfastllm_tools.so')
    state['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
    started = time.monotonic()
    for number, shard in enumerate(shards, 1):
        header, data_start = headers[shard]
        if shard in state['shards']:
            for filename in set(state['shards'][shard]['weight_map'].values()):
                read_header(output / filename)
            print(f'RESUME {number}/{len(shards)} {shard}', flush=True)
            continue
        print(f'EXPORT_BEGIN {number}/{len(shards)} {shard}', flush=True)
        # Exclude copied tensors explicitly, independent of inference settings
        # such as FASTLLM_DSPARK_TOKENS, including shards with mixed contents.
        convert_names = [name for name in header
                         if not (name.startswith('mtp.') or '.engram.embed.' in name)]
        counts = {'q2_k': 0, 'q4_k': 0}
        if not convert_names:
            copy_tensors(source / shard, output / shard, list(header), header, data_start)
            mapping = {name: shard for name in header}
        else:
            with tempfile.TemporaryDirectory(prefix='.q2-export-', dir=output) as work:
                staging, native_output = Path(work) / 'source', Path(work) / 'export'
                staging.mkdir()
                shutil.copyfile(source / 'config.json', staging / 'config.json')
                if len(convert_names) == len(header):
                    (staging / shard).symlink_to(source / shard)
                else:
                    # The native reader consumes the entire safetensors file,
                    # so an index alone cannot filter a mixed shard's tensors.
                    copy_tensors(source / shard, staging / shard, convert_names, header, data_start)
                atomic_json(staging / 'model.safetensors.index.json', {
                    'weight_map': {name: shard for name in convert_names}})
                llm.export_llm_model_fromhf(str(staging), str(native_output),
                                          dtype='float16', dtype_config=rules_text)
                converted, counts = verify_quantized_shard(native_output / shard, header)
                mapping = {name: shard for name in converted}
                # Converted scales are inline. Preserve unmapped primary tensors
                # and their raw scales, especially Engram in mixed tiny fixtures.
                missing = [name for name in header if name not in converted and not (
                    name.endswith('.scale') and name[:-len('scale')] + 'weight' in converted)]
                (native_output / shard).replace(output / shard)
                if missing:
                    preserved = 'preserved-' + shard
                    copy_tensors(source / shard, output / preserved, missing, header, data_start)
                    mapping.update({name: preserved for name in missing})
        state['shards'][shard] = {'weight_map': mapping, 'quantized_counts': counts}
        atomic_json(state_path, state)
        print(json.dumps({'stage': 'shard_complete', 'shard': shard, 'index': number,
                          'total_shards': len(shards), 'seconds': time.monotonic() - started,
                          'q2_k': counts['q2_k'], 'q4_k': counts['q4_k']}, ensure_ascii=False), flush=True)
    mapping = {}
    for shard in shards:
        mapping.update(state['shards'][shard]['weight_map'])
    required = {name for name in source_index if not (name.endswith('.scale') and
                name[:-len('scale')] + 'weight' in mapping and
                name not in mapping)}
    if not required.issubset(mapping):
        raise ValueError(f'Export missing source tensors: {sorted(required - mapping)[:8]}')
    source_quantized = [name for name in source_index if name.endswith('.weight') and
                        name[:-len('weight')] + 'scale' in source_index and name in mapping]
    config[ACTIVATION_NAMES] = sorted(set(config.get(ACTIVATION_NAMES, [])) | set(source_quantized))
    config['fastllm_quantization'] = {'scheme': 'Q2_K/Q4_K mixed', 'routed_gate_up': 'Q2_K',
                                    'routed_down': 'Q4_K', 'dense': 'float16',
                                    'engram': 'source FP8', 'draft': 'source unchanged'}
    for path in source.iterdir():
        if path.is_file() and path.suffix != '.safetensors' and path.name not in {
                'config.json', 'model.safetensors.index.json'}:
            shutil.copy2(path, output / path.name)
    shutil.copy2(args.dtype_config, output / 'q2_dtype_config.json')
    total_bytes = sum((output / name).stat().st_size for name in set(mapping.values()))
    atomic_json(output / 'config.json', config)
    atomic_json(output / 'model.safetensors.index.json', {
        'metadata': {'total_size': total_bytes}, 'weight_map': mapping})
    counts = {kind: sum(part['quantized_counts'][kind] for part in state['shards'].values())
              for kind in ['q2_k', 'q4_k']}
    state.update(complete=True, weight_files_bytes=total_bytes,
                 source_weight_files_bytes=sum((source / name).stat().st_size for name in shards),
                 activation_quantized_linears=len(config[ACTIVATION_NAMES]),
                 quantized_counts=counts, last_run_seconds=time.monotonic() - started)
    atomic_json(state_path, state)
    print('EXPORT_COMPLETE', json.dumps({k: v for k, v in state.items() if k not in {'signature', 'shards'}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dtype-config', type=Path, default=DEFAULT_RULES)
    parser.add_argument('--threads', type=int, default=28)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    export(args)


if __name__ == '__main__':
    main()

"""Standalone V4.1 GGUF discovery, Engram inventory and auto defaults."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from fastllm_pytools import gguf_metadata as gguf, hybrid_auto, auto_config
from fastllm_pytools.util import make_normal_parser


def write_fixture(path, metadata, name):
    def string(value):
        value = value.encode()
        return struct.pack('<Q', len(value)) + value
    def value(x):
        if isinstance(x, str): return 8, string(x)
        if isinstance(x, bool): return 7, struct.pack('<?', x)
        if isinstance(x, int): return 4, struct.pack('<I', x)
        if isinstance(x, float): return 6, struct.pack('<f', x)
        kind = value(x[0])[0]
        return 9, struct.pack('<IQ', kind, len(x)) + b''.join(value(v)[1] for v in x)
    data = b'GGUF' + struct.pack('<IQQ', 3, 1, len(metadata))
    for key, x in metadata.items():
        kind, payload = value(x)
        data += string(key) + struct.pack('<I', kind) + payload
    data += string(name) + struct.pack('<IQIQ', 1, 32, 1, 0)
    path.write_bytes(data + b'\0' * (-len(data) % 32) + b'\0' * 64)


class DeepSeekV41GGUFTest(unittest.TestCase):
    def metadata(self):
        return {'general.architecture': 'deepseek41', 'deepseek41.block_count': 40,
                'deepseek41.embedding_length': 5120, 'deepseek41.context_length': 1048576,
                'deepseek41.attention.head_count': 64, 'deepseek41.attention.key_length': 512,
                'deepseek41.attention.indexer.head_count': 32,
                'deepseek41.engram.present': True, 'deepseek41.engram.file': 'engram.gguf',
                'deepseek41.engram.source': 'source', 'deepseek41.engram.layer_ids': [1,14],
                'deepseek41.engram.num_embeddings': [384006168,384016682],
                'deepseek41.rope.scaling.type': 'yarn', 'deepseek41.rope.scaling.factor': 16.0}

    def test_metadata_and_defaults_without_ori(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'model.gguf')
            write_fixture(path, self.metadata(), 'output.weight')
            config = gguf.get_gguf_model_config(path)
            self.assertEqual(config['model_type'], 'deepseek_v41')
            self.assertEqual(config['num_hidden_layers'], 40)
            self.assertEqual(config['engram_num_embeddings'], [384006168,384016682])
            self.assertEqual(config['rope_scaling'], {'rope_type':'yarn','factor':16.0})
            args = make_normal_parser('test').parse_args(['-p',str(path),'--auto'])
            auto_config.apply_model_defaults(args, config)
            self.assertTrue(args.fast_prefill)
            self.assertEqual(args.dspark, 0)
            args = make_normal_parser('test').parse_args(['-p',str(path),'--auto'])
            auto_config.apply_model_defaults(args, self.metadata())
            self.assertEqual(args.dspark, 0)
            self.assertTrue(args.fast_prefill)

    def test_engram_single_file_and_sharded_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            aux = {'general.architecture':'deepseek41-engram','deepseek41.engram.source':'source'}
            write_fixture(root/'engram.gguf', aux, 'blk.1.engram_embd.weight')
            for count in (1,2):
                for i in range(count):
                    metadata = self.metadata()
                    metadata.update({'split.count':count,'split.no':i,'split.tensors.count':count})
                    path = root/('model.gguf' if count == 1 else f'model-{i+1:05d}-of-{count:05d}.gguf')
                    write_fixture(path, metadata, f'blk.{i}.attn_q_a.weight')
                _, tensors = hybrid_auto.inventory(path)
                self.assertEqual(len(tensors), count+1)
                self.assertTrue(hybrid_auto.is_lookup_table(tensors[-1].name))
            aux['deepseek41.engram.source'] = 'wrong'
            write_fixture(root/'engram.gguf', aux, 'blk.1.engram_embd.weight')
            with self.assertRaisesRegex(ValueError, 'mismatched'):
                hybrid_auto.inventory(path)

    def test_engram_metadata_matches_loader_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'model.gguf'
            for name in ('', '.', '..', '../engram.gguf', 'sub/engram.gguf', r'sub\engram.gguf'):
                with self.subTest(name=name):
                    metadata = self.metadata()
                    metadata['deepseek41.engram.file'] = name
                    write_fixture(path, metadata, 'output.weight')
                    with self.assertRaisesRegex(ValueError, 'adjacent'):
                        hybrid_auto.inventory(path)
            metadata = self.metadata()
            del metadata['deepseek41.engram.source']
            write_fixture(path, metadata, 'output.weight')
            write_fixture(root / 'engram.gguf', {'general.architecture': 'deepseek41-engram'},
                          'blk.1.engram_embd.weight')
            with self.assertRaisesRegex(ValueError, 'mismatched'):
                hybrid_auto.inventory(path)


if __name__ == '__main__':
    unittest.main()

"""Standalone V4.1 GGUF discovery, Engram inventory and auto defaults."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from fastllm_pytools import gguf_metadata as gguf, hybrid_auto, auto_config
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model


def write_fixture(path, metadata, name):
    def string(value):
        value = value.encode()
        return struct.pack('<Q', len(value)) + value
    def value(x):
        if isinstance(x, str): return 8, string(x)
        if isinstance(x, bool): return 7, struct.pack('<?', x)
        if isinstance(x, int): return 10, struct.pack('<Q', x)
        if isinstance(x, float): return 6, struct.pack('<f', x)
        kind = value(x[0])[0]
        return 9, struct.pack('<IQ', kind, len(x)) + b''.join(value(v)[1] for v in x)
    data = b'GGUF' + struct.pack('<IQQ', 3, int(name is not None), len(metadata))
    for key, x in metadata.items():
        kind, payload = value(x)
        data += string(key) + struct.pack('<I', kind) + payload
    if name is not None:
        data += string(name) + struct.pack('<IQIQ', 1, 32, 1, 0)
    path.write_bytes(data + b'\0' * (-len(data) % 32) + (b'\0' * 64 if name is not None else b''))


class DeepSeekV41GGUFTest(unittest.TestCase):
    def upstream_metadata(self):
        fields = {
            'block_count': 40, 'embedding_length': 5120, 'context_length': 1048576,
            'attention.head_count': 64, 'attention.key_length': 512,
            'attention.indexer.key_length': 128, 'expert_count': 384, 'expert_used_count': 6,
            'expert_feed_forward_length': 2304, 'attention.sliding_window': 128,
            'attention.compress_ratios': [0, 0] + [2] * 18 + [1] * 20,
            'attention.kv_source_layers': [2, 8, 14, 20],
            'attention.indexer.source_layers': [2, 8, 14, 20, 24, 28, 32, 36],
            'attention.indexer.candidate_source_layer': 20,
            'attention.indexer.candidate_top_k_blocks': 2048,
            'attention.indexer.candidate_block_size': 8, 'expert_gating_func': 4,
            'ple.layers': [1, 14], 'ple.ngram_size': 3, 'ple.heads_per_ngram': 2,
            'embedding_length_per_layer_input': 32, 'engram.pad_token_id': 2,
            'ple.head_vocab_sizes': [3, 5, 7, 11, 13, 17, 19, 23],
            'ple.head_offsets': [0, 3, 8, 15, 26, 39, 56, 75],
            'ple.layer_multipliers': [2**53 + 1, 3, 5, 7, 11, 13],
            'engram.token_map': [0, 1, 2],
        }
        return {'general.architecture': 'deepseek41', **{'deepseek41.' + k: v for k, v in fields.items()}}

    def test_upstream_metadata_only_first_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'model.gguf')
            metadata = self.upstream_metadata()
            write_fixture(path, metadata, None)
            config = gguf.get_gguf_model_config(path)
            self.assertEqual(config['kv_source_layer_ids'], [2, 8, 14, 20])
            self.assertEqual(config['engram_num_embeddings'], [26, 72])
            self.assertEqual(config['engram_compressed_vocab_size'], 3)
            self.assertEqual(config['scoring_func'], 'sqrtsoftplus')
            self.assertEqual(gguf.read_gguf_metadata(path, {'deepseek41.ple.layer_multipliers'})[
                'deepseek41.ple.layer_multipliers'][0], 2**53 + 1)
            model = hybrid_auto.geometry(config, 'fp4', 0)
            old = dict(metadata)
            for target, source in gguf._V41_UPSTREAM_ALIASES.items():
                old['deepseek41.' + gguf._V41_GGUF_ALIASES[target]] = old.pop('deepseek41.' + source)
            old['deepseek41.engram.num_embeddings'] = [26, 72]
            old['deepseek41.engram_compressed_vocab_size'] = 3
            legacy = hybrid_auto.geometry(gguf.deepseek_v41_config(old), 'fp4', 0)
            self.assertEqual(model.kv, legacy.kv)
            self.assertEqual(model.state, legacy.state)

    def test_upstream_shared_engram_shard_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metadata = self.upstream_metadata()
            metadata.update({'split.count': 3, 'split.no': 0, 'split.tensors.count': 2})
            first = root / 'model-00001-of-00003.gguf'
            write_fixture(first, metadata, None)
            second = root / 'model-00002-of-00003.gguf'
            write_fixture(second, {'split.count': 3, 'split.no': 1}, 'blk.1.engram_gate.weight')
            shared = root / 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'
            write_fixture(shared, {'split.count': 3, 'split.no': 2}, 'per_layer_token_embd.weight')
            for selected in (first, second):
                md, tensors = hybrid_auto.inventory(selected)
                self.assertEqual(md, metadata)
                self.assertEqual(len(tensors), 2)
            from fastllm_pytools.autotune import checkpoint_files
            self.assertEqual(checkpoint_files(first), [first, second, shared])
            args = make_normal_parser('test').parse_args(['-p', str(first), '--auto'])
            _, tensors, _ = hybrid_auto.model_inputs(args)
            self.assertEqual(len(tensors), 6)  # Four materialized integer arrays.
            sizes = hybrid_auto._deepseek_v41_weight_sizes(tensors, args)
            self.assertEqual(sizes['blk.1.engram_gate.weight'], 32 * 8)
            write_fixture(shared, {'split.count': 3, 'split.no': 1}, 'per_layer_token_embd.weight')
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                hybrid_auto.inventory(first)

    def test_upstream_rejects_bad_hash_layout_and_conflicting_aliases(self):
        cases = [('deepseek41.ple.head_offsets', [0] * 8),
                 ('deepseek41.ple.layer_multipliers', [2] * 6),
                 ('deepseek41.ple.layers', [14, 1]),
                 ('deepseek41.ple.layers', [1, 40]),
                 ('deepseek41.ple.ngram_size', 1),
                 ('deepseek41.engram.token_map', [0, 2**31]),
                 ('deepseek41.kv_source_layer_ids', [3]),
                 ('deepseek41.engram.pad_token_id', 99)]
        for key, value in cases:
            with self.subTest(key=key):
                metadata = self.upstream_metadata()
                metadata[key] = value
                with self.assertRaises(ValueError):
                    gguf.deepseek_v41_config(metadata)

    def test_ori_does_not_enable_absent_hf_draft(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps({
                'model_type': 'deepseek_v41', 'text_config': {
                    'model_type': 'deepseek_v41_text', 'num_nextn_predict_layers': 1,
                    'dspark_block_size': 5, 'dspark_target_layer_ids': [3], 'dspark_noise_token_id': 7}}))
            for metadata in (self.metadata(), self.upstream_metadata()):
                with self.subTest(upstream='deepseek41.ple.layers' in metadata):
                    model = root / 'model.gguf'
                    write_fixture(model, metadata, None)
                    args = make_normal_parser('test').parse_args([str(model), '--auto', '--ori', str(root)])
                    fake_ftllm = SimpleNamespace(llm=SimpleNamespace(get_cuda_memory_info=lambda *_: None))
                    with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {'ftllm': fake_ftllm}), \
                            patch.object(auto_config, 'resolve', side_effect=RuntimeError('stop before allocation')):
                        with self.assertRaisesRegex(RuntimeError, 'stop before allocation'):
                            make_normal_llm_model(args)
                        self.assertEqual(args.dspark, 0)
                        self.assertNotIn('FASTLLM_DSPARK_TOKENS', os.environ)

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

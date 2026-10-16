"""One-flag selection and explicit CLI precedence, without allocating weights."""
import copy
import io
import os
from pathlib import Path
import sys
import tempfile
import json
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from fastllm_pytools import auto_config as auto, hybrid_auto as memory
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model, _AutoArgumentParser

GiB = 1 << 30


class AutoConfigTest(unittest.TestCase):
    def setUp(self):
        self.g = memory.Geometry('qwen4', 4, 65536, 128, 8, 2, 64,
                                 [1024] * 4, [1 << 20] * 4, 4, 1024, 1024)
        self.costs = ([GiB] * 4, [1 << 20] * 4, 1 << 20, 1 << 20, GiB, False)
        self.gpu = {d: {'total': 8 * GiB, 'free': 8 * GiB} for d in (0, 1)}
        info = {d: {'sm': 40, 'clock_khz': 1500000, 'memory_bandwidth': 400e9,
                    'link_bandwidth': 12e9, 'capability': 75} for d in (0, 1)}
        self.hardware = info, {(a, b): True for a in info for b in info}
        self.ram = {'total': 32 * GiB, 'available': 32 * GiB}

    def resolve(self, a):
        with patch.object(memory, 'model_inputs', return_value=({}, [], self.g)), \
                patch.object(memory, 'geometry', return_value=self.g), \
                patch.object(memory, 'weight_costs', return_value=self.costs), \
                patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            return auto.resolve(a, self.gpu.__getitem__, hardware=self.hardware,
                                ram=self.ram, use_numa=True)

    def args(self, *options):
        return make_normal_parser('test').parse_args(['model.gguf', '--auto', *options])

    def test_tracking_aliases_zero_and_subcommands(self):
        parser = _AutoArgumentParser()
        sub = parser.add_subparsers(dest='command')
        sub.add_parser('server', parents=[make_normal_parser('test', add_help=False)])
        a = parser.parse_args(['server', 'model', '--auto', '--moe-cuda-cache=0', '-t4', '--moe_device_layers', '-1'])
        self.assertTrue({'auto', 'moe_cuda_cache', 'threads', 'moe_device_layers'} <= set(a._auto_explicit))
        b = parser.parse_args(['server', 'model'])
        self.assertNotIn('threads', b._auto_explicit)
        self.assertFalse(b.auto)

    def test_one_flag_is_concrete_before_loading(self):
        a = self.args()
        p = self.resolve(a)
        self.assertEqual((a.max_context_length, a.tokens, a.max_batch), (32768, 32768, 1))
        self.assertEqual(a.moe_device, 'numa')
        self.assertTrue(all(getattr(a, k) != 'auto' for k in memory.AUTO_OPTIONS))
        self.assertTrue(p['automatic']['cuda_graph'])
        self.assertGreater(a.threads, 0)
        self.assertLessEqual(p['host_estimated_bytes'], p['host_budget_bytes'])
        for d in p['gpu_budget_bytes']:
            self.assertLessEqual(p['gpu_estimated_bytes'][d], p['gpu_budget_bytes'][d])

    def test_thread_environment_matches_native_numa_precedence(self):
        for environment, expected in [
                ({'FT_THREADS': '7', 'FASTLLM_NUMA_THREADS': '4'}, 7),
                ({'FASTLLM_NUMA_THREADS': '4'}, 8),
                ({'FASTLLM_NUMA_THREADS': '4', 'FT_NUMAS': '1'}, 4),
                ({'FASTLLM_NUMA_THREADS': '4', 'FT_NUMAS': '0'}, 8)]:
            with self.subTest(environment=environment), \
                    patch.dict(os.environ, environment, clear=True), \
                    patch.object(Path, 'glob', return_value=[Path('node0'), Path('node1')]):
                self.assertEqual(auto.cpu_threads(), expected)

    def test_explicit_parameters_override_policy_even_when_equal_to_defaults(self):
        a = self.args('--device', 'cuda:1', '--tp', 'off', '--moe_cuda_cache', '0',
                      '--moe_cpu_cache', '0', '--moe_device_layers', '-1', '--ngram_device', 'cpu',
                      '--chunked_prefill_size', '256', '--threads', '3', '--mtp', '1',
                      '--atype', 'float32', '--kv_cache_dtype', 'float32',
                      '--max_context_length', '8192', '--tokens', '16384', '--max_batch', '2')
        expected = {k: copy.deepcopy(getattr(a, k)) for k in a._auto_explicit}
        p = self.resolve(a)
        for k, value in expected.items():
            self.assertEqual(getattr(a, k), value, k)
        self.assertEqual(set(p['gpu_budget_bytes']), {1})
        self.assertEqual(p['resident_layers'], 0)

    def test_full_residency_when_it_fits(self):
        p = self.resolve(self.args('--device', 'cuda:0'))
        self.assertEqual(p['resident_layers'], self.g.layers)
        self.assertEqual((p['moe_device_layers'], p['moe_cuda_cache']), (0, 0))

    def test_insufficient_memory_does_not_modify_request(self):
        a = self.args('--moe_cuda_cache', '100g')
        before = vars(a).copy()
        with self.assertRaises(ValueError):
            self.resolve(a)
        self.assertEqual(vars(a), before)

    def test_disk_experts_are_a_capacity_fallback(self):
        self.costs = ([4 * GiB] * 4, [1 << 20] * 4, 1 << 20, 0, GiB, False)
        self.gpu = {d: {'total': 3 * GiB, 'free': 3 * GiB} for d in (0, 1)}
        self.ram = {'total': 8 * GiB, 'available': 8 * GiB}
        a = self.args()
        p = self.resolve(a)
        self.assertEqual(a.moe_device, 'disk')
        self.assertLessEqual(p['host_estimated_bytes'], p['host_budget_bytes'])
        self.assertGreaterEqual(a.moe_cpu_cache, 0)

    def test_default_capacity_covers_explicit_batch_without_lowering_precision(self):
        a = self.args('--max_batch', '4', '--max_context_length', '8192', '--kv_cache_dtype', 'fp8_e4m3')
        self.resolve(a)
        self.assertEqual((a.max_batch, a.max_context_length, a.tokens, a.kv_cache_dtype), (4, 8192, 32768, 'fp8_e4m3'))

    def test_tp_off_and_heterogeneous_gpus_exclude_auto_tp(self):
        a = self.args('--tp', 'off')
        self.assertTrue(all(row[2] == 'pp' for row in auto.topologies(a, set(a._auto_explicit), self.hardware[0], self.gpu)))
        a = self.args()
        with patch.dict(os.environ, {'FASTLLM_TP': 'off'}, clear=True):
            self.assertTrue(all(row[2] == 'pp' for row in auto.topologies(a, set(a._auto_explicit), self.hardware[0], self.gpu)))
        self.hardware[0][1]['capability'] = 89
        a = self.args()
        self.assertTrue(all(row[2] == 'pp' for row in auto.topologies(a, set(a._auto_explicit), self.hardware[0], self.gpu)))

    def test_explicit_tp_and_hidden_ordinal(self):
        a = self.args('--tp', '2')
        p = self.resolve(a)
        self.assertEqual((p['mode'], set(p['gpu_budget_bytes'])), ('tp', {0, 1}))
        with self.assertRaisesRegex(ValueError, 'not visible'):
            self.resolve(self.args('--device', 'cuda:2'))

    def test_large_lookup_table_prefers_disk_unless_explicit(self):
        tensors = [memory.Tensor('per_layer_token_embd.weight', (1,), 8 * GiB, 'F16')]
        for options, expected in [((), 'disk'), (('--ngram_device', 'cpu'), 'cpu')]:
            with self.subTest(options=options), \
                    patch.object(memory, 'model_inputs', return_value=({}, tensors, self.g)), \
                    patch.object(memory, 'geometry', return_value=self.g), \
                    patch.object(memory, 'weight_costs', return_value=self.costs), \
                    patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                a = self.args(*options)
                auto.resolve(a, self.gpu.__getitem__, hardware=self.hardware,
                             ram=self.ram, use_numa=True)
                self.assertEqual(a.ngram_device, expected)

    def test_missing_cuda_driver_has_actionable_error(self):
        with patch.object(auto.ctypes, 'CDLL', side_effect=OSError('missing')):
            with self.assertRaisesRegex(ValueError, 'NVIDIA CUDA driver'):
                auto.cuda_hardware()

    def test_policy_searches_across_chunks_without_changing_capacity_only_default(self):
        a = self.args('--device', 'cuda:0', '--moe_device', 'numa')
        a.moe_device_layers, a.moe_cuda_cache, a.moe_cpu_cache, a.chunked_prefill_size = -1, 'auto', 0, 'auto'
        a.tokens, a.max_batch = 8192, 1
        p = memory.plan(a, self.g, self.costs, 'pp', {0: 1}, self.gpu, self.ram)
        q = memory.plan(a, self.g, self.costs, 'pp', {0: 1}, self.gpu, self.ram,
                        rank=lambda c: (-c['chunked_prefill_size'],))
        self.assertEqual((p['chunked_prefill_size'], q['chunked_prefill_size']), (4096, 256))

    def test_model_integration_and_saved_json_explicit_zero(self):
        self.check_saved_configuration({}, [], {}, True)

    def test_cli_zero_overrides_json_and_graph_environment_wins(self):
        self.check_saved_configuration({'moe_cuda_cache': '1g'}, ['--moe_cuda_cache', '0'],
                                       {'FASTLLM_CUDA_GRAPH': '0'}, False)

    def check_saved_configuration(self, settings, cli, environment, graph):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp'}))
            cfg = Path(d, 'serve.json')
            cfg.write_text(json.dumps({'path': d, 'auto': True, 'moe_cuda_cache': 0, **settings}))
            a = make_normal_parser('test').parse_args([str(cfg), *cli])
            runtime = MagicMock()
            model = runtime.model.return_value
            model.get_max_input_len.return_value = 32768
            model.get_max_batch.return_value = 1
            def resolve(a, query):
                self.assertIn('moe_cuda_cache', a._auto_explicit)
                self.assertEqual(a.moe_cuda_cache, 0)
                self.assertEqual(a.path, d)
                a.device, a.tp, a.moe_device = 'cuda:0', 'off', 'numa'
                a.threads, a.tokens, a.max_batch = 4, 32768, 1
                return {'automatic': {'cuda_graph': graph}}
            with patch.object(auto, 'resolve', side_effect=resolve), \
                    patch.dict(sys.modules, {'ftllm': types.SimpleNamespace(llm=runtime)}), \
                    patch.dict(os.environ, environment, clear=True), redirect_stdout(io.StringIO()):
                self.assertIs(make_normal_llm_model(a), model)
            runtime.set_cuda_graph.assert_called_once_with(graph)
            runtime.set_moe_cuda_cache.assert_called_once_with(0)
            self.assertEqual(model.hybrid_auto_plan['automatic']['cuda_graph'], graph)


if __name__ == '__main__':
    unittest.main()

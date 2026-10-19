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
        self.assertEqual((a.max_context_length, a.tokens, a.max_batch), (65536, 65536, 1))
        self.assertEqual(a.moe_device, 'numa')
        self.assertTrue(all(getattr(a, k) != 'auto' for k in memory.AUTO_OPTIONS))
        self.assertTrue(p['automatic']['cuda_graph'])
        self.assertGreater(a.threads, 0)
        self.assertLessEqual(p['host_estimated_bytes'], p['host_budget_bytes'])
        for d in p['gpu_budget_bytes']:
            self.assertLessEqual(p['gpu_estimated_bytes'][d], p['gpu_budget_bytes'][d])

    def test_optional_memory_limits_cap_auto_planning_budgets(self):
        a = self.args('--auto-gpu-memory-limit', '6g', '--auto-host-memory-limit', '20g')
        p = self.resolve(a)
        self.assertEqual((a.auto_gpu_memory_limit, a.auto_host_memory_limit), (6 * GiB, 20 * GiB))
        self.assertTrue(all(value <= 6 * GiB for value in p['gpu_budget_bytes'].values()))
        self.assertLessEqual(p['host_budget_bytes'], 20 * GiB)
        self.assertEqual(p['automatic']['gpu_memory_limit'], 6 * GiB)
        self.assertEqual(p['automatic']['host_memory_limit'], 20 * GiB)

    def test_optional_memory_limits_are_unset_by_default(self):
        a = self.args()
        self.assertIsNone(a.auto_gpu_memory_limit)
        self.assertIsNone(a.auto_host_memory_limit)

    def test_memory_limits_cannot_inflate_detected_capacity(self):
        normal = self.resolve(self.args())
        capped = self.resolve(self.args('--auto_gpu_memory_limit', '1t',
                                        '--auto_host_memory_limit', '2t'))
        for field in ('gpu_budget_bytes', 'host_budget_bytes'):
            self.assertEqual(normal[field], capped[field])

    def test_memory_limits_reject_zero_and_insufficient_budgets(self):
        for flag in ('--auto_gpu_memory_limit', '--auto_host_memory_limit'):
            for size in ('0', '1'):
                with self.subTest(flag=flag, size=size), self.assertRaises(ValueError):
                    self.resolve(self.args(flag, size))

    def test_optional_memory_limits_accept_terabyte_suffixes(self):
        a = self.args('--auto-gpu-memory-limit', '1t', '--auto-host-memory-limit', '2TiB')
        self.assertEqual(a.auto_gpu_memory_limit, 1 << 40)
        self.assertEqual(a.auto_host_memory_limit, 2 << 40)

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

    def test_dense_qwen3_uses_real_gqa_geometry_and_budgets_a_tied_output_head(self):
        metadata = dict(model_type="qwen3", num_hidden_layers=28, hidden_size=1024,
                        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                        intermediate_size=3072, max_position_embeddings=40960)
        g = memory.geometry(metadata, "float16", 0)
        self.assertEqual(sum(g.kv), 28 * 8 * 128 * 2 * 2)
        self.assertEqual(g.attention_score_heads, 16)
        embedding = memory.Tensor("model.embed_tokens.weight", (151936, 1024), 151936 * 1024 * 2, "BF16")
        tensors = [embedding, memory.Tensor("model.layers.0.mlp.up_proj.weight", (3072, 1024), GiB, "BF16")]
        a = self.args()
        with patch.object(memory, 'inventory', return_value=(metadata, tensors)), \
                patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            p = auto.resolve(a, self.gpu.__getitem__, hardware=self.hardware, ram=self.ram, use_numa=True)
            self.assertEqual(os.environ['FASTLLM_CUDA_GRAPH'], '0')
            self.assertFalse(p['automatic']['cuda_graph'])
        self.assertEqual((a.max_context_length, a.moe_device_layers, a.moe_cuda_cache, a.moe_cpu_cache),
                         (40960, 0, 0, 0))
        costs = memory.weight_costs(tensors, g, a)
        self.assertEqual(sum(costs[0]), 0)
        self.assertGreaterEqual(costs[2], 2 * embedding.bytes)
        self.assertLessEqual(p['host_estimated_bytes'], p['host_budget_bytes'])
        for d in p['gpu_budget_bytes']:
            self.assertLessEqual(p['gpu_estimated_bytes'][d], p['gpu_budget_bytes'][d])
        for mtp, dspark in ((1, 0), (0, 7)):
            with self.assertRaises(ValueError):
                memory.geometry(metadata, "float16", mtp, dspark)
        with patch.object(memory, 'inventory', return_value=(metadata, tensors)), \
                patch.dict(os.environ, {'FASTLLM_CUDA_GRAPH': '1'}, clear=True), redirect_stdout(io.StringIO()):
            p = auto.resolve(self.args(), self.gpu.__getitem__, hardware=self.hardware, ram=self.ram, use_numa=True)
            self.assertEqual(os.environ['FASTLLM_CUDA_GRAPH'], '1')
            self.assertTrue(p['automatic']['cuda_graph'])

    def test_hybrid_prefers_cache_and_keeps_per_device_headroom(self):
        self.g.experts = 256
        self.costs = ([4 * GiB] * 4, [1 << 20] * 4, 1 << 20, 0, GiB, False)
        self.gpu[1]['free'] -= GiB
        a = self.args('--device', 'cudapp=2')
        p = self.resolve(a)
        self.assertEqual(p['resident_layers'], 0)
        cache = p['moe_cuda_cache_per_device']
        self.assertEqual(cache[0] - cache[1], GiB)
        self.assertEqual(a.moe_cuda_cache, max(cache.values()))
        for d in cache:
            self.assertLessEqual(p['gpu_estimated_bytes'][d], p['gpu_budget_bytes'][d])
            self.assertGreaterEqual(self.gpu[d]['free'] - p['gpu_budget_bytes'][d],
                                    self.gpu[d]['total'] * .099)
        # Explicit cache sizes still mean the same amount on every device.
        explicit = self.resolve(self.args('--device', 'cudapp=2', '--moe_cuda_cache', '1g'))
        self.assertNotIn('moe_cuda_cache_per_device', explicit)
        tp = self.resolve(self.args('--tp', '2'))
        self.assertNotIn('moe_cuda_cache_per_device', tp)

    def test_partial_residency_remains_a_host_capacity_fallback(self):
        self.costs = ([4 * GiB] * 4, [1 << 20] * 4, 1 << 20, 0, GiB, False)
        self.ram = dict(total=16 * GiB, available=16 * GiB)
        p = self.resolve(self.args('--device', 'cudapp=2'))
        self.assertEqual(p['moe_device'], 'numa')
        self.assertGreater(p['resident_layers'], 0)
        self.assertLess(p['resident_layers'], self.g.layers)
        self.assertLessEqual(p['host_estimated_bytes'], p['host_budget_bytes'])

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

    def test_default_128k_and_explicit_total_capacity(self):
        self.g.context = 262144
        for options, expected in [((), 131072), (('--tokens', '16384'), 16384),
                                  (('--max_context_length', '65536'), 65536)]:
            with self.subTest(options=options):
                a = self.args(*options)
                p = self.resolve(a)
                self.assertEqual((a.max_context_length, a.tokens), (expected, expected))
                self.assertIsNone(a.kvmem)
                self.assertFalse(p['automatic']['kv_streaming'])

    def test_deepseek_1m_target_respects_native_window_and_explicit_capacity(self):
        self.g.family, self.g.context = 'deepseek_v41', 2097152
        self.gpu = {0: dict(total=64 * GiB, free=64 * GiB)}
        self.hardware = ({0: self.hardware[0][0]}, {(0, 0): True})
        self.ram = dict(total=128 * GiB, available=128 * GiB)
        for options, expected in [((), 1048576), (('--tokens', '65536'), 65536),
                                  (('--max_context_length', '131072'), 131072)]:
            with self.subTest(options=options):
                a = self.args(*options)
                self.resolve(a)
                self.assertEqual((a.max_context_length, a.tokens), (expected, expected))
        self.g.context = 262144
        a = self.args()
        self.resolve(a)
        self.assertEqual(a.max_context_length, 262144)

    def test_deepseek_auto_defaults_allow_explicit_opt_out_and_leave_other_models_alone(self):
        md = {'model_type': 'deepseek_v41', 'text_config': {'model_type': 'deepseek_v41_text'}}
        for options, drafts, fast in [((), 7, True), (('--dspark', '0'), 0, True),
                                     (('--speculative_algorithm', 'off'), 0, True),
                                     (('--mtp', '0'), 0, True),
                                     (('--dspark', '5', '--no-fast-prefill'), 5, False)]:
            a = self.args(*options)
            auto.apply_model_defaults(a, md)
            self.assertEqual((a.dspark, a.fast_prefill), (drafts, fast))
        for family in ('qwen4_exp', 'glm5_next'):
            a = self.args()
            auto.apply_model_defaults(a, {'model_type': family})
            self.assertEqual((a.dspark, a.fast_prefill), (0, False))
        a = make_normal_parser('test').parse_args(['model'])
        auto.apply_model_defaults(a, md)
        self.assertEqual((a.dspark, a.fast_prefill), (0, False))

    def test_deepseek_auto_defaults_reach_planner_and_runtime_with_cli_json_precedence(self):
        cases = [({}, [], 7, True, .5),
                 ({}, ['--dspark', '5', '--speculative_dspark_confidence_threshold', '.2'], 5, True, .2),
                 ({}, ['--speculative_algorithm', 'off', '--no_fast_prefill'], 0, False, .5),
                 ({'dspark': 0, 'fast_prefill': False}, [], 0, False, .5),
                 ({'dspark': 0, 'fast_prefill': False}, ['--dspark', '7', '--fast_prefill'], 7, True, .5)]
        for saved, cli, drafts, fast, threshold in cases:
            with self.subTest(saved=saved, cli=cli), tempfile.TemporaryDirectory() as d:
                Path(d, 'config.json').write_text(json.dumps({'model_type': 'deepseek_v41',
                    'text_config': {'model_type': 'deepseek_v41_text', 'dspark_block_size': 5,
                                    'dspark_target_layer_ids': [2, 3], 'dspark_noise_token_id': 100}}))
                cfg = Path(d, 'serve.json')
                cfg.write_text(json.dumps({'path': d, 'auto': True, **saved}))
                a = make_normal_parser('test').parse_args([str(cfg), *cli])
                runtime = MagicMock()
                model = runtime.model.return_value
                model.get_max_input_len.return_value = 1048576
                model.get_max_batch.return_value = 1
                def resolve(a, query):
                    self.assertEqual((a.dspark, a.fast_prefill), (drafts, fast))
                    self.assertEqual(os.environ.get('FASTLLM_DSPARK_TOKENS'), str(drafts) if drafts else None)
                    if drafts:
                        self.assertEqual(float(os.environ['FASTLLM_DSPARK_CONFIDENCE_THRESHOLD']), threshold)
                    a.device, a.tp, a.moe_device = 'cuda:0', 'off', 'numa'
                    a.threads, a.tokens, a.max_context_length, a.max_batch = 64, 1048576, 1048576, 1
                    a.chunked_prefill_size, a.moe_cuda_cache = 4096, GiB
                    return {'automatic': {'cuda_graph': True}}
                def load(*unused, **kwargs):
                    self.assertEqual(os.environ['FASTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY'], '1' if fast else '0')
                    runtime.set_cuda_graph.assert_called_once_with(True)
                    return model
                runtime.model.side_effect = load
                with patch.object(auto, 'resolve', side_effect=resolve), \
                        patch.dict(sys.modules, {'ftllm': types.SimpleNamespace(llm=runtime)}), \
                        patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                    self.assertIs(make_normal_llm_model(a), model)

    def streaming_fixture(self):
        self.g.context = 262144
        self.g.kv = [8320] * 4
        self.g.streamed_kv = [8192] * 4
        self.gpu = {d: {'total': 3 * GiB, 'free': 3 * GiB} for d in (0, 1)}

    def test_streaming_preserves_128k_and_budgets_host_and_prefill_staging(self):
        self.streaming_fixture()
        a = self.args('--device', 'cuda:0', '--moe_device', 'numa')
        p = self.resolve(a)
        config = json.loads(a.kvmem)
        self.assertEqual((a.max_context_length, a.tokens, a.max_batch), (131072, 131072, 1))
        self.assertEqual(config['resident_tokens'], 32768)
        self.assertEqual(config['max_tokens'], 131072)
        self.assertGreaterEqual(config['host_mib'], 131072 * 8192 * 5 // memory.MiB)
        self.assertGreater(p['workspace_bytes_per_device'], GiB)  # full one-layer staging
        self.assertLess(p['resident_layers'], self.g.layers)     # Hybrid only
        self.assertLessEqual(p['gpu_estimated_bytes'][0], p['gpu_budget_bytes'][0])
        self.assertLessEqual(p['host_estimated_bytes'], p['host_budget_bytes'])
        self.assertTrue(p['automatic']['kv_streaming'])

    def test_streaming_uses_smaller_resident_pool_only_when_needed(self):
        self.streaming_fixture()
        self.gpu[0] = dict(total=int(2.4 * GiB), free=int(2.4 * GiB))
        a = self.args('--device', 'cuda:0', '--moe_device', 'numa')
        self.resolve(a)
        self.assertEqual(json.loads(a.kvmem)['resident_tokens'], 16384)

    def test_streaming_keeps_explicit_tp(self):
        self.streaming_fixture()
        a = self.args('--tp', '2', '--moe_device', 'numa')
        p = self.resolve(a)
        self.assertEqual((p['mode'], set(p['gpu_budget_bytes'])), ('tp', {0, 1}))
        self.assertTrue(p['automatic']['kv_streaming'])

    def test_streaming_can_increase_expert_cache_when_full_kv_already_fits(self):
        self.streaming_fixture()
        self.g.experts = 256
        options = ('--device', 'cudapp=2', '--moe_device_layers', '-1', '--chunked_prefill_size', '512')
        with patch.object(auto, 'kv_residencies', return_value=[0]):
            full = self.resolve(self.args(*options))
        original = memory.plan
        with patch.object(memory, 'plan', wraps=original) as plans:
            streamed = self.resolve(self.args(*options))
        self.assertEqual(streamed['kvmem']['resident_tokens'], 32768)
        self.assertGreater(streamed['moe_cuda_cache'], full['moe_cuda_cache'])
        self.assertEqual(streamed['chunked_prefill_size'], full['chunked_prefill_size'])
        self.assertEqual({call.kwargs['kv_resident'] for call in plans.call_args_list}, {0, 32768})

    def test_streaming_precedes_latency_estimates_when_experts_do_not_fit(self):
        self.streaming_fixture()
        self.g.experts = 256
        options = ('--device', 'cudapp=2', '--moe_device_layers', '-1', '--chunked_prefill_size', '512')
        for full_score in (9., 10.5, 12.):
            with self.subTest(full_score=full_score), patch.object(auto, 'latency_score',
                    side_effect=lambda p, *unused: 10. if 'kvmem' in p else full_score):
                result = self.resolve(self.args(*options))
            self.assertIn('kvmem', result)

    def test_full_kv_room_is_checked_per_device(self):
        self.streaming_fixture()
        self.g.experts = 256
        self.gpu = {0: dict(total=5 * GiB, free=5 * GiB),
                    1: dict(total=20 * GiB, free=20 * GiB)}
        p = self.resolve(self.args('--device', 'cudapp=2', '--moe_device_layers', '-1',
                                   '--chunked_prefill_size', '512'))
        self.assertIn('kvmem', p)

    def test_smaller_streaming_pool_is_tried_even_when_full_kv_fits(self):
        self.streaming_fixture()
        self.g.experts = 256
        original = memory.plan
        def plan(*args, **kwargs):
            if kwargs.get('kv_resident') == 32768:
                raise ValueError('32K pool does not fit')
            return original(*args, **kwargs)
        with patch.object(memory, 'plan', side_effect=plan):
            p = self.resolve(self.args('--device', 'cudapp=2', '--moe_device_layers', '-1',
                                       '--chunked_prefill_size', '512'))
        self.assertEqual(p['kvmem']['resident_tokens'], 16384)

    def test_full_kv_remains_available_when_streaming_does_not_help_or_fit(self):
        self.streaming_fixture()
        self.gpu = {d: dict(total=12 * GiB, free=12 * GiB) for d in (0, 1)}
        # Do not change a fully resident GPU configuration into hybrid streaming.
        full = self.resolve(self.args('--device', 'cuda:0'))
        self.assertEqual(full['resident_layers'], self.g.layers)
        self.assertNotIn('kvmem', full)
        options = ('--device', 'cudapp=2', '--moe_device_layers', '-1', '--moe_cuda_cache', '0')
        self.assertNotIn('kvmem', self.resolve(self.args(*options)))
        cached = self.resolve(self.args('--device', 'cudapp=2', '--moe_device_layers', '-1'))
        self.assertEqual(cached['resident_layers'], 0)
        self.assertNotIn('kvmem', cached)
        original = memory.plan
        def plan(*args, **kwargs):
            if kwargs.get('kv_resident'):
                raise ValueError('No room for pinned KV')
            return original(*args, **kwargs)
        with patch.object(memory, 'plan', side_effect=plan):
            self.assertNotIn('kvmem', self.resolve(self.args('--device', 'cudapp=2')))

    def test_short_context_does_not_stream_just_to_increase_cache(self):
        self.streaming_fixture()
        with patch.object(memory, 'plan', wraps=memory.plan) as plans:
            p = self.resolve(self.args('--device', 'cudapp=2', '--max_context_length', '32768'))
        self.assertNotIn('kvmem', p)
        self.assertEqual({call.kwargs['kv_resident'] for call in plans.call_args_list}, {0})

    def test_hybrid_pp_mtp_selects_cpu_embedding_without_overriding_explicit_choice(self):
        options = ('--device', 'cudapp=2', '--moe_device_layers', '-1', '--mtp', '3')
        a = self.args(*options)
        p = self.resolve(a)
        self.assertTrue(a.low_gpu_mem)
        self.assertFalse(a.cuda_embedding)
        self.assertTrue(p['low_gpu_mem'])
        for extra in (('--cuda_embedding',), ('--low_gpu_mem',)):
            a = self.args(*options, *extra)
            self.resolve(a)
            self.assertEqual(a.low_gpu_mem, extra == ('--low_gpu_mem',))
            self.assertEqual(a.cuda_embedding, extra == ('--cuda_embedding',))
        a = self.args(*options)
        a._auto_explicit.append('low_gpu_mem')  # an explicit JSON false is an opt-out
        self.resolve(a)
        self.assertFalse(a.low_gpu_mem)

    def test_automatic_cpu_embedding_does_not_change_other_execution_paths(self):
        for options in (('--device', 'cudapp=2', '--mtp', '0'),
                        ('--device', 'cuda:0', '--mtp', '3'),
                        ('--tp', '2', '--mtp', '3'),
                        ('--device', 'cudapp=2', '--mtp', '3', '--moe_device_layers', '0')):
            a = self.args(*options)
            self.resolve(a)
            self.assertFalse(a.low_gpu_mem)
        self.g.family = 'glm5'
        a = self.args('--device', 'cudapp=2', '--mtp', '3', '--moe_device_layers', '-1')
        self.resolve(a)
        self.assertFalse(a.low_gpu_mem)

    def test_streaming_does_not_hide_insufficient_ram_or_mutate_failed_request(self):
        self.streaming_fixture()
        self.ram = dict(total=8 * GiB, available=8 * GiB)
        a = self.args('--device', 'cuda:0', '--moe_device', 'numa')
        before = vars(a).copy()
        with self.assertRaises(ValueError):
            self.resolve(a)
        self.assertEqual(vars(a), before)

    def test_host_streaming_precedes_disk_experts(self):
        self.streaming_fixture()
        self.gpu = {d: dict(total=8 * GiB, free=8 * GiB) for d in (0, 1)}
        original = memory.plan
        def plan(args, *rest, **kwargs):
            if args.moe_device != 'disk' and not kwargs.get('kv_resident'):
                raise ValueError('Full KV leaves too little room for host expert residency')
            return original(args, *rest, **kwargs)
        with patch.object(memory, 'plan', side_effect=plan):
            p = self.resolve(self.args())
        self.assertEqual(p['moe_device'], 'numa')
        self.assertTrue(p['automatic']['kv_streaming'])

    def test_streaming_preserves_explicit_features_and_supported_model_boundary(self):
        self.streaming_fixture()
        a = self.args()
        a.atype = a.kv_cache_dtype = 'float16'
        a.tokens = a.max_context_length = 131072
        a.max_batch = 1
        for changes in ({'max_batch': 2}, {'kv_cache_dtype': 'fp8_e4m3'}, {'atype': 'bfloat16'},
                        {'tokens': 262144}, {'prefix_cache': 'true'}, {'cache_history': 'true'},
                        {'kvmem': '{"enabled":false}'}):
            with self.subTest(changes=changes):
                trial = copy.copy(a)
                vars(trial).update(changes)
                trial._auto_explicit = [*a._auto_explicit, *changes]
                self.assertEqual(auto.kv_residencies(trial, self.g), [0])
        for env in ({'FASTLLM_PREFIX_CACHE': '1'}, {'FASTLLM_SKIP_WARMUP': '1'}):
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(auto.kv_residencies(a, self.g), [0])
        self.g.family = 'glm5'
        self.assertEqual(auto.kv_residencies(a, self.g), [0])

    def test_streaming_disables_default_history_and_preserves_explicit_history(self):
        self.streaming_fixture()
        self.gpu = {d: dict(total=8 * GiB, free=8 * GiB) for d in (0, 1)}
        self.costs = ([4 * GiB] * 4, [1 << 20] * 4, 1 << 20, 0, GiB, False)
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as d:
                Path(d, 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp'}))
                options = ['--cache_history', 'true'] if explicit else []
                a = make_normal_parser('test').parse_args([d, '--auto', *options])
                runtime = MagicMock()
                runtime.get_cuda_memory_info.side_effect = self.gpu.__getitem__
                model = runtime.model.return_value
                model.get_max_input_len.return_value = 131072
                model.get_max_batch.return_value = 1
                def warmup():
                    if explicit:
                        model.set_kvmem.assert_not_called()
                        model.set_save_history.assert_called_once_with(True)
                    else:
                        model.set_kvmem.assert_called_once_with(**json.loads(a.kvmem))
                        model.set_save_history.assert_not_called()
                        runtime.set_cpu_historycache.assert_not_called()
                        self.assertEqual(os.environ['FASTLLM_PREFIX_CACHE'], 'false')
                model.warmup.side_effect = warmup
                with patch.object(memory, 'model_inputs', return_value=({}, [], self.g)), \
                        patch.object(memory, 'geometry', return_value=self.g), \
                        patch.object(memory, 'weight_costs', return_value=self.costs), \
                        patch.object(memory, 'host_memory', return_value=self.ram), \
                        patch.object(auto, 'cuda_hardware', return_value=self.hardware), \
                        patch.dict(sys.modules, {'ftllm': types.SimpleNamespace(llm=runtime)}), \
                        patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                    self.assertIs(make_normal_llm_model(a), model)
                self.assertEqual(model.hybrid_auto_plan['automatic']['kv_streaming'], not explicit)
                self.assertEqual(a.cache_history, 'true' if explicit else 'false')
                self.assertEqual(a.prefix_cache, '' if explicit else 'false')
                model.warmup.assert_called_once()

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

    def test_qwen_prefers_pp_without_overriding_explicit_tp_or_other_models(self):
        def score(candidate, model, costs, args, mode, *unused):
            return 1 if mode == 'tp' else 10
        with patch.object(auto, 'latency_score', side_effect=score):
            self.assertEqual(self.resolve(self.args())['mode'], 'pp')
            self.assertEqual(self.resolve(self.args('--tp', '2'))['mode'], 'tp')
            self.g.family = 'glm5'
            self.assertEqual(self.resolve(self.args())['mode'], 'tp')

    def test_host_experts_take_priority_over_disk_pp(self):
        original = memory.plan
        def plan(args, model, costs, mode, *rest, **kwargs):
            if mode == 'pp' and args.moe_device != 'disk':
                raise ValueError('Host experts only fit with TP')
            return original(args, model, costs, mode, *rest, **kwargs)
        with patch.object(memory, 'plan', side_effect=plan):
            p = self.resolve(self.args())
            self.assertEqual((p['moe_device'], p['mode']), ('numa', 'tp'))

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

    def test_automatic_streaming_configuration_reaches_model_before_warmup(self):
        self.check_saved_configuration({}, [], {}, True, streaming=True)

    def test_per_device_cache_reaches_runtime_without_environment_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp'}))
            a = make_normal_parser('test').parse_args([d, '--auto'])
            runtime = MagicMock()
            model = runtime.model.return_value
            model.get_max_input_len.return_value = 32768
            model.get_max_batch.return_value = 1
            budgets = {0: 2 * GiB, 1: GiB}
            def resolve(a, query):
                a.device, a.tp, a.moe_device = 'cudapp=2', 'off', 'numa'
                a.threads, a.tokens, a.max_batch = 4, 32768, 1
                a.moe_cuda_cache = 2 * GiB
                return {'automatic': {'cuda_graph': True}, 'moe_cuda_cache_per_device': budgets}
            def load(*unused, **kw):
                runtime.set_moe_cuda_cache.assert_called_once_with(2 * GiB, budgets)
                self.assertFalse(any(k.startswith('FASTLLM_MOE_CUDA_CACHE_BYTES_') for k in os.environ))
                return model
            runtime.model.side_effect = load
            with patch.object(auto, 'resolve', side_effect=resolve), \
                    patch.dict(sys.modules, {'ftllm': types.SimpleNamespace(llm=runtime)}), \
                    patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                self.assertIs(make_normal_llm_model(a), model)

    def test_automatic_cpu_embedding_reaches_runtime_before_model_loading(self):
        self.check_saved_configuration({'mtp': 3}, [], {}, True, cpu_embedding=True)

    def check_saved_configuration(self, settings, cli, environment, graph, streaming=False, cpu_embedding=False):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp'}))
            cfg = Path(d, 'serve.json')
            cfg.write_text(json.dumps({'path': d, 'auto': True, 'moe_cuda_cache': 0, **settings}))
            a = make_normal_parser('test').parse_args([str(cfg), *cli])
            runtime = MagicMock()
            model = runtime.model.return_value
            model.get_max_input_len.return_value = 32768
            model.get_max_batch.return_value = 1
            kv = {'max_tokens': 131072, 'resident_tokens': 32768, 'host_mib': 4096, 'prefill_tokens': 512}
            if streaming:
                model.warmup.side_effect = lambda: model.set_kvmem.assert_called_once_with(**kv)
            def resolve(a, query):
                self.assertIn('moe_cuda_cache', a._auto_explicit)
                self.assertEqual(a.moe_cuda_cache, 0)
                self.assertEqual(a.path, d)
                a.device, a.tp, a.moe_device = 'cuda:0', 'off', 'numa'
                a.threads, a.tokens, a.max_batch = 4, 32768, 1
                if streaming:
                    a.kvmem = json.dumps(kv)
                if cpu_embedding:
                    a.low_gpu_mem = True
                return {'automatic': {'cuda_graph': graph}}
            if cpu_embedding:
                def load(*unused, **kw):
                    runtime.set_cuda_embedding.assert_called_once_with(False)
                    return model
                runtime.model.side_effect = load
            with patch.object(auto, 'resolve', side_effect=resolve), \
                    patch.dict(sys.modules, {'ftllm': types.SimpleNamespace(llm=runtime)}), \
                    patch.dict(os.environ, environment, clear=True), redirect_stdout(io.StringIO()):
                self.assertIs(make_normal_llm_model(a), model)
            runtime.set_cuda_graph.assert_called_once_with(graph)
            runtime.set_moe_cuda_cache.assert_called_once_with(0)
            self.assertEqual(model.hybrid_auto_plan['automatic']['cuda_graph'], graph)


if __name__ == '__main__':
    unittest.main()

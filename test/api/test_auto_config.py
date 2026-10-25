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

    def resolve(self, a, *, use_numa=True):
        with patch.object(memory, 'model_inputs', return_value=({}, [], self.g)), \
                patch.object(memory, 'geometry', return_value=self.g), \
                patch.object(memory, 'weight_costs', return_value=self.costs), \
                patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            return auto.resolve(a, self.gpu.__getitem__, hardware=self.hardware,
                                ram=self.ram, use_numa=use_numa)

    def args(self, *options):
        return make_normal_parser('test').parse_args(['model.gguf', '--auto', *options])

    def mixed_fixture(self):
        self.g = memory.Geometry('deepseek_v41', 8, 65536, 128, 128, 8, 64,
                                 [1024] * 8, [1 << 20] * 8, 4, 1024, 1024)
        self.costs = ([4 * GiB] * 8, [1 << 20] * 8, 1 << 20, 1 << 20, 4 * GiB, False)
        self.ram = {'total': 32 * GiB, 'available': 30 * GiB}

    def test_disk_cache_ratio_default_alias_and_validation(self):
        self.assertEqual(self.args().auto_disk_cache_ratio, .5)
        self.assertEqual(self.args('--auto-disk-cache-ratio', '.7').auto_disk_cache_ratio, .7)
        for value in (-.1, 1.1, float('nan'), float('inf'), '.5', None, True):
            a = self.args()
            a.auto_disk_cache_ratio = value  # Also validate saved JSON values.
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'auto_disk_cache_ratio'):
                self.resolve(a)

    def test_disk_cache_ratio_changes_only_the_numa_disk_boundary(self):
        self.mixed_fixture()
        plans = []
        for ratio, expected_layers in ((0, 5), (.25, 4), (.5, 2), (.6, 1)):
            a = self.args('--auto_disk_cache_ratio', str(ratio))
            p = self.resolve(a)
            plans.append(p)
            self.assertEqual(p['numa_layers'], expected_layers)
            self.assertEqual(p['autotune_placement']['front'], {'numa': expected_layers})
            self.assertEqual(a.moe_device_layers, 8 - expected_layers)
            self.assertGreaterEqual(p['moe_cpu_cache'], p['disk_expert_bytes'] * ratio)
            self.assertEqual(p['disk_expert_bytes'], (8 - expected_layers) * 4 * GiB)
            self.assertLessEqual(p['host_estimated_bytes'], p['host_budget_bytes'])
            for d in p['gpu_budget_bytes']:
                self.assertLessEqual(p['gpu_estimated_bytes'][d], p['gpu_budget_bytes'][d])
                self.assertLessEqual(p['moe_cuda_cache_per_device'][d], p['moe_cuda_cache'])
        self.assertEqual(len({(p['device'], p['resident_layers'], p['chunked_prefill_size']) for p in plans}), 1)
        # Even the all-disk cache cannot cover 75%; retain the existing safe
        # fallback and explain the unmet recommendation instead of failing.
        fallback = self.resolve(self.args('--auto_disk_cache_ratio', '.75'))
        self.assertNotIn('autotune_placement', fallback)
        self.assertTrue(any('coverage' in note for note in fallback['notes']))

    def test_disk_cache_ratio_preserves_explicit_placement_and_host_only(self):
        self.mixed_fixture()
        for option, value in (('moe_device', 'disk'), ('moe_device_layers', '-1'),
                              ('moe_cuda_cache', '0'), ('moe_cpu_cache', '1g')):
            with self.subTest(option=option):
                p = self.resolve(self.args('--' + option, value))
                self.assertNotIn('autotune_placement', p)
        self.assertNotIn('autotune_placement', self.resolve(self.args(), use_numa=False))
        explicit_tp = self.resolve(self.args('--tp', 'cuda:0,1'))
        self.assertEqual(explicit_tp['mode'], 'tp')
        self.assertNotIn('autotune_placement', explicit_tp)
        self.ram = {'total': 64 * GiB, 'available': 64 * GiB}
        host = self.resolve(self.args())
        self.assertEqual(host['moe_device'], 'numa')
        self.assertNotIn('autotune_placement', host)

    def test_mixed_disk_budget_includes_both_transport_pools(self):
        self.mixed_fixture()
        a = self.args('--moe_device', 'disk')
        baseline = self.resolve(a)
        with patch.object(memory, 'weight_costs', return_value=self.costs):
            _, original = auto.mixed_disk_plan(a, baseline, {}, [], self.g, self.gpu, self.ram, .5)
            baseline['expert_transport_bytes_per_device'] += GiB
            baseline['workspace_bytes_per_device'] += GiB
            baseline['expert_host_staging_bytes'] += GiB
            _, mixed = auto.mixed_disk_plan(a, baseline, {}, [], self.g, self.gpu, self.ram, .5)
        self.assertEqual(original['numa_layers'], mixed['numa_layers'])
        self.assertEqual(original['moe_cpu_cache'] - mixed['moe_cpu_cache'], GiB)
        self.assertEqual(mixed['expert_host_staging_bytes'], GiB)
        self.assertEqual(mixed['expert_transport_bytes_per_device'], GiB)
        for d in mixed['gpu_budget_bytes']:
            original_cache = original['moe_cuda_cache'] + original['moe_cuda_cache_per_device'][d]
            mixed_cache = mixed['moe_cuda_cache'] + mixed['moe_cuda_cache_per_device'][d]
            self.assertEqual(original_cache - mixed_cache, GiB)
            self.assertLessEqual(mixed['gpu_estimated_bytes'][d], mixed['gpu_budget_bytes'][d])

    def test_mixed_disk_draft_weights_follow_their_target_layers(self):
        self.mixed_fixture()
        self.g.kv += [1024, 1024]
        self.g.state += [1 << 20, 1 << 20]
        self.costs = (self.costs[0] + [3 * GiB, GiB], self.costs[1] + [1 << 20, 1 << 20],
                      *self.costs[2:])
        p = self.resolve(self.args())
        boundary = p['resident_layers'] + p['numa_layers']
        self.assertLessEqual(boundary, 6)
        self.assertEqual(p['disk_expert_bytes'], (8 - boundary) * 4 * GiB + 4 * GiB)
        self.assertGreaterEqual(p['disk_cpu_cache_ratio'], .5)

    def test_streamed_disk_gguf_chunk_priority_preserves_host_policy(self):
        self.g.family = 'deepseek_v41'
        self.g.disk_prefill = {'canonical_gguf': True, 'streamed_mmq': True}
        small = dict(chunked_prefill_size=512, resident_layers=0, moe_cuda_cache=2*GiB,
                     expert_transport_bytes_per_device=256*1024**2)
        large = dict(small, chunked_prefill_size=2048, moe_cuda_cache=GiB)
        self.assertGreater(auto._deepseek_priority(large, self.g, 'disk'),
                           auto._deepseek_priority(small, self.g, 'disk'))
        prompt = dict(large, chunked_prefill_size=4096, moe_cuda_cache=GiB//2)
        self.assertGreater(auto._deepseek_priority(prompt, self.g, 'disk'),
                           auto._deepseek_priority(large, self.g, 'disk'))
        fallback = dict(prompt, chunked_prefill_size=8192, moe_cuda_cache=3*GiB)
        self.assertGreater(auto._deepseek_priority(prompt, self.g, 'disk'),
                           auto._deepseek_priority(fallback, self.g, 'disk'))
        # Concurrent decode does not multiply V4.1's streamed prefill rows.
        self.assertEqual(auto._deepseek_priority(dict(prompt, max_batch=4), self.g, 'disk'),
                         auto._deepseek_priority(prompt, self.g, 'disk'))
        for backend in ('numa', 'cpu'):
            self.assertLess(auto._deepseek_priority(large, self.g, backend),
                            auto._deepseek_priority(small, self.g, backend))
        small['expert_transport_bytes_per_device'] = large['expert_transport_bytes_per_device'] = 0
        self.assertLess(auto._deepseek_priority(large, self.g, 'disk'),
                        auto._deepseek_priority(small, self.g, 'disk'))

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

    def test_deepseek_disk_prefers_large_chunk_and_cpu_embedding(self):
        self.g.family = 'deepseek_v41'
        a = self.args('--moe_device', 'disk', '--moe_device_layers', '-1')
        p = self.resolve(a)
        self.assertEqual(a.chunked_prefill_size, 8192)
        self.assertTrue(a.low_gpu_mem)
        self.assertFalse(a.cuda_embedding)
        self.assertTrue(p['automatic']['low_gpu_mem'])
        explicit = self.args('--moe_device', 'disk', '--moe_device_layers', '-1',
                             '--cuda_embedding', '--chunked_prefill_size', '512')
        self.resolve(explicit)
        self.assertTrue(explicit.cuda_embedding)
        self.assertFalse(explicit.low_gpu_mem)
        self.assertEqual(explicit.chunked_prefill_size, 512)
        # A full-GPU plan does not need the constrained-memory preference.
        full = self.args('--moe_device', 'disk', '--moe_device_layers', '0')
        self.resolve(full)
        self.assertFalse(full.low_gpu_mem)

    def test_deepseek_hybrid_ratio_default_and_explicit_precedence(self):
        self.g.family = 'deepseek_v41'
        for backend in ('numa', 'cpu', 'disk'):
            for options, ratio in (((), .95), (('--gpu_mem_ratio', '.9'), .9),
                                   (('--gpu_mem_ratio', '.97'), .97)):
                with self.subTest(backend=backend, options=options):
                    a = self.args('--device', 'cuda:0', '--moe_device', backend,
                                  '--moe_device_layers', '-1', *options)
                    p = self.resolve(a)
                    self.assertEqual(a.gpu_mem_ratio, ratio)
                    self.assertEqual(p['gpu_mem_ratio'], ratio)
                    self.assertAlmostEqual(p['gpu_budget_bytes'][0], 8 * GiB * ratio, delta=1)
        capped = self.args('--device', 'cuda:0', '--moe_device_layers', '-1',
                           '--auto_gpu_memory_limit', '6g')
        self.assertEqual(self.resolve(capped)['gpu_budget_bytes'][0], 6 * GiB)

    def test_hybrid_ratio_does_not_change_full_gpu_or_other_models(self):
        self.g.family = 'deepseek_v41'
        full = self.args('--device', 'cuda:0', '--moe_device_layers', '0')
        p = self.resolve(full)
        self.assertEqual(p['resident_layers'], self.g.layers)
        self.assertEqual((full.gpu_mem_ratio, p['gpu_mem_ratio']), (.9, .9))
        self.g.family = 'qwen4'
        other = self.args('--device', 'cuda:0', '--moe_device_layers', '-1')
        p = self.resolve(other)
        self.assertEqual((other.gpu_mem_ratio, p['gpu_mem_ratio']), (.9, .9))

    def test_deepseek_numa_prefers_largest_feasible_chunk_and_pp(self):
        self.g.family = 'deepseek_v41'
        a = self.args('--moe_device', 'numa', '--moe_device_layers', '-1')
        p = self.resolve(a)
        self.assertEqual((p['mode'], a.chunked_prefill_size), ('pp', 8192))
        self.assertEqual(a.moe_device, 'numa')

    def test_deepseek_gguf_auto_slabs_honor_explicit_opt_out(self):
        self.g.family = 'deepseek_v41'
        tensors = [memory.Tensor('layers.0.ffn.experts.0.w2.weight', (256, 256), 1 << 20, 'gguf:12')]
        with patch.object(memory, 'model_inputs', return_value=({}, tensors, self.g)), \
                patch.object(memory, 'geometry', return_value=self.g), \
                patch.object(memory, 'weight_costs', return_value=self.costs), \
                patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            for flags, expected in [((), 64), (('--cuda_slab', '0'), 0), (('--cuda_slab', '32'), 32)]:
                a = self.args(*flags)
                p = auto.resolve(a, self.gpu.__getitem__, hardware=self.hardware,
                                 ram=self.ram, use_numa=True)
                self.assertEqual(a.cuda_slab, expected)
                self.assertEqual(p['cuda_slab'], expected)

    def deepseek_cache_fixture(self):
        self.g.family, self.g.experts = 'deepseek_v41', 256
        self.g.activation_width = 512 * 1024
        self.costs = ([4 * GiB] * 4, *self.costs[1:])
        return ('--moe_device', 'numa', '--moe_device_layers', '-1')

    def test_deepseek_cache_target_precedes_larger_chunk(self):
        options = self.deepseek_cache_fixture()
        p = self.resolve(self.args(*options))
        large = self.resolve(self.args(*options, '--chunked_prefill_size', '2048'))
        small = self.resolve(self.args(*options, '--chunked_prefill_size', '512'))
        self.assertEqual((p['mode'], p['chunked_prefill_size']), ('pp', 1024))
        self.assertGreaterEqual(p['moe_cuda_cache'], 3 * GiB)
        self.assertLess(large['moe_cuda_cache'], 3 * GiB)
        # Once 3 GiB fits, grow the chunk rather than maximizing cache alone.
        self.assertGreater(small['moe_cuda_cache'], p['moe_cuda_cache'])

    def test_deepseek_chunk_floor_precedes_cache_target_and_can_fall_back(self):
        # Keep this boundary case at its original budget as defaults evolve.
        options = (*self.deepseek_cache_fixture(), '--device', 'cuda:0', '--gpu_mem_ratio', '.9')
        self.gpu[0]['free'] = 21 * GiB // 4
        p = self.resolve(self.args(*options))
        small = self.resolve(self.args(*options, '--chunked_prefill_size', '256'))
        self.assertEqual(p['chunked_prefill_size'], 512)
        self.assertLess(p['moe_cuda_cache'], 3 * GiB)
        self.assertGreaterEqual(small['moe_cuda_cache'], 3 * GiB)
        # Both targets are soft: retain a safe plan when even 512 cannot fit.
        self.gpu[0]['free'] = 5 * GiB // 2
        fallback = self.resolve(self.args(*options))
        self.assertEqual(fallback['chunked_prefill_size'], 256)
        self.assertLessEqual(fallback['gpu_estimated_bytes'][0], fallback['gpu_budget_bytes'][0])

    def test_deepseek_cache_targets_keep_explicit_values_and_short_context(self):
        options = (*self.deepseek_cache_fixture(), '--device', 'cuda:0')
        cases = [(('--moe_cuda_cache', '0'), 2048, 0),
                 (('--chunked_prefill_size', '2048'), 2048, None),
                 (('--chunked_prefill_size', '256', '--moe_cuda_cache', '1g'), 256, GiB),
                 (('--max_context_length', '128'), 128, None)]
        for flags, chunk, cache in cases:
            with self.subTest(flags=flags):
                a = self.args(*options, *flags)
                p = self.resolve(a)
                self.assertEqual(a.chunked_prefill_size, chunk)
                if cache is not None:
                    self.assertEqual(a.moe_cuda_cache, cache)
                if chunk == 128:
                    self.assertEqual((a.max_context_length, a.tokens), (128, 128))
                self.assertLessEqual(p['gpu_estimated_bytes'][0], p['gpu_budget_bytes'][0])
                self.assertLessEqual(p['host_estimated_bytes'], p['host_budget_bytes'])

    def test_deepseek_auto_excludes_tp_even_with_better_cache(self):
        options = self.deepseek_cache_fixture()
        original = memory.plan
        modes = []
        def plan(args, model, costs, mode, *rest, **kwargs):
            modes.append(mode)
            # Present feasible per-topology results whose priorities differ.
            trial = copy.copy(args)
            trial.chunked_prefill_size = 2048 if mode == 'pp' else 512
            trial.moe_cuda_cache = GiB if mode == 'pp' else 3 * GiB
            return original(trial, model, costs, mode, *rest, **kwargs)
        with patch.object(memory, 'plan', side_effect=plan):
            p = self.resolve(self.args(*options))
            self.assertEqual((p['mode'], p['chunked_prefill_size'], p['moe_cuda_cache']),
                             ('pp', 2048, GiB))
            self.assertEqual(set(modes), {'pp'})
            explicit = self.resolve(self.args(*options, '--tp', '2'))
            self.assertEqual((explicit['mode'], explicit['chunked_prefill_size'], explicit['moe_cuda_cache']),
                             ('tp', 512, 3 * GiB))

    def test_deepseek_auto_uses_disk_pp_instead_of_tp_capacity_fallback(self):
        self.g.family = 'deepseek_v41'
        original = memory.plan
        modes = []
        def plan(args, model, costs, mode, *rest, **kwargs):
            modes.append(mode)
            if mode == 'pp' and args.moe_device != 'disk':
                raise ValueError('Host experts only fit with TP')
            return original(args, model, costs, mode, *rest, **kwargs)
        with patch.object(memory, 'plan', side_effect=plan):
            p = self.resolve(self.args())
            self.assertEqual((p['moe_device'], p['mode']), ('disk', 'pp'))
            self.assertEqual(set(modes), {'pp'})
            with self.assertRaisesRegex(ValueError, 'Host experts only fit with TP'):
                self.resolve(self.args('--moe_device', 'numa'))
        a = self.args()
        with patch.dict(os.environ, {'FASTLLM_TP': '2'}, clear=True):
            layouts = list(auto.topologies(a, set(a._auto_explicit), self.hardware[0], self.gpu,
                                          family='deepseek_v41'))
        self.assertEqual([row[2] for row in layouts], ['tp'])

    def test_deepseek_full_gpu_does_not_need_cache_target(self):
        self.g.family = 'deepseek_v41'
        p = self.resolve(self.args('--device', 'cuda:0'))
        self.assertEqual(p['resident_layers'], self.g.layers)
        self.assertEqual(p['moe_cuda_cache'], 0)

    def test_deepseek_latency_can_choose_full_gpu_over_larger_hybrid_chunk(self):
        self.g.family = 'deepseek_v41'
        self.g.activation_width = 1 << 20
        self.costs = ([23 * GiB // 4] * 4, *self.costs[1:])
        self.gpu[0] = dict(total=30 * GiB, free=30 * GiB)
        # Both candidates meet the chunk/cache targets. A larger hybrid
        # chunk must not veto the faster full-GPU candidate before ranking.
        with patch.object(auto, 'latency_score', side_effect=lambda p, *rest:
                          1 if p['resident_layers'] == self.g.layers else 100):
            p = self.resolve(self.args('--device', 'cuda:0', '--gpu_mem_ratio', '.9', '--max_batch', '2'))
        self.assertEqual((p['resident_layers'], p['chunked_prefill_size']), (self.g.layers, 512))

    def test_deepseek_implicit_embedding_is_charged_before_chunk_selection(self):
        self.g.family = 'deepseek_v41'
        def costs(tensors, model, a):
            c = list(self.costs)
            if a.cuda_embedding and not a.low_gpu_mem:
                c[2] += GiB
            return tuple(c)
        with patch.object(memory, 'model_inputs', return_value=({}, [], self.g)), \
                patch.object(memory, 'geometry', return_value=self.g), \
                patch.object(memory, 'weight_costs', side_effect=costs), \
                patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            plans = []
            for flags in ((), ('--low_gpu_mem',)):
                a = self.args('--device', 'cuda:0', '--moe_device', 'disk', '--moe_device_layers', '-1',
                              '--chunked_prefill_size', '512', '--moe_cuda_cache', '0', *flags)
                # Mark the default false explicitly to exercise normal mode.
                if not flags:
                    a._auto_explicit.append('low_gpu_mem')
                plans.append(auto.resolve(a, self.gpu.__getitem__, hardware=self.hardware,
                                          ram=self.ram, use_numa=True))
            self.assertEqual(sum(plans[0]['gpu_estimated_bytes'].values()) -
                             sum(plans[1]['gpu_estimated_bytes'].values()), GiB)

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
        self.assertIsNone(a.force_ddr_limit)

    def test_host_budget_override_is_final_and_auto_preserves_defaults(self):
        normal = self.resolve(self.args())
        self.assertEqual(normal, self.resolve(self.args('--force_ddr_limit', 'auto')))
        for flag, value, expected in (('--force_ddr_limit', '40g', 40 * GiB),
                                      ('--force-ddr-limit', '20GiB', 20 * GiB)):
            with self.subTest(flag=flag):
                p = self.resolve(self.args(flag, value))
                self.assertEqual(p['host_budget_bytes'], expected)
                self.assertEqual(p['automatic']['force_ddr_limit'], expected)
                self.assertIsNone(p['automatic']['host_memory_limit'])
                self.assertEqual(p['gpu_budget_bytes'], normal['gpu_budget_bytes'])

    def test_host_budget_override_can_avoid_disk_experts(self):
        self.mixed_fixture()
        normal = self.resolve(self.args())
        self.assertGreater(normal['disk_expert_bytes'], 0)
        a = self.args('--force_ddr_limit', '40g')
        forced = self.resolve(a)
        self.assertEqual(a.moe_device, 'numa')
        self.assertEqual(forced['moe_cpu_cache'], 0)
        self.assertGreater(forced['host_estimated_bytes'], normal['host_budget_bytes'])
        self.assertLessEqual(forced['host_estimated_bytes'], 40 * GiB)

    def test_host_budget_and_limit_are_mutually_exclusive_in_cli_and_saved_settings(self):
        with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            self.args('--auto_host_memory_limit', '20g', '--force_ddr_limit', '40g')
        a = self.args('--force_ddr_limit', '40g')
        a.auto_host_memory_limit = '20g'  # Saved JSON bypasses argparse.
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            self.resolve(a)

    def test_host_budget_override_rejects_invalid_saved_values(self):
        for value in (0, -1, float('inf'), float('nan'), True, 'invalid'):
            a = self.args()
            a.force_ddr_limit = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'force_ddr_limit'):
                self.resolve(a)

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

    def test_qwen35_auto_uses_embedded_mtp_tp_and_native_context(self):
        config = dict(model_type='qwen3_5_text', num_hidden_layers=64, hidden_size=5120,
                      num_attention_heads=24, num_key_value_heads=4, head_dim=256,
                      intermediate_size=17408, max_position_embeddings=262144,
                      layer_types=['linear_attention'] * 3 + ['full_attention'] * 1)
        config['layer_types'] *= 16
        config.update(linear_num_key_heads=16, linear_num_value_heads=48,
                      linear_key_head_dim=128, linear_value_head_dim=128,
                      linear_conv_kernel_dim=4, mtp_num_hidden_layers=1)
        metadata = {'model_type': 'qwen3_5', 'text_config': config}
        geometry = memory.geometry(metadata, 'float16', 5)
        self.assertEqual((len(geometry.kv), sum(bool(x) for x in geometry.kv[:64])), (65, 16))
        self.assertEqual(geometry.kv[3], 4096)
        self.assertGreater(geometry.state[0], 0)
        costs = ([0] * 65, [300 << 20] * 65, 3 * GiB, 4 * GiB, GiB, False)
        gpu = {d: {'total': 72 * GiB, 'free': 72 * GiB} for d in (0, 1)}
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'mtp.safetensors').write_bytes(b'draft')
            a = self.args()
            a.path = directory
            a.dtype = 'fp8_e4m3'
            with patch.object(memory, 'model_inputs', return_value=(metadata, [], geometry)), \
                    patch.object(memory, 'weight_costs', return_value=costs), \
                    patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                result = auto.resolve(a, gpu.__getitem__, hardware=self.hardware,
                                      ram={'total': 256 * GiB, 'available': 256 * GiB},
                                      use_numa=True)
                self.assertEqual(os.environ['FASTLLM_QWEN35_MTP_VERIFY_CUDA_GRAPH'], '0')
            self.assertEqual((a.tp, a.max_batch, a.max_context_length, a.tokens, a.mtp),
                             ('cuda:0,1', 4, 262144, 1048576, 5))
            self.assertTrue(a.cuda_embedding)
            self.assertEqual(result['mode'], 'tp')
            self.assertFalse(result['automatic']['mtp_verify_cuda_graph'])

            opt_out = self.args('--mtp', '0', '--max_batch', '2', '--max_context_length', '65536')
            opt_out.path = directory
            opt_out_costs = (costs[0][:-1], costs[1][:-1], *costs[2:])
            with patch.object(memory, 'model_inputs', return_value=(metadata, [], geometry)), \
                    patch.object(memory, 'weight_costs', return_value=opt_out_costs), \
                    patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                auto.resolve(opt_out, gpu.__getitem__, hardware=self.hardware,
                             ram={'total': 256 * GiB, 'available': 256 * GiB}, use_numa=True)
            self.assertEqual((opt_out.mtp, opt_out.max_batch, opt_out.max_context_length),
                             (0, 2, 65536))

            constrained = self.args()
            constrained.path = directory
            smaller = {d: {'total': 44 * GiB, 'free': 44 * GiB} for d in (0, 1)}
            with patch.object(memory, 'model_inputs', return_value=(metadata, [], geometry)), \
                    patch.object(memory, 'weight_costs', return_value=costs), \
                    patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                auto.resolve(constrained, smaller.__getitem__, hardware=self.hardware,
                             ram={'total': 256 * GiB, 'available': 256 * GiB}, use_numa=True)
            self.assertTrue(4096 <= constrained.max_context_length < 262144)
            self.assertEqual(constrained.tokens, 4 * constrained.max_context_length)

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

    def test_unequal_expert_layers_use_weighted_pp_before_disk(self):
        self.g.family = 'deepseek_v41'
        self.g.layers = 6
        self.g.kv = [1024] * 6
        self.g.state = [1 << 20] * 6
        self.costs = ([GiB] * 3 + [2 * GiB] * 3, [1 << 20] * 6, 0, 0, 2 * GiB, False)
        self.gpu = {d: dict(total=4 * GiB, free=4 * GiB) for d in range(3)}
        info = {d: self.hardware[0][0].copy() for d in self.gpu}
        self.hardware = info, {(a, b): True for a in info for b in info}
        self.ram = dict(total=7 * GiB, available=7 * GiB)
        p = self.resolve(self.args('--max_context_length', '4096'))
        self.assertEqual(p['moe_device'], 'numa')
        self.assertEqual(p['mode'], 'pp')
        self.assertEqual(p['resident_layers'], 5)
        self.assertEqual(p['device'], "{'cuda:0': 2, 'cuda:1': 2, 'cuda:2': 1}")
        self.assertEqual(p['moe_cpu_cache'], 0)
        # An explicit layout must not silently become a different partition.
        fixed = self.resolve(self.args('--max_context_length', '4096', '--device', 'cudapp=0,1,2'))
        self.assertEqual(fixed['device'], 'cudapp=0,1,2')
        self.assertEqual(fixed['moe_device'], 'disk')

    def test_v41_shared_cpu_and_numa_have_identical_capacity_plans(self):
        self.g.family = 'deepseek_v41'
        self.g.experts = 128
        self.g.gguf_expert_bytes = {i: 2 * GiB // 128 for i in range(4)}
        self.costs = ([2 * GiB] * 4, [1 << 20] * 4, 1 << 20, 1 << 20, 2 * GiB, False)
        self.gpu = {d: dict(total=6 * GiB, free=6 * GiB) for d in (0, 1)}
        self.ram = dict(total=int(12.5 * GiB), available=int(12.5 * GiB))
        plans = [self.resolve(self.args('--max_context_length', '4096', '--moe_device', backend))
                 for backend in ('cpu', 'numa')]
        for p in plans:
            self.assertGreater(p['moe_cuda_cache'], 0)
            self.assertEqual(p['expert_transport_bytes_per_device'], 96 * (2 * GiB // 128) + 272 * (1 << 20))
            self.assertLessEqual(p['host_estimated_bytes'], p['host_budget_bytes'])
        for key in ('device', 'mode', 'resident_layers', 'chunked_prefill_size',
                    'moe_cuda_cache', 'host_estimated_bytes', 'gpu_estimated_bytes'):
            self.assertEqual(plans[0][key], plans[1][key], key)
        legacy = self.resolve(self.args('--max_context_length', '4096'), use_numa=False)
        self.assertEqual(legacy['moe_device'], 'cpu')
        self.assertEqual(legacy['moe_cuda_cache'], 0)
        self.assertEqual(legacy['expert_transport_bytes_per_device'], 0)

    def test_v41_capacity_fallback_does_not_change_other_families(self):
        for family in ('qwen3', 'qwen4', 'glm5'):
            with self.subTest(family=family):
                self.g.family = family
                backends = []
                original = memory.prepare
                def prepare(a, *rest, **kwargs):
                    backends.append(a.moe_device)
                    if a.moe_device == 'numa':
                        raise ValueError('Force existing disk fallback')
                    return original(a, *rest, **kwargs)
                with patch.object(memory, 'prepare', side_effect=prepare), \
                        patch.object(auto, 'capacity_pp_topologies') as weighted:
                    p = self.resolve(self.args())
                self.assertEqual(set(backends), {'numa', 'disk'})
                self.assertEqual(p['moe_device'], 'disk')
                self.assertNotIn('cuda_slab', p)
                weighted.assert_not_called()

    def test_shared_cpu_models_use_the_numa_cache_capacity_policy(self):
        for family in ('qwen4', 'glm5'):
            with self.subTest(family=family):
                self.g.family, self.g.experts = family, 128
                self.costs = ([8 * GiB] * 4, [1 << 20] * 4, 1 << 20, 1 << 20, 8 * GiB, False)
                self.ram = dict(total=128 * GiB, available=128 * GiB)
                plans = [self.resolve(self.args('--moe_device', backend))
                         for backend in ('cpu', 'numa')]
                self.assertGreater(plans[0]['moe_cuda_cache'], 0)
                for key in ('mode', 'resident_layers', 'chunked_prefill_size',
                            'moe_cuda_cache', 'host_estimated_bytes', 'gpu_estimated_bytes'):
                    self.assertEqual(plans[0][key], plans[1][key], key)
                legacy = self.resolve(self.args('--moe_device', 'cpu'), use_numa=False)
                if family == 'glm5':
                    self.assertEqual(legacy['moe_cuda_cache'], 0)

    def test_capacity_pp_ratios_are_unique_after_normalization(self):
        layouts = [('cudapp=0,1', 'off', 'pp', {0: 1, 1: 1})]
        mappings = [mapping for _, _, _, mapping in
                    auto.capacity_pp_topologies(self.g, self.costs, layouts, self.gpu)]
        self.assertNotIn({0: 1, 1: 1}, mappings)
        self.assertEqual(len(mappings), len({tuple(m.items()) for m in mappings}))

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

    def test_deepseek_auto_packs_native_kv_and_respects_explicit_storage(self):
        self.g.family = 'deepseek_v41'
        md = {'model_type': 'deepseek_v41', 'head_dim': 512, 'index_head_dim': 128}
        for flag, expected in [(None, 'fp4'), ('float16', 'float16'),
                               ('fp4_e2m1', 'fp4_e2m1'), ('auto', 'auto')]:
            with self.subTest(flag=flag), \
                    patch.object(memory, 'model_inputs', return_value=(md, [], self.g)), \
                    patch.object(memory, 'geometry', return_value=self.g) as geometry, \
                    patch.object(memory, 'weight_costs', return_value=self.costs), \
                    patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                a = self.args(*(['--kv_cache_dtype', flag] if flag else []))
                p = auto.resolve(a, self.gpu.__getitem__, hardware=self.hardware,
                                 ram=self.ram, use_numa=True)
                self.assertEqual(a.kv_cache_dtype, expected)
                self.assertEqual(p['kv_cache_dtype'], expected)
                self.assertEqual(geometry.call_args.args[1], expected)

    def test_lossless_kv_default_checks_native_debug_flag_and_geometry(self):
        self.g.family = 'deepseek_v41'
        md = {'text_config': {'head_dim': 512, 'index_head_dim': 128}}
        for flag, expected in [('', 'fp4'), ('0', 'fp4'), ('1', 'float16'),
                               ('false', 'float16'), ('off', 'float16')]:
            with patch.dict(os.environ, {'FASTLLM_DSV41_DISABLE_FAKE_QUANT': flag}, clear=True):
                self.assertEqual(auto.default_kv_dtype(self.g, md), expected)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(auto.default_kv_dtype(self.g, {'head_dim': 513, 'index_head_dim': 128}), 'float16')
            self.assertEqual(auto.default_kv_dtype(self.g, {'head_dim': None, 'index_head_dim': 128}), 'float16')
            self.g.family = 'qwen4'
            self.assertEqual(auto.default_kv_dtype(self.g, md), 'float16')
            self.assertEqual(auto.default_kv_dtype(self.g, {'text_config': None}), 'float16')

    def test_deepseek_auto_defaults_allow_explicit_opt_out_and_leave_other_models_alone(self):
        md = {'model_type': 'deepseek_v41', 'text_config': {'model_type': 'deepseek_v41_text'}}
        for options, drafts, fast in [((), 5, True), (('--dspark', '0'), 0, True),
                                     (('--speculative_algorithm', 'off'), 0, True),
                                     (('--mtp', '0'), 0, True),
                                     (('--dspark', '3', '--no-fast-prefill'), 3, False)]:
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
        cases = [({}, [], 5, True, .5),
                 ({}, ['--dspark', '3', '--speculative_dspark_confidence_threshold', '.2'], 3, True, .2),
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

    def streaming_args(self, *options):
        return self.args('--prefix_cache', 'false', '--cache_history', 'false', *options)

    def test_default_reuse_is_not_disabled_to_fit_memory(self):
        self.streaming_fixture()
        a = self.args('--device', 'cuda:0', '--moe_device', 'numa')
        before = vars(a).copy()
        with self.assertRaises(ValueError):
            self.resolve(a)
        self.assertEqual(vars(a), before)
        self.assertTrue(self.resolve(self.streaming_args(
            '--device', 'cuda:0', '--moe_device', 'numa'))['automatic']['kv_streaming'])

    def test_streaming_preserves_128k_and_budgets_host_and_prefill_staging(self):
        self.streaming_fixture()
        a = self.streaming_args('--device', 'cuda:0', '--moe_device', 'numa')
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
        a = self.streaming_args('--device', 'cuda:0', '--moe_device', 'numa')
        self.resolve(a)
        self.assertEqual(json.loads(a.kvmem)['resident_tokens'], 16384)

    def test_streaming_keeps_explicit_tp(self):
        self.streaming_fixture()
        a = self.streaming_args('--tp', '2', '--moe_device', 'numa')
        p = self.resolve(a)
        self.assertEqual((p['mode'], set(p['gpu_budget_bytes'])), ('tp', {0, 1}))
        self.assertTrue(p['automatic']['kv_streaming'])

    def test_streaming_can_increase_expert_cache_when_full_kv_already_fits(self):
        self.streaming_fixture()
        self.g.experts = 256
        options = ('--device', 'cudapp=2', '--moe_device_layers', '-1', '--chunked_prefill_size', '512')
        with patch.object(auto, 'kv_residencies', return_value=[0]):
            full = self.resolve(self.streaming_args(*options))
        original = memory.plan
        with patch.object(memory, 'plan', wraps=original) as plans:
            streamed = self.resolve(self.streaming_args(*options))
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
                result = self.resolve(self.streaming_args(*options))
            self.assertIn('kvmem', result)

    def test_full_kv_room_is_checked_per_device(self):
        self.streaming_fixture()
        self.g.experts = 256
        self.gpu = {0: dict(total=5 * GiB, free=5 * GiB),
                    1: dict(total=20 * GiB, free=20 * GiB)}
        p = self.resolve(self.streaming_args('--device', 'cudapp=2', '--moe_device_layers', '-1',
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
            p = self.resolve(self.streaming_args('--device', 'cudapp=2', '--moe_device_layers', '-1',
                                       '--chunked_prefill_size', '512'))
        self.assertEqual(p['kvmem']['resident_tokens'], 16384)

    def test_full_kv_remains_available_when_streaming_does_not_help_or_fit(self):
        self.streaming_fixture()
        self.gpu = {d: dict(total=12 * GiB, free=12 * GiB) for d in (0, 1)}
        # Do not change a fully resident GPU configuration into hybrid streaming.
        full = self.resolve(self.streaming_args('--device', 'cuda:0'))
        self.assertEqual(full['resident_layers'], self.g.layers)
        self.assertNotIn('kvmem', full)
        options = ('--device', 'cudapp=2', '--moe_device_layers', '-1', '--moe_cuda_cache', '0')
        self.assertNotIn('kvmem', self.resolve(self.streaming_args(*options)))
        cached = self.resolve(self.streaming_args('--device', 'cudapp=2', '--moe_device_layers', '-1'))
        self.assertEqual(cached['resident_layers'], 0)
        self.assertNotIn('kvmem', cached)
        original = memory.plan
        def plan(*args, **kwargs):
            if kwargs.get('kv_resident'):
                raise ValueError('No room for pinned KV')
            return original(*args, **kwargs)
        with patch.object(memory, 'plan', side_effect=plan):
            self.assertNotIn('kvmem', self.resolve(self.streaming_args('--device', 'cudapp=2')))

    def test_short_context_does_not_stream_just_to_increase_cache(self):
        self.streaming_fixture()
        with patch.object(memory, 'plan', wraps=memory.plan) as plans:
            p = self.resolve(self.streaming_args('--device', 'cudapp=2', '--max_context_length', '32768'))
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
        a = self.streaming_args('--device', 'cuda:0', '--moe_device', 'numa')
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
            p = self.resolve(self.streaming_args())
        self.assertEqual(p['moe_device'], 'numa')
        self.assertTrue(p['automatic']['kv_streaming'])

    def test_streaming_preserves_explicit_features_and_supported_model_boundary(self):
        self.streaming_fixture()
        a = self.streaming_args()
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
            trial = copy.copy(a)
            if 'FASTLLM_PREFIX_CACHE' in env:
                trial.prefix_cache = ''
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(auto.kv_residencies(trial, self.g), [0])
        self.g.family = 'glm5'
        self.assertEqual(auto.kv_residencies(a, self.g), [0])

    def test_default_reuse_reaches_runtime_and_streaming_requires_opt_out(self):
        self.streaming_fixture()
        self.gpu = {d: dict(total=8 * GiB, free=8 * GiB) for d in (0, 1)}
        self.costs = ([4 * GiB] * 4, [1 << 20] * 4, 1 << 20, 0, GiB, False)
        cases = (
            ((), True, '', False),
            (('--cache_history', 'true'), True, '', False),
            (('--prefix_cache', 'false'), True, 'false', False),
            (('--cache_history', 'false'), False, '', False),
            (('--prefix_cache', 'false', '--cache_history', 'false'), False, 'false', True),
        )
        for options, history, prefix, streaming in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as d:
                Path(d, 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp'}))
                a = make_normal_parser('test').parse_args([d, '--auto', *options])
                runtime = MagicMock()
                runtime.get_cuda_memory_info.side_effect = self.gpu.__getitem__
                model = runtime.model.return_value
                model.get_max_input_len.return_value = 131072
                model.get_max_batch.return_value = 1
                def warmup():
                    if streaming:
                        model.set_kvmem.assert_called_once_with(**json.loads(a.kvmem))
                    else:
                        model.set_kvmem.assert_not_called()
                    if history:
                        model.set_save_history.assert_called_once_with(True)
                    else:
                        model.set_save_history.assert_not_called()
                        runtime.set_cpu_historycache.assert_not_called()
                    self.assertEqual(os.environ.get('FASTLLM_PREFIX_CACHE', ''), prefix)
                model.warmup.side_effect = warmup
                with patch.object(memory, 'model_inputs', return_value=({}, [], self.g)), \
                        patch.object(memory, 'geometry', return_value=self.g), \
                        patch.object(memory, 'weight_costs', return_value=self.costs), \
                        patch.object(memory, 'host_memory', return_value=self.ram), \
                        patch.object(auto, 'cuda_hardware', return_value=self.hardware), \
                        patch.dict(sys.modules, {'ftllm': types.SimpleNamespace(llm=runtime)}), \
                        patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                    self.assertIs(make_normal_llm_model(a), model)
                self.assertEqual(model.hybrid_auto_plan['automatic']['kv_streaming'], streaming)
                self.assertEqual(a.cache_history, 'true' if history else 'false')
                self.assertEqual(a.prefix_cache, prefix)
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

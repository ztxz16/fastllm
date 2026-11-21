import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools.util import add_server_args, make_normal_llm_model, make_normal_parser, _is_naive_n05_family


class NaiveTensorParallelCliTest(unittest.TestCase):
    def configure(self, options, model_type="naive_n05_flash", saved=None, config=None, environment=None, native_chunk=8192):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(
                {"model_type": model_type, **(config or {})}))
            model_path = str(root)
            if saved is not None:
                path = root / "launch.json"
                path.write_text(json.dumps({"model": str(root), **saved}))
                model_path = str(path)
            parser = make_normal_parser("test")
            add_server_args(parser)
            args = parser.parse_args([model_path, "-t", "4", *options])
            fake = types.ModuleType("ftllm")
            fake.llm = MagicMock()
            fake.llm.model.return_value.get_max_input_len.return_value = 4096
            fake.llm.model.return_value.get_max_batch.return_value = 1
            fake.llm.model.return_value.get_chunked_prefill_size.return_value = native_chunk
            with patch.dict(os.environ, environment or {}, clear=True), \
                    patch.dict(sys.modules, {"ftllm": fake}), \
                    patch("fastllm_pytools.util._has_cuda_device", return_value=True), \
                    redirect_stdout(io.StringIO()):
                make_normal_llm_model(args)
                self.native_env = dict(os.environ)
            fake.llm.set_cuda_slab.assert_called_once_with(args.cuda_slab)
            self.native = fake.llm
            self.loaded_model = fake.llm.model.return_value
            self.model_options = fake.llm.model.call_args.kwargs
            return args

    def test_pd_prefill_warmup_receives_the_scheduler_limits(self):
        limits = dict(role="prefill", model_id="test", transport="rdma",
                      listen="127.0.0.1:1234", chunk_tokens=32768,
                      max_requests_per_rank=16, max_new_tokens_per_rank=32768, max_workspace_mib_per_rank=6144)
        self.configure(["--tp", "8", "--attn-dp-size", "8", "--pd", json.dumps(limits)])
        self.loaded_model.warmup.assert_called_once_with(pd_prefill=limits)
        self.loaded_model.configure_pd.assert_called_once_with(**limits)
        decode = dict(role="decode", model_id="test", transport="rdma", prefill=["127.0.0.1:1234"])
        self.configure(["--tp", "8", "--attn-dp-size", "8", "--pd", json.dumps(decode)])
        self.loaded_model.warmup.assert_called_once_with()

    def test_dspark_parallel_placement_before_warmup(self):
        with tempfile.TemporaryDirectory() as draft:
            (Path(draft) / "config.json").write_text(json.dumps({"architectures": ["DSparkDraftModel"], "block_size": 7}))
            target = ["--draft", draft, "--tp", "4", "--attn-dp-size", "4"]
            pd = target + ["--pd", '{"role":"decode"}']
            for opts, legacy, expected in (
                    (pd, None, (4, True)),
                    (pd, "mlp:4", None),
                    (pd + ["--dspark_dp_size", "4"], "1", (4, True)),
                    (target + ["--dspark_tp_size", "1"], "dp:4", (1, False)),
                    (target, None, None)):
                with self.subTest(opts=opts, legacy=legacy):
                    env = {"FASTLLM_DSPARK_TP": legacy} if legacy else {}
                    self.configure(opts, environment=env)
                    self.assertEqual(self.native_env.get("FASTLLM_DSPARK_TP"), legacy)
                    configure = self.loaded_model.configure_dspark_parallel
                    if expected is None:
                        configure.assert_not_called()
                    else:
                        configure.assert_called_once_with(expected[0], replicas=expected[1])
                        calls = [call[0] for call in self.loaded_model.mock_calls]
                        self.assertLess(calls.index("configure_dspark_parallel"), calls.index("warmup"))

    def test_pd_prefill_role_before_warmup(self):
        for role in ("prefill", "decode"):
            self.configure(["--tp", "4", "--pd", json.dumps({"role": role})])
            self.assertEqual(self.model_options["pd_prefill"], role == "prefill")
            warmup = self.loaded_model.warmup
            if role == "prefill":
                warmup.assert_called_once_with(pd_prefill=True)
            else:
                warmup.assert_called_once_with()
            order = [call[0] for call in self.loaded_model.mock_calls]
            self.assertLess(order.index("warmup"), order.index("configure_pd"))

    def test_naive_host_prefix_cache(self):
        self.configure(['--tp', '8'])
        self.loaded_model.set_save_history.assert_called_once_with(True)
        self.loaded_model.configure_host_prefix_cache.assert_not_called()
        self.native.set_cpu_historycache.assert_not_called()
        self.configure(['--tp', '8', '--prefix_cache_l2_bytes', '8589934592'])
        self.loaded_model.configure_host_prefix_cache.assert_called_once_with(8589934592)
        names = [call[0] for call in self.loaded_model.mock_calls]
        self.assertLess(names.index('set_save_history'), names.index('warmup'))
        self.assertLess(names.index('warmup'), names.index('configure_host_prefix_cache'))
        with self.assertRaisesRegex(ValueError, 'L2|l2'):
            self.configure(['--tp', '8', '--cache_history', 'false', '--prefix_cache_l2_bytes', '1'])

    def test_saved_history_booleans(self):
        for value in (True, False):
            self.configure([], saved=dict(tp='8', cache_history=value))
            self.loaded_model.set_save_history.assert_called_once_with(value)
        with self.assertRaisesRegex(ValueError, 'L2'):
            self.configure([], saved=dict(tp='8', cache_history=False, prefix_cache_l2_bytes=4096))

    def test_split_host_prefix_budget(self):
        self.configure(['--tp', '8', '--prefix_cache_l2_kv_bytes', '8192', '--prefix_cache_l2_swa_bytes', '4096'])
        self.loaded_model.configure_host_prefix_cache.assert_called_once_with(0, kv_bytes=8192, swa_bytes=4096)
        self.configure([], saved=dict(tp='8', prefix_cache_l2_bytes=8192, prefix_cache_l2_swa_bytes=0))
        self.loaded_model.configure_host_prefix_cache.assert_called_once_with(8192, kv_bytes=None, swa_bytes=0)
        self.configure([], saved=dict(tp='8', cache_history=False, prefix_cache_l2_bytes=8192,
                                      prefix_cache_l2_swa_bytes=0, prefix_cache_l2_kv_bytes=0))
        self.loaded_model.configure_host_prefix_cache.assert_not_called()
        for opts in (['--prefix_cache_l2_swa_bytes', '-1'],
                     ['--prefix_cache_l2_bytes', '4', '--prefix_cache_l2_kv_bytes', '5']):
            with self.assertRaisesRegex(ValueError, 'L2'):
                self.configure(['--tp', '8', *opts])

    def test_fp8_compute_cli_and_saved_settings(self):
        self.configure([])
        self.assertEqual(self.model_options['indexer_compute_dtype'], 'auto')
        self.assertNotIn('FASTLLM_CUDA_NATIVE_FP8_PREFILL', self.native_env)
        for mode in ('fp8', 'fp8_direct'):
            for saved in (False, True):
                settings = {'indexer_compute_dtype': mode, 'fp8_prefill': 'true'}
                options = [] if saved else [part for k, v in settings.items() for part in ('--' + k, v)]
                self.configure(options, saved=settings if saved else None)
                self.assertEqual(self.model_options['indexer_compute_dtype'], mode)
                self.assertEqual(self.native_env['FASTLLM_CUDA_NATIVE_FP8_PREFILL'], '1')

    def test_fp8_prefill_environment_and_saved_booleans(self):
        name = 'FASTLLM_CUDA_NATIVE_FP8_PREFILL'
        self.configure(['--fp8_prefill', 'auto'], environment={name: '1'})
        self.assertEqual(self.native_env[name], '1')
        for value in ('true', 'false', True, False):
            with self.subTest(value=value):
                self.configure([], saved={'fp8_prefill': value}, environment={name: '1'})
                self.assertEqual(self.native_env[name], '1' if str(value).lower() == 'true' else '0')
        self.configure(['--fp8_prefill', 'false'], environment={name: '1'})
        self.assertEqual(self.native_env[name], '0')

    def test_fp8_prefill_rejects_invalid_saved_value(self):
        with self.assertRaisesRegex(ValueError, 'fp8_prefill'):
            self.configure([], saved={'fp8_prefill': 'invalid'})

    def test_indexer_tp_storage(self):
        self.configure(['--tp', '8'])
        self.assertEqual(self.model_options['indexer_tp_storage'], 'replicated')
        for options, saved in [(['--tp', '8', '--indexer_tp_storage', 'layer_shard'], None),
                               ([], {'tp': '4', 'indexer_tp_storage': 'layer_shard'})]:
            with self.subTest(saved=saved):
                self.configure(options, saved=saved)
                self.assertEqual(self.model_options['indexer_tp_storage'], 'layer_shard')
        for options, model, saved in [([], 'naive_n05_flash', None),
                                      (['--tp', '1'], 'naive_n05_flash', None),
                                      (['--tp', '8'], 'llama', None),
                                      (['--tp', '8'], 'naive_n05_flash', {'indexer_tp_storage': 'invalid'})]:
            with self.subTest(options=options, model=model), self.assertRaisesRegex(ValueError, '[Ii]ndexer'):
                self.configure(options + ([] if saved else ['--indexer_tp_storage', 'layer_shard']),
                               model_type=model, saved=saved)

    def test_context_sharded_indexer_with_gpu_kv_or_kvmem(self):
        for saved in (False, True):
            for kvmem in (False, True):
                settings = {'tp': '4', 'indexer_tp_storage': 'context_shard'}
                if kvmem:
                    settings['kvmem'] = '{}'
                options = [] if saved else [v for k, value in settings.items() for v in ('--' + k, value)]
                self.configure(options, saved=settings if saved else None)
                self.assertEqual(self.model_options['indexer_tp_storage'], 'context_shard')
                self.assertEqual(self.loaded_model.set_kvmem.call_count, int(kvmem))
        for options, model in [(['--tp', '1'], 'naive_n05_flash'),
                               (['--tp', '1', '--kvmem', '{}'], 'naive_n05_flash'),
                               (['--tp', '4', '--kvmem', '{}'], 'llama')]:
            with self.subTest(options=options, model=model), self.assertRaisesRegex(ValueError, '[Ii]ndexer'):
                self.configure(options + ['--indexer_tp_storage', 'context_shard'], model_type=model)

    def test_prefill_batching_cli_and_saved_settings(self):
        for options, saved in [(['--tp', '8', '--prefill_batch_tokens', '16384',
                                  '--prefill_max_wait_ms', '20', '--chunked_prefill_size', '8192'], None),
                               ([], {'tp': '8', 'prefill_batch_tokens': 16384,
                                     'prefill_max_wait_ms': 20, 'chunked_prefill_size': 8192})]:
            with self.subTest(saved=saved):
                self.configure(options, saved=saved)
                self.loaded_model.set_prefill_batching.assert_called_once_with(16384, 20)
                self.loaded_model.set_chunked_prefill_size.assert_called_once_with(8192)

    def test_prefill_batching_defaults_and_invalid_values(self):
        self.configure(['--tp', '8'])
        self.loaded_model.set_prefill_batching.assert_not_called()
        for options in [['--prefill_batch_tokens', '-1'], ['--prefill_max_wait_ms', '10'],
                        ['--prefill_batch_tokens', '8192', '--prefill_max_wait_ms', '-1']]:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'prefill'):
                self.configure(options)
        with self.assertRaisesRegex(ValueError, 'prefill'):
            self.configure([], saved={'prefill_batch_tokens': 2147483648})

    def test_attention_dp_default_budget_and_wait(self):
        for world, dp, chunk, limit, wait, expected in (
                (4, 4, 4096, 32, 0, 16384), (8, 8, 4096, 64, 0, 32768),
                (8, 4, 8192, -1, 0, 32768), (8, 8, 4096, 2, 0, 8192),
                (8, 8, 4096, 64, 2000, 32768)):
            with self.subTest(world=world, dp=dp, chunk=chunk, limit=limit, wait=wait):
                args = self.configure(['--tp', str(world), '--attn-dp-size', str(dp),
                                       '--max_batch', str(limit), '--chunked_prefill_size', str(chunk),
                                       '--prefill_max_wait_ms', str(wait)],
                                      native_chunk=chunk)
                self.assertEqual(args.prefill_batch_tokens, expected)
                self.loaded_model.set_prefill_batching.assert_called_once_with(expected, wait)
                names = [call[0] for call in self.loaded_model.mock_calls]
                self.assertLess(names.index('set_chunked_prefill_size'), names.index('get_chunked_prefill_size'))
                self.assertLess(names.index('set_prefill_batching'), names.index('warmup'))

    def test_attention_dp_preserves_explicit_batch_budget(self):
        for budget in (0, 8192):
            for saved in (False, True):
                with self.subTest(budget=budget, saved=saved):
                    settings = dict(tp='4', attn_dp_size=4, prefill_batch_tokens=budget)
                    options = ['--tp', '4', '--attn-dp-size', '4', '--prefill_batch_tokens', str(budget)]
                    args = self.configure([] if saved else options, saved=settings if saved else None)
                    self.assertEqual(args.prefill_batch_tokens, budget)
                    self.loaded_model.get_chunked_prefill_size.assert_not_called()
                    if budget:
                        self.loaded_model.set_prefill_batching.assert_called_once_with(budget, 0)
                    else:
                        self.loaded_model.set_prefill_batching.assert_not_called()

    def test_attention_dp_default_budget_with_scheduling(self):
        settings = dict(tp='4', attn_dp_size=4, prefill_batch_requests=8,
                        prefill_max_requests=16, prefill_max_wait_ms=2000,
                        prefill_dp_balance=True, prefill_chunked=True,
                        prefill_chunk_max_wait_ms=100, prefill_fairness_ms=100)
        self.configure([], saved=settings)
        self.loaded_model.set_prefill_batching.assert_called_once_with(
            32768, 2000, target_requests=8, max_requests=16, min_decode_requests=0)
        self.loaded_model.set_prefill_scheduling.assert_called_once_with(
            balance_dp=True, adaptive_requests=0, chunked=True,
            chunk_max_wait_ms=100, fairness_ms=100)
        for override in ({'prefill_batch_tokens': 0}, {'attn_dp_size': 1}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, 'prefill'):
                self.configure([], saved={**settings, **override})

    def test_kv_admission_reserve(self):
        settings = ['--tp', '4', '--prefill_batch_tokens', '16384', '--prefill_chunked', 'true']
        self.configure(settings)
        self.loaded_model.set_kv_admission_reserve.assert_not_called()
        for reserve in (0, 1024, 4096):
            self.configure(settings + ['--kv_admission_reserve', str(reserve)])
            self.loaded_model.set_kv_admission_reserve.assert_called_once_with(reserve)
        self.configure([], saved={'tp': '4', 'kv_admission_reserve': 0})
        self.loaded_model.set_kv_admission_reserve.assert_called_once_with(0)
        for options in (['--kv_admission_reserve', '4096'], settings + ['--kv_admission_reserve', '-1']):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'kv_admission_reserve'):
                self.configure(options)

    def test_attention_dp_saved_default_and_kvmem_chunk(self):
        args = self.configure([], saved=dict(tp='4', attn_dp_size=4,
                              kvmem='{"prefill_tokens": 2048}'), native_chunk=2048)
        self.assertEqual(args.prefill_batch_tokens, 8192)
        self.loaded_model.set_prefill_batching.assert_called_once_with(8192, 0)
        names = [call[0] for call in self.loaded_model.mock_calls]
        self.assertLess(names.index('set_kvmem'), names.index('get_chunked_prefill_size'))

    def test_deepseek_attention_dp_uses_native_chunk(self):
        self.configure(['--tp', '8', '--attn-dp-size', '8'],
                       model_type='deepseek_v41', native_chunk=4096)
        self.loaded_model.set_prefill_batching.assert_called_once_with(32768, 0)

    def test_attention_dp_default_budget_rejects_overflow(self):
        with self.assertRaisesRegex(ValueError, 'prefill budget'):
            self.configure(['--tp', '8', '--attn-dp-size', '8'], native_chunk=2147483647)

    def test_prefill_coalescing_targets_and_caps(self):
        values = {'prefill_batch_tokens': 16384, 'prefill_max_wait_ms': 20000,
                  'prefill_batch_requests': 16, 'prefill_max_requests': 24,
                  'prefill_min_decode_requests': 8}
        for saved in (False, True):
            options = [] if saved else [part for k, v in values.items() for part in ('--' + k, str(v))]
            self.configure(options, saved=values if saved else None)
            self.loaded_model.set_prefill_batching.assert_called_once_with(
                16384, 20000, target_requests=16, max_requests=24, min_decode_requests=8)
        for changes in ({'prefill_batch_requests': -1}, {'prefill_max_requests': 15},
                        {'prefill_min_decode_requests': -1}, {'prefill_batch_tokens': 0},
                        {'prefill_max_requests': 2147483648}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'prefill'):
                self.configure([], saved={**values, **changes})

    def test_prefill_scheduling_cli_and_saved_settings(self):
        values = dict(prefill_batch_tokens=32768, prefill_dp_balance='true',
                      prefill_adaptive_requests=256, prefill_chunked='true',
                      prefill_chunk_max_wait_ms=1000, prefill_fairness_ms=1000)
        for saved in (False, True):
            options = [] if saved else [p for k, v in values.items() for p in ('--' + k, str(v))]
            self.configure(options, saved=values if saved else None)
            self.loaded_model.set_prefill_scheduling.assert_called_once_with(balance_dp=True,
                adaptive_requests=256, chunked=True, chunk_max_wait_ms=1000, fairness_ms=1000)
        self.configure([])
        self.loaded_model.set_prefill_scheduling.assert_not_called()
        for changes in ({'prefill_batch_tokens': 0}, {'prefill_chunked': 'false'},
                        {'prefill_adaptive_requests': -1}, {'prefill_dp_balance': 'invalid'},
                        {'prefill_chunk_max_wait_ms': True}, {'prefill_fairness_ms': 2147483648}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'prefill'):
                self.configure([], saved={**values, **changes})

    def test_tp_defaults_and_explicit_slab(self):
        for ranks in (2, 4, 8):
            for slab in (None, 0, 32):
                with self.subTest(ranks=ranks, slab=slab):
                    options = ["--tp", str(ranks)]
                    if slab is not None:
                        options += ["--cuda_slab", str(slab)]
                    args = self.configure(options)
                    self.assertEqual(args.cuda_slab, 16 if slab is None else slab)
                    self.assertEqual(args.atype, "bfloat16")

    def test_saved_explicit_zero(self):
        args = self.configure([], saved={"tp": "8", "cuda_slab": 0})
        self.assertEqual(args.cuda_slab, 0)

    def test_mimo_dsa_tp_defaults(self):
        args = self.configure(["--tp", "8"], model_type="mimo_v2", config={
            "architectures": ["MiMoV2ForCausalLM"], "enable_dsa": True})
        self.assertEqual(args.atype, "bfloat16")
        self.assertEqual(args.cuda_slab, 16)
        self.assertEqual(args.cache_history, "true")

    def test_mimo_family_requires_dsa(self):
        for enabled in (None, False, "true"):
            self.assertFalse(_is_naive_n05_family(
                {"model_type": "mimo_v2", "enable_dsa": enabled}))
        self.assertTrue(_is_naive_n05_family({
            "architectures": ["MiMoV2DSAForCausalLM"], "enable_dsa": True}))

    def test_serial_and_single_device_defaults(self):
        for options in ([], ["--tp", "1"], ["--device", '["cuda:0", "cuda:1"]']):
            with self.subTest(options=options):
                self.assertEqual(self.configure(options).cuda_slab, 0)

    def test_other_model_defaults(self):
        args = self.configure(["--tp", "2"], model_type="llama")
        self.assertEqual(args.cuda_slab, 0)
        self.assertEqual(args.atype, "float16")

    def test_ep_cli_and_saved_settings(self):
        for ep in (4, 8):
            for options, saved in [(["--tp", str(ep), "--ep", str(ep)], None),
                                   ([], {"tp": str(ep), "ep": ep})]:
                with self.subTest(ep=ep, options=options, saved=saved):
                    args = self.configure(options, saved=saved)
                    self.assertEqual(args.ep, ep)
                    self.assertEqual(self.native_env["FASTLLM_EP"], str(ep))
                    self.assertEqual(self.native_env["FASTLLM_TP"], "cuda:" + ",".join(map(str, range(ep))))

    def test_ep_explicit_disable_overrides_environment(self):
        self.configure(["--tp", "8", "--ep", "1"], environment={
            "FASTLLM_EP": "8"})
        self.assertEqual(self.native_env["FASTLLM_EP"], "1")

    def test_ep_environment_and_explicit_enable(self):
        args = self.configure(["--tp", "8"], environment={"FASTLLM_EP": "8"})
        self.assertEqual(args.ep, 8)
        self.configure(["--tp", "8", "--ep", "8"], environment={"FASTLLM_EP": "1"})
        self.assertEqual(self.native_env["FASTLLM_EP"], "8")

    def test_ep_requires_matching_tp_devices(self):
        for ep, sizes in ((4, (None, "1", "8")), (8, (None, "1", "4"))):
            for tp in sizes:
                with self.subTest(ep=ep, tp=tp), self.assertRaisesRegex(ValueError, f"requires exactly {ep}"):
                    self.configure(["--ep", str(ep)] + (["--tp", tp] if tp else []))

    def test_ep_rejects_unsupported_model_and_saved_value(self):
        with self.assertRaisesRegex(ValueError, "currently supports MiMo"):
            self.configure(["--tp", "8", "--ep", "8"], model_type="llama")
        with self.assertRaisesRegex(ValueError, "supports 1"):
            self.configure([], saved={"tp": "8", "ep": 2})

    def test_tp_does_not_enable_ep_implicitly(self):
        args = self.configure(["--tp", "8"])
        self.assertIsNone(args.ep)
        self.assertNotIn("FASTLLM_EP", self.native_env)


if __name__ == "__main__":
    unittest.main()

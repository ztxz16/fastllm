"""Offline tuner safety, independent cache accounting, and legacy-auto isolation."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import autotune as tune, auto_config as auto, hybrid_auto as memory
from fastllm_pytools.env import env
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model

GiB, MiB = memory.GiB, memory.MiB


def measurement(score, spread=0, **changes):
    samples = [{"input_hash": str(i), "input_tokens": 4096, "output_tokens": 512,
                "cached_input_tokens": 0, "ttft_seconds": 1., "decode_seconds_per_token": .1,
                "score_seconds": score + delta, **changes} for i, delta in enumerate((-spread, spread))]
    return {"score_seconds": score, "samples": samples}


class AutotuneTest(unittest.TestCase):
    def setUp(self):
        self.g = memory.Geometry("deepseek_v41", 8, 65536, 128, 128, 8, 64,
                                 [1024] * 8, [MiB] * 8, 4, 1024, 1024)
        self.costs = ([4 * GiB] * 8, [MiB] * 8, MiB, MiB, 4 * GiB, False)
        self.gpu = {d: {"total": 8 * GiB, "free": 8 * GiB} for d in (0, 1)}
        self.ram = {"total": 24 * GiB, "available": 24 * GiB}
        info = {d: {"sm": 40, "clock_khz": 1500000, "memory_bandwidth": 400e9,
                    "link_bandwidth": 12e9, "capability": 75} for d in (0, 1)}
        self.hardware = info, {(a, b): True for a in info for b in info}
        self.a = make_normal_parser("test").parse_args(["model", "--auto", "--dspark", "0"])
        self.a.device, self.a.tp, self.a.moe_device = "cudapp=0,1", "off", "disk"
        self.a.threads, self.a.tokens, self.a.max_context_length, self.a.max_batch = 4, 4096, 4096, 1
        self.a.low_gpu_mem = True
        self.a.moe_device_layers, self.a.chunked_prefill_size = -1, 4096
        self.a.moe_cpu_cache, self.a.moe_cuda_cache = "auto", "auto"
        self.a.cuda_slab = 0

    def baseline(self):
        p = memory.plan(self.a, self.g, self.costs, "pp", {0: 1, 1: 1}, self.gpu, self.ram)
        p["automatic"] = {"cuda_graph": True}
        for k in memory.AUTO_OPTIONS:
            if getattr(self.a, k) == "auto":
                setattr(self.a, k, p[k])
        self.a.gpu_mem_ratio = p["gpu_mem_ratio"]
        return p

    def pool(self, baseline):
        with patch.object(memory, "weight_costs", return_value=self.costs), \
                patch.object(env, "_build_info", {"USE_NUMAS": True}), patch.dict(os.environ, {}, clear=True):
            return tune.candidates(self.a, baseline, {}, [], self.g, self.gpu, self.ram)[0]

    def test_parser_is_opt_in(self):
        old = make_normal_parser("t").parse_args(["m", "--auto"])
        new = make_normal_parser("t").parse_args(["m", "--autotune", "--autotune-budget", "60"])
        self.assertFalse(old.autotune)
        self.assertTrue(new.autotune)
        self.assertFalse(new.auto)
        self.assertEqual(new.autotune_budget, 60)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                make_normal_parser("t").parse_args(["--autotune-budget", "0"])
        # Saved JSON settings bypass argparse's value converter.
        for value in (0, -1, float("inf"), "1800"):
            new.autotune_budget = value
            with self.assertRaisesRegex(ValueError, "positive integer"):
                tune.resolve(new, None)
        from fastllm_pytools.cli import save_defaults_to_json
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            path = Path(d) / "config.json"
            save_defaults_to_json(make_normal_parser("test"), str(path))
            saved = json.loads(path.read_text())
            self.assertEqual(saved["autotune_budget"], 1800)
            self.assertIn("autotune_profile", saved)
            self.assertIn("autotune_prompt", saved)

    def test_forced_host_budget_survives_candidate_planning_and_live_recheck(self):
        self.a.force_ddr_limit = '32g'
        baseline = self.baseline()
        pool = self.pool(baseline)
        self.assertGreater(len(pool), 1)
        for candidate in pool:
            self.assertEqual(candidate['plan']['host_budget_bytes'], 32 * GiB)
            with patch.object(memory, 'host_memory', return_value={'total': GiB, 'available': 0}):
                self.assertTrue(tune.fits(candidate, self.a, self.gpu.__getitem__))
                self.assertFalse(tune.fits(candidate, self.a, lambda d: {'total': 8 * GiB, 'free': GiB}))
        self.a.force_ddr_limit = baseline['host_estimated_bytes'] - 1
        with patch.object(memory, 'host_memory', return_value=self.ram):
            self.assertFalse(tune.fits(pool[0], self.a, self.gpu.__getitem__))

    def test_candidates_keep_baseline_and_account_for_both_gpu_pools(self):
        p = self.baseline()
        before = copy.deepcopy(vars(self.a)), copy.deepcopy(p)
        pool = self.pool(p)
        mixed = [x for x in pool if x["placement"]]
        self.assertTrue(mixed)
        self.assertEqual(pool[0]["plan"], before[1])
        self.assertEqual((vars(self.a), p), before)
        for item in pool:
            plan = item["plan"]
            self.assertLessEqual(plan["host_estimated_bytes"], plan["host_budget_bytes"])
            for d in self.gpu:
                self.assertLessEqual(plan["gpu_estimated_bytes"][d], plan["gpu_budget_bytes"][d])
        positive = [x for x in mixed if max(x["host_gpu_cache"].values()) > 0]
        self.assertTrue(positive)
        for item in positive:
            self.assertLessEqual(max(item["host_gpu_cache"].values()), item["settings"]["moe_cuda_cache"])
            self.assertEqual(sum(item["placement"]["front"].values()) + item["placement"]["disk_layers"], self.g.layers)
            # Disk pool was charged on top of the normal NUMA solver's CUDA cache.
            disk = item["settings"]["moe_cuda_cache"]
            self.assertTrue(all(n >= disk + item["host_gpu_cache"][d]
                                for d, n in item["plan"]["gpu_estimated_bytes"].items()))

    def test_every_explicit_placement_option_freezes_backend_and_caches(self):
        p = self.baseline()
        self.a._auto_explicit += [*tune.TUNABLE]
        pool = self.pool(p)
        self.assertTrue(all(not x["placement"] for x in pool))
        self.assertTrue(all(x["settings"] == pool[0]["settings"] for x in pool))

    def test_mixed_auto_baseline_keeps_gpu_prefix_in_new_candidates(self):
        mixed = next(x for x in self.pool(self.baseline()) if x['placement'])
        vars(self.a).update(mixed['settings'])
        pool = self.pool(mixed['plan'])
        self.assertEqual(pool[0]['numa_layers'], mixed['numa_layers'])
        self.assertEqual(pool[0]['placement'], mixed['placement'])
        self.assertTrue(all(x['gpu_layers'] == mixed['gpu_layers'] for x in pool))

    def test_gpu_front_mapping_matches_native_prefix_partition(self):
        self.assertEqual(tune._front_map(6, 26, {0: 1, 1: 1}), {"cuda:0": 3, "cuda:1": 3, "numa": 26})
        self.assertEqual(tune._front_map(0, 12, {0: 1, 1: 1}), {"numa": 12})
        # Draft experts are charged according to their target layer.
        self.g.kv += [1024, 1024]
        self.assertEqual(memory.expert_layer(8, self.g), 6)

    def test_history_only_reorders_feasible_candidates(self):
        pool = self.pool(self.baseline())
        tune.seed_estimates(pool, self.a, self.g, self.costs, self.hardware)
        predictor = tune.cost.CostModel(self.g.layers, {0: 1, 1: 1})
        chosen, ranked = tune.cost.select_next(pool, [], predictor, {})
        self.assertEqual(chosen, pool[0])
        self.assertEqual(len(ranked), len(pool))
        other = pool[-1]
        chosen, ranked = tune.cost.select_next(pool, [pool[0]], predictor,
                                               {other["history_key"]: 1., "stale": .1})
        self.assertEqual(next(p for p in ranked if p['id'] == other['id'])['seconds'], 1.)
        self.assertEqual(chosen['numa_layers'], max(p['numa_layers'] for p in pool))
        cuts = {p['numa_layers'] for p in pool if p['numa_layers']}
        self.assertEqual(cuts, set(range(1, max(cuts)+1)))

    def test_budgeted_resolve_measures_more_than_three_candidates(self):
        baseline = self.baseline()
        pool = self.pool(baseline)[:5]
        self.assertEqual(len(pool), 5)
        args = copy.deepcopy(self.a)
        args.autotune = True
        args.autotune_budget = 1800
        with tempfile.TemporaryDirectory() as d:
            args.autotune_profile = str(Path(d)/'profile.json')
            calls = []
            def probe(*a):
                calls.append(a[1]['id'])
                return measurement(100-len(calls)*5)
            with patch.object(tune.sys, 'platform', 'linux'), \
                    patch.object(memory, 'model_inputs', return_value=({},[],self.g)), \
                    patch.object(memory, 'host_memory', return_value=self.ram), \
                    patch.object(auto, 'cuda_hardware', return_value=self.hardware), \
                    patch.object(auto, 'resolve', return_value=baseline), \
                    patch.object(tune, 'candidates', return_value=(pool,self.costs)) as plan, \
                    patch.object(tune, 'fingerprint', return_value='test'), \
                    patch.object(tune, 'fits', return_value=True), \
                    patch.object(tune, 'probe', side_effect=probe), redirect_stdout(io.StringIO()):
                tune.resolve(args, self.gpu.__getitem__)
            self.assertEqual(plan.call_args.args[-2][0]['capability'], 75)
            report = json.loads(Path(args.autotune_profile).read_text())
            self.assertEqual(len(calls),5)
            self.assertEqual(len(set(calls)),5)
            self.assertEqual(len(report['predictions']),5)
            self.assertEqual(report['stop_reason'],'all feasible candidates tested')
            self.assertFalse(report['baseline_retained'])

    def test_no_clear_gain_keeps_verified_baseline(self):
        baseline = {"id": "base", "measurement": measurement(100)}
        for candidate in ({"id": "failed", "error": "OOM"},
                          {"id": "small", "measurement": measurement(97)},
                          {"id": "noisy", "measurement": measurement(90, 15)},
                          {"id": "cached", "measurement": measurement(50, cached_input_tokens=128)},
                          {"id": "short", "measurement": measurement(50, output_tokens=4)},
                          {"id": "different", "measurement": measurement(50, input_hash="wrong")}):
            self.assertIs(tune.choose_winner([baseline, candidate]), baseline)
        fast = {"id": "fast", "measurement": measurement(80, 1)}
        self.assertIs(tune.choose_winner([baseline, fast]), fast)
        with self.assertRaises(RuntimeError):
            tune.choose_winner([{"error": "timeout"}, fast])
        with self.assertRaises(RuntimeError):
            tune.choose_winner([])

    def test_profile_corruption_invalidation_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "profile.json"
            for content in ("{", "null", "[]", '{"schema":1,"fingerprint":"key","trials":[1]}'):
                path.write_text(content)
                self.assertEqual(tune.read_history(path, "key"), [])
            tune._write(path, {"schema": tune.SCHEMA, "fingerprint": "key",
                               "trials": [{"history_key": "good", "measurement": measurement(10)}]})
            history = tune.read_history(path, "key")
            self.assertEqual([(t['history_key'],t['measurement']['score_seconds']) for t in history], [('good',10)])
            self.assertEqual(tune.read_history(path, "changed-code-model-or-settings"), [])
            self.assertEqual(path.stat().st_mode & 0o077, 0)

    def test_recheck_rejects_decreased_host_and_gpu_memory(self):
        p = self.baseline()
        candidate = tune._candidate(self.a, p)
        with patch.object(memory, "host_memory", return_value=self.ram):
            self.assertTrue(tune.fits(candidate, self.a, self.gpu.__getitem__))
            self.assertFalse(tune.fits(candidate, self.a, lambda d: {"total": 8 * GiB, "free": GiB}))
        with patch.object(memory, "host_memory", return_value={"total": 24 * GiB, "available": 2 * GiB}):
            self.assertFalse(tune.fits(candidate, self.a, self.gpu.__getitem__))

    def test_auto_entry_applies_mixed_plan_without_calling_tuner(self):
        self.ram = {'total': 32 * GiB, 'available': 30 * GiB}
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
            args = make_normal_parser("test").parse_args([d, "--auto", "--dspark", "0"])
            runtime = MagicMock()
            runtime.get_cuda_memory_info.side_effect = self.gpu.__getitem__
            runtime.model.return_value.get_max_input_len.return_value = 4096
            runtime.model.return_value.get_max_batch.return_value = 1
            with patch.object(memory, "model_inputs", return_value=({}, [], self.g)), \
                    patch.object(memory, "geometry", return_value=self.g), \
                    patch.object(memory, "weight_costs", return_value=self.costs), \
                    patch.object(memory, "host_memory", return_value=self.ram), \
                    patch.object(auto, "cuda_hardware", return_value=self.hardware), \
                    patch.object(env, "_build_info", {"USE_NUMAS": True}), \
                    patch.dict(sys.modules, {"ftllm": types.SimpleNamespace(llm=runtime)}), \
                    patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()), \
                    patch.object(tune, "resolve", side_effect=AssertionError("old auto entered tuner")):
                expected_args = copy.deepcopy(args)
                expected_args.cuda_slab = 0
                expected = auto.resolve(expected_args, self.gpu.__getitem__)
                model = make_normal_llm_model(args)
            self.assertEqual(model.hybrid_auto_plan, expected)
            placement = expected['autotune_placement']
            runtime.set_device_map.assert_any_call(placement['front'], True)
            runtime.set_moe_device_layers.assert_called_with(placement['disk_layers'])
            runtime.set_moe_cuda_cache.assert_called_once_with(
                expected['moe_cuda_cache'], expected['moe_cuda_cache_per_device'])
            runtime.model.assert_called_once()

    def test_tuned_mapping_and_cache_overrides_reach_native_before_load(self):
        p = self.baseline()
        candidate = next(x for x in self.pool(p) if x["placement"] and max(x["host_gpu_cache"].values()) > 0)
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
            a = copy.deepcopy(self.a)
            a.model = a.path = d
            a.auto = a.autotune = False
            a._autotune_plan = candidate["plan"]
            vars(a).update(candidate["settings"])
            runtime = MagicMock()
            model = runtime.model.return_value
            model.get_max_input_len.return_value, model.get_max_batch.return_value = 4096, 1
            def loaded(*args, **kwargs):
                runtime.set_device_map.assert_any_call(candidate["placement"]["front"], True)
                runtime.set_layered_moe_device_map.assert_called_with("disk")
                runtime.set_moe_device_layers.assert_called_with(candidate["placement"]["disk_layers"])
                runtime.set_moe_cuda_cache.assert_called_once_with(a.moe_cuda_cache, candidate["host_gpu_cache"])
                return model
            runtime.model.side_effect = loaded
            with patch.dict(sys.modules, {"ftllm": types.SimpleNamespace(llm=runtime)}), \
                    patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                make_normal_llm_model(a)

    def test_timeout_kills_worker_and_removes_private_arguments(self):
        p = self.baseline()
        child = MagicMock(pid=12345)
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as d, \
                patch.object(tune.subprocess, "Popen", return_value=child), \
                patch.object(tune.os, "killpg") as kill, redirect_stdout(io.StringIO()):
            directory = Path(d) / "trial"
            with self.assertRaises(TimeoutError):
                tune.probe(self.a, tune._candidate(self.a, p), "prompt", directory, 0)
            kill.assert_called_once_with(child.pid, tune.signal.SIGTERM)
            child.wait.assert_called_once()
            self.assertFalse((directory / "input.json").exists())

    def test_parent_exit_cannot_leave_detached_model_worker(self):
        libc = MagicMock()
        libc.prctl.return_value = 0
        with patch.object(tune.ctypes, "CDLL", return_value=libc), patch.object(os, "getppid", return_value=42):
            tune._parent_lifetime(42)
            libc.prctl.assert_called_with(1, tune.signal.SIGKILL, 0, 0, 0)
            with self.assertRaisesRegex(RuntimeError, "parent exited"):
                tune._parent_lifetime(41)

    def test_large_worker_teardown_waits_after_kill(self):
        child = MagicMock(pid=12345)
        child.poll.return_value = None
        child.wait.side_effect = [tune.subprocess.TimeoutExpired("worker", 5), 0]
        with patch.object(tune.os, "killpg") as kill:
            tune._stop(child)
        self.assertEqual([c.kwargs["timeout"] for c in child.wait.call_args_list], [5, 30])
        self.assertEqual([c.args[1] for c in kill.call_args_list], [tune.signal.SIGTERM, tune.signal.SIGKILL])

    def test_exited_leader_does_not_skip_live_helper_cleanup(self):
        child = MagicMock(pid=12345)
        child.poll.return_value = 0
        with patch.object(tune, '_group_alive', side_effect=[True, False]), \
                patch.object(tune.os, 'killpg') as kill:
            tune._stop(child)
        self.assertEqual([c.args[1] for c in kill.call_args_list], [tune.signal.SIGTERM, tune.signal.SIGKILL])

    def test_gguf_files_include_all_shards_and_engram(self):
        from fastllm_pytools import gguf_metadata
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shards = [root/('model-%05d-of-00002.gguf' % i) for i in (1,2)]
            for p in shards+[root/'engram.gguf']:
                p.write_bytes(b'fixture')
            metadata = {'split.count':2, 'deepseek41.engram.present':True, 'deepseek41.engram.file':'engram.gguf'}
            with patch.object(gguf_metadata, 'read_gguf_metadata', return_value=metadata):
                self.assertEqual(tune.checkpoint_files(shards[0]), [p.resolve() for p in shards+[root/'engram.gguf']])
                metadata['deepseek41.engram.file'] = '../external.gguf'
                with self.assertRaisesRegex(ValueError, 'adjacent'):
                    tune.checkpoint_files(shards[0])

    def test_standalone_gguf_uses_official_deepseek_prompt(self):
        model = MagicMock(hf_tokenizer=None, force_chat_template=False, enable_thinking=False)
        model._is_deepseek_v4.return_value = True
        model._deepseek_encode_messages.return_value.return_value = 'rendered'
        messages = [{'role':'user','content':'fixture'}]
        self.assertEqual(tune._render_prompt(model, messages), 'rendered')
        model._deepseek_encode_messages.return_value.assert_called_once_with(messages, thinking_mode='chat')
        model.force_chat_template = True
        tune._render_prompt(model, messages)
        model.apply_chat_template.assert_called_once_with(messages, add_generation_prompt=True)

    def test_unreaped_worker_is_fatal_and_private_input_is_removed(self):
        child = MagicMock(pid=12345)
        child.poll.return_value = None
        child.wait.side_effect = tune.subprocess.TimeoutExpired("worker", 30)
        with tempfile.TemporaryDirectory() as d, \
                patch.object(tune.subprocess, "Popen", return_value=child), \
                patch.object(tune.os, "killpg"), redirect_stdout(io.StringIO()):
            directory = Path(d) / "trial"
            with self.assertRaises(tune.WorkerCleanupError):
                tune.probe(self.a, tune._candidate(self.a, self.baseline()), "prompt", directory, 0)
            self.assertFalse((directory / "input.json").exists())

    def test_profile_identity_includes_format_code_and_speculation(self):
        with tempfile.TemporaryDirectory() as d:
            a = copy.deepcopy(self.a)
            a.path = d
            library = types.SimpleNamespace(fastllm_lib=types.SimpleNamespace(_name="native.so"))
            with patch.dict(sys.modules, {"fastllm_pytools.llm": library}), \
                    patch.object(tune, "_file_hash", return_value="code-v1") as digest, \
                    patch.object(Path, "read_text", return_value="model name: CPU\n"), \
                    patch.object(os, "sched_getaffinity", return_value={0, 1}, create=True):
                key = tune.fingerprint(a, [], self.hardware, "prompt")
                a.dspark = 5
                self.assertNotEqual(key, tune.fingerprint(a, [], self.hardware, "prompt"))
                a.dspark = 0
                digest.return_value = "code-v2"
                self.assertNotEqual(key, tune.fingerprint(a, [], self.hardware, "prompt"))
                digest.return_value = "code-v1"
                tensor = memory.Tensor("expert", (16, 16), 64, "gguf:19")
                self.assertNotEqual(key, tune.fingerprint(a, [tensor], self.hardware, "prompt"))

    def test_calibration_timeout_keeps_completed_measurements_and_reaps_worker(self):
        plan = self.baseline()
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as d:
            trial_dir = Path(d)/'trial'
            def launched(*a, **kw):
                (trial_dir/'result.json').write_text(json.dumps(measurement(10)))
                return process
            with patch.object(tune.subprocess,'Popen',side_effect=launched), \
                    patch.object(tune,'_stop') as stop:
                result = tune.probe(self.a,tune._candidate(self.a,plan),'prompt',trial_dir,0)
            self.assertEqual(result['score_seconds'],10)
            self.assertIn('calibration reached',result['calibration_error'])
            stop.assert_called_once_with(process)
            self.assertFalse((trial_dir/'input.json').exists())

    def test_cma_pressure_stops_worker_before_oom(self):
        p = self.baseline()
        child = MagicMock(pid=12345)
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as d, \
                patch.object(tune.subprocess, "Popen", return_value=child), \
                patch.object(tune.os, "killpg") as kill, \
                patch.object(memory, "host_memory", return_value={"total": 24 * GiB, "available": MiB}), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "safety floor"):
                tune.probe(self.a, tune._candidate(self.a, p), "prompt", Path(d) / "trial", float("inf"))
            kill.assert_called_once()


class TuningMemoryTest(unittest.TestCase):
    def layout(self, directory, available=49 * GiB):
        root = Path(directory)
        proc, mount = root / "proc", root / "cgroup"
        (proc / "self").mkdir(parents=True)
        child = mount / "parent/worker"
        child.mkdir(parents=True)
        (proc / "meminfo").write_text("MemTotal: %d kB\nMemAvailable: %d kB\nCmaTotal: %d kB\n" %
                                      (128 * GiB // 1024, available // 1024, 9 * GiB // 1024))
        (proc / "self/cgroup").write_text("0::/parent/worker\n")
        (proc / "self/mountinfo").write_text("1 0 0:1 / %s rw - cgroup2 cgroup rw\n" % mount)
        (child / "memory.max").write_text(str(96 * GiB))
        (child / "memory.current").write_text(str(96 * GiB))
        self.stats(child, file=24 * GiB, inactive_file=24 * GiB)
        return proc, child

    def stats(self, group, **values):
        (group / "memory.stat").write_text("".join("%s %d\n" % item for item in values.items()))

    def test_default_auto_credits_clean_file_cache(self):
        with tempfile.TemporaryDirectory() as d:
            proc, _ = self.layout(d)
            old = memory.host_memory(proc, reclaim_file_cache=False)
            tuned = memory.host_memory(proc)
            self.assertEqual(old, {"total": 96 * GiB, "available": 0, "cma": 9 * GiB})
            self.assertEqual(tuned, {"total": 96 * GiB, "available": 24 * GiB, "cma": 9 * GiB})
            self.assertEqual(memory.host_memory(proc), tuned)

    def test_active_mapped_file_pages_are_credited_but_not_shared_or_dirty(self):
        with tempfile.TemporaryDirectory() as d:
            proc, child = self.layout(d)
            self.stats(child, file=24 * GiB, inactive_file=24 * GiB,
                       file_mapped=4 * GiB, shmem=3 * GiB, file_dirty=2 * GiB, file_writeback=GiB)
            self.assertEqual(memory.host_memory(proc, reclaim_file_cache=True)["available"], 18 * GiB)
            self.stats(child, file=24 * GiB, inactive_file=0, active_file=24 * GiB)
            self.assertEqual(memory.host_memory(proc, reclaim_file_cache=True)["available"], 24 * GiB)

    def test_v1_hierarchical_file_cache_credit(self):
        with tempfile.TemporaryDirectory() as d:
            proc, child = self.layout(d)
            mount = child.parent.parent
            (proc / 'self/cgroup').write_text('3:memory:/parent/worker\n')
            (proc / 'self/mountinfo').write_text('1 0 0:1 / %s rw - cgroup cgroup rw,memory\n' % mount)
            (child / 'memory.limit_in_bytes').write_text(str(96 * GiB))
            (child / 'memory.usage_in_bytes').write_text(str(96 * GiB))
            self.stats(child, cache=GiB, total_cache=24 * GiB, total_shmem=3 * GiB,
                       total_dirty=2 * GiB, total_writeback=GiB)
            self.assertEqual(memory.host_memory(proc)['available'], 18 * GiB)

    def test_parent_usage_and_siblings_still_bound_available_memory(self):
        with tempfile.TemporaryDirectory() as d:
            proc, child = self.layout(d)
            parent = child.parent
            (parent / "memory.max").write_text(str(100 * GiB))
            (parent / "memory.current").write_text(str(99 * GiB))
            self.stats(parent, file=2 * GiB, inactive_file=2 * GiB)
            self.assertEqual(memory.host_memory(proc, reclaim_file_cache=True)["available"], 3 * GiB)

    def test_physical_cma_limit_remains_binding(self):
        with tempfile.TemporaryDirectory() as d:
            proc, _ = self.layout(d, available=14 * GiB)
            self.assertEqual(memory.host_memory(proc, reclaim_file_cache=True)["available"], 5 * GiB)

    def test_missing_or_invalid_counters_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            proc, child = self.layout(d)
            (child / "memory.stat").unlink()
            self.assertEqual(memory.host_memory(proc, reclaim_file_cache=True)["available"], 0)
            for text in ("file invalid\ninactive_file 100\n", "inactive_file 100\n"):
                (child / "memory.stat").write_text(text)
                self.assertEqual(memory.host_memory(proc, reclaim_file_cache=True)["available"], 0)

    def test_counter_sampling_race_never_exceeds_limit(self):
        with tempfile.TemporaryDirectory() as d:
            proc, child = self.layout(d, available=120 * GiB)
            (child / "memory.current").write_text(str(GiB))
            self.assertEqual(memory.host_memory(proc, reclaim_file_cache=True)["available"], 96 * GiB)


if __name__ == "__main__":
    unittest.main()

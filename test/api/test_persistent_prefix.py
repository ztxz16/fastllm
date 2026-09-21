"""CPU-only cache identity/configuration contracts using tiny model fixtures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import persistent_prefix as cache
from fastllm_pytools.util import make_normal_parser


def write_tensors(path, tensors):
    """Minimal valid safetensors fixture; values are raw U8 buffers."""
    payload = bytearray()
    header = {}
    for name, value in tensors.items():
        begin = len(payload)
        payload.extend(value)
        header[name] = {"dtype": "U8", "shape": [len(value)],
                        "data_offsets": [begin, len(payload)]}
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


class PersistentPrefixTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.tensors = {"model.language_model.layer.weight": b"target",
                        "model.visual.layer.weight": b"vision"}
        self.write_model(self.model, self.tensors)
        self.library = self.root / "libfixture.so"
        self.library.write_bytes(b"native-v2")
        self.native = types.SimpleNamespace(
            get_persistent_prefix_cache_version=Mock(return_value=2))
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def write_model(self, directory, tensors, draft=False):
        directory.mkdir(exist_ok=True)
        config = ({"model_type": "qwen3", "architectures": ["DFlash2DraftModel"]}
                  if draft else {"model_type": "qwen3_5", "text_config": {
                      "model_type": "qwen3_5_text", "head_dim": 256},
                      "vision_config": {"patch_size": 16}})
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "tokenizer.json").write_text('{"version":"1"}')
        (directory / "chat_template.jinja").write_text("{{ messages }}")
        (directory / "preprocessor_config.json").write_text('{"merge_size":2}')
        write_tensors(directory / "model.safetensors", tensors)

    def execution(self, **values):
        execution = {"dtype": "auto", "atype": "float16", "kv_cache_dtype": "float16",
                     "chunked_prefill_size": 2048, "tp": "0,1,2,3", "env": {}}
        execution.update(values)
        return execution

    def manifest(self, model=None, execution=None, draft=None):
        return cache.build_identity(model or self.model, execution or self.execution(),
                                    self.library, draft)

    def keys(self, **kwargs):
        return cache.identity_keys(self.manifest(**kwargs))

    def args(self, *argv):
        return make_normal_parser("persistent fixture").parse_args([
            "--path", str(self.model), *argv])

    def configure(self, args):
        return cache.persistent_prefix_environment(args, True, str(self.library), self.native)

    def test_identity_follows_content_not_path_or_mtime(self):
        baseline = self.keys()
        replica = self.root / "replica"
        shutil.copytree(self.model, replica)
        self.assertEqual(baseline, self.keys(model=replica))
        weights = replica / "model.safetensors"
        previous = weights.stat()
        changed = dict(self.tensors)
        changed["model.language_model.layer.weight"] = b"TARGET"
        write_tensors(weights, changed)
        os.utime(weights, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        updated = self.keys(model=replica)
        self.assertNotEqual(baseline["identity"], updated["identity"])
        self.assertNotEqual(baseline["family_identity"], updated["family_identity"])
        self.assertEqual(baseline["encoder_identity"], updated["encoder_identity"])

    def test_shard_repacking_preserves_semantic_identity(self):
        baseline = self.keys()
        target_name, vision_name = self.tensors
        write_tensors(self.model / "target.safetensors", {target_name: self.tensors[target_name]})
        write_tensors(self.model / "vision.safetensors", {vision_name: self.tensors[vision_name]})
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {target_name: "target.safetensors", vision_name: "vision.safetensors"}}))
        (self.model / "model.safetensors").unlink()
        self.assertEqual(baseline, self.keys())

    def test_encoder_and_target_tensor_groups_are_independent(self):
        baseline = self.keys()
        changed = dict(self.tensors)
        changed["model.visual.layer.weight"] = b"VISION"
        write_tensors(self.model / "model.safetensors", changed)
        updated = self.keys()
        self.assertNotEqual(baseline["identity"], updated["identity"])
        self.assertNotEqual(baseline["encoder_identity"], updated["encoder_identity"])
        self.assertEqual(baseline["target_identity"], updated["target_identity"])

    def test_parallel_shards_preserve_serial_identity_with_draft(self):
        mapping = {}
        for number in range(7):
            name = "model.language_model.layer%d.weight" % number
            shard = "part%d.safetensors" % number
            write_tensors(self.model / shard, {name: bytes([number]) * (4096 * (number + 1))})
            mapping[name] = shard
        (self.model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        draft = self.root / "draft"
        self.write_model(draft, {"draft.layer.weight": b"draft"}, draft=True)
        with patch.object(cache, "_HASH_WORKERS", 1):
            serial = self.manifest(draft=draft)
        with patch.object(cache, "_HASH_WORKERS", 8):
            parallel = self.manifest(draft=draft)
        self.assertEqual(serial, parallel)
        self.assertEqual(cache.identity_keys(serial), cache.identity_keys(parallel))

    def test_parallel_shard_failure_rejects_identity(self):
        mapping = {}
        for number in range(4):
            name, shard = "layer%d.weight" % number, "part%d.safetensors" % number
            write_tensors(self.model / shard, {name: b"weight"})
            mapping[name] = shard
        (self.model / "part2.safetensors").write_bytes(b"short")
        (self.model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        with self.assertRaisesRegex(ValueError, "header"):
            self.manifest()

    def test_file_replaced_during_hash_is_rejected(self):
        path = self.model / "model.safetensors"
        changed = False

        def replace_during_read(*_, **__):
            nonlocal changed
            if not changed:
                changed = True
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)

        before, offset, tensors = cache._tensor_layout(path)
        with self.assertRaisesRegex(ValueError, "changed while"):
            cache._hash_tensor((0, path, before, offset, tensors[0]),
                               threading.local(), 16, replace_during_read)

    def test_tensor_pool_bounds_and_shared_target_draft(self):
        target = {"target.%d" % n: bytes([n]) * 4096 for n in range(2)}
        draft_weights = {"draft.%d" % n: bytes([n]) * 4096 for n in range(32)}
        self.write_model(self.model, target)
        draft = self.root / "draft"
        self.write_model(draft, draft_weights, draft=True)
        real_executor, real_hash, real_wait = cache.ThreadPoolExecutor, cache._hash_tensor, cache.wait
        gate, lock = threading.Barrier(8), threading.Lock()
        state = {"pools": 0, "peak_pending": 0, "started": 0}
        buffer_ids, early_models = {}, set()

        class CountedExecutor(real_executor):
            def __init__(inner, *args, **kwargs):
                state["pools"] += 1
                state["max_workers"] = kwargs["max_workers"]
                super().__init__(*args, **kwargs)

        def observed_wait(pending, **kwargs):
            state["peak_pending"] = max(state["peak_pending"], len(pending))
            return real_wait(pending, **kwargs)

        def observed_hash(task, buffers, size, progress):
            with lock:
                state["started"] += 1
                early = state["started"] <= 8
                if early:
                    early_models.add(task[0])
            if early:
                gate.wait(timeout=5)
            result = real_hash(task, buffers, size, progress)
            with lock:
                buffer_ids.setdefault(threading.get_ident(), set()).add(id(buffers.data))
                self.assertLessEqual(len(buffers.data), cache._READ_BYTES)
            return result

        with patch.object(cache, "ThreadPoolExecutor", CountedExecutor), \
                patch.object(cache, "_hash_tensor", observed_hash), patch.object(cache, "wait", observed_wait):
            manifest = self.manifest(draft=draft)
        self.assertEqual(state["pools"], 1)
        self.assertEqual(state["max_workers"], 8)
        self.assertLessEqual(state["peak_pending"], 16)
        self.assertEqual(early_models, {0, 1})
        self.assertEqual(len(buffer_ids), 8)
        self.assertTrue(all(len(ids) == 1 for ids in buffer_ids.values()))
        for name, value in target.items():
            self.assertEqual(manifest["target"]["tensors"][name]["sha256"], hashlib.sha256(value).hexdigest())
        for name, value in draft_weights.items():
            self.assertEqual(manifest["draft"]["tensors"][name]["sha256"], hashlib.sha256(value).hexdigest())

    def test_change_to_finished_shard_is_rejected(self):
        write_tensors(self.model / "first.safetensors", {"first": b"first-long"})
        write_tensors(self.model / "last.safetensors", {"last": b"last"})
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"first": "first.safetensors", "last": "last.safetensors"}}))
        original = cache._hash_tensor

        def mutate_completed(task, *args):
            result = original(task, *args)
            if task[1].name == "last.safetensors":
                path = self.model / "first.safetensors"
                before = path.stat()
                with path.open("r+b") as handle:
                    handle.seek(-1, 2)
                    handle.write(b"X")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            return result

        with patch.object(cache, "_HASH_WORKERS", 1), patch.object(cache, "_hash_tensor", mutate_completed):
            with self.assertRaisesRegex(ValueError, "changed while"):
                self.manifest()

    def test_streaming_digest_short_reads_and_empty_tensor(self):
        weights = {"empty": b"", "tail": b"0123456789abcdefg"}
        self.write_model(self.model, weights)
        with patch.object(cache, "_READ_BYTES", 3):
            manifest = self.manifest()
        for name, value in weights.items():
            self.assertEqual(manifest["target"]["tensors"][name]["sha256"], hashlib.sha256(value).hexdigest())

    def test_tensor_truncated_while_reading_is_rejected(self):
        path = self.model / "model.safetensors"
        before, offset, tensors = cache._tensor_layout(path)

        def truncate(*_, **__):
            with path.open("r+b") as handle:
                handle.truncate(offset + 1)

        with self.assertRaisesRegex(ValueError, "Truncated"):
            cache._hash_tensor((0, path, before, offset, tensors[0]), threading.local(), 2, truncate)

    def test_unindexed_tensors_loaded_by_native_are_hashed(self):
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"model.language_model.layer.weight": "model.safetensors"}}))
        baseline = self.keys()
        changed = dict(self.tensors)
        changed["model.visual.layer.weight"] = b"VISION"
        write_tensors(self.model / "model.safetensors", changed)
        self.assertNotEqual(baseline["identity"], self.keys()["identity"])

    def test_exact_dtype_differs_but_fp16_fp8_share_only_family(self):
        fp16 = self.keys()
        fp8 = self.keys(execution=self.execution(kv_cache_dtype="fp8_e4m3"))
        self.assertNotEqual(fp16["identity"], fp8["identity"])
        for name in ("family_identity", "target_identity", "encoder_identity", "draft_identity"):
            self.assertEqual(fp16[name], fp8[name])
        for field, value in (("chunked_prefill_size", 4096), ("tp", "0,1"),
                             ("atype", "bfloat16"), ("rope_scaling", "yarn")):
            with self.subTest(field=field):
                changed = self.keys(execution=self.execution(**{field: value}))
                self.assertNotEqual(fp16["family_identity"], changed["family_identity"])

    def test_draft_content_invalidates_complete_identity_only(self):
        draft = self.root / "draft"
        self.write_model(draft, {"draft.layer.weight": b"draft1"}, draft=True)
        baseline = self.keys(draft=draft)
        write_tensors(draft / "model.safetensors", {"draft.layer.weight": b"draft2"})
        changed = self.keys(draft=draft)
        for name in ("identity", "family_identity", "draft_identity"):
            self.assertNotEqual(baseline[name], changed[name])
        for name in ("target_identity", "encoder_identity"):
            self.assertEqual(baseline[name], changed[name])

    def test_template_processor_runtime_and_numerical_env_invalidate(self):
        baseline = self.keys()
        for filename in ("config.json", "tokenizer.json", "chat_template.jinja",
                         "preprocessor_config.json"):
            path = self.model / filename
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(filename=filename):
                self.assertNotEqual(baseline["identity"], self.keys()["identity"])
            path.write_bytes(original)
        self.library.write_bytes(b"native-v3")
        self.assertNotEqual(baseline["identity"], self.keys()["identity"])
        self.library.write_bytes(b"native-v2")
        changed = self.keys(execution=self.execution(env={"FASTLLM_CUDA_DFLASH_TP_BACKBONE": "force"}))
        self.assertNotEqual(baseline["family_identity"], changed["family_identity"])

    def test_manifest_hashes_original_tensor_bytes(self):
        manifest = self.manifest()
        target = manifest["target"]["tensors"]["model.language_model.layer.weight"]
        self.assertEqual(target["bytes"], 6)
        self.assertEqual(target["sha256"], hashlib.sha256(b"target").hexdigest())

    def test_invalid_tensor_and_index_are_rejected(self):
        weights = self.model / "model.safetensors"
        original = weights.read_bytes()
        weights.write_bytes(original[:-1])
        with self.assertRaises(ValueError):
            self.manifest()
        weights.write_bytes(original)
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"missing.weight": "model.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "missing tensor"):
            self.manifest()
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"missing.weight": "../model.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "relative"):
            self.manifest()

    def test_env_only_enable_preserves_raw_env_and_snapshot_interval(self):
        directory = str(self.root / "ssd")
        os.environ.update({"FASTLLM_PREFIX_CACHE_DIR": directory,
                           "FASTLLM_PREFIX_CACHE_DISK_BYTES": str(100 << 30),
                           "FASTLLM_PREFIX_CACHE_RESTORE_POLICY": "always",
                           "FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES": "128"})
        with self.configure(self.args()) as exported:
            self.assertEqual(exported["FASTLLM_PREFIX_CACHE_DIR"], directory)
            self.assertEqual(exported["FASTLLM_PREFIX_CACHE_MANIFEST_VERSION"], "2")
            self.assertEqual(exported["FASTLLM_PREFIX_CACHE_DISK_BYTES"], str(100 << 30))
            self.assertEqual(exported["FASTLLM_PREFIX_CACHE_RESTORE_POLICY"], "always")
            identity = exported["FASTLLM_PREFIX_CACHE_IDENTITY"]
            manifest = json.loads((Path(directory) / "v2" / "identities" / (identity + ".json")).read_text())
            self.assertEqual(cache.identity_keys(manifest)["identity"], identity)
        self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_DIR"], directory)
        self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_DISK_BYTES"], str(100 << 30))
        self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_RESTORE_POLICY"], "always")
        self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES"], "128")
        for name in cache._DERIVED_ENV:
            self.assertNotIn(name, os.environ)

    def test_cli_overrides_env_and_does_not_enable_next_model(self):
        os.environ.update({"FASTLLM_PREFIX_CACHE_DIR": "environment-directory",
                           "FASTLLM_PREFIX_CACHE_DISK_BYTES": "invalid",
                           "FASTLLM_PREFIX_CACHE_RESTORE_POLICY": "invalid"})
        args = self.args("--prefix_cache_dir", str(self.root / "cli"),
                         "--prefix_cache_disk_gb", "0.5",
                         "--prefix_cache_restore_policy", "never")
        with self.configure(args) as settings:
            self.assertEqual(settings["FASTLLM_PREFIX_CACHE_DIR"], str(self.root / "cli"))
            self.assertEqual(settings["FASTLLM_PREFIX_CACHE_DISK_BYTES"], str(1 << 29))
            self.assertEqual(settings["FASTLLM_PREFIX_CACHE_RESTORE_POLICY"], "never")
        self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_DIR"], "environment-directory")
        with self.configure(self.args("--prefix_cache_dir", "")) as settings:
            self.assertIsNone(settings)
            self.assertNotIn("FASTLLM_PREFIX_CACHE_DIR", os.environ)
        self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_DIR"], "environment-directory")

    def test_defaults_no_snapshot_mutation_and_failed_construction_cleanup(self):
        args = self.args("--prefix_cache_dir", str(self.root / "ssd"))
        with self.assertRaisesRegex(RuntimeError, "constructor failed"):
            with self.configure(args) as settings:
                self.assertEqual(settings["FASTLLM_PREFIX_CACHE_DISK_BYTES"], str(256 << 30))
                self.assertEqual(settings["FASTLLM_PREFIX_CACHE_RESTORE_POLICY"], "auto")
                self.assertNotIn("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", os.environ)
                raise RuntimeError("constructor failed")
        for name in (*cache._DERIVED_ENV, *cache._INPUT_ENV):
            self.assertNotIn(name, os.environ)

    def test_second_model_rehashes_and_disabled_model_clears_stale_identity(self):
        os.environ["FASTLLM_PREFIX_CACHE_DIR"] = str(self.root / "ssd")
        with self.configure(self.args()) as settings:
            first = settings["FASTLLM_PREFIX_CACHE_IDENTITY"]
        changed = dict(self.tensors)
        changed["model.language_model.layer.weight"] = b"TARGET"
        write_tensors(self.model / "model.safetensors", changed)
        with self.configure(self.args()) as settings:
            self.assertNotEqual(first, settings["FASTLLM_PREFIX_CACHE_IDENTITY"])
        os.environ["FASTLLM_PREFIX_CACHE_IDENTITY"] = first
        with self.configure(self.args("--prefix_cache_dir", "")):
            self.assertNotIn("FASTLLM_PREFIX_CACHE_IDENTITY", os.environ)

    def test_missing_native_capability_fails_before_hashing(self):
        args = self.args("--prefix_cache_dir", str(self.root / "ssd"))
        for native in (None, types.SimpleNamespace(), types.SimpleNamespace(
                get_persistent_prefix_cache_version=Mock(return_value=0))):
            with self.subTest(native=native), patch.object(cache, "build_identity") as build:
                with self.assertRaisesRegex(RuntimeError, "native"):
                    with cache.persistent_prefix_environment(args, True, self.library, native):
                        self.fail("Unsupported library must not construct a model")
                build.assert_not_called()

    def test_configuration_rejects_unsupported_paths_and_invalid_limits(self):
        base = self.args("--prefix_cache_dir", str(self.root / "ssd"))
        for name, value in (("mtp", 1), ("dspark", 2), ("speculative_algorithm", "mtp"),
                            ("lora", "/lora"), ("custom", "/custom.py"),
                            ("prefix_cache_disk_gb", float("nan")),
                            ("prefix_cache_disk_gb", 1e308),
                            ("prefix_cache_disk_gb", -1), ("prefix_cache", "false")):
            with self.subTest(name=name, value=value):
                args = copy.copy(base)
                setattr(args, name, value)
                with self.assertRaises(ValueError), self.configure(args):
                    self.fail("Invalid configuration must be rejected")
        os.environ["FASTLLM_PREFIX_CACHE_DISK_BYTES"] = "0"
        with self.assertRaisesRegex(ValueError, "quota"), self.configure(base):
            self.fail("Zero capacity must be rejected")

    def test_dflash_enabled_and_actual_chunk_and_numerical_env_recorded(self):
        draft = self.root / "draft"
        self.write_model(draft, {"draft.layer.weight": b"draft1"}, draft=True)
        args = self.args("--prefix_cache_dir", str(self.root / "ssd"),
                         "--speculative_algorithm", "dflash", "--draft", str(draft),
                         "--draft_tokens", "6", "--chunked_prefill_size", "2048")
        os.environ.update({"FASTLLM_DFLASH_MODEL_PATH": str(draft),
                           "FASTLLM_DFLASH_BLOCK_SIZE": "7",
                           "FASTLLM_CUDA_DFLASH_TP_BACKBONE": "force"})
        with self.configure(args) as settings:
            self.assertNotEqual(settings["FASTLLM_PREFIX_CACHE_DRAFT_IDENTITY"], "none")
            manifest = json.loads((self.root / "ssd" / "v2" / "identities" /
                (settings["FASTLLM_PREFIX_CACHE_IDENTITY"] + ".json")).read_text())
            self.assertEqual(manifest["execution"]["chunked_prefill_size"], 2048)
            self.assertEqual(manifest["execution"]["env"]["FASTLLM_DFLASH_BLOCK_SIZE"], "7")
            self.assertNotIn("FASTLLM_DFLASH_MODEL_PATH", manifest["execution"]["env"])


if __name__ == "__main__":
    unittest.main()

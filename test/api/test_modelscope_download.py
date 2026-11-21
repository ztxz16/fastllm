import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.modelscope_download import (
    AggregateProgress,
    PROGRESS_PREFIX,
    _filter_file_sizes,
    download, download_bundle, _bundle_plan,
)


from fastllm_pytools.model_catalog import bundle, source


class ModelScopeProgressAdapterTest(unittest.TestCase):
    def test_unmatched_filter_is_not_reported_as_a_successful_download(self):
        snapshot = Mock()
        args = SimpleNamespace(model="test/model", revision="master", local_dir="/tmp/model",
                               max_workers=2, include=["missing/*"])
        with patch.dict(sys.modules, {
            "modelscope.hub.snapshot_download": SimpleNamespace(snapshot_download=snapshot),
        }), patch("fastllm_pytools.modelscope_download._download_plan",
                  return_value=({}, "master", None)):
            with self.assertRaisesRegex(ValueError, "No repository files match"):
                download(args, None)
        snapshot.assert_not_called()

    def test_file_patterns_select_nested_quantization_directory(self):
        files = {
            "Q2_0/model-00001.gguf": 100,
            "Q2_0/model-00002.gguf": 200,
            "IQ3_S/model-00001.gguf": 300,
            "mmproj-Qwen3.8-Flash-Next-BF16.gguf": 40,
        }
        self.assertEqual(
            _filter_file_sizes(files, ["Q2_0/*", "mmproj-*.gguf"]),
            {
                "Q2_0/model-00001.gguf": 100,
                "Q2_0/model-00002.gguf": 200,
                "mmproj-Qwen3.8-Flash-Next-BF16.gguf": 40,
            },
        )

    def test_aggregate_progress_is_byte_weighted(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            progress = AggregateProgress({"a": 100, "b": 900})
            callback = progress.callback_type()("a", 100)
            callback.update(50)
            callback.end()

        events = [
            json.loads(line[len(PROGRESS_PREFIX):])
            for line in stream.getvalue().splitlines()
        ]
        self.assertEqual(events[0]["totalBytes"], 1000)
        self.assertTrue(any(event["downloadedBytes"] == 50 for event in events))
        self.assertEqual(events[-1]["downloadedBytes"], 100)
        self.assertEqual(events[-1]["completedFiles"], 1)

    def test_success_counts_cached_files_without_callbacks(self):
        def snapshot_download(**kwargs):
            callback = kwargs["progress_callbacks"][0]("new", 100)
            callback.update(100)
            callback.end()
            return kwargs["local_dir"]

        args = SimpleNamespace(model="test/model", revision="master", local_dir="/tmp/model", max_workers=4, include=[])
        stream = io.StringIO()
        with patch.dict(sys.modules, {
            "modelscope.hub.snapshot_download": SimpleNamespace(snapshot_download=snapshot_download),
        }), patch("fastllm_pytools.modelscope_download._download_plan",
                  return_value=({"cached": 900, "new": 100}, "master", None)), redirect_stdout(stream):
            self.assertEqual(download(args, None), "/tmp/model")
        events = [json.loads(line[len(PROGRESS_PREFIX):]) for line in stream.getvalue().splitlines()]
        self.assertEqual(events[-2]["downloadedBytes"], 100)
        self.assertEqual(events[-1]["downloadedBytes"], 1000)
        self.assertEqual(events[-1]["completedFiles"], 2)

    def test_include_patterns_are_forwarded_to_modelscope(self):
        captured = {}

        def snapshot_download(**kwargs):
            captured.update(kwargs)
            return kwargs["local_dir"]

        args = SimpleNamespace(
            model="test/model",
            revision="master",
            local_dir="/tmp/model",
            max_workers=2,
            include=["Q2_0/*", "mmproj-*.gguf"],
        )
        with patch.dict(sys.modules, {
            "modelscope.hub.snapshot_download": SimpleNamespace(snapshot_download=snapshot_download),
        }), patch("fastllm_pytools.modelscope_download._download_plan",
                  return_value=({"Q2_0/model.gguf": 100}, "master", None)):
            self.assertEqual(download(args, None), "/tmp/model")
        self.assertEqual(captured["allow_patterns"], ["Q2_0/*", "mmproj-*.gguf"])


class BundleDownloadTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.files = {
            "org/base": {"config.json": b"{}", "model-1.safetensors": b"base",
                         "model.safetensors.index.json": b'{"weight_map":{"w":"model-1.safetensors"}}'},
            "org/adapter": {"config.json": b"adapter", "unused": b"unselected"},
        }
        self.recipe = bundle("example", "FP8", [
            source("org/adapter", "config.json", rename={"config.json": "adapter.json"}),
            source("org/base", "config.json model*.safetensors*", "base_model")],
            manifest={"base_model": "base_model"})
        self.calls = []

        def plan(repo, revision, token, patterns=None, *, checksums):
            checksums.update({name: hashlib.sha256(data).hexdigest() for name, data in self.files[repo].items()})
            return _filter_file_sizes({k: len(v) for k, v in self.files[repo].items()}, patterns), revision, None
        mock = patch("fastllm_pytools.modelscope_download._download_plan", side_effect=plan)
        mock.start()
        self.addCleanup(mock.stop)

    def snapshot(self, **kwargs):
        repo = kwargs["model_id"]
        self.calls.append(repo)
        for name in kwargs["allow_patterns"]:
            path = Path(kwargs["local_dir"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.files[repo][name])
        return kwargs["local_dir"]

    def fetch(self, target="model", snapshot=None):
        self.stream = io.StringIO()
        with redirect_stdout(self.stream):
            return download_bundle(self.recipe, self.root / target, 1, None, snapshot or self.snapshot)

    @unittest.skipUnless(importlib.util.find_spec("filelock"), "requires the ModelScope filelock dependency")
    def test_public_checkpoint_resumes_without_sending_modelscope_credentials(self):
        data = b"public checkpoint"
        quant = source("org/public", "dit.safetensors")
        quant.update(hub="Hugging Face", revision="abc", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.recipe["sources"][0] = quant
        target = self.root / "model"
        calls = []

        def fetch(request, timeout):
            calls.append(request)
            response = io.BytesIO(data[:4] if len(calls) == 1 else data[4:])
            response.status = 200 if len(calls) == 1 else 206
            response.headers = {"Content-Range": f"bytes 4-{len(data)-1}/{len(data)}"}
            return response

        with patch("urllib.request.urlopen", side_effect=fetch), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Incomplete transfer"):
                download_bundle(self.recipe, target, 1, "modelscope-secret", self.snapshot)
            self.assertFalse(target.exists())
            download_bundle(self.recipe, target, 1, "modelscope-secret", self.snapshot)
        self.assertEqual((target / "dit.safetensors").read_bytes(), data)
        self.assertEqual(calls[1].get_header("Range"), "bytes=4-")
        self.assertTrue(all(request.full_url == "https://huggingface.co/org/public/resolve/abc/dit.safetensors"
                            and "modelscope-secret" not in str(request.header_items()) for request in calls))

    def test_rejects_missing_sources_overlaps_and_unsafe_paths_before_transfer(self):
        self.recipe["sources"][1]["files"].append("missing-*.json")
        with self.assertRaisesRegex(ValueError, "org/base: missing component"):
            _bundle_plan(self.recipe, None)
        self.recipe["sources"][1] = source("org/base", "config.json", rename={"config.json": "adapter.json"})
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            _bundle_plan(self.recipe, None)
        self.recipe["sources"][1]["rename"]["config.json"] = "../outside"
        with self.assertRaisesRegex(ValueError, "Invalid repository file path"):
            _bundle_plan(self.recipe, None)

    @unittest.skipUnless(importlib.util.find_spec("filelock"), "requires filelock")
    def test_pinned_huggingface_snapshot_verifies_git_blobs_and_weight_shards(self):
        files = self.files["org/base"]
        revision = "c" * 40
        manifest = {"sha": revision, "siblings": []}
        for name, data in files.items():
            item = {"rfilename": name, "size": len(data)}
            if name.endswith(".safetensors"):
                item["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
            else:
                item["blobId"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            manifest["siblings"].append(item)
        weights = source("org/base", "*.json *.safetensors")
        weights.update(hub="Hugging Face", revision=revision)
        self.recipe = bundle("hf-snapshot", "FP8", [weights])

        def fetch(request, timeout):
            if isinstance(request, str):
                self.assertIn("/revision/" + revision + "?blobs=true", request)
                return io.BytesIO(json.dumps(manifest).encode())
            self.assertIn("/resolve/" + revision + "/", request.full_url)
            self.assertIsNone(request.get_header("Authorization"))
            response = io.BytesIO(files[request.full_url.rsplit("/", 1)[1]])
            response.status = 200
            return response

        with patch("urllib.request.urlopen", side_effect=fetch):
            files["config.json"] = b"xx"
            with self.assertRaisesRegex(ValueError, "corrupt component"):
                self.fetch()
            files["config.json"] = b"{}"
            target = Path(self.fetch())
            self.assertEqual((target / "model-1.safetensors").read_bytes(), b"base")

    @unittest.skipUnless(importlib.util.find_spec("filelock"), "requires the ModelScope filelock dependency")
    def test_reuses_verified_installed_weights_and_fetches_only_missing_files(self):
        installed = self.root / "base"
        installed.mkdir()
        (installed / "model-1.safetensors").write_bytes(b"base")
        (installed / "config.json").write_bytes(b"xx")  # Same size, different content.
        fetched = {}
        def capture(**kwargs):
            fetched[kwargs["model_id"]] = kwargs["allow_patterns"]
            return self.snapshot(**kwargs)
        target = Path(self.fetch(snapshot=capture))
        self.assertEqual(set(fetched["org/base"]), {"config.json", "model.safetensors.index.json"})
        self.assertEqual((installed / "config.json").read_bytes(), b"xx")
        self.assertEqual((target / "base_model/config.json").read_bytes(), b"{}")
        self.assertTrue(os.path.samefile(target / "base_model/model-1.safetensors", installed / "model-1.safetensors"))

    @unittest.skipUnless(importlib.util.find_spec("filelock"), "requires the ModelScope filelock dependency")
    def test_failure_retry_cache_sharing_and_completed_handoff(self):
        def interrupt(**kwargs):
            if kwargs["model_id"] == "org/base":
                raise OSError("connection interrupted")
            return self.snapshot(**kwargs)

        with self.assertRaisesRegex(OSError, "interrupted"):
            self.fetch(snapshot=interrupt)
        self.assertFalse((self.root / "model").exists())
        (self.root / "model").mkdir()  # An existing empty directory can be published to as well.
        target = Path(self.fetch())
        self.assertEqual(self.calls, ["org/adapter", "org/base"])
        self.assertEqual((target / "adapter.json").read_bytes(), b"adapter")
        self.assertEqual((target / "base_model/config.json").read_bytes(), b"{}")
        self.assertFalse((target / "unused").exists())
        self.assertEqual(json.loads((target / "h3_quantization.json").read_text()), self.recipe["manifest"])
        last = json.loads(self.stream.getvalue().splitlines()[-1][len(PROGRESS_PREFIX):])
        self.assertEqual((last["totalFiles"], last["completedFiles"]), (4, 4))
        self.assertEqual(last["totalBytes"], sum(map(len, self.files["org/base"].values())) + len(b"adapter"))
        self.assertEqual(last["totalBytes"], last["downloadedBytes"])
        self.fetch()
        other = Path(self.fetch("sibling"))
        self.assertEqual(self.calls, ["org/adapter", "org/base"])
        weight = "base_model/model-1.safetensors"
        self.assertTrue(os.path.samefile(other / weight, target / weight))
        self.assertFalse(os.path.samefile(other / "adapter.json", target / "adapter.json"))
        (target / "adapter.json").write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.fetch()
        self.recipe["preset"] = "different"
        with self.assertRaisesRegex(ValueError, "another model"):
            self.fetch()
        self.assertEqual((target / "adapter.json").read_bytes(), b"damaged")

    @unittest.skipUnless(importlib.util.find_spec("filelock"), "requires the ModelScope filelock dependency")
    def test_rejects_corrupt_files_and_missing_index_shards(self):
        def corrupt(**kwargs):
            path = self.snapshot(**kwargs)
            (Path(path) / "config.json").write_bytes(b"corrupt")
            return path
        with self.assertRaisesRegex(ValueError, "corrupt component"):
            self.fetch(snapshot=corrupt)
        self.files["org/base"]["model.safetensors.index.json"] = b'{"weight_map":{"w":"missing.safetensors"}}'
        with self.assertRaisesRegex(ValueError, "missing shard"):
            self.fetch()
        self.assertFalse((self.root / "model").exists())

    @unittest.skipUnless(importlib.util.find_spec("filelock"), "requires the ModelScope filelock dependency")
    def test_lingbot_conversion_must_produce_both_native_files(self):
        self.recipe["prepare"] = "lingbot"
        def convert(argv, check):
            native = Path(argv[-1]) / "native"
            native.mkdir()
            (native / "t5.safetensors").write_bytes(b"t5")
            (native / "vae.safetensors").write_bytes(b"vae")
        with patch("fastllm_pytools.modelscope_download.preparation_error", return_value=""):
            with patch("fastllm_pytools.modelscope_download.subprocess.run"):
                with self.assertRaises(FileNotFoundError):
                    self.fetch()
                self.assertFalse((self.root / "model").exists())
            with patch("fastllm_pytools.modelscope_download.subprocess.run", side_effect=convert):
                self.fetch()
        self.assertEqual((self.root / "model/native/vae.safetensors").read_bytes(), b"vae")


if __name__ == "__main__":
    unittest.main()

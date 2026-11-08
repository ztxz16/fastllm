#!/usr/bin/env python3
"""SSD startup rejects obsolete formats before model hashing or filesystem I/O."""
import contextlib
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/fastllm_pytools"))
import persistent_prefix as pp
import util


def native(caps=0x1c):
    def capability():
        return caps
    def config(value):
        return 1
    return types.SimpleNamespace(
        get_persistent_prefix_cache_storage_capabilities=capability,
        set_persistent_prefix_cache_storage_config=config)


class V4OnlyStartup(unittest.TestCase):
    def setUp(self):
        self.parser = util.make_normal_parser("v4-only SSD")

    def args(self, *extra):
        return self.parser.parse_args(["unused-model", "--prefix_cache_dir", "/unused/cache", *extra])

    def test_default_requires_v4_native(self):
        self.assertEqual(self.args().prefix_cache_storage_version, 4)
        # A pre-capability runtime and a v3-only runtime cannot serve a v4 deployment.
        for library in (types.SimpleNamespace(), native(0x13)):
            with self.subTest(library=library), self.assertRaisesRegex(RuntimeError, "v4"):
                pp._negotiate_storage(self.args(), library)
        self.assertEqual(pp._negotiate_storage(self.args(), native()), (4, 0x1c))

    def test_obsolete_cli_is_rejected(self):
        for argv in (("--prefix_cache_storage_version", "3"),
                     ("--prefix_cache_file_gb", "2"),
                     ("--prefix_cache_legacy_root", "/old/v3")):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args(*argv)

    def test_obsolete_namespace_cannot_bypass_parser(self):
        for field, value in (("prefix_cache_storage_version", 3),
                             ("prefix_cache_legacy_root", ["/old/v3"]),
                             ("prefix_cache_file_gb", 2)):
            args = self.args()
            setattr(args, field, value)
            with self.subTest(field=field), patch.object(pp, "build_identity") as hashing:
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(ValueError, "v4|removed|Unclassified"):
                        pp.configure_persistent_prefix(args, True, "unused", native())
                hashing.assert_not_called()

    def test_obsolete_file_limit_env_cannot_affect_v4(self):
        with patch.dict(os.environ, {"FASTLLM_PREFIX_CACHE_FILE_BYTES": "2147483648"}, clear=True):
            with patch.object(pp, "build_identity") as hashing:
                with self.assertRaisesRegex(ValueError, "FILE_BYTES.*removed"):
                    pp.configure_persistent_prefix(self.args(), True, "unused", native())
                hashing.assert_not_called()

    def test_manifest_builder_refuses_v3_before_opening_models(self):
        with patch.object(pp, "_model_plan") as model:
            with self.assertRaisesRegex(ValueError, "v4"):
                pp.build_identity("unused", {}, "unused", storage_format=3)
            model.assert_not_called()

    def test_source_defaults_keep_capacity_controls_explicit(self):
        args = self.args()
        self.assertEqual(args.tokens, -1)
        self.assertEqual(args.max_context_length, -1)
        self.assertEqual(pp.DEFAULT_DISK_BYTES, 256 << 30)
        self.assertEqual(args.prefix_cache_disk_gb, 256.0)
        self.assertEqual(args.prefix_cache_restore_policy, "auto")
        self.assertEqual(args.prefix_cache_checkpoint_interval, 256)
        self.assertEqual(args.prefix_cache_checkpoint_jobs, 8)
        self.assertEqual(args.prefix_cache_host_limit_mb, 0)
        self.assertEqual(args.prefix_cache_l2, "off")
        self.assertEqual(args.prefix_cache_l2_gb, 4.0)
        self.assertEqual(self.args("--prefix_cache_disk_gb", "128").prefix_cache_disk_gb, 128.0)


    def test_unset_ssd_directory_never_uses_user_or_xdg_cache(self):
        args = self.parser.parse_args(["unused-model"])
        with patch.dict(os.environ, {
                "HOME": "/unused/home",
                "XDG_CACHE_HOME": "/unused/xdg"}, clear=True), \
                patch.object(pp, "build_identity") as hashing, \
                patch.object(pp, "_negotiate_storage") as negotiation:
            self.assertIsNone(util._configured_prefix_cache_dir())
            self.assertIsNone(pp.configure_persistent_prefix(
                args, True, "unused", native()))
            self.assertNotIn("FASTLLM_PREFIX_CACHE_DIR", os.environ)
            hashing.assert_not_called()
            negotiation.assert_not_called()

    def test_explicit_ssd_environment_and_cli_override_are_preserved(self):
        args = self.args()
        keys = {"identity": "a" * 64, "family_identity": "b" * 64,
                "target_identity": "c" * 64, "encoder_identity": "d" * 64,
                "draft_identity": "none"}
        with patch.dict(os.environ, {
                "FASTLLM_PREFIX_CACHE_DIR": "/unused/user-selected-env"}, clear=True), \
                patch.object(pp, "build_identity", return_value={}), \
                patch.object(pp, "identity_keys", return_value=keys), \
                patch.object(pp, "_native_manifest_function", return_value=lambda _: 1):
            self.assertEqual(
                util._configured_prefix_cache_dir(), "/unused/user-selected-env")
            pp.configure_persistent_prefix(args, True, "unused", native())
            self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_DIR"], "/unused/cache")

    def test_defaults_match_the_explicit_source_profile(self):
        defaults = self.args()
        explicit = self.args(
            "--prefix_cache_disk_gb", "256", "--prefix_cache_l2_gb", "4",
            "--prefix_cache_storage_version", "4",
            "--prefix_cache_checkpoint_interval", "256",
            "--prefix_cache_checkpoint_jobs", "8")
        for name in ("prefix_cache_disk_gb", "prefix_cache_l2_gb",
                     "prefix_cache_storage_version", "prefix_cache_checkpoint_interval",
                     "prefix_cache_checkpoint_jobs"):
            with self.subTest(option=name):
                self.assertEqual(getattr(defaults, name), getattr(explicit, name))
        self.assertEqual(defaults.prefix_cache_l2_gb, 4.0)
        self.assertIsNone(self.parser.parse_args(["unused-model"]).prefix_cache_dir)

    def test_cli_overrides_survive_environment_export(self):
        args = self.args(
            "--tokens", "16384", "--max_context_length", "8192",
            "--prefix_cache_disk_gb", "64", "--prefix_cache_l2", "on",
            "--prefix_cache_l2_gb", "2", "--prefix_cache_checkpoint_interval", "512",
            "--prefix_cache_checkpoint_jobs", "3")
        self.assertEqual((args.tokens, args.max_context_length), (16384, 8192))
        self.assertEqual(args.prefix_cache_disk_gb, 64.0)
        self.assertEqual(args.prefix_cache_dir, "/unused/cache")
        with patch.dict(os.environ, {
                "FASTLLM_PREFIX_CACHE_CHECKPOINT_INTERVAL": "1024",
                "FASTLLM_PREFIX_CACHE_CHECKPOINT_JOBS": "2"}, clear=True):
            util.apply_prefix_cache_env(args)
            self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_CHECKPOINT_INTERVAL"], "512")
            self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_CHECKPOINT_JOBS"], "3")
            self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_L2_HOST_KV_MB"], "2048")

    def test_default_and_explicit_profiles_export_the_same_storage_settings(self):
        profiles = (
            self.args("--prefix_cache_l2", "on"),
            self.args(
                "--prefix_cache_l2", "on", "--prefix_cache_disk_gb", "256",
                "--prefix_cache_l2_gb", "4", "--prefix_cache_storage_version", "4",
                "--prefix_cache_checkpoint_interval", "256",
                "--prefix_cache_checkpoint_jobs", "8"),
        )
        exports = []
        keys = {"identity": "a" * 64, "family_identity": "b" * 64,
                "target_identity": "c" * 64, "encoder_identity": "d" * 64,
                "draft_identity": "none"}
        for args in profiles:
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(pp, "build_identity", return_value={}), \
                    patch.object(pp, "identity_keys", return_value=keys), \
                    patch.object(pp, "_native_manifest_function", return_value=lambda _: 1):
                util.apply_prefix_cache_env(args)
                pp.configure_persistent_prefix(args, True, "unused", native())
                exports.append({name: os.environ[name] for name in (
                    "FASTLLM_PREFIX_CACHE_DISK_BYTES",
                    "FASTLLM_PREFIX_CACHE_CHECKPOINT_INTERVAL",
                    "FASTLLM_PREFIX_CACHE_CHECKPOINT_JOBS",
                    "FASTLLM_PREFIX_CACHE_L2_HOST_KV_MB")})
        self.assertEqual(exports[0], exports[1])
        self.assertEqual(exports[0]["FASTLLM_PREFIX_CACHE_DISK_BYTES"], str(256 << 30))
        self.assertEqual(exports[0]["FASTLLM_PREFIX_CACHE_CHECKPOINT_INTERVAL"], "256")
        self.assertEqual(exports[0]["FASTLLM_PREFIX_CACHE_CHECKPOINT_JOBS"], "8")



if __name__ == "__main__":
    unittest.main()

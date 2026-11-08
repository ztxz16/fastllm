#!/usr/bin/env python3
"""CPU checks for the v4-only persistent-prefix startup contract."""
import copy
import contextlib
import io
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "fastllm_pytools"))
import persistent_prefix as pp  # noqa: E402
import util  # noqa: E402

FAILURES = []


def check(condition, label):
    if condition:
        print("PASS %s" % label)
    else:
        print("FAIL %s" % label)
        FAILURES.append(label)


def expect_error(label, action, fragment):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            action()
    except SystemExit as error:
        check(error.code != 0, "%s -> argparse rejected obsolete option" % label)
        return
    except (RuntimeError, ValueError) as error:
        check(fragment in str(error), "%s -> %s" % (label, error))
        return
    check(False, "%s (no error raised)" % label)


def native(capabilities=pp.STORAGE_CAP_READ_V4 | pp.STORAGE_CAP_WRITE_V4 | pp.STORAGE_CAP_CONFIG):
    def capability():
        return capabilities

    def config(value):
        return 1

    return types.SimpleNamespace(
        get_persistent_prefix_cache_storage_capabilities=capability,
        set_persistent_prefix_cache_storage_config=config,
    )


def args(storage_version=4):
    return types.SimpleNamespace(prefix_cache_storage_version=storage_version)


def negotiation_rules():
    capable = native()
    version, capabilities = pp._negotiate_storage(args(), capable)
    check(version == 4 and capabilities == 0x1c, "capable native accepts v4")
    expect_error("v3 request rejected", lambda: pp._negotiate_storage(args(3), capable),
                 "only v4")
    expect_error("v3-only native rejected", lambda: pp._negotiate_storage(args(), native(0x13)),
                 "v4 prefix-cache writer")
    expect_error("old native rejected", lambda: pp._negotiate_storage(args(), types.SimpleNamespace()),
                 "v4 prefix-cache writer")


def parser_defaults():
    parser = util.make_normal_parser("v4-only SSD")
    parsed = parser.parse_args(["unused-model"])
    check(parsed.prefix_cache_storage_version == 4, "storage format defaults to v4")
    check(parsed.prefix_cache_disk_gb == 256.0, "SSD quota defaults to 256 GiB")
    check(parsed.prefix_cache_restore_policy == "auto", "restore policy defaults to auto")
    check(parsed.prefix_cache_checkpoint_interval == 256, "checkpoint interval defaults to 256")
    check(parsed.prefix_cache_checkpoint_jobs == 8, "checkpoint jobs defaults to 8")
    check(parsed.prefix_cache_host_limit_mb == 0, "host budget defaults to automatic")
    for argv in (("--prefix_cache_storage_version", "3"),
                 ("--prefix_cache_file_gb", "2"),
                 ("--prefix_cache_legacy_root", "/old/v3")):
        expect_error("obsolete CLI rejected %s" % (argv[0],),
                     lambda argv=argv: parser.parse_args(["unused-model", *argv]),
                     "unrecognized arguments" if argv[0] != "--prefix_cache_storage_version"
                     else "invalid choice")


def storage_config_fields():
    record = {"prefix_cache_storage_version": 4, "kv_cache_dtype": "fp8_e4m3"}
    pp._group_storage_config(record)
    check(record.get("storage_config") == {"prefix_cache_storage_version": 4},
          "storage config contains only v4 format")
    check("prefix_cache_storage_version" not in record,
          "storage format is nested in execution record")


def identity_is_storage_free():
    base = {
        "target": {"math_version": "qwen35-target-v2"},
        "state_abi": copy.deepcopy(pp.STATE_ABI),
        "encoding": {"kv_dtype": "fp8_e4m3"},
        "target_identity": "a" * 64,
        "encoder_identity": "b" * 64,
        "draft_identity": "none",
    }
    v4 = dict(base, storage_format_version=4, storage_manifest_schema_version=2)
    check(pp.identity_keys(base) == pp.identity_keys(v4),
          "storage format does not enter identity_keys")


def configure_rejects_old_environment_without_hashing():
    parser = util.make_normal_parser("v4-only SSD")
    parsed = parser.parse_args(["unused-model", "--prefix_cache_dir", "/unused/cache"])
    with patch.object(pp, "build_identity") as hashing:
        with patch.dict(os.environ, {"FASTLLM_PREFIX_CACHE_FILE_BYTES": "2147483648"}, clear=True):
            expect_error("v3 file quota environment rejected",
                         lambda: pp.configure_persistent_prefix(parsed, True, "unused", native()),
                         "FILE_BYTES")
        hashing.assert_not_called()


def main():
    negotiation_rules()
    parser_defaults()
    storage_config_fields()
    identity_is_storage_free()
    configure_rejects_old_environment_without_hashing()
    if FAILURES:
        print("FAILED: %d check(s)" % len(FAILURES))
        return 1
    print("PASS test_persistent_cache_capability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

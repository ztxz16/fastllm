#!/usr/bin/env python3
"""纯 Python 校验迁移工具的下载完整性和拒绝路径；不安装软件、不编译。"""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import export_env as env


class ExportTests(unittest.TestCase):
    """/** 覆盖真实 APT 格式、缓存损坏、下载失败和恢复入口的保护。 */"""

    def setUp(self):
        """/** 在独立临时目录创建固定样本，不触碰系统 APT 缓存。 */"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = b"fixed package fixture"
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.name = "libnccl2_2.28.3-1+cuda12.9_amd64.deb"
        self.plan = (f"'https://example.invalid/{self.name}' {self.name} "
                     f"{len(self.data)} SHA256:{self.digest}\n")

    def test_apt_format_with_logging(self):
        """/** APT 的状态行不影响有效 URI 的解析。 */"""
        rows = env.parse_uris("Reading package lists... Done\n" + self.plan)
        self.assertEqual(rows[0]["name"], self.name)
        self.assertEqual(rows[0]["sha256"], self.digest)

    def test_reject_unsafe_name_or_checksum(self):
        """/** 拒绝越界文件名、不可在 Windows 传输的名称及缺少 SHA256 的计划。 */"""
        for old, new in [(f" {self.name} ", " ../outside.deb "),
                         (f" {self.name} ", " with:colon.deb "),
                         ("SHA256:", "MD5Sum:"), ("https://", "file://")]:
            with self.subTest(replacement=new), self.assertRaises(ValueError):
                env.parse_uris(self.plan.replace(old, new))

    def test_reuse_verified_cache_without_downloader(self):
        """/** 缓存完整时，即使没有 aria2 也直接复用且不执行下载命令。 */"""
        cache = self.root / "cache"
        cache.mkdir()
        (cache / self.name).write_bytes(self.data)
        with patch.object(env, "capture", return_value=self.plan), patch.object(env, "run") as run:
            env.fetch_debs(["libnccl2=2.28.3-1+cuda12.9"], self.root / "out", [cache])
        run.assert_not_called()
        self.assertEqual((self.root / "out" / self.name).read_bytes(), self.data)

    def test_corrupt_cache_is_not_accepted(self):
        """/** 大小相同但内容损坏的缓存不能绕过 SHA256 检查。 */"""
        cache = self.root / "cache"
        cache.mkdir()
        (cache / self.name).write_bytes(b"x" * len(self.data))
        with patch.object(env, "capture", return_value=self.plan), \
                patch.object(env.shutil, "which", return_value=None), self.assertRaises(RuntimeError):
            env.fetch_debs(["libnccl2=2.28.3-1+cuda12.9"], self.root / "out", [cache])
        self.assertEqual((cache / self.name).read_bytes(), b"x" * len(self.data))

    def test_download_output_is_verified(self):
        """/** 即使下载进程返回成功，错误文件也必须导致导出失败。 */"""
        output = self.root / "out"

        def incomplete_download(command):
            """/** 模拟成功退出但留下错误内容的下载器。 */"""
            self.assertEqual(command[0], "aria2c")
            self.assertIn("checksum=sha-256=" + self.digest,
                          (output / "aria2-input.txt").read_text())
            (output / self.name).write_bytes(b"incomplete")

        with patch.object(env, "capture", return_value=self.plan), \
                patch.object(env.shutil, "which", return_value="aria2c"), \
                patch.object(env, "run", side_effect=incomplete_download), self.assertRaises(RuntimeError):
            env.fetch_debs(["libnccl2=2.28.3-1+cuda12.9"], output, [])

    def test_resolve_uses_separate_planning_cache(self):
        """/** 构建依赖计划不因全局 APT 缓存已有包而省略这些包。 */"""
        with patch.object(env, "capture", return_value="") as capture:
            env.fetch_debs(["example=1"], self.root / "out", [], resolve=True)
        command = capture.call_args.args[0]
        self.assertIn("Dir::Cache::archives=" + str(self.root / "out/apt-plan"), command)
        self.assertIn("--print-uris", command)
        self.assertIn("--no-remove", command)

    def test_package_selection_preserves_native_architecture(self):
        """/** i386 包不能覆盖同名 amd64 包的锁定版本，CUDA 13 包不纳入目标。 */"""
        versions = {name: "1" for name in env.REQUIRED}
        versions.update(libnccl2=env.NCCL_VERSION)
        versions["libnccl-dev"] = env.NCCL_VERSION
        listing = "".join(f"{name}\t{version}\tamd64\thold ok installed\n"
                          for name, version in versions.items())
        listing += "libnccl2\twrong\ti386\tinstall ok installed\n"
        listing += "cuda-nvcc-13-0\t13.0\tamd64\tinstall ok installed\n"
        with patch.object(env, "capture", return_value=listing):
            actual, _ = env.selected_packages()
        self.assertEqual(actual, versions)

    def test_restore_rejected_outside_build(self):
        """/** 误在原服务器运行恢复入口时，不得安装包或删除 Python 环境。 */"""
        with patch.dict(env.os.environ, {}, clear=True), patch.object(env, "run") as run, \
                self.assertRaises(RuntimeError):
            env.restore(self.root)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

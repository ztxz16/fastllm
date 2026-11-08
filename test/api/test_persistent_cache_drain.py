#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SSD 持久前缀缓存「正常停服 → native drain 自己排空」的可执行验收（方案 §15.2 / §13 关闭顺序）。

本文件验证三件事（§13 第 1–6 条、§15.2）：
  ① 正常 SIGTERM 后，native `CloseAdmission → Wait(timeout=300s，支持配置)` 自己把「已受理但未落盘」的
     检查点排空；进程退出码为 0，日志里**没有** `drain timeout`；
  ② 排空失败（store 仍有在途）时**必须 fail closed**：日志打印 `drain timeout`，进程以
     退出码 3 退出，编排层能识别"这次关停不完整"；不得静默成功；
  ③ 不丢已受理的检查点：重启同一缓存目录后重放，`cached_tokens > 0`（数据真的 durable 了）。

默认**全部跳过**（pytest 可收集、显示 skip 原因）。要真正跑起来需要 GPU + 模型，先设：
    export FASTLLM_DRAIN_ACCEPTANCE=1
可覆盖的环境变量：
    FASTLLM_DRAIN_BIN        ftllm 可执行文件
    FASTLLM_DRAIN_LIB        被测 libfastllm_tools.so
    FASTLLM_DRAIN_RUN_DIR    产物目录（默认 <repo>/.scratch-drain-test）
    FASTLLM_DRAIN_PORT       服务端口（默认 8099）
    FASTLLM_DRAIN_GPU_GROUP  GPU 组 0/1（默认 1，即 5-9 卡）
    FASTLLM_DRAIN_FORCE_TIMEOUT=1  额外启用"排空超时 fail closed"用例（需人工制造在途积压）

CPython 依赖：transformers（与验收脚本同环境）。CPU-only 机器上只跑 `test_shutdown_wiring_contract`。

单独运行：
    python3 test/api/test_persistent_cache_drain.py -v
"""
import importlib.util
import ast
import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.parent
ACC_PATH = ROOT / '测试' / 'ssd_write_acceptance.py'

BIN = Path(os.environ.get('FASTLLM_DRAIN_BIN', 'ftllm'))
LIB = Path(os.environ.get('FASTLLM_DRAIN_LIB',
                         str(REPO / 'build-rewrite/tools/ftllm/libfastllm_tools.so')))
RUN_DIR = Path(os.environ.get('FASTLLM_DRAIN_RUN_DIR', str(REPO / '.scratch-drain-test')))
PORT = int(os.environ.get('FASTLLM_DRAIN_PORT', '8099'))
GPU_GROUP = os.environ.get('FASTLLM_DRAIN_GPU_GROUP', '1')
PROMPT = ROOT / '测试' / 'prompt.txt'

# 正常停服的退出码契约（tools/fastllm_pytools/server.py 注册的 atexit 值）。
DRAIN_TIMEOUT_EXIT_CODE = 3
DRAIN_MESSAGE = 'SSD prefix cache drain timed out'

_ACC = None


def _load_acc():
    """按验收脚本的真实入口加载它；模块级常量在 import 期读环境变量，所以先设环境。"""
    global _ACC
    if _ACC is not None:
        return _ACC
    os.environ.setdefault('SSD_RUN', str(RUN_DIR))
    os.environ.setdefault('SSD_LIB', str(LIB))
    os.environ.setdefault('SSD_GPU_GROUP', GPU_GROUP)
    os.environ.setdefault('SSD_PORT', str(PORT))
    os.environ.setdefault('SSD_INTERVAL', '2048')
    os.environ.setdefault('SSD_MAX_TOKENS', '64')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    spec = importlib.util.spec_from_file_location('ssd_acc_for_drain', ACC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # 让被测命令指向本次测试显式指定的 bin/port/library。
    module.BIN = str(BIN)
    module.PORT = PORT
    module.URL = 'http://127.0.0.1:%d/v1/chat/completions' % PORT
    _ACC = module
    return module


def _missing_prerequisites():
    problems = []
    if os.environ.get('FASTLLM_DRAIN_ACCEPTANCE') != '1':
        problems.append('FASTLLM_DRAIN_ACCEPTANCE=1 未设置（GPU 行为用例默认 skip）')
    if not ACC_PATH.is_file():
        problems.append('缺少 %s' % ACC_PATH)
    if not BIN.is_file():
        problems.append('缺少 ftllm: %s' % BIN)
    if not LIB.is_file():
        problems.append('缺少 libfastllm_tools.so: %s' % LIB)
    if not PROMPT.is_file():
        problems.append('缺少 %s' % PROMPT)
    if shutil.which('nvidia-smi') is None:
        problems.append('nvidia-smi 不可用')
    try:
        import transformers  # noqa: F401
    except Exception as exc:                                     # noqa: BLE001
        problems.append('transformers 不可用: %r' % exc)
    return problems


SKIP_REASON = '; '.join(_missing_prerequisites())
RUNNABLE = not SKIP_REASON


class DrainHarness:
    """启动真实服务、发一个长请求，然后按需求发 SIGTERM / SIGKILL。"""

    def __init__(self, tag):
        self.tag = tag
        self.acc = _load_acc()
        self.out = RUN_DIR / tag
        self.cache = self.out / 'cache'
        self.logfile = self.out / 'server.log'
        self.proc = None
        self.out.mkdir(parents=True, exist_ok=True)

    # ---- 启动/就绪/库核验 -------------------------------------------------
    def start(self, logfile=None, timeout=900):
        self.acc.idle_check()
        logfile = Path(logfile or self.logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        env = self.acc.environment()
        # This acceptance test inspects checkpoint diagnostics that are
        # intentionally hidden from normal user-facing `info` logs.
        env['FASTLLM_PREFIX_CACHE_LOG_LEVEL'] = 'debug'
        os.environ.update(env)
        command = self.acc.server_cmd(str(self.cache), 4)
        with logfile.open('x') as handle:
            self.proc = subprocess.Popen(command, cwd=str(self.acc.REPO), env=env,
                                         stdout=handle, stderr=subprocess.STDOUT)
        (logfile.parent / 'server.pid').write_text(str(self.proc.pid) + '\n')
        deadline = time.monotonic() + timeout
        while 'Uvicorn running on' not in self.acc.log_text(logfile):
            if self.proc.poll() is not None:
                raise RuntimeError('服务启动即退出 code=%s: %s'
                                   % (self.proc.returncode, self.acc.log_text(logfile)[-4000:]))
            if time.monotonic() > deadline:
                raise RuntimeError('服务启动超时')
            time.sleep(1)
        # §13.6：证明实际加载的就是本测试指定的 .so（路径/inode/sha256）。
        evidence = self.acc.verify_loaded_library(
            self.proc.pid, self.acc.LIB, self.proc,
            [logfile.parent / 'loaded-library.json', self.out / 'loaded-library.json'])
        return evidence

    def ready(self):
        return 'Uvicorn running on' in self.acc.log_text(self.logfile)

    def log_text(self, logfile=None):
        return self.acc.log_text(Path(logfile or self.logfile))

    # ---- 请求 --------------------------------------------------------------
    def build_prompt(self, approx_tokens=2500):
        """用项目内 prompt.txt 造一个足够触发检查点的请求（不依赖具体分词结果）。"""
        raw = PROMPT.read_bytes()
        try:
            source = raw.decode('utf-8')
        except UnicodeDecodeError:
            source = raw.decode('gbk', errors='strict')
        chars = max(4096, approx_tokens * 2)
        return [{'role': 'user',
                 'content': 'SSD-drain-验收。请阅读以下资料，最后只回答收到。\n'
                            + source[:chars] + '\n资料结束。只回答：收到。'}]

    def send(self, messages, max_tokens=64, timeout=3600):
        body = {'model': 'Qwen3.8-27B-FP8', 'messages': messages, 'temperature': 0,
                'max_tokens': max_tokens, 'stream': False}
        request = urllib.request.Request(
            self.acc.URL, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response), body

    def wait_until(self, predicate, timeout=180, interval=0.5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self.log_text()):
                return True
            if self.proc is not None and self.proc.poll() is not None:
                return False
            time.sleep(interval)
        return False

    # ---- 收尾 --------------------------------------------------------------
    def stop(self, timeout=90):
        """只对自己保存的 PID 发 SIGTERM；不强杀，返回退出码。"""
        if self.proc is None or self.proc.poll() is not None:
            return self.proc.returncode if self.proc is not None else None
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise AssertionError('SIGTERM 后 %ss 未退出（未强杀，PID=%s）' % (timeout, self.proc.pid))
        return self.proc.returncode

    def force_timeout_scenario(self):
        """制造 drain 超时的钩子：本仓库没有故障注入开关，需人工制造 store 在途积压。

        返回 True 表示已制造（例如通过环境变量 + 外部压测），False 表示无法制造，
        让调用方 skip 而不是伪造通过。
        """
        return os.environ.get('FASTLLM_DRAIN_FORCE_TIMEOUT') == '1'


def _legit_skip(reason):
    return any(marker in reason for marker in
               ('checkpoint_queue_limit', 'host_budget', 'gpu_pin_budget', 'queue_limit'))


@unittest.skipUnless(RUNNABLE, SKIP_REASON)
class PersistentCacheDrainBehaviorTest(unittest.TestCase):
    """真实服务行为用例；默认 skip（见模块头部前置条件）。"""

    @classmethod
    def setUpClass(cls):
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    def test_sigterm_drains_admitted_checkpoints_and_exits_zero(self):
        harness = DrainHarness('sigterm-drain')
        if harness.out.exists():
            shutil.rmtree(harness.out)
        try:
            harness.start()
            _result, _body = harness.send(harness.build_prompt())
            admitted = harness.wait_until(
                lambda text: 'checkpoint=yes' in text, timeout=180)
            self.assertTrue(admitted, '请求返回后仍未见任何已受理检查点（enqueue ... checkpoint=yes）')
            # 关键：不等脚本排空，立刻按关闭顺序发 SIGTERM，让 native drain 自己收尾。
            code = harness.stop()
            text = harness.log_text()
        finally:
            if harness.proc is not None and harness.proc.poll() is None:
                harness.proc.kill()
        self.assertEqual(code, 0, '正常停服应以退出码 0 结束，实际 %r\n%s' % (code, text[-4000:]))
        self.assertNotIn('drain timeout', text,
                         '正常停服不应出现 drain timeout（说明存有在途未排空）')
        totals = harness.acc.inspect_log(text)
        failures = sum(1 for reason in totals['write_skipped'] if not _legit_skip(reason))
        self.assertGreaterEqual(totals['enqueued'], 1, '至少应有一个已受理检查点')
        self.assertEqual(
            totals['enqueued'],
            totals['committed'] + totals['already_present'] + failures,
            '已受理检查点必须全部 committed/already_present（不丢点）: %r' % totals)
        durable = harness.acc.durable_count(text)
        self.assertGreaterEqual(
            durable, totals['committed'] - totals['already_present'],
            '每个新提交的检查点都应有 v4-durable 落盘行: committed=%d durable=%d already_present=%d'
            % (totals['committed'], durable, totals['already_present']))

    def test_drain_timeout_fails_closed_with_exit_code_three(self):
        harness = DrainHarness('drain-timeout')
        if not harness.force_timeout_scenario():
            self.skipTest('无法制造 drain 超时（需 FASTLLM_DRAIN_FORCE_TIMEOUT=1 并人工制造在途积压）；'
                          '不伪造通过')
        if harness.out.exists():
            shutil.rmtree(harness.out)
        try:
            harness.start()
            _result, _body = harness.send(harness.build_prompt())
            harness.wait_until(lambda text: 'checkpoint=yes' in text, timeout=180)
            code = harness.stop()
            text = harness.log_text()
        finally:
            if harness.proc is not None and harness.proc.poll() is None:
                harness.proc.kill()
        self.assertIn(DRAIN_MESSAGE, text, '排空超时必须打印 fail-closed 日志')
        self.assertEqual(code, DRAIN_TIMEOUT_EXIT_CODE,
                         '排空超时必须用退出码 %d 退出（不得静默成功），实际 %r'
                         % (DRAIN_TIMEOUT_EXIT_CODE, code))

    def test_admitted_checkpoint_survives_drain_and_restart(self):
        harness = DrainHarness('drain-restart')
        if harness.out.exists():
            shutil.rmtree(harness.out)
        restart_log = None
        try:
            harness.start()
            messages = harness.build_prompt()
            first, _body = harness.send(messages)
            harness.wait_until(lambda text: 'checkpoint=yes' in text, timeout=180)
            code = harness.stop()
            self.assertEqual(code, 0, '第一次停服应正常退出')
            # 重启同一缓存目录，重放同一请求 → 必须命中 SSD 缓存（证明已受理的检查点真的落盘）。
            restart_log = harness.out / 'server-restart.log'
            harness.start(logfile=restart_log)
            second, _body = harness.send(messages)
            cached = second.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens', 0)
            text = harness.log_text(restart_log)
            self.assertGreater(cached, 0,
                               '重启后重放应命中（cached_tokens>0），实际 usage=%r\n%s'
                               % (second.get('usage'), text[-4000:]))
            self.assertGreaterEqual(len(re.findall(r'restored (\d+)', text)), 1,
                                    '重启日志应有 restore 事件')
        finally:
            if harness.proc is not None and harness.proc.poll() is None:
                harness.proc.kill()
            if restart_log is not None and restart_log.exists() and harness.proc is not None:
                # 保留日志与缓存证据，只记录最终状态，不删除。
                sys.stdout.write('drain-restart 证据保留在 %s\n' % harness.out)


class ShutdownWiringContractTest(unittest.TestCase):
    """CPU-only 结构契约：校验 server.py 的关闭顺序接线（不替代 GPU 行为用例）。"""

    def test_server_shutdown_orders_admission_close_before_drain(self):
        import ast
        source = (REPO / 'tools' / 'fastllm_pytools' / 'server.py').read_text()
        tree = ast.parse(source)
        shutdown = next((node for node in ast.walk(tree)
                         if isinstance(node, ast.AsyncFunctionDef) and node.name == 'shutdown'), None)
        self.assertIsNotNone(shutdown, 'server.py 找不到 FastLLMUvicornServer.shutdown')
        # §13 第 2 条：关闭准入必须在排空之前（顺序由源码行号保证）。
        close_line = next(node.lineno for node in ast.walk(shutdown)
                          if getattr(node, 'attr', None) == 'close_persistent_prefix_admission')
        wait_line = next(node.lineno for node in ast.walk(shutdown)
                         if getattr(node, 'attr', None) == 'wait_persistent_prefix_cache')
        self.assertLess(close_line, wait_line,
                        'close_persistent_prefix_admission 必须在 wait_persistent_prefix_cache 之前')
        # §13 第 6 条 + S1-3：超时必须 fail closed（退出码 3），不能只打 warning。
        self.assertIn('drain timed out', source)
        self.assertRegex(source, r'_exit\(3\)')




def method_source(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method_name)
    return ast.get_source_segment(path.read_text(), method) or ''


class NativeDrainBridgeTests(unittest.TestCase):
    def test_model_exposes_native_drain_and_stats(self):
        source = REPO / 'tools/fastllm_pytools/llm.py'
        wait = method_source(source, 'model', 'wait_persistent_prefix_cache')
        stats = method_source(source, 'model', 'persistent_prefix_cache_statistics')
        self.assertIn('wait_persistent_prefix_cache', wait)
        self.assertIn('ctypes.c_bool', wait)
        self.assertIn('get_persistent_prefix_cache_statistics', stats)
        self.assertIn('json.loads', stats)

    def test_server_shutdown_calls_drain(self):
        source = (REPO / 'tools/fastllm_pytools/server.py').read_text()
        method = method_source(REPO / 'tools/fastllm_pytools/server.py',
                              'FastLLMUvicornServer', 'shutdown')
        self.assertIn('wait_persistent_prefix_cache', method)
        self.assertIn('FASTLLM_PREFIX_CACHE_DRAIN_TIMEOUT_S', method)
        self.assertIn('SSD prefix cache drain timed out', source)


class ShutdownTimeoutBehaviorTest(unittest.IsolatedAsyncioTestCase):
    """Run the real shutdown method with a fake native backend and no model/GPU."""

    async def run_shutdown(self, configured_timeout=None, finished=True):
        events = []

        async def to_thread(func, *args):
            return func(*args)

        class ParentServer:
            async def shutdown(self, sockets=None):
                events.append(('server', sockets))

        class Backend:
            def close_persistent_prefix_admission(self):
                events.append(('close', None))

            def wait_persistent_prefix_cache(self, timeout):
                events.append(('wait', timeout))
                return finished

        path = REPO / 'tools/fastllm_pytools/server.py'
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == 'FastLLMUvicornServer')
        cls.body = [node for node in cls.body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == 'shutdown']
        cls.bases = [ast.Name(id='ParentServer', ctx=ast.Load())]
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        namespace = {'ParentServer': ParentServer,
                     'asyncio': SimpleNamespace(to_thread=to_thread),
                     'logging': logging, 'os': os,
                     'fastllm_completion': SimpleNamespace(model=Backend())}
        exec(compile(module, str(path), 'exec'), namespace)
        environment = dict(os.environ)
        environment.pop('FASTLLM_PREFIX_CACHE_DRAIN_TIMEOUT_S', None)
        if configured_timeout is not None:
            environment['FASTLLM_PREFIX_CACHE_DRAIN_TIMEOUT_S'] = configured_timeout
        with patch.dict(os.environ, environment, clear=True), \
                patch('atexit.register') as register, patch('os._exit') as exit_process:
            await namespace['FastLLMUvicornServer']().shutdown(sockets=['listener'])
            if finished:
                register.assert_not_called()
            else:
                register.assert_called_once()
                register.call_args.args[0]()
                exit_process.assert_called_once_with(DRAIN_TIMEOUT_EXIT_CODE)
        self.assertEqual([event[0] for event in events], ['server', 'close', 'wait'])
        self.assertEqual(events[0][1], ['listener'])
        return events[-1][1]

    async def test_default_timeout_and_successful_drain(self):
        self.assertEqual(await self.run_shutdown(), 300.0)

    async def test_configured_timeout_reaches_native_backend(self):
        self.assertEqual(await self.run_shutdown('45.5'), 45.5)

    async def test_invalid_timeout_falls_back_and_small_timeout_is_clamped(self):
        for value, expected in [('invalid', 300.0), ('0', 1.0), ('-10', 1.0)]:
            with self.subTest(value=value):
                self.assertEqual(await self.run_shutdown(value), expected)

    async def test_failed_drain_registers_exit_code_three(self):
        self.assertEqual(await self.run_shutdown('2', finished=False), 2.0)




if __name__ == '__main__':
    unittest.main(verbosity=2)

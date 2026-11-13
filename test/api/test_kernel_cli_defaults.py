"""Startup defaults and saved-config precedence without allocating weights."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from fastllm_pytools import kernel_policy
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model


class StopBeforeWeights(Exception):
    pass


class KernelCliDefaultsTest(unittest.TestCase):
    def startup_args(self, options, saved=None):
        resolve = kernel_policy.resolve_mode
        def stop(args):
            resolve(args)
            raise StopBeforeWeights
        with tempfile.TemporaryDirectory() as directory:
            model = 'unused-model'
            if saved is not None:
                model = str(Path(directory) / 'serve.json')
                Path(model).write_text(json.dumps({'path': 'unused-model', **saved}))
            args = make_normal_parser('test').parse_args([model, *options])
            with patch.object(kernel_policy, 'resolve_mode', side_effect=stop), \
                    self.assertRaises(StopBeforeWeights):
                make_normal_llm_model(args)
            return args

    def test_auto_and_autotune_default_to_quick_at_model_startup(self):
        for flags, mode in [((), 'off'), (('--auto',), 'quick'), (('--autotune',), 'quick')]:
            with self.subTest(flags=flags):
                self.assertEqual(self.startup_args(flags).kernel_tune, mode)

    def test_explicit_mode_is_respected_in_either_cli_order(self):
        for mode in ('off', 'quick', 'cache'):
            for flags in [('--auto', '--kernel-tune', mode), ('--kernel-tune', mode, '--auto')]:
                with self.subTest(flags=flags):
                    self.assertEqual(self.startup_args(flags).kernel_tune, mode)

    def test_saved_auto_and_mode_are_resolved_before_loading(self):
        for saved, mode in [({'auto': True}, 'quick'),
                            ({'auto': True, 'kernel_tune': 'off'}, 'off'),
                            ({'auto': True, 'kernel_tune': 'cache'}, 'cache')]:
            with self.subTest(saved=saved):
                self.assertEqual(self.startup_args([], saved).kernel_tune, mode)

    def test_cli_mode_overrides_saved_auto_mode(self):
        for saved, cli in [('quick', 'off'), ('off', 'quick'), ('quick', 'cache')]:
            with self.subTest(saved=saved, cli=cli):
                args = self.startup_args(['--kernel-tune', cli], {'auto': True, 'kernel_tune': saved})
                self.assertEqual(args.kernel_tune, cli)

    def test_policy_configuration_accepts_unresolved_parser_defaults(self):
        for flags, mode in [(('--auto',), 'quick'), (('--auto', '--kernel-tune', 'cache'), 'cache')]:
            args = make_normal_parser('test').parse_args(['model', *flags])
            with patch.object(kernel_policy, '_active', None), \
                    patch.object(kernel_policy, 'KernelPolicy') as policy:
                self.assertIs(kernel_policy.configure(args), policy.return_value)
                policy.assert_called_once_with(mode, '', '')
        args = make_normal_parser('test').parse_args(['model', '--auto', '--kernel-tune', 'off'])
        with patch.object(kernel_policy, '_active', None), \
                patch.object(kernel_policy, 'KernelPolicy') as policy:
            self.assertIsNone(kernel_policy.configure(args))
            policy.assert_not_called()

    def test_low_memory_opt_out_is_explicit_and_overrides_saved_true(self):
        for flag in ('--no-low-gpu-mem', '--no_low_gpu_mem'):
            with self.subTest(flag=flag):
                args = self.startup_args([flag], {'auto': True, 'low_gpu_mem': True})
                self.assertFalse(args.low_gpu_mem)
                self.assertIn('low_gpu_mem', args._auto_explicit)
        args = make_normal_parser('test').parse_args(['model', '--auto'])
        self.assertFalse(args.low_gpu_mem)
        self.assertNotIn('low_gpu_mem', args._auto_explicit)

    def test_conflicting_low_memory_flags_are_rejected(self):
        for flags in [('--low_gpu_mem', '--no-low-gpu-mem'), ('--no-low-gpu-mem', '--low_gpu_mem')]:
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                make_normal_parser('test').parse_args(['model', *flags])


if __name__ == '__main__':
    unittest.main()

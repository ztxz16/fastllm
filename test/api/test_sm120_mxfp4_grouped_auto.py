import copy
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tools')))
from fastllm_pytools.util import _configure_sm120_mxfp4_grouped_triton

AUTO = 'FASTLLM_CUDA_TRITON_MXFP4_GROUPED_AUTO'
CONFIG = {
    'model_type': 'deepseek_v41',
    'text_config': {'hidden_size': 5120, 'moe_intermediate_size': 2304,
                    'num_experts_per_tok': 6, 'dspark_num_experts_per_tok': 3,
                    'swiglu_limit': 10.0},
    'quantization_config': {'expert_dtype': 'fp4', 'scale_fmt': 'ue8m0',
                            'weight_block_size': [32, 32]},
}

def args(**changes):
    values = dict(auto=True, dspark=5, max_batch=1, tp='off', device='cuda:0', moe_device='numa',
                  moe_cuda_cache=48 << 30, path='/models/native')
    values.update(changes)
    return SimpleNamespace(**values)

class Sm120GroupedAutoTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        for name, value in [('_is_nvidia_cuda_platform', True),
                            ('_nvidia_cuda_compute_capabilities', {0: 120}),
                            ('_find_triton_python', '/test/venv/bin/python')]:
            mock = patch('fastllm_pytools.util.' + name, return_value=value)
            setattr(self, name, mock.start())
            self.addCleanup(mock.stop)
        stdout = redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)

    def configure(self, options=None, config=None):
        return _configure_sm120_mxfp4_grouped_triton(options or args(), CONFIG if config is None else config)

    def test_selects_only_grouped_for_the_eligible_auto_model(self):
        self.assertEqual(self.configure(), '/test/venv/bin/python')
        self.assertEqual(dict(os.environ), {
            AUTO: '1', 'FASTLLM_CUDA_TRITON_PYTHON': '/test/venv/bin/python'})
        self._nvidia_cuda_compute_capabilities.assert_called_once_with([0])

    def test_respects_both_explicit_switches(self):
        for name in ('FASTLLM_CUDA_TRITON', 'FASTLLM_CUDA_TRITON_MXFP4_GROUPED'):
            for value in ('0', '1', ''):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name:value}, clear=True):
                    self.assertEqual(self.configure(), '')
                    self.assertEqual(dict(os.environ), {name:value})
        self._find_triton_python.assert_not_called()

    def test_does_not_expand_to_unmeasured_architectures_or_unknown_devices(self):
        for capabilities in ({0:75}, {0:89}, {0:100}, {0:121}, {}):
            with self.subTest(capabilities=capabilities):
                self._nvidia_cuda_compute_capabilities.return_value = capabilities
                self.assertEqual(self.configure(), '')
                self.assertNotIn(AUTO, os.environ)
        self._find_triton_python.assert_not_called()

    def test_accepts_disabled_tensor_parallel_spellings(self):
        for tp in ('', 'off', 'false', 'none', 'disable', None):
            with self.subTest(tp=tp):
                self.assertTrue(self.configure(args(tp=tp)))
                self.assertEqual(os.environ[AUTO], '1')

    def test_queries_the_selected_cuda_ordinal(self):
        self._nvidia_cuda_compute_capabilities.return_value = {3:120}
        self.assertTrue(self.configure(args(device='cuda:3')))
        self._nvidia_cuda_compute_capabilities.assert_called_once_with([3])

    def test_ineligible_execution_modes_do_not_probe_or_enable(self):
        for changes in ({'auto':False}, {'dspark':0}, {'dspark':8}, {'max_batch':2},
                        {'tp':'cuda:0,1'}, {'device':'cuda:0,1'},
                        {'device':"{'cuda:0': 1, 'cuda:1': 1}"}, {'device':'cpu'},
                        {'moe_device':'cuda:0'}, {'moe_cuda_cache':0}, {'path':'/models/a.gguf'}):
            with self.subTest(changes=changes):
                self.assertEqual(self.configure(args(**changes)), '')
                self.assertNotIn(AUTO, os.environ)
        self._find_triton_python.assert_not_called()

    def test_rejects_different_weight_formats_and_model_shapes(self):
        variants = [{}, {'model_type':'qwen3_moe'}]
        for section, key, value in [('text_config','hidden_size',4096),
                                    ('text_config','moe_intermediate_size',2048),
                                    ('text_config','num_experts_per_tok',8),
                                    ('text_config','dspark_num_experts_per_tok',6),
                                    ('text_config','swiglu_limit',7.0),
                                    ('quantization_config','expert_dtype','fp8'),
                                    ('quantization_config','scale_fmt','e4m3'),
                                    ('quantization_config','weight_block_size',[16,16])]:
            config=copy.deepcopy(CONFIG);config[section][key]=value;variants.append(config)
        for config in variants:
            with self.subTest(config=config):
                self.assertEqual(self.configure(config=config), '')
                self.assertNotIn(AUTO, os.environ)
        self._find_triton_python.assert_not_called()

    def test_missing_triton_keeps_native_cuda_and_compiler_override(self):
        os.environ['FASTLLM_CUDA_TRITON_PYTHON'] = '/user/python'
        self._find_triton_python.return_value = ''
        self.assertEqual(self.configure(), '')
        self.assertEqual(dict(os.environ), {'FASTLLM_CUDA_TRITON_PYTHON':'/user/python'})

    def test_non_nvidia_build_does_not_probe_cuda(self):
        self._is_nvidia_cuda_platform.return_value = False
        self.assertEqual(self.configure(), '')
        self._nvidia_cuda_compute_capabilities.assert_not_called()

    def test_next_model_rechecks_auto_policy(self):
        self.assertTrue(self.configure())
        self.assertEqual(self.configure(args(auto=False)), '')
        self.assertNotIn(AUTO, os.environ)

    def test_later_explicit_disable_clears_auto_selection(self):
        self.assertTrue(self.configure())
        os.environ['FASTLLM_CUDA_TRITON_MXFP4_GROUPED'] = '0'
        self.assertEqual(self.configure(), '')
        self.assertNotIn(AUTO, os.environ)
        self.assertEqual(os.environ['FASTLLM_CUDA_TRITON_MXFP4_GROUPED'], '0')

if __name__ == '__main__':
    unittest.main()

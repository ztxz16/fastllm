"""Launcher hardware must agree with the GPUs visible to inference children."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastllm_pytools.ui_hardware import _gpu_info


class VisibleGPUInfoTest(unittest.TestCase):
    def query(self, visible=None):
        output = '\n'.join(f'{i}, GPU {i}, 24000, 20000, 0, 40, 580, GPU-abc{i}' for i in range(3))
        with patch.dict(os.environ, {}, clear=True), \
                patch('fastllm_pytools.ui_hardware.shutil.which', return_value='/bin/nvidia-smi'), \
                patch('fastllm_pytools.ui_hardware.subprocess.run',
                      return_value=SimpleNamespace(returncode=0, stdout=output)):
            if visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = visible
            return _gpu_info()

    def test_visible_devices_and_cuda_logical_indices(self):
        self.assertEqual(len(self.query()), 3)
        for visible in ('', '-1', 'invalid'):
            self.assertEqual(self.query(visible), [])
        gpus = self.query('2,0')
        self.assertEqual([gpu['name'] for gpu in gpus], ['GPU 2', 'GPU 0'])
        self.assertEqual([gpu['index'] for gpu in gpus], ['0', '1'])
        self.assertEqual([gpu['name'] for gpu in self.query('GPU-abc1')], ['GPU 1'])
        self.assertEqual(len(self.query('0,-1,2')), 1)
        self.assertEqual(len(self.query('0,0')), 1)

"""Build the standalone trace fixture; validate records without CUDA or a model."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class InferenceTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++')
        if not compiler:
            raise unittest.SkipTest('C++ compiler is required')
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.binary = cls.root/'trace-test'
        repo = Path(__file__).resolve().parents[2]
        subprocess.run([compiler, '-std=c++17', '-pthread', '-I', str(repo/'include'),
                        str(repo/'test/basic/test_inference_trace.cpp'), '-o', str(cls.binary)],
                       check=True, capture_output=True, text=True)

    def test_disabled_by_default(self):
        env = {k: v for k, v in os.environ.items() if k != 'FASTLLM_INFERENCE_TRACE'}
        result = subprocess.run([str(self.binary)], env=env, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, '')
        self.assertEqual(result.stderr, '')

    def test_chunks_contexts_timing_accounting_and_failures(self):
        path = self.root/'trace.jsonl'
        subprocess.run([str(self.binary)], env={**os.environ, 'FASTLLM_INFERENCE_TRACE': str(path)},
                       check=True, capture_output=True, text=True)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        forwards = [r for r in rows if r['event'] == 'forward']
        steps = [r for r in rows if r['event'] == 'scheduler_step']
        self.assertEqual(len(forwards), 4)
        self.assertEqual(len(steps), 3)
        self.assertEqual([r['input_tokens'] for r in forwards], [8, 16, 2, 1])
        self.assertEqual([r['context_after'] for r in forwards], [[108], [124], [125, 251], [1]])
        self.assertEqual(steps[0]['forward_count'], 2)
        self.assertEqual(steps[0]['cached_prefix_tokens'], [100])
        self.assertEqual(forwards[2]['mode'], 'cuda_graph')
        self.assertAlmostEqual(forwards[2]['gpu_model_max_ms'], .3)
        self.assertFalse(forwards[-1]['completed'])
        self.assertFalse(steps[-1]['completed'])
        for row in forwards:
            self.assertAlmostEqual(sum(row['timings_ms'].values()) + row['tail_ms'], row['forward_ms'], places=4)
            if row['forward_ms'] > .001:
                self.assertAlmostEqual(row['tokens_per_second'] * row['forward_ms'] / 1000,
                                       row['input_tokens'], places=2)
        for row in steps:
            self.assertAlmostEqual(row['outside_forward_ms'] + row['forward_api_ms'], row['total_ms'], places=4)
            self.assertEqual(row['forward_count'], sum(f['step_id'] == row['step_id'] for f in forwards))
            self.assertGreaterEqual(row['forward_trace_emit_ms'], 0)

    def test_unwritable_path_does_not_break_inference(self):
        result = subprocess.run([str(self.binary)],
            env={**os.environ, 'FASTLLM_INFERENCE_TRACE': str(self.root/'missing'/'trace.jsonl')},
            capture_output=True, text=True, check=True)
        self.assertIn('Cannot open inference trace', result.stderr)


if __name__ == '__main__':
    unittest.main()

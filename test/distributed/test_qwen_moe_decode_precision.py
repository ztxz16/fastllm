"""缓存定位探针的CPU检查：教师强制历史、缓存传递、阈值边界及失败报告。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import probe_qwen_moe_decode_precision as probe


def source():
    """/** 采样Token与教师强制Token不同，确保探针重放实际输入而不是模型自己生成的历史。 */"""
    report = dict(dp_rank=0, request=dict(token_ids=[1, 2], max_new_tokens=2, eos_token_ids=[],
                  temperature=0., top_k=0, top_p=1., seed=1), teacher_forced_tokens=[9],
                  steps=[dict(logits=[0., 0.], token_id=8, execution='eager'),
                         dict(logits=[.3, 0.], token_id=8, execution='eager')])
    reference = dict(rows=[dict(ids=[1, 2], logits=[0., 0.]), dict(ids=[1, 2, 9], logits=[0., 0.])])
    return report, reference


class DecodePrecisionTests(unittest.TestCase):
    def test_true_violations_are_separate_from_largest_absolute_error(self):
        row = probe.comparison('logits', [12.9, .06], [12.8, 0.])
        self.assertEqual(row['status'], 'FAIL')
        self.assertEqual(row['worst_index'], 0)
        self.assertEqual([x['vocab_id'] for x in row['failed_vocab']], [1])
        self.assertEqual(row['failed_elements'], 1)
        self.assertEqual(row['failed_vocab'][0]['threshold'], .05)
        self.assertEqual(probe.comparison('boundary', [.05], [0.])['status'], 'PASS')

    def test_invalid_logits_are_rejected(self):
        for actual, expected in (([], []), ([1.], []), ([1., 2.], [1.]), ([float('nan')], [0.])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                probe.comparison('invalid', actual, expected)

    def test_teacher_forced_plan_and_prefill_failure_rejection(self):
        report, reference = source()
        plan = probe.replay_plan(report, reference)
        self.assertEqual(plan['ids'], [1, 2, 9])
        self.assertEqual(plan['suffix'], [9])
        self.assertEqual(plan['step'], 1)
        report['steps'][0]['logits'] = [.3, 0.]
        with self.assertRaisesRegex(ValueError, 'failing decode step'):
            probe.replay_plan(report, reference)

    def test_hf_cache_carries_history_and_checks_length(self):
        import torch

        class Model:
            """/** 独立求和oracle；检查完整输入与缓存逐Token输入是否数学等价。 */"""
            def __init__(self):
                self.calls, self.bad = [], False

            def get_input_embeddings(self):
                return SimpleNamespace(weight=torch.empty(1))

            def __call__(self, input_ids, use_cache, past_key_values=None):
                ids = input_ids[0].tolist()
                self.calls.append((ids, use_cache, past_key_values))
                history = ([] if past_key_values is None else past_key_values.ids) + ids
                past = SimpleNamespace(ids=history, get_seq_length=lambda: -1 if self.bad else len(history))
                return SimpleNamespace(logits=torch.tensor([[[float(sum(history))]]]), past_key_values=past)

        plan = probe.replay_plan(*source())
        model = Model()
        result = probe.hf_replay(model, torch, plan)
        self.assertEqual(result['full'], [12.])
        self.assertEqual(result['cached'], result['full'])
        self.assertEqual([c[0] for c in model.calls], [[1, 2, 9], [1, 2], [9]])
        self.assertEqual([c[1] for c in model.calls], [False, True, True])
        self.assertIsNone(model.calls[1][2])
        self.assertEqual(model.calls[2][2].ids, [1, 2])
        self.assertEqual([s['cache_length'] for s in result['steps']], [2, 3])
        model.bad = True
        with self.assertRaisesRegex(RuntimeError, 'cache length mismatch'):
            probe.hf_replay(model, torch, plan)

    def test_native_modes_use_public_generate_with_same_history(self):
        report, reference = source()
        plan = probe.replay_plan(report, reference)
        args = SimpleNamespace(build=Path('/unused'), model=Path('/model'), timeout=60)
        with patch.object(probe, 'QwenMoeProcessSession') as session_type:
            session = session_type.return_value.__enter__.return_value
            session.generate.return_value = dict(status='PASS', dp_rank=0, steps=[
                dict(execution='eager', joint_round=dict(actions=[1, 0, 0, 0], submitted_token_counts=[2, 0, 0, 0])),
                dict(execution='eager', joint_round=dict(actions=[2, 0, 0, 0], submitted_token_counts=[1, 0, 0, 0]))])
            probe.native_replay(args, Path('/cached'), report, plan, True)
            request = session.generate.call_args.args[0]
            self.assertEqual(request['token_ids'], [1, 2])
            self.assertEqual(request['max_new_tokens'], 2)
            self.assertEqual(session.generate.call_args.kwargs['teacher_forced_tokens'], [9])
            self.assertFalse(session_type.call_args.kwargs['prefix_cache'])
            self.assertFalse(session_type.call_args.kwargs['cuda_graph'])
            session.generate.return_value = dict(status='PASS', dp_rank=0, steps=[
                dict(execution='eager', joint_round=dict(actions=[1, 0, 0, 0], submitted_token_counts=[3, 0, 0, 0]))])
            probe.native_replay(args, Path('/fresh'), report, plan, False)
            self.assertEqual(session.generate.call_args.args[0]['token_ids'], [1, 2, 9])
            self.assertEqual(session.generate.call_args.args[0]['max_new_tokens'], 1)
            self.assertIsNone(session.generate.call_args.kwargs['teacher_forced_tokens'])
            session.generate.return_value['steps'][0]['execution'] = 'decode_cuda_graph'
            with self.assertRaisesRegex(RuntimeError, 'execution path'):
                probe.native_replay(args, Path('/bad'), report, plan, False)

    def test_main_keeps_all_comparisons_and_partial_failure_reports(self):
        for reference_exit in (0, 7):
            with self.subTest(reference_exit=reference_exit), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / 'CMakeCache.txt').write_text('CUDA_ARCH:STRING=70')
                for name in ('fastllm_moe_generate', 'fastllm_tp_worker'):
                    (root / name).write_bytes(b'CPU mock, never executed')
                report, reference = source()
                (root / 'request.json').write_text(json.dumps(report))
                (root / 'transformers-reference.json').write_text(json.dumps(reference))
                (root / 'summary.json').write_text(json.dumps(dict(metadata=dict(git='source'))))

                def run_reference(command, *_):
                    Path(command[-1]).write_text(json.dumps(dict(ids=[1, 2, 9], full=[0., 0.],
                                                                 cached=[0., 0.], metadata={})))
                    return reference_exit

                argv = ['probe', '--model', temp, '--build', temp, '--result-dir', temp,
                        '--request', 'request.json', '--reference-python', '/mock/python', '--output', str(root / 'out')]
                with patch.object(probe, 'capture', return_value='CPU mock'), patch('sys.argv', argv), \
                        patch.object(probe, 'run_logged', side_effect=run_reference), \
                        patch.object(probe, 'native_replay', return_value=dict(steps=[dict(logits=[.2, 0.])])) as native, \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(probe.main(), 1)
                output = next((root / 'out').iterdir())
                for extension in ('json', 'md', 'csv', 'xlsx'):
                    self.assertTrue((output / f'summary.{extension}').is_file())
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual(summary['status'], 'FAIL')
                self.assertEqual(summary['cases'][0]['case'], 'saved_native_vs_saved_hf')
                if reference_exit:
                    self.assertEqual(native.call_count, 0)
                    self.assertFalse(summary['complete'])
                    self.assertEqual(summary['cases'][-1]['case'], 'probe_error')
                    self.assertTrue((output / 'error.log').exists())
                else:
                    self.assertTrue(summary['complete'])
                    self.assertEqual(len(summary['cases']), 8)
                    self.assertEqual(native.call_count, 2)


if __name__ == '__main__':
    unittest.main()

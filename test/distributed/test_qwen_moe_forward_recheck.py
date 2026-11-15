"""历史前向数据复核：无需GPU，保留真实失败，拒绝失配证据且不覆盖历史报告。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import probe_qwen_moe_forward_precision as probe
import recheck_qwen_moe_forward_precision as recheck
from test_qwen_moe_forward_precision import reference_fixture, trace_fixture


def saved_probe(root):
    """/** 经真实报告流程保存小规模轨迹，再恢复旧版未处理JSON格式的精确门禁。 */"""
    source, diagnostic, build, model = [root / name for name in ('source', 'diagnostic', 'build', 'model')]
    for directory in (source, diagnostic, build, model):
        directory.mkdir()
    (model / 'config.json').write_text(json.dumps(dict(model_type='qwen2_moe', hidden_size=2,
        num_hidden_layers=1, vocab_size=2)))
    for name in ('CMakeCache.txt', 'fastllm_moe_generate', 'fastllm_tp_worker'):
        (build / name).write_text('CPU fixture; never executable')
    replay = [struct.unpack('<f', struct.pack('<f', v))[0] for v in (.1, .3)]
    wire = [float(format(v, '.12f')) for v in replay]
    original = dict(dp_rank=0, request=dict(token_ids=[1, 2]),
        steps=[dict(logits=wire, execution='eager', token_id=0)])
    reference = dict(rows=[dict(ids=[1, 2], logits=[0., 0.])])
    (source / 'request.json').write_text(json.dumps(original))
    (source / 'transformers-reference.json').write_text(json.dumps(reference))
    (diagnostic / 'summary.json').write_text(json.dumps(dict(metadata=dict(source_dir=str(source),
        request='request.json', target=probe.failing_history(original, reference), rank=0,
        git='original', completed_tasks=3), cases=[dict(case='saved_full_model_logits', status='FAIL')])))
    (diagnostic / 'layer-inputs.json').write_text('[[1, 2], [], [], []]')
    layers = reference_fixture(1)
    layers['metadata'] = dict(dtype='float16')
    (diagnostic / 'reference.json').write_text(json.dumps(layers))

    def native(args, output, original, ids, trace=None):
        """/** 三次会话共用确定性日志；只在采集会话生成FP16快照。 */"""
        if trace is not None:
            trace_fixture(trace)
        return wire

    def head(args, config, hf, native, output):
        """/** 模拟已结束的GPU输出头采集，留下CPU复核所需的原始文件。 */"""
        values = {key: [0., 0.] for key in ('hf_norm', 'hf_norm_on_native', 'hf_head_full',
            'hf_head_last', 'native_head_on_hf', 'hf_head_on_native_norm')}
        values['native_head_replay'] = replay
        metadata = dict(device='CPU fixture')
        (output / 'head-replay.json').write_text(json.dumps(dict(metadata=metadata, values=values)))
        return metadata, values

    argv = ['probe', '--diagnostic', str(diagnostic), '--model', str(model), '--build', str(build),
            '--output', str(root / 'reports')]
    with patch('sys.argv', argv), patch.object(probe, 'capture', return_value='original'), \
            patch.dict('os.environ', {probe.TRACE_ENV: ''}), \
            patch.object(probe, 'native_forward', side_effect=native), \
            patch.object(probe, 'head_probe', side_effect=head), redirect_stdout(io.StringIO()):
        if probe.main() != 1:
            raise AssertionError('the saved real numerical failure must remain FAIL')
    output, = (root / 'reports').iterdir()
    report = json.loads((output / 'summary.json').read_text())
    report['cases'] = [probe.compare(row['case'], replay, wire, 2, exact=True)
                      if row['case'] == 'native_head_reproduces_traced_logits' else row for row in report['cases']]
    report['metadata']['head_attribution_valid'] = False
    (output / 'summary.json').write_text(json.dumps(report))
    return output


class ForwardRecheckTests(unittest.TestCase):
    def test_saved_data_recheck_fixes_only_wire_gate_and_preserves_failed_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = saved_probe(root)
            before = {str(p): probe.digest(p) for p in root.rglob('*') if p.is_file()}
            argv = ['recheck', '--probe', str(source), '--output', str(root / 'rechecks')]
            with patch('sys.argv', argv), patch.object(recheck, 'capture', return_value='analysis') as capture, \
                    patch.object(probe, 'native_forward', side_effect=AssertionError('GPU must not run')), \
                    patch.object(probe, 'head_probe', side_effect=AssertionError('weights must not load')), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(recheck.main(), 1)
            capture.assert_called_once_with(['git', 'rev-parse', 'HEAD'])
            output, = (root / 'rechecks').iterdir()
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertTrue(report['metadata']['trace_attribution_valid'])
            self.assertTrue(report['metadata']['head_attribution_valid'])
            self.assertEqual(report['metadata']['git'], 'original')
            self.assertEqual(report['metadata']['recheck_git'], 'analysis')
            self.assertEqual(report['metadata']['head_gate_wire_max_abs'], 0.)
            self.assertGreater(report['metadata']['head_gate_raw_max_abs'], 0.)
            rows = {row['case']: row for row in report['cases']}
            self.assertEqual(rows['saved_native_vs_saved_hf']['status'], 'FAIL')
            self.assertEqual(rows['native_head_reproduces_traced_logits']['status'], 'PASS')
            self.assertEqual(len(report['metadata']['first_failure_by_token']), 2)
            for name, sha in before.items():
                self.assertEqual(probe.digest(Path(name)), sha)
            for directory in (output, output / 'tokens'):
                for suffix in ('json', 'md', 'csv', 'xlsx'):
                    self.assertTrue((directory / f'summary.{suffix}').is_file())

    def test_recheck_rejects_changed_trace_logits_reference_and_missing_head(self):
        for mode in ('trace', 'logits', 'reference', 'head', 'incomplete', 'threshold'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = saved_probe(root)
                if mode == 'trace':
                    path = source / 'trace' / 'trace.f16'
                    path.write_bytes(struct.pack('<e', 2.) + path.read_bytes()[2:])
                elif mode == 'logits':
                    (source / 'native-traced.json').write_text('[1.0, 2.0]')
                elif mode == 'reference':
                    (root / 'diagnostic' / 'layer-inputs.json').write_text('[[2, 1], [], [], []]')
                elif mode == 'head':
                    (source / 'head-replay.json').unlink()
                else:
                    path = source / 'summary.json'
                    report = json.loads(path.read_text())
                    if mode == 'threshold':
                        report['metadata']['atol'] = .1
                    else:
                        report['cases'].pop()
                    path.write_text(json.dumps(report))
                argv = ['recheck', '--probe', str(source), '--output', str(root / 'rechecks')]
                with patch('sys.argv', argv), patch.object(recheck, 'capture', return_value='analysis'), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(recheck.main(), 1)
                output, = (root / 'rechecks').iterdir()
                report = json.loads((output / 'summary.json').read_text())
                self.assertEqual(report['cases'][-1]['case'], 'recheck_error')
                self.assertEqual(report['cases'][0]['status'], 'FAIL')
                self.assertFalse(report['metadata']['head_attribution_valid'])
                for suffix in ('json', 'md', 'csv', 'xlsx'):
                    self.assertTrue((output / f'summary.{suffix}').is_file())


if __name__ == '__main__':
    unittest.main()

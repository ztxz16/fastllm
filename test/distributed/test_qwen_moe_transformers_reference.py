"""参考模型设备分配及失败报告的纯Python检查；不加载GPU库、不编译。"""

from collections import Counter
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from qwen_moe_transformers_reference import reference_device_map, check_reference_placement


def fake_model(placement):
    """/** 构造带独立权重设备与执行hook设备的模型替身，可注入两者不一致的故障。 */"""
    parameters, buffers, blocks = {}, {}, {}
    for name, device in placement.items():
        hook = SimpleNamespace(execution_device=device, offload=False)
        block = SimpleNamespace(_hf_hook=hook)
        children = [('', block)]
        if name.startswith('model.layers.'):
            norm = SimpleNamespace(_hf_hook=SimpleNamespace(execution_device=device))
            children.append(('input_layernorm', norm))
            parameters[name + '.input_layernorm.weight'] = SimpleNamespace(device=f'cuda:{device}')
            block.norm = norm
        elif name == 'model.rotary_emb':
            buffers[name + '.inv_freq'] = SimpleNamespace(device=f'cuda:{device}')
        else:
            parameters[name + '.weight'] = SimpleNamespace(device=f'cuda:{device}')
        block.named_modules = lambda children=children: iter(children)
        blocks[name] = block
    return SimpleNamespace(
        named_parameters=lambda: iter(parameters.items()), named_buffers=lambda: iter(buffers.items()),
        get_submodule=blocks.__getitem__, parameters=parameters, buffers=buffers, blocks=blocks)


class ReferencePlacementTests(unittest.TestCase):
    def test_map_covers_model_without_ancestor_overlap(self):
        """/** 检查四卡完整覆盖、连续分层与非整除层数；根映射回归会触发祖先重叠失败。 */"""
        for count in (4, 7, 24, 30):
            with self.subTest(layers=count):
                placement = reference_device_map(SimpleNamespace(num_hidden_layers=count, tie_word_embeddings=False))
                for name in placement:
                    self.assertTrue(name)
                    self.assertFalse(any(name.startswith(other + '.') for other in placement if name != other))
                devices = [placement[f'model.layers.{i}'] for i in range(count)]
                self.assertEqual(set(devices), {0, 1, 2, 3})
                self.assertEqual(devices, sorted(devices))
                counts = Counter(devices)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
                self.assertEqual(placement['model.embed_tokens'], 0)
                self.assertEqual(placement['model.rotary_emb'], 0)
                self.assertEqual(placement['model.norm'], devices[-1])
                self.assertEqual(placement['lm_head'], devices[-1])

    def test_tied_head_stays_with_embedding_and_short_models_fail(self):
        """/** 绑定权重不跨卡，少于四层的模型不能伪装为四卡参考。 */"""
        config = SimpleNamespace(num_hidden_layers=4, tie_word_embeddings=True)
        placement = reference_device_map(config)
        self.assertEqual(placement['lm_head'], placement['model.embed_tokens'])
        for count in (0, 1, 3):
            config.num_hidden_layers = count
            with self.assertRaisesRegex(ValueError, 'four decoder layers'):
                reference_device_map(config)

    def setUp(self):
        """/** 每例使用独立的四层模型，故障注入不污染其他检查。 */"""
        self.placement = reference_device_map(SimpleNamespace(num_hidden_layers=4, tie_word_embeddings=False))
        self.model = fake_model(self.placement)

    def test_valid_dispatch_records_all_blocks(self):
        """/** 正常分配保留所有模块证据，RoPE非持久buffer也纳入设备核查。 */"""
        evidence = check_reference_placement(self.model, self.placement)
        self.assertEqual({row['module'] for row in evidence}, set(self.placement))
        self.assertEqual({row['device'] for row in evidence}, {f'cuda:{i}' for i in range(4)})

    def test_rejects_stale_root_execution_hook(self):
        """/** 重现权重在GPU 1但LayerNorm的输入hook仍搬往GPU 0的真实失败。 */"""
        self.model.blocks['model.layers.1'].norm._hf_hook.execution_device = 0
        with self.assertRaisesRegex(RuntimeError, r'input_layernorm: execution_device=cuda:0, expected=cuda:1'):
            check_reference_placement(self.model, self.placement)

    def test_sequential_hooks_are_checked_recursively(self):
        """/** 组合hook中的错误执行设备不能被外层空execution_device掩盖。 */"""
        norm = self.model.blocks['model.layers.2'].norm
        norm._hf_hook = SimpleNamespace(hooks=(SimpleNamespace(execution_device='cuda:2'),))
        check_reference_placement(self.model, self.placement)
        norm._hf_hook.hooks[0].execution_device = 'cuda:0'
        with self.assertRaisesRegex(RuntimeError, r'input_layernorm: execution_device=cuda:0, expected=cuda:2'):
            check_reference_placement(self.model, self.placement)

    def test_rejects_wrong_weights_buffers_and_uncovered_tensors(self):
        """/** 即使hook正确，权重错卡、RoPE残留CPU或新参数未分配也必须失败。 */"""
        for kind in ('weight', 'buffer', 'uncovered'):
            model = fake_model(self.placement)
            if kind == 'weight':
                model.parameters['model.layers.1.input_layernorm.weight'].device = 'cuda:0'
            elif kind == 'buffer':
                model.buffers['model.rotary_emb.inv_freq'].device = 'cpu'
            else:
                model.parameters['model.new_module.weight'] = SimpleNamespace(device='cuda:0')
            with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, 'reference tensor'):
                check_reference_placement(model, self.placement)

    def test_missing_block_hook_and_offload_are_rejected(self):
        """/** 显式块缺少输入搬运hook或被静默卸载时，不继续前向生成参考。 */"""
        del self.model.blocks['model.layers.1']._hf_hook
        with self.assertRaisesRegex(RuntimeError, 'missing Accelerate'):
            check_reference_placement(self.model, self.placement)
        self.model = fake_model(self.placement)
        self.model.blocks['model.layers.2']._hf_hook.offload = True
        with self.assertRaisesRegex(RuntimeError, 'offload is not allowed'):
            check_reference_placement(self.model, self.placement)


class ReferenceFailureReportTests(unittest.TestCase):
    def test_reference_exit_keeps_traceback_and_all_reports(self):
        """/** 模拟参考子进程退出，正式runner须返回失败并保存退出码、异常栈及四种报告。 */"""
        import run_qwen_moe_full as runner
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, requests = root / 'cache', root / 'requests.jsonl'
            cache.write_text('CUDA_ARCH:STRING=70\n')
            requests.write_text(json.dumps(dict(messages=[], max_new_tokens=1)) + '\n')
            argv = ['runner', '--binary', '/unused', '--worker', '/unused', '--model', str(root),
                    '--build-cache', str(cache), '--requests', str(requests), '--output', str(root / 'reports')]
            transformers = SimpleNamespace(AutoTokenizer=mock.Mock())
            traces = [dict(request=dict(token_ids=[1]), steps=[dict(token_id=2)])]
            stderr = io.StringIO()
            with mock.patch('sys.argv', argv), mock.patch.dict('sys.modules', transformers=transformers), \
                    mock.patch.object(runner, 'capture', return_value='unit test'), \
                    mock.patch.object(runner.Suite, 'session', return_value=nullcontext()), \
                    mock.patch.object(runner, 'batch', return_value=traces), \
                    mock.patch('fastllm_pytools.qwen3_process_chat.prepare_request', return_value={}), \
                    mock.patch.object(runner, 'case_process', return_value=nullcontext()), \
                    mock.patch.object(runner, 'wait_case', return_value=7), \
                    redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                self.assertEqual(runner.main(), 1)

            output, = (root / 'reports').iterdir()
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertIn('exited 7', report['cases'][0]['reason'])
            self.assertIn(str(output / 'reference.log'), stderr.getvalue())
            self.assertIn('AssertionError', (output / 'setup.traceback.log').read_text())
            for suffix in ('md', 'csv', 'json', 'xlsx'):
                self.assertTrue((output / f'summary.{suffix}').is_file())


if __name__ == '__main__':
    unittest.main()

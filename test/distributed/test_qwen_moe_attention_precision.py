"""用已有PyTorch CPU算子独立核对MoE舍入用例；不编译、不访问GPU或下载。"""

import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from qwen3_layer_reference import linear, matrix
from qwen_moe_reference import forward, full_logits
from run_qwen_moe_layer import make_case


class AttentionPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """/** 仅使用现有PyTorch的CPU算子；缺少时明确跳过而不安装。 */"""
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('requires an existing CPU PyTorch installation')
        cls.torch = torch

    def attention(self, config, tensors, task, input_, fp16_intermediates=True, split_bias=False):
        """/**
         * 用HF Eager同样的公开张量操作独立计算Attention子层与残差。
         * 输入、权重、位置和mask保持一致；旧FP32中间值分支只证明固定用例可检出回归。
         * @param config 小模型配置。
         * @param tensors 完整未分片权重。
         * @param task 正式单层入口的任务。
         * @param input_ 当前rank的原始隐藏输入与起始位置。
         * @param fp16_intermediates true使用HF FP16张量边界，false模拟旧精度语义。
         * @param split_bias true仅模拟旧QKV的点积FP16舍入后再加bias，验证用例能拒绝此行为。
         * @return CPU计算的FP16残差输出，展平为列表；不代表GPU验收结果。
         */"""
        torch = self.torch
        h, d = config['hidden_size'], config['head_dim']
        qh, kh = config['num_attention_heads'], config['num_key_value_heads']
        x = torch.tensor(input_['hidden'], dtype=torch.float16).reshape(-1, h)
        count = len(x)
        if not count:
            return []
        prefix = f"model.layers.{task['layer']}."
        def weight(name):
            """/** 将相同checkpoint权重读取为CPU FP16张量，不重排或改变精度。 */"""
            return torch.tensor(matrix(tensors[prefix+name]), dtype=torch.float16)

        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + config['rms_norm_eps'])
        normalized = normalized.half() * weight('input_layernorm.weight')
        projections = []
        for name, heads in (('q', qh), ('k', kh), ('v', kh)):
            bias_name = f'self_attn.{name}_proj.bias'
            bias = weight(bias_name) if prefix+bias_name in tensors else None
            values = torch.nn.functional.linear(normalized, weight(f'self_attn.{name}_proj.weight'),
                                                None if split_bias else bias)
            if split_bias and bias is not None:
                values = values + bias
            projections.append(values.reshape(count, heads, d).transpose(0, 1))
        q, k, v = projections
        positions = torch.arange(count, dtype=torch.float32) + input_['position_begin']
        inv_freq = 1.0 / config['rope_theta'] ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
        angles = positions[:, None] * inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cosine, sine = angles.cos().half(), angles.sin().half()
        q = q*cosine + torch.cat((-q[..., d//2:], q[..., :d//2]), dim=-1)*sine
        k = k*cosine + torch.cat((-k[..., d//2:], k[..., :d//2]), dim=-1)*sine
        k, v = [value.repeat_interleave(qh//kh, dim=0) for value in (k, v)]

        scores = torch.matmul(q if fp16_intermediates else q.float(),
                              (k if fp16_intermediates else k.float()).transpose(-1, -2)) / math.sqrt(d)
        mask = torch.ones(count, count, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float('-inf'))
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        if fp16_intermediates:
            probabilities = probabilities.half()
        context = torch.matmul(probabilities, v if fp16_intermediates else v.float()).half()
        context = context.transpose(0, 1).reshape(count, h)
        output_bias = weight('self_attn.o_proj.bias') if prefix+'self_attn.o_proj.bias' in tensors else None
        return (x + torch.nn.functional.linear(context, weight('self_attn.o_proj.weight'), output_bias)).reshape(-1).tolist()

    def test_public_layer_fixture_matches_hf_precision_and_detects_old_path(self):
        """/** 同一正式用例与独立HF算子一致，旧中间精度必须超过原阈值；包含空rank。 */"""
        with tempfile.TemporaryDirectory() as temp:
            config, tensors, tasks = make_case(Path(temp), dict(bias=False, attention_rounding=True), 4)
            task = tasks[0]
            legacy_failed = 0
            for rank, input_ in enumerate(task['inputs']):
                expected = self.attention(config, tensors, task, input_)
                actual = forward(config, tensors, task, input_)['output']
                old = self.attention(config, tensors, task, input_, False)
                self.assertEqual(len(actual), len(expected))
                with self.subTest(rank=rank):
                    self.assertTrue(all(abs(a-b) <= .008 + .008*abs(b) for a, b in zip(actual, expected)))
                legacy_failed += sum(abs(a-b) > .008 + .008*abs(b) for a, b in zip(old, expected))
            self.assertGreater(legacy_failed, 0, 'fixture must reject missing FP16 rounding')

    def test_qkv_bias_fixture_matches_torch_and_rejects_double_rounding(self):
        """/** 解析消去用例经正式任务覆盖prefill/decode及空rank，旧舍入须超过原容差。 */"""
        with tempfile.TemporaryDirectory() as temp:
            config, tensors, tasks = make_case(Path(temp), dict(qkv_bias_rounding=True), 4)
            failures = 0
            for task in tasks:
                for rank, input_ in enumerate(task['inputs']):
                    with self.subTest(task=task['name'], rank=rank):
                        expected = self.attention(config, tensors, task, input_)
                        actual = forward(config, tensors, task, input_)['output']
                        old = self.attention(config, tensors, task, input_, split_bias=True)
                        analytical = [1.03125 if i % 8 == 0 else 1.0 for i in range(len(input_['hidden']))]
                        self.assertEqual(expected, analytical)
                        self.assertEqual(actual, expected)
                        self.assertEqual(old, [1.0]*len(input_['hidden']))
                        failures += sum(abs(a-b) > .008+.008*abs(b) for a, b in zip(old, expected))
            self.assertGreater(failures, 0)

    def test_bias_fixture_detects_regression_in_full_logits(self):
        """/** 同一checkpoint经全模型参考放大可见差异，避免最终Norm消除测试信号。 */"""
        with tempfile.TemporaryDirectory() as temp:
            config, tensors, _ = make_case(Path(temp), dict(qkv_bias_rounding=True), 4)
            def old_linear(x, weight, bias=None, rounded=True):
                """/** 仅恢复有bias的旧舍入；此用例O投影bias为0，不影响其他Linear语义。 */"""
                return linear(x, weight, bias, rounded=True if bias is not None else rounded)
            for ids in ([1], [1, 2, 3], [1, 2, 3, 4]):
                expected = full_logits(config, tensors, ids)
                with mock.patch('qwen_moe_reference.linear', side_effect=old_linear):
                    old = full_logits(config, tensors, ids)
                self.assertTrue(any(abs(a-b) > .008+.008*abs(b) for a, b in zip(old, expected)))


if __name__ == '__main__':
    unittest.main()

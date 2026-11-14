"""用已有PyTorch CPU算子独立核对MoE舍入用例；不编译、不访问GPU或下载。"""

import math
from pathlib import Path
import tempfile
import unittest

from qwen3_layer_reference import matrix
from qwen_moe_reference import forward
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

    def attention(self, config, tensors, task, input_, fp16_intermediates=True):
        """/**
         * 用HF Eager同样的公开张量操作独立计算Attention子层与残差。
         * 输入、权重、位置和mask保持一致；旧FP32中间值分支只证明固定用例可检出回归。
         * @param config 小模型配置。
         * @param tensors 完整未分片权重。
         * @param task 正式单层入口的任务。
         * @param input_ 当前rank的原始隐藏输入与起始位置。
         * @param fp16_intermediates true使用HF FP16张量边界，false模拟旧精度语义。
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
        q, k, v = [torch.nn.functional.linear(normalized, weight(f'self_attn.{name}_proj.weight'))
                   .reshape(count, heads, d).transpose(0, 1)
                   for name, heads in (('q', qh), ('k', kh), ('v', kh))]
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
        return (x + torch.nn.functional.linear(context, weight('self_attn.o_proj.weight'))).reshape(-1).tolist()

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


if __name__ == '__main__':
    unittest.main()

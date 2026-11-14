"""使用已有PyTorch的CPU算子验证参考采集，不编译、不访问GPU或下载模型。"""

from types import SimpleNamespace
import unittest
from unittest import mock

from qwen_moe_transformers_reference import layer_reference, moe_routing_observer


class RoutingObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """/** PyTorch仅用于可选CPU集成检查，主验收环境缺少时明确标记跳过。 */"""
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('CPU observer integration requires an existing PyTorch installation')
        cls.torch = torch

    def model(self, dtype, normalize, *, fail=False, corrupt=False):
        """/** 创建无下载的两层MoE，真实调用公开Top-K/专家/index_add_并独立保存实际输出。 */"""
        torch = self.torch

        class Moe(torch.nn.Module):
            def __init__(self):
                """/** 四专家与Top-2制造等值路由，逐专家累加保持输入dtype。 */"""
                super().__init__()
                self.gate = torch.nn.Linear(3, 4, bias=False)
                self.experts = torch.nn.ModuleList(torch.nn.Linear(3, 3, bias=False) for _ in range(4))
                self.shared_expert = torch.nn.Linear(3, 3, bias=False)
                self.shared_expert_gate = torch.nn.Linear(3, 1, bias=False)
                self.evidence = []
                torch.nn.init.zeros_(self.gate.weight)

            def forward(self, hidden):
                """/** 按实际Top-K结果调度专家，独立保存采集器应观察到的路由与合成结果。 */"""
                shape = hidden.shape
                hidden = hidden.reshape(-1, 3)
                logits = self.gate(hidden)
                weights, selected = torch.topk(torch.nn.functional.softmax(logits, dim=1, dtype=torch.float), 2, dim=-1)
                if normalize:
                    weights /= weights.sum(dim=-1, keepdim=True)
                weights = weights.to(hidden.dtype)
                routed = torch.zeros_like(hidden)
                mask = torch.nn.functional.one_hot(selected, num_classes=4).permute(2, 1, 0)
                for expert_id in mask.sum(dim=(-1, -2)).nonzero().reshape(-1):
                    slot, token = torch.where(mask[expert_id])
                    value = self.experts[expert_id](hidden[token]) * weights[token, slot, None]
                    routed.index_add_(0, token, value.to(hidden.dtype))
                if fail:
                    raise RuntimeError('injected expert failure')

                shared = self.shared_expert(hidden)
                shared = torch.sigmoid(self.shared_expert_gate(hidden)) * shared
                combined = (routed + shared).reshape(shape)
                self.evidence.append(dict(router_logits=logits.clone(), selected_experts=selected.clone(),
                    routing_weights=weights.clone(), routed_output=routed.clone(), shared_output=shared.clone()))
                if corrupt:
                    combined = combined + 1
                return combined, logits

        class Layer(torch.nn.Module):
            def __init__(self):
                """/** 保持正式层的MoE子模块入口与残差相加。 */"""
                super().__init__()
                self.mlp = Moe()

            def forward(self, hidden):
                """/** 返回包含残差的真实层输出，供逐层采集入口验证。 */"""
                return (hidden + self.mlp(hidden)[0],)

        class Model(torch.nn.Module):
            def __init__(self):
                """/** 构造Embedding和两层公开Module入口，不模拟Tensor计算。 */"""
                super().__init__()
                self.config = SimpleNamespace(hidden_size=3, num_experts_per_tok=2, norm_topk_prob=normalize)
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(8, 3)
                self.model.layers = torch.nn.ModuleList(Layer() for _ in range(2))

            def get_input_embeddings(self):
                """/** 返回采集入口用于确定输入设备的Embedding。 */"""
                return self.model.embed_tokens

            def forward(self, input_ids, use_cache=False):
                """/** 沿公开模块入口执行实际CPU前向。 */"""
                hidden = self.model.embed_tokens(input_ids)
                for layer in self.model.layers:
                    hidden = layer(hidden)[0]
                return hidden

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261114)
            return Model().to(dtype=dtype).eval()

    def assert_no_hooks(self, model):
        """/** 成功或异常退出都不得遗留hook，防止污染后续历史参考前向。 */"""
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_actual_routes_and_dtype_accumulation_without_reselecting(self):
        """/** 等值路由、归一化开关及三种dtype均与实际返回值一致，不容许CPU重算softmax。 */"""
        torch = self.torch
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for normalize in (False, True):
                with self.subTest(dtype=dtype, normalize=normalize):
                    model = self.model(dtype, normalize)
                    ids = torch.tensor([[1, 2]])
                    with torch.inference_mode():
                        expected_output = model(ids).clone()
                    for layer in model.model.layers:
                        layer.mlp.evidence.clear()
                    # 正式路径调用F.softmax；旧采集hook调用torch.softmax重选专家，应被此检查拒绝。
                    with mock.patch.object(torch, 'softmax', side_effect=AssertionError('observer recomputed routing')):
                        report = layer_reference(model, torch)

                    for index, layer in enumerate(model.model.layers):
                        for rank, observed in enumerate(layer.mlp.evidence):
                            for key, value in observed.items():
                                self.assertEqual(report[index]['reference'][rank][key], value.reshape(-1).tolist())
                    self.assert_no_hooks(model)
                    with torch.inference_mode():
                        self.assertTrue(torch.equal(expected_output, model(ids)))

    def test_failed_forward_removes_observer_and_all_hooks(self):
        """/** 专家中途异常后可再次运行普通前向，线程局部观察器及层hook均须释放。 */"""
        torch = self.torch
        from torch.overrides import _get_current_function_mode
        previous = _get_current_function_mode()
        model = self.model(torch.float16, True, fail=True)
        with self.assertRaisesRegex(RuntimeError, 'injected expert failure'):
            layer_reference(model, torch)
        self.assert_no_hooks(model)
        self.assertIs(_get_current_function_mode(), previous)

    def test_observation_rejects_changed_moe_composition(self):
        """/** 模型输出含未捕获贡献时必须拒绝，不能把错误中间量作为参考。 */"""
        model = self.model(self.torch.float16, True, corrupt=True)
        with self.assertRaisesRegex(RuntimeError, r'layer=0 rank=0 tokens=2.*do not reconstruct'):
            layer_reference(model, self.torch)
        self.assert_no_hooks(model)

    def test_unknown_operator_path_is_rejected(self):
        """/** 没有观察到实际路由算子时拒绝静默退回CPU重算。 */"""
        observer = moe_routing_observer(self.torch)
        observer.begin({}, 'layer=3 rank=1 tokens=4')
        with self.assertRaisesRegex(RuntimeError, r'layer=3 rank=1 tokens=4.*topk=0'):
            observer.finish(None)

    def test_unsupported_router_layout_is_rejected_before_hooks(self):
        """/** 新版集合Router或非MoE层不能被错误地当作原始logits入口。 */"""
        model = self.model(self.torch.float16, True)
        model.model.layers[1].mlp.gate = self.torch.nn.Identity()
        with self.assertRaisesRegex(RuntimeError, r'layer=1: reference requires'):
            layer_reference(model, self.torch)
        self.assert_no_hooks(model)


if __name__ == '__main__':
    unittest.main()

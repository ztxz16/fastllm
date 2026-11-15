"""MoE 独立数学参考和验收器的纯 Python 检查；不编译、不加载 GPU 库。"""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from qwen3_layer_reference import half, matrix
from qwen_moe_reference import fixture, forward, full_logits, route, weights_for_rank
from run_qwen3_shards import encode
from run_qwen_moe_layer import BACKEND, compare, make_case, tasks_for, verify, cases


class MoeReferenceTests(unittest.TestCase):
    def setUp(self):
        """/** 为每项检查创建隔离的小模型与空误差计数。 */"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.tensors = fixture(self.root / "model")
        self.task = tasks_for(1)[0]
        self.input = self.task["inputs"][0]
        self.metrics = dict(max=0.0, sum=0.0, count=0, failed=0)

    def test_topk_uses_full_softmax_then_optional_normalization(self):
        """/** 已知 softmax=[1,2,3]/6，验证 Top-K 前后归一化的区别。 */"""
        logits = [0, math.log(2), math.log(3)]
        ids, probs = route(logits, 2, False)
        self.assertEqual(ids, [2, 1])
        self.assertEqual(probs, [half(0.5), half(1/3)])
        self.assertEqual(route(logits, 2, True), ([2, 1], [half(0.6), half(0.4)]))
        self.assertEqual(route([0, 0, 0], 2, True), ([0, 1], [0.5, 0.5]))

    def test_partition_covers_each_expert_once_and_replicates_shared(self):
        """/** 独立检查 1/2/4 rank 的专家互斥性、覆盖和非专家完整复制。 */"""
        for world in (1, 2, 4):
            plans = [weights_for_rank(self.config, self.tensors, rank, world) for rank in range(world)]
            for expert in range(5):
                key = f"model.layers.0.mlp.experts.{expert}.down_proj.weight"
                self.assertEqual([rank for rank, p in enumerate(plans) if key in p], [expert % world])
            for key in plans[0]:
                if ".experts." not in key:
                    self.assertTrue(all(p[key] == plans[0][key] for p in plans))

    def test_zero_projections_preserve_residual(self):
        """/** 全零投影下任意阶段输出必须是输入 FP16 残差。 */"""
        for name, tensor in self.tensors.items():
            if "layernorm.weight" not in name:
                tensor["data"] = bytes(len(tensor["data"]))
        for stage in (0, 1, 2):
            actual = forward(self.config, self.tensors, dict(self.task, stage=stage), self.input)
            self.assertEqual(actual["output"], [half(x) for x in self.input["hidden"]])

    def test_full_layer_equals_attention_then_moe(self):
        """/** 两个单阶段与完整层必须遵循相同归一化和残差顺序。 */"""
        attention = forward(self.config, self.tensors, dict(self.task, stage=1), self.input)
        moe = forward(self.config, self.tensors, dict(self.task, stage=2), dict(self.input, hidden=attention["output"]))
        self.assertEqual(moe, forward(self.config, self.tensors, self.task, self.input))

    def test_shared_gate_sigmoid_zero_halves_shared_output(self):
        """/** 用只含共享专家的解析输入验证 sigmoid 门控和舍入边界。 */"""
        for name, tensor in self.tensors.items():
            if "layernorm.weight" not in name:
                tensor["data"] = bytes(len(tensor["data"]))
        p = "model.layers.0.mlp.shared_expert."
        for projection in ("gate", "up"):
            key = p + projection + "_proj.weight"
            self.tensors[key]["data"] = encode([0.125]*88, "F16")
        self.tensors[p + "down_proj.weight"]["data"] = encode([0.125]*88, "F16")
        input_ = dict(position_begin=0, hidden=[1.0]*8)
        # 归一化权重固定为 1，得到 gate=up=1，SiLU(1) 后由 11 个通道求和。
        norm_tensor = self.tensors["model.layers.0.post_attention_layernorm.weight"]
        norm_tensor["data"] = encode([1.0]*8, "F16")
        out = forward(self.config, self.tensors, dict(self.task, stage=2), input_)
        expected = half(half(half(1/(1+math.exp(-1))) * 11/8) * 0.5)
        self.assertEqual(out["shared_output"], [expected]*8)
        self.assertEqual(out["routed_output"], [0.0]*8)

    def test_causal_prefix_and_idle_inputs(self):
        """/** 修改未来 Token 不影响前缀，空 rank 返回空输出和空路由。 */"""
        actual = forward(self.config, self.tensors, self.task, self.input)
        changed = dict(self.input, hidden=self.input["hidden"][:-8] + [3.0]*8)
        after = forward(self.config, self.tensors, self.task, changed)
        self.assertEqual(actual["output"][:-8], after["output"][:-8])
        idle = forward(self.config, self.tensors, self.task, dict(position_begin=0, hidden=[]))
        self.assertTrue(all(not value for value in idle.values()))

    def test_expert_accumulation_fixture_has_analytic_result_on_public_tasks(self):
        """/** 解析消去用例覆盖不同EP、Prefill/Decode及空闲输入，固定原验收阈值。 */"""
        for world in (1, 2, 4):
            directory = self.root / f'accumulation{world}'
            directory.mkdir()
            config, tensors, tasks = make_case(directory,
                dict(experts=4, topk=4, expert_accumulation=True), world)
            for task in tasks:
                for rank, input_ in enumerate(task['inputs']):
                    count = len(input_['hidden']) // 8
                    actual = forward(config, tensors, task, input_)
                    with self.subTest(world=world, task=task['name'], rank=rank):
                        self.assertEqual(actual['routed_output'], [1.0, *([0.0]*7)] * count)
                        self.assertEqual(actual['output'], [2.0, *([1.0]*7)] * count)
                        self.assertEqual(actual['selected_experts'], [2, 0, 3, 1] * count)
            logits = full_logits(config, tensors, [1, 2, 3])
            # 输出头读取规范化后的首通道；以解析残差[2,1,...,1]独立求期望。
            expected = half(2 / math.sqrt(11/8 + config['rms_norm_eps'])) * 4
            self.assertEqual(logits, [expected] + [0.0]*12)
            with self.assertRaises(AssertionError):
                compare([2.0], [1.0], dict(max=0., sum=0., count=0, failed=0), 'legacy FP32 reduction')

    def test_expert_accumulation_fixture_matches_public_torch_operations(self):
        """/** 用独立PyTorch CPU Linear/TopK/index_add_核对参考，并拒绝错精度及错顺序。 */"""
        try:
            import torch
        except ImportError:
            self.skipTest('requires an existing CPU PyTorch installation')
        config, tensors = fixture(self.root / 'torch_accumulation', experts=4, topk=4, expert_accumulation=True)
        x = torch.ones((3, 8), dtype=torch.float16)
        prefix = 'model.layers.0.'
        def weight(name):
            """/** 直接读取未分片checkpoint为CPU FP16张量，保持HF布局。 */"""
            return torch.tensor(matrix(tensors[prefix+name]), dtype=torch.float16)

        normed = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + config['rms_norm_eps'])).half()
        normed = normed * weight('post_attention_layernorm.weight')
        scores = torch.nn.functional.linear(normed, weight('mlp.gate.weight'))
        probs, ids = torch.topk(torch.softmax(scores, dim=-1, dtype=torch.float32), 4, dim=-1)
        probs = probs.half()
        output = torch.zeros_like(x)
        contributions = []
        for expert in range(4):
            name = f'mlp.experts.{expert}.'
            gate = torch.nn.functional.linear(normed, weight(name+'gate_proj.weight'))
            up = torch.nn.functional.linear(normed, weight(name+'up_proj.weight'))
            down = torch.nn.functional.linear(torch.nn.functional.silu(gate)*up, weight(name+'down_proj.weight'))
            probability = (probs * (ids == expert)).sum(dim=-1, keepdim=True)
            contribution = down * probability
            contributions.append(contribution)
            output.index_add_(0, torch.arange(len(x)), contribution)
        self.assertEqual([value[0, 0].item() for value in contributions], [2048, 1, -2048, 1])
        actual = forward(config, tensors, dict(layer=0, stage=2), dict(position_begin=0, hidden=x.flatten().tolist()))
        self.assertEqual(actual['routed_output'], output.flatten().tolist())
        self.assertEqual(actual['output'], (x+output).flatten().tolist())

        old = torch.stack(contributions).float().sum(dim=0).half()
        wrong_order = torch.zeros_like(x)
        for expert in ids[0].tolist():
            wrong_order.add_(contributions[expert])
        for wrong in (old, wrong_order):
            self.assertEqual(wrong[0, 0].item(), 2)
            with self.assertRaises(AssertionError):
                compare(wrong.flatten().tolist(), output.flatten().tolist(),
                        dict(max=0., sum=0., count=0, failed=0), 'incorrect expert accumulation')

    def test_expert_accumulation_case_registered_for_all_execution_modes(self):
        """/** 正式完整生成矩阵必须包含EP=1/2/4及Graph、Prefix全部组合。 */"""
        from run_qwen_moe_full import SYNTHETIC_CASES
        for world in (1, 2, 4):
            for graph in (0, 1):
                for prefix in (0, 1):
                    self.assertIn(f'expert_accumulation_ep{world}_graph{graph}_prefix{prefix}', SYNTHETIC_CASES)

    def test_expert_accumulation_fixture_rejects_incompatible_shapes(self):
        """/** 解析用例的精度和专家数量固定，拒绝产生失去消去语义的模型。 */"""
        for options in (dict(experts=5), dict(topk=2), dict(dtype='BF16'), dict(normalize=True)):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'accumulation fixture'):
                fixture(self.root / 'invalid_accumulation', **dict(dict(experts=4, topk=4, expert_accumulation=True), **options))

    def test_nonfinite_and_shape_mismatches_fail(self):
        """/** 非有限参考值不得因 inf 容差被放行。 */"""
        for actual, expected in [(float("nan"), 0), (0, float("inf")), (float("inf"), float("inf")), (2, 0)]:
            with self.assertRaises(AssertionError):
                compare([actual], [expected], self.metrics, "corrupt")
        self.assertEqual(self.metrics["failed"], 4)
        with self.assertRaisesRegex(AssertionError, "shape"):
            compare([], [0.0], self.metrics, "truncated")

    def evidence(self):
        """/** 构造仅用于验收器拒绝逻辑自测的证据，不能冒充 GPU 验收结果。 */"""
        tasks = tasks_for(1)[:1]
        ref = forward(self.config, self.tensors, tasks[0], tasks[0]["inputs"][0])
        counts = [ref["selected_experts"].count(e) for e in range(5)]
        result = dict(ref, expert_tokens=counts, allgather_calls=3, reduce_scatter_calls=1, indexed_decode=False)
        expected = weights_for_rank(self.config, self.tensors, 0, 1)
        gpu = self.root / "gpu"
        gpu.mkdir(exist_ok=True)
        entries = []
        for index, (name, tensor) in enumerate(expected.items()):
            filename = f"{index}.bin"
            (gpu / filename).write_bytes(encode(tensor["values"], "F16"))
            entries.append(dict(name=name, shape=tensor["shape"], file=filename))
        report = dict(status="PASS", backend=BACKEND, workers=[dict(rank=0, device=0, pid=100,
                      cuda_runtime_version=12060, nccl_version=22203)], workers_reaped=True, reload_passed=True,
                      weights=[dict(tensors=entries, device_bytes=sum(len(x["values"])*2 for x in expected.values()),
                                    source_bytes_read=sum(x["source_bytes"] for x in expected.values()))],
                      results=[dict(task=tasks[0]["name"], ranks=[result])])
        return report, tasks

    def test_verifier_rejects_missing_paths_and_wrong_ownership(self):
        """/** 检查器必须拒绝静默 fallback、缺 rank、额外专家及篡改的路由。 */"""
        report, tasks = self.evidence()
        verify(report, self.root, [0], self.config, self.tensors, tasks, self.metrics)
        for mutation in ("backend", "rank", "collective", "expert", "route", "cleanup"):
            broken = copy.deepcopy(report)
            if mutation == "backend": broken["backend"] = "fallback"
            if mutation == "rank": broken["results"][0]["ranks"] = []
            if mutation == "collective": broken["results"][0]["ranks"][0]["allgather_calls"] = 0
            if mutation == "expert": broken["weights"][0]["tensors"].pop()
            if mutation == "route": broken["results"][0]["ranks"][0]["selected_experts"][0] = 100
            if mutation == "cleanup": broken["workers_reaped"] = False
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                verify(broken, self.root, [0], self.config, self.tensors, tasks, self.metrics)

    def test_runner_failure_still_emits_four_reports(self):
        """/** 模拟已有检查器执行失败，验证非零状态及完整报告，不启动原生产物。 */"""
        import run_qwen_moe_layer as runner
        cache = self.root / "cache"
        cache.write_text("CUDA_ARCH:STRING=70\n")
        args = ["runner", "--check", "/missing", "--worker", "/missing", "--devices", "0",
                "--build-cache", str(cache), "--output-root", str(self.root / "reports")]
        with mock.patch("sys.argv", args), mock.patch.object(runner, "cases", return_value=[("failure", {})]), \
                mock.patch.object(runner, "capture", return_value="not queried: unit test"), \
                mock.patch.object(runner, "invoke", side_effect=RuntimeError("injected worker failure")):
            self.assertEqual(runner.main(), 1)
        directories = list((self.root / "reports").iterdir())
        self.assertEqual(len(directories), 1)
        for suffix in ("md", "csv", "json", "xlsx"):
            self.assertTrue((directories[0] / ("summary." + suffix)).is_file())

    def test_all_fixture_cases_construct_without_native_code(self):
        """/** 所有 GPU 用例均可生成可复现输入，非法模型由原生入口负责拒绝。 */"""
        names = set()
        for index, (name, spec) in enumerate(cases()):
            self.assertNotIn(name, names)
            names.add(name)
            path = self.root / str(index)
            path.mkdir()
            _, _, tasks = make_case(path, spec, 4)
            self.assertTrue(tasks)
            self.assertTrue((path / "tasks.json").is_file())


if __name__ == "__main__":
    unittest.main()

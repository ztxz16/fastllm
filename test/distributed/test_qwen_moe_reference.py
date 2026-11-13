"""MoE 独立数学参考和验收器的纯 Python 检查；不编译、不加载 GPU 库。"""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from qwen3_layer_reference import half
from qwen_moe_reference import fixture, forward, route, weights_for_rank
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

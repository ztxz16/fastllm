"""External GGUF drafts must be validated and included in capacity planning."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import deepseek_v41_gguf as aux, hybrid_auto as memory
from fastllm_pytools.util import make_normal_llm_model, make_normal_parser


class AddonTest(unittest.TestCase):
    def setUp(self):
        self.meta = {"general.architecture": "dflash", "dflash.block_count": 1,
                     "dflash.embedding_length": 32, "dflash.block_size": 5,
                     "dflash.target_layers": [3], "tokenizer.ggml.mask_token_id": 7,
                     "dflash.attention.compress_ratios": [0],
                     "dflash.expert_count": 2, "dflash.expert_used_count": 1}
        self.target = {"gguf_architecture": "deepseek41", "hidden_size": 32, "num_hidden_layers": 4}
        for key, value in {"num_attention_heads": 4, "head_dim": 8, "qk_rope_head_dim": 4,
                           "q_lora_rank": 16, "o_lora_rank": 8, "o_groups": 2,
                           "hc_mult": 4, "moe_intermediate_size": 16}.items():
            self.target[key] = value
            self.meta["dflash." + aux._V41_GGUF_ALIASES[key]] = value
        self.tensors = {"markov_w1.weight": ((64, 8), 30), "markov_w2.weight": ((64, 8), 30),
                        "fc.weight": ((32, 32), 8)}
        self.tensors.update({"blk.0." + n + ".weight": ((32, 32), 8) for n in
                            ("attn_q_a", "attn_q_b", "attn_kv", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")})
        for name, value in (("read_gguf_metadata", self.meta), ("read_gguf_tensor_info", self.tensors)):
            mock = patch.object(aux, name, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def test_draft_configuration(self):
        config = aux.dspark_config("draft.gguf", self.target)
        self.assertEqual((config["num_nextn_predict_layers"], config["dspark_block_size"],
                          config["dspark_markov_rank"], config["vocab_size"]), (1, 5, 8, 64))

    def test_target_mismatch_fails_before_loading(self):
        self.target["hidden_size"] = 64
        with self.assertRaisesRegex(ValueError, "mismatch"):
            aux.dspark_config("draft.gguf", self.target)

    def test_bad_markov_and_projection(self):
        for name in ("markov_w2.weight", "fc.weight"):
            with self.subTest(name=name):
                saved = self.tensors[name]
                self.tensors[name] = ((8, 64), 30)
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    aux.dspark_config("draft.gguf", self.target)
                self.tensors[name] = saved

    def test_incomplete_draft_and_bad_targets(self):
        del self.tensors["blk.0.ffn_down_exps.weight"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            aux.dspark_config("draft.gguf", self.target)
        self.tensors["blk.0.ffn_down_exps.weight"] = ((32, 32), 39)
        self.meta["dflash.target_layers"] = [4]
        with self.assertRaisesRegex(ValueError, "out of range"):
            aux.dspark_config("draft.gguf", self.target)

    def test_planner_counts_raw_draft_experts_in_draft_stage(self):
        config = dict(model_type="deepseek_v41", num_hidden_layers=4, max_position_embeddings=1024,
                      hidden_size=32, n_routed_experts=2, num_experts_per_tok=1, moe_intermediate_size=16,
                      num_attention_heads=4, head_dim=8, index_head_dim=8, compress_ratios=[0,0,2,1,0],
                      kv_source_layer_ids=[2,3], index_source_layer_ids=[2], sliding_window=8)
        config.update(aux.dspark_config("draft.gguf"))
        model = memory.geometry(config, "fp4", 0, 5)
        args = make_normal_parser("test").parse_args([])
        args.dspark = 5
        tensors = [memory.Tensor("blk.0.ffn_gate_exps.weight", (2,16,32), 544, "gguf:16")]
        tensors += [memory.Tensor(aux.draft_tensor_name(n, 1), (2,16,32), 544, "gguf:39")
                    for n in ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight")]
        costs = memory.weight_costs(tensors, model, args)
        self.assertEqual(costs[0][-1], 3 * 544)
        self.assertEqual(model.eager_expert_layers, {4})
        self.assertEqual(set(model.gguf_expert_bytes), {0})
        args.mmproj = "vision.gguf"
        vision = memory.Tensor("vision.mm.1.weight", (32, 32), 1088, "gguf:8")
        with_vision = memory.weight_costs(tensors + [vision], model, args)
        self.assertEqual(with_vision[3] - costs[3], 2048)  # FP16 import, not Q8 storage
        self.assertEqual(with_vision[2] - costs[2], 2 * memory.GiB + 2048)

    def test_explicit_off_does_not_read_draft(self):
        args = make_normal_parser("test").parse_args(["--draft", "/missing.gguf", "--speculative_algorithm", "off"])
        with self.assertRaises(SystemExit):
            make_normal_llm_model(args)
        aux.read_gguf_metadata.assert_not_called()

    def test_cli_selects_embedded_dspark_engine_for_gguf(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as file, patch.dict(os.environ, {}, clear=True), \
                patch("fastllm_pytools.util.read_gguf_metadata", return_value={"general.architecture": "dflash"}):
            args = make_normal_parser("test").parse_args(["--draft", file.name, "--draft_tokens", "3"])
            with self.assertRaises(SystemExit):
                make_normal_llm_model(args)
            self.assertEqual(args.speculative_algorithm, "dspark")
            self.assertEqual(args.dspark, 3)
            self.assertEqual(os.environ["FASTLLM_DSPARK_TOKENS"], "3")
            self.assertNotIn("FASTLLM_DSPARK_MODEL_PATH", os.environ)


if __name__ == "__main__":
    unittest.main()

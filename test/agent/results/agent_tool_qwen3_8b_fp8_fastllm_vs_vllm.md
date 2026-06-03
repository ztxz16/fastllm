# Agent Tool-use Qwen3-8B-FP8 FastLLM vs vLLM

Date: 2026-06-02

## Setup

- Benchmark: `agent_tool`
- Cases: `test/agent/baseline/default_cases.jsonl`, 24 samples
- Model: `/root/hfmodels/Qwen3-8B-FP8`
- Served model name: `ds`
- API: OpenAI-compatible `/v1/chat/completions`
- Protocol: JSON ReAct, no native server-side function calling required
- Generation: `temperature=0`, `max_tokens=384`
- Chat template: `{"chat_template_kwargs":{"enable_thinking":false}}`
- Client workers: `4`
- Max agent steps per case: `5`

FastLLM service:

```bash
FASTLLM_CUDA_CUTLASS_LINEAR_FP8=1 ftllm serve ~/hfmodels/Qwen3-8B-FP8 --model_name ds --port 1616 --hide_input
```

vLLM service:

```bash
CUDA_VISIBLE_DEVICES=0 /root/vllm/bin/vllm serve /root/hfmodels/Qwen3-8B-FP8 --served-model-name ds --host 0.0.0.0 --port 1617 --gpu-memory-utilization 0.90 --max-model-len 8192
```

Evaluation command shape:

```bash
python3 test/agent/agent_tool_eval.py \
  --base-url http://127.0.0.1:<port> \
  --model ds \
  --workers 4 \
  --max-steps 5 \
  --max-tokens 384 \
  --output-file test/agent/results/<name>.jsonl \
  --overwrite \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

## Results

| Metric | FastLLM | vLLM | Delta |
| --- | ---: | ---: | ---: |
| Correct / total | 22 / 24 | 22 / 24 | same |
| Accuracy | 91.67% | 91.67% | same |
| Tool plan correct / total | 20 / 24 | 20 / 24 | same |
| Tool plan accuracy | 83.33% | 83.33% | same |
| Invalid JSON cases | 0 | 0 | same |
| Tool error cases | 0 | 0 | same |
| Max-step cases | 0 | 0 | same |
| Avg steps | 2.42 | 2.38 | vLLM -0.04 |
| Avg case latency | 722.73 ms | 626.98 ms | vLLM -13.25% |
| P50 case latency | 591.82 ms | 532.66 ms | vLLM -10.00% |
| P90 case latency | 1068.58 ms | 950.81 ms | vLLM -11.02% |
| P95 case latency | 1320.28 ms | 990.01 ms | vLLM -25.02% |
| Avg request latency | 298.87 ms | 263.77 ms | vLLM -11.74% |
| Runtime | 4.46 s | 3.93 s | vLLM -12.03% |
| Items/s | 5.38 | 6.11 | vLLM +13.67% |
| Input tok/s | 7143.79 | 7911.01 | vLLM +10.74% |
| Output tok/s | 376.48 | 439.94 | vLLM +16.85% |

## Category Accuracy

| Category | FastLLM | vLLM |
| --- | ---: | ---: |
| lookup | 100.00% | 100.00% |
| multi_step | 100.00% | 100.00% |
| inventory | 100.00% | 100.00% |
| calculation | 100.00% | 100.00% |
| knowledge | 100.00% | 100.00% |
| policy | 50.00% | 50.00% |

## Failed Cases

Both runtimes failed the same two policy cases:

- `agent:010`: Asked whether cancelled order `ORD-1004` can be returned. The model searched the return policy but did not look up the order, then answered `yes`; expected `no`.
- `agent:019`: Asked whether order `ORD-1002` qualifies for free shipping. The model looked up the order but searched `shipping policy`; the free-shipping rule is in `discount policy`, so it answered `no`; expected `yes`.

## Notes

- This is a lightweight local agent regression test, not a public leaderboard benchmark.
- The script tests JSON instruction following, tool selection, argument generation, multi-step tool chaining, policy lookup, calculator use, and final-answer extraction.
- vLLM and FastLLM produced the same accuracy on this small suite. vLLM had lower latency and higher throughput in this run.

Summary files:

- `test/agent/results/fastllm_qwen3_8b_fp8_agent_tool_full.summary.json`
- `test/agent/results/vllm_qwen3_8b_fp8_agent_tool_full.summary.json`

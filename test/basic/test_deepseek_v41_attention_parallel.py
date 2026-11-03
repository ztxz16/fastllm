#!/usr/bin/env python3
"""Eight-GPU Attention DP/TP regression, including load-time TP weight splitting.

The native test checks mixed-length requests, owner-local KV, prefix relocation,
and request-handle reuse. This driver compares teacher-forced logits against TP8.
Requires numpy; fixture creation additionally requires torch and safetensors.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def run(args, directory):
    fixture = args.fixture
    if fixture is None:
        from test_deepseek_v41_tp_graph import prepare
        fixture = directory / "fixture"
        fixture.mkdir()
        prepare(fixture, hc_mult=4, expert_cache=True, fp8_dense=True, expert_count=16)
    env = dict(os.environ, FASTLLM_SKIP_WARMUP="1", FASTLLM_EXECUTION_MODE="gpu",
               FASTLLM_TP="8", FASTLLM_CUDA_GRAPH="0", OMP_NUM_THREADS="1",
               FASTLLM_CUDA_CUSTOM_ALLREDUCE="0",
               FASTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY="1" if args.fast_prefill else "0")
    results = {}
    for kv in ("bf16", "fp4"):
        reference = None
        for dp in (1, 2, 4, 8):
            output = directory / f"{kv}-dp{dp}.f32"
            command = [str(args.binary), str(fixture), str(dp), str(output)]
            if kv == "fp4":
                command.append("fp4")
            log = directory / f"{kv}-dp{dp}.log"
            with log.open("w") as stream:
                result = subprocess.run(command, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=300)
            if result.returncode == 77:
                raise SystemExit(77)
            if result.returncode:
                raise RuntimeError(f"native regression failed; see {log}")
            values = np.fromfile(output, dtype=np.float32).astype(np.float64)
            assert values.size == 9 * (1 + 3 + 9 + 32) * 1024, output
            assert np.isfinite(values).all(), output
            if reference is None:
                reference = values
            delta = values - reference
            metrics = dict(max_abs=float(abs(delta).max()),
                           rms=float(np.sqrt(np.mean(delta * delta))),
                           cosine=float(np.dot(values, reference) /
                                        np.linalg.norm(values) / np.linalg.norm(reference)),
                           greedy_agreement=float(np.mean(values.reshape(-1, 1024).argmax(1) ==
                                                          reference.reshape(-1, 1024).argmax(1))))
            # TP reduction width changes BF16 rounding. Tolerances are fixed
            # before measurement; prefix relocation is separately exact (1e-5).
            assert metrics["max_abs"] <= 0.5 and metrics["rms"] <= 0.05, metrics
            assert metrics["cosine"] >= 0.99999, metrics
            results[f"{kv}-dp{dp}"] = metrics
            print(kv, f"DP{dp}/TP{8//dp}", json.dumps(metrics), flush=True)
            if args.full_graphs or (args.graphs and dp > 1):
                graph_output = directory / f"{kv}-dp{dp}-graph.f32"
                graph_log = directory / f"{kv}-dp{dp}-graph.log"
                graph_command = command.copy()
                graph_command[3] = str(graph_output)
                graph_command.append("full-graph" if args.full_graphs else "graph")
                with graph_log.open("w") as stream:
                    result = subprocess.run(graph_command,
                        env=dict(env, FASTLLM_DSV41_CUDA_GRAPH_DEBUG="1"),
                        stdout=stream, stderr=subprocess.STDOUT, timeout=300)
                if result.returncode:
                    raise RuntimeError(f"graph regression failed; see {graph_log}")
                replay = np.fromfile(graph_output, dtype=np.float32).astype(np.float64)
                assert np.array_equal(values, replay), f"graph/eager logits differ: {graph_output}"
                log_text = graph_log.read_text()
                assert "graph captured:" in log_text and "graph replay:" in log_text, graph_log
                if args.full_graphs:
                    assert "full graph replay:" in log_text, graph_log
                assert "graph disabled" not in log_text, graph_log
                results[f"{kv}-dp{dp}-graph"] = dict(bitwise_equal=True)
                print(kv, f"DP{dp}/TP{8//dp} graph/eager bitwise equal", flush=True)
                if args.full_graphs and kv == "fp4" and dp == 8:
                    sampling_log = directory / "fp4-dp8-sampling.log"
                    sampling_command = graph_command.copy()
                    sampling_command[3] = str(directory / "fp4-dp8-sampling.f32")
                    sampling_command.append("sampling")
                    with sampling_log.open("w") as stream:
                        result = subprocess.run(sampling_command, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, timeout=300)
                    if result.returncode:
                        raise RuntimeError(f"cross-thread sampling regression failed; see {sampling_log}")
                    results["fp4-dp8-sampling"] = dict(greedy_to_random=True, cross_thread=True)
    (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--fixture", type=lambda value: Path(value).resolve())
    parser.add_argument("--fast-prefill", action="store_true", help="exercise bounded prefill before decode")
    parser.add_argument("--graphs", action="store_true", help="also compare DP graphs against eager logits")
    parser.add_argument("--full-graphs", action="store_true", help="compare full DP/TP decode graphs against eager logits")
    parser.add_argument("--output-dir", type=Path, help="retain logs and logits")
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run(args, args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="v41-attention-dp-") as path:
            run(args, Path(path))


if __name__ == "__main__":
    main()

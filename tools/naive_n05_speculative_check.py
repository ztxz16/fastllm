#!/usr/bin/env python3
"""Execute independent coding checks for naive_n05_speculative_bench reports."""
import argparse
import ast
import builtins
import itertools
import json
from pathlib import Path
import random
import re
import subprocess
import sys


PROMPTS = {
    "merge": "Write a Python function merge_intervals(intervals). Input is a list of [start,end] pairs of integers with start <= end. Merge all overlapping closed intervals, including touching endpoints. Return a sorted list of [start,end] pairs. Do not mutate the input. Empty input returns []. Output only the function, no tests or explanations.",
    "binary": "Write a Python function binary_search_first(values, target). The input list values is sorted in ascending order and may contain duplicates. Return the lowest index whose value equals target, or -1 when absent. Use O(log n) time and O(1) extra space. Output only the function, no tests or explanations.",
    "topo": "Write a Python function topological_sort(n, edges). Vertices are 0 through n-1. Each edge (u,v) requires u before v. Return the lexicographically smallest valid topological ordering, using heapq. Include isolated vertices. Raise ValueError if there is a cycle. Parallel edges are allowed. Output only imports and the function, no tests or explanations.",
}


def write_cases(tokenizer, destination):
    cases = []
    for kind, prompt in PROMPTS.items():
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_dict=False)
        base = {"input_ids": ids, "output_tokens": 768, "confidence_threshold": .5}
        # Prime both paths: rejected proposals can visit experts that the
        # target-only answer never selects, and packing those weights is lazy.
        for draft in [0, 7]:
            cases.append(dict(base, name=f"{kind}_{draft}_warmup", draft_tokens=draft))
        for repeat in range(2):
            for draft in [0, 7]:
                cases.append(dict(base, name=f"{kind}_greedy_{draft}_{repeat}", draft_tokens=draft))
        for draft in [0, 7]:
            cases.append(dict(base, name=f"{kind}_sample_{draft}_0", draft_tokens=draft,
                              top_k=50, top_p=.95, temperature=.8))
    Path(destination).write_text(json.dumps(cases, indent=2) + "\n")


def check(kind, source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__import__":
            raise ValueError("Unexpected private name in generated function")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Unexpected private attribute in generated function")
    def import_module(name, *args, **kwargs):
        if name not in {"heapq", "collections", "bisect", "itertools"}:
            raise ValueError("Unexpected import: " + name)
        return builtins.__import__(name, *args, **kwargs)
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in (
        "len", "range", "sorted", "list", "tuple", "dict", "set", "min", "max", "abs",
        "enumerate", "zip", "reversed", "iter", "next", "ValueError", "bool", "int", "float")}}
    namespace["__builtins__"]["__import__"] = import_module
    exec(compile(tree, "generated.py", "exec"), namespace)
    rng = random.Random(42)
    count = 0
    if kind == "merge":
        function = namespace["merge_intervals"]
        cases = [[], [[1, 2], [2, 3]], [[1, 10], [2, 4]], [[6, 7], [1, 3], [2, 5]]]
        for _ in range(500):
            cases.append([sorted([rng.randrange(-5, 7), rng.randrange(-5, 7)]) for _ in range(rng.randrange(15))])
        for intervals in cases:
            expected = []
            for a, b in sorted(intervals):
                if expected and a <= expected[-1][1]:
                    expected[-1][1] = max(b, expected[-1][1])
                else:
                    expected.append([a, b])
            original = [row[:] for row in intervals]
            assert function(intervals) == expected, (intervals, expected)
            assert intervals == original, "Input was mutated"
            count += 1
    elif kind == "binary":
        function = namespace["binary_search_first"]
        for _ in range(500):
            values = sorted(rng.randrange(-6, 7) for _ in range(rng.randrange(33)))
            for target in range(-7, 8):
                expected = values.index(target) if target in values else -1
                assert function(values, target) == expected, (values, target, expected)
                count += 1
    elif kind == "topo":
        function = namespace["topological_sort"]
        cases = [(0, []), (4, []), (3, [(0, 1), (0, 1), (1, 2)]), (2, [(0, 1), (1, 0)])]
        for _ in range(150):
            n = rng.randrange(1, 8)
            cases.append((n, [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randrange(12))]))
        for n, edges in cases:
            expected = None
            for order in itertools.permutations(range(n)):
                positions = {vertex: i for i, vertex in enumerate(order)}
                if all(positions[u] < positions[v] for u, v in edges):
                    expected = list(order)
                    break
            try:
                actual = function(n, edges)
            except ValueError:
                assert expected is None, (n, edges, expected)
            else:
                assert expected is not None and actual == expected, (n, edges, expected, actual)
            count += 1
    else:
        raise ValueError(kind)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--report")
    parser.add_argument("--output")
    parser.add_argument("--write-cases", help="Generate paired coding benchmark inputs instead of checking a report")
    parser.add_argument("--worker", nargs=2, metavar=("KIND", "SOURCE"))
    args = parser.parse_args()
    if args.worker:
        print(check(args.worker[0], Path(args.worker[1]).read_text()))
        return
    if not args.model:
        parser.error("--model is required")
    if not args.write_cases and (not args.report or not args.output):
        parser.error("--report and --output are required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.write_cases:
        write_cases(tokenizer, args.write_cases)
        return
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    report = json.loads(Path(args.report).read_text())
    rows = report["paired_results"] if isinstance(report, dict) else report
    for row in rows:
        name = row["name"]
        if "output_ids" not in row or name.endswith("warmup"):
            continue
        text = tokenizer.decode(row["output_ids"], skip_special_tokens=True)
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
        source = "\n".join(blocks) if blocks else text
        path = directory / (name + ".py")
        path.write_text(source)
        try:
            process = subprocess.run([sys.executable, __file__, "--worker", name.split("_")[0], str(path)],
                                     capture_output=True, text=True, timeout=15)
            result = {"name": name, "passed": process.returncode == 0,
                      "checks": int(process.stdout.strip()) if process.returncode == 0 else 0,
                      "error": process.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            result = {"name": name, "passed": False, "checks": 0, "error": "Timed out"}
        results.append(result)
        print(json.dumps(result), flush=True)
    (directory / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
    if not results or not all(row["passed"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TP4/DFlash multimodal prefix regression; every measured prompt exceeds 8K.

Starts an isolated server process with the production arguments, plus
--multimodal. Saves the exact command, library hash, responses, cache usage and
server log. Does not stop or modify any pre-existing server.
"""
import argparse
import base64
import copy
import hashlib
import io
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MODEL = "/mnt/nvme/Qwen/Qwen3.8-27B-FP8"
DRAFT = "/mnt/nvme/z-lab/Qwen3.8-27B-DFlash2"
FTLLM = "/home/sy/miniconda3/envs/fastllm2/bin/ftllm"
LIBRARY = "/home/sy/miniconda3/envs/fastllm2/lib/python3.10/site-packages/ftllm/libfastllm_tools.so"
NAME = "Qwen3.8-27B-FP8"


def say(message):
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def image_url(color, size=256):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 100, 100), fill="white")
    data = io.BytesIO()
    image.save(data, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()}}


def scenarios():
    # Count actual tokens through response usage; repetitions only construct the
    # input. A change to tokenization cannot silently turn this into a short test.
    material = ("记录：FastLLM 支持张量并行和视觉语言模型。缓存只应复用输入相同的前缀，"
                "图片内容变化时必须重新计算相关部分。项目代号是青山。\n") * 74
    red, blue = image_url("red"), image_url("blue")
    result = []
    for placement in ("text", "first", "middle", "last"):
        messages = [{"role": "system", "content": "这是缓存回归测试。根据给定材料和图片回答，保持简短。场景=" + placement}]
        image_round = {"first": 0, "middle": 1, "last": 2}.get(placement)
        for turn in range(3):
            content = [{"type": "text", "text": f"第{turn + 1}轮材料：\n" + material}]
            if turn == image_round:
                content.insert(0, copy.deepcopy(red))
            if turn == 2:
                content.append({"type": "text", "text": "请说出项目代号；若有图片，也说出图片的背景颜色。"})
            messages.append({"role": "user", "content": content})
            if turn < 2:
                messages.append({"role": "assistant", "content": "已阅读材料。"})
        result.append((placement + "/cold", copy.deepcopy(messages), False))
        result.append((placement + "/repeat", copy.deepcopy(messages), placement != "text"))
        follow = copy.deepcopy(messages) + [
            {"role": "assistant", "content": "项目代号是青山。"},
            {"role": "user", "content": "结合前面的完整历史，再次回答项目代号和图中背景颜色。"}]
        result.append((placement + "/followup", follow, placement != "text"))
        if placement != "text":
            changed = copy.deepcopy(messages)
            changed[1 + 2 * image_round]["content"][0] = copy.deepcopy(blue)
            result.append((placement + "/changed", changed, False))
            if placement == "first":
                additional = follow + [{"role": "assistant", "content": "已确认。"},
                    {"role": "user", "content": [copy.deepcopy(blue), {"type": "text", "text": "新图与第一张图的背景颜色有何不同？"}]}]
                result.append(("first/append-B", copy.deepcopy(additional), True))
                additional += [{"role": "assistant", "content": "两张图颜色不同。"},
                    {"role": "user", "content": [copy.deepcopy(red), {"type": "text", "text": "按出现顺序列出三张图的背景颜色。"}]}]
                result.append(("first/append-A", additional, True))
    # Local tokenizer + native preprocessing gives 8316 prompt tokens and an
    # image span [8049, 8305): the original 8192-token prefill boundary cuts
    # through its feature rows, exercising restoration of a partial image.
    boundary_material = "".join((material * 3).splitlines(keepends=True)[:205])
    boundary = [{"role": "system", "content": "缓存边界测试。简短回答。"},
        {"role": "user", "content": [{"type": "text", "text": boundary_material},
                                    image_url("red", 512), {"type": "text", "text": "什么颜色？"}]}]
    result += [("inside/cold", copy.deepcopy(boundary), False),
               ("inside/repeat", copy.deepcopy(boundary), True),
               ("inside/followup", boundary + [{"role": "assistant", "content": "红色。"},
                   {"role": "user", "content": "再确认一次背景颜色。"}], True)]
    return result


def chat(messages):
    payload = {"model": NAME, "messages": messages, "temperature": 0,
               "top_p": 1, "max_tokens": 48, "stream": True}
    request = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start, first = time.monotonic(), None
    content, reasoning, usage, finish, done = "", "", {}, None, False
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "): continue
            data = line[6:].strip()
            if data == b"[DONE]": done = True; break
            chunk = json.loads(data)
            if "error" in chunk: raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
                if first is None and (content or reasoning): first = time.monotonic() - start
                finish = choice.get("finish_reason") or finish
    if not done: raise RuntimeError("stream ended without DONE")
    return {"content": content, "reasoning": reasoning, "usage": usage,
            "finish": finish, "ttft": first, "seconds": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cache", choices=("0", "1"), default="1")
    parser.add_argument("--expect-hits", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--case", default="")
    args = parser.parse_args()
    output = ROOT / "build-acceptance/20260920-multimodal-prefix-fix"
    output.mkdir(parents=True, exist_ok=True)
    logfile, resultfile = output / (args.label + ".log"), output / (args.label + ".json")
    if logfile.exists() or resultfile.exists(): raise RuntimeError("Choose a new label; preserving prior evidence")
    proc = None
    result = {"status": "failed", "requests": [], "cache": args.cache, "failures": []}
    try:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8080)) == 0: raise RuntimeError("8080 already in use")
        gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        result["gpu_before"] = gpus
        for line in gpus.splitlines():
            index, memory = map(int, line.split(","))
            if index < 4 and memory > 64: raise RuntimeError("GPU 0-3 are occupied")
        command = [FTLLM, "server", MODEL, "--tp", "0,1,2,3", "--gpu_mem_ratio", "0.95",
            "--chunked_prefill_size", "2048", "--kv_cache_dtype", "fp8_e4m3", "--enable_thinking", "true",
            "--speculative_algorithm", "dflash", "--speculative_draft_model_path", DRAFT, "--draft_tokens", "6",
            "--model_name", NAME, "--host", "0.0.0.0", "--port", "8080", "--multimodal"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", FASTLLM_ACTIVATE_NUMA="ON",
            FASTLLM_NUMA_THREADS="27", FASTLLM_CUDA_DFLASH_TP_BACKBONE="force",
            FASTLLM_MULTIMODAL_PREFIX_CACHE=args.cache, PYTHONUNBUFFERED="1")
        result.update(command=command, library_sha256=hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest())
        with logfile.open("wb") as sink:
            proc = subprocess.Popen(command, env=env, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 600
        say("server starting, pid=" + str(proc.pid))
        while True:
            if proc.poll() is not None: raise RuntimeError("server exited: " + str(proc.returncode))
            if time.monotonic() > deadline: raise RuntimeError("server startup timeout")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2) as response:
                    if any(m["id"] == NAME for m in json.load(response)["data"]): break
            except (OSError, ValueError): pass
            time.sleep(1)
        say("server ready")
        reference = {}
        if args.compare:
            reference = {r["name"]: r for r in json.loads(args.compare.read_text())["requests"]}
        for name, messages, should_hit in scenarios():
            if args.case and not name.startswith(args.case): continue
            say(name + " sending")
            offset = logfile.stat().st_size
            response = chat(messages)
            details = response["usage"].get("prompt_tokens_details") or {}
            cached = int(details.get("cached_tokens") or 0)
            prompt = int(response["usage"].get("prompt_tokens") or 0)
            with logfile.open("rb") as source:
                source.seek(offset)
                markers = [line for line in source.read().decode(errors="replace").splitlines()
                           if "[PrefixCache]" in line or "prefix cache" in line or "[Vision]" in line]
            record = dict(name=name, prompt_tokens=prompt, cached_tokens=cached, response=response, markers=markers)
            result["requests"].append(record)
            resultfile.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            say(f"{name}: cached={cached}/{prompt}, time={response['seconds']:.2f}s")
            assert prompt > 8192, (name, "prompt must exceed 8192 tokens", prompt)
            if args.expect_hits and (should_hit or name in ("text/repeat", "text/followup")):
                assert cached > 0, (name, "expected prefix cache hit")
            if args.expect_hits and name.endswith("/changed"):
                layout = next(m for m in markers if "image layout:" in m)
                first_image = int(re.search(r"spans=\[(\d+),", layout).group(1))
                assert cached <= first_image, (name, "reused KV affected by changed image", cached, first_image)
            if args.cache == "0" and not name.startswith("text/"):
                assert cached == 0, (name, "disabled cache hit")
            if name in reference:
                previous = reference[name]["response"]
                assert reference[name]["prompt_tokens"] == prompt, (name, "comparison prompt length changed")
                record["output_equal"] = all(response[k] == previous[k] for k in ("content", "reasoning", "finish"))
                if not record["output_equal"]:
                    result["failures"].append(name + ": output differs from cold baseline")
                    say(result["failures"][-1])
        assert not result["failures"], result["failures"]
        result["status"] = "ok"
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)
        resultfile.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        say("result: " + result["status"] + " " + str(resultfile))


if __name__ == "__main__":
    main()

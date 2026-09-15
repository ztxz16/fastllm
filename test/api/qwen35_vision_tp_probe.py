"""Exercise a running Qwen3.5 API server with deterministic multimodal requests.

Run once per TP/MTP configuration, using a fresh server for cold-cache evidence:
  python3 test/api/qwen35_vision_tp_probe.py --url http://localhost:8082/v1 \
      --output /tmp/tp3-mtp3 --server-log /tmp/server.log
Pass --compare /tmp/tp3-mtp0/summary.json to require identical greedy responses
and tool argument objects. Start with qwen35_vision_tp_server.py and pass
--require-token-ids to also compare original generated IDs from the server log.
The API does not expose logits; native numeric tests validate feature precision.
Requires Pillow. It never starts or stops a service.
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.request

from PIL import Image, ImageDraw


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def image_part(image, output, name):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    (output / (name + ".png")).write_bytes(data)
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(data).decode()}}, {
            "name": name, "size": list(image.size), "sha256": hashlib.sha256(data).hexdigest()}


class Probe:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.baseline = None
        if args.compare:
            reference = json.loads(args.compare.read_text())
            require(reference.get("passed") is True, "comparison reference did not pass")
            self.baseline = {row["name"]: row for row in reference["cases"]}
        self.model = args.model or self.get_json("models")["data"][0]["id"]

    def get_json(self, path):
        with urllib.request.urlopen(self.args.url.rstrip("/") + "/" + path,
                                    timeout=self.args.timeout) as response:
            return json.load(response)

    def request(self, name, messages, stream=False, tools=None, max_tokens=96):
        payload = {"model": self.model, "messages": messages, "temperature": 0,
                   "top_p": 1, "max_tokens": max_tokens, "stream": stream,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        if stream:
            payload["stream_options"] = {"include_usage": True}
        write_json(self.output / (name + "-request.json"), payload)
        log_start = self.args.server_log.stat().st_size if self.args.server_log else 0
        started = time.monotonic()
        request = urllib.request.Request(self.args.url.rstrip("/") + "/chat/completions",
                                         json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
            raw = response.read().decode()
        (self.output / (name + "-response.txt")).write_text(raw)
        if stream:
            result = self.parse_stream(raw)
        else:
            result = json.loads(raw)
        require("error" not in result, f"{name}: API error {result}")
        require(len(result["choices"]) == 1, f"{name}: expected one choice")
        choice = result["choices"][0]
        require(choice["finish_reason"] in ("stop", "tool_calls"),
                f"{name}: incomplete response: {choice}")
        message = choice["message"]
        usage = result["usage"]
        prompt = usage["prompt_tokens"]
        completion = usage["completion_tokens"]
        details = usage["prompt_tokens_details"]
        cached = details["cached_tokens"]
        require(isinstance(cached, int) and 0 <= cached <= prompt, f"{name}: invalid cache usage")
        require(prompt > 0 and completion > 0, f"{name}: empty token accounting")
        require(usage["total_tokens"] == prompt + completion, f"{name}: invalid total usage")
        if "missed_tokens" in details:
            require(details["missed_tokens"] + cached == prompt, f"{name}: invalid miss accounting")
        multimodal = any(isinstance(m.get("content"), list) and
                         any(p.get("type") == "image_url" for p in m["content"])
                         for m in messages)
        if multimodal:
            require(cached == 0, f"{name}: unverified multimodal KV hit: {cached}")
        normalized = {"content": message.get("content") or "", "tool_calls": []}
        for call in message.get("tool_calls") or []:
            require(isinstance(call.get("id"), str) and call["id"], f"{name}: missing tool ID")
            require(call.get("type") == "function", f"{name}: invalid tool type")
            normalized["tool_calls"].append({"name": call["function"]["name"],
                                              "arguments": json.loads(call["function"]["arguments"])})
        row = {"name": name, "seconds": time.monotonic() - started, "stream": stream,
               "finish_reason": choice["finish_reason"], "usage": usage,
               "greedy_visible_response": normalized}
        if self.args.server_log:
            with self.args.server_log.open("rb") as log:
                log.seek(log_start)
                delta = log.read().decode(errors="replace")
            (self.output / (name + "-server.log")).write_text(delta)
            row["image_cache_hits"] = delta.count("Image embedding cache hit")
            row["image_cache_misses"] = delta.count("Image embedding cache miss")
            row["mtp_profiles"] = re.findall(r"\[Qwen3\.5 MTP profile\][^\n]*", delta)
            captures = [json.loads(line.split("[Vision TP probe tokens] ", 1)[1])
                        for line in delta.splitlines() if "[Vision TP probe tokens] " in line]
            if captures or self.args.require_token_ids:
                require(len(captures) == 1, f"{name}: expected one completed native token capture")
                require(captures[0]["end"] == -1 and captures[0]["token_ids"],
                        f"{name}: incomplete native token capture")
                row["generated_token_ids"] = captures[0]["token_ids"]
                require(len(row["generated_token_ids"]) == completion,
                        f"{name}: captured token count differs from completion usage")
        self.rows.append(row)
        # Save partial evidence even when a later assertion fails.
        self.save(False)
        if self.baseline is not None:
            require(name in self.baseline, f"{name}: missing comparison case")
            expected = self.baseline[name]
            require(normalized == expected["greedy_visible_response"],
                    f"{name}: greedy visible response changed: {normalized} != "
                    f"{expected['greedy_visible_response']}")
            require(choice["finish_reason"] == expected["finish_reason"],
                    f"{name}: finish reason changed")
            if self.args.require_token_ids:
                require("generated_token_ids" in expected, f"{name}: reference has no original token IDs")
                require(row["generated_token_ids"] == expected["generated_token_ids"],
                        f"{name}: original greedy token IDs changed")
        print(f"{name}: {normalized} cached={cached} seconds={row['seconds']:.2f}", flush=True)
        return message, row

    @staticmethod
    def parse_stream(raw):
        content, calls, usage, finish, done = "", {}, None, None, False
        for line in raw.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                continue
            event = json.loads(data)
            require("error" not in event, f"stream error: {event}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                require(choice.get("index", 0) == 0, "unexpected streamed choice index")
                delta = choice.get("delta") or {}
                content += delta.get("content") or ""
                for part in delta.get("tool_calls") or []:
                    index = part["index"]
                    call = calls.setdefault(index, {"id": "", "type": "function",
                                                   "function": {"name": "", "arguments": ""}})
                    if part.get("id"):
                        call["id"] += part["id"]
                    if part.get("type"):
                        require(part["type"] == "function", "invalid streamed tool type")
                    for key in ("name", "arguments"):
                        call["function"][key] += (part.get("function") or {}).get(key) or ""
                if choice.get("finish_reason"):
                    require(finish is None, "multiple stream finish reasons")
                    finish = choice["finish_reason"]
        require(done and usage is not None and finish is not None, "incomplete SSE stream or missing usage")
        require(sorted(calls) == list(range(len(calls))), "noncontiguous streamed tool indexes")
        return {"choices": [{"message": {"role": "assistant", "content": content,
                                         "tool_calls": [calls[i] for i in sorted(calls)]},
                              "finish_reason": finish}], "usage": usage}

    def answer(self, name, messages, expected, **kwargs):
        message, row = self.request(name, messages, **kwargs)
        require(not message.get("tool_calls"), f"{name}: unexpected tool call")
        actual = re.sub(r"\s+", "", message.get("content") or "").upper().strip(".。")
        require(actual == expected, f"{name}: expected {expected!r}, got {message.get('content')!r}")
        return message, row

    def save(self, passed):
        write_json(self.output / "summary.json", {
            "passed": passed, "model": self.model, "url": self.args.url,
            "comparison": str(self.args.compare) if self.args.compare else None,
            "precision_scope": "Original generated token IDs" if self.args.require_token_ids else
                               "Visible greedy responses; native tests must validate features/logits.",
            "cases": self.rows})

    def run(self):
        red, red_meta = image_part(Image.new("RGB", (384, 384), "red"), self.output, "red")
        blue, blue_meta = image_part(Image.new("RGB", (384, 384), "blue"), self.output, "blue")
        shapes = Image.new("RGB", (640, 384), "white")
        draw = ImageDraw.Draw(shapes)
        for x in (40, 240, 440):
            draw.ellipse((x, 40, x + 120, 160), fill="red")
        for x in (140, 380):
            draw.rectangle((x, 230, x + 100, 330), fill="blue")
        shape, shape_meta = image_part(shapes, self.output, "shapes")
        write_json(self.output / "images.json", [red_meta, blue_meta, shape_meta])

        def media(parts, prompt):
            return [{"role": "user", "content": parts + [{"type": "text", "text": prompt}]}]

        color_prompt = "What is the solid background color? Reply with exactly RED or BLUE and nothing else."
        a, cold = self.answer("red-cold", media([red], color_prompt), "RED")
        _, warm = self.answer("red-warm", media([red], color_prompt), "RED")
        if self.args.server_log:
            require(cold["image_cache_misses"] >= 1, "red-cold: expected fresh-server image cache miss")
            require(warm["image_cache_hits"] >= 1 and warm["image_cache_misses"] == 0,
                    "red-warm: expected image cache hit without recomputing vision")
        self.answer("blue-same-size", media([blue], color_prompt), "BLUE")
        order_prompt = "List the solid background colors of the images in order. Output only comma-separated uppercase color names, with one name per image."
        self.answer("red-blue", media([red, blue], order_prompt), "RED,BLUE")
        self.answer("blue-red", media([blue, red], order_prompt), "BLUE,RED")
        self.answer("red-blue-red", media([red, blue, red], order_prompt), "RED,BLUE,RED")
        history = media([red], color_prompt) + [a] + media([blue],
            "What is the solid background color in the NEW image in this message? Output only RED or BLUE.")
        self.answer("append-blue", history, "BLUE")
        self.answer("count-shapes", media([shape],
            "How many red circles are shown? Output only the digit."), "3")
        self.answer("count-shapes-warm", media([shape],
            "How many blue squares are shown? Output only the digit."), "2")
        description = ("Red circles: 3\nBlue squares: 2\nTotal shapes: 5\nCircle color: red\n"
                       "Square color: blue\nBackground color: white\nTop row: circles\nBottom row: squares")
        _, row = self.answer("describe-shapes", media([shape],
            "Describe the image using exactly these eight lines, filling each blank: "
            "Red circles: _\nBlue squares: _\nTotal shapes: _\nCircle color: _\n"
            "Square color: _\nBackground color: _\nTop row: _\nBottom row: _\n"
            "Use digits for counts and lowercase color/shape names. No extra text."),
            re.sub(r"\s+", "", description).upper())
        require(row["usage"]["completion_tokens"] >= 32,
                "describe-shapes: too few tokens to exercise multi-step speculative decoding")
        if self.args.require_mtp:
            acceptance = [tuple(map(int, match.groups()))
                          for profile in row.get("mtp_profiles", [])
                          for match in [re.search(r"accept=\{spec=(\d+),full=(\d+),partial=(\d+)", profile)]
                          if match]
            require(len(acceptance) >= 2 and acceptance[-1][0] > acceptance[0][0] and
                    sum(acceptance[-1][1:]) > sum(acceptance[0][1:]),
                    "describe-shapes: no new accepted speculative drafts within this request")

        tools = [{"type": "function", "function": {"name": "record_color",
            "description": "Record the observed image background color.",
            "parameters": {"type": "object", "properties": {
                "color": {"type": "string", "enum": ["red", "blue"]}},
                "required": ["color"], "additionalProperties": False}}}]
        for stream, part, color in ((False, red, "red"), (True, blue, "blue")):
            name = "tool-stream" if stream else "tool-nonstream"
            conversation = media([part], "Call record_color with the solid background color of this image. You must call the tool; do not answer in prose.")
            message, row = self.request(name, conversation, stream=stream, tools=tools)
            require(row["finish_reason"] == "tool_calls", f"{name}: missing tool_calls finish reason")
            require(row["greedy_visible_response"]["tool_calls"] == [
                {"name": "record_color", "arguments": {"color": color}}], f"{name}: wrong tool arguments")
            call = message["tool_calls"][0]
            conversation += [message, {"role": "tool", "tool_call_id": call["id"],
                "content": json.dumps({"receipt": "ALPHA_731", "color": color})},
                {"role": "user", "content":
                    "Using the receipt and color from the tool result, output exactly this sentence, "
                    "replacing the bracketed fields: The tool returned receipt [receipt] for an image "
                    "with a [color] background. The recorded color is [color], and the receipt confirms "
                    "that the image was processed successfully."}]
            expected = (f"The tool returned receipt ALPHA_731 for an image with a {color} background. "
                        f"The recorded color is {color}, and the receipt confirms that the image was processed successfully.")
            _, row = self.answer(name + "-result", conversation,
                re.sub(r"\s+", "", expected).upper().rstrip("."), stream=stream, tools=tools)
            require(row["usage"]["completion_tokens"] >= 32,
                    f"{name}-result: too few tokens for a multi-step tool continuation")

        prefix = "Reference table for cache regression.\n" + "\n".join(
            f"Entry {i} stores value {i + 17}." for i in range(400))
        messages = [{"role": "user", "content": prefix + "\nIgnore the reference table. What is 1+1? Output only the digit."}]
        self.answer("text-prefix-cold", messages, "2")
        _, row = self.answer("text-prefix-warm", messages, "2")
        require(row["usage"]["prompt_tokens_details"]["cached_tokens"] > 0,
                "text-prefix-warm: expected actual cross-card prefix reuse")
        messages[-1]["content"] = prefix + "\nIgnore the reference table. What is 2+3? Output only the digit."
        self.answer("text-prefix-branch", messages, "5")

        # Exercise snapshot boundaries, not just short requests that happen to
        # miss KV reuse. Image placeholders have identical IDs for both colors.
        # The probe measures tokenizer length first and verifies the alignment;
        # it does not assume that a repeated string is always one token.
        question = "\nIgnore the reference table. What is the solid background color? Reply exactly RED or BLUE."
        _, row = self.answer("boundary-calibration", media([red], prefix + question), "RED")
        length = row["usage"]["prompt_tokens"]
        padding = (self.args.page_size - 1 - length % self.args.page_size) % self.args.page_size
        for offset in (-1, 0, 1):
            count = padding + offset
            if count < 0:
                count += self.args.page_size
            prompt = prefix + " x" * count + question
            name = f"boundary-{offset + 1}"
            _, row = self.answer(name + "-red", media([red], prompt), "RED")
            require(row["usage"]["prompt_tokens"] % self.args.page_size ==
                    (self.args.page_size - 1 + offset) % self.args.page_size,
                    f"{name}: tokenizer padding did not produce the requested page boundary")
            self.answer(name + "-red-warm", media([red], prompt), "RED")
            self.answer(name + "-blue", media([blue], prompt), "BLUE")
        if self.baseline is not None:
            require(set(self.baseline) == {row["name"] for row in self.rows}, "comparison case set differs")
        self.save(True)
        print(f"PASS: {len(self.rows)} cases; artifacts: {self.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8082/v1")
    parser.add_argument("--model")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--server-log", type=Path,
                        help="Fresh server log; verifies actual image cache miss/hit events.")
    parser.add_argument("--require-token-ids", action="store_true",
                        help="Require original token IDs recorded by qwen35_vision_tp_server.py.")
    parser.add_argument("--require-mtp", action="store_true",
                        help="Require draft acceptance logs; start server with existing FASTLLM_QWEN35_MTP_PROFILE=2.")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--page-size", type=int, default=128,
                        help="Actual server KV page length, used for boundary regressions.")
    args = parser.parse_args()
    require(args.page_size > 0, "page size must be positive")
    require(not args.require_token_ids or args.server_log,
            "--require-token-ids requires --server-log")
    require(not args.require_mtp or args.server_log, "--require-mtp requires --server-log")
    require(not (args.output / "summary.json").exists(), "output already contains a run; use a fresh directory")
    Probe(args).run()


if __name__ == "__main__":
    main()

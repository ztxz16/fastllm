"""Check Inkling image order and cache isolation. Use --chunked_prefill_size 64 on the server to split an image span."""

import argparse
import base64
import io
import json
import re
import urllib.request

from PIL import Image


def image_part(color):
    stream = io.BytesIO()
    Image.new("RGB", (123, 81), color).save(stream, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": uri}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    # A cacheable text prefix followed by different pixels must not restore
    # a previous image's KV state when the expanded token IDs are identical.
    prefix = {"type": "text", "text": "Inspect the supplied images. " * 98}

    def run(colors):
        content = [prefix, *(image_part(color) for color in colors)]
        content.append({"type": "text", "text": "Name the solid color of each image in order. Answer only with color names."})
        payload = {"model": args.model, "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "max_tokens": 256, "reasoning_effort": "none", "chat_template_kwargs": {"reasoning_effort": "none"}}
        req = urllib.request.Request(args.url + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.load(response)
        answer = data["choices"][0]["message"]["content"].strip().lower()
        if re.findall(r"\b(?:red|blue)\b", answer) != colors:
            raise AssertionError((colors, answer))
        usage = data["usage"]
        if usage["prompt_tokens_details"]["cached_tokens"] != 0:
            raise AssertionError(usage)
        return usage["prompt_tokens"]

    red, blue = run(["red"]), run(["blue"])
    red_blue, blue_red = run(["red", "blue"]), run(["blue", "red"])
    if not (red == blue < red_blue == blue_red):
        raise AssertionError((red, blue, red_blue, blue_red))


if __name__ == "__main__":
    main()

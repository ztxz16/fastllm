"""Check image-cache reuse and subsequent text batching on an existing server."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import time
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw


def check(base_url, model, api_key="no-key"):
    def request(path, payload=None):
        req = Request(
            base_url.rstrip("/") + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + api_key},
        )
        with urlopen(req, timeout=120) as response:
            return json.load(response)

    if not model:
        model = request("/v1/models")["data"][0]["id"]
    results = {}

    def chat(messages):
        start = time.perf_counter()
        response = request("/v1/chat/completions", {
            "model": model, "messages": messages, "stream": False,
            "max_tokens": 80, "temperature": 0, "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        return {"seconds": time.perf_counter() - start, "response": response}

    def content(result):
        return result["response"]["choices"][0]["message"]["content"]

    image = Image.new("RGB", (448, 448), "blue")
    ImageDraw.Draw(image).ellipse((90, 90, 358, 358), fill="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "请用一句话描述图片中的颜色和形状。"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}]
    results["image_first"] = chat(messages)
    answer = content(results["image_first"])
    if "红" not in answer or ("圆" not in answer and "圈" not in answer):
        raise AssertionError("Unexpected image description: " + answer)
    messages += [{"role": "assistant", "content": answer},
                 {"role": "user", "content": "图片背景是什么颜色？只说颜色。"}]
    results["image_followup"] = chat(messages)
    answer = content(results["image_followup"])
    if "蓝" not in answer:
        raise AssertionError("Unexpected image follow-up: " + answer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [(n, pool.submit(chat, [
            {"role": "user", "content": f"只输出计算结果：{n}乘以{n}。"}]))
            for n in (19, 22)]
        for n, future in futures:
            result = future.result()
            results["concurrent_" + str(n)] = result
            if str(n * n) not in content(result):
                raise AssertionError("Unexpected arithmetic result: " + content(result))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="no-key")
    args = parser.parse_args()
    print(json.dumps(check(args.base_url, args.model, args.api_key),
                     ensure_ascii=False, indent=2))

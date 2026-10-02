"""EPD 代理：对外 OpenAI 兼容入口，编排 encoder / consumer 两个实例。

拓扑：
    客户端 → epd_proxy(:10000)
        ① 抽取请求中的图片 → POST encoder /encode_images（预编码进共享缓存）
        ② 原请求透传 → consumer（ftllm server，命中缓存则跳过 ViT）

encoder 预编码失败只告警不阻断：consumer 会回退到自跑 ViT，服务保持可用。

用法：
    python -m ftllm.epd_proxy --port 10000 \
        --encoder_url http://127.0.0.1:10001 --consumer_url http://127.0.0.1:10002
"""

import argparse
import logging
from typing import Any, Dict, List

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("epd_proxy")

app = FastAPI(title="fastllm EPD proxy")


class ProxyState:
    encoder_url: str = ""
    consumer_url: str = ""
    client: httpx.AsyncClient = None


state = ProxyState()

_HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding"}


def _extract_image_urls_from_openai(body: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return urls
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "input_image"):
                url = part.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if isinstance(url, str) and url:
                    urls.append(url)
    return urls


def _extract_image_urls_from_anthropic(body: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return urls
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            source = part.get("source")
            if isinstance(source, dict) and source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data")
                if isinstance(data, str) and data:
                    urls.append(f"data:{media_type};base64,{data}")
    return urls


async def _prime_encoder(urls: List[str]) -> None:
    if not urls:
        return
    try:
        resp = await state.client.post(
            state.encoder_url + "/encode_images", json={"images": urls}
        )
        resp.raise_for_status()
        result = resp.json()
        logger.info("[EPD-proxy] encoder primed %d image(s)", len(result.get("results", [])))
    except Exception as exc:
        logger.warning(
            "[EPD-proxy] encoder prime failed (%s); consumer 将回退自跑 ViT", exc
        )


def _forward_headers(request: Request) -> Dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_HEADERS
    }


async def _forward_streaming(path: str, body: Dict[str, Any], request: Request) -> StreamingResponse:
    req = state.client.build_request(
        "POST",
        state.consumer_url + path,
        json=body,
        headers=_forward_headers(request),
    )
    resp = await state.client.send(req, stream=True)

    async def _stream():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        _stream(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/event-stream"),
    )


async def _forward_plain(path: str, body: Dict[str, Any], request: Request) -> Response:
    resp = await state.client.post(
        state.consumer_url + path, json=body, headers=_forward_headers(request)
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    await _prime_encoder(_extract_image_urls_from_openai(body))
    if body.get("stream"):
        return await _forward_streaming("/v1/chat/completions", body, request)
    return await _forward_plain("/v1/chat/completions", body, request)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    await _prime_encoder(_extract_image_urls_from_anthropic(body))
    if body.get("stream"):
        return await _forward_streaming("/v1/messages", body, request)
    return await _forward_plain("/v1/messages", body, request)


@app.get("/health")
async def health():
    encoder_ok = False
    consumer_ok = False
    try:
        resp = await state.client.get(state.encoder_url + "/health", timeout=5.0)
        encoder_ok = resp.status_code == 200
    except Exception:
        pass
    try:
        resp = await state.client.get(state.consumer_url + "/v1/models", timeout=5.0)
        consumer_ok = resp.status_code == 200
    except Exception:
        pass
    payload = {
        "encode_cluster": "healthy" if encoder_ok else "unhealthy",
        "decode_cluster": "healthy" if consumer_ok else "unhealthy",
    }
    return JSONResponse(payload, status_code=200 if (encoder_ok and consumer_ok) else 503)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def passthrough(path: str, request: Request):
    body = await request.body()
    resp = await state.client.request(
        request.method,
        f"{state.consumer_url}/{path}",
        content=body,
        params=request.query_params,
        headers=_forward_headers(request),
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


def main():
    parser = argparse.ArgumentParser(description="fastllm EPD proxy")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--encoder_url", type=str, default="http://127.0.0.1:10001")
    parser.add_argument("--consumer_url", type=str, default="http://127.0.0.1:10002")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    state.encoder_url = args.encoder_url.rstrip("/")
    state.consumer_url = args.consumer_url.rstrip("/")
    state.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    logger.info(
        "[EPD-proxy] listening on %s:%d, encoder=%s, consumer=%s",
        args.host, args.port, state.encoder_url, state.consumer_url,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

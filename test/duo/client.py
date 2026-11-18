"""Binary HTTP client for the native Duo realization service (NumPy only)."""

import json
import struct
from urllib.request import Request, urlopen

import numpy as np


class DuoClient:
    def __init__(self, url="http://127.0.0.1:8099", timeout=600):
        self.url, self.timeout, self.session = url.rstrip("/"), timeout, None
        self.info = self._json("/health")
        if self.info["protocol"] != "fastllm-duo-v1":
            raise ValueError("Unsupported Duo service protocol")
        self.h, self.w = self.info["latent_height"], self.info["latent_width"]

    def _json(self, route, data=None):
        request = Request(
            self.url + route,
            None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _packet(self, route, meta, tensors):
        header = json.dumps(meta, separators=(",", ":")).encode()
        body = (
            struct.pack("<I", len(header))
            + header
            + b"".join(np.asarray(x, dtype="<f4").tobytes() for x in tensors)
        )
        request = Request(
            self.url + route,
            body,
            headers={"Content-Type": "application/x-duo-tensors"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            data = response.read()
            if response.headers.get_content_type() == "application/json":
                return json.loads(data)
        if len(data) < 4:
            raise ValueError("Truncated Duo response")
        (size,) = struct.unpack_from("<I", data)
        if size > 65536 or size + 4 > len(data):
            raise ValueError("Invalid Duo response header")
        meta = json.loads(data[4 : 4 + size])
        values = np.frombuffer(data, dtype="<f4", offset=4 + size)
        nv = 5 * self.h * self.w // 4 * 96
        if len(values) != nv + 58 * 32:
            raise ValueError("Invalid latent payload size")
        video = (
            values[:nv]
            .reshape(5, self.h // 2, self.w // 2, 24, 2, 2)
            .transpose(3, 0, 1, 4, 2, 5)
            .reshape(24, 5, self.h, self.w)
        )
        audio = values[nv:].reshape(2, 29, 32).transpose(0, 2, 1)
        return {"video": video.copy(), "audio": audio.copy(), "metadata": meta}

    def reset(self, reference, shared):
        reference = np.asarray(reference)
        shared = np.asarray(shared)
        if (
            reference.shape != (24, 1, self.h, self.w)
            or shared.ndim != 2
            or shared.shape[1] != self.info["text_dim"]
        ):
            raise ValueError("Expected reference[24,1,H,W], shared[N,text_dim]")
        packed = (
            reference.reshape(24, 1, self.h // 2, 2, self.w // 2, 2)
            .transpose(1, 2, 4, 0, 3, 5)
            .reshape(-1, 96)
        )
        result = self._packet(
            "/v1/session", {"shared_rows": len(shared)}, [packed, shared]
        )
        self.session = result["session"]
        return result

    def chunk(self, index, intent, seed=0):
        intent = np.asarray(intent)
        if (
            self.session is None
            or intent.ndim != 2
            or intent.shape[1] != self.info["text_dim"]
        ):
            raise ValueError("Reset a session and provide intent[N,text_dim]")
        return self._packet(
            "/v1/chunk",
            {
                "session": self.session,
                "chunk": index,
                "rows": len(intent),
                "seed": seed,
            },
            [intent],
        )

    def cancel(self):
        return self._json("/v1/cancel", {"session": self.session})


class OfficialFeatureBridge:
    """Use official Intent feature dictionaries and normalized Codec latents.

    Keeps Intent and Codec independent of the native realization process. This
    adapter does not load Intent or change the official decoder's EOF policy.
    """

    def __init__(self, client):
        self.client = client

    @staticmethod
    def _cpu(value):
        return (
            value.detach().float().cpu().numpy()
            if hasattr(value, "detach")
            else np.asarray(value)
        )

    def features(self, features, block):
        if features["protocol"] != "reciprocal_frame_chatml_v5":
            raise ValueError("Expected Duo frame-v5 features")
        values = self._cpu(features["intent_states"])
        spans = [
            values[g["start"] : g["end"]]
            for g in features["text_groups"]
            if g["block"] == block
        ]
        if not spans:
            raise ValueError("Missing Intent feature span")
        return np.concatenate(spans)

    def reset(self, reference, features):
        return self.client.reset(self._cpu(reference), self.features(features, -1))

    def generate(self, features, block, seed=0):
        return self.client.chunk(block, self.features(features, block), seed)

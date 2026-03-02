from typing import Generator, List, Dict

from openai import OpenAI


class ChatClient:
    def __init__(self, port: int, model_name: str):
        self._client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="none",
        )
        self._model_name = model_name

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repeat_penalty: float = 1.0,
    ) -> Generator[str, None, None]:
        extra = {}
        if top_k > 0:
            extra["top_k"] = top_k
        if repeat_penalty != 1.0:
            extra["frequency_penalty"] = repeat_penalty

        stream = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            **extra,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

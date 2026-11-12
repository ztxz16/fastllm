"""JSONL聊天请求预检与正式分词；仅用于可复现的greedy输出一致性验证。"""

import json
import re

from run_process_tp import ROOT  # 初始化与正式包一致的导入路径。
from fastllm_pytools.qwen3_process_chat import prepare_request
from fastllm_pytools.qwen3_process_server import validate_payload


def read_requests(text):
    """/** 逐行拒绝非法结构、重名和不支持语义；补齐明确的greedy默认值，不改写用户对话。 */"""
    items, names = [], set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {'name', 'request'}:
                raise ValueError('each line requires exactly name and request')
            name, request = row['name'], row['request']
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) or name in names:
                raise ValueError('name must be unique and contain 1..64 letters/digits/_/-')
            if not isinstance(request, dict) or 'messages' not in request:
                raise ValueError('request requires messages')
            body = dict(model='qwen3', max_tokens=96, temperature=0.0, top_k=0, top_p=1.0, seed=0, stream=False)
            body.update(request)
            if body['stream'] is True:
                body.setdefault('stream_options', {'include_usage': True})
            validate_payload(body, 'qwen3')
            if any(body[k] != v for k, v in dict(temperature=0.0, top_k=0, top_p=1.0, seed=0).items()):
                raise ValueError('baseline requires temperature=0, top_k=0, top_p=1, seed=0')
            if body['stream'] and body['stream_options'] != {'include_usage': True}:
                raise ValueError('SSE baseline requires stream_options.include_usage=true')
            items.append(dict(name=name, payload=body))
            names.add(name)
        except (ValueError, TypeError) as error:
            raise ValueError(f'JSONL line {number}: {error}') from error
    if not items:
        raise ValueError('JSONL contains no requests')
    return items


def prepare_inputs(items, model, tokenizer):
    """/** 所有对话经正式模板完整分词，超上下文直接拒绝；不截断、不补长或重复请求。 */"""
    inputs = []
    for item in items:
        body = item['payload']
        try:
            request = prepare_request(model, tokenizer, None, messages=body['messages'],
                max_new_tokens=body['max_tokens'], temperature=body['temperature'],
                top_k=body['top_k'], top_p=body['top_p'], seed=body['seed'])
        except ValueError as error:
            raise ValueError(f'input {item["name"]}: {error}') from error
        inputs.append(dict(item, request=request))
    return inputs

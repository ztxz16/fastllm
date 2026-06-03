#!/usr/bin/env python3
import argparse
import ast
import concurrent.futures
import csv
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm


BENCHMARK_ID = "agent_tool"
DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "baseline" / "default_cases.jsonl"
DEFAULT_SYSTEM_PROMPT = """You are an agent in a deterministic tool-use benchmark.
You must respond with exactly one JSON object and no markdown.

To call a tool, use:
{"thought":"short reason","tool":"tool_name","arguments":{"arg":"value"}}

To finish, use:
{"thought":"short reason","final":"answer"}

Rules:
- Use tools when the answer depends on orders, customers, inventory, policies, dates, or arithmetic.
- For arithmetic, always call calculator before giving the final answer. Mental arithmetic is not allowed.
- If the requested final format is a percentage, include the percent sign.
- Do not invent values that can be looked up.
- Keep the final answer concise and match the user's requested format.
- Do not write any text outside the JSON object.
"""


ORDERS = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "customer_id": "C-001",
        "sku": "USB-C-1M",
        "quantity": 3,
        "status": "delivered",
        "total": "35.97",
        "paid": True,
        "created_date": "2026-05-10",
        "delivered_date": "2026-05-12",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "customer_id": "C-002",
        "sku": "NB-14",
        "quantity": 1,
        "status": "processing",
        "total": "1299.00",
        "paid": True,
        "created_date": "2026-05-29",
        "delivered_date": None,
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "customer_id": "C-003",
        "sku": "HD-2TB",
        "quantity": 2,
        "status": "shipped",
        "total": "179.98",
        "paid": True,
        "created_date": "2026-05-20",
        "delivered_date": None,
    },
    "ORD-1004": {
        "order_id": "ORD-1004",
        "customer_id": "C-004",
        "sku": "WM-RED",
        "quantity": 5,
        "status": "cancelled",
        "total": "64.75",
        "paid": False,
        "created_date": "2026-04-16",
        "delivered_date": None,
    },
    "ORD-1005": {
        "order_id": "ORD-1005",
        "customer_id": "C-001",
        "sku": "BAT-AA",
        "quantity": 12,
        "status": "delivered",
        "total": "18.00",
        "paid": True,
        "created_date": "2026-05-02",
        "delivered_date": "2026-05-06",
    },
}


CUSTOMERS = {
    "C-001": {
        "customer_id": "C-001",
        "name": "Ada Chen",
        "tier": "gold",
        "city": "Seattle",
        "reward_points": 8420,
    },
    "C-002": {
        "customer_id": "C-002",
        "name": "Ben Ortiz",
        "tier": "standard",
        "city": "Austin",
        "reward_points": 120,
    },
    "C-003": {
        "customer_id": "C-003",
        "name": "Mira Patel",
        "tier": "platinum",
        "city": "Boston",
        "reward_points": 15200,
    },
    "C-004": {
        "customer_id": "C-004",
        "name": "Omar Green",
        "tier": "silver",
        "city": "Denver",
        "reward_points": 900,
    },
}


INVENTORY = {
    "USB-C-1M": {
        "sku": "USB-C-1M",
        "name": "USB-C cable 1m",
        "product_type": "accessory",
        "stock": 42,
        "unit_price": "11.99",
        "weight_kg": "0.05",
    },
    "NB-14": {
        "sku": "NB-14",
        "name": "14 inch notebook laptop",
        "product_type": "laptop",
        "stock": 3,
        "unit_price": "1299.00",
        "weight_kg": "1.40",
    },
    "HD-2TB": {
        "sku": "HD-2TB",
        "name": "2TB hard drive",
        "product_type": "accessory",
        "stock": 0,
        "unit_price": "89.99",
        "weight_kg": "0.20",
    },
    "WM-RED": {
        "sku": "WM-RED",
        "name": "red wireless mouse",
        "product_type": "accessory",
        "stock": 24,
        "unit_price": "12.95",
        "weight_kg": "0.10",
    },
    "BAT-AA": {
        "sku": "BAT-AA",
        "name": "AA battery pack",
        "product_type": "battery",
        "stock": 200,
        "unit_price": "1.50",
        "weight_kg": "0.02",
    },
    "KB-MECH": {
        "sku": "KB-MECH",
        "name": "mechanical keyboard",
        "product_type": "accessory",
        "stock": 7,
        "unit_price": "84.50",
        "weight_kg": "0.90",
    },
    "MON-27": {
        "sku": "MON-27",
        "name": "27 inch monitor",
        "product_type": "monitor",
        "stock": 2,
        "unit_price": "249.99",
        "weight_kg": "5.30",
    },
    "BAG-LITE": {
        "sku": "BAG-LITE",
        "name": "lightweight laptop bag",
        "product_type": "accessory",
        "stock": 15,
        "unit_price": "39.99",
        "weight_kg": "0.50",
    },
}


KB_DOCS = [
    {
        "doc_id": "return_policy",
        "title": "Return policy",
        "keywords": ["return", "returns", "refund", "退货", "退款"],
        "body": (
            "Delivered orders can be returned within 30 calendar days after delivered_date. "
            "Cancelled orders cannot be returned."
        ),
    },
    {
        "doc_id": "shipping_policy",
        "title": "Shipping policy",
        "keywords": ["shipping", "delivery", "expedited", "business days", "配送", "物流"],
        "body": "Standard shipping takes 5 business days. Expedited shipping takes 2 business days.",
    },
    {
        "doc_id": "warranty_policy",
        "title": "Warranty policy",
        "keywords": ["warranty", "laptop", "battery", "batteries", "保修"],
        "body": (
            "Laptop SKUs such as NB-14 have a 2 years warranty. Accessories have a 1 year warranty. "
            "Batteries such as BAT-AA have a 90 days warranty."
        ),
    },
    {
        "doc_id": "discount_policy",
        "title": "Discount policy",
        "keywords": ["discount", "tier", "gold", "platinum", "silver", "折扣", "会员"],
        "body": (
            "Customer tier discounts: platinum 15%, gold 10%, silver 5%, standard 0%. "
            "Free shipping applies to orders over $100."
        ),
    },
    {
        "doc_id": "escalation_policy",
        "title": "Escalation policy",
        "keywords": ["approval", "finance", "refund", "out-of-stock", "审批", "升级"],
        "body": "Refunds over $500 require finance approval. Out-of-stock items require a supplier ticket.",
    },
]


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "model"


def parse_json_object(value: str, name: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{name} must be a JSON object.")
    return parsed


def load_local_rows(data_file: Path) -> Iterable[Dict[str, Any]]:
    suffix = data_file.suffix.lower()
    if suffix == ".jsonl":
        with data_file.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if suffix == ".json":
        with data_file.open("r", encoding="utf-8") as fin:
            parsed = json.load(fin)
        if isinstance(parsed, dict):
            parsed = parsed.get("data", parsed.get("examples", []))
        if not isinstance(parsed, list):
            raise ValueError("JSON data file must be a list, or contain a data/examples list.")
        for row in parsed:
            yield row
        return
    if suffix == ".csv":
        with data_file.open("r", encoding="utf-8", newline="") as fin:
            for row in csv.DictReader(fin):
                yield row
        return
    raise ValueError("Unsupported local data file. Use .jsonl, .json, or .csv.")


def normalize_example(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    task = str(row.get("task", "")).strip()
    expected = str(row.get("expected", "")).strip()
    if not task:
        raise ValueError(f"Example {idx} has no task.")
    if expected == "":
        raise ValueError(f"Example {idx} has no expected answer.")
    case_id = str(row.get("case_id", row.get("id", f"agent:{idx:04d}"))).strip()
    aliases = row.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [item.strip() for item in aliases.split("|") if item.strip()]
    required_tools = row.get("required_tools", [])
    if isinstance(required_tools, str):
        required_tools = [item.strip() for item in required_tools.split(",") if item.strip()]
    return {
        "benchmark_id": BENCHMARK_ID,
        "case_id": case_id,
        "category": str(row.get("category", "unknown")).strip() or "unknown",
        "task": task,
        "expected": expected,
        "answer_type": str(row.get("answer_type", "text")).strip() or "text",
        "aliases": [str(item).strip() for item in aliases],
        "required_tools": [str(item).strip() for item in required_tools],
    }


def load_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data_file = Path(args.data_file) if args.data_file else DEFAULT_DATA_FILE
    examples = []
    for idx, row in enumerate(load_local_rows(data_file)):
        examples.append(normalize_example(dict(row), idx))
    return examples


def filter_examples(examples: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = examples
    if args.category:
        wanted = {category.lower() for category in args.category}
        selected = [item for item in selected if item["category"].lower() in wanted]
    if args.shuffle:
        rng = random.Random(args.seed)
        selected = list(selected)
        rng.shuffle(selected)
    if args.start:
        selected = selected[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def decimal_to_json(value: Decimal) -> Any:
    if value == value.to_integral_value():
        return int(value)
    return format(value.normalize(), "f")


def normalize_key(value: Any) -> str:
    return str(value).strip().upper()


def get_single_or_many(arguments: Dict[str, Any], primary: str, plural: str) -> Tuple[List[str], bool]:
    value = arguments.get(primary)
    if value is None:
        value = arguments.get(plural)
    if isinstance(value, list):
        return [normalize_key(item) for item in value], True
    if value is None:
        return [], False
    return [normalize_key(value)], False


def tool_lookup_order(arguments: Dict[str, Any]) -> Dict[str, Any]:
    keys, many = get_single_or_many(arguments, "order_id", "order_ids")
    if not keys:
        return {"error": "missing order_id"}
    results = []
    for key in keys:
        item = ORDERS.get(key)
        results.append(item if item else {"order_id": key, "error": "not_found"})
    return {"orders": results} if many else results[0]


def tool_lookup_customer(arguments: Dict[str, Any]) -> Dict[str, Any]:
    keys, many = get_single_or_many(arguments, "customer_id", "customer_ids")
    if not keys:
        return {"error": "missing customer_id"}
    results = []
    for key in keys:
        item = CUSTOMERS.get(key)
        results.append(item if item else {"customer_id": key, "error": "not_found"})
    return {"customers": results} if many else results[0]


def tool_inventory_check(arguments: Dict[str, Any]) -> Dict[str, Any]:
    keys, many = get_single_or_many(arguments, "sku", "skus")
    if not keys:
        return {"error": "missing sku"}
    results = []
    for key in keys:
        item = INVENTORY.get(key)
        results.append(item if item else {"sku": key, "error": "not_found"})
    return {"items": results} if many else results[0]


def preprocess_expression(expression: str) -> str:
    expression = expression.replace("$", "").replace(",", "")
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expression)
    return expression


def eval_decimal_node(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return eval_decimal_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.Num):
        return Decimal(str(node.n))
    if isinstance(node, ast.UnaryOp):
        value = eval_decimal_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    if isinstance(node, ast.BinOp):
        left = eval_decimal_node(node.left)
        right = eval_decimal_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if right != right.to_integral_value() or abs(int(right)) > 12:
                raise ValueError("unsupported exponent")
            return left ** int(right)
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def tool_calculator(arguments: Dict[str, Any]) -> Dict[str, Any]:
    expression = str(arguments.get("expression", "")).strip()
    if not expression:
        return {"error": "missing expression"}
    expression = preprocess_expression(expression)
    try:
        parsed = ast.parse(expression, mode="eval")
        result = eval_decimal_node(parsed)
    except Exception as exc:
        return {"error": "invalid_expression", "message": str(exc)}
    return {"expression": expression, "result": decimal_to_json(result)}


def tool_date_diff(arguments: Dict[str, Any]) -> Dict[str, Any]:
    start_date = arguments.get("start_date", arguments.get("from_date"))
    end_date = arguments.get("end_date", arguments.get("to_date"))
    if not start_date or not end_date:
        return {"error": "missing start_date or end_date"}
    try:
        start = date.fromisoformat(str(start_date))
        end = date.fromisoformat(str(end_date))
    except ValueError as exc:
        return {"error": "invalid_date", "message": str(exc)}
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days": (end - start).days,
    }


def keyword_score(query: str, doc: Dict[str, Any]) -> int:
    query_norm = query.lower()
    score = 0
    for keyword in doc["keywords"]:
        if keyword.lower() in query_norm:
            score += 5
    words = set(re.findall(r"[a-z0-9]+", query_norm))
    haystack = f"{doc['title']} {doc['body']}".lower()
    for word in words:
        if len(word) > 2 and word in haystack:
            score += 1
    return score


def tool_search_kb(arguments: Dict[str, Any]) -> Dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"error": "missing query"}
    try:
        top_k = int(arguments.get("top_k", 2))
    except (TypeError, ValueError):
        top_k = 2
    scored = [(keyword_score(query, doc), doc) for doc in KB_DOCS]
    scored.sort(key=lambda item: item[0], reverse=True)
    docs = [
        {key: doc[key] for key in ("doc_id", "title", "body")}
        for score, doc in scored
        if score > 0
    ][: max(1, min(top_k, 5))]
    if not docs:
        docs = [{key: KB_DOCS[0][key] for key in ("doc_id", "title", "body")}]
    return {"query": query, "results": docs}


ToolFn = Callable[[Dict[str, Any]], Dict[str, Any]]


TOOLS: Dict[str, Tuple[str, ToolFn]] = {
    "lookup_order": (
        "Lookup order details. Arguments: {\"order_id\":\"ORD-1001\"} or {\"order_ids\":[...]}",
        tool_lookup_order,
    ),
    "lookup_customer": (
        "Lookup customer profile. Arguments: {\"customer_id\":\"C-001\"} or {\"customer_ids\":[...]}",
        tool_lookup_customer,
    ),
    "inventory_check": (
        "Lookup inventory, price, product type, and weight. Arguments: {\"sku\":\"USB-C-1M\"} or {\"skus\":[...]}",
        tool_inventory_check,
    ),
    "search_kb": (
        "Search policy docs. Useful queries: return policy, shipping policy, warranty policy, discount policy, escalation policy. Arguments: {\"query\":\"discount policy\",\"top_k\":2}",
        tool_search_kb,
    ),
    "calculator": (
        "Evaluate a numeric expression with +, -, *, /, parentheses, and percentages. Arguments: {\"expression\":\"84.50 * 2 * 0.90\"}",
        tool_calculator,
    ),
    "date_diff": (
        "Compute calendar day difference. Arguments: {\"start_date\":\"2026-05-12\",\"end_date\":\"2026-06-02\"}",
        tool_date_diff,
    ),
}


def build_system_prompt(base_prompt: str) -> str:
    tool_lines = "\n".join(f"- {name}: {description}" for name, (description, _) in TOOLS.items())
    return f"{base_prompt.strip()}\n\nAvailable tools:\n{tool_lines}"


def build_task_prompt(example: Dict[str, Any]) -> str:
    return (
        f"Task:\n{example['task']}\n\n"
        "Remember: respond with exactly one JSON object. Use a tool if needed."
    )


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = strip_think_blocks(text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_action(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if "final" not in parsed and "final_answer" in parsed:
        parsed["final"] = parsed["final_answer"]
    if "final" not in parsed and parsed.get("action") in {"finish", "final"}:
        parsed["final"] = parsed.get("answer", parsed.get("content", ""))
    if "tool" not in parsed and "action" in parsed and parsed.get("action") not in {"finish", "final"}:
        parsed["tool"] = parsed["action"]
    if "arguments" not in parsed:
        parsed["arguments"] = parsed.get("args", {})
    if isinstance(parsed.get("arguments"), str):
        try:
            args = json.loads(parsed["arguments"])
            parsed["arguments"] = args if isinstance(args, dict) else {"value": parsed["arguments"]}
        except json.JSONDecodeError:
            parsed["arguments"] = {"value": parsed["arguments"]}
    if parsed.get("arguments") is None:
        parsed["arguments"] = {}
    return parsed


def usage_value(usage: Dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def build_payload(
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "stream": False,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    payload.update(extra_body)
    return payload


def post_chat_completion(
    session: requests.Session,
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    url = f"{normalize_base_url(args.base_url)}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    payload = build_payload(args, messages, extra_body)

    last_error = ""
    for attempt in range(args.max_retries + 1):
        start = time.perf_counter()
        try:
            response = session.post(url, headers=headers, json=payload, timeout=args.timeout)
            latency_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            usage = data.get("usage") or {}
            return {
                "content": content,
                "latency_ms": latency_ms,
                "input_tokens": usage_value(usage, "prompt_tokens", "input_tokens"),
                "output_tokens": usage_value(usage, "completion_tokens", "output_tokens"),
                "raw": data,
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= args.max_retries:
                break
            time.sleep(args.retry_sleep * (attempt + 1))

    raise RuntimeError(f"API request failed after retries: {last_error}; last_latency_ms={latency_ms:.2f}")


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = strip_think_blocks(text)
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    text = text.strip(" \t\r\n\"'`")
    text = text.replace(" percent", "%").replace(" pct", "%")
    text = re.sub(r"\s+%", "%", text)
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".。")
    return text


def normalize_bool(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if text in {"yes", "true", "y", "1", "是", "可以", "满足", "足够", "有货"}:
        return "yes"
    if text in {"no", "false", "n", "0", "否", "不", "不可以", "不能", "不足", "没货"}:
        return "no"
    if re.match(r"^(yes|true)\b", text):
        return "yes"
    if re.match(r"^(no|false)\b", text):
        return "no"
    return None


def decimal_from_text(value: Any) -> Optional[Decimal]:
    text = str(value).strip()
    text = text.replace(",", "").replace("$", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def answer_matches(prediction: Optional[str], example: Dict[str, Any]) -> bool:
    if prediction is None:
        return False
    expected_values = [example["expected"]] + list(example.get("aliases", []))
    answer_type = example.get("answer_type", "text")

    if answer_type == "number":
        pred_num = decimal_from_text(prediction)
        if pred_num is None:
            return False
        for expected in expected_values:
            exp_num = decimal_from_text(expected)
            if exp_num is not None and abs(pred_num - exp_num) <= Decimal("0.000001"):
                return True
        return False

    if answer_type == "boolean":
        pred_bool = normalize_bool(prediction)
        if pred_bool is None:
            return False
        return any(pred_bool == normalize_bool(expected) for expected in expected_values)

    pred_text = normalize_text(prediction)
    return any(pred_text == normalize_text(expected) for expected in expected_values)


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name not in TOOLS:
        return {"error": "unknown_tool", "tool": name}
    _, tool_fn = TOOLS[name]
    try:
        return tool_fn(arguments)
    except Exception as exc:
        return {"error": "tool_exception", "message": f"{type(exc).__name__}: {exc}"}


def has_tool_error(result: Any) -> bool:
    if isinstance(result, dict):
        if "error" in result:
            return True
        return any(has_tool_error(value) for value in result.values())
    if isinstance(result, list):
        return any(has_tool_error(item) for item in result)
    return False


def run_case(
    example: Dict[str, Any],
    args: argparse.Namespace,
    extra_body: Dict[str, Any],
    system_prompt: str,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_task_prompt(example)},
    ]
    raw_outputs: List[str] = []
    observations: List[Dict[str, Any]] = []
    used_tools: List[str] = []
    request_latencies: List[float] = []
    invalid_actions = 0
    tool_errors = 0
    input_tokens = 0
    output_tokens = 0
    final_answer: Optional[str] = None
    error = ""

    case_start = time.perf_counter()
    with requests.Session() as session:
        session.trust_env = args.use_env_proxy
        for _ in range(args.max_steps):
            try:
                response = post_chat_completion(session, args, messages, extra_body)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break

            request_latencies.append(response["latency_ms"])
            input_tokens += response["input_tokens"]
            output_tokens += response["output_tokens"]
            content = response["content"]
            raw_outputs.append(content)
            messages.append({"role": "assistant", "content": content})

            parsed = extract_json_object(content)
            if parsed is None:
                invalid_actions += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last response was not a valid JSON object. "
                            "Continue with exactly one JSON object only."
                        ),
                    }
                )
                continue

            action = normalize_action(parsed)
            if "final" in action:
                final_answer = str(action.get("final", "")).strip()
                break

            tool_name = str(action.get("tool", "")).strip()
            arguments = action.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            result = execute_tool(tool_name, arguments)
            used_tools.append(tool_name)
            if has_tool_error(result):
                tool_errors += 1
            observation = {
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            }
            observations.append(observation)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Observation: "
                        + json.dumps(observation, ensure_ascii=False, sort_keys=True)
                        + "\nContinue with exactly one JSON object. "
                        "Call another tool or finish with a final answer."
                    ),
                }
            )
        else:
            error = "max_steps_exceeded"

    latency_ms = (time.perf_counter() - case_start) * 1000
    required_tools = example.get("required_tools", [])
    tool_plan_correct = all(tool in used_tools for tool in required_tools)
    correct = answer_matches(final_answer, example)
    return {
        "benchmark_id": BENCHMARK_ID,
        "case_id": example["case_id"],
        "category": example["category"],
        "task": example["task"],
        "expected": example["expected"],
        "answer_type": example["answer_type"],
        "prediction": final_answer,
        "correct": correct,
        "required_tools": required_tools,
        "used_tools": used_tools,
        "tool_plan_correct": tool_plan_correct,
        "steps": len(raw_outputs),
        "requests": len(raw_outputs),
        "invalid_actions": invalid_actions,
        "tool_errors": tool_errors,
        "raw_outputs": raw_outputs,
        "observations": observations,
        "request_latency_ms": request_latencies,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": error,
    }


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def make_summary(results: List[Dict[str, Any]], runtime_seconds: float, args: argparse.Namespace) -> Dict[str, Any]:
    total = len(results)
    answered = sum(1 for item in results if item.get("prediction") not in (None, ""))
    correct = sum(1 for item in results if item.get("correct"))
    tool_plan_correct = sum(1 for item in results if item.get("tool_plan_correct"))
    invalid_cases = sum(1 for item in results if item.get("invalid_actions", 0) > 0)
    tool_error_cases = sum(1 for item in results if item.get("tool_errors", 0) > 0)
    api_error_cases = sum(1 for item in results if item.get("error") and item.get("error") != "max_steps_exceeded")
    max_step_cases = sum(1 for item in results if item.get("error") == "max_steps_exceeded")
    latencies = [float(item["latency_ms"]) for item in results if item.get("latency_ms") is not None]
    request_latencies = [
        float(value)
        for item in results
        for value in item.get("request_latency_ms", [])
        if value is not None
    ]
    input_tokens = sum(int(item.get("input_tokens") or 0) for item in results)
    output_tokens = sum(int(item.get("output_tokens") or 0) for item in results)
    steps = [int(item.get("steps") or 0) for item in results]

    by_category: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[item.get("category", "unknown")].append(item)
    for category, items in sorted(grouped.items()):
        by_category[category] = {
            "total": len(items),
            "correct": sum(1 for item in items if item.get("correct")),
            "accuracy": (sum(1 for item in items if item.get("correct")) / len(items)) if items else 0,
            "tool_plan_accuracy": (
                sum(1 for item in items if item.get("tool_plan_correct")) / len(items)
            )
            if items
            else 0,
        }

    return {
        "benchmark_id": BENCHMARK_ID,
        "total": total,
        "answered": answered,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "tool_plan_correct": tool_plan_correct,
        "tool_plan_accuracy": tool_plan_correct / total if total else 0,
        "invalid_action_cases": invalid_cases,
        "tool_error_cases": tool_error_cases,
        "api_error_cases": api_error_cases,
        "max_step_cases": max_step_cases,
        "steps_avg": sum(steps) / len(steps) if steps else 0,
        "steps_p50": percentile(steps, 50),
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else None,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p90": percentile(latencies, 90),
        "latency_ms_p95": percentile(latencies, 95),
        "request_latency_ms_avg": sum(request_latencies) / len(request_latencies)
        if request_latencies
        else None,
        "request_latency_ms_p50": percentile(request_latencies, 50),
        "request_latency_ms_p95": percentile(request_latencies, 95),
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "runtime_seconds": runtime_seconds,
        "throughput_items_per_second": total / runtime_seconds if runtime_seconds > 0 else 0,
        "input_tokens_per_second": input_tokens / runtime_seconds if runtime_seconds > 0 else 0,
        "output_tokens_per_second": output_tokens / runtime_seconds if runtime_seconds > 0 else 0,
        "by_category": by_category,
        "settings": {
            "model": args.model,
            "base_url": args.base_url,
            "workers": args.workers,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
    }


def default_output_path(model: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model = sanitize_filename(model)
    return Path("test/agent/results") / f"{BENCHMARK_ID}_{safe_model}_{stamp}.jsonl"


def load_done_case_ids(output_file: Path) -> set:
    done = set()
    if not output_file.exists():
        return done
    with output_file.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = row.get("case_id")
            if case_id:
                done.add(case_id)
    return done


def write_summary(output_file: Path, summary: Dict[str, Any]) -> Path:
    summary_file = output_file.with_suffix(".summary.json")
    with summary_file.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
        fout.write("\n")
    return summary_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent tool-use evaluation via OpenAI-compatible chat API.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL, e.g. http://localhost:1616")
    parser.add_argument("--model", required=True, help="Model name passed to the API.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE), help="Local .jsonl/.json/.csv case file.")
    parser.add_argument("--category", action="append", help="Only run a category. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--extra-body", default="", help="Extra JSON object merged into request body.")
    parser.add_argument("--use-env-proxy", action="store_true", help="Honor HTTP_PROXY/HTTPS_PROXY from environment.")
    parser.add_argument("--output-file", default="", help="JSONL output path.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")

    extra_body = parse_json_object(args.extra_body, "--extra-body")
    examples = filter_examples(load_examples(args), args)
    output_file = Path(args.output_file) if args.output_file else default_output_path(args.model)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and args.overwrite and not args.resume:
        output_file.unlink()
    if output_file.exists() and not args.overwrite and not args.resume:
        raise SystemExit(f"Output file exists. Use --overwrite or --resume: {output_file}")

    done_case_ids = load_done_case_ids(output_file) if args.resume else set()
    pending = [example for example in examples if example["case_id"] not in done_case_ids]
    if not pending:
        print("No pending examples.")
        return

    system_prompt = build_system_prompt(args.system_prompt)
    results: List[Dict[str, Any]] = []
    start = time.perf_counter()

    progress = tqdm(total=len(pending), desc="Agent", disable=args.no_progress)
    with output_file.open("a", encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(run_case, example, args, extra_body, system_prompt): example
                for example in pending
            }
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                results.append(result)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                completed = len(results)
                correct = sum(1 for item in results if item.get("correct"))
                tool_plan = sum(1 for item in results if item.get("tool_plan_correct"))
                progress.set_postfix(
                    acc=f"{correct / completed:.4f}",
                    tool=f"{tool_plan / completed:.4f}",
                )
                progress.update(1)
    progress.close()

    runtime_seconds = time.perf_counter() - start
    if args.resume and done_case_ids:
        all_results = []
        with output_file.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))
        results_for_summary = all_results
    else:
        results_for_summary = results
    summary = make_summary(results_for_summary, runtime_seconds, args)
    summary_file = write_summary(output_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()

"""解析并校验FastLLM Attention Backend正式路径记录。"""


TRACE_PREFIX = "[fastllm][attention] "
KV_CACHE_TRACE_PREFIX = "[fastllm][kv-cache] "


def _parse_fields(line, prefix):
    """解析由空格分隔的key=value正式路径记录。"""
    fields = {}
    for item in line[len(prefix):].split():
        key, separator, value = item.partition("=")
        if separator:
            fields[key] = value
    return fields


def _normalize_kv_cache_layout(layout):
    """统一底层contiguous名称和CLI continuous名称。"""
    return "continuous" if layout == "contiguous" else layout


def parse_attention_backend_traces(output):
    """把每个静态调用签名的首次Backend选择记录解析为字段字典。"""
    traces = []
    for line in output.splitlines():
        if not line.startswith(TRACE_PREFIX):
            continue
        fields = _parse_fields(line, TRACE_PREFIX)
        if fields.get("actual"):
            traces.append(fields)
    return traces


def actual_attention_backends(output):
    """返回日志中确认执行且非none的Backend名称有序集合。"""
    return sorted({
        trace["actual"] for trace in parse_attention_backend_traces(output)
        if trace["actual"] != "none"
    })


def attention_backend_confirmed(requested, actual_backends):
    """判断auto是否产生实际路径，或显式Backend是否确实执行。"""
    return (bool(actual_backends) if requested == "auto" else
            requested in actual_backends)


def actual_kv_cache_layouts(output):
    """返回模型布局记录或Attention选择记录中的实际布局有序集合。"""
    layouts = set()
    for trace in parse_attention_backend_traces(output):
        if trace.get("cache"):
            layouts.add(_normalize_kv_cache_layout(trace["cache"]))
    for line in output.splitlines():
        if not line.startswith(KV_CACHE_TRACE_PREFIX):
            continue
        actual = _parse_fields(line, KV_CACHE_TRACE_PREFIX).get("actual")
        if actual:
            layouts.add(_normalize_kv_cache_layout(actual))
    return sorted(layouts)


def kv_cache_layout_confirmed(requested, actual_layouts):
    """判断auto是否产生实际布局，或显式布局是否确实执行。"""
    return (bool(actual_layouts) if requested == "auto" else
            requested in actual_layouts)

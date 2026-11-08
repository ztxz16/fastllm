class PromptTooLongError(ValueError):
    """The backend rejected the context and has already released its handle."""

    def __init__(self):
        super().__init__(
            "上下文长度不足（prompt too long）：对话内容超过当前模型或 KV 缓存允许的长度。"
            "请新建对话，或在模型和显存允许的情况下增大上下文长度后重试。")


class CapacityUnavailableError(ValueError):
    """The backend could not obtain a safe replacement KV page."""

    def __init__(self):
        super().__init__(
            "KV 缓存容量不足：没有可安全回收或迁移的页面，当前请求已失败。"
            "请减少并发或上下文长度后重试。")

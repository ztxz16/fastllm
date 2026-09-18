# JSON 结构化输出

Chat Completions 的 `response_format` 与 Responses 的 `text.format` 使用同一套服务层实现，适用于不同模型，不需要额外环境变量。

支持 `text`（普通文本，默认）、`json_object`（JSON 对象）和 `json_schema`（指定 JSON Schema）。例如：

```json
{
  "model": "your-model",
  "input": "为检查目录内容的任务起一个标题",
  "text": {
    "format": {
      "type": "json_schema",
      "name": "task_title",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"title": {"type": "string", "maxLength": 36}},
        "required": ["title"],
        "additionalProperties": false
      }
    }
  }
}
```

Chat Completions 对应写法为 `"response_format": {"type": "json_schema", "json_schema": {"name": "task_title", "strict": true, "schema": {...}}}`。
JSON 对象模式为 `"response_format": {"type": "json_object"}`，或 `"text": {"format": {"type": "json_object"}}`。

实现采用通用系统提示引导生成，并在完成时校验 JSON 和 Schema；**尚未使用逐 token 的语法约束解码**，成功率仍取决于模型的指令遵循能力。无论 `strict` 是否设置，完成的答案都会接受格式校验。嵌套对象、数组、枚举和文档内 `$ref` 由 `jsonschema` 校验；不会联网获取外部 `$ref`。

流式输出仍逐段发送，收到成功的结束事件之前，内容只是未完成的候选结果。若最终校验失败，Chat Completions 返回 `invalid_response_format` 错误，Responses 流以 `response.failed` 结束，不会伪报成功。非流式校验失败返回 HTTP 500；非法格式或 Schema 在推理前返回 HTTP 400。

格式要求作用于最终文本答案，不会禁止正常工具调用。达到输出 token 上限时仍按截断处理（`length` / `incomplete`），不误判为 Schema 错误。普通文本请求不增加格式提示或输出校验。

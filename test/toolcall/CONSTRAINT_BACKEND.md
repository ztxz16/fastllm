# Tool-call constraint contract

The Python API builds request-aware tool constraints and passes them through `set_tool_call_constraint_llm_model` in `tools/src/pytools.cpp` before generation. Native builds expose this entry point; Python tolerates older libraries that lack it.

The native bridge accepts supported `tool_name_enum` formats, optional `tool_parameter_name_enum` mappings, and the supported Kimi tool-content sampling payload. Pending state is consumed into the next generation configuration; JSON `null` clears it. Requests must not inherit another request's allowed names.

Name constraints restrict tool and parameter names in supported formats. Carrying `json_schemas`, `structural_tag` or grammar metadata does not by itself enforce arbitrary JSON Schema during decoding. Parser/API validation, name constraints and general structured output are separate contracts; see [structured output](../../docs/structured-output.md).

Run the [deterministic parser tests](README.md) for protocol behavior and the manual live runner for model adherence. Golden parser tests do not establish real-model quality or arbitrary-schema enforcement.

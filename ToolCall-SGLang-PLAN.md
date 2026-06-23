# FastLLM ToolCall SGLang-Style Plan

This plan replaces the earlier alias-first roadmap. The target is to implement a SGLang-inspired toolcall stack for FastLLM:

- a unified function-call parser layer;
- model-specific detectors/parsers;
- request-tool-aware validation;
- deterministic diagnostics for unknown or malformed calls;
- optional compatibility mode;
- future generation-time structural constraints.

The key design point is that the parser must not fuzzy-correct unknown tool names. If the model emits `get_wearher` while the request exposes `get_weather`, that should be reported as an invalid tool call unless a user explicitly configured a deterministic alias outside this SGLang-style path.

## SGLang Reference Model

SGLang organizes tool calls around these concepts:

- `FunctionCallParser`
  - Owns available request tools.
  - Owns one model-specific detector.
  - Provides non-stream and stream parse entry points.
  - Produces structural/json constraints when requested.

- `BaseFormatDetector`
  - Parses a model-specific toolcall wire format.
  - Maintains streaming parser state.
  - Validates emitted tool names against the request tool list.
  - Unknown tools are not silently accepted by default.

- Model-specific detectors
  - DeepSeek V4 detector mostly configures DSML start/end tokens and structural tag name.

- Constraint generation
  - For `tool_choice="required"`, named tool choice, or strict schemas, SGLang builds a structural tag or JSON schema constraint.
  - Function names are represented as enums so the model is constrained to valid tool names during generation.

FastLLM should not copy SGLang line-for-line, but it should copy the separation of responsibilities. In particular, FastLLM should be stricter than SGLang's legacy fail-open parser path when producing diagnostics: unknown tools should be observable as invalid tool calls by default, not silently treated as successful calls.

## Non-Goals

- Do not fuzzy-correct parser output.
- Do not silently map unknown tool names based on string similarity.
- Do not introduce model downloads, server startup, or GPU-dependent tests.
- Do not make live tests part of deterministic CI.
- Do not implement full xgrammar/structural decoding in the first milestone.
- Do not make alias mapping the primary solution in this plan.

Explicit aliasing may remain a separate opt-in mitigation, but it is not the SGLang-style core.

## Milestone 1: FunctionCall Parser Facade And Validation Layer

Goal: introduce a SGLang-style parser facade that wraps existing FastLLM parsers and validates parsed tool names against `request.tools`.

Implementation:

- Add a module such as:
  - `tools/fastllm_pytools/openai_server/toolcall_parser.py`
- Add a request-local parser facade:
  - `FunctionCallParser`
- Inputs:
  - `tools`
  - `tool_choice`
  - `tool_parser_name` or model type
  - tokenizer
- Public methods:
  - `has_tool_call(text) -> bool`
  - `parse_non_stream(text, request) -> ToolCallParseResult`
  - `parse_stream_chunk(...) -> DeltaMessage or parse result`
  - `validate_tool_calls(tool_calls) -> ToolCallValidationResult`
- Keep the existing `ToolParserManager` parsers underneath.
- Do not implement a second detector framework in this milestone. The facade is an orchestration and validation wrapper over existing FastLLM parsers.
- Add `ToolCallParseResult`:
  - `content`
  - `tools_called`
  - `valid_tool_calls`
  - `invalid_tool_calls`
  - `diagnostics`
  - `has_invalid_tool_block`
- Add a request-tool-name index:
  - `{"get_weather": 0, "get_time": 1}`
- Validate parser output:
  - tool name in request tools: valid
  - tool name not in request tools: invalid
- Default behavior for invalid names should be strict diagnostics, not fuzzy correction.
- Initially, do not change OpenAI response behavior until tests pin the desired policy.

Unit tests:

- Build tool index from OpenAI `tools`.
- Empty tools means no toolcall parsing.
- Non-stream valid tool name passes.
- Non-stream unknown tool name produces `invalid_tool_name`.
- Stream valid tool name passes.
- Stream unknown tool name produces `invalid_tool_name`.
- `tool_choice` is stored but not enforced yet.
- No mutation of request/tools.

Integration tests:

- Use scripted DeepSeek V4 DSML:
  - valid `<invoke name="get_weather">` passes;
  - invalid `<invoke name="get_wearher">` is detected.
- Cover non-stream and stream.
- Run through deterministic unittest only.

Acceptance:

- Existing parser golden tests still pass.
- New validation tests pass.
- Unknown names are observable with stable diagnostics.

## Milestone 2: Wire Facade Into OpenAI Non-Stream Response Path

Goal: route non-stream OpenAI toolcall parsing through the new facade.

Implementation:

- In `FastLLmCompletion.chat_completion_full_generator`, replace direct parser calls:
  - from `tool_parser.extract_tool_calls(...)`
  - to `FunctionCallParser.parse_non_stream(...)`
- If all tool calls are valid:
  - preserve current OpenAI-compatible response shape.
- If invalid tool calls are found:
  - choose one explicit policy and test it.

Recommended default policy:

- Do not expose invalid unknown tool calls as OpenAI `tool_calls`.
- Do not return raw toolcall markup such as DSML blocks as normal assistant content.
- If the output contains an invalid tool block and no valid tool calls, return an explicit OpenAI error response or a sanitized assistant response according to one tested policy.
- Preferred first implementation: return an OpenAI error response with `invalid_tool_name` diagnostics.
- Attach server logs/diagnostics internally.
- For deterministic tests, expose diagnostics through helper-level tests rather than API response unless production protocol supports it.

Alternative policy:

- Return sanitized assistant content with no raw toolcall markup and log diagnostics.

Do not pick both. Prefer the explicit error response initially because it avoids leaking raw DSML/toolcall markup as assistant content.

Unit tests:

- `FunctionCallParser` returns valid parse result for known tool.
- Unknown tool is filtered or rejected according to chosen policy.
- Content is not corrupted when invalid tool call is rejected.
- Parser result preserves arguments JSON for valid tool.

Integration tests:

- Server-mock non-stream valid toolcall response.
- Server-mock non-stream unknown tool name:
  - scripted output calls `get_wearher`;
  - expected no external `tool_calls` or expected error, depending on chosen policy.
- Regression: plain text no-tool response unchanged.

Acceptance:

- Non-stream unknown tool name is no longer silently exposed as a valid OpenAI tool call.
- No parser fuzzy correction.

## Milestone 3: Wire Facade Into OpenAI Streaming Response Path

Goal: apply the same validation semantics to streaming `delta.tool_calls`.

Implementation:

- Wrap existing parser streaming calls through `FunctionCallParser.parse_stream_chunk`.
- Maintain validation state per stream tool index.
- If a tool name is known:
  - stream as before.
- If a tool name is unknown:
  - do not emit it as a valid `delta.tool_calls`;
  - record diagnostic state.
- Once a stream tool index is marked invalid:
  - suppress all subsequent deltas for that tool index;
  - do not stream its argument fragments;
  - keep the invalid tool name and raw fragments in diagnostics if available;
  - do not let that invalid index contribute to `finish_reason="tool_calls"`.
- Set `finish_reason="tool_calls"` only if at least one valid tool call was emitted.
- Preserve content streaming behavior.
- Preserve `[DONE]`.
- Preserve `finish_reason` logic for valid tool calls.

Unit tests:

- Streaming valid tool name emits tool delta.
- Streaming unknown tool name does not emit a valid external tool delta.
- Arguments fragments for valid tools still reconstruct.
- Parallel valid tools preserve index order.
- Unknown first tool followed by valid second tool has deterministic behavior.

Integration tests:

- Server-mock stream valid single tool.
- Server-mock stream unknown tool name.
- Server-mock stream parallel tools with one invalid name.
- Existing stream parser golden tests still pass.

Acceptance:

- Stream and non-stream invalid-name behavior are consistent.
- No fuzzy correction.
- `[DONE]` and finish reason remain correct for valid calls.

## Milestone 4: Compatibility Mode Diagnostics

Goal: add SGLang-like compatibility mode as diagnostics and controlled recovery, not silent correction.

Implementation:

- Add opt-in environment variable, for example:
  - `FT_TOOLCALL_COMPAT_MODE=ON`
- Add structured diagnostics:
  - `invalid_tool_name`
  - `malformed_arguments_json`
  - `missing_required_argument`
  - `tool_choice_violation`
  - `parallel_tool_calls_violation`
  - `malformed_tool_block`
- Add closest-match diagnostics for reporting only:
  - actual name;
  - allowed names;
  - closest allowed name;
  - ratio.
- Do not use closest match to pass validation.

Optional compat behavior:

- Forward unknown tool calls only if explicitly enabled by a separate variable:
  - `FT_TOOLCALL_FORWARD_UNKNOWN_TOOLS=ON`
- Default remains strict.

Unit tests:

- Compat mode records diagnostics.
- Closest match is reported but not used for passing.
- Unknown forwarding is off by default.
- Unknown forwarding on preserves raw unknown name.

Integration tests:

- Server-mock invalid-name diagnostics.
- Live runner report-only error summary.
- Deterministic tests remain strict by default.

Acceptance:

- Developers can understand model/toolcall failures without changing parser correctness.

## Milestone 5: Tool Choice And Parallel Validation

Goal: validate protocol semantics after parsing.

Implementation:

- `tool_choice="required"`:
  - if no valid tool call is produced, record violation.
- Named tool choice:
  - only the named function is valid.
- `parallel_tool_calls=false`:
  - more than one valid tool call is a violation.
- Keep this as response validation first, not generation constraint.

Unit tests:

- Required tool choice with no call.
- Required tool choice with valid call.
- Named function matched.
- Named function mismatched.
- Parallel disabled with one call.
- Parallel disabled with two calls.

Integration tests:

- Server-mock cases for:
  - required no-call;
  - named mismatch;
  - parallel disabled mismatch.
- Existing protocol tests still parse request fields.

Acceptance:

- FastLLM reports tool choice violations deterministically.
- No decoding changes required.

## Milestone 6: Basic Strict Schema Validation

Goal: validate arguments against tool schema enough to catch common failures without adding a large dependency.

Implementation:

- For each valid tool call:
  - parse `function.arguments` as JSON object.
  - if tool schema has `required`, ensure required keys exist.
  - optionally validate primitive top-level types:
    - string;
    - number/integer;
    - boolean;
    - array;
    - object.
- Do not implement full JSON Schema in this milestone.
- Record diagnostics; default response policy should be explicit and tested.

Unit tests:

- Required key present.
- Required key missing.
- Wrong primitive type.
- Arguments not JSON.
- Arguments JSON but not object.

Integration tests:

- Server-mock strict schema missing required field.
- Valid strict schema case.

Acceptance:

- Common schema errors are caught deterministically.
- No new dependency is required.

## Milestone 7: DeepSeek V4 Constraint Descriptor Dry Run

Goal: model SGLang's structural/json constraint selection without wiring it into decoding yet. This is long-term design/prototype work, not a near-term production commitment.

Implementation:

- Add a constraint descriptor:
  - `constraint_type`
  - `tool_names`
  - `tool_choice`
  - `parallel_tool_calls`
  - `schemas`
  - `model_type`
- For DeepSeek V4:
  - descriptor should identify `deepseek_v4` structural format.
  - function names should be represented as enums.
- Do not invoke xgrammar or constrained decoding yet.

Unit tests:

- Auto without strict returns no constraint or dry-run descriptor only.
- Required returns descriptor with at-least-one semantics.
- Named function returns descriptor restricted to that function.
- Strict schema includes parameter schema.
- Parallel flag included.

Integration tests:

- Server-mock dry-run does not change output.
- Debug endpoint/helper can print descriptor in tests.

Acceptance:

- Clear future interface for generation-time constraints.
- No runtime behavior change unless explicitly enabled.

## Milestone 8: Generation-Time Constraint Hook

Goal: add the server-side hook needed to pass a future constraint into model generation. This remains prototype work until the FastLLM decoding path can safely consume constraints.

Implementation:

- Extend generation config internally with optional toolcall constraint.
- Do not require every backend to support it.
- If backend does not support constraint:
  - log/debug diagnostics;
  - continue current behavior unless strict mode requires error.
- Keep interface backend-agnostic.

Unit tests:

- Request with required tool builds constraint.
- Unsupported backend path does not crash.
- Constraint is absent when no tools.

Integration tests:

- Mock model records received constraint.
- No real model/GPU test.

Acceptance:

- FastLLM has a clean place to carry SGLang/vLLM-style constraints.

## Milestone 9: Real Constraint Backend Exploration

Goal: evaluate actual constrained decoding implementation options. This is explicitly exploratory and should not block the earlier validation/diagnostics milestones.

Options:

- xgrammar structural tags;
- JSON schema constrained sampling;
- DeepSeek V4-specific structural tag;
- minimal name-only grammar for function names.

Work items:

- Compare with SGLang `get_structural_tag`.
- Compare with vLLM `structured_outputs`.
- Decide whether FastLLM should vendor, depend on, or implement a small subset.

Unit tests:

- Pure grammar/descriptor tests.
- No GPU.

Integration tests:

- Mock decoder constraint acceptance.
- Optional manual live constrained smoke.

Acceptance:

- Decision document plus prototype.
- No hidden dependency on live server.

## Milestone 10: Manual Live Regression Matrix

Goal: use live tests only to measure real model quality and constraint impact.

Live cases:

- Baseline `get_weather`.
- Short name `weather`.
- Unknown typo detection.
- Required tool choice.
- Named tool choice.
- Parallel tools.
- Tool result roundtrip.
- Strict schema missing required argument.

Rules:

- Manual only.
- Not in unittest discovery.
- Not in default `--all`.
- Report-only mode for quality experiments.

Acceptance:

- Live results can distinguish:
  - parser/server bug;
  - protocol bug;
  - model adherence failure;
  - constraint effectiveness.

## Milestone 11: DeepSeek V4 Name-Only Constraint Spike

Goal: make the first generation-time constraint target explicit and testable:
the model must only emit request-approved function names inside DeepSeek V4
DSML `invoke name="..."` attributes.

Scope:

- Build a backend-neutral name enum constraint from the existing descriptor.
- Keep parser-side strict validation as the correctness fallback.
- Do not fuzzy-correct unknown model output.
- Do not implement full DSML structural decoding or argument JSON schema
  decoding yet.
- Do not require GPU or live server in deterministic tests.

Implementation:

- Extend `ToolCallConstraintSpec` with a `name_constraint` payload:
  - format: `deepseek_v4_dsml`;
  - trigger prefixes for standard and alternate DSML invoke tags;
  - allowed function names from `request.tools` or named `tool_choice`;
  - tokenizer-agnostic matching mode.
- Add a focused name-only adapter path:
  - prefer `set_tool_name_constraint(payload)` if a backend exposes it;
  - fallback to `set_tool_call_constraint(spec)` for backends that consume the
    full spec;
  - otherwise report that no native backend support exists.
- In `llm.py`, attempt to apply `tool_call_constraint` before launch if the
  underlying native object exposes a future constraint API. If no API exists,
  log and continue.
- Treat `tool_call_constraint=None` as an explicit clear signal for future
  model-level native backends so constraints cannot leak across requests.
- Keep all live tests manual.

Unit tests:

- DeepSeek V4 spec includes name-only constraint with allowed names.
- Named `tool_choice` reduces allowed names to the named function.
- Empty tools produce no constraint.
- Decoder exposing `set_tool_name_constraint` receives only the name payload.
- Decoder exposing only `set_tool_call_constraint` still receives the full spec.
- Unsupported backend reports not applied and does not crash.

Integration tests:

- Mock model records that `tool_call_constraint` reaches `launch_stream_response`.
- Mock decoder accepts the name-only payload.
- Existing parser/server deterministic tests still pass.

Manual live acceptance for a future native backend:

- `live_openai_baseline_get_weather --repeat 20`: no invalid tool names.
- `live_openai_named_weather_function --repeat 20`: no invalid tool names.
- `live_openai_required_weather_stream --repeat 20`: no invalid tool names.
- `live_openai_parallel_weather_time --repeat 20`: no `get_w eather` or
  similar variants.

## Milestone 12: Native DeepSeek V4 Name Constraint Backend

Goal: consume the M11 name-only constraint in native FastLLM generation and
mask sampling at the DeepSeek V4 DSML `invoke name="..."` position.

Scope:

- Native backend only constrains function names.
- Allowed names come from `request.tools` or named `tool_choice` through the
  existing M11 `name_constraint` payload.
- No parser fuzzy correction.
- No unknown-name auto repair.
- No full DSML grammar, JSON argument grammar, strict schema, or
  `parallel_tool_calls=false` enforcement.
- Live tests remain manual.

Implementation:

- Export `set_tool_call_constraint_llm_model(model, json_payload)` from
  `tools/src/pytools.cpp`.
- Store the payload in thread-local pending request state and consume it into
  the next `GenerationConfig` at `launch_response*`.
- Treat JSON `null` as an explicit clear signal.
- Extend `GenerationConfig` with:
  - enabled flag;
  - allowed function names;
  - DeepSeek V4 invoke-name trigger prefixes;
  - name terminator;
  - per-step allowed token ids.
- Track generated output text per `ResponseContext`, excluding prompt text.
- Before each decode/prefill sampling step, detect the latest unclosed
  DeepSeek V4 invoke-name prefix in generated text and compute allowed next
  token ids by tokenizer-decoded string prefix matching.
- Apply the mask in common `LLMSampling` / `LLMSamplingOnly`; when the mask is
  active, sample from allowed ids rather than filtering an already selected
  global top-k.
- Keep `LLMSamplingBlock` on the CPU sampling path when a name mask is active,
  so CUDA top-k/top-p does not bypass the mask.
- Wire the prepare/update hooks into both generic `basellm` schedulers and the
  DeepSeek V4 model-specific scheduler.

Unit tests:

- Existing M11 Python hook tests verify the native setter receives JSON and
  receives JSON `null` on clear.
- Existing constraint compiler tests verify allowed names and named
  `tool_choice` reduction.
- C++ compile is required to catch native API and scheduler integration.

Integration tests:

- Deterministic OpenAI parser/server/toolcall suites remain green.
- Native build/compile checks succeed.
- Manual live acceptance after rebuilding/restarting the server:
  - `live_openai_baseline_get_weather --repeat 20`;
  - `live_openai_named_weather_function --repeat 20`;
  - `live_openai_required_weather_stream --repeat 20`;
  - `live_openai_parallel_weather_time --repeat 20`.

Acceptance:

- No generated DeepSeek V4 DSML invoke name outside request-approved names in
  live strict-path cases.
- Existing short-name/alias live diagnostics still work, but are mitigation
  paths rather than required for correctness.

## Recommended PR Split

1. Parser facade and validation layer, no server behavior change.
2. Non-stream OpenAI integration.
3. Streaming OpenAI integration.
4. Compatibility diagnostics.
5. Tool choice and parallel validation.
6. Basic strict schema validation.
7. Constraint descriptor dry run.
8. Generation-time constraint hook.
9. Optional real constraint backend prototype.

## Immediate Next Step

The next implementation step should be:

1. Revert or set aside alias-first M2 work if it conflicts with this plan.
2. Implement Milestone 1:
   - `FunctionCallParser` facade;
   - request tool index;
   - valid/invalid tool name diagnostics;
   - non-stream and stream deterministic tests.

Do not implement fuzzy correction. Do not implement generation-time constraints in Milestone 1.

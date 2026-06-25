# Toolcall Constraint Backend Decision

This is the Milestone 9 decision note for the SGLang-style toolcall work.

## References

- SGLang routes tool calls through a `FunctionCallParser` and lets model-specific detectors expose structural-tag support.
- SGLang can hand a structural tag to xgrammar when the detector supports it.
- vLLM carries constrained decoding through `structured_outputs`, including JSON schema and structural tag payloads.

## Decision

FastLLM should not vendor xgrammar or copy vLLM/SGLang internals in this milestone.

Instead, FastLLM now builds an internal prototype constraint spec:

- `descriptor`: the existing request-aware toolcall descriptor;
- `structural_tag`: a DeepSeek V4 DSML structural-tag-shaped payload;
- `name_constraint`: a focused DeepSeek V4 `invoke name` enum payload for the
  first name-only backend spike;
- `name_grammar`: a minimal EBNF-style name grammar with allowed tool names as enum literals;
- `json_schemas`: strict tool schemas copied from request tools.

The generation hook can pass this spec to a future backend. `llm.py` now attempts
to call a future native C API named `set_tool_call_constraint_llm_model(model,
json_payload)` before launch. Current native FastLLM builds do not expose that
API yet, so the hook logs and continues unless a mock or future decoder accepts
it. For no-tools requests the hook sends JSON `null`; future model-level native
backends must treat that as "clear any previous tool-call constraint" so one
request's tool-name enum cannot leak into the next request.

## Why

- No new dependency is required.
- Unit tests can validate the payload deterministically.
- The shape is close enough to SGLang/vLLM to evaluate future backend integration.
- Parser fuzzy correction remains intentionally out of scope.

## Current Limitations

- The prototype does not enforce constraints in current native decoding.
- The name-only payload is ready for a native backend, but it is not a parser
  fuzzy-correction mechanism.
- The name grammar is a transport artifact, not an xgrammar-validated grammar.
- Strict schemas are carried for future use but are not enforced during decoding.
- Live model-quality tests remain manual and outside deterministic CI.

## Next Backend Options

1. Add a small native name-enum constraint in FastLLM decoding.
2. Add an optional xgrammar bridge behind a feature flag.
3. Add a backend adapter that consumes `structured_outputs`-style payloads.
4. Keep deterministic parser-side validation as the correctness fallback.

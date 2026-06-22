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
- `name_grammar`: a minimal EBNF-style name grammar with allowed tool names as enum literals;
- `json_schemas`: strict tool schemas copied from request tools.

The generation hook can pass this spec to a future backend. Native FastLLM decoding currently ignores it unless a mock or future decoder explicitly accepts it.

## Why

- No new dependency is required.
- Unit tests can validate the payload deterministically.
- The shape is close enough to SGLang/vLLM to evaluate future backend integration.
- Parser fuzzy correction remains intentionally out of scope.

## Current Limitations

- The prototype does not enforce constraints in native decoding.
- The name grammar is a transport artifact, not an xgrammar-validated grammar.
- Strict schemas are carried for future use but are not enforced during decoding.
- Live model-quality tests remain manual and outside deterministic CI.

## Next Backend Options

1. Add a small native name-enum constraint in FastLLM decoding.
2. Add an optional xgrammar bridge behind a feature flag.
3. Add a backend adapter that consumes `structured_outputs`-style payloads.
4. Keep deterministic parser-side validation as the correctness fallback.

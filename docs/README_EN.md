# Documentation index

[中文](README.md)

- [Quick start](../README_EN.md)
- [Model deployment guides](../README_EN.md#model-deployment-guides)
- [`--auto` tuning and Launcher controls](auto_en.md): common options, effects, examples, and startup logs
- [Launcher and Studio](launcher_en.md): profiles, chat, and agents
- [Manual deployment and option reference](deployment_en.md): device layouts, formats, download/export, and source builds
- [DeepSeek-V4.1 deployment](deepseek_v41_en.md)
- [Benchmarks](benchmark_en.md): measured results and their scope by model
- [Changelog](version_en.md)

Further technical guides are currently in Chinese:

- [Planner rules and model defaults](hybrid-auto.md)
- [Development and validation](development-validation.md)
- [Agent deployment reference](model-deployment-agent.md)
- [Execution modes and module boundaries](execution_modes.md)
- [Hybrid inference](mixforward.md) and [CUDA expert caching](cuda-expert-cache.md)
- [Context extension](context-length-extension.md) and [KVMem](kvmem.md)
- [Harness integration](launcher-harness.md), [native agents](launcher-agents.md), and [interface customization](launcher-plugins.md)

## Documentation conventions

Deployment guides describe current options, support boundaries, fallbacks, and verification entry points. Performance measurements belong in the corresponding model benchmark page.

Keep hardware, weight format, tested revision when known, input/output lengths, concurrency, cache/MTP settings, timing definitions, and limitations with each result. Measurements without a known source revision must not be presented as current-HEAD performance.

Replace obsolete conclusions when results change instead of appending optimization diaries. Keep raw logs, profiler traces, full generated outputs, temporary debugging commands, and cross-framework experiment archives outside the repository. Tests retain runnable regressions, required fixtures, and instructions. Preserve independent numerical references, third-party licenses, and source attributions; use changelogs for release changes.

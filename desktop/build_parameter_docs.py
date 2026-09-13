#!/usr/bin/env python3
"""Build the offline reference with the bundled Python and installed CLI parser."""

import argparse
from html import escape
from pathlib import Path
import re

from ftllm import __version__
from ftllm.cli import args_parser
from ftllm.util import make_normal_parser


# Extra explanations accompany the CLI's own help, defaults, aliases and types.
# A note is (explanation, example value); examples are values, not shell commands.
MODEL_NOTES = {
    "model": ("放在子命令后。离线使用本地 HF 模型目录、GGUF 或 FLM 文件；远程仓库 ID 需要联网。也可读取 FastLLM 配置文件。", '"/path/to/model"'),
    "path": ("与位置参数 model 二选一；两者同时提供时，非空 --path 优先。含空格的路径要加引号。", '"/path/模型目录"'),
    "threads": ("CPU 计算线程数；-1 由运行时选择。CPU 或 NUMA 承担 MoE 专家计算时，此项对速度尤其重要。", "16"),
    "low": ("开启低内存模式。控制的是主机内存策略，与 --low_gpu_mem 是两个独立开关。", ""),
    "low_gpu_mem": ("不带值。强制关闭 CUDA embedding 与 GPU token handoff，优先于 --cuda_embedding；保留原有 CUDA Graph 策略，不保证所有模型都能避免显存不足。", ""),
    "device": ("常用 cpu、cuda:0、cuda:1；cuda:1 是第 2 张卡。未指定时由模型和环境决定；多卡优先使用 --tp，MoE 专家可另设 --moe_device。", "cuda:0"),
    "tp": ("2 表示前两张卡（0、1），0 表示仅 0 号卡；0,2 明确选择 0、2 号卡，auto 自动选择。支持情况取决于模型的并行后端。", "2"),
    "dtype": ("常用 auto、float16、bfloat16、int8、int4、int4g128、fp8_e4m3。主要控制 HF 权重的加载/量化；auto 通常使用 float16，也会识别源模型量化配置。GGUF 已有的量化格式不由此项任意转换。", "int4g128"),
    "moe_dtype": ("仅覆盖 MoE 专家权重的类型，可与主权重 --dtype 不同；留空沿用运行时策略。较低精度可节省内存，模型支持和精度需自行验证。", "int4g128"),
    "atype": ("中间计算精度，常用 auto、float16、float32；区别于存储权重的 --dtype。具体支持取决于模型和设备。", "float16"),
    "moe_atype": ("MoE 层的激活计算精度：auto、float32、float16、bfloat16。留空时沿用运行时策略。", "bfloat16"),
    "kv_cache_dtype": ("控制普通注意力的 KV 缓存精度，不改变权重精度。auto、float16、bfloat16、fp8_e4m3；fp4 仅适用于已支持的 Qwen3.5 CUDA 路径与 DeepSeek-V4.1（后者为无损存储，890 B/token），并非所有模型通用。", "fp8_e4m3"),
    "cuda_embedding": ("不带值。让支持的模型在 GPU 上执行 embedding，会增加显存需求；同时指定 --low_gpu_mem 时此开关被覆盖。", ""),
    "kv_cache_limit": ("KV 缓存容量，支持带单位的值，例如 5G、512M；auto 让运行时处理。它不是模型权重占用上限，也不是单会话长度。", "5G"),
    "max_batch": ("同时处理的请求数量上限；-1 使用运行时默认策略。提高并发通常需要更多 KV 缓存，单请求延迟可能增加。", "4"),
    "chunked_prefill_size": ("长输入分块预填充的 token 数，首块与后续块相同；-1 不显式覆盖后端设置。减小切片可降低单次预填充峰值，但可能增加调度开销。", "8192"),
    "moe_device": ("只控制 MoE 专家部分，常用 cpu、numa、cuda:0；主干仍由 --device/--tp 控制。numa 适合已支持的多 CPU 混合推理；disk 仅限支持磁盘专家读取的模型路径。", "cpu"),
    "moe_device_layers": ("让最后 N 层专家使用 --moe_device；-1 表示所有 MoE 层。此项与模型的混合放置策略共同决定实际设备分布。", "8"),
    "moe_cuda_cache": ("为支持的混合 MoE 推理预留专家 GPU 缓存，容量可写 3g；0 关闭。此显存需要与权重和 KV 缓存一起预算。", "3g"),
    "image_embedding_cache": ("Qwen3.5 native 图片 embedding 的 CPU 缓存；未传时有效默认上限为 512m，0 关闭。首次处理图片时按需分配，不是固定常驻的 GPU 显存。", "512m"),
    "ngram_device": ("仅影响带 N-gram 表的模型。cpu 保存在内存；disk 从 checkpoint 按行读取，以降低主机内存占用，同时增加磁盘访问。", "disk"),
    "moe_experts": ("每次路由使用的专家数；-1 沿用模型配置。改变此值可能影响模型质量，通常保留默认值。", ""),
    "cache_history": ("带值的开关，使用 true/false；留空由模型策略决定。控制历史对话缓存，不是浏览器会话记录的保存目录。", "true"),
    "cache_fast": ("带值的开关，使用 true/false；启用快速缓存可能额外消耗显存。", "true"),
    "enable_thinking": ("使用 true/false 控制支持该功能的模型的硬思考开关；留空沿用模型设置。它与服务端处理思考标签的 --think 不同。", "false"),
    "cuda_shared_expert": ("使用 true/false 控制共享专家是否在 CUDA 执行；--cuda_se 是同义写法。关闭后仍需考虑模型的其他设备设置。", "false"),
    "enable_amx": ("使用 true/false，开启需要 CPU 和内核环境支持 AMX；不能让不具备 AMX 的机器获得此指令能力。", "true"),
    "max_context_length": ("单会话输入与输出合计的 token 上限；-1 不显式指定。超出模型声明窗口需要有效的 RoPE 扩展，warmup 后缓存容量不足会导致启动失败。", "32768"),
    "rope_scaling": ("RoPE 扩展配置，接受 yarn 或 JSON。仅支持已接入的 HF Qwen2/3/3.5 布局；Qwen3 等模型需要明确 original_max_position_embeddings，不能只增大上下文数字。", '\'{"rope_type":"yarn","factor":2,"original_max_position_embeddings":32768}\''),
    "tokens": ("用于估算分页 KV 缓存总页数的 token 总额度；-1 不显式覆盖。区别于单会话 --max_context_length，也不是一次输出长度。", "65536"),
    "page_size": ("分页缓存每页的 token 数；-1 采用后端默认值，multicuda 默认 16。通常保留默认，修改时需要考虑后端支持。", "16"),
    "prefix_cache": ("带值的开关 true/false，复用相同输入前缀的缓存；留空不覆盖 FASTLLM_PREFIX_CACHE 环境设置。", "true"),
    "prefix_cache_snapshot_interval_pages": ("按多少页保存一次前缀缓存快照；-1 不覆盖对应环境设置。更密集的快照需要额外缓存管理开销。", "16"),
    "prefix_cache_snapshot_max_per_request": ("每个请求最多保留的前缀快照数；-1 不覆盖对应环境设置。", "4"),
    "prefix_cache_snapshot_max_records": ("全局前缀快照记录数上限；-1 不覆盖对应环境设置。", "128"),
    "gpu_mem_ratio": ("运行时预算使用的 GPU 显存比例，例如 0.9 为 90%。这不是进程显存的硬隔离配额；降低它可以给其他显存用途留余量。", "0.85"),
    "cuda_slab": ("CUDA 权重分配块的大小，单位 MB。命令行初始值为 0，部分多卡 MoE 模型仍会自动选择模型专用 slab 大小。大量小权重场景可减少碎片。", "1024"),
    "mtp": ("模型自身必须具备 MTP 支持和权重；0 关闭，当前每轮最多 8 个 draft token。不能与 DFlash 或 DSpark 同时启用。", "3"),
    "dspark": ("使用模型内置 DSpark，正整数指定 draft token 数；0 关闭。当前内置路径针对具备相应权重的 DeepSeek-V4。", "7"),
    "speculative_algorithm": ("mtp、dspark 或 dflash；留空时可根据 draft checkpoint 推断。算法、目标模型与 draft 模型必须匹配，不是任意两个模型都可组合。", "dflash"),
    "speculative_draft_model_path": ("指定匹配的 draft checkpoint。--draft、--draft_model_path、--dspark_model 均为别名；MTP 也可提供 mtp.safetensors。", '"/path/to/draft-model"'),
    "draft_tokens": ("每轮实际 draft token 数；未指定时读取 draft 配置。DFlash 的 block 还包含 1 个 anchor，因此 block size = draft_tokens + 1。", "7"),
    "speculative_num_draft_tokens": ("DFlash 的 block 大小，包含 anchor；默认读取 draft config。必须处于 checkpoint 支持范围，与 --draft_tokens 同时设置时需一致。", "8"),
    "speculative_dspark_block_size": ("DSpark block 大小，默认读取 draft config；当前外置 DSpark 路径要求与 checkpoint 的 block_size 相同。", ""),
    "speculative_dspark_confidence_threshold": ("DSpark 接受前缀的置信度阈值，范围 0 到 1；0 表示固定验证完整 block。", "0.5"),
    "triton": ("不带值。启用已接入的 Triton CUDA 算子路径，需要可用的 Triton 编译环境；常规启动无需传入此开关。", ""),
    "custom": ("自定义模型描述 Python 文件，适合开发者接入模型结构。", '"/path/to/model_definition.py"'),
    "lora": ("LoRA 权重路径，需要与基础模型和加载器兼容。", '"/path/to/lora"'),
    "cache_dir": ("模型下载/缓存文件目录；不是 KV 缓存、桌面应用配置或网页历史目录。", '"/path/to/model-cache"'),
    "dtype_config": ("按权重名等规则指定不同权重类型的 JSON 文件；用于比 --dtype 更细的量化配置。", '"/path/to/dtype.json"'),
    "ori": ("GGUF 主模型需要额外 tokenizer、配置等信息时，提供原始模型目录。", '"/path/to/original-hf-model"'),
    "mmproj": ("与 GGUF 主模型匹配的多模态投影文件；仅在读取 GGUF 主模型时生效。", '"/path/to/mmproj.gguf"'),
    "tool_call_parser": ("工具调用输出解析器；auto 根据模型选择。手动指定必须使用本版本支持且适配模型的解析器名称。", "auto"),
    "chat_template": ("自定义聊天模板文件，影响消息如何组成模型输入；应与模型训练时的格式匹配。", '"/path/to/chat_template.jinja"'),
}

SERVER_NOTES = {
    "model_name": ("API 请求中的 model 名称。留空时由模型路径等信息推导；客户端指定的名称应与服务公布的名称一致。", "my-model"),
    "host": ("模型 API 的监听地址。默认 0.0.0.0 监听所有网卡；只在本机使用时显式设置 127.0.0.1。", "127.0.0.1"),
    "port": ("模型 API 端口，通常访问 http://127.0.0.1:8080/v1；与 Launcher 的 8000 端口相互独立。", "8080"),
    "api_key": ("为模型 API 设置访问密钥，客户端使用相同密钥。留空不设置密钥；它与 Launcher URL 中的控制 token 是两种凭据。", "your-api-key"),
    "temperature": ("服务端默认采样温度；未传时使用模型默认值，请求中可覆盖。较高温度通常让输出更随机。", "0.7"),
    "top_p": ("服务端默认核采样概率阈值；未传时使用模型默认值，请求中可覆盖。", "0.9"),
    "top_k": ("服务端默认候选 token 数；未传时使用模型默认值，请求中可覆盖。", "20"),
    "repeat_penalty": ("默认重复惩罚，1.0 表示不施加额外惩罚；未传时使用模型默认值。", "1.05"),
    "think": ("带值的 true/false。兼容模型未输出起始 <think> 标签的思考内容解析，不等同于 --enable_thinking 模型开关。", "true"),
    "hide_input": ("不带值，减少服务端请求输入日志。", ""),
    "dev_mode": ("不带值，启用开发用的对话列表和主动停止等接口。常规部署通常不需要。", ""),
    "startup_progress": ("off 关闭结构化启动进度；ndjson 向标准错误输出 FTLLM_PROGRESS 事件，供 Launcher 等程序读取。", "ndjson"),
}

LAUNCH_NOTES = {
    "host": ("Launcher 管理页面的监听地址。127.0.0.1 仅本机访问；0.0.0.0 供其他设备访问。不会改变已保存模型服务的监听地址。", "0.0.0.0"),
    "port": ("Launcher 管理页面的端口；不改变模型 API 服务端口。", "8000"),
    "no_browser": ("不带值，只启动管理服务而不自动打开浏览器；适合无桌面服务器。", ""),
    "config": ("保存启动项的 JSON 配置文件；留空使用当前用户的 Launcher 配置。桌面应用会自动指定包内配置路径。", '"/path/to/profiles.json"'),
    "agent_workspace_root": ("目录 Agent 可选择项目的根目录；留空默认使用当前用户主目录。", '"/path/to/projects"'),
    "allow_remote_workspace_agent": ("本版本 Launcher 中默认已经开启，保留此开关用于兼容。关闭目录 Agent 要使用 --disable-workspace-agent。", ""),
    "disable_workspace_agent": ("不带值，关闭本机和远程访问中的目录浏览、新建目录 Agent 及相关任务执行。普通聊天仍可使用。", ""),
}

WEBUI_NOTES = {
    "model": ("可省略；只用于推导 API 模型名，不在此进程加载模型。没有模型提示时通过 /v1/models 自动发现。", ""),
    "path": ("model 的兼容选项写法；同样只用于推导模型名称。", '"/path/to/model"'),
    "host": ("独立聊天页面的监听地址；它与 ftllm launch 是两个不同命令。", "127.0.0.1"),
    "port": ("独立聊天页面端口，默认 1616；不是模型 API 或 Launcher 管理端口。", "1616"),
    "title": ("浏览器页面标题，含空格时加引号。", '"本地模型工作室"'),
    "max_token": ("每次回复的最大输出 token 数；小于等于 0 不设置人工上限，仍受模型上下文与服务容量限制。", "4096"),
    "history_dir": ("SQLite 会话数据库及上传文件目录。默认位于当前用户的 ~/.fastllm/webui。", '"/path/to/conversations"'),
    "max_upload_mb": ("单个文档、图片或视频文件的上传大小上限，单位 MiB。", "128"),
    "web_search_timeout": ("Web Agent 单次联网搜索/读取请求的超时时间，单位秒；不是模型推理超时。", "20"),
    "data_max_rows": ("数据分析每个表最多读取的行数，用于限制表格处理规模。", "100000"),
    "code_max_context_chars": ("代码项目 Agent 一次注入模型的最大源码字符数；字符数不是 token 数。", "60000"),
    "agent_runtime": ("pi 使用包内 Pi；auto 在 Pi 可用时优先使用它；builtin 使用原有内置链路。", "pi"),
    "pi_agent_timeout": ("单个 Pi Agent 任务的总超时，单位秒。", "600"),
    "pi_agent_max_turns": ("单个 Pi Agent 任务最多调用模型的轮数，包含多轮工具使用。", "8"),
    "pi_agent_context_window": ("传给 Pi 的模型上下文窗口，单位 token；应与实际模型服务能力匹配。", "32768"),
    "agent_workspace_root": ("独立 WebUI 中目录 Agent 可选择的项目根目录。", '"/path/to/projects"'),
    "allow_remote_workspace_agent": ("独立 WebUI 默认关闭远程目录 Agent；需要时显式添加。与 Launcher 命令默认开启的行为不同。", ""),
    "disable_workspace_agent": ("不带值，关闭本机及远程的目录 Agent。", ""),
    "api_base": ("连接已启动的 OpenAI 兼容模型服务，地址应包含 /v1；WebUI 自身不加载模型。", "http://127.0.0.1:8080/v1"),
    "api_key": ("WebUI 后端访问模型 API 使用的密钥，应与 server 的 --api_key 相同。", "your-api-key"),
    "api_model": ("发送到模型 API 的 model 名称；需匹配 server 的 --model_name。留空时使用模型提示或服务发现结果。", "my-model"),
    "api_timeout": ("单次模型 API 请求超时，单位秒。", "3600"),
    "api_ready_timeout": ("启动时等待模型 API 就绪的最长时间，单位秒。", "3600"),
}

BENCHMARK_NOTES = {
    "input_tokens": ("构造的输入 token 长度，必须大于 0。", "1024"),
    "output_tokens": ("每个基准请求的最大输出 token 数，必须大于 0。", "256"),
    "batch": ("同时发起的基准请求数，必须大于 0；与通用参数 --max_batch 的服务容量上限不同。", "4"),
    "warmup": ("正式测量前的预热请求数，必须大于等于 0。", "1"),
    "prompt_unit": ("反复拼接以构造指定长度输入的文本单元，不能是空字符串。", '"FastLLM benchmark context block. "'),
    "temperature": ("基准请求的温度；小于等于 0 使用贪心解码。", "0"),
    "top_p": ("基准请求的核采样概率阈值。", "0.9"),
    "top_k": ("基准请求的候选 token 数；默认 1。", "1"),
    "repeat_penalty": ("基准请求的重复惩罚系数。", "1.0"),
}

DOWNLOAD_NOTES = {
    "repo_id": ("Hugging Face 仓库 ID，放在 download 后；下载需要联网。", "Qwen/Qwen3-0.6B"),
    "include": ("只下载匹配的文件，可给多个模式。为避免 shell 提前展开，模式要加引号。", '\'*.json\' \'*.safetensors\''),
    "exclude": ("排除匹配的文件，可给多个模式；注意保留模型运行所需文件。", '\'*.md\''),
    "hf_username": ("需要认证的 Hugging Face 用户名。", "your-username"),
    "hf_token": ("访问受限仓库的令牌；只有需要认证时才设置。", "your-token"),
    "tool": ("aria2c 或 wget，系统需要提供相应下载程序；Launcher 页面也提供模型下载入口。", "wget"),
    "x": ("aria2c 单文件连接数，范围 1–10。", "4"),
    "j": ("并发下载任务数，范围 1–10。", "5"),
    "dataset": ("不带值，将仓库按数据集类型处理。", ""),
    "local_dir": ("下载目标目录；未指定时使用当前目录下的仓库末级名称。", '"/path/to/models/Qwen3-0.6B"'),
    "revision": ("仓库分支、标签或 revision，默认 main。", "main"),
}

GROUPS = {
    "model": "model path custom lora cache_dir dtype_config ori mmproj tool_call_parser chat_template".split(),
    "devices": "device tp threads low low_gpu_mem cuda_embedding enable_amx triton".split(),
    "precision": "dtype atype moe_dtype moe_atype enable_thinking".split(),
    "memory": "kv_cache_dtype kv_cache_limit max_context_length rope_scaling tokens page_size gpu_mem_ratio max_batch chunked_prefill_size cuda_slab image_embedding_cache".split(),
    "moe": "moe_device moe_device_layers moe_cuda_cache moe_experts cuda_shared_expert ngram_device".split(),
    "cache": "cache_history cache_fast prefix_cache prefix_cache_snapshot_interval_pages prefix_cache_snapshot_max_per_request prefix_cache_snapshot_max_records".split(),
    "speculative": "mtp dspark speculative_algorithm speculative_draft_model_path draft_tokens speculative_num_draft_tokens speculative_dspark_block_size speculative_dspark_confidence_threshold".split(),
}


def default_text(action):
    value = action.default
    if value is None:
        return "未指定"
    if value == "":
        return "未指定（空字符串）"
    if value is True:
        return "开启"
    if value is False:
        return "关闭"
    if isinstance(value, list):
        return "无" if not value else ", ".join(map(str, value))
    value = str(value)
    home = str(Path.home())
    if value == home or value.startswith(home + "/"):
        value = "~" + value[len(home):]
    return value


def value_text(action):
    if action.nargs == 0:
        return "开关，不带值"
    if action.choices:
        return " / ".join(map(str, action.choices))
    kind = getattr(action.type, "__name__", "str")
    label = {"int": "整数", "float": "数值", "_positive_int": "正整数", "_memory_size_bytes": "容量（如 512m、3g）"}.get(kind, "文本 / 路径")
    if action.nargs in ("+", "*"):
        label += "，可传多个值"
    return label


def render_table(key, parser, actions, notes):
    rows = []
    for action in actions:
        names = action.option_strings or [action.dest]
        explanation, example = notes.get(action.dest, ("", ""))
        help_text = parser._get_formatter()._expand_help(action) if action.help else ""
        description = explanation or help_text
        # Preserve the parser's short description alongside richer guidance.
        if explanation and help_text and not re.search(r"[\u4e00-\u9fff]", help_text):
            help_text = ""  # Chinese notes replace English-only help.
        lead = f'<p class="meaning">{escape(help_text)}</p>' if explanation and help_text and help_text != explanation else ""
        usage = ""
        if example or action.nargs == 0:
            option = next((n for n in names if n.startswith("--")), names[0])
            usage = f'<div class="example">写法：<code>{escape((option + " " + example).strip())}</code></div>'
        rows.append(
            f'<tr id="{key}-{action.dest}" data-parameter="{escape(action.dest)}">'
            f'<th scope="row">{"<br>".join("<code>" + escape(n) + "</code>" for n in names)}</th>'
            f'<td><code>{escape(default_text(action))}</code><span class="value-type">{escape(value_text(action))}</span></td>'
            f'<td>{lead}<p>{escape(description)}</p>{usage}</td></tr>'
        )
    return '<div class="table-wrap"><table class="parameters"><thead><tr><th scope="col">参数 / 别名</th><th scope="col">命令行默认值 · 取值方式</th><th scope="col">作用与用法</th></tr></thead><tbody>' + "\n".join(rows) + '</tbody></table></div>'


def build(template):
    parser = args_parser()
    shared = make_normal_parser("ftllm")
    actions = {a.dest: a for a in shared._actions if a.dest != "help"}
    subcommands = next(a.choices for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    substitutions = {"FTLLM_VERSION": escape(__version__)}
    known = set()
    count = 0
    for key, destinations in GROUPS.items():
        selected = [actions[d] for d in destinations if d in actions]
        substitutions[key.upper() + "_TABLE"] = render_table(key, shared, selected, MODEL_NOTES)
        known.update(a.dest for a in selected)
        count += len(selected)
    # New CLI flags remain documented even before hand-written guidance exists.
    extra = [a for name, a in actions.items() if name not in known]
    substitutions["EXTRA_TABLE"] = render_table("extra", shared, extra, MODEL_NOTES) if extra else "<p>本版本通用参数均已列在上方分组中。</p>"
    count += len(extra)
    for command, notes in (("launch", LAUNCH_NOTES), ("server", SERVER_NOTES), ("webui", WEBUI_NOTES), ("benchmark", BENCHMARK_NOTES), ("download", DOWNLOAD_NOTES)):
        command_parser = subcommands[command]
        selected = [a for a in command_parser._actions if a.dest != "help" and not (command in ("server", "benchmark") and a.dest in actions)]
        substitutions[command.upper() + "_TABLE"] = render_table(command, command_parser, selected, notes)
        count += len(selected)
    substitutions["PARAMETER_COUNT"] = str(count)
    result = template
    for key, value in substitutions.items():
        result = result.replace("@" + key + "@", value)
    unresolved = re.findall(r"@[A-Z_]+@", result)
    if unresolved:
        raise ValueError("Unresolved documentation placeholders: " + ", ".join(unresolved))
    return result, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document, count = build(args.template.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    args.output.chmod(0o644)
    print(f"Offline parameter reference: ftllm {__version__}, {count} parameter rows.")


if __name__ == "__main__":
    main()

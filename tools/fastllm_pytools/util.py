import argparse
import os
import sys
import subprocess
import glob

def _has_cuda_device() -> bool:
    if os.path.exists("/dev/nvidia0") or os.path.isdir("/proc/driver/nvidia/gpus"):
        return True
    try:
        return subprocess.run(["nvidia-smi", "-L"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=8).returncode == 0
    except Exception:
        return False

def _total_memory_gib() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0

def _uses_cuda_device(device) -> bool:
    if not device:
        return False
    return "cuda" in str(device).lower() or str(device).lower().startswith("cudapp=")

def _uses_multicuda_device(device) -> bool:
    if not device:
        return False
    return "multicuda" in str(device).lower()

def _uses_thread_tp(tp) -> bool:
    if tp is None:
        return False
    spec = str(tp).strip().lower()
    return spec not in ["", "false", "off", "none", "disable"]

def _cuda_device_count() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "-L"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                text=True,
                                timeout=8)
        if result.returncode == 0:
            return len([line for line in result.stdout.splitlines() if line.strip()])
    except Exception:
        pass
    try:
        return len(glob.glob("/dev/nvidia[0-9]*"))
    except Exception:
        return 0

def _first_thread_tp_cuda_device(tp) -> str:
    spec = str(tp or "").strip()
    lower = spec.lower()
    if lower in ["", "auto", "true", "on", "1"]:
        return "cuda:0"
    if lower.isdigit() and int(lower) > 1:
        return "cuda:0"

    first_part = spec.split(",")[0].strip()
    first_lower = first_part.lower()
    if first_lower.startswith("multicuda:") or first_lower.startswith("cuda:"):
        first_part = first_part.split(":", 1)[1].strip()
    elif first_lower in ["multicuda", "cuda"]:
        return "cuda:0"

    device_id = ""
    for ch in first_part:
        if ch.isdigit():
            device_id += ch
        elif device_id:
            break
    return "cuda:" + (device_id if device_id != "" else "0")

def _thread_tp_cuda_device_spec(tp) -> str:
    spec = str(tp or "").strip()
    lower = spec.lower()
    if lower in ["", "false", "off", "none", "disable"]:
        return ""
    if lower in ["auto", "true", "on", "1"]:
        count = _cuda_device_count()
        if count <= 1:
            return "cuda:0"
        return "cuda:" + ",".join(str(i) for i in range(count))
    if lower.isdigit() and int(lower) > 1:
        requested = int(lower)
        count = _cuda_device_count()
        if count > 0:
            requested = min(requested, count)
        return "cuda:" + ",".join(str(i) for i in range(requested))

    if lower.startswith("multicuda:") or lower.startswith("cuda:"):
        spec = spec.split(":", 1)[1].strip()
    elif lower in ["multicuda", "cuda"]:
        return "cuda:0"
    return "cuda:" + spec

def apply_page_size_default(args):
    if (getattr(args, "page_size", -1) <= 0 and
        (_uses_multicuda_device(getattr(args, "device", "")) or
         _uses_multicuda_device(getattr(args, "moe_device", "")))):
        try:
            args.page_size = int(os.environ.get("FASTLLM_MULTICUDA_PAGE_SIZE", "16"))
        except:
            args.page_size = 16
    return args

def _is_moe_architecture(architecture: str, model_type: str = "", text_model_type: str = "") -> bool:
    return (architecture in [
        "DeepseekV3ForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV4ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "MiniMaxM1ForCausalLM",
        "MiniMaxText01ForCausalLM",
        "HunYuanMoEV1ForCausalLM",
        "Ernie4_5_MoeForCausalLM",
        "PanguProMoEForCausalLM",
        "Glm4MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "MiniMaxM2ForCausalLM",
    ] or model_type in ["deepseek_v4", "qwen3_5_moe"] or text_model_type == "qwen3_5_moe_text")

def make_normal_parser(des: str, add_help = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = des, add_help = add_help)
    parser.add_argument('model', nargs='?', help = '模型路径，fastllm模型文件或HF模型文件夹或配置文件')
    parser.add_argument('-p', '--path', type = str, required = False, default = '', help = '模型路径，fastllm模型文件或HF模型文件夹')
    parser.add_argument('-t', '--threads', type = int, default = -1,  help = '线程数量')
    parser.add_argument('-l', '--low', action = 'store_true', help = '是否使用低内存模式')
    parser.add_argument('--dtype', type = str, default = "auto", help = '权重类型（读取HF模型时有效）')
    parser.add_argument('--moe_dtype', type = str, default = "", help = 'MOE层使用的权重类型（读取HF模型时有效）')
    parser.add_argument('--moe_atype', type = str, default = "", help = 'MOE层激活类型，可使用float32、float16或bfloat16')
    parser.add_argument('--atype', type = str, default = "auto", help = '推理类型，可使用float32或float16')
    parser.add_argument('--kv_cache_dtype', type = str, default = "auto", help = 'KV Cache类型，可使用auto、float16、bfloat16或fp8_e4m3')
    parser.add_argument('--cuda_embedding', action = 'store_true', help = '在cuda上进行embedding')
    parser.add_argument('--kv_cache_limit', type = str, default = "auto",  help = 'kv缓存最大使用量')
    parser.add_argument('--max_batch', type = int, default = -1,  help = '每次最多同时推理的询问数量')
    parser.add_argument('--chunked_prefill_size', type = int, default = -1, help = '分块 prefill 的切片大小（首块与后续块相同），如 8192')
    parser.add_argument('--device', type = str, help = '使用的设备')
    parser.add_argument('--tp', type = str, default = "", help = '线程级张量并行设备，如 0,1 或 auto')
    parser.add_argument('--moe_device', type = str, default = "", help = 'moe使用的设备')
    parser.add_argument('--moe_device_layers', type = int, default = -1, help = '后面多少层moe使用moe_device，-1表示全部moe层使用moe_device')
    parser.add_argument('--moe_experts', type = int, default = -1, help = 'moe使用的专家数')
    parser.add_argument("--cache_history", type = str, default = "", help = "缓存历史对话")
    parser.add_argument("--cache_fast", type = str, default = "", help = "是否启用快速缓存（会消耗一定显存）")
    parser.add_argument("--enable_thinking", type = str, default = "", help = "是否开启硬思考开关（需要模型支持）")
    parser.add_argument("--cuda_shared_expert", "--cuda_se", type = str, default = "true", help = "是否使用cuda来执行共享专家")
    parser.add_argument("--enable_amx", "--amx", type = str, default = "false", help = "是否开启amx加速")
    parser.add_argument("--tokens", type = int, default = -1, help = "设置总的token数量（用于计算paged cache的最大页数）")
    parser.add_argument("--page_size", type = int, default = -1, help = "设置paged cache每页的大小（token数），默认multicuda为16，其它设备使用后端默认值")
    parser.add_argument("--gpu_mem_ratio", type = float, default = 0.9, help = "GPU显存使用比例，如0.9表示使用90%%的显存")
    parser.add_argument("--cuda_slab", type = int, default = 0, help = "CUDA模型权重slab大小（MB），0表示关闭")
    
    parser.add_argument('--custom', type = str, default = "", help = '指定描述自定义模型的python文件')
    parser.add_argument('--lora', type = str, default = "", help = '指定lora路径')
    parser.add_argument('--cache_dir', type = str, default = "", help = '指定缓存模型文件的路径')
    parser.add_argument('--dtype_config', type = str, default = "", help = '指定权重类型配置文件')
    parser.add_argument('--ori', type = str, default = "", help = '原始模型权重，读取GGUF文件时可以使用')

    parser.add_argument('--tool_call_parser', type = str, default = "auto", help = '使用的tool_call_parser类型')
    parser.add_argument('--chat_template', type = str, default = "", help = '使用的chat_template文件')

    return parser

def add_server_args(parser):
    parser.add_argument("--model_name", type = str, default = '', help = "部署的模型名称, 调用api时会进行名称核验")
    parser.add_argument("--host", type = str, default="0.0.0.0", help = "API server host")
    parser.add_argument("--port", type = int, default = 8080, help = "API server port")
    parser.add_argument("--api_key", type = str, default = "", help = "API Key")
    parser.add_argument("--temperature", type = float, default = None, help = "覆盖服务端默认 temperature，未指定则使用模型默认值")
    parser.add_argument("--top_p", type = float, default = None, help = "覆盖服务端默认 top_p，未指定则使用模型默认值")
    parser.add_argument("--top_k", type = int, default = None, help = "覆盖服务端默认 top_k，未指定则使用模型默认值")
    parser.add_argument("--repeat_penalty", "--repetition_penalty", dest = "repeat_penalty",
                        type = float, default = None, help = "覆盖服务端默认 repeat_penalty，未指定则使用模型默认值")
    parser.add_argument("--think", type = str, default = "false", help="if <think> lost")
    parser.add_argument("--hide_input", action = 'store_true', help = "不显示请求信息")
    parser.add_argument("--dev_mode", action = 'store_true', help = "开发模式, 启用后能够获取对话列表并主动停止")

def expand_cudapp_device(device_str):
    if not device_str or not device_str.startswith("cudapp="):
        return device_str
    spec = device_str[len("cudapp="):]
    if ':' in spec:
        weights = [int(w) for w in spec.split(':')]
    else:
        n = int(spec)
        weights = [1] * n
    return str({f'cuda:{i}': w for i, w in enumerate(weights)})

def make_normal_llm_model(args):
    if (args.model and args.model != ''):
        if (args.model.endswith(".json") and os.path.exists(args.model)):
            import json
            with open(args.model, "r", encoding = "utf-8") as file:
                args_config = json.load(file)
                for it in args_config.keys():
                    if (it == "FASTLLM_USE_NUMA" or it == "FASTLLM_NUMA_THREADS"):
                        os.environ[it] = str(args_config[it])
                    setattr(args, it, args_config[it])

    user_set_device = bool(args.device and args.device != "")
    user_set_moe_device = bool(args.moe_device and args.moe_device != "")

    usenuma = False
    try:
        from ftllm.env import env
        usenuma = env.use_numas
    except:
        pass
    if (args.path == '' or args.path is None):
        args.path = args.model
    if (args.path == '' or args.path is None):
        print("model can't be empty. (Example: ftllm run MODELNAME)")
        exit(0)
    if not(os.path.exists(args.path)):
        if (hasattr(args, "model_name") and args.model_name == ''):
            args.model_name = args.path
        from ftllm.download import HFDNormalDownloader
        from ftllm.download import find_metadata
        from ftllm.download import search_model
        if (not(os.path.exists(get_fastllm_cache_path(args.path, args.cache_dir))) and not(find_metadata(args.path))):
            print("Can't find model \"" + args.path + "\", try to find similar one.")
            search_result = search_model(args.path)
            if (len(search_result) > 0):
                args.path = search_result[0]["id"]
                print("Replace model to \"" + args.path + "\"")
            else:
                exit(0)
        downloader = HFDNormalDownloader(args.path, local_dir = get_fastllm_cache_path(args.path, args.cache_dir))
        downloader.run()
        args.path = str(downloader.local_dir)
    
    config_path = os.path.join(args.path, "config.json")
    if (not(os.path.exists(config_path)) and args.ori != "" and os.path.exists(os.path.join(args.ori, "config.json"))):
        config_path = os.path.join(args.ori, "config.json")
    is_moe_model = False
    is_thread_tp_moe_model = False
    if (os.path.exists(config_path)):
        try:
            import json
            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file)
            architecture = config["architectures"][0]
            model_type = config.get("model_type", "")
            text_model_type = ""
            if isinstance(config.get("text_config"), dict):
                text_model_type = config["text_config"].get("model_type", "")
            is_moe_model = _is_moe_architecture(architecture, model_type, text_model_type)

            is_step3p5 = (architecture == 'Step3p5ForCausalLM' or
                          model_type == 'step3p5' or
                          text_model_type == 'step3p5')
            is_step3p7 = (architecture == 'Step3p7ForConditionalGeneration' or
                          model_type == 'step3p7')
            if is_step3p5:
                is_thread_tp_moe_model = True
                if (args.cache_history == ""):
                    args.cache_history = "true"
                if (args.moe_device == "" and not(args.device and args.device != "")):
                    total_mem_gib = _total_memory_gib()
                    can_hold_cpu_moe = total_mem_gib >= 220.0
                    if (_has_cuda_device() and can_hold_cpu_moe):
                        args.device = "cuda"
                        args.moe_device = "cpu"
                    else:
                        args.device = "cpu"
                        args.moe_device = "disk"
                if (args.chunked_prefill_size <= 0):
                    args.chunked_prefill_size = 128
                if (args.tokens <= 0 and not is_step3p7 and not _uses_thread_tp(getattr(args, "tp", ""))):
                    args.tokens = 32768

            if (architecture == 'Qwen3ForCausalLM' or architecture == 'Qwen3MoeForCausalLM' or
                architecture == 'DeepseekV4ForCausalLM' or model_type == 'deepseek_v4' or
                architecture == 'Qwen3_5MoeForConditionalGeneration' or
                model_type == 'qwen3_5_moe' or text_model_type == 'qwen3_5_moe_text' or
                architecture == 'Glm4MoeForCausalLM'):
                if (args.enable_thinking == ""):
                    args.enable_thinking = "true"
            if (architecture == 'Qwen3MoeForCausalLM' or model_type == 'qwen3_moe'):
                is_thread_tp_moe_model = True
            if (architecture == 'MiniMaxM2ForCausalLM' or model_type == 'minimax_m2'):
                is_thread_tp_moe_model = True
            if (is_moe_model):
                if (args.cache_history == ""):
                    args.cache_history = "true"
                if ((not(args.device and args.device != ""))):
                    args.device = "cuda"
                    if (not user_set_moe_device):
                        args.moe_device = "cpu"
                        if (usenuma):
                            args.moe_device = "numa"
            if ("quantization_config" in config):
                quantization_config = config["quantization_config"]
                try:
                    if (args.dtype == "auto" and quantization_config['bits'] == 4 and quantization_config['group_size']):
                        args.dtype = "int4g" + str(quantization_config["group_size"])
                except:
                    pass
                try:
                    if (args.dtype == "auto" and quantization_config['quant_method'] == "fp8" and 
                        (quantization_config['fmt'] == "e4m3" or quantization_config['fmt'] == "float8_e4m3fn")):
                        args.dtype = "fp8_e4m3"
                except:
                    pass
                try:
                    if (args.path.lower().find("-fp8") != -1):
                        args.dtype = "fp8_e4m3";
                except:
                    pass
        except:
            pass
    if (_uses_thread_tp(getattr(args, "tp", ""))):
        tp_device = _first_thread_tp_cuda_device(args.tp)
        if (not user_set_device):
            args.device = tp_device
        if (not user_set_moe_device):
            args.moe_device = (_thread_tp_cuda_device_spec(args.tp) or args.device) if is_thread_tp_moe_model else args.device
    if ((args.device and args.device.find("numa") != -1) or args.moe_device.find("numa") != -1 or
        (args.device and args.device.find("tfacc") != -1) or args.moe_device.find("tfacc") != -1):
        os.environ["FASTLLM_ACTIVATE_NUMA"] = "ON"
        if (args.threads == -1):
            try:
                import glob
                numa_nodes = sorted(glob.glob("/sys/devices/system/node/node[0-9]*"))
                numa_count = len(numa_nodes)
                if numa_count > 0:
                    physical_cores_per_numa = set()
                    for entry in os.listdir(numa_nodes[0]):
                        if entry.startswith("cpu") and entry[3:].isdigit():
                            siblings_path = os.path.join(numa_nodes[0], entry, "topology", "thread_siblings_list")
                            if os.path.exists(siblings_path):
                                with open(siblings_path, "r") as f:
                                    physical_cores_per_numa.add(f.read().strip())
                    cpus_per_numa = len(physical_cores_per_numa) if physical_cores_per_numa else 1
                    args.threads = max(1, numa_count * (cpus_per_numa - 4))
                else:
                    args.threads = 4
            except:
                args.threads = 4
    if (args.threads == -1):
        try:
            available_cores = len(os.sched_getaffinity(0))  # 参数 0 表示当前进程
            args.threads = max(1, min(32, available_cores - 2))
        except:
            args.threads = max(1, min(32, os.cpu_count() - 2))
    if ("FT_THREADS" not in os.environ and "FASTLLM_NUMA_THREADS" not in os.environ):
        os.environ["FT_THREADS"] = str(args.threads)
    atype_was_auto = (args.atype == "auto")
    if (args.atype == "auto"):
        if (args.device in ["cpu", "numa", "tfacc"]):
            args.atype = "float32"
    if (args.dtype == "auto"):
        args.dtype = "float16"
    if (args.moe_device == ""):
        args.moe_device = args.device
    tp_arg = getattr(args, "tp", "")
    if (tp_arg != ""):
        os.environ["FASTLLM_TP"] = tp_arg
        if (_uses_thread_tp(tp_arg)):
            if (atype_was_auto):
                args.atype = "float16"
            if (not(args.device and args.device != "")):
                args.device = _first_thread_tp_cuda_device(tp_arg)
    if (args.moe_atype == "" and is_moe_model and args.dtype == "fp8_e4m3"):
        if (_uses_cuda_device(args.moe_device)):
            args.moe_atype = "float16"
        elif (_uses_thread_tp(tp_arg)):
            args.moe_atype = "bfloat16"
    if (args.device and args.device != ""):
        expanded = expand_cudapp_device(args.device)
        if expanded != args.device:
            print(f"[device] cudapp expand: {args.device} => {expanded}")
            args.device = expanded
    if (args.moe_device and args.moe_device != ""):
        args.moe_device = expand_cudapp_device(args.moe_device)
    from ftllm import llm
    llm.set_moe_device_layers(-1)
    if (args.device and args.device != ""):
        try:
            import ast
            device_map = ast.literal_eval(args.device)
            if (isinstance(device_map, list) or isinstance(device_map, dict)):
                llm.set_device_map(device_map)
            else:
                llm.set_device_map(args.device)
        except:
            llm.set_device_map(args.device)
    if (args.moe_device and args.device != ""):
        try:
            import ast
            moe_device_map = ast.literal_eval(args.moe_device)
            if (args.moe_device_layers >= 0):
                front_moe_device = args.device
                if (_uses_thread_tp(tp_arg) and is_thread_tp_moe_model):
                    front_moe_device = _thread_tp_cuda_device_spec(tp_arg) or args.device
                llm.set_device_map(front_moe_device, True)
                if (isinstance(moe_device_map, list) or isinstance(moe_device_map, dict)):
                    llm.set_layered_moe_device_map(moe_device_map)
                else:
                    llm.set_layered_moe_device_map(args.moe_device)
                llm.set_moe_device_layers(args.moe_device_layers)
            elif (isinstance(moe_device_map, list) or isinstance(moe_device_map, dict)):
                llm.set_device_map(moe_device_map, True)
            else:
                llm.set_device_map(args.moe_device, True)
        except:
            if (args.moe_device_layers >= 0):
                front_moe_device = args.device
                if (_uses_thread_tp(tp_arg) and is_thread_tp_moe_model):
                    front_moe_device = _thread_tp_cuda_device_spec(tp_arg) or args.device
                llm.set_device_map(front_moe_device, True)
                llm.set_layered_moe_device_map(args.moe_device)
                llm.set_moe_device_layers(args.moe_device_layers)
            else:
                llm.set_device_map(args.moe_device, True)
    llm.set_cpu_threads(args.threads)
    llm.set_cpu_low_mem(args.low)
    if (args.cuda_embedding):
        llm.set_cuda_embedding(True)
    if (args.cuda_shared_expert.lower() not in ["", "false", "0", "off"]):
        llm.set_cuda_shared_expert(True)
    if (args.enable_amx.lower() not in ["", "false", "0", "off"]):
        llm.set_enable_amx(True)
    if (args.tokens > 0):
        llm.set_max_tokens(args.tokens)
    apply_page_size_default(args)
    if (args.page_size > 0):
        llm.set_page_size(args.page_size)
    if (hasattr(args, 'gpu_mem_ratio')):
        llm.set_gpu_mem_ratio(args.gpu_mem_ratio)
    if (hasattr(args, 'cuda_slab') and hasattr(llm, 'set_cuda_slab')):
        llm.set_cuda_slab(args.cuda_slab)
    graph = None
    if (args.custom != ""):
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_module", args.custom)
        if spec is None:
            raise ImportError(f"Cannot load module at {args.custom}")
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        if (hasattr(custom_module, "__model__")):
            graph = getattr(custom_module, "__model__")
    if (args.dtype_config != "" and os.path.exists(args.dtype_config)):
        with open(args.dtype_config, "r", encoding="utf-8") as file:
            args.dtype_config = file.read()
    if (args.chat_template != "" and os.path.exists(args.chat_template)):
        with open(args.chat_template, "r", encoding="utf-8") as file:
            args.chat_template = file.read()
    model = llm.model(args.path, dtype = args.dtype, kv_cache_dtype = args.kv_cache_dtype,
                        moe_dtype = args.moe_dtype, graph = graph, tokenizer_type = "auto", lora = args.lora, 
                        dtype_config = args.dtype_config, ori_model_path = args.ori, chat_template = args.chat_template, tool_call_parser = args.tool_call_parser)
    if (args.enable_thinking.lower() in ["", "false", "0", "off"]):
        model.enable_thinking = False
    model.set_atype(args.atype)
    if (args.moe_atype != ""):
        model.set_moe_atype(args.moe_atype)
    if (args.cache_history.lower() not in ["", "false", "0", "off"]):
        model.set_save_history(True)
        if (args.cache_fast in ["", "false", "0", "off"]):
            llm.set_cpu_historycache(True)
    if (args.moe_experts > 0):
        model.set_moe_experts(args.moe_experts)
    if (args.max_batch > 0):
        model.set_max_batch(args.max_batch)
    if (args.kv_cache_limit != "" and args.kv_cache_limit != "auto"):
        model.set_kv_cache_limit(args.kv_cache_limit)
    if (args.chunked_prefill_size > 0):
        model.set_chunked_prefill_size(args.chunked_prefill_size)
    model.warmup()
    return model

def make_download_parser(add_help = True):
    parser = argparse.ArgumentParser(
            description="Downloads a model or dataset from Hugging Face",
            usage="ftllm download [REPO_ID] [OPTIONS]",
            add_help = add_help
    )
        
    # 位置参数
    parser.add_argument("repo_id", nargs="?", help="Hugging Face repo ID")
    # 选项参数
    parser.add_argument("--include", nargs="+", default=[], help="Include patterns")
    parser.add_argument("--exclude", nargs="+", default=[], help="Exclude patterns")
    parser.add_argument("--hf_username", help="HF username")
    parser.add_argument("--hf_token", help="HF access token")
    parser.add_argument("--tool", choices=["aria2c", "wget"], default="aria2c", help="Download tool")
    parser.add_argument("-x", type=int, default=4, help="Threads for aria2c")
    parser.add_argument("-j", type=int, default=5, help="Concurrent downloads")
    parser.add_argument("--dataset", action="store_true", help="Download dataset")
    parser.add_argument("--local-dir", help="Local directory path")
    parser.add_argument("--revision", default="main", help="Revision to download")
    #parser.add_argument("-h", "--help", action="store_true", help="Show help")
        
    return parser

def get_fastllm_cache_path(model_name: str, cache_path = ""):
    system = sys.platform

    if cache_path == "":
        if system == "win32":
            # Windows: %LOCALAPPDATA%\Temp 或 C:\Users\<user>\AppData\Local\Temp
            cache_path = os.getenv('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local')) + '\\Temp'
        elif system == "darwin":
            # macOS: ~/Library/Caches
            cache_path = os.path.expanduser('~/Library/Caches')
        else:
            # Linux 和其他 Unix-like 系统: ~/.cache 或 $XDG_CACHE_HOME
            cache_path = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        cache_path = os.path.join(cache_path, "fastllm")

        cache_dir = os.getenv("FASTLLM_CACHEDIR")
        if (cache_dir and os.path.isdir(cache_dir)):
            cache_path = cache_dir

    cache_path = os.path.join(cache_path, model_name)
    return cache_path

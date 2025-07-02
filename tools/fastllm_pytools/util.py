import argparse
import os
import sys

def make_normal_parser(des: str, add_help = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = des, add_help = add_help)
    parser.add_argument('model', nargs='?', help = '模型路径，fastllm模型文件或HF模型文件夹或配置文件')
    parser.add_argument('-p', '--path', type = str, required = False, default = '', help = '模型路径，fastllm模型文件或HF模型文件夹')
    parser.add_argument('-t', '--threads', type = int, default = -1,  help = '线程数量')
    parser.add_argument('-l', '--low', action = 'store_true', help = '是否使用低内存模式')
    parser.add_argument('--dtype', type = str, default = "auto", help = '权重类型（读取HF模型时有效）')
    parser.add_argument('--moe_dtype', type = str, default = "", help = 'MOE层使用的权重类型（读取HF模型时有效）')
    parser.add_argument('--atype', type = str, default = "auto", help = '推理类型，可使用float32或float16')
    parser.add_argument('--cuda_embedding', action = 'store_true', help = '在cuda上进行embedding')
    parser.add_argument('--kv_cache_limit', type = str, default = "auto",  help = 'kv缓存最大使用量')
    parser.add_argument('--max_batch', type = int, default = -1,  help = '每次最多同时推理的询问数量')
    parser.add_argument('--device', type = str, help = '使用的设备')
    parser.add_argument('--moe_device', type = str, default = "", help = 'moe使用的设备')
    parser.add_argument('--moe_experts', type = int, default = -1, help = 'moe使用的专家数')
    parser.add_argument("--cache_history", type = str, default = "", help = "缓存历史对话")
    parser.add_argument("--cache_fast", type = str, default = "", help = "是否启用快速缓存（会消耗一定显存）")
    parser.add_argument("--enable_thinking", type = str, default = "", help = "是否开启硬思考开关（需要模型支持）")
    parser.add_argument("--cuda_shared_expert", "--cuda_se", type = str, default = "true", help = "是否使用cuda来执行共享专家")
    
    parser.add_argument('--custom', type = str, default = "", help = '指定描述自定义模型的python文件')
    parser.add_argument('--lora', type = str, default = "", help = '指定lora路径')
    parser.add_argument('--cache_dir', type = str, default = "", help = '指定缓存模型文件的路径')
    parser.add_argument('--dtype_config', type = str, default = "", help = '指定权重类型配置文件')
    return parser

def add_server_args(parser):
    parser.add_argument("--model_name", type = str, default = '', help = "部署的模型名称, 调用api时会进行名称核验")
    parser.add_argument("--host", type = str, default="0.0.0.0", help = "API server host")
    parser.add_argument("--port", type = int, default = 8080, help = "API server port")
    parser.add_argument("--api_key", type = str, default = "", help = "API Key")
    parser.add_argument("--think", type = str, default = "false", help="if <think> lost")
    parser.add_argument("--hide_input", action = 'store_true', help = "不显示请求信息")
    parser.add_argument("--dev_mode", action = 'store_true', help = "开发模式, 启用后能够获取对话列表并主动停止")

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
                
    usenuma = False
    try:
        env_FASTLLM_USE_NUMA = os.getenv("FASTLLM_USE_NUMA")
        if (env_FASTLLM_USE_NUMA and env_FASTLLM_USE_NUMA != '' and env_FASTLLM_USE_NUMA != "OFF" and env_FASTLLM_USE_NUMA != "0"):
            usenuma = True
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
    if (os.path.exists(os.path.join(args.path, "config.json"))):
        try:
            import json
            with open(os.path.join(args.path, "config.json"), "r", encoding="utf-8") as file:
                config = json.load(file)
            if (config["architectures"][0] == 'Qwen3ForCausalLM' or config["architectures"][0] == 'Qwen3MoeForCausalLM'):
                if (args.enable_thinking == ""):
                    args.enable_thinking = "true"
            if (config["architectures"][0] == 'DeepseekV3ForCausalLM' or 
                config["architectures"][0] == 'DeepseekV2ForCausalLM' or 
                config["architectures"][0] == 'Qwen3MoeForCausalLM' or 
                config["architectures"][0] == 'MiniMaxM1ForCausalLM' or 
                config["architectures"][0] == 'MiniMaxText01ForCausalLM' or 
                config["architectures"][0] == 'HunYuanMoEV1ForCausalLM' or 
                config["architectures"][0] == 'Ernie4_5_MoeForCausalLM' or 
                config["architectures"][0] == 'PanguProMoEForCausalLM'):
                if (args.cache_history == ""):
                    args.cache_history = "true"
                if ((not(args.device and args.device != ""))):
                    args.device = "cuda"
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
                    if (args.dtype == "auto" and quantization_config['quant_method'] == "fp8" and quantization_config['fmt'] == "e4m3"):
                        args.dtype = "fp8_e4m3"
                except:
                    pass
        except:
            pass
    if ((args.device and args.device.find("numa") != -1) or args.moe_device.find("numa") != -1 or
        (args.device and args.device.find("tfacc") != -1) or args.moe_device.find("tfacc") != -1):
        os.environ["FASTLLM_ACTIVATE_NUMA"] = "ON"
        if (args.threads == -1):
            args.threads = 4
    if (args.threads == -1):
        try:
            available_cores = len(os.sched_getaffinity(0))  # 参数 0 表示当前进程
            args.threads = max(1, min(32, available_cores - 2))
        except:
            args.threads = max(1, min(32, os.cpu_count() - 2))
    if (args.atype == "auto"):
        if (args.device in ["cpu", "numa", "tfacc"]):
            args.atype = "float32"
    if (args.dtype == "auto"):
        args.dtype = "float16"
    if (args.moe_device == ""):
        args.moe_device = args.device
    from ftllm import llm
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
            if (isinstance(moe_device_map, list) or isinstance(moe_device_map, dict)):
                llm.set_device_map(moe_device_map, True)
            else:
                llm.set_device_map(args.moe_device, True)
        except:
            llm.set_device_map(args.moe_device, True)
    llm.set_cpu_threads(args.threads)
    llm.set_cpu_low_mem(args.low)
    if (args.cuda_embedding):
        llm.set_cuda_embedding(True)
    if (args.cuda_shared_expert.lower() not in ["", "false", "0", "off"]):
        llm.set_cuda_shared_expert(True)
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
    model = llm.model(args.path, dtype = args.dtype, moe_dtype = args.moe_dtype, graph = graph, tokenizer_type = "auto", lora = args.lora, 
                        dtype_config = args.dtype_config)
    if (args.enable_thinking.lower() in ["", "false", "0", "off"]):
        model.enable_thinking = False
    model.set_atype(args.atype)
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
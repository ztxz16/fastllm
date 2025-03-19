import argparse
from ftllm import llm

def make_normal_parser(des: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = des)
    parser.add_argument('-p', '--path', type = str, required = True, default = '', help = '模型路径，fastllm模型文件或HF模型文件夹')
    parser.add_argument('-t', '--threads', type = int, default = 4,  help = '线程数量')
    parser.add_argument('-l', '--low', action = 'store_true', help = '是否使用低内存模式')
    parser.add_argument('--dtype', type = str, default = "float16", help = '权重类型（读取HF模型时有效）')
    parser.add_argument('--atype', type = str, default = "auto", help = '推理类型，可使用float32或float16')
    parser.add_argument('--cuda_embedding', action = 'store_true', help = '在cuda上进行embedding')
    parser.add_argument('--kv_cache_limit', type = str, default = "auto",  help = 'kv缓存最大使用量')
    parser.add_argument('--max_batch', type = int, default = -1,  help = '每次最多同时推理的询问数量')
    parser.add_argument('--device', type = str, help = '使用的设备')
    parser.add_argument('--moe_device', type = str, default = "", help = 'moe使用的设备')
    parser.add_argument('--moe_experts', type = int, default = -1, help = 'moe使用的专家数')
    parser.add_argument('--custom', type = str, default = "", help = '指定描述自定义模型的python文件')
    parser.add_argument('--lora', type = str, default = "", help = '指定lora路径')
    return parser

def make_normal_llm_model(args):
    if (args.moe_device == ""):
        args.moe_device = args.device
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
    model = llm.model(args.path, dtype = args.dtype, graph = graph, tokenizer_type = "auto", lora = args.lora)
    model.set_atype(args.atype)
    if (args.moe_experts > 0):
        model.set_moe_experts(args.moe_experts)
    if (args.max_batch > 0):
        model.set_max_batch(args.max_batch)
    if (args.kv_cache_limit != "" and args.kv_cache_limit != "auto"):
        model.set_kv_cache_limit(args.kv_cache_limit)
    return model
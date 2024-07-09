import argparse
from ftllm import llm

def make_normal_parser(des: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = des)
    parser.add_argument('-p', '--path', type = str, required = True, default = '', help = '模型路径，fastllm模型文件或HF模型文件夹')
    parser.add_argument('-t', '--threads', type = int, default = 4,  help = '线程数量')
    parser.add_argument('-l', '--low', action = 'store_true', help = '是否使用低内存模式')
    parser.add_argument('--dtype', type = str, default = "float16", help = '权重类型（读取HF模型时有效）')
    parser.add_argument('--atype', type = str, default = "float32", help = '推理类型，可使用float32或float16')
    parser.add_argument('--cuda_embedding', action = 'store_true', help = '在cuda上进行embedding')
    parser.add_argument('--device', type = str, help = '使用的设备')
    return parser

def make_normal_llm_model(args):
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
    llm.set_cpu_threads(args.threads)
    llm.set_cpu_low_mem(args.low)
    if (args.cuda_embedding):
        llm.set_cuda_embedding(True)
    model = llm.model(args.path, dtype = args.dtype, tokenizer_type = "auto")
    model.set_atype(args.atype)
    return model
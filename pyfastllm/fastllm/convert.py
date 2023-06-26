import sys
import logging
import sys
import struct
import numpy as np
import argparse
from .utils import torch2flm 

try:
    import torch
    from transformers import AutoTokenizer, AutoModel # chatglm
    from transformers import LlamaTokenizer, LlamaForCausalLM # alpaca
    from transformers import AutoModelForCausalLM, AutoTokenizer  # baichuan, moss
    from peft import PeftModel
except Exception as e:
    logging.error("Make sure that you installed transformers and peft!!!")
    sys.exit(1)

MODEL_DICT = {
    "alpaca":{
        "tokenizer": "minlik/chinese-alpaca-33b-merged",
        "model": "minlik/chinese-alpaca-33b-merged"
    },
    "baichuan7B":{
        "model": "baichuan-inc/baichuan-7B",
        "tokenizer": "baichuan-inc/baichuan-7B",
        "peft": "hiyouga/baichuan-7b-sft",
    },
    "chatglm6B":{
        "tokenizer": "THUDM/chatglm-6b",
        "model": "THUDM/chatglm-6b"
    },
    "moss":{
        "model": "fnlp/moss-moon-003-sft",
        "tokenizer": "fnlp/moss-moon-003-sft",
    }
}

def parse_args():
    # -p 模型路径或hf路径
    # -o --out_path 导出路径
    # -q 量化位数
    parser = argparse.ArgumentParser(description='build fastllm libs')
    parser.add_argument('-o', dest='export_path', default=None,
                    help='output export path')
    parser.add_argument('-p', dest='model_path', type=list,
                    help='a list with "tokenizer", "model", "peft model", such as: -p THUDM/chatglm-6b THUDM/chatglm-6b')
    parser.add_argument('-m', dest='model', default='',
                    help='model name with(alpaca, baichuan7B, chatglm6B, moss)')
    args = parser.parse_args()
    return args


def convert(model, tokenizer, export_path):
    torch2flm.tofile(model, tokenizer, export_path)


def alpaca(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path).float()
    return model, tokenizer

def baichuan7B(model_path, peft_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, peft_path).float()
    layers = model.model.model.layers
    for i in range(len(layers)):
        layers[i].self_attn.W_pack.weight += torch.mm(layers[i].self_attn.W_pack.lora_B.default.weight, layers[i].self_attn.W_pack.lora_A.default.weight) * layers[i].self_attn.W_pack.scaling["default"]
    
    return model, tokenizer

def chatglm6B(model_path, ):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()
    model = model.eval()
    return model, tokenizer

def moss(model_path, ):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).float()
    model = model.eval()
    return model, tokenizer

def main(args):
    if args.model not in MODEL_DICT:
        assert f"Not Support {args.model} Yet!!!"
    

    model_args = {}
    model_args["model_path"] = MODEL_DICT[args.model].get("model")
    if MODEL_DICT[args.model].has_key("peft"):
        model_args["peft_path"] = MODEL_DICT[args.model].get("peft")
    
    if args.model_path:
        model_args["model_path"] = args.model_path[0]
        if len(args.model_path) > 2:
            model_args["peft_path"] = args.model_path[2]
    
    model, tokenizer = globals().get(args.model)(**model_args)
    export_path = args.export_path or f"{args.model}-fp32.bin"
    torch2flm.tofile(export_path, model.model, tokenizer)

if __name__ == "__main__":
    args = parse_args()
    main(args)
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    modelpath = "baichuan-inc/Baichuan2-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(modelpath, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True)

    # normalize lm_head
    state_dict = model.state_dict()
    state_dict['lm_head.weight'] = torch.nn.functional.normalize(state_dict['lm_head.weight'])
    model.load_state_dict(state_dict)

    try:
        model.generation_config = GenerationConfig.from_pretrained(modelpath)
    except:
        pass

    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "baichuan2-7b-" + dtype + ".flm"
    torch2flm.tofile(exportPath, model.to('cpu'), tokenizer, dtype=dtype)
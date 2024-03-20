import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    modelNameOrPath = sys.argv[3] if len(sys.argv) >= 4 else "openbmb/MiniCPM-2B-dpo-fp16"
    tokenizer = AutoTokenizer.from_pretrained(modelNameOrPath, use_fast=False, trust_remote_code=True)
    # `torch_dtype=torch.float16` is set by default, if it will not cause an OOM Error, you can load model in float32.
    model = AutoModelForCausalLM.from_pretrained(modelNameOrPath, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.eval()

    model.config.__dict__['model_type'] = 'minicpm'

    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "minicpm-2b-" + dtype + ".flm"
    torch2flm.tofile(exportPath, model, tokenizer, pre_prompt = "<s>", 
                     user_role = "<用户>", bot_role = "<AI>", 
                     history_sep = "", dtype = dtype)

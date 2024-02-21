import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    modelNameOrPath = sys.argv[3] if len(sys.argv) >= 4 else "internlm/internlm-chat-7b-v1_1"
    tokenizer = AutoTokenizer.from_pretrained(modelNameOrPath, trust_remote_code=True);
    # `torch_dtype=torch.float16` is set by default, if it will not cause an OOM Error, you can load model in float32.
    model = AutoModelForCausalLM.from_pretrained(modelNameOrPath, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.eval()
    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "internlm-7b-" + dtype + ".flm"
    torch2flm.tofile(exportPath, model, tokenizer, pre_prompt = "<s><s>", 
                     user_role = "<|User|>:", bot_role = "<eoh>\n<|Bot|>:", 
                     history_sep = "<eoa>\n<s>", dtype = dtype)

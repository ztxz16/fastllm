import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    model_name = sys.argv[3] if len(sys.argv) >= 4 else 'qwen/Qwen1.5-7B-Chat'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float16)
    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "qwen1.5-7b-" + dtype + ".flm"
    # add custom code here
    torch2flm.tofile(exportPath, model, tokenizer, pre_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>", user_role="<|im_start|>user\n",
                     bot_role="<|im_end|><|im_start|>assistant\n", history_sep="<|im_end|>\n", dtype = dtype)

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    modelNameOrPath = sys.argv[3] if len(sys.argv) >= 4 else 'qwen/Qwen1.5-7B-Chat'
    tokenizer = AutoTokenizer.from_pretrained(modelNameOrPath, trust_remote_code=True);
    # `torch_dtype=torch.float16` is set by default, if it will not cause an OOM Error, you can load model in float32.
    model = AutoModelForCausalLM.from_pretrained(modelNameOrPath, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.eval()
    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else model.config.model_type + "-7b-" + dtype + ".flm"
    if model.config.model_type == "internlm":
        torch2flm.tofile(exportPath, model, tokenizer, pre_prompt = "<s>", 
                         user_role = "<|User|>:", bot_role = "<eoh>\n<|Bot|>:", 
                         history_sep = "<eoa>\n<s>", dtype = dtype)
    elif model.config.model_type == "internlm2":
        torch2flm.tofile(exportPath, model, tokenizer, pre_prompt="<s><|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语).\n<|im_end|>", 
                         user_role="<|im_start|>user\n", bot_role="<|im_end|><|im_start|>assistant\n", history_sep="<|im_end|>\n", dtype = dtype)
    elif model.config.model_type == "qwen2":
        torch2flm.tofile(exportPath, model, tokenizer, pre_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>", user_role="<|im_start|>user\n",
                         bot_role="<|im_end|><|im_start|>assistant\n", history_sep="<|im_end|>\n", dtype = dtype)
    elif model.config.model_type == "qwen2_moe":
        torch2flm.tofile(exportPath, model, tokenizer, pre_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", user_role="<|im_start|>user\n",
                        bot_role="<|im_end|>\n<|im_start|>assistant\n", history_sep="<|im_end|>\n", eos_id = tokenizer.eos_token_id, dtype = dtype)
    # add custom code here
    else:
        torch2flm.tofile(exportPath, model, tokenizer, pre_prompt = "", user_role = "", 
                         bot_role = "", history_sep = "", dtype = dtype)

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    modelNameOrPath = sys.argv[3] if len(sys.argv) >= 4 else 'meta-llama/Meta-Llama-3-8B'
    tokenizer = AutoTokenizer.from_pretrained(modelNameOrPath, trust_remote_code=True);
    # `torch_dtype=torch.float16` is set by default, if it will not cause an OOM Error, you can load model in float32.
    model = AutoModelForCausalLM.from_pretrained(modelNameOrPath, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.eval()
    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else model.config.model_type + "-7b-" + dtype + ".flm"
    torch2flm.tofile(exportPath, model, tokenizer, 
                     pre_prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>", 
                     user_role="<|start_header_id|>user<|end_header_id|>\n",
                     bot_role="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n", 
                     history_sep="<|eot_id|>\n",
                     eos_id = tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                     dtype = dtype)

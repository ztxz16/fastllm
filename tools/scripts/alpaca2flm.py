import sys
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from ftllm import torch2flm

if __name__ == "__main__":
    model_name = sys.argv[3] if len(sys.argv) >= 4 else 'minlik/chinese-alpaca-33b-merged'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # `torch_dtype=torch.float16` is set by default, if it will not cause an OOM Error, you can load model in float32.
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    conf = model.config.__dict__
    conf["model_type"] = "llama"
    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "alpaca-33b-" + dtype + ".flm"
    # add custom code here
    torch2flm.tofile(exportPath, model, tokenizer, dtype = dtype)

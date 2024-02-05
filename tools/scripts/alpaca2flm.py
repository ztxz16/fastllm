import sys
from transformers import LlamaTokenizer, LlamaForCausalLM
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    model_name = sys.argv[3] if len(sys.argv) >= 4 else 'minlik/chinese-alpaca-33b-merged'
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name).float()
    conf = model.config.__dict__
    conf["model_type"] = "llama"
    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "alpaca-33b-' + dtype + '.flm"
    # add custom code here
    torch2flm.tofile(exportPath, model, tokenizer, dtype = dtype)

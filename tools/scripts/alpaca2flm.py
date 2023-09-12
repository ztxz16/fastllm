import sys
from transformers import LlamaTokenizer, LlamaForCausalLM
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    exportPath = sys.argv[1] if (sys.argv[1] is not None) else "alpaca-fp32.flm";
    tokenizer = LlamaTokenizer.from_pretrained('minlik/chinese-alpaca-33b-merged');
    model = LlamaForCausalLM.from_pretrained('minlik/chinese-alpaca-33b-merged').float();
    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "alpaca-33b-' + dtype + '.flm"
    torch2flm.tofile(exportPath, model, tokenizer, dtype = dtype)

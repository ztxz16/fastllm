import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastllm_pytools import torch2flm

tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True);
model = AutoModelForCausalLM.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True).float();
model = model.eval();

if __name__ == "__main__":
    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "moss-' + dtype + '.flm"
    torch2flm.tofile(exportPath, model, tokenizer, dtype = dtype)
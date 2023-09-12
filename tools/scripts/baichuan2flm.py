import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    modelpath = "baichuan-inc/baichuan-13B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(modelpath, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.to("cpu")
    try:
        model.generation_config = GenerationConfig.from_pretrained(modelpath)
    except:
        pass
    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "baichuan-13b-' + dtype + '.flm"
    torch2flm.tofile(exportPath, model, tokenizer, dtype = dtype)

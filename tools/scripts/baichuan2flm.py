import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    exportPath = sys.argv[1] if (sys.argv[1] is not None) else "baichuan-13b-fp32.flm";

    modelpath = "baichuan-inc/baichuan-13B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(modelpath, device_map="auto", trust_remote_code=True)
    try:
        model.generation_config = GenerationConfig.from_pretrained(modelpath);
    except:
        pass;
    torch2flm.tofile(exportPath, model, tokenizer);

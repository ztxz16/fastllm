import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    exportPath = sys.argv[1] if (sys.argv[1] is not None) else "baichuan-fp32.flm";

    modelpath = "baichuan-inc/baichuan-7B"
    peftpath = "hiyouga/baichuan-7b-sft"
    tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(modelpath, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, peftpath).float()

    layers = model.model.model.layers;
    for i in range(len(layers)):
        layers[i].self_attn.W_pack.weight += torch.mm(layers[i].self_attn.W_pack.lora_B.default.weight, layers[i].self_attn.W_pack.lora_A.default.weight) * layers[i].self_attn.W_pack.scaling["default"];

    torch2flm.tofile(exportPath, model.model, tokenizer);

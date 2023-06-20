import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch2flm

tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True);
model = AutoModelForCausalLM.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True).float();
model = model.eval();

if __name__ == "__main__":
    exportPath = sys.argv[1] if (sys.argv[1] is not None) else "moss-fp32.flm";
    torch2flm.tofile(exportPath, model, tokenizer)
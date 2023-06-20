import sys
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch2flm

if __name__ == "__main__":
    exportPath = sys.argv[1] if (sys.argv[1] is not None) else "alpaca-fp32.flm";
    tokenizer = LlamaTokenizer.from_pretrained('minlik/chinese-alpaca-33b-merged');
    model = LlamaForCausalLM.from_pretrained('minlik/chinese-alpaca-33b-merged').float();
    torch2flm.tofile(exportPath, model, tokenizer);

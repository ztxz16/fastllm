import sys
from transformers import AutoTokenizer, AutoModel
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float()
    model = model.eval()

    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "chatglm-6b-fp32.flm";
    torch2flm.tofile(exportPath, model, tokenizer)

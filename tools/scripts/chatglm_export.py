import sys
from transformers import AutoTokenizer, AutoModel
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model = model.eval()

    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "chatglm-6b-' + dtype + '.flm"
    torch2flm.tofile(exportPath, model, tokenizer, dtype = dtype)

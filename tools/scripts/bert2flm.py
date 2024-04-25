import sys
from transformers import AutoTokenizer, AutoModel
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    modelpath = sys.argv[3] if len(sys.argv) >= 4 else 'BAAI/bge-small-zh-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModel.from_pretrained(modelpath).cpu().float()
    model = model.eval()

    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "bert-" + dtype + ".flm"
    torch2flm.tofile(exportPath, model, tokenizer, dtype = dtype)
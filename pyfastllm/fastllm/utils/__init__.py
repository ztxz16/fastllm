from . import torch2flm

def convert(model, tokenizer, output_path, **args):
    torch2flm.tofile(output_path, model, tokenizer, **args)

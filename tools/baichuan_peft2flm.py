import os
import platform
import signal
import sys
import struct
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def writeString(fo, s):
    fo.write(struct.pack('i', len(s)));
    fo.write(s.encode());

if __name__ == "__main__":
    exportPath = sys.argv[1] if (sys.argv[1] is not None) else "fastllm.flm";

    modelpath = "baichuan-inc/baichuan-7B"
    peftpath = "hiyouga/baichuan-7b-sft"
    tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(modelpath, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, peftpath).float()

    layers = model.model.model.layers;
    for i in range(len(layers)):
        layers[i].self_attn.W_pack.weight += torch.mm(layers[i].self_attn.W_pack.lora_B.default.weight, layers[i].self_attn.W_pack.lora_A.default.weight) * layers[i].self_attn.W_pack.scaling["default"];

    dict = model.model.state_dict();
    fo = open(exportPath, "wb");

    # 0. version id
    fo.write(struct.pack('i', 1));

    # 0.1 bos, eos
    fo.write(struct.pack('i', 2));
    writeString(fo, "bos");
    writeString(fo, str(tokenizer.sp_model.bos_id()));
    writeString(fo, "eos");
    writeString(fo, str(tokenizer.sp_model.eos_id()));

    # 1. vocab
    piece_size = tokenizer.sp_model.piece_size();
    fo.write(struct.pack('i', piece_size));
    for i in range(piece_size):
        s = tokenizer.sp_model.id_to_piece(i).encode();
        fo.write(struct.pack('i', len(s)));
        for c in s:
            fo.write(struct.pack('i', c));
        fo.write(struct.pack('i', i));

    # 2. weight
    fo.write(struct.pack('i', len(dict)));
    for key in dict:
        cur = dict[key].numpy().astype(np.float32);
        #cur = dict[key].numpy();

        fo.write(struct.pack('i', len(key)));
        fo.write(key.encode());
        fo.write(struct.pack('i', len(cur.shape)));
        for i in cur.shape:
            fo.write(struct.pack('i', i));
        fo.write(struct.pack('i', 0));
        fo.write(cur.data);

    fo.close();

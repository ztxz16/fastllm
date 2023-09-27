import sys
import struct
import numpy as np
import torch
import binascii
from transformers import AutoTokenizer, AutoModel
from fastllm_pytools import torch2flm

def glmtofile(exportPath,
           model,
           tokenizer = None,
           dtype = "float16"):
    if (dtype not in torch2flm.fastllm_data_type_dict):
        print("dtype should in ", list(torch2flm.fastllm_data_type_dict.keys()))
        exit(0)

    dict = model.state_dict()
    fo = open(exportPath, "wb")

    # 0. version id
    fo.write(struct.pack('i', 2))

    # 0.1 model info
    modelInfo = model.config.__dict__
    if model.generation_config is not None:
        modelInfo.update(model.generation_config.__dict__)
    if ("model_type" not in modelInfo):
        print("unknown model_type.")
        exit(0)

    modelInfo["tokenizer_use_score"] = "1" # 分词带分数
    modelInfo["tokenizer_serialized"]=binascii.hexlify(tokenizer.sp_model.serialized_model_proto()).decode("latin-1") # sentencepiece分词器序列化存储

    if hasattr(model, "peft_config"):
        adapter_size = len(model.peft_config)
        modelInfo["peft_size"] = adapter_size

    fo.write(struct.pack('i', len(modelInfo)))
    for it in modelInfo.keys():
        torch2flm.writeKeyValue(fo, str(it), str(modelInfo[it]))

    if hasattr(model, "peft_config"):
        for adapter_name in model.peft_config.keys():
            adapter_dict = model.peft_config[adapter_name].__dict__
            torch2flm.writeString(fo, adapter_name)
            fo.write(struct.pack('i', len(adapter_dict)))
            for it in adapter_dict.keys():
                torch2flm.writeKeyValue(fo, str(it), str(adapter_dict[it]))

    # 1. vocab
    if (tokenizer):
        if (hasattr(tokenizer, "tokenizer")):
            tokenizer = tokenizer.tokenizer
        if (hasattr(tokenizer, "sp_model")):
            piece_size = tokenizer.sp_model.piece_size()
            fo.write(struct.pack('i', piece_size))
            for i in range(piece_size):
                s = tokenizer.sp_model.id_to_piece(i).encode()
                fo.write(struct.pack('i', len(s)))
                for c in s:
                    fo.write(struct.pack('i', c))
                fo.write(struct.pack('i', i))
                fo.write(struct.pack('f', float(tokenizer.sp_model.get_score(i))))
        else:
            vocab = tokenizer.get_vocab()
            fo.write(struct.pack('i', len(vocab)))
            for v in vocab.keys():
                s = v.encode()
                fo.write(struct.pack('i', len(s)))
                for c in s:
                    fo.write(struct.pack('i', c))
                fo.write(struct.pack('i', vocab[v]))
                fo.write(struct.pack('f', 1.0))
    else:
        fo.write(struct.pack('i', 0))

    weight_type_dict = {}
    module_dict = {}
    for key, m in model.named_modules():
        if (isinstance(m, torch.nn.Linear)):
            weight_type_dict[key + ".weight"] = "linear"
            module_dict[key + ".weight"] = m
        if (isinstance(m, torch.nn.Embedding)):
            weight_type_dict[key] = "embedding"

    # 2. weight
    fo.write(struct.pack('i', len(dict)))
    tot = 0
    for key in dict:
        ori_data_type = 0
        ori_np_data_type = np.float32
        cur_weight_type = 0
        if (key in weight_type_dict and weight_type_dict[key] in torch2flm.fastllm_weight_type_dict):
            cur_weight_type = torch2flm.fastllm_weight_type_dict[weight_type_dict[key]]
        to_data_type = 0
        if (cur_weight_type == 1):
            to_data_type = torch2flm.fastllm_data_type_dict[dtype]
            if (to_data_type == 7):
                ori_data_type = 7
                ori_np_data_type = np.float16

        cur = dict[key].numpy().astype(ori_np_data_type)
        
        if hasattr(model, "peft_config"):
            weight_name = key.replace('base_model.model.', '')
            fo.write(struct.pack('i', len(weight_name)))
            fo.write(weight_name.encode())
        else:
            fo.write(struct.pack('i', len(key)))
            fo.write(key.encode())
        fo.write(struct.pack('i', len(cur.shape)))
        for i in cur.shape:
            fo.write(struct.pack('i', i))
        if (to_data_type == 3):
            write_int8(fo, cur)
        elif (to_data_type == 8):
            write_int4(fo, cur)
        else:
            fo.write(struct.pack('i', to_data_type))
            fo.write(cur.data)
        tot += 1
        print("output (", tot, "/", len(dict), end = " )\r")
    print("\nfinish.")
    fo.close()

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-large-chinese", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/glm-large-chinese", trust_remote_code=True)
    model = model.eval()

    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float32"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "glm-" + dtype + ".flm"
    glmtofile(exportPath, model, tokenizer, dtype = dtype)

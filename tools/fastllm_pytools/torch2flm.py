import struct
import numpy as np

def writeString(fo, s):
    fo.write(struct.pack('i', len(s)))
    fo.write(s.encode())

def writeKeyValue(fo, key, value):
    writeString(fo, key)
    writeString(fo, value)

def tofile(exportPath,
           model,
           tokenizer = None,
           pre_prompt = None,
           user_role = None,
           bot_role = None,
           history_sep = None):
    dict = model.state_dict()
    fo = open(exportPath, "wb")

    # 0. version id
    fo.write(struct.pack('i', 2))

    # 0.1 model info
    modelInfo = model.config.__dict__
    if ("model_type" not in modelInfo):
        print("unknown model_type.")
        exit(0)

    if (pre_prompt):
        modelInfo["pre_prompt"] = pre_prompt
    if (user_role):
        modelInfo["user_role"] = user_role
    if (bot_role):
        modelInfo["bot_role"] = bot_role
    if (history_sep):
        modelInfo["history_sep"] = history_sep
    if (modelInfo["model_type"] == "baichuan" and hasattr(model, "model") and hasattr(model.model, "get_alibi_mask")):
        # Baichuan 2代
        modelInfo["use_alibi"] = "1"
        modelInfo["pre_prompt"] = ""
        modelInfo["user_role"] = tokenizer.decode([model.generation_config.user_token_id])
        modelInfo["bot_role"] = tokenizer.decode([model.generation_config.assistant_token_id])
        modelInfo["history_sep"] = ""

    fo.write(struct.pack('i', len(modelInfo)))
    for it in modelInfo.keys():
        writeKeyValue(fo, str(it), str(modelInfo[it]))

    # 1. vocab
    if (tokenizer):
        if (hasattr(tokenizer, "sp_model")):
            piece_size = tokenizer.sp_model.piece_size()
            fo.write(struct.pack('i', piece_size))
            for i in range(piece_size):
                s = tokenizer.sp_model.id_to_piece(i).encode()
                fo.write(struct.pack('i', len(s)))
                for c in s:
                    fo.write(struct.pack('i', c))
                fo.write(struct.pack('i', i))
        else:
            vocab = tokenizer.get_vocab()
            fo.write(struct.pack('i', len(vocab)))
            for v in vocab.keys():
                s = v.encode()
                fo.write(struct.pack('i', len(s)))
                for c in s:
                    fo.write(struct.pack('i', c))
                fo.write(struct.pack('i', vocab[v]))
    else:
        fo.write(struct.pack('i', 0))

    # 2. weight
    fo.write(struct.pack('i', len(dict)))
    tot = 0;
    for key in dict:
        cur = dict[key].numpy().astype(np.float32)
        fo.write(struct.pack('i', len(key)))
        fo.write(key.encode())
        fo.write(struct.pack('i', len(cur.shape)))
        for i in cur.shape:
            fo.write(struct.pack('i', i))
        fo.write(struct.pack('i', 0))
        fo.write(cur.data)
        tot += 1
        print("output (", tot, "/", len(dict), end = " )\r")
    print("\nfinish.")
    fo.close()
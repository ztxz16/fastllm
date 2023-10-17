import logging
import struct
import numpy as np

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

def write_int(fp, i):
    fp.write(struct.pack('i', i))

def write_str(fp, s):
    fp.write(struct.pack('i', len(s)))
    fp.write(s.encode())

def write_dict(fp, kv_dict):
    fp.write(struct.pack('i', len(kv_dict)))
    for k, v in kv_dict.items():
        write_str(fp, str(k))
        write_str(fp, str(v))

def write_tensor(fp, key, value):
    fp.write(struct.pack('i', len(key)))
    fp.write(key.encode())
    fp.write(struct.pack('i', len(value.shape)))
    for axis in value.shape: fp.write(struct.pack('i', axis))
    fp.write(struct.pack('i', 0))
    fp.write(value.data)


def convert_configs(f, model_config):
    modelInfo = model_config.__dict__
    # modelInfo["pre_prompt"] = pre_prompt or None
    # modelInfo["user_role"] = user_role or None
    # modelInfo["bot_role"] = bot_role or None
    # modelInfo["history_sep"] = history_sep or None
    write_dict(f, modelInfo)


def convert_tokenizer(fp, tokenizer):
    if (hasattr(tokenizer, "sp_model")):
        piece_size = tokenizer.sp_model.piece_size()
        fp.write(struct.pack('i', piece_size))
        for i in range(piece_size):
            s = tokenizer.sp_model.id_to_piece(i).encode()
            fp.write(struct.pack('i', len(s)))
            for c in s:
                fp.write(struct.pack('i', c))
            fp.write(struct.pack('i', i))
    else:
        vocab = tokenizer.get_vocab()
        fp.write(struct.pack('i', len(vocab)))
        for v in vocab.keys():
            s = v.encode()
            fp.write(struct.pack('i', len(s)))
            for c in s:
                fp.write(struct.pack('i', c))
            fp.write(struct.pack('i', vocab[v]))


def convert_weights(fp, state_dict):
    fp.write(struct.pack('i', len(dict)))
    tot = 0
    for key in state_dict:
        cur = dict[key].numpy().astype(np.float32)
        write_tensor(fp, key, cur)
        tot += 1


def convert(file_path,
           model,
           tokenizer = None,
           verbose=False):
    with open(file_path, encoding='utf-8') as f:
        # 0. version id
        f.write(struct.pack('i', 2))

        # 0.1 model info
        convert_configs(f, model.config)

        # 1. vocab
        if tokenizer:
            convert_tokenizer(f, tokenizer)
        else:
            f.write(struct.pack('i', 0))

        # 2. weight
        state_dict = model.cpu().state_dict()
        convert_weights(f, state_dict )
   
        
    
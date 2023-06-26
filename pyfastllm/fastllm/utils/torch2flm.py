import struct
import numpy as np

def write_str(fp, s):
    fp.write(struct.pack('i', len(s)))
    fp.write(s.encode())

def write_kv(fp, key, value):
    write_str(fp, key)
    write_str(fp, value)


def write_dict(fp, kv_dict):
    fp.write(struct.pack('i', len(kv_dict)))
    for k, v in kv_dict.items():
        write_kv(fp, str(k), str(v))


def tofile(export_path,
           model,
           tokenizer = None,
           pre_prompt = None,
           user_role = None,
           bot_role = None,
           history_sep = None):
    dict = model.state_dict()

    with open(export_path, "wb") as fp:
        # 0. version id
        fp.write(struct.pack('i', 2))

        # 0.1 model infp
        modelInfp = model.config.__dict__
        modelInfp["pre_prompt"] = pre_prompt or None
        modelInfp["user_role"] = user_role or None
        modelInfp["bot_role"] = bot_role or None
        modelInfp["history_sep"] = history_sep or None
        write_dict(fp, modelInfp)
            

        # 1. vocab
        if tokenizer:
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
        else:
            fp.write(struct.pack('i', 0))

        # 2. weight
        fp.write(struct.pack('i', len(dict)))
        tot = 0
        for key in dict:
            cur = dict[key].numpy().astype(np.float32)
            fp.write(struct.pack('i', len(key)))
            fp.write(key.encode())
            fp.write(struct.pack('i', len(cur.shape)))
            for i in cur.shape: fp.write(struct.pack('i', i))
            fp.write(struct.pack('i', 0))
            fp.write(cur.data)
            tot += 1
            print("output (", tot, "/", len(dict), end = " )\r")
        print("\nfinish.")
        fp.close()
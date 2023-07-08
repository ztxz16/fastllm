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

def tofile(export_path,
           model,
           tokenizer = None,
           pre_prompt = None,
           user_role = None,
           bot_role = None,
           history_sep = None,
           verbose=False):
    print(verbose)
    logger.info('start to convert torch model to fastllm model...') if verbose else None
    dict = model.cpu().state_dict()
        
    with open(export_path, "wb") as fp:
        # 0. version id
        fp.write(struct.pack('i', 2))

        logger.info('start to convert model info...') if verbose else None
        # 0.1 model info
        modelInfo = model.config.__dict__
        modelInfo["pre_prompt"] = pre_prompt or None
        modelInfo["user_role"] = user_role or None
        modelInfo["bot_role"] = bot_role or None
        modelInfo["history_sep"] = history_sep or None
        write_dict(fp, modelInfo)
        
        # 1. vocab
        if tokenizer:
            logger.info('start to convert model tokenizer...') if verbose else None
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
        logger.info('start to convert model weights...') if verbose else None
        fp.write(struct.pack('i', len(dict)))
        tot = 0
        for key in dict:
            cur = dict[key].numpy().astype(np.float32)
            write_tensor(fp, key, cur)
            tot += 1
            logger.info(f"output ( {tot} / {len(dict)} ) \r")
        logger.info("\nfinish.")
        fp.close()
import torch
import warnings
import platform
import sys
import struct
import numpy as np;
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True);
model = AutoModelForCausalLM.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True).float();
model = model.eval();

if __name__ == "__main__":
    exportPath = sys.argv[1] if (sys.argv[1] is not None) else "moss.bin";
    dict = model.state_dict();
    fo = open(exportPath, "wb");

    # 0. version id
    fo.write(struct.pack('i', 0));

    # 1. vocab
    vocab = tokenizer.get_vocab();
    fo.write(struct.pack('i', len(vocab)));
    for v in vocab.keys():
        s = [(ord(c) if c not in tokenizer.byte_decoder else tokenizer.byte_decoder[c]) for c in v]
        fo.write(struct.pack('i', len(s)));
        for c in s:
            fo.write(struct.pack('i', c));
        fo.write(struct.pack('i', vocab[v]));
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
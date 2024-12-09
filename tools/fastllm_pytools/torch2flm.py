import struct
import builtins, os, json
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers.decoders import ByteLevel

def writeString(fo, s):
    bytes = s.encode()
    fo.write(struct.pack('i', len(bytes)))
    fo.write(bytes)

def writeKeyValue(fo, key, value):
    writeString(fo, key)
    writeString(fo, value)

fastllm_data_type_dict = {
    "int4g": 9,
    "int4": 8,
    "int8": 3,
    "float16": 7,
    "float32": 0,
}
fastllm_weight_type_dict = {
    "linear": 1,
    "embedding": 2,
    "QuantizedLinear": 111
}

v = np.random.randint(-127, 127, [10, 20]);
temp = v;
c_max = np.expand_dims(np.abs(v).max(axis = -1), -1)
c_scale = c_max / 127.0
v = (v / c_scale + 128.5).clip(1, 255).astype(np.uint8)

def write_int8(fo, v):
    c_max = np.expand_dims(np.abs(v).max(axis = -1), -1).clip(0.1, 1e100)
    c_scale = c_max / 127.0
    v = (v / c_scale + 128.5).clip(1, 255).astype(np.uint8)
    fo.write(struct.pack('i', 3))
    fo.write(struct.pack('i', 0))
    for i in range(c_max.shape[0]):
        fo.write(struct.pack('f', -c_max[i][0]));
        fo.write(struct.pack('f', c_max[i][0]));
    fo.write(v.data)

def write_int4g(fo, v, groupCnt = -1):
    if (groupCnt == -1):
        groupCnt = 128;
    k = v.shape[0]
    m = v.shape[1]
    group = (m - 1) // groupCnt + 1
    pad = group * groupCnt - m
    if (pad > 0):
        v = np.concatenate((v, np.zeros([k, pad])), axis = 1)
    v.resize(k * group, groupCnt)
    c_min = np.expand_dims(v.min(axis = -1), -1)
    c_max = np.expand_dims(v.max(axis = -1), -1)
    c_scale = (c_max - c_min) / 15.0
    c_zero = np.round(0.0 - c_min / c_scale)
    c_zero = c_zero.clip(0, 15)
    c_min = -c_scale * c_zero

    v = (v - c_min) / c_scale
    v = (v + 0.5).astype(np.int8).clip(0, 15).astype(np.uint8)

    if (pad > 0):
        v.resize(k, group * groupCnt)
        v = v[:, :-pad].copy(order = 'C')

    v = v[:, 0::2] * 16 + v[:, 1::2]
    fo.write(struct.pack('i', 9))
    fo.write(struct.pack('i', 0))
    fo.write(struct.pack('i', group))
    fo.write(struct.pack('i', groupCnt))
    for i in range(c_min.shape[0]):
        fo.write(struct.pack('f', c_min[i][0]));
        fo.write(struct.pack('f', c_max[i][0]));
    fo.write(v)

def write_int4(fo, v):
    # c_min = np.expand_dims(-np.abs(v).max(axis = -1), -1)
    # c_max = np.expand_dims(np.abs(v).max(axis = -1), -1)
    # c_scale = c_max / 7.0
    # c_min = c_scale * -8.0

    c_min = np.expand_dims(v.min(axis = -1), -1)
    c_max = np.expand_dims(v.max(axis = -1), -1)
    c_scale = (c_max - c_min) / 15.0
    c_zero = np.round(0.0 - c_min / c_scale)
    c_zero = c_zero.clip(0, 15)
    c_min = -c_scale * c_zero

    v = (v - c_min) / c_scale
    v = (v + 0.5).astype(np.int8).clip(0, 15).astype(np.uint8)
    v = v[:, 0::2] * 16 + v[:, 1::2]
    fo.write(struct.pack('i', 8))
    fo.write(struct.pack('i', 0))
    for i in range(c_min.shape[0]):
        fo.write(struct.pack('f', c_min[i][0]));
        fo.write(struct.pack('f', c_max[i][0]));
    fo.write(v.data)

def tofile(exportPath,
           model,
           tokenizer = None,
           pre_prompt = None,
           user_role = None,
           bot_role = None,
           history_sep = None,
           eos_id = None,
           dtype = "float16"):
    int4g_groupcnt = -1
    if (dtype.startswith("int4g") and len(dtype) > 5):
        try:
            int4g_groupcnt = int(dtype[5:])
            dtype = "int4g";
        except:
            print("dtype should be like \"int4g256\"")
            exit(0)
    if (dtype not in fastllm_data_type_dict):
        print("dtype should be one of ", list(fastllm_data_type_dict.keys()))
        exit(0)

    # 0.1 model info
    modelInfo = model.config.__dict__
    if model.generation_config is not None:
        modelInfo.update(model.generation_config.__dict__)
    if ("model_type" not in modelInfo):
        print("unknown model_type.")
        exit(0)

    fo = open(exportPath, "wb")

    # 0. version id
    fo.write(struct.pack('i', 2))

    if (pre_prompt is not None):
        modelInfo["pre_prompt"] = pre_prompt
    if (user_role is not None):
        modelInfo["user_role"] = user_role
    if (bot_role is not None):
        modelInfo["bot_role"] = bot_role
    if (history_sep):
        modelInfo["history_sep"] = history_sep
    if (modelInfo["model_type"] == "baichuan"):
        if (hasattr(model, "model") and hasattr(model.model, "get_alibi_mask")):
            # Baichuan / Baichuan2 13B
            modelInfo["use_alibi"] = "1"
        modelInfo["pre_prompt"] = ""
        if (modelInfo["vocab_size"] == 125696):
            # Baichuan 2代
            modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.user_token_id) + ">") if hasattr(model.generation_config, "user_token_id") else "";
        else:
            # Baichuan-13B-chat
            modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.user_token_id) + "> ") if hasattr(model.generation_config, "user_token_id") else "";
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.assistant_token_id) + ">") if hasattr(model.generation_config, "assistant_token_id") else "";
        modelInfo["history_sep"] = ""
    if (modelInfo["model_type"] == "qwen"):
        if modelInfo["chat_format"] == "chatml":
            modelInfo["im_end_id"] = tokenizer.im_end_id
            modelInfo["im_start_id"] = tokenizer.im_start_id
    elif (modelInfo["model_type"] == "qwen2"):
        modelInfo["eos_token_id"] = "151645"
    elif (modelInfo["model_type"] == "internlm"):
        modelInfo["eos_token_id"] = "103028"
        if "rotary" in modelInfo:
            rope_scaling = modelInfo.pop("rotary")
            if isinstance(rope_scaling, builtins.dict):
                modelInfo["rope_scaling.type"] = rope_scaling["type"]
                modelInfo["rope_theta"] = rope_scaling["base"]
    elif (modelInfo["model_type"] == "internlm2"):
        modelInfo["eos_token_id"] = "92542"
    if (modelInfo["model_type"] == "chatglm" and hasattr(tokenizer, "build_chat_input")):
        # chatglm3
        modelInfo["pre_prompt"] = "";
        modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.get_command("<|user|>")) + "> \n");
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.get_command("<|assistant|>")) + ">");
        modelInfo["history_sep"] = "";
    if (modelInfo["model_type"] == "chatglm" and hasattr(tokenizer, "name") and tokenizer.name == "GLM4Tokenizer"):
        # glm-4-chat
        modelInfo["pre_prompt"] = "";
        modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.convert_tokens_to_ids("<|user|>")) + ">\n");
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.convert_tokens_to_ids("<|assistant|>")) + ">");
        modelInfo["history_sep"] = "";
        modelInfo["tokenizer_class"] = tokenizer.name;
    if "rope_scaling" in modelInfo and isinstance(modelInfo["rope_scaling"], builtins.dict):
        rope_scaling = modelInfo.pop("rope_scaling")
        for key, value in rope_scaling.items():
            modelInfo["rope_scaling." + key] = value
    if eos_id:
        modelInfo["eos_token_id"] = str(eos_id)

    merges = {}
    if tokenizer:
        modelInfo["tokenizer_use_score"] = "1" # 分词带分数
        if len(tokenizer.all_special_tokens) > 0:
            token_set = set()
            for token in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token, tokenizer.pad_token]:
                for prompt in [pre_prompt, user_role, bot_role, history_sep]:
                    if prompt and str(token) in prompt:
                        modelInfo["tokenizer_has_special_tokens"] = "1"
                token_set.add(str(token))
            if len(tokenizer.all_special_tokens) > len(token_set):
                modelInfo["tokenizer_has_special_tokens"] = "1"
        if hasattr(tokenizer, "sp_model") or (hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "sp_model")):
            try:
                import sentencepiece.sentencepiece_model_pb2 as model_pb2
                with open(tokenizer.vocab_file, "rb") as f:
                    sp_model_data = f.read()
                    sp_model_proto = model_pb2.ModelProto.FromString(sp_model_data)
                    modelInfo["tokenizer_add_dummy_prefix"] = sp_model_proto.normalizer_spec.add_dummy_prefix
                    if sp_model_proto.normalizer_spec.remove_extra_whitespaces:
                        modelInfo["tokenizer_remove_extra_whitespaces"] = True
            except:
                pass
        elif isinstance(tokenizer, PreTrainedTokenizerFast):
            modelInfo["tokenizer_add_dummy_prefix"] = False
            tokenizer_file_name = tokenizer.vocab_file if (hasattr(tokenizer, "vocab_file") and tokenizer.vocab_file) else tokenizer.vocab_files_names['tokenizer_file']
            tokenizer_file = os.path.join(tokenizer.name_or_path, tokenizer_file_name)
            if os.path.exists(tokenizer_file):
                with open(tokenizer_file, "r", encoding='utf-8') as f:
                    tokenizer_data = json.load(f)
                    if "normalizer" in tokenizer_data and tokenizer_data["normalizer"] and "normalizers" in tokenizer_data["normalizer"]:
                        for normalizer in tokenizer_data["normalizer"]["normalizers"]:
                            if normalizer["type"] == "Prepend" and \
                                    (normalizer["prepend"] == '▁' or normalizer["prepend"] == ' '):
                                modelInfo["tokenizer_add_dummy_prefix"] = True
                    if "merges" in tokenizer_data["model"]:
                        bpe_merges = tokenizer_data["model"]["merges"]
                        bpe_merges = [pair.replace(" ", "") for pair in bpe_merges]
                        merges = builtins.dict(zip(bpe_merges, range(0, -len(bpe_merges), -1)))
            if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "decoder") \
                    and isinstance(tokenizer._tokenizer.decoder, ByteLevel):
                modelInfo["tokenizer_byte_as_char"] = True
        else:
            if hasattr(tokenizer, "byte_encoder") and hasattr(tokenizer, "byte_decoder"):
                modelInfo["tokenizer_byte_as_char"] = True
            if not hasattr(tokenizer, "add_prefix_space") or not getattr(tokenizer, "add_prefix_space", True):
                modelInfo["tokenizer_add_dummy_prefix"] = False

    if hasattr(model, "peft_config"):
        adapter_size = len(model.peft_config)
        modelInfo["peft_size"] = adapter_size

    fo.write(struct.pack('i', len(modelInfo)))
    for it in sorted(modelInfo.keys()):
        writeKeyValue(fo, str(it), str(modelInfo[it]))

    if hasattr(model, "peft_config"):
        for adapter_name in model.peft_config.keys():
            adapter_dict = model.peft_config[adapter_name].__dict__
            writeString(fo, adapter_name)
            fo.write(struct.pack('i', len(adapter_dict)))
            for it in adapter_dict.keys():
                writeKeyValue(fo, str(it), str(adapter_dict[it]))

    weight_type_dict = {}
    model = model.cpu();
    dict = model.state_dict()

    # 1. vocab
    if (tokenizer):
        if (hasattr(tokenizer, "tokenizer")):
            if (str(type(tokenizer.tokenizer)).find("Encoding") == -1):
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
            if hasattr(tokenizer, "bpe_ranks"):
                merges = {("".join(bpe_tokens), token_index) for bpe_tokens, token_index in sorted(tokenizer.bpe_ranks.items(), key=lambda kv: kv[1])}
            vocab = tokenizer.get_vocab()
            fo.write(struct.pack('i', len(vocab)))
            for v in vocab.keys():
                score = merges[v] if v in merges else 1.0
                # if (modelInfo["model_type"] == "moss"):
                #     s = [(ord(c) if c not in tokenizer.byte_decoder else tokenizer.byte_decoder[c]) for c in v]
                if (isinstance(v, str)):
                    s = v.encode()
                else:
                    s = v
                fo.write(struct.pack('i', len(s)))
                for c in s:
                    fo.write(struct.pack('i', c))
                fo.write(struct.pack('i', vocab[v]))
                fo.write(struct.pack('f', score))
        if ("tokenizer_has_special_tokens" in modelInfo):
            all_special_tokens = tokenizer.all_special_tokens
            if hasattr(tokenizer, "added_tokens_decoder"):
                for i in tokenizer.added_tokens_decoder:
                    all_special_tokens.append(str(tokenizer.added_tokens_decoder[i]))
            fo.write(struct.pack('i', len(all_special_tokens)))
            for special_token in all_special_tokens:
                writeString(fo, special_token)
    else:
        fo.write(struct.pack('i', 0))

    module_dict = {}
    for key, m in model.named_modules():
        if (isinstance(m, torch.nn.Linear)):
            weight_type_dict[key + ".weight"] = "linear"
            module_dict[key + ".weight"] = m
        if (isinstance(m, torch.nn.Embedding)):
            weight_type_dict[key + ".weight"] = "embedding"

    # 2. weight
    fo.write(struct.pack('i', len(dict)))
    tot = 0
    for key in dict:
        ori_data_type = 0
        ori_np_data_type = np.float32
        cur_weight_type = 0
        if (key in weight_type_dict and weight_type_dict[key] in fastllm_weight_type_dict):
            cur_weight_type = fastllm_weight_type_dict[weight_type_dict[key]]
        to_data_type = 0

        if (cur_weight_type == 1):
            to_data_type = fastllm_data_type_dict[dtype]
            if (to_data_type == 7):
                ori_data_type = 7
                ori_np_data_type = np.float16

        if (dict[key].dtype == torch.bfloat16):
            cur = dict[key].half().numpy().astype(ori_np_data_type)
        else:
            cur = dict[key].numpy().astype(ori_np_data_type)
        
        weight_name = key
        if hasattr(model, "peft_config"):
            weight_name = weight_name.replace('base_model.model.', '')
        writeString(fo, weight_name)
        fo.write(struct.pack('i', len(cur.shape)))
        for i in cur.shape:
            fo.write(struct.pack('i', i))
        if (to_data_type == 3):
            write_int8(fo, cur)
        elif (to_data_type == 8):
            write_int4(fo, cur)
        elif (to_data_type == 9):
            write_int4g(fo, cur, groupCnt = int4g_groupcnt)
        else:
            fo.write(struct.pack('i', to_data_type))
            fo.write(cur.data)
        tot += 1
        print("output (", tot, "/", len(dict), end = " )\r")
    print("\nfinish.")
    fo.close()
